#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Inference script for Titans models with HuggingFace tokenizers.

Usage:
    # Generate from trained model
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Hello"

    # Generate with HuggingFace tokenizer
    uv run python scripts/inference.py --checkpoint model.pt --tokenizer meta-llama/Llama-2-7b-hf

    # Interactive mode with streaming
    uv run python scripts/inference.py --checkpoint model.pt --interactive --stream

    # Quantized inference
    uv run python scripts/inference.py --checkpoint model.pt --quantize int8
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.memory import MemoryState

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Optional HuggingFace transformers
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Tokenizers
# =============================================================================


class SimpleTokenizer:
    """Simple character-level tokenizer for demo purposes."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        """Encode text to token IDs."""
        ids = [self.char_to_id.get(c, 0) for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return ids

    def decode(
        self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def __call__(self, text: str, return_tensors: str | None = None) -> Any:
        return self.encode(text, return_tensors=return_tensors)


def load_tokenizer(
    tokenizer_name: str | None,
    vocab_size: int,
) -> SimpleTokenizer | PreTrainedTokenizerBase:
    """Load tokenizer - HuggingFace or simple character-level."""
    if tokenizer_name and HAS_TRANSFORMERS:
        logger.info(f"Loading HuggingFace tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    elif tokenizer_name and not HAS_TRANSFORMERS:
        logger.warning("transformers not installed, using simple tokenizer")

    logger.info("Using simple character-level tokenizer")
    return SimpleTokenizer(vocab_size)


# =============================================================================
# Model Loading
# =============================================================================


def create_model(model_type: str, config: TitansConfig) -> torch.nn.Module:
    """Create Titans model based on type."""
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type](config)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    quantize: str | None = None,
) -> tuple[torch.nn.Module, TitansConfig, str, str | None]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        quantize: Quantization mode (None, "int8", "int4")

    Returns:
        Tuple of (model, config, model_type, tokenizer_name)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = TitansConfig(**checkpoint["config"])
    model_type = checkpoint["model_type"]
    tokenizer_name = checkpoint.get("tokenizer_name")

    model = create_model(model_type, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Apply quantization if requested
    if quantize:
        model = quantize_model(model, quantize, device)

    model.eval()

    logger.info(f"Loaded {model_type.upper()} model from {checkpoint_path}")
    if quantize:
        logger.info(f"Applied {quantize} quantization")

    return model, config, model_type, tokenizer_name


def quantize_model(
    model: torch.nn.Module,
    mode: str,
    device: torch.device,
) -> torch.nn.Module:
    """Apply quantization to model.

    Args:
        model: Model to quantize
        mode: Quantization mode ("int8", "int4")
        device: Device

    Returns:
        Quantized model
    """
    if mode == "int8":
        # Dynamic int8 quantization
        if device.type == "cuda":
            try:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
            except Exception as e:
                logger.warning(f"Int8 quantization failed: {e}, using fp16")
                model = model.half()
        else:
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
    elif mode == "int4":
        # Int4 requires bitsandbytes
        import importlib.util

        if importlib.util.find_spec("bitsandbytes") is not None:
            logger.warning("Int4 quantization requires manual conversion, using int8")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
        else:
            logger.warning("bitsandbytes not installed, using fp16")
            if device.type == "cuda":
                model = model.half()
    elif mode == "fp16" and device.type in ("cuda", "mps"):
        model = model.half()

    return model


# =============================================================================
# KV Cache for Fast Generation
# =============================================================================


class KVCache:
    """Key-Value cache for fast autoregressive generation."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        # Allocate cache tensors
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

        for _ in range(num_layers):
            self.k_cache.append(
                torch.zeros(
                    batch_size,
                    num_heads,
                    max_seq_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            self.v_cache.append(
                torch.zeros(
                    batch_size,
                    num_heads,
                    max_seq_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )

        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full k, v.

        Args:
            layer_idx: Layer index
            k: New keys (batch, heads, seq, head_dim)
            v: New values (batch, heads, seq, head_dim)

        Returns:
            Full cached k, v up to current position
        """
        new_seq_len = k.shape[2]

        # Update cache
        self.k_cache[layer_idx][:, :, self.seq_len : self.seq_len + new_seq_len] = k
        self.v_cache[layer_idx][:, :, self.seq_len : self.seq_len + new_seq_len] = v

        # Return full cache
        return (
            self.k_cache[layer_idx][:, :, : self.seq_len + new_seq_len],
            self.v_cache[layer_idx][:, :, : self.seq_len + new_seq_len],
        )

    def increment_seq_len(self, delta: int) -> None:
        """Increment sequence length after processing."""
        self.seq_len += delta

    def reset(self) -> None:
        """Reset cache to empty."""
        self.seq_len = 0
        for i in range(self.num_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()


# =============================================================================
# Text Generation
# =============================================================================


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    states: list[MemoryState] | None = None,
    stream: bool = False,
) -> (
    Iterator[tuple[torch.Tensor, list[MemoryState] | None]]
    | tuple[torch.Tensor, list[MemoryState] | None]
):
    """Generate tokens autoregressively.

    Args:
        model: Titans model
        input_ids: Input token IDs (batch, seq)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        eos_token_id: Stop generation at this token
        states: Initial memory states
        stream: If True, yield tokens one by one

    Returns:
        Generated token IDs and final memory states
    """
    generated = input_ids.clone()

    def _generate_step():
        nonlocal generated, states

        for _ in range(max_new_tokens):
            # Forward pass (use last chunk_size tokens for efficiency)
            context_size = min(generated.shape[1], model.config.chunk_size)
            context = generated[:, -context_size:]

            logits, states = model(context, states=states)

            # Get logits for last position
            next_logits = logits[:, -1, :] / max(temperature, 1e-7)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(generated.shape[0]):
                    for token_id in set(generated[i].tolist()):
                        next_logits[i, token_id] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_logits
                    < torch.topk(next_logits, min(top_k, next_logits.shape[-1]))[0][
                        ..., -1, None
                    ]
                )
                next_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            if stream:
                yield generated, states

        if not stream:
            yield generated, states

    if stream:
        return _generate_step()
    else:
        return next(_generate_step())


def generate_streaming(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    states: list[MemoryState] | None = None,
) -> Iterator[str]:
    """Generate tokens with streaming output.

    Yields decoded text incrementally.
    """
    # Encode prompt
    if callable(tokenizer):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
    else:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

    input_ids.shape[1]

    generator = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        states=states,
        stream=True,
    )

    for generated, _ in generator:
        # Decode only the new token
        new_token = generated[0, -1:].tolist()
        text = tokenizer.decode(new_token, skip_special_tokens=True)
        yield text


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with Titans models")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (e.g., meta-llama/Llama-2-7b-hf)",
    )

    # Generation arguments
    parser.add_argument("--prompt", type=str, default="", help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
    )

    # Mode arguments
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--stream", action="store_true", help="Stream output token by token"
    )

    # Optimization arguments
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int8", "int4", "fp16"],
        default=None,
        help="Quantization mode",
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )

    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    model, config, model_type, saved_tokenizer = load_model(
        Path(args.checkpoint), device, args.quantize
    )

    # Load tokenizer (prefer command line, fallback to saved, then simple)
    tokenizer_name = args.tokenizer or saved_tokenizer
    tokenizer = load_tokenizer(tokenizer_name, config.vocab_size)

    if args.interactive:
        # Interactive mode
        logger.info("Interactive mode. Type 'quit' to exit, 'reset' to clear memory.")
        states = None

        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() == "quit":
                    break
                if prompt.lower() == "reset":
                    states = None
                    logger.info("Memory cleared.")
                    continue

                if args.stream:
                    # Streaming output
                    print("Model: ", end="", flush=True)
                    start_time = time.time()
                    token_count = 0

                    for text in generate_streaming(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        states=states,
                    ):
                        print(text, end="", flush=True)
                        token_count += 1

                    elapsed = time.time() - start_time
                    print(
                        f"\n[{token_count} tokens in {elapsed:.2f}s = {token_count / elapsed:.1f} tok/s]"
                    )
                else:
                    # Normal generation
                    encoded = tokenizer(prompt, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(device)

                    start_time = time.time()
                    output_ids, states = generate(
                        model,
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        states=states,
                    )
                    elapsed = time.time() - start_time

                    generated_text = tokenizer.decode(
                        output_ids[0].tolist(),
                        skip_special_tokens=True,
                    )
                    new_tokens = output_ids.shape[1] - input_ids.shape[1]
                    print(f"Model: {generated_text}")
                    print(
                        f"[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s]"
                    )

            except KeyboardInterrupt:
                break

        logger.info("Goodbye!")

    else:
        # Single generation
        if not args.prompt:
            logger.warning("No prompt provided. Using empty prompt.")

        encoded = tokenizer(args.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generating {args.max_tokens} tokens...")

        start_time = time.time()

        if args.stream:
            print("\n" + "=" * 50)
            print("Generated text:")
            print("=" * 50)
            print(args.prompt, end="", flush=True)

            token_count = 0
            for text in generate_streaming(
                model,
                tokenizer,
                args.prompt,
                device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                print(text, end="", flush=True)
                token_count += 1

            elapsed = time.time() - start_time
            print(f"\n{'=' * 50}")
            print(
                f"[{token_count} tokens in {elapsed:.2f}s = {token_count / elapsed:.1f} tok/s]"
            )
        else:
            output_ids, _ = generate(
                model,
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            elapsed = time.time() - start_time

            generated_text = tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True,
            )

            print("\n" + "=" * 50)
            print("Generated text:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)

            new_tokens = output_ids.shape[1] - input_ids.shape[1]
            print(
                f"[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s]"
            )


if __name__ == "__main__":
    main()
