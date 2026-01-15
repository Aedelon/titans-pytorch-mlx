#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Inference script for Titans models.

Usage:
    # Generate from trained model
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Hello"

    # Generate with custom parameters
    uv run python scripts/inference.py --checkpoint model.pt --max-tokens 100 --temperature 0.8

    # Interactive mode
    uv run python scripts/inference.py --checkpoint model.pt --interactive
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.memory import MemoryState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
) -> tuple[torch.nn.Module, TitansConfig, str]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = TitansConfig(**checkpoint["config"])
    model_type = checkpoint["model_type"]

    model = create_model(model_type, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded {model_type.upper()} model from {checkpoint_path}")

    return model, config, model_type


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    states: list[MemoryState] | None = None,
) -> tuple[torch.Tensor, list[MemoryState]]:
    """Generate tokens autoregressively.

    Args:
        model: Titans model
        input_ids: Input token IDs (batch, seq)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = normal, <1 = less random, >1 = more random)
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling (keep tokens with cumulative prob <= top_p)
        states: Initial memory states

    Returns:
        Generated token IDs and final memory states
    """
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass (use last chunk_size tokens for efficiency)
        context_size = min(generated.shape[1], model.config.chunk_size)
        context = generated[:, -context_size:]

        logits, states = model(context, states=states)

        # Get logits for last position
        next_logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = (
                next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            )
            next_logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob above threshold
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

    return generated, states


class SimpleTokenizer:
    """Simple character-level tokenizer for demo purposes."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        # Use ASCII characters as base vocabulary
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.id_to_char.get(i, "?") for i in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with Titans models")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )

    # Mode arguments
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)",
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
    model, config, model_type = load_model(Path(args.checkpoint), device)

    # Create tokenizer (simple character-level for demo)
    tokenizer = SimpleTokenizer(config.vocab_size)

    if args.interactive:
        # Interactive mode
        logger.info("Interactive mode. Type 'quit' to exit.")
        states = None

        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() == "quit":
                    break

                # Encode prompt
                input_ids = torch.tensor(
                    [tokenizer.encode(prompt)], dtype=torch.long, device=device
                )

                # Generate
                output_ids, states = generate(
                    model,
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    states=states,
                )

                # Decode output
                generated_text = tokenizer.decode(output_ids[0].tolist())
                print(f"Model: {generated_text}")

            except KeyboardInterrupt:
                break

        logger.info("Goodbye!")

    else:
        # Single generation
        if not args.prompt:
            logger.warning("No prompt provided. Using empty prompt.")

        # Encode prompt
        input_ids = torch.tensor(
            [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
        )

        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generating {args.max_tokens} tokens...")

        # Generate
        output_ids, _ = generate(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        # Decode output
        generated_text = tokenizer.decode(output_ids[0].tolist())

        print("\n" + "=" * 50)
        print("Generated text:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)


if __name__ == "__main__":
    main()
