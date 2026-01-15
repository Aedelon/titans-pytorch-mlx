#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Perplexity evaluation script for Titans models.

Evaluates model perplexity on standard benchmarks:
- WikiText-2
- WikiText-103
- PG-19
- Custom text files

Usage:
    # Evaluate on WikiText-2
    uv run python scripts/eval_perplexity.py --checkpoint model.pt --dataset wikitext-2

    # Evaluate on custom text
    uv run python scripts/eval_perplexity.py --checkpoint model.pt --data test.txt

    # Evaluate with specific stride
    uv run python scripts/eval_perplexity.py --checkpoint model.pt --dataset wikitext-103 --stride 512
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

if TYPE_CHECKING:
    pass

# Optional imports
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type](config)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, TitansConfig, str, str | None]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = TitansConfig(**checkpoint["config"])
    model_type = checkpoint["model_type"]
    tokenizer_name = checkpoint.get("tokenizer_name")

    model = create_model(model_type, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded {model_type.upper()} model from {checkpoint_path}")

    return model, config, model_type, tokenizer_name


# =============================================================================
# Data Loading
# =============================================================================


def load_evaluation_data(
    dataset_name: str | None,
    data_path: str | None,
    split: str = "test",
) -> str:
    """Load evaluation data from HuggingFace or file.

    Args:
        dataset_name: HuggingFace dataset name
        data_path: Path to local text file
        split: Dataset split to use

    Returns:
        Full text for evaluation
    """
    if data_path:
        logger.info(f"Loading data from {data_path}")
        with open(data_path) as f:
            return f.read()

    if not HAS_DATASETS:
        raise ImportError("datasets library required for HuggingFace datasets")

    dataset_map = {
        "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
        "pg19": ("pg19", None),
        "lambada": ("lambada", None),
    }

    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(dataset_map.keys())}"
        )

    hf_name, hf_config = dataset_map[dataset_name]
    logger.info(f"Loading {dataset_name} from HuggingFace")

    if hf_config:
        dataset = load_dataset(hf_name, hf_config, split=split)
    else:
        dataset = load_dataset(hf_name, split=split)

    # Concatenate all text
    if "text" in dataset.column_names:
        texts = dataset["text"]
    elif "content" in dataset.column_names:
        texts = dataset["content"]
    else:
        raise ValueError("Dataset has no 'text' or 'content' column")

    return "\n\n".join(texts)


# =============================================================================
# Perplexity Calculation
# =============================================================================


@torch.no_grad()
def calculate_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_length: int = 4096,
    stride: int = 512,
    batch_size: int = 1,
) -> dict[str, float]:
    """Calculate perplexity using sliding window.

    Uses the standard sliding window approach where each position
    is evaluated with maximum context available.

    Args:
        model: Titans model
        tokenizer: Tokenizer
        text: Text to evaluate
        device: Device
        max_length: Maximum context length
        stride: Stride for sliding window
        batch_size: Batch size (for future use)

    Returns:
        Dictionary with perplexity metrics
    """
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)

    seq_len = input_ids.shape[1]
    logger.info(
        f"Evaluating {seq_len} tokens with max_length={max_length}, stride={stride}"
    )

    nlls = []
    prev_end_loc = 0

    pbar = tqdm(range(0, seq_len, stride), desc="Calculating perplexity")

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Number of tokens to evaluate

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()

        # Mask out tokens we've already evaluated (for sliding window)
        target_chunk[:, :-trg_len] = -100

        # Forward pass
        outputs, _ = model(input_chunk)

        # Shift for next-token prediction
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = target_chunk[:, 1:].contiguous()

        # Calculate loss only on non-masked tokens
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        # Count valid tokens
        valid_tokens = (shift_labels != -100).sum()
        if valid_tokens > 0:
            nlls.append((loss.item(), valid_tokens.item()))

        prev_end_loc = end_loc

        # Update progress bar
        if nlls:
            current_ppl = math.exp(sum(n for n, _ in nlls) / sum(t for _, t in nlls))
            pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})

        if end_loc >= seq_len:
            break

    # Calculate final perplexity
    total_nll = sum(n for n, _ in nlls)
    total_tokens = sum(t for _, t in nlls)

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "bits_per_byte": avg_nll / math.log(2),  # Convert to bits
    }


@torch.no_grad()
def calculate_perplexity_with_memory(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: torch.device,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Calculate perplexity using Titans memory.

    This method evaluates perplexity by processing text in chunks
    while maintaining memory state across chunks. This is the
    proper way to evaluate Titans on long sequences.

    Args:
        model: Titans model
        tokenizer: Tokenizer
        text: Text to evaluate
        device: Device
        chunk_size: Size of each chunk

    Returns:
        Dictionary with perplexity metrics
    """
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)

    seq_len = input_ids.shape[1]
    logger.info(f"Evaluating {seq_len} tokens with memory (chunk_size={chunk_size})")

    nlls = []
    states = None

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    pbar = tqdm(range(num_chunks), desc="Evaluating with memory")

    for i in pbar:
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, seq_len)

        chunk = input_ids[:, start_idx:end_idx]

        # Forward pass with memory
        outputs, states = model(chunk, states=states)

        # Shift for next-token prediction
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        # Calculate loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="sum",
        )

        valid_tokens = shift_labels.numel()
        nlls.append((loss.item(), valid_tokens))

        # Update progress bar
        current_ppl = math.exp(sum(n for n, _ in nlls) / sum(t for _, t in nlls))
        pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})

    # Calculate final perplexity
    total_nll = sum(n for n, _ in nlls)
    total_tokens = sum(t for _, t in nlls)

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "bits_per_byte": avg_nll / math.log(2),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Titans model perplexity")

    # Model arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="HuggingFace tokenizer"
    )

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext-2", "wikitext-103", "pg19", "lambada"],
        default=None,
        help="HuggingFace dataset to evaluate on",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")

    # Evaluation arguments
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Maximum context length"
    )
    parser.add_argument(
        "--stride", type=int, default=512, help="Stride for sliding window"
    )
    parser.add_argument(
        "--use-memory", action="store_true", help="Use Titans memory for evaluation"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size for memory evaluation"
    )

    # Device arguments
    parser.add_argument("--device", type=str, default="auto", help="Device")

    # Output arguments
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )

    args = parser.parse_args()

    if not args.dataset and not args.data:
        parser.error("Either --dataset or --data must be provided")

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
        Path(args.checkpoint), device
    )

    # Load tokenizer
    tokenizer_name = args.tokenizer or saved_tokenizer
    if tokenizer_name and HAS_TRANSFORMERS:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        # Simple character tokenizer
        from scripts.inference import SimpleTokenizer

        tokenizer = SimpleTokenizer(config.vocab_size)

    # Load data
    text = load_evaluation_data(args.dataset, args.data, args.split)
    logger.info(f"Loaded {len(text)} characters")

    # Evaluate
    if args.use_memory:
        results = calculate_perplexity_with_memory(
            model,
            tokenizer,
            text,
            device,
            chunk_size=args.chunk_size,
        )
    else:
        results = calculate_perplexity(
            model,
            tokenizer,
            text,
            device,
            max_length=args.max_length,
            stride=args.stride,
        )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Model: {model_type.upper()}")
    print(f"Dataset: {args.dataset or args.data}")
    print(f"Tokens: {results['total_tokens']:,}")
    print("-" * 50)
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Avg NLL: {results['avg_nll']:.4f}")
    print(f"Bits/byte: {results['bits_per_byte']:.4f}")
    print("=" * 50)

    # Save results
    if args.output:
        import json

        results["model_type"] = model_type
        results["dataset"] = args.dataset or args.data
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
