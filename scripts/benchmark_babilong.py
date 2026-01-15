#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
BABILong benchmark evaluation for Titans models.

BABILong is a benchmark for testing long-context reasoning abilities.
It extends the original bAbI tasks with much longer context lengths
(up to 1M tokens) by inserting irrelevant "haystack" text.

Paper: "BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack"

Usage:
    # Evaluate on BABILong
    uv run python scripts/benchmark_babilong.py --checkpoint model.pt --task qa1

    # Evaluate all tasks
    uv run python scripts/benchmark_babilong.py --checkpoint model.pt --task all

    # Evaluate at specific context length
    uv run python scripts/benchmark_babilong.py --checkpoint model.pt --task qa1 --context-length 16k
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

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
# BABILong Tasks
# =============================================================================

BABILONG_TASKS = {
    "qa1": "single-supporting-fact",
    "qa2": "two-supporting-facts",
    "qa3": "three-supporting-facts",
    "qa4": "two-arg-relations",
    "qa5": "three-arg-relations",
    "qa6": "yes-no-questions",
    "qa7": "counting",
    "qa8": "lists-sets",
    "qa9": "simple-negation",
    "qa10": "indefinite-knowledge",
}

CONTEXT_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


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
# BABILong Dataset
# =============================================================================


def load_babilong_task(
    task: str,
    context_length: str = "0k",
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load BABILong task data.

    Args:
        task: Task name (qa1-qa10)
        context_length: Context length variant
        split: Dataset split

    Returns:
        List of examples with 'input', 'question', 'answer' keys
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required for BABILong")

    task_name = BABILONG_TASKS.get(task)
    if not task_name:
        raise ValueError(
            f"Unknown task: {task}. Choose from {list(BABILONG_TASKS.keys())}"
        )

    logger.info(f"Loading BABILong {task} ({task_name}) at {context_length}")

    try:
        # Try loading from HuggingFace
        dataset = load_dataset(
            "RMT-team/babilong",
            f"{task_name}_{context_length}",
            split=split,
        )
    except Exception as e:
        logger.warning(f"Could not load from HuggingFace: {e}")
        logger.info("Generating synthetic BABILong-style data")
        return generate_synthetic_babilong(task, context_length, num_examples=100)

    examples = []
    for item in dataset:
        examples.append(
            {
                "input": item.get("input", item.get("context", "")),
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("target", "")),
            }
        )

    return examples


def generate_synthetic_babilong(
    task: str,
    context_length: str,
    num_examples: int = 100,
) -> list[dict[str, Any]]:
    """Generate synthetic BABILong-style examples for testing.

    This is a fallback when the actual dataset is not available.
    """
    import random

    # Parse context length
    length_map = {
        "0k": 0,
        "1k": 1000,
        "2k": 2000,
        "4k": 4000,
        "8k": 8000,
        "16k": 16000,
        "32k": 32000,
        "64k": 64000,
        "128k": 128000,
    }
    target_length = length_map.get(context_length, 0)

    # Simple templates for different tasks
    names = ["Mary", "John", "Sandra", "Daniel", "Fred", "Bill", "Julie", "Emily"]
    locations = ["garden", "kitchen", "bedroom", "bathroom", "hallway", "office"]
    objects = ["apple", "football", "milk", "keys", "book", "phone"]

    # Filler text for haystack
    filler_sentences = [
        "The weather was nice today.",
        "Birds were singing in the trees.",
        "The sun was shining brightly.",
        "A gentle breeze was blowing.",
        "Flowers were blooming in the garden.",
        "The sky was clear and blue.",
    ]

    examples = []

    for _ in range(num_examples):
        # Generate a simple story
        name1 = random.choice(names)
        random.choice([n for n in names if n != name1])
        loc1 = random.choice(locations)
        loc2 = random.choice([loc for loc in locations if loc != loc1])
        obj = random.choice(objects)

        # Core fact
        story = f"{name1} went to the {loc1}. "
        story += f"{name1} picked up the {obj}. "
        story += f"{name1} went to the {loc2}. "

        # Add haystack text
        if target_length > 0:
            haystack = " ".join(random.choices(filler_sentences, k=target_length // 50))
            # Insert haystack in the middle
            story = (
                story[: len(story) // 2]
                + " "
                + haystack
                + " "
                + story[len(story) // 2 :]
            )

        # Question and answer depend on task
        if task == "qa1":
            question = f"Where is {name1}?"
            answer = loc2
        elif task == "qa2":
            question = f"Where is the {obj}?"
            answer = loc2
        else:
            question = f"Where did {name1} go?"
            answer = loc2

        examples.append(
            {
                "input": story,
                "question": question,
                "answer": answer,
            }
        )

    return examples


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    context: str,
    question: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:
    """Generate answer for a question given context.

    Args:
        model: Titans model
        tokenizer: Tokenizer
        context: Context text
        question: Question to answer
        device: Device
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated answer string
    """
    # Format prompt
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    # Tokenize
    if callable(tokenizer):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.chunk_size * 10,
        )
        input_ids = encoded["input_ids"].to(device)
    else:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Process context in chunks for long sequences
    chunk_size = model.config.chunk_size
    states = None

    if input_ids.shape[1] > chunk_size:
        # Process in chunks, keeping memory
        num_chunks = (input_ids.shape[1] - 1) // chunk_size + 1
        for i in range(num_chunks - 1):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = input_ids[:, start:end]
            _, states = model(chunk, states=states)

        # Last chunk for generation
        input_ids = input_ids[:, (num_chunks - 1) * chunk_size :]

    # Generate
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs, states = model(generated[:, -chunk_size:], states=states)
        next_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # Stop at newline or period
        if hasattr(tokenizer, "decode"):
            token_str = tokenizer.decode(next_token[0])
            if "\n" in token_str or token_str.strip() in [".", "?", "!"]:
                break

    # Decode only the generated part
    generated_ids = generated[0, input_ids.shape[1] :].tolist()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clean up answer
    answer = answer.strip().split("\n")[0].strip()

    return answer


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    answer = answer.lower()
    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)
    # Remove extra whitespace
    answer = " ".join(answer.split())
    return answer


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Evaluate model on a BABILong task.

    Args:
        model: Titans model
        tokenizer: Tokenizer
        examples: List of examples
        device: Device
        max_examples: Maximum examples to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    if max_examples:
        examples = examples[:max_examples]

    correct = 0
    total = 0
    results = []

    for example in tqdm(examples, desc="Evaluating"):
        predicted = generate_answer(
            model,
            tokenizer,
            example["input"],
            example["question"],
            device,
        )

        # Compare normalized answers
        pred_norm = normalize_answer(predicted)
        gold_norm = normalize_answer(example["answer"])

        is_correct = gold_norm in pred_norm or pred_norm in gold_norm

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "question": example["question"],
                "predicted": predicted,
                "gold": example["answer"],
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="BABILong benchmark evaluation")

    # Model arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="HuggingFace tokenizer"
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="qa1",
        help="Task to evaluate (qa1-qa10 or 'all')",
    )
    parser.add_argument(
        "--context-length",
        type=str,
        default="0k",
        choices=CONTEXT_LENGTHS,
        help="Context length variant",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Max examples to evaluate"
    )

    # Device arguments
    parser.add_argument("--device", type=str, default="auto", help="Device")

    # Output arguments
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")

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
        Path(args.checkpoint), device
    )

    # Load tokenizer
    tokenizer_name = args.tokenizer or saved_tokenizer
    if tokenizer_name and HAS_TRANSFORMERS:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        from scripts.inference import SimpleTokenizer

        tokenizer = SimpleTokenizer(config.vocab_size)

    # Determine tasks to evaluate
    tasks = list(BABILONG_TASKS.keys()) if args.task == "all" else [args.task]

    # Evaluate
    all_results = {}

    for task in tasks:
        logger.info(f"\nEvaluating {task}...")

        try:
            examples = load_babilong_task(task, args.context_length)
            results = evaluate_task(
                model,
                tokenizer,
                examples,
                device,
                max_examples=args.max_examples,
            )
            all_results[task] = results

            print(
                f"\n{task}: {results['accuracy'] * 100:.1f}% ({results['correct']}/{results['total']})"
            )

        except Exception as e:
            logger.error(f"Error evaluating {task}: {e}")
            all_results[task] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 50)
    print("BABILong Results Summary")
    print("=" * 50)
    print(f"Model: {model_type.upper()}")
    print(f"Context Length: {args.context_length}")
    print("-" * 50)

    accuracies = []
    for task, results in all_results.items():
        if "accuracy" in results:
            acc = results["accuracy"] * 100
            accuracies.append(acc)
            print(f"{task}: {acc:.1f}%")
        else:
            print(f"{task}: ERROR")

    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        print("-" * 50)
        print(f"Average: {avg_acc:.1f}%")
    print("=" * 50)

    # Save results
    if args.output:
        output_data = {
            "model_type": model_type,
            "context_length": args.context_length,
            "results": {
                k: {kk: vv for kk, vv in v.items() if kk != "results"}
                for k, v in all_results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
