#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pretraining script for Titans models.

Usage:
    uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

    # With custom data
    uv run python scripts/pretrain.py --model mag --data path/to/data.txt

    # Resume from checkpoint
    uv run python scripts/pretrain.py --model mac --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
    ) -> None:
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing/demo purposes."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate reproducible synthetic data
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size, (num_samples, seq_len + 1), generator=generator
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.data[idx, :-1],
            "labels": self.data[idx, 1:],
        }


def create_model(
    model_type: str,
    config: TitansConfig,
) -> nn.Module:
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


def load_text_data(path: Path, vocab_size: int) -> torch.Tensor:
    """Load text file and convert to token IDs."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # Simple character-level tokenization for demo
    chars = sorted(set(text))
    char_to_idx = {c: i % vocab_size for i, c in enumerate(chars)}

    tokens = torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)
    return tokens


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(input_ids)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        batch_tokens = labels.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        # Update progress bar
        avg_loss = total_loss / total_tokens
        pbar.set_postfix(
            {"loss": f"{avg_loss:.4f}", "ppl": f"{math.exp(avg_loss):.2f}"}
        )

    return {
        "loss": total_loss / total_tokens,
        "perplexity": math.exp(total_loss / total_tokens),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        batch_tokens = labels.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    config: TitansConfig,
    model_type: str,
    path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "config": config.__dict__,
        "model_type": model_type,
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {path}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Titans models")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="mac",
        choices=["mac", "mag", "mal", "lmm"],
        help="Model variant to train",
    )
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument(
        "--chunk-size", type=int, default=128, help="Chunk size for MAC"
    )
    parser.add_argument(
        "--window-size", type=int, default=128, help="Window size for MAG/MAL"
    )

    # Data arguments
    parser.add_argument("--data", type=str, default=None, help="Path to text data file")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=10000,
        help="Synthetic samples if no data",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")

    # Checkpoint arguments
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--save-every", type=int, default=1, help="Save checkpoint every N epochs"
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

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

    # Create config
    config = TitansConfig(
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        dropout=0.1,
        num_persistent_tokens=8,
        num_memory_layers=2,
    )

    # Create model
    model = create_model(args.model, config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create dataset
    if args.data:
        data = load_text_data(Path(args.data), args.vocab_size)
        train_size = int(0.9 * len(data))
        train_data = data[:train_size]
        val_data = data[train_size:]

        train_dataset = TextDataset(train_data, args.seq_len)
        val_dataset = TextDataset(val_data, args.seq_len)
    else:
        logger.info("Using synthetic data for demo")
        train_dataset = SyntheticDataset(
            args.vocab_size, args.seq_len, args.synthetic_samples, seed=args.seed
        )
        val_dataset = SyntheticDataset(
            args.vocab_size,
            args.seq_len,
            args.synthetic_samples // 10,
            seed=args.seed + 1,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_pct = min(args.warmup_steps / total_steps, 0.3) if total_steps > 0 else 0.1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_pct,
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'=' * 50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.grad_clip
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}"
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}"
        )

        # Save checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                args.model,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                args.model,
                checkpoint_dir / "best_model.pt",
            )
            logger.info(f"New best model! Val loss: {best_val_loss:.4f}")

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        args.epochs,
        config,
        args.model,
        checkpoint_dir / "final_model.pt",
    )

    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed:.2f}s")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")


if __name__ == "__main__":
    main()
