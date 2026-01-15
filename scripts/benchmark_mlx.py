#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Benchmark MLX vs PyTorch implementation of Titans.

Compare performance between:
- PyTorch CPU
- PyTorch MPS (Apple Silicon GPU)
- MLX (Apple Silicon optimized)

Usage:
    uv run python scripts/benchmark_mlx.py [--batch-size 4] [--seq-len 512] [--warmup 3] [--repeat 10]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

# PyTorch imports
from titans import TitansConfig as TitansConfigPT
from titans import TitansLMM as TitansLMMPT
from titans import TitansMAC as TitansMACPT
from titans import TitansMAG as TitansMAGPT
from titans import TitansMAL as TitansMALPT

# MLX imports
try:
    import mlx.core as mx

    from titans_mlx import TitansConfig as TitansConfigMLX
    from titans_mlx import TitansLMM as TitansLMMMLX
    from titans_mlx import TitansMAC as TitansMACMLX
    from titans_mlx import TitansMAG as TitansMAGMLX
    from titans_mlx import TitansMAL as TitansMALMLX

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available")


def create_config(
    dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    chunk_size: int = 128,
    window_size: int = 128,
) -> dict[str, Any]:
    """Create a shared config dict for both PyTorch and MLX."""
    return {
        "dim": dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": 32000,
        "num_memory_layers": 2,
        "memory_hidden_mult": 4.0,
        "num_persistent_tokens": 16,
        "chunk_size": chunk_size,
        "window_size": window_size,
        "max_seq_len": 8192,
        "dropout": 0.0,
        "use_conv": False,
        "use_rope": True,
    }


def benchmark_pytorch(
    model_class,
    config_dict: dict,
    batch_size: int,
    seq_len: int,
    device: str,
    warmup: int = 3,
    repeat: int = 10,
) -> dict[str, float]:
    """Benchmark PyTorch model."""
    config = TitansConfigPT(**config_dict)
    model = model_class(config).to(device)
    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
            if device == "mps":
                torch.mps.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(input_ids)
            if device == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def benchmark_mlx(
    model_class,
    config_dict: dict,
    batch_size: int,
    seq_len: int,
    warmup: int = 3,
    repeat: int = 10,
) -> dict[str, float]:
    """Benchmark MLX model."""
    config = TitansConfigMLX(**config_dict)
    model = model_class(config)

    input_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)

    # Warmup
    for _ in range(warmup):
        logits, _ = model(input_ids)
        mx.eval(logits)

    # Benchmark
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        logits, _ = model(input_ids)
        mx.eval(logits)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def print_results(
    model_name: str,
    results: dict[str, dict[str, float]],
) -> None:
    """Print benchmark results in a table format."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"{'Backend':<20} {'Mean (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15}")
    print(f"{'-' * 60}")

    for backend, stats in results.items():
        print(
            f"{backend:<20} {stats['mean_ms']:<15.2f} {stats['min_ms']:<15.2f} {stats['max_ms']:<15.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Titans MLX vs PyTorch")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="Benchmark iterations")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["MAC", "MAG", "MAL", "LMM"],
        choices=["MAC", "MAG", "MAL", "LMM"],
        help="Models to benchmark",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print("  Titans MLX vs PyTorch Benchmark")
    print(f"{'#' * 60}")
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Model dimension: {args.dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Benchmark iterations: {args.repeat}")

    config_dict = create_config(
        dim=args.dim,
        num_layers=args.num_layers,
        chunk_size=min(args.seq_len, 128),
        window_size=min(args.seq_len, 128),
    )

    # Model mapping
    pt_models = {
        "MAC": TitansMACPT,
        "MAG": TitansMAGPT,
        "MAL": TitansMALPT,
        "LMM": TitansLMMPT,
    }
    mlx_models = (
        {
            "MAC": TitansMACMLX,
            "MAG": TitansMAGMLX,
            "MAL": TitansMALMLX,
            "LMM": TitansLMMMLX,
        }
        if HAS_MLX
        else {}
    )

    # Check device availability
    has_mps = torch.backends.mps.is_available()

    for model_name in args.models:
        results = {}

        # PyTorch CPU
        print(f"\nBenchmarking {model_name} on PyTorch CPU...")
        results["PyTorch CPU"] = benchmark_pytorch(
            pt_models[model_name],
            config_dict,
            args.batch_size,
            args.seq_len,
            "cpu",
            args.warmup,
            args.repeat,
        )

        # PyTorch MPS
        if has_mps:
            print(f"Benchmarking {model_name} on PyTorch MPS...")
            results["PyTorch MPS"] = benchmark_pytorch(
                pt_models[model_name],
                config_dict,
                args.batch_size,
                args.seq_len,
                "mps",
                args.warmup,
                args.repeat,
            )

        # MLX
        if HAS_MLX:
            print(f"Benchmarking {model_name} on MLX...")
            results["MLX"] = benchmark_mlx(
                mlx_models[model_name],
                config_dict,
                args.batch_size,
                args.seq_len,
                args.warmup,
                args.repeat,
            )

        print_results(f"Titans{model_name}", results)

        # Calculate speedup
        if HAS_MLX and "MLX" in results:
            cpu_time = results["PyTorch CPU"]["mean_ms"]
            mlx_time = results["MLX"]["mean_ms"]
            print(f"\n  MLX vs PyTorch CPU: {cpu_time / mlx_time:.2f}x speedup")

            if has_mps:
                mps_time = results["PyTorch MPS"]["mean_ms"]
                print(f"  MLX vs PyTorch MPS: {mps_time / mlx_time:.2f}x speedup")

    print(f"\n{'#' * 60}")
    print("  Benchmark Complete")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
