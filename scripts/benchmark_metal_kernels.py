#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Benchmark Metal Kernels vs Standard MLX Operations.

This script compares the performance of custom Metal kernels
against standard MLX operations for Titans architecture.

Usage:
    uv run python scripts/benchmark_metal_kernels.py
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx import (
    metal_causal_attention,
    metal_memory_update,
    metal_rope,
    metal_silu_gate,
    MetalFeedForward,
    MetalRotaryEmbedding,
    get_metal_kernel_info,
)


def benchmark_operation(
    name: str,
    metal_fn,
    standard_fn,
    warmup: int = 5,
    repeat: int = 20,
) -> dict:
    """Benchmark Metal kernel vs standard operation."""
    # Warmup Metal
    for _ in range(warmup):
        result = metal_fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    # Benchmark Metal
    metal_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = metal_fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        metal_times.append(time.perf_counter() - start)

    # Warmup Standard
    for _ in range(warmup):
        result = standard_fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    # Benchmark Standard
    std_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = standard_fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        std_times.append(time.perf_counter() - start)

    metal_mean = sum(metal_times) / len(metal_times) * 1000
    std_mean = sum(std_times) / len(std_times) * 1000
    speedup = std_mean / metal_mean if metal_mean > 0 else 0

    return {
        "name": name,
        "metal_ms": metal_mean,
        "standard_ms": std_mean,
        "speedup": speedup,
    }


def benchmark_silu_gate(batch: int, seq: int, dim: int) -> dict:
    """Benchmark SiLU gating operation."""
    gate = mx.random.normal((batch, seq, dim))
    up = mx.random.normal((batch, seq, dim))
    mx.eval(gate, up)

    return benchmark_operation(
        name=f"SiLU Gate ({batch}x{seq}x{dim})",
        metal_fn=lambda: metal_silu_gate(gate, up),
        standard_fn=lambda: nn.silu(gate) * up,
    )


def benchmark_memory_update(dim: int) -> dict:
    """Benchmark memory update operation."""
    grad = mx.random.normal((dim, dim))
    momentum = mx.zeros((dim, dim))
    weights = mx.random.normal((dim, dim)) * 0.02
    mx.eval(grad, momentum, weights)

    eta, theta, alpha = 0.9, 0.1, 0.01

    def std_update():
        m_new = eta * momentum - theta * grad
        w_new = (1 - alpha) * weights + m_new
        return m_new, w_new

    return benchmark_operation(
        name=f"Memory Update ({dim}x{dim})",
        metal_fn=lambda: metal_memory_update(
            grad, momentum, weights, eta, theta, alpha
        ),
        standard_fn=std_update,
    )


def benchmark_rope(batch: int, heads: int, seq: int, head_dim: int) -> dict:
    """Benchmark RoPE operation."""
    x = mx.random.normal((batch, heads, seq, head_dim))
    inv_freq = 1.0 / (
        10000 ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
    )
    positions = mx.arange(seq, dtype=mx.float32)
    freqs = mx.outer(positions, inv_freq)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    mx.eval(x, cos, sin)

    def std_rope(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_exp = mx.expand_dims(mx.expand_dims(cos, 0), 0)
        sin_exp = mx.expand_dims(mx.expand_dims(sin, 0), 0)
        rot_even = x1 * cos_exp - x2 * sin_exp
        rot_odd = x1 * sin_exp + x2 * cos_exp
        result = mx.stack([rot_even, rot_odd], axis=-1)
        return result.reshape(x.shape)

    return benchmark_operation(
        name=f"RoPE ({batch}x{heads}x{seq}x{head_dim})",
        metal_fn=lambda: metal_rope(x, cos, sin),
        standard_fn=lambda: std_rope(x, cos, sin),
    )


def print_results(results: list[dict]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("  Metal Kernels Benchmark Results")
    print("=" * 80)
    print(f"{'Operation':<35} {'Metal (ms)':<12} {'Standard (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for r in results:
        speedup_str = f"{r['speedup']:.2f}x"
        if r["speedup"] > 1:
            speedup_str = f"✅ {speedup_str}"
        elif r["speedup"] < 1:
            speedup_str = f"⚠️ {speedup_str}"

        print(
            f"{r['name']:<35} {r['metal_ms']:<12.3f} {r['standard_ms']:<14.3f} {speedup_str:<10}"
        )

    print("=" * 80)


def verify_correctness() -> None:
    """Verify that Metal kernels produce correct results."""
    print("\n" + "=" * 80)
    print("  Correctness Verification")
    print("=" * 80)

    # Test SiLU Gate
    gate = mx.random.normal((2, 64, 128))
    up = mx.random.normal((2, 64, 128))
    metal_out = metal_silu_gate(gate, up)
    std_out = nn.silu(gate) * up
    mx.eval(metal_out, std_out)
    silu_match = np.allclose(
        np.array(metal_out), np.array(std_out), rtol=1e-4, atol=1e-5
    )
    print(f"SiLU Gate: {'✅ PASS' if silu_match else '❌ FAIL'}")

    # Test Memory Update
    grad = mx.random.normal((256, 256))
    momentum = mx.zeros((256, 256))
    weights = mx.random.normal((256, 256)) * 0.02
    eta, theta, alpha = 0.9, 0.1, 0.01

    m_metal, w_metal = metal_memory_update(grad, momentum, weights, eta, theta, alpha)
    m_std = eta * momentum - theta * grad
    w_std = (1 - alpha) * weights + m_std
    mx.eval(m_metal, w_metal, m_std, w_std)

    m_match = np.allclose(np.array(m_metal), np.array(m_std), rtol=1e-4, atol=1e-5)
    w_match = np.allclose(np.array(w_metal), np.array(w_std), rtol=1e-4, atol=1e-5)
    print(f"Memory Update (momentum): {'✅ PASS' if m_match else '❌ FAIL'}")
    print(f"Memory Update (weights): {'✅ PASS' if w_match else '❌ FAIL'}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Metal Kernels")
    parser.add_argument(
        "--skip-verification", action="store_true", help="Skip correctness verification"
    )
    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("  Titans MLX - Metal Kernels Benchmark")
    print("#" * 80)

    # Show available kernels
    info = get_metal_kernel_info()
    print(f"\nAvailable Metal Kernels: {', '.join(info['kernels'])}")
    print(f"Supported dtypes: {', '.join(info['supported_dtypes'])}")

    if not args.skip_verification:
        verify_correctness()

    # Run benchmarks
    results = []

    # SiLU Gate benchmarks
    print("\nRunning SiLU Gate benchmarks...")
    results.append(benchmark_silu_gate(2, 256, 512))
    results.append(benchmark_silu_gate(4, 512, 1024))
    results.append(benchmark_silu_gate(8, 1024, 2048))

    # Memory Update benchmarks
    print("Running Memory Update benchmarks...")
    results.append(benchmark_memory_update(256))
    results.append(benchmark_memory_update(512))
    results.append(benchmark_memory_update(1024))

    # RoPE benchmarks (if kernel works)
    print("Running RoPE benchmarks...")
    try:
        results.append(benchmark_rope(2, 8, 128, 64))
        results.append(benchmark_rope(4, 8, 256, 64))
    except Exception as e:
        print(f"  RoPE benchmark skipped: {e}")

    # Print results
    print_results(results)

    # Summary
    speedups = [r["speedup"] for r in results]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    faster_count = sum(1 for s in speedups if s > 1)

    print(f"\nSummary:")
    print(f"  - {faster_count}/{len(results)} operations faster with Metal kernels")
    print(f"  - Average speedup: {avg_speedup:.2f}x")
    print(f"\nNote: MLX is already highly optimized for Apple Silicon.")
    print("      Custom kernels are most beneficial for fused operations")
    print("      that reduce memory bandwidth.")

    print("\n" + "#" * 80)


if __name__ == "__main__":
    main()
