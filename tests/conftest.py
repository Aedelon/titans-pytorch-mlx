# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Pytest fixtures for Titans tests."""

import pytest
import torch

from titans.config import TitansConfig


@pytest.fixture
def default_config() -> TitansConfig:
    """Default configuration for tests."""
    return TitansConfig(
        dim=64,
        num_heads=4,
        num_layers=2,
        ffn_mult=2.0,
        num_memory_layers=2,
        memory_hidden_mult=2.0,
        num_persistent_tokens=4,
        num_memory_tokens=8,
        chunk_size=32,
        window_size=16,
        dropout=0.0,  # Disable dropout for deterministic tests
        use_conv=True,
        conv_kernel_size=4,
        use_rope=True,
        max_seq_len=256,
        vocab_size=100,
        memory_lr=0.1,
        memory_momentum=0.9,
        memory_decay=0.01,
    )


@pytest.fixture
def small_config() -> TitansConfig:
    """Minimal configuration for fast tests."""
    return TitansConfig(
        dim=32,
        num_heads=2,
        num_layers=1,
        ffn_mult=2.0,
        num_memory_layers=1,
        memory_hidden_mult=2.0,
        num_persistent_tokens=2,
        num_memory_tokens=4,
        chunk_size=16,
        window_size=8,
        dropout=0.0,
        use_conv=False,
        use_rope=False,
        max_seq_len=64,
        vocab_size=50,
    )


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def seq_len() -> int:
    """Default sequence length for tests."""
    return 32


@pytest.fixture
def device() -> torch.device:
    """Device for tests."""
    return torch.device("cpu")
