# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for TitansConfig."""

import pytest

from titans.config import TitansConfig


class TestTitansConfig:
    """Tests for TitansConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TitansConfig()

        assert config.dim == 512
        assert config.num_heads == 8
        assert config.num_layers == 12
        assert config.ffn_mult == 4.0
        assert config.num_memory_layers == 2
        assert config.num_persistent_tokens == 16
        assert config.chunk_size == 512
        assert config.window_size == 512
        assert config.dropout == 0.1
        assert config.use_conv is True
        assert config.use_rope is True
        assert config.vocab_size == 32000

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TitansConfig(
            dim=256,
            num_heads=4,
            num_layers=6,
            vocab_size=1000,
        )

        assert config.dim == 256
        assert config.num_heads == 4
        assert config.num_layers == 6
        assert config.vocab_size == 1000

    def test_head_dim_property(self) -> None:
        """Test head_dim computed property."""
        config = TitansConfig(dim=512, num_heads=8)
        assert config.head_dim == 64

        config = TitansConfig(dim=256, num_heads=4)
        assert config.head_dim == 64

    def test_ffn_dim_property(self) -> None:
        """Test ffn_dim computed property."""
        config = TitansConfig(dim=512, ffn_mult=4.0)
        assert config.ffn_dim == 2048

        config = TitansConfig(dim=256, ffn_mult=2.0)
        assert config.ffn_dim == 512

    def test_memory_hidden_dim_property(self) -> None:
        """Test memory_hidden_dim computed property."""
        config = TitansConfig(dim=512, memory_hidden_mult=4.0)
        assert config.memory_hidden_dim == 2048

    def test_validation_dim_divisible_by_heads(self) -> None:
        """Test that dim must be divisible by num_heads."""
        with pytest.raises(AssertionError, match="dim must be divisible"):
            TitansConfig(dim=100, num_heads=8)

    def test_validation_memory_layers(self) -> None:
        """Test that num_memory_layers must be >= 1."""
        with pytest.raises(AssertionError, match="num_memory_layers"):
            TitansConfig(num_memory_layers=0)

    def test_validation_chunk_size(self) -> None:
        """Test that chunk_size must be positive."""
        with pytest.raises(AssertionError, match="chunk_size"):
            TitansConfig(chunk_size=0)

    def test_validation_window_size(self) -> None:
        """Test that window_size must be positive."""
        with pytest.raises(AssertionError, match="window_size"):
            TitansConfig(window_size=0)

    def test_validation_persistent_tokens(self) -> None:
        """Test that num_persistent_tokens must be >= 0."""
        with pytest.raises(AssertionError, match="num_persistent_tokens"):
            TitansConfig(num_persistent_tokens=-1)

    def test_validation_dropout(self) -> None:
        """Test dropout bounds."""
        # Valid values
        TitansConfig(dropout=0.0)
        TitansConfig(dropout=0.5)

        # Invalid values
        with pytest.raises(AssertionError, match="dropout"):
            TitansConfig(dropout=-0.1)
        with pytest.raises(AssertionError, match="dropout"):
            TitansConfig(dropout=1.0)

    def test_validation_memory_lr(self) -> None:
        """Test memory_lr bounds."""
        # Valid values
        TitansConfig(memory_lr=0.01)
        TitansConfig(memory_lr=1.0)

        # Invalid values
        with pytest.raises(AssertionError, match="memory_lr"):
            TitansConfig(memory_lr=0.0)
        with pytest.raises(AssertionError, match="memory_lr"):
            TitansConfig(memory_lr=1.5)

    def test_validation_memory_momentum(self) -> None:
        """Test memory_momentum bounds."""
        # Valid values
        TitansConfig(memory_momentum=0.0)
        TitansConfig(memory_momentum=0.99)

        # Invalid value
        with pytest.raises(AssertionError, match="memory_momentum"):
            TitansConfig(memory_momentum=1.0)

    def test_validation_memory_decay(self) -> None:
        """Test memory_decay bounds."""
        # Valid values
        TitansConfig(memory_decay=0.0)
        TitansConfig(memory_decay=0.5)

        # Invalid value
        with pytest.raises(AssertionError, match="memory_decay"):
            TitansConfig(memory_decay=1.0)
