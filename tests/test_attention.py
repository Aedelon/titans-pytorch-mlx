# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for attention modules."""

import torch

from titans.attention import (
    RotaryPositionEmbedding,
    SegmentedAttention,
    SlidingWindowAttention,
)
from titans.config import TitansConfig


class TestRotaryPositionEmbedding:
    """Tests for RoPE."""

    def test_forward(self) -> None:
        """Test RoPE forward pass."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=256)

        q = torch.randn(2, 4, 16, 64)  # batch, heads, seq, head_dim
        k = torch.randn(2, 4, 16, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_with_offset(self) -> None:
        """Test RoPE with sequence offset."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=256)

        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)

        q_rot, k_rot = rope(q, k, seq_offset=10)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_cache_rebuild(self) -> None:
        """Test cache rebuild for long sequences."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=32)

        q = torch.randn(2, 4, 64, 64)  # Longer than max_seq_len
        k = torch.randn(2, 4, 64, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert rope.cos_cached.shape[0] >= 64

    def test_apply_rotary_preserves_shape(self) -> None:
        """Test _apply_rotary preserves tensor shape."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=256)
        rope._build_cache(16)

        x = torch.randn(2, 4, 16, 64)
        cos = rope.cos_cached[:16]
        sin = rope.sin_cached[:16]

        rotated = rope._apply_rotary(x, cos, sin)

        assert rotated.shape == x.shape


class TestSlidingWindowAttention:
    """Tests for Sliding Window Attention."""

    def test_forward(self, default_config: TitansConfig) -> None:
        """Test SWA forward pass."""
        attn = SlidingWindowAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)

        output = attn(x)

        assert output.shape == x.shape

    def test_forward_with_prefix(self, default_config: TitansConfig) -> None:
        """Test SWA with prefix tokens."""
        attn = SlidingWindowAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)
        prefix = torch.randn(2, 8, default_config.dim)

        output = attn(x, prefix=prefix)

        assert output.shape == x.shape  # Only x positions returned

    def test_forward_with_offset(self, default_config: TitansConfig) -> None:
        """Test SWA with sequence offset."""
        attn = SlidingWindowAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)

        output = attn(x, seq_offset=16)

        assert output.shape == x.shape

    def test_sliding_window_mask(self, default_config: TitansConfig) -> None:
        """Test sliding window mask creation."""
        attn = SlidingWindowAttention(default_config)
        device = torch.device("cpu")

        mask = attn._create_sliding_window_mask(16, device)

        assert mask.shape == (16, 16)
        assert mask.dtype == torch.bool

        # Check causality: position 0 can only attend to itself
        assert mask[0, 0]
        assert not mask[0, 1]

        # Check window: last position cannot attend to first if outside window
        if default_config.window_size < 16:
            assert not mask[15, 0]

    def test_extended_mask(self, default_config: TitansConfig) -> None:
        """Test extended mask for prefix attention."""
        attn = SlidingWindowAttention(default_config)
        device = torch.device("cpu")

        mask = attn._create_extended_mask(
            query_len=8, key_len=16, prefix_len=8, device=device
        )

        assert mask.shape == (1, 1, 8, 16)

        # Queries can attend to all prefix positions
        assert mask[0, 0, 0, :8].all()

    def test_without_rope(self) -> None:
        """Test SWA without RoPE."""
        config = TitansConfig(dim=64, num_heads=4, use_rope=False)
        attn = SlidingWindowAttention(config)
        x = torch.randn(2, 16, config.dim)

        output = attn(x)

        assert output.shape == x.shape

    def test_different_window_sizes(self) -> None:
        """Test different window sizes."""
        for window_size in [4, 8, 16]:
            config = TitansConfig(dim=64, num_heads=4, window_size=window_size)
            attn = SlidingWindowAttention(config)
            x = torch.randn(2, 32, config.dim)

            output = attn(x)

            assert output.shape == x.shape


class TestSegmentedAttention:
    """Tests for Segmented Attention (MAC Core)."""

    def test_forward(self, default_config: TitansConfig) -> None:
        """Test segmented attention forward pass."""
        attn = SegmentedAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)

        output = attn(x)

        assert output.shape == x.shape

    def test_forward_with_persistent(self, default_config: TitansConfig) -> None:
        """Test with persistent memory tokens."""
        attn = SegmentedAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)
        persistent = torch.randn(2, 8, default_config.dim)

        output = attn(x, persistent=persistent)

        assert output.shape == x.shape

    def test_forward_with_memory(self, default_config: TitansConfig) -> None:
        """Test with memory tokens."""
        attn = SegmentedAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)
        memory = torch.randn(2, 16, default_config.dim)

        output = attn(x, memory=memory)

        assert output.shape == x.shape

    def test_forward_with_all_components(self, default_config: TitansConfig) -> None:
        """Test with persistent and memory tokens."""
        attn = SegmentedAttention(default_config)
        x = torch.randn(2, 32, default_config.dim)
        persistent = torch.randn(2, 8, default_config.dim)
        memory = torch.randn(2, 16, default_config.dim)

        output = attn(x, persistent=persistent, memory=memory)

        assert output.shape == x.shape

    def test_causal_mask(self, default_config: TitansConfig) -> None:
        """Test causal mask creation."""
        attn = SegmentedAttention(default_config)
        device = torch.device("cpu")

        mask = attn._create_causal_mask(16, device)

        assert mask.shape == (1, 1, 16, 16)
        assert mask.dtype == torch.bool

        # Lower triangular
        for i in range(16):
            for j in range(16):
                if j <= i:
                    assert mask[0, 0, i, j]
                else:
                    assert not mask[0, 0, i, j]

    def test_without_rope(self) -> None:
        """Test without RoPE."""
        config = TitansConfig(dim=64, num_heads=4, use_rope=False)
        attn = SegmentedAttention(config)
        x = torch.randn(2, 16, config.dim)

        output = attn(x)

        assert output.shape == x.shape
