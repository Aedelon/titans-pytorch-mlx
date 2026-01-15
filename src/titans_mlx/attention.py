# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Attention modules for Titans architecture (MLX Implementation).

This module implements:
1. Sliding Window Attention (SWA) - for MAG and MAL variants
2. Segmented Attention - for MAC variant with full causal attention per segment
3. Rotary Position Embeddings (RoPE)

MLX-specific optimizations:
- Vectorized operations for Apple Silicon
- Efficient memory usage with unified memory
- Lazy evaluation for optimized computation graphs
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - MLX Implementation.

    Applies rotary position embeddings to queries and keys.
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(
        self, dim: int, max_seq_len: int = 8192, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self._inv_freq = inv_freq

        # Precompute cos and sin for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        positions = mx.arange(seq_len, dtype=mx.float32)
        # Outer product: positions @ inv_freq^T
        freqs = mx.outer(positions, self._inv_freq)

        # Compute cos and sin
        self._cos_cached = mx.cos(freqs)
        self._sin_cached = mx.sin(freqs)
        self.max_seq_len = seq_len

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        seq_offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: Queries (batch, heads, seq, head_dim)
            k: Keys (batch, heads, seq, head_dim)
            seq_offset: Offset for position indices

        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.shape[2]

        # Get cached cos/sin or recompute if needed
        if seq_offset + seq_len > self.max_seq_len:
            self._build_cache(seq_offset + seq_len)

        cos = self._cos_cached[seq_offset : seq_offset + seq_len]
        sin = self._sin_cached[seq_offset : seq_offset + seq_len]

        # Apply rotation
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply rotary embedding to tensor.

        Args:
            x: Input tensor (batch, heads, seq, head_dim)
            cos: Cosine values (seq, head_dim // 2)
            sin: Sine values (seq, head_dim // 2)

        Returns:
            Rotated tensor
        """
        # Split into even and odd parts
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Expand cos/sin for broadcasting: (seq, dim//2) -> (1, 1, seq, dim//2)
        cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=0)
        sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=0)

        # Apply rotation
        rotated_even = x1 * cos - x2 * sin
        rotated_odd = x1 * sin + x2 * cos

        # Interleave even and odd
        # Stack along last dim then reshape
        batch, heads, seq, half_dim = rotated_even.shape
        rotated = mx.stack([rotated_even, rotated_odd], axis=-1)
        return rotated.reshape(batch, heads, seq, half_dim * 2)


def _rearrange_to_heads(x: mx.array, num_heads: int) -> mx.array:
    """Rearrange from (batch, seq, dim) to (batch, heads, seq, head_dim)."""
    batch, seq, dim = x.shape
    head_dim = dim // num_heads
    x = x.reshape(batch, seq, num_heads, head_dim)
    return mx.transpose(x, (0, 2, 1, 3))


def _rearrange_from_heads(x: mx.array) -> mx.array:
    """Rearrange from (batch, heads, seq, head_dim) to (batch, seq, dim)."""
    batch, heads, seq, head_dim = x.shape
    x = mx.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq, heads * head_dim)


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention (SWA) - MLX Implementation.

    Implements local attention with a fixed window size.
    Each position can only attend to positions within the window.
    Used in MAG and MAL variants of Titans.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.scale = self.head_dim**-0.5

        # Projections
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Rotary embeddings
        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
            )

        # Dropout
        self.dropout_p = config.dropout

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            module.weight = mx.random.normal(module.weight.shape) * self.config.init_std

    def _create_sliding_window_mask(
        self,
        seq_len: int,
    ) -> mx.array:
        """Create sliding window causal mask.

        Args:
            seq_len: Sequence length

        Returns:
            Boolean mask where True = attend, False = mask out
        """
        # Create position indices
        positions = mx.arange(seq_len)

        # Compute distances
        row_idx = mx.expand_dims(positions, axis=1)  # (seq, 1)
        col_idx = mx.expand_dims(positions, axis=0)  # (1, seq)

        # Causal: can only attend to past (including self)
        causal_mask = col_idx <= row_idx

        # Window: can only attend within window
        window_mask = (row_idx - col_idx) < self.window_size

        # Combine
        mask = causal_mask & window_mask

        return mask

    def __call__(
        self,
        x: mx.array,
        prefix: mx.array | None = None,
        seq_offset: int = 0,
    ) -> mx.array:
        """Forward pass with sliding window attention.

        Args:
            x: Input tensor (batch, seq, dim)
            prefix: Optional prefix tokens that can be attended to (batch, prefix_len, dim)
            seq_offset: Offset for rotary embeddings

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape

        # If prefix provided, concatenate it
        if prefix is not None:
            full_x = mx.concatenate([prefix, x], axis=1)
            prefix_len = prefix.shape[1]
        else:
            full_x = x
            prefix_len = 0

        full_len = full_x.shape[1]

        # Project Q, K, V
        q = self.proj_q(x)  # Only from x, not prefix
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        # Reshape for multi-head attention
        q = _rearrange_to_heads(q, self.num_heads)
        k = _rearrange_to_heads(k, self.num_heads)
        v = _rearrange_to_heads(v, self.num_heads)

        # Apply RoPE
        if self.rope is not None:
            # For k/v, we need to account for prefix
            q, _ = self.rope(q, q, seq_offset=prefix_len + seq_offset)
            k, _ = self.rope(k, k, seq_offset=seq_offset)

        # Compute attention scores
        attn_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Create attention mask
        mask = self._create_extended_mask(seq_len, full_len, prefix_len)
        # Mask: True = attend, False = mask out -> fill False with -inf
        attn_scores = mx.where(mask, attn_scores, mx.array(float("-inf")))

        # Softmax
        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Apply dropout during training (if needed)
        if self.dropout_p > 0:
            attn_weights = nn.Dropout(self.dropout_p)(attn_weights)

        # Apply attention
        output = mx.matmul(attn_weights, v)

        # Reshape back
        output = _rearrange_from_heads(output)

        # Output projection
        output = self.proj_out(output)

        return output

    def _create_extended_mask(
        self,
        query_len: int,
        key_len: int,
        prefix_len: int,
    ) -> mx.array:
        """Create mask for queries attending to keys (including prefix).

        Args:
            query_len: Length of query sequence
            key_len: Length of key sequence (prefix + query)
            prefix_len: Length of prefix

        Returns:
            Boolean mask (1, 1, query_len, key_len)
        """
        # Queries can always attend to all prefix tokens
        prefix_mask = mx.ones((query_len, prefix_len), dtype=mx.bool_)

        # For non-prefix positions, use sliding window causal mask
        if key_len > prefix_len:
            main_mask = self._create_sliding_window_mask(query_len)
        else:
            main_mask = mx.zeros((query_len, 0), dtype=mx.bool_)

        # Combine
        mask = mx.concatenate([prefix_mask, main_mask], axis=1)

        # Add batch and head dimensions
        return mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)


class SegmentedAttention(nn.Module):
    """Segmented/Chunked Attention for MAC variant - MLX Implementation.

    Implements full causal attention within each segment/chunk.
    The segment includes:
    1. Persistent memory tokens (fixed)
    2. Retrieved long-term memory tokens
    3. Current input chunk

    This is the "Core" module in the MAC architecture.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # Projections
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Rotary embeddings
        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
            )

        # Dropout
        self.dropout_p = config.dropout

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            module.weight = mx.random.normal(module.weight.shape) * self.config.init_std

    def __call__(
        self,
        x: mx.array,
        persistent: mx.array | None = None,
        memory: mx.array | None = None,
    ) -> mx.array:
        """Forward pass with segmented attention.

        The full sequence is: [persistent] || [memory] || [input]

        Args:
            x: Input tensor (batch, seq, dim)
            persistent: Persistent memory tokens (batch, num_persistent, dim)
            memory: Retrieved long-term memory (batch, num_memory, dim)

        Returns:
            Output tensor (batch, seq, dim) - only for input positions
        """
        batch_size, seq_len, _ = x.shape

        # Build full sequence
        components = []
        prefix_lens = []

        if persistent is not None:
            components.append(persistent)
            prefix_lens.append(persistent.shape[1])

        if memory is not None:
            components.append(memory)
            prefix_lens.append(memory.shape[1])

        components.append(x)

        full_x = mx.concatenate(components, axis=1)
        full_len = full_x.shape[1]
        prefix_len = sum(prefix_lens)

        # Project Q, K, V
        q = self.proj_q(full_x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        # Reshape for multi-head attention
        q = _rearrange_to_heads(q, self.num_heads)
        k = _rearrange_to_heads(k, self.num_heads)
        v = _rearrange_to_heads(v, self.num_heads)

        # Apply RoPE
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Compute attention scores
        attn_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Create causal mask
        mask = self._create_causal_mask(full_len)
        attn_scores = mx.where(mask, attn_scores, mx.array(float("-inf")))

        # Softmax
        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Apply dropout during training
        if self.dropout_p > 0:
            attn_weights = nn.Dropout(self.dropout_p)(attn_weights)

        # Apply attention
        output = mx.matmul(attn_weights, v)

        # Reshape back
        output = _rearrange_from_heads(output)

        # Output projection
        output = self.proj_out(output)

        # Return only the input positions (not persistent/memory)
        return output[:, prefix_len:]

    def _create_causal_mask(
        self,
        seq_len: int,
    ) -> mx.array:
        """Create full causal mask.

        Args:
            seq_len: Sequence length

        Returns:
            Boolean mask (1, 1, seq, seq) where True = attend
        """
        # Create lower triangular mask
        mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        # Add batch and head dimensions
        return mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)
