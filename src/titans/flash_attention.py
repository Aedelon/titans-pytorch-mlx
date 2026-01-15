# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Flash Attention 2 integration for Titans.

This module provides optional Flash Attention 2 support for faster
and more memory-efficient attention computation.

Flash Attention 2 (Dao et al., 2023) is a fused attention kernel
that achieves significant speedups (2-4x) and memory savings
compared to standard PyTorch attention.

Installation:
    pip install flash-attn --no-build-isolation

Note: Flash Attention requires:
    - NVIDIA GPU with compute capability >= 8.0 (Ampere or newer)
    - CUDA 11.6+
    - PyTorch 2.0+
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from titans.config import TitansConfig

# Check for Flash Attention availability
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None
    flash_attn_varlen_func = None

# Check for PyTorch 2.0+ scaled_dot_product_attention
HAS_TORCH_SDPA = hasattr(F, "scaled_dot_product_attention")


def is_flash_attention_available() -> bool:
    """Check if Flash Attention is available."""
    return HAS_FLASH_ATTN


def is_torch_sdpa_available() -> bool:
    """Check if PyTorch SDPA is available."""
    return HAS_TORCH_SDPA


class FlashSlidingWindowAttention(nn.Module):
    """Sliding Window Attention using Flash Attention 2.

    This implements efficient sliding window attention using
    Flash Attention's native sliding window support.
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

        # Dropout
        self.dropout_p = config.dropout if self.training else 0.0

        # Select attention backend
        self.use_flash = HAS_FLASH_ATTN and torch.cuda.is_available()
        self.use_sdpa = HAS_TORCH_SDPA and not self.use_flash

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Flash Attention.

        Args:
            x: Input tensor (batch, seq, dim)
            attention_mask: Optional attention mask

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        if self.use_flash:
            # Use Flash Attention 2 with sliding window
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
                window_size=(self.window_size, 0),  # (left, right)
            )
        elif self.use_sdpa:
            # Use PyTorch SDPA
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            v = rearrange(v, "b s h d -> b h s d")

            # Create sliding window mask
            mask = self._create_sliding_window_mask(seq_len, x.device)

            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # We provide explicit mask
            )
            output = rearrange(output, "b h s d -> b s h d")
        else:
            # Fallback to standard attention
            output = self._standard_attention(q, k, v, seq_len, x.device)

        # Reshape and project output
        output = rearrange(output, "b s h d -> b s (h d)")
        output = self.proj_out(output)

        return output

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create sliding window causal mask for SDPA."""
        positions = torch.arange(seq_len, device=device)
        row_idx = positions.unsqueeze(1)
        col_idx = positions.unsqueeze(0)

        # Causal mask
        causal_mask = col_idx <= row_idx

        # Window mask
        window_mask = (row_idx - col_idx) < self.window_size

        # Combine
        mask = causal_mask & window_mask

        # Convert to float mask for SDPA
        mask = mask.float()
        mask = mask.masked_fill(~mask.bool(), float("-inf"))
        mask = mask.masked_fill(mask.bool(), 0.0)

        return mask.unsqueeze(0).unsqueeze(0)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Standard attention fallback."""
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window causal mask
        mask = self._create_sliding_window_mask(seq_len, device)
        attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return rearrange(output, "b h s d -> b s h d")


class FlashSegmentedAttention(nn.Module):
    """Segmented Attention using Flash Attention 2.

    Used in MAC variant for full causal attention within segments.
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

        self.dropout_p = config.dropout

        # Select backend
        self.use_flash = HAS_FLASH_ATTN and torch.cuda.is_available()
        self.use_sdpa = HAS_TORCH_SDPA and not self.use_flash

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(
        self,
        x: torch.Tensor,
        persistent: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with full causal attention.

        Args:
            x: Input tensor (batch, seq, dim)
            persistent: Persistent tokens (batch, n_persistent, dim)
            memory: Memory tokens (batch, n_memory, dim)

        Returns:
            Output for input positions only (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Build full sequence
        components = []
        if persistent is not None:
            components.append(persistent)
        if memory is not None:
            components.append(memory)
        components.append(x)

        full_x = torch.cat(components, dim=1)
        full_len = full_x.shape[1]
        prefix_len = full_len - seq_len

        # Project
        q = self.proj_q(full_x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        # Reshape
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        if self.use_flash:
            # Use Flash Attention with full causal mask
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
            )
        elif self.use_sdpa:
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            v = rearrange(v, "b s h d -> b h s d")

            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
            output = rearrange(output, "b h s d -> b s h d")
        else:
            output = self._standard_causal_attention(q, k, v, full_len, x.device)

        # Reshape and project
        output = rearrange(output, "b s h d -> b s (h d)")
        output = self.proj_out(output)

        # Return only input positions
        return output[:, prefix_len:]

    def _standard_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Standard causal attention fallback."""
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return rearrange(output, "b h s d -> b s h d")
