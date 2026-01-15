# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans Model Architectures.

This module implements the three variants of Titans:
1. MAC (Memory as Context): Memory retrieval concatenated with input before attention
2. MAG (Memory as Gate): Memory and attention combined via gating
3. MAL (Memory as Layer): Memory used as a layer before attention

Plus the standalone LMM (Long-term Memory Module) without attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.attention import SegmentedAttention, SlidingWindowAttention
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.persistent import PersistentMemory


class FeedForward(nn.Module):
    """Feed-forward network with gating (following recent architectures)."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.ffn_dim

        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SiLU gating."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# MAC: Memory as Context
# =============================================================================


class MACBlock(nn.Module):
    """Memory as Context Block.

    Architecture:
    1. Retrieve from long-term memory using input as query
    2. Concatenate: [persistent] || [memory] || [input]
    3. Apply segmented attention
    4. Feed-forward network

    At test time:
    - Persistent memory parameters are fixed
    - Attention performs in-context learning
    - Long-term memory continues learning (weight updates)
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Segmented attention (Core module)
        self.attention = SegmentedAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass for MAC block.

        Args:
            x: Input tensor (batch, seq, dim) - single chunk/segment
            state: Memory state from previous chunk

        Returns:
            Tuple of (output, new_state)
        """
        batch_size, seq_len, _ = x.shape

        # Get persistent memory tokens
        persistent = self.persistent(batch_size)

        # Retrieve from long-term memory and update it
        memory_out, new_state = self.memory(x, state=state)
        memory_tokens = self.norm_mem(memory_out)

        # Attention with persistent + memory + input
        normed = self.norm1(x)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        x = x + self.dropout(attn_out)

        # Feed-forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context.

    Segments the sequence into chunks and processes each with MAC blocks.
    Long-term memory persists across chunks within a sequence.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAC blocks
        self.blocks = nn.ModuleList(
            [MACBlock(config) for _ in range(config.num_layers)]
        )

        # Output normalization and head
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states for each layer

        Returns:
            Tuple of (logits, new_states)
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process in chunks
        outputs = []
        new_states = [None] * len(self.blocks)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk = x[:, chunk_start:chunk_end]

            # Process through blocks
            chunk_states = states
            for i, block in enumerate(self.blocks):
                chunk, new_state = block(chunk, state=chunk_states[i])
                new_states[i] = new_state

            outputs.append(chunk)

            # Update states for next chunk
            states = new_states

        # Concatenate outputs
        x = torch.cat(outputs, dim=1)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# MAG: Memory as Gate
# =============================================================================


class MAGBlock(nn.Module):
    """Memory as Gate Block.

    Architecture:
    1. Sliding window attention (short-term memory)
    2. Long-term memory in parallel
    3. Combine via gating: output = y ⊗ M(x)

    The attention handles precise local dependencies,
    while memory provides fading long-range context.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Persistent memory (prepended to input)
        self.persistent = PersistentMemory(config)

        # Sliding window attention
        self.attention = SlidingWindowAttention(config)

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Gating projections (for combining attention and memory)
        self.gate_attn = nn.Linear(config.dim, config.dim, bias=False)
        self.gate_mem = nn.Linear(config.dim, config.dim, bias=False)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_attn = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass for MAG block.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Get persistent memory
        persistent = self.persistent(batch_size)

        # Build prefix for attention
        prefix = persistent

        # Attention branch
        normed = self.norm1(x)
        attn_out = self.attention(normed, prefix=prefix)
        attn_out = self.norm_attn(attn_out)

        # Memory branch
        mem_out, new_state = self.memory(normed, state=state)
        mem_out = self.norm_mem(mem_out)

        # Gating combination: o = y ⊗ M(x)
        gate_a = torch.sigmoid(self.gate_attn(attn_out))
        gate_m = torch.sigmoid(self.gate_mem(mem_out))
        combined = gate_a * attn_out + gate_m * mem_out

        x = x + self.dropout(combined)

        # Feed-forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, new_state


class TitansMAG(nn.Module):
    """Titans with Memory as Gate.

    Uses sliding window attention and long-term memory in parallel,
    combined via a gating mechanism.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAG blocks
        self.blocks = nn.ModuleList(
            [MAGBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# MAL: Memory as Layer
# =============================================================================


class MALBlock(nn.Module):
    """Memory as Layer Block.

    Architecture:
    1. Long-term memory processes input
    2. Sliding window attention on memory output
    3. Feed-forward network

    Memory acts as a preprocessing layer before attention.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Long-term memory (first layer)
        self.memory = NeuralLongTermMemory(config)

        # Sliding window attention (second layer)
        self.attention = SlidingWindowAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm3 = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass for MAL block.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Get persistent memory
        persistent = self.persistent(batch_size)

        # Memory layer
        normed = self.norm1(x)
        mem_out, new_state = self.memory(normed, state=state)
        x = x + self.dropout(mem_out)

        # Attention layer with persistent prefix
        normed = self.norm2(x)
        attn_out = self.attention(normed, prefix=persistent)
        x = x + self.dropout(attn_out)

        # Feed-forward
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, new_state


class TitansMAL(nn.Module):
    """Titans with Memory as Layer.

    Memory processes input before attention in a sequential manner.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAL blocks
        self.blocks = nn.ModuleList(
            [MALBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# LMM: Long-term Memory Module (standalone)
# =============================================================================


class LMMBlock(nn.Module):
    """Standalone Long-term Memory Block (no attention).

    Uses only the neural memory module as a sequence model.
    This tests the memory's ability to work independently.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        # Memory
        normed = self.norm1(x)
        mem_out, new_state = self.memory(normed, state=state)
        x = x + self.dropout(mem_out)

        # Feed-forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, new_state


class TitansLMM(nn.Module):
    """Titans with only Long-term Memory (no attention).

    A sequence model using only the neural memory module.
    Tests memory's standalone capability.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of LMM blocks
        self.blocks = nn.ModuleList(
            [LMMBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states
