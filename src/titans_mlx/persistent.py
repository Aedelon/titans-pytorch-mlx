# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Persistent Memory Module for Titans (MLX Implementation).

Persistent memory consists of learnable but data-independent parameters
that encode knowledge about the task. These tokens are prepended to the
sequence and remain fixed during inference.

Three perspectives from the paper:
1. Memory: Stores task knowledge abstraction
2. FFN replacement: Acts like data-independent attention weights
3. Technical: Mitigates attention sink effect on initial tokens
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig


class PersistentMemory(nn.Module):
    """Persistent Memory tokens - MLX Implementation.

    These are learnable parameters that:
    - Are data-independent (same for all inputs)
    - Are prepended to the input sequence
    - Encode task-specific knowledge
    - Help stabilize attention across chunks/segments

    Attributes:
        tokens: Learnable token embeddings (num_tokens, dim)
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_tokens = config.num_persistent_tokens
        self.dim = config.dim

        # Learnable persistent tokens
        if self.num_tokens > 0:
            self.tokens = (
                mx.random.normal((self.num_tokens, self.dim)) * config.init_std
            )
        else:
            self.tokens = None

    def __call__(self, batch_size: int) -> mx.array | None:
        """Get persistent memory tokens expanded for batch.

        Args:
            batch_size: Batch size

        Returns:
            Persistent tokens (batch, num_tokens, dim) or None if num_tokens=0
        """
        if self.tokens is None:
            return None

        # Expand for batch dimension by broadcasting
        # (num_tokens, dim) -> (1, num_tokens, dim) -> (batch, num_tokens, dim)
        tokens_expanded = mx.expand_dims(self.tokens, axis=0)
        return mx.broadcast_to(tokens_expanded, (batch_size, self.num_tokens, self.dim))

    def get_tokens(self) -> mx.array | None:
        """Get raw token embeddings.

        Returns:
            Token embeddings (num_tokens, dim) or None
        """
        return self.tokens
