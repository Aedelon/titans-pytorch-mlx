# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Configuration classes for Titans architecture."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TitansConfig:
    """Configuration for Titans models.

    The Titans architecture consists of three main components:
    1. Core (short-term memory): Attention with limited window size
    2. Long-term Memory: Neural memory module that learns to memorize
    3. Persistent Memory: Learnable data-independent parameters

    Attributes:
        dim: Model dimension (d_in in the paper)
        num_heads: Number of attention heads
        num_layers: Number of Titans blocks
        num_memory_layers: Depth of the neural memory MLP (L_M >= 1)
        memory_hidden_mult: Hidden dimension multiplier for memory MLP
        chunk_size: Segment/chunk size for MAC variant
        window_size: Sliding window size for MAG/MAL variants
        num_persistent_tokens: Number of persistent memory tokens (N_p)
        num_memory_tokens: Number of long-term memory tokens (N_l)
        dropout: Dropout probability
        use_conv: Whether to use 1D depthwise convolution after Q/K/V projections
        conv_kernel_size: Kernel size for the 1D convolution
        use_rope: Whether to use Rotary Position Embeddings
        max_seq_len: Maximum sequence length (for position embeddings)
        vocab_size: Vocabulary size (for language modeling)
        memory_lr: Learning rate for memory updates (theta_t in paper)
        memory_momentum: Momentum coefficient for surprise (eta_t in paper)
        memory_decay: Weight decay/forgetting factor (alpha_t in paper)
        activation: Activation function for memory MLP
    """

    # Model dimensions
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    ffn_mult: float = 4.0

    # Memory configuration
    num_memory_layers: int = 2
    memory_hidden_mult: float = 4.0
    num_persistent_tokens: int = 16
    num_memory_tokens: int = 64

    # Attention configuration
    chunk_size: int = 512
    window_size: int = 512

    # Regularization
    dropout: float = 0.1

    # Convolution (optional, following Mamba2/GatedDeltaNet)
    use_conv: bool = True
    conv_kernel_size: int = 4

    # Position embeddings
    use_rope: bool = True
    max_seq_len: int = 8192

    # Language modeling
    vocab_size: int = 32000

    # Memory learning parameters (data-dependent in full implementation)
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_decay: float = 0.01

    # Activation
    activation: Literal["silu", "gelu", "relu"] = "silu"

    # Initialization
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        assert self.num_memory_layers >= 1, "num_memory_layers must be >= 1"
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.window_size > 0, "window_size must be positive"
        assert self.num_persistent_tokens >= 0, "num_persistent_tokens must be >= 0"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert 0.0 < self.memory_lr <= 1.0, "memory_lr must be in (0, 1]"
        assert 0.0 <= self.memory_momentum < 1.0, "memory_momentum must be in [0, 1)"
        assert 0.0 <= self.memory_decay < 1.0, "memory_decay must be in [0, 1)"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.dim // self.num_heads

    @property
    def ffn_dim(self) -> int:
        """Feed-forward network hidden dimension."""
        return int(self.dim * self.ffn_mult)

    @property
    def memory_hidden_dim(self) -> int:
        """Memory MLP hidden dimension."""
        return int(self.dim * self.memory_hidden_mult)
