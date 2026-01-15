# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Configuration for Titans MLX models.

Identical to PyTorch TitansConfig for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TitansConfig:
    """Configuration for Titans models.

    This configuration is identical to the PyTorch version to ensure
    compatibility and identical behavior.

    Attributes:
        dim: Model dimension (d_model)
        num_heads: Number of attention heads
        num_layers: Number of Titans blocks
        vocab_size: Vocabulary size for embedding
        num_memory_layers: Depth of memory MLP (L_M >= 1)
        memory_hidden_mult: Hidden dimension multiplier for memory MLP
        num_persistent_tokens: Number of persistent memory tokens (N_p)
        chunk_size: Segment size for MAC variant
        window_size: Sliding window size for MAG/MAL variants
        memory_lr: Learning rate for memory updates (theta)
        memory_momentum: Momentum coefficient for memory (eta)
        memory_decay: Forgetting/decay factor for memory (alpha)
        max_seq_len: Maximum sequence length
        use_conv: Whether to use 1D convolution
        conv_kernel_size: Kernel size for convolution
        use_rope: Whether to use Rotary Position Embeddings
        dropout: Dropout probability
        activation: Activation function name
        init_std: Standard deviation for weight initialization
    """

    # Core dimensions
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    ffn_mult: float = 4.0

    # Memory configuration
    num_memory_layers: int = 2
    memory_hidden_mult: float = 4.0
    num_persistent_tokens: int = 16
    num_memory_tokens: int = 64

    # Sequence configuration
    chunk_size: int = 512
    window_size: int = 512
    max_seq_len: int = 8192

    # Memory learning parameters
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_decay: float = 0.01

    # Architecture options
    use_conv: bool = True
    conv_kernel_size: int = 4
    use_rope: bool = True

    # Training
    dropout: float = 0.0
    activation: str = "silu"
    init_std: float = 0.02

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
        """Hidden dimension for memory MLP."""
        return int(self.dim * self.memory_hidden_mult)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "ffn_mult": self.ffn_mult,
            "num_memory_layers": self.num_memory_layers,
            "memory_hidden_mult": self.memory_hidden_mult,
            "num_persistent_tokens": self.num_persistent_tokens,
            "num_memory_tokens": self.num_memory_tokens,
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
            "max_seq_len": self.max_seq_len,
            "memory_lr": self.memory_lr,
            "memory_momentum": self.memory_momentum,
            "memory_decay": self.memory_decay,
            "use_conv": self.use_conv,
            "conv_kernel_size": self.conv_kernel_size,
            "use_rope": self.use_rope,
            "dropout": self.dropout,
            "activation": self.activation,
            "init_std": self.init_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TitansConfig:
        """Create config from dictionary."""
        return cls(**d)
