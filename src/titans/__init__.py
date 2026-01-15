# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans: Learning to Memorize at Test Time

A PyTorch implementation of the Titans architecture from Google Research.
Titans introduce a neural long-term memory module that learns to memorize
historical context and helps attention attend to current context while
utilizing long past information.

Reference:
    Behrouz, A., Zhong, P., & Mirrokni, V. (2024).
    Titans: Learning to Memorize at Test Time.
    arXiv preprint arXiv:2501.00663
"""

from titans.attention import SegmentedAttention, SlidingWindowAttention
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.models import (
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
)
from titans.persistent import PersistentMemory

__version__ = "0.1.0"
__author__ = "Delanoe Pirard / Aedelon"
__license__ = "Apache-2.0"

__all__ = [
    # Config
    "TitansConfig",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    # Attention
    "SlidingWindowAttention",
    "SegmentedAttention",
    # Persistent Memory
    "PersistentMemory",
    # Models
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
]
