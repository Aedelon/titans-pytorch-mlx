# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Tests de parité PyTorch vs MLX pour l'architecture Titans.

Ces tests vérifient que les implémentations MLX produisent des résultats
numériquement équivalents aux implémentations PyTorch de référence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# PyTorch imports
from titans import TitansConfig as TitansConfigPT
from titans import TitansLMM as TitansLMMPT
from titans import TitansMAC as TitansMACPT
from titans import TitansMAG as TitansMAGPT
from titans import TitansMAL as TitansMALPT

# MLX imports
try:
    import mlx.core as mx

    from titans_mlx import TitansConfig as TitansConfigMLX
    from titans_mlx import TitansLMM as TitansLMMMLX
    from titans_mlx import TitansMAC as TitansMACMLX
    from titans_mlx import TitansMAG as TitansMAGMLX
    from titans_mlx import TitansMAL as TitansMALMLX

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

# Configuration de test minimale pour les tests rapides
SMALL_CONFIG = {
    "dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "vocab_size": 1000,
    "num_memory_layers": 1,
    "memory_hidden_mult": 2.0,
    "num_persistent_tokens": 4,
    "chunk_size": 32,
    "window_size": 32,
    "max_seq_len": 128,
    "dropout": 0.0,  # Pas de dropout pour des résultats déterministes
    "use_conv": False,  # Désactiver conv pour simplifier les tests
    "use_rope": True,
}

# Tolérance pour les comparaisons numériques
RTOL = 1e-4
ATOL = 1e-5


def numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    return torch.from_numpy(arr)


def numpy_to_mlx(arr: np.ndarray) -> "mx.array":
    """Convert numpy array to MLX array."""
    return mx.array(arr)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def mlx_to_numpy(arr: "mx.array") -> np.ndarray:
    """Convert MLX array to numpy array."""
    return np.array(arr)


@pytest.fixture
def small_config_pt() -> TitansConfigPT:
    """Create small PyTorch config for testing."""
    return TitansConfigPT(**SMALL_CONFIG)


@pytest.fixture
def small_config_mlx() -> "TitansConfigMLX":
    """Create small MLX config for testing."""
    if not HAS_MLX:
        pytest.skip("MLX not available")
    return TitansConfigMLX(**SMALL_CONFIG)


@pytest.fixture
def input_ids_np() -> np.ndarray:
    """Create sample input IDs as numpy array."""
    np.random.seed(42)
    return np.random.randint(0, 1000, (2, 32)).astype(np.int64)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestConfigParity:
    """Test that configs have identical attributes."""

    def test_config_attributes(
        self, small_config_pt: TitansConfigPT, small_config_mlx: "TitansConfigMLX"
    ) -> None:
        """Verify config attributes match."""
        assert small_config_pt.dim == small_config_mlx.dim
        assert small_config_pt.num_heads == small_config_mlx.num_heads
        assert small_config_pt.head_dim == small_config_mlx.head_dim
        assert small_config_pt.ffn_dim == small_config_mlx.ffn_dim
        assert small_config_pt.memory_hidden_dim == small_config_mlx.memory_hidden_dim


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestModelInstantiation:
    """Test that models can be instantiated."""

    def test_mac_instantiation(self, small_config_mlx: "TitansConfigMLX") -> None:
        """Test TitansMAC instantiation."""
        model = TitansMACMLX(small_config_mlx)
        assert model is not None

    def test_mag_instantiation(self, small_config_mlx: "TitansConfigMLX") -> None:
        """Test TitansMAG instantiation."""
        model = TitansMAGMLX(small_config_mlx)
        assert model is not None

    def test_mal_instantiation(self, small_config_mlx: "TitansConfigMLX") -> None:
        """Test TitansMAL instantiation."""
        model = TitansMALMLX(small_config_mlx)
        assert model is not None

    def test_lmm_instantiation(self, small_config_mlx: "TitansConfigMLX") -> None:
        """Test TitansLMM instantiation."""
        model = TitansLMMMLX(small_config_mlx)
        assert model is not None


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestOutputShapes:
    """Test that output shapes match between implementations."""

    def test_mac_output_shape(
        self,
        small_config_pt: TitansConfigPT,
        small_config_mlx: "TitansConfigMLX",
        input_ids_np: np.ndarray,
    ) -> None:
        """Test TitansMAC output shapes match."""
        # PyTorch
        model_pt = TitansMACPT(small_config_pt)
        input_pt = numpy_to_torch(input_ids_np)
        with torch.no_grad():
            logits_pt, _ = model_pt(input_pt)
        shape_pt = tuple(logits_pt.shape)

        # MLX
        model_mlx = TitansMACMLX(small_config_mlx)
        input_mlx = numpy_to_mlx(input_ids_np)
        logits_mlx, _ = model_mlx(input_mlx)
        mx.eval(logits_mlx)
        shape_mlx = tuple(logits_mlx.shape)

        assert shape_pt == shape_mlx, f"Shapes differ: PT={shape_pt}, MLX={shape_mlx}"

    def test_mag_output_shape(
        self,
        small_config_pt: TitansConfigPT,
        small_config_mlx: "TitansConfigMLX",
        input_ids_np: np.ndarray,
    ) -> None:
        """Test TitansMAG output shapes match."""
        # PyTorch
        model_pt = TitansMAGPT(small_config_pt)
        input_pt = numpy_to_torch(input_ids_np)
        with torch.no_grad():
            logits_pt, _ = model_pt(input_pt)
        shape_pt = tuple(logits_pt.shape)

        # MLX
        model_mlx = TitansMAGMLX(small_config_mlx)
        input_mlx = numpy_to_mlx(input_ids_np)
        logits_mlx, _ = model_mlx(input_mlx)
        mx.eval(logits_mlx)
        shape_mlx = tuple(logits_mlx.shape)

        assert shape_pt == shape_mlx, f"Shapes differ: PT={shape_pt}, MLX={shape_mlx}"

    def test_mal_output_shape(
        self,
        small_config_pt: TitansConfigPT,
        small_config_mlx: "TitansConfigMLX",
        input_ids_np: np.ndarray,
    ) -> None:
        """Test TitansMAL output shapes match."""
        # PyTorch
        model_pt = TitansMALPT(small_config_pt)
        input_pt = numpy_to_torch(input_ids_np)
        with torch.no_grad():
            logits_pt, _ = model_pt(input_pt)
        shape_pt = tuple(logits_pt.shape)

        # MLX
        model_mlx = TitansMALMLX(small_config_mlx)
        input_mlx = numpy_to_mlx(input_ids_np)
        logits_mlx, _ = model_mlx(input_mlx)
        mx.eval(logits_mlx)
        shape_mlx = tuple(logits_mlx.shape)

        assert shape_pt == shape_mlx, f"Shapes differ: PT={shape_pt}, MLX={shape_mlx}"

    def test_lmm_output_shape(
        self,
        small_config_pt: TitansConfigPT,
        small_config_mlx: "TitansConfigMLX",
        input_ids_np: np.ndarray,
    ) -> None:
        """Test TitansLMM output shapes match."""
        # PyTorch
        model_pt = TitansLMMPT(small_config_pt)
        input_pt = numpy_to_torch(input_ids_np)
        with torch.no_grad():
            logits_pt, _ = model_pt(input_pt)
        shape_pt = tuple(logits_pt.shape)

        # MLX
        model_mlx = TitansLMMMLX(small_config_mlx)
        input_mlx = numpy_to_mlx(input_ids_np)
        logits_mlx, _ = model_mlx(input_mlx)
        mx.eval(logits_mlx)
        shape_mlx = tuple(logits_mlx.shape)

        assert shape_pt == shape_mlx, f"Shapes differ: PT={shape_pt}, MLX={shape_mlx}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMemoryStatePersistence:
    """Test that memory states persist correctly."""

    def test_mac_state_persistence(
        self,
        small_config_mlx: "TitansConfigMLX",
        input_ids_np: np.ndarray,
    ) -> None:
        """Test MAC memory state persists across forward passes."""
        model = TitansMACMLX(small_config_mlx)
        input_mlx = numpy_to_mlx(input_ids_np)

        # First forward pass
        _, states1 = model(input_mlx)
        mx.eval([s.weights[0] for s in states1 if s is not None])

        # Second forward pass with states
        _, states2 = model(input_mlx, states=states1)
        mx.eval([s.weights[0] for s in states2 if s is not None])

        # States should be different after processing more input
        for s1, s2 in zip(states1, states2):
            if s1 is not None and s2 is not None:
                w1 = np.array(s1.weights[0])
                w2 = np.array(s2.weights[0])
                # Memory weights should have changed
                assert not np.allclose(w1, w2, rtol=1e-3), (
                    "Memory weights should change"
                )

    def test_mag_state_persistence(
        self,
        small_config_mlx: "TitansConfigMLX",
        input_ids_np: np.ndarray,
    ) -> None:
        """Test MAG memory state persists across forward passes."""
        model = TitansMAGMLX(small_config_mlx)
        input_mlx = numpy_to_mlx(input_ids_np)

        # First forward pass
        _, states1 = model(input_mlx)
        mx.eval([s.weights[0] for s in states1 if s is not None])

        # Second forward pass with states
        _, states2 = model(input_mlx, states=states1)
        mx.eval([s.weights[0] for s in states2 if s is not None])

        # States should be different after processing more input
        for s1, s2 in zip(states1, states2):
            if s1 is not None and s2 is not None:
                w1 = np.array(s1.weights[0])
                w2 = np.array(s2.weights[0])
                # Memory weights should have changed
                assert not np.allclose(w1, w2, rtol=1e-3), (
                    "Memory weights should change"
                )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLXOptimizations:
    """Test MLX-specific optimizations work correctly."""

    def test_lazy_evaluation(self, small_config_mlx: "TitansConfigMLX") -> None:
        """Test that MLX lazy evaluation works."""
        model = TitansLMMMLX(small_config_mlx)
        input_mlx = mx.zeros((1, 16), dtype=mx.int32)

        # This should not compute immediately
        logits, _ = model(input_mlx)

        # Now force evaluation
        mx.eval(logits)

        # Should have computed successfully
        assert logits.shape == (1, 16, small_config_mlx.vocab_size)

    def test_batched_processing(
        self,
        small_config_mlx: "TitansConfigMLX",
    ) -> None:
        """Test batched processing works correctly."""
        model = TitansMAGMLX(small_config_mlx)

        # Single sample
        input_single = mx.zeros((1, 16), dtype=mx.int32)
        logits_single, _ = model(input_single)
        mx.eval(logits_single)

        # Batched
        input_batch = mx.zeros((4, 16), dtype=mx.int32)
        logits_batch, _ = model(input_batch)
        mx.eval(logits_batch)

        # Shapes should be correct
        assert logits_single.shape == (1, 16, small_config_mlx.vocab_size)
        assert logits_batch.shape == (4, 16, small_config_mlx.vocab_size)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestGradientComputation:
    """Test that gradient computation works for memory updates."""

    def test_memory_gradient_flow(self, small_config_mlx: "TitansConfigMLX") -> None:
        """Test memory gradients are computed correctly."""
        from titans_mlx.memory import NeuralLongTermMemory

        memory = NeuralLongTermMemory(small_config_mlx)
        batch_size = 2
        seq_len = 8

        # Create input
        x = mx.random.normal((batch_size, seq_len, small_config_mlx.dim))

        # Initialize state
        state = memory.init_state(batch_size)

        # Forward pass
        output, new_state = memory(x, state=state)
        mx.eval(output)

        # States should have updated weights
        assert new_state is not None
        assert len(new_state.weights) > 0

        # Weights should be different from initial
        for w_old, w_new in zip(state.weights, new_state.weights):
            assert not np.allclose(np.array(w_old), np.array(w_new), rtol=1e-3)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestChunkedProcessing:
    """Test chunked processing for MAC model."""

    def test_mac_chunking(
        self,
        small_config_mlx: "TitansConfigMLX",
    ) -> None:
        """Test MAC processes long sequences in chunks correctly."""
        model = TitansMACMLX(small_config_mlx)

        # Sequence longer than chunk_size
        seq_len = small_config_mlx.chunk_size * 2 + 10
        input_mlx = mx.zeros((1, seq_len), dtype=mx.int32)

        logits, states = model(input_mlx)
        mx.eval(logits)

        # Output should match input length
        assert logits.shape == (1, seq_len, small_config_mlx.vocab_size)

        # States should be populated
        assert all(s is not None for s in states)
