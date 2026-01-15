# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for Neural Long-term Memory module."""

import pytest
import torch

from titans.config import TitansConfig
from titans.memory import (
    MemoryMLP,
    MemoryState,
    NeuralLongTermMemory,
    get_activation,
)


class TestGetActivation:
    """Tests for get_activation utility."""

    def test_valid_activations(self) -> None:
        """Test valid activation functions."""
        assert isinstance(get_activation("silu"), torch.nn.SiLU)
        assert isinstance(get_activation("gelu"), torch.nn.GELU)
        assert isinstance(get_activation("relu"), torch.nn.ReLU)

    def test_invalid_activation(self) -> None:
        """Test invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("invalid")


class TestMemoryState:
    """Tests for MemoryState dataclass."""

    def test_creation(self) -> None:
        """Test MemoryState creation."""
        weights = [torch.randn(64, 64)]
        momentum = [torch.zeros(64, 64)]
        state = MemoryState(weights=weights, momentum=momentum)

        assert len(state.weights) == 1
        assert len(state.momentum) == 1

    def test_detach(self) -> None:
        """Test detach creates new tensors without gradients."""
        weights = [torch.randn(64, 64, requires_grad=True)]
        momentum = [torch.randn(64, 64, requires_grad=True)]
        state = MemoryState(weights=weights, momentum=momentum)

        detached = state.detach()

        assert not detached.weights[0].requires_grad
        assert not detached.momentum[0].requires_grad
        # Original should be unchanged
        assert weights[0].requires_grad

    def test_clone(self) -> None:
        """Test clone creates independent copy."""
        weights = [torch.randn(64, 64)]
        momentum = [torch.randn(64, 64)]
        state = MemoryState(weights=weights, momentum=momentum)

        cloned = state.clone()

        # Modify original
        state.weights[0][0, 0] = 999.0

        # Clone should be unchanged
        assert cloned.weights[0][0, 0] != 999.0


class TestMemoryMLP:
    """Tests for MemoryMLP module."""

    def test_linear_memory(self, default_config: TitansConfig) -> None:
        """Test linear memory (single layer)."""
        config = TitansConfig(
            dim=default_config.dim,
            num_memory_layers=1,
        )
        mlp = MemoryMLP(config)

        assert len(mlp.layers) == 1

        x = torch.randn(2, 16, config.dim)
        y = mlp(x)

        assert y.shape == x.shape

    def test_deep_memory(self, default_config: TitansConfig) -> None:
        """Test deep memory (multiple layers)."""
        config = TitansConfig(
            dim=default_config.dim,
            num_memory_layers=3,
            memory_hidden_mult=2.0,
        )
        mlp = MemoryMLP(config)

        assert len(mlp.layers) == 3

        x = torch.randn(2, 16, config.dim)
        y = mlp(x)

        assert y.shape == x.shape

    def test_get_weights(self, default_config: TitansConfig) -> None:
        """Test get_weights returns cloned weights."""
        mlp = MemoryMLP(default_config)
        weights = mlp.get_weights()

        assert len(weights) == default_config.num_memory_layers

        # Modify returned weights
        weights[0][0, 0] = 999.0

        # Original should be unchanged
        assert mlp.layers[0].weight[0, 0] != 999.0

    def test_set_weights(self, default_config: TitansConfig) -> None:
        """Test set_weights updates weights."""
        mlp = MemoryMLP(default_config)

        new_weights = [torch.randn_like(layer.weight) for layer in mlp.layers]
        mlp.set_weights(new_weights)

        for i, layer in enumerate(mlp.layers):
            assert torch.allclose(layer.weight.data, new_weights[i])

    def test_compute_loss(self, default_config: TitansConfig) -> None:
        """Test associative memory loss computation."""
        mlp = MemoryMLP(default_config)

        keys = torch.randn(2, 8, default_config.dim)
        values = torch.randn(2, 8, default_config.dim)

        loss = mlp.compute_loss(keys, values)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0


class TestNeuralLongTermMemory:
    """Tests for NeuralLongTermMemory module."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward pass without initial state."""
        memory = NeuralLongTermMemory(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        output, state = memory(x)

        assert output.shape == x.shape
        assert state is not None
        assert len(state.weights) == default_config.num_memory_layers

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward pass with existing state."""
        memory = NeuralLongTermMemory(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        # First forward
        _, state1 = memory(x)

        # Second forward with state
        output2, state2 = memory(x, state=state1)

        assert output2.shape == x.shape
        assert state2 is not None

    def test_forward_no_return_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without returning state."""
        memory = NeuralLongTermMemory(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        output, state = memory(x, return_state=False)

        assert output.shape == x.shape
        assert state is None

    def test_init_state(self, default_config: TitansConfig, batch_size: int) -> None:
        """Test memory state initialization."""
        memory = NeuralLongTermMemory(default_config)
        device = torch.device("cpu")

        state = memory.init_state(batch_size, device)

        assert len(state.weights) == default_config.num_memory_layers
        assert len(state.momentum) == default_config.num_memory_layers

        # Momentum should be zeros
        for m in state.momentum:
            assert torch.allclose(m, torch.zeros_like(m))

    def test_retrieve(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test memory retrieval without update."""
        memory = NeuralLongTermMemory(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        # Get initial state
        state = memory.init_state(batch_size, x.device)

        # Retrieve
        retrieved = memory.retrieve(x, state)

        assert retrieved.shape == x.shape

    def test_with_conv(self, batch_size: int, seq_len: int) -> None:
        """Test memory with convolution enabled."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            use_conv=True,
            conv_kernel_size=4,
            num_memory_layers=2,
        )
        memory = NeuralLongTermMemory(config)
        x = torch.randn(batch_size, seq_len, config.dim)

        output, state = memory(x)

        assert output.shape == x.shape
        assert state is not None

    def test_without_conv(self, batch_size: int, seq_len: int) -> None:
        """Test memory without convolution."""
        config = TitansConfig(
            dim=64,
            num_heads=4,
            use_conv=False,
            num_memory_layers=2,
        )
        memory = NeuralLongTermMemory(config)
        x = torch.randn(batch_size, seq_len, config.dim)

        output, state = memory(x)

        assert output.shape == x.shape

    def test_memory_update_changes_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test that memory updates change the state."""
        memory = NeuralLongTermMemory(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        _, state1 = memory(x)
        _, state2 = memory(x, state=state1)

        # States should be different after update
        assert not torch.allclose(state1.weights[0], state2.weights[0])

    def test_gradient_computation(self, small_config: TitansConfig) -> None:
        """Test gradient computation for memory update."""
        memory = NeuralLongTermMemory(small_config)

        keys = torch.randn(2, 8, small_config.dim)
        values = torch.randn(2, 8, small_config.dim)

        grads = memory._compute_gradients(keys, values)

        assert len(grads) == small_config.num_memory_layers
        for g in grads:
            assert g is not None
            assert not torch.isnan(g).any()
