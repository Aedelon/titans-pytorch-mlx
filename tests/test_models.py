# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""Tests for Titans model variants."""

import torch

from titans.config import TitansConfig
from titans.models import (
    FeedForward,
    LMMBlock,
    MACBlock,
    MAGBlock,
    MALBlock,
    RMSNorm,
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_forward(self) -> None:
        """Test RMS normalization."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)

        y = norm(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_output_scale(self) -> None:
        """Test output is properly scaled."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64) * 10  # Large values

        y = norm(x)

        # RMS norm should produce values with similar RMS to 1
        rms = torch.sqrt(torch.mean(y**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)


class TestFeedForward:
    """Tests for FeedForward network."""

    def test_forward(self, default_config: TitansConfig) -> None:
        """Test FFN forward pass."""
        ffn = FeedForward(default_config)
        x = torch.randn(2, 16, default_config.dim)

        y = ffn(x)

        assert y.shape == x.shape

    def test_dimensions(self, default_config: TitansConfig) -> None:
        """Test FFN hidden dimensions."""
        ffn = FeedForward(default_config)

        assert ffn.gate_proj.in_features == default_config.dim
        assert ffn.gate_proj.out_features == default_config.ffn_dim
        assert ffn.down_proj.in_features == default_config.ffn_dim
        assert ffn.down_proj.out_features == default_config.dim


# =============================================================================
# MAC Block and Model Tests
# =============================================================================


class TestMACBlock:
    """Tests for MACBlock."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = MACBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        output, state = block(x)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = MACBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        _, state1 = block(x)
        output, state2 = block(x, state=state1)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansMAC:
    """Tests for TitansMAC model."""

    def test_forward(self, small_config: TitansConfig, batch_size: int) -> None:
        """Test full forward pass."""
        model = TitansMAC(small_config)
        seq_len = small_config.chunk_size * 2  # Multiple chunks
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansMAC(small_config)
        input_ids = torch.randint(
            0, small_config.vocab_size, (batch_size, small_config.chunk_size)
        )

        _, states1 = model(input_ids)
        logits2, states2 = model(input_ids, states=states1)

        assert logits2.shape[0] == batch_size
        assert len(states2) == small_config.num_layers

    def test_chunking(self, small_config: TitansConfig, batch_size: int) -> None:
        """Test sequence is processed in chunks."""
        model = TitansMAC(small_config)
        chunk_size = small_config.chunk_size
        seq_len = chunk_size * 3  # Three chunks

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansMAC(small_config)

        assert model.head.weight is model.embed.weight


# =============================================================================
# MAG Block and Model Tests
# =============================================================================


class TestMAGBlock:
    """Tests for MAGBlock."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = MAGBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        output, state = block(x)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = MAGBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        _, state1 = block(x)
        output, state2 = block(x, state=state1)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansMAG:
    """Tests for TitansMAG model."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test full forward pass."""
        model = TitansMAG(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansMAG(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        _, states1 = model(input_ids)
        logits2, states2 = model(input_ids, states=states1)

        assert logits2.shape[0] == batch_size

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansMAG(small_config)

        assert model.head.weight is model.embed.weight


# =============================================================================
# MAL Block and Model Tests
# =============================================================================


class TestMALBlock:
    """Tests for MALBlock."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = MALBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        output, state = block(x)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = MALBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        _, state1 = block(x)
        output, state2 = block(x, state=state1)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansMAL:
    """Tests for TitansMAL model."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test full forward pass."""
        model = TitansMAL(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansMAL(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        _, states1 = model(input_ids)
        logits2, states2 = model(input_ids, states=states1)

        assert logits2.shape[0] == batch_size

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansMAL(small_config)

        assert model.head.weight is model.embed.weight


# =============================================================================
# LMM Block and Model Tests
# =============================================================================


class TestLMMBlock:
    """Tests for LMMBlock (standalone memory)."""

    def test_forward_without_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward without initial state."""
        block = LMMBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        output, state = block(x)

        assert output.shape == x.shape
        assert state is not None

    def test_forward_with_state(
        self, default_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with existing state."""
        block = LMMBlock(default_config)
        x = torch.randn(batch_size, seq_len, default_config.dim)

        _, state1 = block(x)
        output, state2 = block(x, state=state1)

        assert output.shape == x.shape
        assert state2 is not None


class TestTitansLMM:
    """Tests for TitansLMM model (memory only)."""

    def test_forward(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test full forward pass."""
        model = TitansLMM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(states) == small_config.num_layers

    def test_forward_with_states(
        self, small_config: TitansConfig, batch_size: int, seq_len: int
    ) -> None:
        """Test forward with continuing states."""
        model = TitansLMM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        _, states1 = model(input_ids)
        logits2, states2 = model(input_ids, states=states1)

        assert logits2.shape[0] == batch_size

    def test_weight_tying(self, small_config: TitansConfig) -> None:
        """Test embedding and output weights are tied."""
        model = TitansLMM(small_config)

        assert model.head.weight is model.embed.weight


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelsIntegration:
    """Integration tests for all model variants."""

    def test_all_models_produce_valid_logits(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test all models produce valid logits."""
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        models = [
            TitansMAC(small_config),
            TitansMAG(small_config),
            TitansMAL(small_config),
            TitansLMM(small_config),
        ]

        for model in models:
            logits, _ = model(input_ids)

            assert not torch.isnan(logits).any(), f"{type(model).__name__} produced NaN"
            assert not torch.isinf(logits).any(), f"{type(model).__name__} produced Inf"

    def test_all_models_return_states(
        self, small_config: TitansConfig, batch_size: int
    ) -> None:
        """Test all models return memory states."""
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        models = [
            TitansMAC(small_config),
            TitansMAG(small_config),
            TitansMAL(small_config),
            TitansLMM(small_config),
        ]

        for model in models:
            _, states = model(input_ids)

            assert states is not None, f"{type(model).__name__} returned None states"
            assert len(states) == small_config.num_layers

    def test_gradients_flow(self, small_config: TitansConfig, batch_size: int) -> None:
        """Test gradients flow through models."""
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        models = [
            TitansMAC(small_config),
            TitansMAG(small_config),
            TitansMAL(small_config),
            TitansLMM(small_config),
        ]

        for model in models:
            logits, _ = model(input_ids)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, small_config.vocab_size),
                targets.view(-1),
            )

            # Backprop
            loss.backward()

            # Check gradients exist for key model parameters
            # Note: NeuralLongTermMemory parameters and PersistentMemory parameters
            # don't receive gradients via backprop by design:
            # - Memory state is detached after forward pass
            # - Memory updates happen via test-time learning with _compute_gradients()
            # - Persistent memory tokens are data-independent
            has_embed_grad = model.embed.weight.grad is not None
            has_norm_grad = model.norm.weight.grad is not None

            assert has_embed_grad, f"{type(model).__name__}.embed has no gradient"
            assert has_norm_grad, f"{type(model).__name__}.norm has no gradient"

            # Check attention/ffn layers have gradients (not memory-related)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Skip all memory-related parameters
                    if ".memory" in name or ".persistent" in name:
                        continue
                    assert param.grad is not None, (
                        f"{type(model).__name__}.{name} has no gradient"
                    )
