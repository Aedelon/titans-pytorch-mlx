# Titans: Learning to Memorize at Test Time

A PyTorch implementation of the Titans architecture from Google Research.

Titans introduce a **Neural Long-term Memory (LMM)** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. This enables attention mechanisms to focus on local context while utilizing long-range information through neural memory.

## Paper Reference

> Behrouz, A., Zhong, P., & Mirrokni, V. (2024). Titans: Learning to Memorize at Test Time. arXiv preprint arXiv:2501.00663

## Features

- Complete implementation of all three Titans variants (MAC, MAG, MAL)
- Standalone Neural Long-term Memory module
- Production-ready training and inference scripts
- 99% test coverage
- Ready-to-use examples

## Architecture

The implementation includes three variants of Titans:

| Variant | Description |
|---------|-------------|
| **MAC** (Memory as Context) | Memory retrieval concatenated with input before attention |
| **MAG** (Memory as Gate) | Memory and attention combined via learned gating |
| **MAL** (Memory as Layer) | Memory used as a preprocessing layer before attention |

Plus a standalone **LMM** module (memory only, without attention).

### Key Components

- **Neural Long-term Memory**: Deep MLP that learns key-value associations through test-time gradient updates
- **Persistent Memory**: Learnable data-independent tokens prepended to sequences
- **Sliding Window Attention**: Local attention with configurable window size (MAG, MAL)
- **Segmented Attention**: Full causal attention within segments (MAC)

## Installation

```bash
uv sync
```

For development:

```bash
uv sync --all-extras
```

## Quick Start

### Training

```bash
# Train MAC model with synthetic data (demo)
uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

# Train with custom data
uv run python scripts/pretrain.py --model mag --data path/to/data.txt --epochs 50

# Resume training from checkpoint
uv run python scripts/pretrain.py --model mac --resume checkpoints/latest.pt
```

### Inference

```bash
# Generate text from trained model
uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Hello" --max-tokens 100

# Interactive mode
uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --interactive
```

### Examples

```bash
# Run basic usage examples
uv run python examples/basic_usage.py

# Run long sequence processing examples
uv run python examples/long_sequence.py
```

## Usage

### Basic Usage with TitansMAC

```python
import torch
from titans import TitansConfig, TitansMAC

# Configuration
config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    vocab_size=32000,
    chunk_size=512,
    num_persistent_tokens=16,
    num_memory_layers=2,
)

# Create model
model = TitansMAC(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 1024))  # (batch, seq)
logits, states = model(input_ids)

# Continue with states for next chunk
input_ids_next = torch.randint(0, config.vocab_size, (2, 512))
logits_next, states_next = model(input_ids_next, states=states)
```

### Using TitansMAG (Memory as Gate)

```python
from titans import TitansConfig, TitansMAG

config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    window_size=256,  # Sliding window size for local attention
)

model = TitansMAG(config)
input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
```

### Using TitansMAL (Memory as Layer)

```python
from titans import TitansConfig, TitansMAL

config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
)

model = TitansMAL(config)
input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
```

### Standalone Neural Memory Module

```python
from titans import TitansConfig, NeuralLongTermMemory

config = TitansConfig(
    dim=512,
    num_memory_layers=2,
    memory_hidden_mult=4.0,
    memory_lr=0.1,
    memory_momentum=0.9,
    memory_decay=0.01,
)

memory = NeuralLongTermMemory(config)

# Forward pass with memory update
x = torch.randn(2, 100, 512)  # (batch, seq, dim)
output, state = memory(x)

# Continue with state
x_next = torch.randn(2, 50, 512)
output2, state2 = memory(x_next, state=state)

# Retrieve without updating
retrieved = memory.retrieve(queries, state2)
```

## Configuration

Key configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 512 | Model dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 12 | Number of Titans blocks |
| `num_memory_layers` | 2 | Depth of neural memory MLP (L_M >= 1) |
| `memory_hidden_mult` | 4.0 | Hidden dimension multiplier for memory |
| `num_persistent_tokens` | 16 | Number of persistent memory tokens |
| `chunk_size` | 512 | Segment size for MAC variant |
| `window_size` | 512 | Sliding window size for MAG/MAL |
| `memory_lr` | 0.1 | Learning rate for memory updates (θ) |
| `memory_momentum` | 0.9 | Momentum coefficient (η) |
| `memory_decay` | 0.01 | Weight decay/forgetting factor (α) |

## Memory Update Equations

The neural memory is updated using the following equations from the paper:

**Surprise (momentum)**:
```
S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
```

**Memory update**:
```
M_t = (1 - α_t) * M_{t-1} + S_t
```

**Associative memory loss**:
```
ℓ(M; x) = ||M(k) - v||²
```

where:
- `α_t`: Forgetting/decay factor (weight decay)
- `η_t`: Surprise decay (momentum coefficient)
- `θ_t`: Learning rate for momentary surprise
- `k, v`: Key-value pairs derived from input

## Project Structure

```
titans/
├── src/titans/
│   ├── __init__.py         # Public API exports
│   ├── config.py           # TitansConfig dataclass
│   ├── memory.py           # Neural Long-term Memory
│   ├── attention.py        # Attention modules (SWA, Segmented)
│   ├── persistent.py       # Persistent Memory
│   └── models.py           # Full model implementations
├── scripts/
│   ├── pretrain.py         # Training script
│   └── inference.py        # Generation script
├── examples/
│   ├── basic_usage.py      # Basic API examples
│   └── long_sequence.py    # Long sequence processing
└── tests/                  # Unit tests (99% coverage)
```

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### With Coverage

```bash
uv run pytest tests/ --cov=titans --cov-report=term-missing
```

### Linting

```bash
uv run ruff check src/titans/ tests/ scripts/ examples/
uv run ruff format src/titans/ tests/ scripts/ examples/
```

## License

Apache License 2.0

Copyright (c) 2024 Delanoe Pirard / Aedelon
