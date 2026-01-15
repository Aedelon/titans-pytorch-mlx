# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
HuggingFace Hub integration for Titans models.

This module provides utilities for:
1. Pushing trained Titans models to HuggingFace Hub
2. Loading Titans models from HuggingFace Hub
3. Converting between checkpoint formats

Usage:
    # Push to Hub
    from titans.hub import push_to_hub
    push_to_hub(
        checkpoint_path="checkpoints/best_model.pt",
        repo_id="username/titans-mac-340m",
        tokenizer_name="meta-llama/Llama-2-7b-hf",
    )

    # Load from Hub
    from titans.hub import load_from_hub
    model, tokenizer = load_from_hub("username/titans-mac-340m")
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import torch

# Optional imports
try:
    from huggingface_hub import HfApi, hf_hub_download, upload_folder

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


def _get_model_classes() -> dict:
    """Get model classes lazily to avoid circular imports."""
    from titans.config import TitansConfig
    from titans.models import TitansLMM, TitansMAC, TitansMAG, TitansMAL

    return {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }


def push_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    tokenizer_name: str | None = None,
    private: bool = False,
    commit_message: str = "Upload Titans model",
    token: str | None = None,
    create_model_card: bool = True,
) -> str:
    """Push a Titans model to HuggingFace Hub.

    Args:
        checkpoint_path: Path to model checkpoint
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        tokenizer_name: Name of tokenizer used during training
        private: Whether to make the repo private
        commit_message: Commit message for the upload
        token: HuggingFace API token (uses cached token if None)
        create_model_card: Whether to create a model card

    Returns:
        URL of the uploaded model

    Raises:
        ImportError: If huggingface_hub is not installed
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub required. Install with: pip install huggingface_hub"
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    model_type = checkpoint["model_type"]

    # Create temporary directory for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save model weights
        model_path = tmpdir / "pytorch_model.bin"
        torch.save(checkpoint["model_state_dict"], model_path)

        # Save config
        config_path = tmpdir / "config.json"
        full_config = {
            "model_type": model_type,
            "titans_config": config_dict,
            "tokenizer_name": tokenizer_name or checkpoint.get("tokenizer_name"),
        }
        with open(config_path, "w") as f:
            json.dump(full_config, f, indent=2)

        # Create model card
        if create_model_card:
            card_path = tmpdir / "README.md"
            _create_model_card(
                card_path,
                repo_id,
                model_type,
                config_dict,
                tokenizer_name,
            )

        # Upload to Hub
        api = HfApi(token=token)

        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
        )

        # Upload folder
        upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
        )

        logger.info(f"Model uploaded to: https://huggingface.co/{repo_id}")

    return f"https://huggingface.co/{repo_id}"


def load_from_hub(
    repo_id: str,
    revision: str | None = None,
    device: str | torch.device = "auto",
    token: str | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Load a Titans model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID
        revision: Git revision (branch, tag, or commit hash)
        device: Device to load model on
        token: HuggingFace API token

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If required libraries are not installed
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub required. Install with: pip install huggingface_hub"
        )

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Download files
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        token=token,
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="pytorch_model.bin",
        revision=revision,
        token=token,
    )

    # Load config
    with open(config_path) as f:
        full_config = json.load(f)

    model_type = full_config["model_type"]
    config_dict = full_config["titans_config"]
    tokenizer_name = full_config.get("tokenizer_name")

    # Create model
    from titans.config import TitansConfig

    config = TitansConfig(**config_dict)
    model_classes = _get_model_classes()
    model_class = model_classes.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_class(config)

    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = None
    if tokenizer_name and HAS_TRANSFORMERS:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loaded {model_type.upper()} model from {repo_id}")

    return model, tokenizer


def convert_checkpoint_to_hub_format(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    tokenizer_name: str | None = None,
) -> Path:
    """Convert a training checkpoint to Hub-compatible format.

    Args:
        checkpoint_path: Path to training checkpoint
        output_dir: Output directory
        tokenizer_name: Name of tokenizer used

    Returns:
        Path to output directory
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights only
    weights_path = output_dir / "pytorch_model.bin"
    torch.save(checkpoint["model_state_dict"], weights_path)

    # Save config
    config_path = output_dir / "config.json"
    full_config = {
        "model_type": checkpoint["model_type"],
        "titans_config": checkpoint["config"],
        "tokenizer_name": tokenizer_name or checkpoint.get("tokenizer_name"),
    }
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    logger.info(f"Converted checkpoint to Hub format in {output_dir}")

    return output_dir


def _create_model_card(
    path: Path,
    repo_id: str,
    model_type: str,
    config: dict,
    tokenizer_name: str | None,
) -> None:
    """Create a model card for HuggingFace Hub."""
    # Extract key config values
    dim = config.get("dim", "?")
    num_layers = config.get("num_layers", "?")
    num_heads = config.get("num_heads", "?")
    vocab_size = config.get("vocab_size", "?")

    # Estimate parameters
    try:
        params = _estimate_params(config, model_type)
        params_str = f"{params / 1e6:.0f}M" if params > 1e6 else f"{params / 1e3:.0f}K"
    except Exception:
        params_str = "?"

    card = f"""---
library_name: titans
tags:
- titans
- neural-memory
- long-context
- pytorch
license: apache-2.0
---

# {repo_id.split("/")[-1]}

A **Titans {model_type.upper()}** model trained with neural long-term memory.

## Model Description

This is a Titans model implementing the {model_type.upper()} (Memory as {"Context" if model_type == "mac" else "Gate" if model_type == "mag" else "Layer" if model_type == "mal" else "Memory"}) architecture from the paper ["Titans: Learning to Memorize at Test Time"](https://arxiv.org/abs/2501.00663).

Titans introduce a **Neural Long-term Memory (LMM)** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay.

## Model Details

| Parameter | Value |
|-----------|-------|
| Model Type | {model_type.upper()} |
| Hidden Size | {dim} |
| Num Layers | {num_layers} |
| Num Heads | {num_heads} |
| Vocab Size | {vocab_size} |
| Parameters | ~{params_str} |

## Usage

```python
from titans.hub import load_from_hub

model, tokenizer = load_from_hub("{repo_id}")

# Generate text
input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]
logits, states = model(input_ids)
```

## Training

{"Tokenizer: " + tokenizer_name if tokenizer_name else "Character-level tokenization"}

## Citation

```bibtex
@article{{behrouz2024titans,
  title={{Titans: Learning to Memorize at Test Time}},
  author={{Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab}},
  journal={{arXiv preprint arXiv:2501.00663}},
  year={{2024}}
}}
```

## License

Apache 2.0
"""

    with open(path, "w") as f:
        f.write(card)


def _estimate_params(config: dict, model_type: str) -> int:
    """Estimate number of parameters from config."""
    dim = config.get("dim", 512)
    num_layers = config.get("num_layers", 12)
    config.get("num_heads", 8)
    vocab_size = config.get("vocab_size", 32000)
    num_memory_layers = config.get("num_memory_layers", 2)
    memory_hidden_mult = config.get("memory_hidden_mult", 4.0)

    # Embedding
    params = vocab_size * dim

    # Per layer
    # Attention: Q, K, V, O projections
    attn_params = 4 * dim * dim
    # Memory: MLP layers
    memory_hidden = int(dim * memory_hidden_mult)
    if num_memory_layers == 1:
        memory_params = dim * dim
    else:
        memory_params = (
            dim * memory_hidden
            + (num_memory_layers - 2) * memory_hidden * memory_hidden
            + memory_hidden * dim
        )

    # Memory projections and gates
    memory_extras = 3 * dim * dim + 3 * dim * dim  # proj_k, proj_v, proj_q + gates

    layer_params = attn_params + memory_params + memory_extras

    params += num_layers * layer_params

    # Output projection
    params += dim * vocab_size

    return params
