"""Checkpoint utilities for loading and saving models."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    model: Module,
    filepath: str | Path,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    **kwargs: Any,
) -> None:
    """
    Save model checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    filepath : str or Path
        Destination path for the checkpoint file.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to save state from.
    epoch : int, optional
        Epoch number to record.
    **kwargs
        Additional metadata to include.
    """
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        **kwargs,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str | Path,
    model: Module | None = None,
    optimizer: Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a PyTorch model checkpoint.

    Parameters
    ----------
    filepath : str or Path
        Path to the checkpoint file.
    model : torch.nn.Module, optional
        Model to load state into.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into.
    device : torch.device, optional
        Device on which to map the checkpoint. Defaults to CPU.

    Returns
    -------
    checkpoint : dict[str, Any]
        Dictionary containing the checkpoint data.
    """
    map_location = device or torch.device("cpu")
    checkpoint: dict[str, Any]
    try:
        # PyTorch >=2.0 supports weights_only argument
        checkpoint = torch.load(
            filepath, map_location=map_location, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(filepath, map_location=map_location)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def load_jax_checkpoint(
    filepath: str | Path,
    model: Module,
    device: torch.device | None = None,
) -> None:
    """
    Load checkpoint from JAX/NumPy format into a PyTorch model.

    Parameters
    ----------
    filepath : str or Path
        Path to the NumPy checkpoint (.npz) file.
    model : torch.nn.Module
        PyTorch model to load weights into.
    device : torch.device, optional
        Device to map tensors to. Defaults to CPU.
    """
    map_location = device or torch.device("cpu")
    data: Any = np.load(filepath)
    state_dict: dict[str, Tensor] = {}

    # Map numpy arrays to PyTorch model keys
    if "wk" in data:
        state_dict["transformer.attention.w_k"] = torch.from_numpy(data["wk"])
    if "wq" in data:
        state_dict["transformer.attention.w_q"] = torch.from_numpy(data["wq"])
    if "xi" in data:
        state_dict["transformer.hopfield.xi"] = torch.from_numpy(data["xi"])
    if "Wenc" in data:
        state_dict["encoder.weight"] = torch.from_numpy(data["Wenc"].T)
    if "benc" in data:
        state_dict["encoder.bias"] = torch.from_numpy(data["benc"])
    if "wdec" in data:
        state_dict["decoder.weight"] = torch.from_numpy(data["wdec"].T)
    if "bdec" in data:
        state_dict["decoder.bias"] = torch.from_numpy(data["bdec"])
    if "lnorm_gamma" in data:
        gamma = torch.from_numpy(data["lnorm_gamma"])
        state_dict["transformer.layer_norm.gamma"] = gamma
        state_dict["output_norm.gamma"] = gamma
    if "lnorm_bias" in data:
        delta = torch.from_numpy(data["lnorm_bias"])
        state_dict["transformer.layer_norm.delta"] = delta
        state_dict["output_norm.delta"] = delta
    if "cls_token" in data:
        state_dict["cls_token"] = torch.from_numpy(data["cls_token"])
    if "mask_token" in data:
        state_dict["mask_token"] = torch.from_numpy(data["mask_token"])
    if "pos_embed" in data:
        state_dict["pos_embed"] = torch.from_numpy(data["pos_embed"])

    # Transfer to device and load
    for key, tensor in state_dict.items():
        state_dict[key] = tensor.to(map_location)
    model.load_state_dict(state_dict, strict=False)

    print(f"JAX checkpoint loaded from {filepath}")
