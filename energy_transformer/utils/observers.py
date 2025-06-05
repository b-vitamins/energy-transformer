"""Observation utilities for Energy Transformer monitoring.

Provides hooks and utilities for monitoring energy descent dynamics,
including component-wise energy tracking and convergence detection.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class StepInfo:
    """Information about a single optimization step."""

    iteration: int
    total_energy: Tensor
    attention_energy: Tensor
    hopfield_energy: Tensor
    grad_norm: Tensor
    step_size: Tensor | None
    tokens: Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        d = {
            "iteration": self.iteration,
            "total_energy": self.total_energy.item(),
            "attention_energy": self.attention_energy.item(),
            "hopfield_energy": self.hopfield_energy.item(),
            "grad_norm": self.grad_norm.item(),
        }
        if self.step_size is not None:
            d["step_size"] = self.step_size.item()
        return d


class EnergyTracker:
    """Tracks energy components during optimization.

    Provides batch-level statistics and convergence detection.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: list[StepInfo] = []
        self.convergence_threshold = 1e-5

    def update(self, info: StepInfo) -> None:
        """Add a new step to the history."""
        self.history.append(info)

    def is_converged(self) -> bool:
        """Check if optimization has converged.

        Returns True if the relative change in energy over the
        last window_size steps is below threshold.
        """
        if len(self.history) < self.window_size:
            return False

        recent = self.history[-self.window_size :]
        energies = torch.stack([s.total_energy for s in recent])

        # Compute relative change
        rel_change = (energies[:-1] - energies[1:]).abs() / (
            energies[:-1].abs() + 1e-8
        )
        return rel_change.max() < self.convergence_threshold

    def get_batch_statistics(self) -> dict[str, Tensor]:
        """Get statistics over the current batch."""
        if not self.history:
            return {}

        last_step = self.history[-1]
        return {
            "energy_mean": last_step.total_energy.mean(),
            "energy_std": last_step.total_energy.std(),
            "attention_mean": last_step.attention_energy.mean(),
            "attention_std": last_step.attention_energy.std(),
            "hopfield_mean": last_step.hopfield_energy.mean(),
            "hopfield_std": last_step.hopfield_energy.std(),
            "grad_norm_mean": last_step.grad_norm.mean(),
            "grad_norm_std": last_step.grad_norm.std(),
        }

    def get_trajectory(self) -> dict[str, np.ndarray]:
        """Get complete trajectory of all tracked values."""
        if not self.history:
            return {}

        return {
            "iterations": np.array([s.iteration for s in self.history]),
            "total_energy": torch.stack(
                [s.total_energy.mean() for s in self.history]
            )
            .cpu()
            .numpy(),
            "attention_energy": torch.stack(
                [s.attention_energy.mean() for s in self.history]
            )
            .cpu()
            .numpy(),
            "hopfield_energy": torch.stack(
                [s.hopfield_energy.mean() for s in self.history]
            )
            .cpu()
            .numpy(),
            "grad_norm": torch.stack([s.grad_norm.mean() for s in self.history])
            .cpu()
            .numpy(),
        }


# Hook factories for common monitoring tasks


def make_logger_hook(log_fn: Callable = print, log_every: int = 1) -> Callable:
    """Create a hook that logs energy values.

    Parameters
    ----------
    log_fn : callable
        Logging function (default: print)
    log_every : int
        Log every N steps

    Returns
    -------
    callable
        Hook function
    """

    def hook(_module: nn.Module, info: StepInfo) -> None:
        if info.iteration % log_every == 0:
            log_fn(
                f"Step {info.iteration}: "
                f"E_total={info.total_energy.mean():.6f} "
                f"(att={info.attention_energy.mean():.6f}, "
                f"hop={info.hopfield_energy.mean():.6f}) "
                f"grad={info.grad_norm.mean():.6f}"
            )

    return hook


def make_tensorboard_hook(
    writer: SummaryWriter, tag_prefix: str = "et", global_step_offset: int = 0
) -> Callable:
    """Create a hook for TensorBoard logging.

    Parameters
    ----------
    writer : SummaryWriter
        TensorBoard writer instance
    tag_prefix : str
        Prefix for all tags
    global_step_offset : int
        Offset for global step counter

    Returns
    -------
    callable
        Hook function
    """

    def hook(_module: nn.Module, info: StepInfo) -> None:
        global_step = global_step_offset + info.iteration
        stats = info.to_dict()

        # Log scalars
        for key, value in stats.items():
            if key != "iteration":
                writer.add_scalar(f"{tag_prefix}/{key}", value, global_step)

        # Log energy breakdown
        att_ratio = info.attention_energy.mean() / (
            info.total_energy.mean() + 1e-8
        )
        writer.add_scalar(
            f"{tag_prefix}/attention_ratio", att_ratio, global_step
        )

    return hook


def make_convergence_hook(
    callback: Callable[[int], None], window: int = 10, threshold: float = 1e-5
) -> Callable:
    """Create a hook that detects convergence.

    Parameters
    ----------
    callback : callable
        Function to call when convergence is detected
    window : int
        Window size for convergence check
    threshold : float
        Convergence threshold

    Returns
    -------
    callable
        Hook function
    """
    tracker = EnergyTracker(window_size=window)
    tracker.convergence_threshold = threshold

    def hook(_module: nn.Module, info: StepInfo) -> None:
        tracker.update(info)
        if tracker.is_converged():
            callback(info.iteration)

    return hook


def make_wandb_hook(run: object, prefix: str = "et") -> Callable:
    """Create a hook for Weights & Biases logging.

    Parameters
    ----------
    run : wandb.Run
        W&B run object
    prefix : str
        Prefix for all metrics

    Returns
    -------
    callable
        Hook function
    """

    def hook(_module: nn.Module, info: StepInfo) -> None:
        metrics = {
            f"{prefix}/{k}": v
            for k, v in info.to_dict().items()
            if k != "iteration"
        }
        metrics[f"{prefix}/energy_ratio_att"] = (
            info.attention_energy.mean() / (info.total_energy.mean() + 1e-8)
        ).item()
        run.log(metrics, step=info.iteration)

    return hook
