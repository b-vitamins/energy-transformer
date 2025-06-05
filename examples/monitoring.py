"""Examples of Energy Transformer monitoring and optimization."""

import torch
from torch.utils.tensorboard import SummaryWriter

from energy_transformer.layers import (
    EnergyLayerNorm,
    HopfieldNetwork,
    MultiheadEnergyAttention,
)
from energy_transformer.models import EnergyTransformer
from energy_transformer.utils import (
    SGD,
    AdaptiveGD,
    EnergyTracker,
    Momentum,
    make_convergence_hook,
    make_logger_hook,
    make_tensorboard_hook,
)


def example_basic_monitoring():
    """Basic energy monitoring during optimization."""
    # Create model
    model = EnergyTransformer(
        EnergyLayerNorm(128),
        MultiheadEnergyAttention(128, 8),
        HopfieldNetwork(128, hidden_dim=512),
        steps=20,
        optimizer=SGD(alpha=0.1),
    )

    # Add simple logger
    handle = model.register_step_hook(make_logger_hook(log_every=5))

    # Run
    x = torch.randn(4, 50, 128)
    model(x)

    # Clean up
    handle.remove()


def example_component_tracking():
    """Track attention vs hopfield energy contributions."""
    model = EnergyTransformer(
        EnergyLayerNorm(128),
        MultiheadEnergyAttention(128, 8),
        HopfieldNetwork(128),
        steps=15,
    )

    # Track energy components
    tracker = EnergyTracker()
    model.register_step_hook(lambda _m, info, tr=tracker: tr.update(info))

    # Run
    x = torch.randn(8, 100, 128)
    model(x)

    # Analyze
    trajectory = tracker.get_trajectory()
    print(f"Attention energy: {trajectory['attention_energy'][-1]:.4f}")
    print(f"Hopfield energy: {trajectory['hopfield_energy'][-1]:.4f}")

    # Get batch statistics
    stats = tracker.get_batch_statistics()
    print(f"Energy variance across batch: {stats['energy_std']:.4f}")


def example_convergence_detection():
    """Detect when optimization converges."""
    model = EnergyTransformer(
        EnergyLayerNorm(256),
        MultiheadEnergyAttention(256, 8),
        HopfieldNetwork(256),
        steps=50,  # Max steps
    )

    converged_at = None

    def on_convergence(step):
        nonlocal converged_at
        converged_at = step
        print(f"Converged at step {step}")

    # Add convergence detection
    model.register_step_hook(
        make_convergence_hook(on_convergence, window=5, threshold=1e-4)
    )

    x = torch.randn(4, 64, 256)
    model(x)

    if converged_at:
        print(
            f"Optimization converged early at step {converged_at}/{model.steps}"
        )


def example_tensorboard_logging():
    """Log to TensorBoard for visualization."""
    writer = SummaryWriter("runs/energy_transformer")

    model = EnergyTransformer(
        EnergyLayerNorm(128),
        MultiheadEnergyAttention(128, 8),
        HopfieldNetwork(128),
        optimizer=Momentum(alpha=0.1, momentum=0.9),
    )

    # Add TensorBoard logging
    handle = model.register_step_hook(make_tensorboard_hook(writer))

    # Run multiple batches
    for batch_idx in range(10):
        x = torch.randn(16, 50, 128)
        out = model(x)

        # Log final output norm
        writer.add_scalar("output_norm", out.norm().item(), batch_idx)

    writer.close()
    handle.remove()


def example_custom_optimizer():
    """Use different optimizers and compare."""
    configs = [
        ("SGD", SGD(alpha=0.1)),
        ("Momentum", Momentum(alpha=0.1, momentum=0.9)),
        ("Adaptive", AdaptiveGD(alpha=0.1)),
    ]

    x = torch.randn(8, 100, 128)

    for name, optimizer in configs:
        model = EnergyTransformer(
            EnergyLayerNorm(128),
            MultiheadEnergyAttention(128, 8),
            HopfieldNetwork(128),
            steps=20,
            optimizer=optimizer,
        )

        tracker = EnergyTracker()
        model.register_step_hook(lambda _m, info, tr=tracker: tr.update(info))

        model(x.clone())
        trajectory = tracker.get_trajectory()

        final_energy = trajectory["total_energy"][-1]
        print(f"{name}: Final energy = {final_energy:.6f}")


if __name__ == "__main__":
    print("Running Energy Transformer monitoring examples...")
    example_basic_monitoring()
    example_component_tracking()
    example_convergence_detection()
    example_custom_optimizer()
    print("Done!")
