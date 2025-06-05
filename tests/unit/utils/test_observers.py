import numpy as np
import pytest
import torch

from energy_transformer.utils.observers import (
    EnergyTracker,
    StepInfo,
    make_convergence_hook,
    make_logger_hook,
)

pytestmark = pytest.mark.unit


def _make_step(iteration: int, energy: float) -> StepInfo:
    return StepInfo(
        iteration=iteration,
        total_energy=torch.tensor(energy),
        attention_energy=torch.tensor(energy / 2),
        hopfield_energy=torch.tensor(energy / 4),
        grad_norm=torch.tensor(1.0),
        step_size=torch.tensor(0.1),
    )


def test_stepinfo_to_dict() -> None:
    info = _make_step(1, 2.0)
    d = info.to_dict()
    assert d["iteration"] == 1
    assert d["total_energy"] == pytest.approx(2.0)
    assert d["step_size"] == pytest.approx(0.1)


def test_energy_tracker_convergence() -> None:
    tracker = EnergyTracker(window_size=3)
    tracker.convergence_threshold = 1e-6
    for i in range(3):
        tracker.update(_make_step(i, 1.0))
    assert tracker.is_converged()


def test_batch_statistics_and_trajectory() -> None:
    tracker = EnergyTracker()
    tracker.update(_make_step(0, 1.0))
    stats = tracker.get_batch_statistics()
    assert "energy_mean" in stats
    traj = tracker.get_trajectory()
    assert isinstance(traj["total_energy"], np.ndarray)


def test_make_logger_hook(capsys: pytest.CaptureFixture[str]) -> None:
    hook = make_logger_hook(log_every=2)
    hook(None, _make_step(2, 1.0))
    captured = capsys.readouterr().out
    assert "Step 2" in captured


def test_make_convergence_hook() -> None:
    called: list[int] = []

    def cb(step: int) -> None:
        called.append(step)

    hook = make_convergence_hook(cb, window=2, threshold=1e-6)
    for i in range(2):
        hook(None, _make_step(i, 1.0))
    assert called == [1]
