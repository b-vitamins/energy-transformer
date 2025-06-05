import pytest
import torch

from energy_transformer.utils.observers import (
    StepInfo,
    make_tensorboard_hook,
    make_wandb_hook,
)

pytestmark = pytest.mark.unit


class DummyWriter:
    def __init__(self) -> None:
        self.logged: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.logged.append((tag, float(value), step))


def _make_step(iteration: int) -> StepInfo:
    return StepInfo(
        iteration=iteration,
        total_energy=torch.tensor(2.0),
        attention_energy=torch.tensor(0.5),
        hopfield_energy=torch.tensor(0.5),
        grad_norm=torch.tensor(1.0),
        step_size=torch.tensor(0.1),
    )


def test_tensorboard_hook_logs_scalars() -> None:
    writer = DummyWriter()
    hook = make_tensorboard_hook(writer, tag_prefix="test")
    hook(None, _make_step(3))
    tags = {t for t, _, _ in writer.logged}
    assert "test/total_energy" in tags
    assert "test/attention_ratio" in tags


class DummyRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, float]] = []

    def log(self, metrics: dict[str, float], **_unused: int) -> None:
        self.logged.append(metrics)


def test_wandb_hook_logs_metrics() -> None:
    run = DummyRun()
    hook = make_wandb_hook(run, prefix="test")
    hook(None, _make_step(4))
    assert run.logged
    metrics = run.logged[0]
    assert metrics["test/total_energy"] == pytest.approx(2.0)
    assert "test/energy_ratio_att" in metrics
