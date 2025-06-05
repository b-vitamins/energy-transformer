import pytest
import torch

from energy_transformer.utils.optimizers import SGD, AdaptiveGD, Momentum

pytestmark = pytest.mark.unit


def test_sgd_step_and_reset() -> None:
    opt = SGD(alpha=0.5)
    grad = torch.ones(2, 3)
    step, step_size = opt.step(grad, grad, grad.sum(), 0)
    assert torch.allclose(step, grad * 0.5)
    assert step_size.item() == pytest.approx(0.5)
    opt.reset()  # should not raise


def test_momentum_accumulates_and_resets() -> None:
    opt = Momentum(alpha=0.5, momentum=0.9)
    grad1 = torch.tensor([[1.0, -1.0]])
    step1, _ = opt.step(grad1, grad1, torch.tensor(0.0), 0)
    assert torch.allclose(step1, grad1 * 0.5)

    grad2 = torch.tensor([[2.0, -2.0]])
    step2, _ = opt.step(grad2, grad2, torch.tensor(0.0), 1)
    expected_vel = 0.9 * step1 + 0.5 * grad2
    assert torch.allclose(step2, expected_vel)

    opt.reset()
    assert opt.velocity is None


def test_adaptivegd_scales_by_norm() -> None:
    opt = AdaptiveGD(alpha=2.0, eps=0.0)
    grad = torch.tensor([[3.0, 4.0]])
    step, step_alpha = opt.step(grad, grad, grad.sum(), 0)
    grad_norm = grad.norm(p=2, dim=-1, keepdim=True)
    expected_alpha = opt.alpha / grad_norm
    assert torch.allclose(step, expected_alpha * grad)
    assert torch.allclose(step_alpha, expected_alpha.mean())
    opt.reset()  # no internal state
