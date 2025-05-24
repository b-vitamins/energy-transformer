import math
import warnings

import pytest
import torch

from energy_transformer.layers.hopfield import (
    ActivationFunction,
    HopfieldNetwork,
    get_energy_function,
    get_energy_transformer_init_std,
    he_scaled_init_std,
    quadratic_energy,
    register_energy_function,
)


def test_he_scaled_init_std_formula() -> None:
    assert math.isclose(he_scaled_init_std(4), math.sqrt(2.0 / 4))
    assert math.isclose(he_scaled_init_std(4, 8), math.sqrt(2.0 / 8))
    assert math.isclose(
        get_energy_transformer_init_std(3, 6), he_scaled_init_std(3, 6)
    )


def test_get_energy_function_values() -> None:
    h = torch.tensor([1.0, -2.0])
    relu_fn = get_energy_function(ActivationFunction.RELU)
    softmax_fn = get_energy_function(ActivationFunction.SOFTMAX)
    power_fn = get_energy_function(ActivationFunction.POWER)
    tanh_fn = get_energy_function(ActivationFunction.TANH)

    assert torch.allclose(relu_fn(h), -0.5 * (torch.relu(h) ** 2).sum())
    assert torch.allclose(softmax_fn(h), -torch.exp(h).sum())
    assert torch.allclose(power_fn(h), -(h.pow(4).mean()))
    assert torch.allclose(
        tanh_fn(h), -torch.sum(torch.log(torch.cosh(h.clamp(-10, 10))))
    )


def test_register_energy_function_decorator() -> None:
    @register_energy_function("dummy")
    def dummy_energy(h: torch.Tensor) -> torch.Tensor:
        return -h.sum()

    assert "dummy" in HopfieldNetwork.list_available_energy_functions()
    assert dummy_energy(torch.tensor([1.0, 2.0])) == -3.0


def test_hopfieldnetwork_custom_activation_requires_energy_fn() -> None:
    with pytest.raises(ValueError):
        HopfieldNetwork(in_dim=1, activation=ActivationFunction.CUSTOM)


def test_hopfieldnetwork_warning_energy_fn_ignored() -> None:
    with warnings.catch_warnings(record=True) as w:
        HopfieldNetwork(
            in_dim=1,
            activation=ActivationFunction.RELU,
            energy_fn=lambda x: x.sum(),
        )
        assert any(
            "energy_fn provided but activation is not CUSTOM" in str(wi.message)
            for wi in w
        )


def test_hopfieldnetwork_forward_quadratic_energy() -> None:
    net = HopfieldNetwork(
        in_dim=2,
        hidden_dim=1,
        bias=False,
        activation=ActivationFunction.CUSTOM,
        energy_fn=quadratic_energy,
    )
    with torch.no_grad():
        net.ξ.fill_(1.0)
    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_h = g @ net.ξ.T
    expected_energy = quadratic_energy(expected_h)
    result = net(g)
    assert torch.allclose(result, expected_energy)
