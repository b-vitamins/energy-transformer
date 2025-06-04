"""Regression tests for ViSET bug fixes."""

import torch
from energy_transformer.models.vision import viset_2l_e50_t50_cifar


def test_energy_scale_is_reasonable():
    model = viset_2l_e50_t50_cifar(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    energies = []

    def capture_energy(_module, _input, output):
        energies.append(output.item())

    for block in model.et_blocks:
        block.hopfield.register_forward_hook(capture_energy)

    with torch.no_grad():
        model(x, et_kwargs={"detach": True})

    assert energies, "No energies captured"
    for e in energies:
        assert torch.isfinite(torch.tensor(e))
        assert abs(e) < 1e6


def test_simplices_cover_all_tokens():
    model = viset_2l_e50_t50_cifar(num_classes=100)
    hopfield = model.et_blocks[0].hopfield
    assert hopfield.max_vertex == 64
    has_cls = False
    for tensor in hopfield.simps_by_size.values():
        if (tensor == 0).any():
            has_cls = True
            break
    assert has_cls


def test_forward_pass_completes():
    model = viset_2l_e50_t50_cifar(num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x, et_kwargs={"detach": True})
    assert out.shape == (4, 100)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
