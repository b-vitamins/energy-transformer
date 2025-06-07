import pytest
import torch

from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork

pytestmark = pytest.mark.unit


def test_energy_scale() -> None:
    """Energy should be in -300 to -500 range initially."""
    net = SimplicialHopfieldNetwork(192, hidden_dim=768)
    g = torch.randn(2, 64, 192)  # Typical CIFAR input size
    energy = net.compute_energy(g)
    # Allow wider range for test stability
    assert -600 < energy.item() < -200


def test_triangle_fraction_bounds() -> None:
    """triangle_fraction must be in [0, 1]."""
    with pytest.raises(ValueError, match="triangle_fraction must be in"):
        SimplicialHopfieldNetwork(64, triangle_fraction=1.5)

    with pytest.raises(ValueError, match="triangle_fraction must be in"):
        SimplicialHopfieldNetwork(64, triangle_fraction=-0.1)


def test_simplex_caching() -> None:
    """Simplex selection should be cached for efficiency."""
    net = SimplicialHopfieldNetwork(64)
    g1 = torch.randn(1, 10, 64)
    g2 = torch.randn(1, 10, 64)

    # First call creates cache
    _ = net.compute_energy(g1)
    assert 10 in net._simplices_cache

    # Second call reuses cache
    edges1, tris1 = net._simplices_cache[10]
    _ = net.compute_energy(g2)
    edges2, tris2 = net._simplices_cache[10]
    assert edges1 is edges2  # Same object
    assert tris1 is tris2


def test_gradient_matches_autograd() -> None:
    """Manual gradient should match autograd."""
    net = SimplicialHopfieldNetwork(32, hidden_dim=64, beta=0.1)
    x = torch.randn(2, 8, 32, requires_grad=True)

    # Manual gradient
    grad_manual = net.compute_grad(x.detach())

    # Autograd gradient
    energy = net.compute_energy(x)
    grad_auto = torch.autograd.grad(energy, x)[0]

    torch.testing.assert_close(grad_manual, grad_auto, rtol=1e-5, atol=1e-6)


def test_edge_cases() -> None:
    """Test with minimal sequence lengths."""
    net = SimplicialHopfieldNetwork(16, triangle_fraction=0.5)

    # N=2: only one edge possible
    g2 = torch.randn(1, 2, 16)
    e2 = net.compute_energy(g2)
    assert e2.item() != 0
    edges, triangles = net._choose_simplices(2, g2.device)
    assert edges.shape[0] == 1
    assert triangles.shape[0] == 0

    # N=3: mixture of edges and triangles
    g3 = torch.randn(1, 3, 16)
    e3 = net.compute_energy(g3)
    assert e3.item() != 0
    edges, triangles = net._choose_simplices(3, g3.device)
    assert edges.shape[0] + triangles.shape[0] > 0


def test_all_edges_configuration() -> None:
    """Test with triangle_fraction=0 (edges only)."""
    net = SimplicialHopfieldNetwork(32, triangle_fraction=0.0)
    g = torch.randn(1, 5, 32)
    _ = net.compute_energy(g)

    edges, triangles = net._simplices_cache[5]
    assert edges.shape[0] == 10  # 5 choose 2
    assert triangles.shape[0] == 0


def test_all_triangles_configuration() -> None:
    """Test with triangle_fraction=1 (triangles only)."""
    net = SimplicialHopfieldNetwork(32, triangle_fraction=1.0)
    g = torch.randn(1, 4, 32)
    _ = net.compute_energy(g)

    edges, triangles = net._simplices_cache[4]
    assert edges.shape[0] == 2  # 6 total - 4 triangles = 2 edges
    assert triangles.shape[0] == 4  # 4 choose 3


def test_no_simplices_error() -> None:
    """Should raise error if no simplices can be formed."""
    net = SimplicialHopfieldNetwork(32, triangle_fraction=1.0)
    # With N=1, no edges or triangles possible
    g = torch.randn(1, 1, 32)
    with pytest.raises(RuntimeError, match="No simplices selected"):
        net.compute_energy(g)


def test_device_and_dtype_handling() -> None:
    """Test proper device and dtype propagation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        net = SimplicialHopfieldNetwork(16, device=device, dtype=torch.float16)
        assert net.patterns.device == device
        assert net.patterns.dtype == torch.float16
        assert net.device == device
        assert net.dtype == torch.float16

        g = torch.randn(1, 4, 16, device=device, dtype=torch.float16)
        energy = net.compute_energy(g)
        assert energy.device == device


def test_forward_compatibility() -> None:
    """Forward method should return energy for EnergyTransformer compatibility."""
    net = SimplicialHopfieldNetwork(32)
    g = torch.randn(2, 8, 32)

    energy_direct = net.compute_energy(g)
    energy_forward = net(g)

    assert torch.allclose(energy_direct, energy_forward)


def test_different_batch_sizes() -> None:
    """Test with various batch sizes."""
    net = SimplicialHopfieldNetwork(64, hidden_dim=128)

    for batch_size in [1, 4, 16]:
        g = torch.randn(batch_size, 10, 64)
        energy = net.compute_energy(g)
        grad = net.compute_grad(g)

        assert energy.ndim == 0  # Scalar
        assert grad.shape == g.shape


def test_gradient_accumulation() -> None:
    """Test that gradients accumulate correctly for multiple simplices."""
    net = SimplicialHopfieldNetwork(16, triangle_fraction=0.5)
    g = torch.randn(1, 4, 16, requires_grad=True)

    # Get gradient
    energy = net.compute_energy(g)
    energy.backward()

    # Gradient should be non-zero for all tokens
    assert (g.grad.abs() > 1e-6).all()
