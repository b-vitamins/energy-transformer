"""Direct comparison tests with JAX implementation behavior."""

import numpy as np
import torch

from energy_transformer import (
    EnergyTransformer,
    ETConfig,
    ImageEnergyTransformer,
    ImageETConfig,
)
from energy_transformer.layers import (
    EnergyAttention,
    EnergyLayerNorm,
    HopfieldNetwork,
)


class TestJAXParityValidation:
    """Validate PyTorch implementation against expected JAX behavior"""

    def test_initialization_patterns(self) -> None:
        """Test that initialization matches JAX patterns"""
        torch.manual_seed(42)

        # LayerNorm initialization
        ln = EnergyLayerNorm(64)
        assert torch.allclose(
            ln.gamma, torch.ones(()), atol=1e-6
        )  # Scalar gamma
        assert ln.delta.shape == (64,)  # Vector bias

        # Attention initialization
        attn = EnergyAttention(64, 4, 16)
        # JAX uses nn.initializers.normal(0.002)
        assert attn.wq.std().item() < 0.1  # Should be small initialization
        assert attn.wk.std().item() < 0.1

        # Hopfield initialization
        hopfield = HopfieldNetwork(64, 256)
        assert hopfield.xi.std().item() < 0.1

    def test_einsum_compatibility(self) -> None:
        """Test that einsum operations match JAX patterns exactly"""
        torch.manual_seed(42)

        # Attention einsums
        g = torch.randn(10, 64)  # No batch
        wq = torch.randn(4, 16, 64)

        # JAX pattern: "qd,hzd->qhz"
        q_jax_pattern = torch.einsum("qd,hzd->qhz", g, wq)

        # Our pattern should handle both
        g_batch = g.unsqueeze(0)  # Add batch
        q_our_pattern = torch.einsum("...qd,hzd->...qhz", g_batch, wq)

        # Should match (modulo batch dimension)
        assert torch.allclose(q_jax_pattern, q_our_pattern[0], atol=1e-6)

    def test_energy_components(self) -> None:
        """Test that energy components match expected mathematical forms"""
        torch.manual_seed(42)

        # Create components
        ln = EnergyLayerNorm(32)
        attn = EnergyAttention(32, 2, 16)
        hopfield = HopfieldNetwork(32, 64)

        x = torch.randn(4, 8, 32)

        # Layer norm energy
        ln_energy = ln.energy(x)
        g = ln(x)

        # Attention energy (should be negative)
        attn_energy = attn.energy(g)
        assert attn_energy <= 0, "Attention energy must be non-positive"

        # Hopfield energy (should be negative)
        hop_energy = hopfield.energy(g)
        assert hop_energy <= 0, "Hopfield energy must be non-positive"

        # Total energy
        total = ln_energy + attn_energy + hop_energy

        print(f"LayerNorm energy: {ln_energy:.6f}")
        print(f"Attention energy: {attn_energy:.6f}")
        print(f"Hopfield energy: {hop_energy:.6f}")
        print(f"Total energy: {total:.6f}")

    def test_gradient_descent_behavior(self) -> None:
        """Test that gradient descent behaves like JAX implementation"""
        torch.manual_seed(42)

        config = ETConfig(
            d_model=64, n_heads=4, d_head=16, n_steps=20, alpha=0.1
        )
        model = EnergyTransformer(config)

        x = torch.randn(1, 10, 64)

        # Manual gradient descent (matching JAX style)
        x_manual = x.clone().detach().requires_grad_(True)
        manual_energies = []

        for _ in range(config.n_steps):
            # JAX style: E, dEdg = jax.value_and_grad(et.energy)(g)
            energy = model.energy(x_manual)
            manual_energies.append(energy.item())

            # JAX style: x = x - alpha * dEdg
            grad = torch.autograd.grad(energy, x_manual)[0]
            with torch.no_grad():
                x_manual = x_manual - config.alpha * grad
            x_manual = x_manual.detach().requires_grad_(True)

        # Model forward pass
        x_model, model_energies = model.compute_energy_trajectory(x)

        # Energies should match
        for i, (e1, e2) in enumerate(
            zip(manual_energies, model_energies, strict=False)
        ):
            assert abs(e1 - e2) < 1e-5, (
                f"Energy mismatch at step {i}: {e1} vs {e2}"
            )

        # Final states should match
        assert torch.allclose(x_manual, x_model, atol=1e-5)

    def test_masking_behavior(self) -> None:
        """Test that masking behavior matches JAX implementation"""
        torch.manual_seed(42)

        config = ImageETConfig(image_shape=(3, 64, 64), patch_size=16, n_mask=8)
        model = ImageEnergyTransformer(config)

        # Create test data
        images = torch.randn(2, 3, 64, 64)
        masks = torch.zeros(2, 16, dtype=torch.bool)
        masks[:, :8] = True  # First 8 patches masked

        # Apply masking
        results = model(images, mask=masks)

        # Check that masked patches are different
        patches_orig = model.patcher.patchify(images)
        patches_recon = model.patcher.patchify(results["reconstruction"])

        # Masked patches should be different
        masked_diff = (patches_orig[:, :8] - patches_recon[:, :8]).abs().mean()
        assert masked_diff > 0.1, (
            "Masked patches should be reconstructed differently"
        )

        # Unmasked patches should be similar (not exact due to processing)
        unmasked_diff = (
            (patches_orig[:, 8:] - patches_recon[:, 8:]).abs().mean()
        )
        assert unmasked_diff < 0.1, "Unmasked patches should be preserved"

    def test_numerical_equivalence(self) -> None:
        """Test numerical equivalence of key operations"""
        torch.manual_seed(42)
        np.random.seed(42)

        # Test LayerNorm equivalence
        ln = EnergyLayerNorm(64, eps=1e-5)
        x = torch.randn(2, 10, 64)

        # Our implementation
        out_torch = ln(x)

        # Manual calculation (matching JAX)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        var = (x_centered**2).mean(dim=-1, keepdim=True)
        out_manual = ln.gamma * x_centered / torch.sqrt(var + ln.eps)
        if ln.use_bias:
            out_manual = out_manual + ln.delta

        assert torch.allclose(out_torch, out_manual, atol=1e-6)

    def test_energy_minimization_properties(self) -> None:
        """Test that energy minimization has expected properties"""
        torch.manual_seed(42)

        config = ETConfig(
            d_model=64, n_heads=4, d_head=16, n_steps=100, alpha=0.05
        )
        model = EnergyTransformer(config)

        x = torch.randn(1, 10, 64)

        # Run minimization
        x_final, energies = model.compute_energy_trajectory(x)

        # Energy should generally decrease
        energy_diffs = np.diff(energies.numpy())
        decreases = energy_diffs < 0
        assert decreases.sum() > len(decreases) * 0.6, (
            "Energy should mostly decrease"
        )

        # Final energy should be lower than initial
        assert energies[-1] < energies[0], "Final energy should be lower"

        # Should reach approximate fixed point (relaxed threshold)
        final_diffs = abs(energy_diffs[-10:])
        assert final_diffs.mean() < 0.5, (
            "Should converge to approximate fixed point"
        )


def run_parity_tests():
    """Run all parity validation tests"""
    print("Running JAX Parity Validation Tests")
    print("=" * 40)

    validator = TestJAXParityValidation()

    tests = [
        ("Initialization patterns", validator.test_initialization_patterns),
        ("Einsum compatibility", validator.test_einsum_compatibility),
        ("Energy components", validator.test_energy_components),
        ("Gradient descent behavior", validator.test_gradient_descent_behavior),
        ("Masking behavior", validator.test_masking_behavior),
        ("Numerical equivalence", validator.test_numerical_equivalence),
        (
            "Energy minimization properties",
            validator.test_energy_minimization_properties,
        ),
    ]

    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        try:
            test_func()
            print(f"✓ {test_name} passed")
        except AssertionError as e:
            print(f"✗ {test_name} failed: {e}")
        except Exception as e:
            print(f"✗ {test_name} error: {e}")

    print("\n" + "=" * 40)
    print("Parity validation complete")


if __name__ == "__main__":
    run_parity_tests()
