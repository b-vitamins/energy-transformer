"""Unit tests for base Energy Transformer components."""

import pytest
import torch
from torch import Tensor

from energy_transformer.layers import (
    HopfieldNetwork,
    LayerNorm,
    MultiHeadEnergyAttention,
    SimplicialHopfieldNetwork,
)
from energy_transformer.layers.base import (
    BaseEnergyAttention,
    BaseHopfieldNetwork,
    BaseLayerNorm,
)


class TestBaseClasses:
    """Test suite for base Energy Transformer component classes."""

    def test_inheritance(self) -> None:
        """Test that concrete implementations inherit from base classes."""
        # Test LayerNorm
        assert issubclass(LayerNorm, BaseLayerNorm)
        assert isinstance(LayerNorm(768), BaseLayerNorm)

        # Test MultiHeadEnergyAttention
        assert issubclass(MultiHeadEnergyAttention, BaseEnergyAttention)
        assert isinstance(MultiHeadEnergyAttention(768), BaseEnergyAttention)

        # Test HopfieldNetwork
        assert issubclass(HopfieldNetwork, BaseHopfieldNetwork)
        assert isinstance(HopfieldNetwork(768), BaseHopfieldNetwork)

        # Test SimplicialHopfieldNetwork
        assert issubclass(SimplicialHopfieldNetwork, BaseHopfieldNetwork)
        assert isinstance(SimplicialHopfieldNetwork(768), BaseHopfieldNetwork)

    def test_abstract_methods(self) -> None:
        """Test that abstract base classes cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseLayerNorm()

        with pytest.raises(TypeError):
            BaseEnergyAttention()

        with pytest.raises(TypeError):
            BaseHopfieldNetwork()

    def test_custom_implementation(self, device: torch.device) -> None:
        """Test creating custom implementations of base classes."""

        class CustomLayerNorm(BaseLayerNorm):
            def forward(self, x: Tensor) -> Tensor:
                # Simple implementation that does nothing
                return x

        class CustomEnergyAttention(BaseEnergyAttention):
            def forward(self, g: Tensor) -> Tensor:
                # Simple implementation that returns a scalar
                return torch.tensor(0.0, device=g.device)

        class CustomHopfieldNetwork(BaseHopfieldNetwork):
            def forward(self, g: Tensor) -> Tensor:
                # Simple implementation that returns a scalar
                return torch.tensor(0.0, device=g.device)

        # Test that these custom implementations can be instantiated
        custom_norm = CustomLayerNorm().to(device)
        custom_attention = CustomEnergyAttention().to(device)
        custom_hopfield = CustomHopfieldNetwork().to(device)

        # Test forward passes
        x = torch.randn(2, 3, 768, device=device)

        # Custom norm should return the input unchanged
        norm_output = custom_norm(x)
        assert torch.allclose(norm_output, x)

        # Custom attention should return a scalar
        attention_output = custom_attention(x)
        assert attention_output.ndim == 0
        assert attention_output.item() == 0.0

        # Custom hopfield should return a scalar
        hopfield_output = custom_hopfield(x)
        assert hopfield_output.ndim == 0
        assert hopfield_output.item() == 0.0

    def test_incomplete_implementation(self) -> None:
        """Test that incomplete implementations raise TypeError."""

        # Define classes in a way that won't raise immediately to test
        # instantiation
        class IncompleteLayerNorm(BaseLayerNorm):
            pass

        class IncompleteEnergyAttention(BaseEnergyAttention):
            pass

        class IncompleteHopfieldNetwork(BaseHopfieldNetwork):
            pass

        # Test that instantiating these incomplete classes raises TypeError
        with pytest.raises(TypeError):
            IncompleteLayerNorm()

        with pytest.raises(TypeError):
            IncompleteEnergyAttention()

        with pytest.raises(TypeError):
            IncompleteHopfieldNetwork()


class TestBaseInterfaces:
    """Test suite for ensuring base interfaces are consistent."""

    def test_forward_signatures(self) -> None:
        """Test forward method signature consistent across implementations."""
        # Get forward methods of concrete implementations
        layer_norm_forward = LayerNorm.forward
        attention_forward = MultiHeadEnergyAttention.forward
        hopfield_forward = HopfieldNetwork.forward

        # Check that parameter counts match
        # Don't check parameter names as implementations may vary in naming
        assert (
            layer_norm_forward.__code__.co_argcount >= 2
        )  # self + at least one arg
        assert (
            attention_forward.__code__.co_argcount >= 2
        )  # self + at least one arg
        assert (
            hopfield_forward.__code__.co_argcount >= 2
        )  # self + at least one arg

        # Check return type annotations if available
        if hasattr(layer_norm_forward, "__annotations__"):
            assert "return" in layer_norm_forward.__annotations__
            assert layer_norm_forward.__annotations__["return"] == Tensor

        if hasattr(attention_forward, "__annotations__"):
            assert "return" in attention_forward.__annotations__
            assert attention_forward.__annotations__["return"] == Tensor

        if hasattr(hopfield_forward, "__annotations__"):
            assert "return" in hopfield_forward.__annotations__
            assert hopfield_forward.__annotations__["return"] == Tensor
