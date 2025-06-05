"""Vision Simplicial Energy Transformer (ViSET) implementation.

This module implements the Vision Simplicial Energy Transformer, which enhances
the Vision Energy Transformer by replacing standard Hopfield Networks with
Simplicial Hopfield Networks that leverage spatial topology through higher-order
interactions.

ViSET combines insights from two key papers:
1. "Energy Transformer" (Hoover et al., 2023) - Energy-based transformers
2. "Simplicial Hopfield Networks" (Burns & Fukai, 2023) - Topology-aware memory

Classes
-------
VisionSimplicialEnergyTransformer
    Extends VisionEnergyTransformer with simplicial Hopfield networks

Factory Functions
-----------------
viset_tiny, viset_small, viset_base
    Standard configurations with topology-aware simplices
viset_2l_e50_t50_cifar
    2-layer model with 50% edges, 50% triangles
viset_2l_e100_cifar
    2-layer model with edges only (k-NN graph)
viset_2l_t100_cifar
    2-layer model with triangles only (Delaunay)
viset_2l_random_cifar
    Baseline with random simplices (no topology)
viset_2l_e40_t40_tet20_cifar
    Experimental config with tetrahedra

Example
-------
>>> # Create topology-aware ViSET for CIFAR-100
>>> model = viset_2l_e50_t50_cifar(num_classes=100)
>>>
>>> # Compare with random baseline
>>> baseline = viset_2l_random_cifar(num_classes=100)
>>>
>>> # Process images
>>> images = torch.randn(32, 3, 32, 32)
>>> logits = model(images)  # Benefits from spatial structure

Naming Convention
-----------------
Model names follow the pattern: ViSET-{depth}L-{simplex_distribution}
- ViSET-2L-E50-T50: 2 layers, 50% edges, 50% triangles
- ViSET-2L-E100: 2 layers, 100% edges
- ViSET-2L-Random: 2 layers, random topology

References
----------
.. [1] Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H.,
       Zaki, M. J., & Krotov, D. (2023). Energy Transformer.
       arXiv preprint arXiv:2302.07253.
.. [2] Burns, T. & Fukai, T. (2023). Simplicial Hopfield networks.
       arXiv preprint arXiv:2305.05179.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from energy_transformer.layers.attention import MultiheadEnergyAttention
from energy_transformer.layers.layer_norm import EnergyLayerNorm
from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork
from energy_transformer.models.base import EnergyTransformer
from energy_transformer.models.vision.viet import VisionEnergyTransformer
from energy_transformer.utils.optimizers import SGD


class VisionSimplicialEnergyTransformer(VisionEnergyTransformer):
    """Vision Simplicial Energy Transformer (ViSET).

    Extends VisionEnergyTransformer by replacing regular Hopfield Networks
    with Simplicial Hopfield Networks that can leverage spatial topology
    through k-NN graphs and Delaunay triangulation.

    Parameters
    ----------
    img_size : int
        Input image size (assumed square).
    patch_size : int
        Size of image patches (assumed square).
    in_chans : int
        Number of input channels.
    num_classes : int
        Number of output classes.
    embed_dim : int
        Embedding dimension.
    depth : int
        Number of Energy Transformer blocks.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    hopfield_hidden_dim : int
        Hidden dimension for Simplicial Hopfield networks.
    et_steps : int
        Number of energy optimization steps per block.
    et_alpha : float
        Step size for energy optimization.
    drop_rate : float
        Dropout rate.
    representation_size : int | None
        Size of representation layer before classification head.
    use_topology : bool
        Whether to use topology-aware simplex generation.
    simplex_budget : float
        Fraction of full edge budget to use.
    simplex_max_dim : int
        Maximum simplex dimension (1=edges, 2=triangles, etc).
    simplex_dim_weights : dict[int, float] | None
        Weight distribution across simplex dimensions.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        hopfield_hidden_dim: int,
        et_steps: int,
        et_alpha: float,
        drop_rate: float = 0.0,
        _representation_size: int | None = None,
        # Simplicial-specific parameters
        use_topology: bool = True,
        simplex_budget: float = 0.15,
        simplex_max_dim: int = 2,
        simplex_dim_weights: dict[int, float] | None = None,
    ) -> None:
        """Initialize Vision Simplicial Energy Transformer.

        Parameters
        ----------
        img_size : int
            Input image size (assumed square).
        patch_size : int
            Size of image patches (assumed square).
        in_chans : int
            Number of input channels.
        num_classes : int
            Number of output classes.
        embed_dim : int
            Embedding dimension.
        depth : int
            Number of Energy Transformer blocks.
        num_heads : int
            Number of attention heads.
        head_dim : int
            Dimension of each attention head.
        hopfield_hidden_dim : int
            Hidden dimension for Simplicial Hopfield networks.
        et_steps : int
            Number of energy optimization steps per block.
        et_alpha : float
            Step size for energy optimization.
        drop_rate : float, optional
            Dropout rate. Default is 0.0.
        representation_size : int | None, optional
            Size of representation layer before classification head.
            If None, uses embed_dim directly.
        use_topology : bool, optional
            Whether to use topology-aware simplex generation.
            Default is True.
        simplex_budget : float, optional
            Fraction of full edge budget to use. Default is 0.15.
        simplex_max_dim : int, optional
            Maximum simplex dimension (1=edges, 2=triangles, etc).
            Default is 2.
        simplex_dim_weights : dict[int, float] | None, optional
            Weight distribution across simplex dimensions.
            Default is {1: 0.5, 2: 0.5}.
        """
        # Initialize parent class first (this sets up most components)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            _head_dim=head_dim,
            hopfield_hidden_dim=hopfield_hidden_dim,
            et_steps=et_steps,
            et_alpha=et_alpha,
            drop_rate=drop_rate,
            _representation_size=_representation_size,
        )

        # Replace regular Hopfield ET blocks with Simplicial ones
        num_patches = self.patch_embed.num_patches

        # Prepare topology configuration if enabled
        if use_topology:
            grid_size = int(num_patches**0.5)  # Assumes square image
            patch_coords = [
                (i, j) for i in range(grid_size) for j in range(grid_size)
            ]
        else:
            patch_coords = None

        # Default dim_weights if not provided
        if simplex_dim_weights is None:
            simplex_dim_weights = {1: 0.5, 2: 0.5}  # Default 50-50 split

        # Replace ET blocks with Simplicial Hopfield versions
        self.et_blocks = nn.ModuleList(
            [
                EnergyTransformer(
                    layer_norm=EnergyLayerNorm(embed_dim),
                    attention=MultiheadEnergyAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                    ),
                    hopfield=SimplicialHopfieldNetwork(
                        in_dim=embed_dim,
                        simplices=None,  # Auto-generate
                        num_vertices=num_patches,
                        coordinates=patch_coords,
                        max_dim=simplex_max_dim,
                        budget=simplex_budget,
                        dim_weights=simplex_dim_weights,
                        hidden_dim=hopfield_hidden_dim,
                    ),
                    steps=et_steps,
                    optimizer=SGD(alpha=et_alpha),
                )
                for _ in range(depth)
            ],
        )


# Factory functions


def viset_tiny(**kwargs: Any) -> VisionSimplicialEnergyTransformer:
    """ViSET-Tiny configuration with topology-aware simplices."""
    config: dict[str, Any] = {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "head_dim": 64,
        "hopfield_hidden_dim": 768,  # 4x embed_dim
        "et_steps": 4,
        "et_alpha": 0.125,
        "in_chans": 3,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_small(**kwargs: Any) -> VisionSimplicialEnergyTransformer:
    """ViSET-Small configuration with topology-aware simplices."""
    config: dict[str, Any] = {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "head_dim": 64,
        "hopfield_hidden_dim": 1536,  # 4x embed_dim
        "et_steps": 4,
        "et_alpha": 0.125,
        "in_chans": 3,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_base(**kwargs: Any) -> VisionSimplicialEnergyTransformer:
    """ViSET-Base configuration with topology-aware simplices."""
    config: dict[str, Any] = {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "head_dim": 64,
        "hopfield_hidden_dim": 3072,  # 4x embed_dim
        "et_steps": 4,
        "et_alpha": 0.125,
        "in_chans": 3,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


# CIFAR-specific configurations


def viset_2l_e50_t50_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-2L with 50% edges, 50% triangles for CIFAR.

    Shallow 2-layer model with topology-aware simplices optimized
    for 32x32 images with 4x4 patches.
    """
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,  # Shallow!
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,  # 3x embed_dim
        "et_steps": 6,
        "et_alpha": 10.0,
        "drop_rate": 0.1,
        "use_topology": True,
        "simplex_budget": 0.2,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_2l_e100_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-2L with 100% edges (k-NN only) for CIFAR.

    Uses only edges from k-NN graph, no higher-order simplices.
    Good baseline for understanding the value of triangles.
    """
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 6,
        "et_alpha": 10.0,
        "drop_rate": 0.1,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 1,  # Only edges!
        "simplex_dim_weights": {1: 1.0},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_2l_t100_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-2L with 100% triangles for CIFAR.

    Uses only Delaunay triangles, no edges. Tests whether
    higher-order interactions alone are sufficient.
    """
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 6,
        "et_alpha": 10.0,
        "drop_rate": 0.1,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {2: 1.0},  # Only triangles!
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_2l_random_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-2L with RANDOM simplices (no topology awareness).

    Important baseline: Uses random edges and triangles,
    not k-NN + Delaunay. Shows the value of topology awareness.
    """
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 6,
        "et_alpha": 10.0,
        "drop_rate": 0.1,
        "use_topology": False,  # KEY: Random simplices!
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_4l_e50_t50_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-4L with 50% edges, 50% triangles for CIFAR."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 4,
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 5,
        "et_alpha": 5.0,
        "drop_rate": 0.1,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_6l_e50_t50_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-6L with 50% edges, 50% triangles for CIFAR."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 4,
        "et_alpha": 2.5,
        "drop_rate": 0.1,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 2,
        "simplex_dim_weights": {1: 0.5, 2: 0.5},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


def viset_2l_e40_t40_tet20_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionSimplicialEnergyTransformer:
    """ViSET-2L with edges, triangles, AND tetrahedra.

    Experimental: 40% edges, 40% triangles, 20% random 4-cliques
    to capture 2x2 patch interactions.
    """
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 8,
        "head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 6,
        "et_alpha": 10.0,
        "drop_rate": 0.1,
        "use_topology": True,
        "simplex_budget": 0.15,
        "simplex_max_dim": 3,  # Include tetrahedra!
        "simplex_dim_weights": {1: 0.4, 2: 0.4, 3: 0.2},
    }
    config.update(kwargs)
    return VisionSimplicialEnergyTransformer(**config)


# Utility function for model naming


def get_viset_name(
    depth: int,
    dim_weights: dict[int, float],
    use_topology: bool = True,
) -> str:
    """Generate standardized ViSET model name from configuration.

    Examples
    --------
    - ViSET-2L-E50-T50 (2 layers, 50% edges, 50% triangles)
    - ViSET-6L-E100 (6 layers, 100% edges)
    - ViSET-2L-Random (2 layers, random simplices)
    """
    if not use_topology:
        return f"ViSET-{depth}L-Random"

    parts = [f"ViSET-{depth}L"]

    # Add dimension percentages
    dim_chars = {1: "E", 2: "T", 3: "Tet"}
    for dim, weight in sorted(dim_weights.items()):
        if weight > 0 and dim in dim_chars:
            percentage = int(weight * 100)
            parts.append(f"{dim_chars[dim]}{percentage}")

    return "-".join(parts)
