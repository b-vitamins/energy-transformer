"""Realisation of model specs into actual PyTorch modules."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from .combinators import ParallelSpec, SequentialSpec
from .primitives import (
    CLSTokenSpec,
    EmbeddingDim,
    ETBlockSpec,
    MaskTokenSpec,
    NormSpec,
    PatchSpec,
    PosEncSpec,
    Spec,
    TokenCount,
)

__all__ = [
    "realise",
    "register_realiser",
    "register_module",
    "SpecInfo",
]

# Type alias for realiser functions
RealiserFunc = Callable[[Any, "SpecInfo"], nn.Module]

# Type registries for realisation functions
_REALISERS: dict[type, RealiserFunc] = {}
_MODULE_REGISTRY: dict[str, type[nn.Module]] = {}


class SpecInfo:
    """Information about a specification and its context in a model.

    Provide dimensions, token counts, and other information needed
    for realisation, with proper dependency tracking and validation.

    Attributes
    ----------
    embedding_dim : Optional[EmbeddingDim]
        The current embedding dimension.
    token_count : Optional[TokenCount]
        The current token count.
    has_cls_token : bool
        Whether a CLS token has been added.
    has_mask_token : bool
        Whether a mask token has been added.
    """

    def __init__(self) -> None:
        """Initialize empty SpecInfo."""
        self.embedding_dim: EmbeddingDim | None = None
        self.token_count: TokenCount | None = None
        self.has_cls_token: bool = False
        self.has_mask_token: bool = False

    def update_from_spec(self, spec: Spec) -> None:
        """Update information from a specification.

        Parameters
        ----------
        spec : Spec
            The specification to update from.
        """
        # Update dimension information
        dim = spec.get_embedding_dim()
        if dim is not None:
            self.embedding_dim = dim

        # Update token count information
        count = spec.get_token_count()
        if count is not None:
            self.token_count = count

        # Track token additions
        if spec.adds_cls_token():
            self.has_cls_token = True
            if self.token_count is not None:
                self.token_count += 1

        if spec.adds_mask_token():
            self.has_mask_token = True

    def create_child(self) -> SpecInfo:
        """Create a child info that inherits from this one.

        Returns
        -------
        SpecInfo
            A new SpecInfo with the same state as this one.
        """
        child = SpecInfo()
        child.embedding_dim = self.embedding_dim
        child.token_count = self.token_count
        child.has_cls_token = self.has_cls_token
        child.has_mask_token = self.has_mask_token
        return child

    def validate_spec(self, spec: Spec) -> None:
        """Validate a spec against this info.

        Parameters
        ----------
        spec : Spec
            The specification to validate.

        Raises
        ------
        ValueError
            If the spec cannot be validated against this info.
        """
        spec.validate(self.embedding_dim, self.token_count)

    def __str__(self) -> str:
        """Return string representation of SpecInfo.

        Returns
        -------
        str
            Human-readable representation of the SpecInfo state.
        """
        return (
            f"SpecInfo(embed_dim={self.embedding_dim}, "
            f"token_count={self.token_count}, "
            f"has_cls={self.has_cls_token}, "
            f"has_mask={self.has_mask_token})"
        )


def register_realiser(
    spec_type: type,
) -> Callable[[RealiserFunc], RealiserFunc]:
    """Register a function that realises a spec type.

    Parameters
    ----------
    spec_type : type
        The specification type to register a realiser for.

    Returns
    -------
    Callable[[RealiserFunc], RealiserFunc]
        Decorator function that takes and returns a realiser function.
    """

    def decorator(fn: RealiserFunc) -> RealiserFunc:
        _REALISERS[spec_type] = fn
        return fn

    return decorator


def register_module(name: str, module_cls: type[nn.Module]) -> None:
    """Register a module class with a string identifier.

    Parameters
    ----------
    name : str
        String identifier for the module.
    module_cls : type[nn.Module]
        The PyTorch module class to register.
    """
    _MODULE_REGISTRY[name] = module_cls


def get_module_class(name: str) -> type[nn.Module]:
    """Get a registered module class by name.

    Parameters
    ----------
    name : str
        String identifier of the module class.

    Returns
    -------
    type[nn.Module]
        The registered module class.

    Raises
    ------
    KeyError
        If the module class is not registered.
    """
    if name not in _MODULE_REGISTRY:
        raise KeyError(f"Module class '{name}' not registered")
    return _MODULE_REGISTRY[name]


def realise(spec: Any, info: SpecInfo | None = None) -> nn.Module | None:
    """Convert a specification to a PyTorch module.

    Parameters
    ----------
    spec : Any
        A specification object.
    info : Optional[SpecInfo], default=None
        Information from upstream specs.

    Returns
    -------
    Optional[nn.Module]
        A PyTorch module constructed according to the spec, or None if spec is None.

    Raises
    ------
    TypeError
        If the spec type is not registered.
    ValueError
        If the spec cannot be validated against the context.
    """
    # Initialize info if not provided
    if info is None:
        info = SpecInfo()

    # Handle None case
    if spec is None:
        return None

    # For modules, return as is
    if isinstance(spec, nn.Module):
        return spec

    # Validate spec against current info
    if isinstance(spec, Spec):
        info.validate_spec(spec)

    # Handle sequential specs
    if isinstance(spec, SequentialSpec):
        return _realise_seq(spec, info)

    # Handle parallel specs
    if isinstance(spec, ParallelSpec):
        return _realise_parallel(spec, info)

    # For leaf specs, use the registered realiser
    spec_type = type(spec)
    if spec_type in _REALISERS:
        # Create module
        module = _REALISERS[spec_type](spec, info)

        # Update info with this spec's contributions
        if isinstance(spec, Spec):
            info.update_from_spec(spec)

        return module

    raise TypeError(f"Don't know how to realise {spec!r}")


def _realise_seq(spec: SequentialSpec, info: SpecInfo) -> nn.Module:
    """Realise a sequential specification.

    Process parts in order, propagating dimensions.

    Parameters
    ----------
    spec : SequentialSpec
        The sequential specification to realise.
    info : SpecInfo
        Context information from upstream specs.

    Returns
    -------
    nn.Module
        The realised sequential module.

    Raises
    ------
    ValueError
        If the sequential spec cannot be validated.
    """
    modules: list[nn.Module] = []

    # Validate whole sequence first
    spec.validate(info.embedding_dim, info.token_count)

    # Process each part in order
    for part in spec.parts:
        module = realise(part, info)
        if module is not None:
            modules.append(module)

    # Check if this sequence forms a vision encoder pattern
    if _is_vision_encoder_pattern(modules):
        return _assemble_vision_encoder(modules, info)

    return nn.Sequential(*modules)


def _realise_parallel(spec: ParallelSpec, info: SpecInfo) -> nn.Module:
    """Realise a parallel specification.

    Process branches independently with the same upstream info.

    Parameters
    ----------
    spec : ParallelSpec
        The parallel specification to realise.
    info : SpecInfo
        Context information from upstream specs.

    Returns
    -------
    nn.Module
        The realised parallel module.

    Raises
    ------
    ValueError
        If no valid branches exist or parallel spec cannot be validated.
    """
    # Validate whole parallel spec first
    spec.validate(info.embedding_dim, info.token_count)

    # Process each branch with a copy of the info
    branches: list[nn.Module] = []
    for branch in spec.branches:
        branch_info = info.create_child()
        module = realise(branch, branch_info)
        if module is not None:
            branches.append(module)

    if not branches:
        raise ValueError(
            "No valid branches to combine in parallel specification"
        )

    # Create combining module
    class ParallelModule(nn.Module):  # type: ignore
        """Module that combines parallel branches."""

        def __init__(self, branches: list[nn.Module], join_mode: str) -> None:
            """Initialize parallel module.

            Parameters
            ----------
            branches : list[nn.Module]
                List of branch modules to combine.
            join_mode : str
                How to combine the branch outputs.
            """
            super().__init__()
            self.branches = nn.ModuleList(branches)
            self.join_mode = join_mode

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through all branches.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            torch.Tensor
                Combined output from all branches.

            Raises
            ------
            ValueError
                If join_mode is not supported.
            """
            outputs = [branch(x) for branch in self.branches]

            if self.join_mode == "concat":
                return torch.cat(outputs, dim=-1)
            elif self.join_mode == "add":
                return sum(outputs)
            elif self.join_mode == "multiply":
                result = outputs[0]
                for o in outputs[1:]:
                    result = result * o
                return result
            else:
                raise ValueError(f"Unsupported join mode: {self.join_mode}")

    # Update info based on parallel module's output dimension
    output_dim = spec.get_embedding_dim()
    if output_dim is not None:
        info.embedding_dim = output_dim

    return ParallelModule(branches, spec.join_mode)


def _is_vision_encoder_pattern(modules: Sequence[nn.Module]) -> bool:
    """Check if modules form a vision transformer encoder pattern.

    The pattern is:
    1. Patch embedder
    2. Optional position encoder
    3. Optional CLS token
    4. Optional mask token
    5. Multiple transformer blocks
    6. Final normalization (optional)

    Parameters
    ----------
    modules : Sequence[nn.Module]
        List of modules to check.

    Returns
    -------
    bool
        True if modules form a vision encoder pattern.
    """
    if not modules:
        return False

    # First module should be patch embedder
    if (
        not hasattr(modules[0], "__class__")
        or modules[0].__class__.__name__ != "PatchEmbed"
    ):
        return False

    # Check for transformer blocks
    has_transformer = False
    for module in modules:
        if (
            hasattr(module, "__class__")
            and module.__class__.__name__ == "EnergyTransformer"
        ):
            has_transformer = True
            break

    return has_transformer


def _assemble_vision_encoder(
    modules: Sequence[nn.Module], info: SpecInfo
) -> nn.Module:
    """Assemble modules into a vision transformer encoder.

    Parameters
    ----------
    modules : Sequence[nn.Module]
        List of modules to assemble.
    info : SpecInfo
        Context information for creating default components.

    Returns
    -------
    nn.Module
        Assembled vision encoder.

    Raises
    ------
    ValueError
        If required components are missing or cannot be created.
    ImportError
        If ViETEncoder cannot be imported.
    """
    # Extract components by role
    patch_embed: nn.Module | None = None
    pos_encoder: nn.Module | None = None
    cls_token: nn.Module | None = None
    mask_token: nn.Module | None = None
    transformer_blocks: list[nn.Module] = []
    norm_layer: nn.Module | None = None

    for module in modules:
        class_name = module.__class__.__name__

        if class_name == "PatchEmbed":
            patch_embed = module
        elif class_name in ("Learnable2DPosEnc", "SinCos2DPosEnc"):
            pos_encoder = module
        elif class_name == "CLSToken":
            cls_token = module
        elif class_name == "MaskToken":
            mask_token = module
        elif class_name == "EnergyTransformer":
            transformer_blocks.append(module)
        elif isinstance(module, nn.LayerNorm):
            norm_layer = module

    # Ensure required components are present
    if patch_embed is None:
        raise ValueError("Vision encoder requires a PatchEmbed component")

    if not transformer_blocks:
        raise ValueError(
            "Vision encoder requires at least one transformer block"
        )

    # Get encoder class
    try:
        from energy_transformer.models.vision import ViETEncoder
    except ImportError as err:
        raise ImportError(
            "Could not import ViETEncoder - are you missing the models module?"
        ) from err

    # Create default normalization if needed
    if norm_layer is None:
        if info.embedding_dim is None:
            raise ValueError(
                "Cannot create default normalization without known embedding dimension"
            )
        norm_layer = nn.LayerNorm(info.embedding_dim)

    # Assemble encoder
    return ViETEncoder(
        patch_embedder=patch_embed,
        transformer_blocks=transformer_blocks,
        norm_layer=norm_layer,
        pos_encoder=pos_encoder,
        cls_token=cls_token,
        mask_token=mask_token,
    )


# Register realisers for each spec type
@register_realiser(PatchSpec)
def _realise_patch_spec(spec: PatchSpec, info: SpecInfo) -> nn.Module:
    """Realise a patch embedding specification.

    Parameters
    ----------
    spec : PatchSpec
        The patch embedding specification.
    info : SpecInfo
        Context information (unused for patch embedding).

    Returns
    -------
    nn.Module
        The patch embedding module.

    Raises
    ------
    ImportError
        If PatchEmbed cannot be imported.
    """
    from energy_transformer.utils.vision import PatchEmbed

    return PatchEmbed(
        img_size=spec.img_size,
        patch_size=spec.patch_size,
        in_chans=spec.in_chans,
        embed_dim=spec.embed_dim,
        flatten=spec.flatten,
    )


@register_realiser(PosEncSpec)
def _realise_pos_enc_spec(spec: PosEncSpec, info: SpecInfo) -> nn.Module:
    """Realise a positional encoding specification.

    Parameters
    ----------
    spec : PosEncSpec
        The positional encoding specification.
    info : SpecInfo
        Context information containing embedding dimension and token count.

    Returns
    -------
    nn.Module
        The positional encoding module.

    Raises
    ------
    ImportError
        If positional encoding modules cannot be imported.
    ValueError
        If required context information is missing.
    """
    from energy_transformer.utils.vision import (
        Learnable2DPosEnc,
        SinCos2DPosEnc,
    )

    # These values are guaranteed to be available by validation
    embed_dim = info.embedding_dim
    num_tokens = info.token_count

    if embed_dim is None or num_tokens is None:
        raise ValueError(
            "embed_dim and num_tokens must be available from context"
        )

    if spec.kind == "learned":
        return Learnable2DPosEnc(num_tokens=num_tokens, embed_dim=embed_dim)
    else:  # sincos
        return SinCos2DPosEnc(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            include_cls=spec.include_cls,
        )


@register_realiser(CLSTokenSpec)
def _realise_cls_token_spec(spec: CLSTokenSpec, info: SpecInfo) -> nn.Module:
    """Realise a CLS token specification.

    Parameters
    ----------
    spec : CLSTokenSpec
        The CLS token specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The CLS token module.

    Raises
    ------
    ImportError
        If CLSToken cannot be imported.
    ValueError
        If embedding dimension is not available from context.
    """
    from energy_transformer.utils.vision import CLSToken

    # Embedding dimension is guaranteed to be available by validation
    embed_dim = info.embedding_dim

    if embed_dim is None:
        raise ValueError("embed_dim must be available from context")

    return CLSToken(embed_dim=embed_dim)


@register_realiser(MaskTokenSpec)
def _realise_mask_token_spec(spec: MaskTokenSpec, info: SpecInfo) -> nn.Module:
    """Realise a mask token specification.

    Parameters
    ----------
    spec : MaskTokenSpec
        The mask token specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The mask token module.

    Raises
    ------
    ImportError
        If MaskToken cannot be imported.
    ValueError
        If embedding dimension is not available from context.
    """
    from energy_transformer.utils.vision import MaskToken

    # Embedding dimension is guaranteed to be available by validation
    embed_dim = info.embedding_dim

    if embed_dim is None:
        raise ValueError("embed_dim must be available from context")

    return MaskToken(embed_dim=embed_dim)


@register_realiser(ETBlockSpec)
def _realise_et_block_spec(spec: ETBlockSpec, info: SpecInfo) -> nn.Module:
    """Realise an Energy Transformer block specification.

    Parameters
    ----------
    spec : ETBlockSpec
        The Energy Transformer block specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The Energy Transformer block module.

    Raises
    ------
    ImportError
        If required Energy Transformer components cannot be imported.
    ValueError
        If embedding dimension is not available from context.
    """
    from energy_transformer.layers.attention import MultiHeadEnergyAttention
    from energy_transformer.layers.hopfield import HopfieldNetwork
    from energy_transformer.layers.layer_norm import LayerNorm
    from energy_transformer.models.base import EnergyTransformer

    # Try to import SimplicialHopfieldNetwork - it's optional
    simplicial_cls: type | None = None
    try:
        from energy_transformer.layers.simplicial import (
            SimplicialHopfieldNetwork,
        )

        simplicial_cls = SimplicialHopfieldNetwork
    except ImportError:
        pass

    # Embedding dimension is guaranteed to be available by validation
    embed_dim = info.embedding_dim

    if embed_dim is None:
        raise ValueError("embed_dim must be available from context")

    # Create components
    layer_norm = LayerNorm(embed_dim, eps=spec.layer_norm_eps)

    # Calculate beta if not provided in spec
    attention_beta = spec.attention_beta
    if attention_beta is None:
        attention_beta = 1.0 / math.sqrt(spec.head_dim)  # Default from paper

    attention = MultiHeadEnergyAttention(
        in_dim=embed_dim,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        beta=attention_beta,  # Pass beta parameter
    )

    # Choose Hopfield type based on spec
    if spec.hopfield_type == "simplicial" and simplicial_cls is not None:
        hopfield = simplicial_cls(
            in_dim=embed_dim,
            hidden_dim=spec.hidden_dim,
        )
    else:
        hopfield = HopfieldNetwork(
            in_dim=embed_dim,
            hidden_dim=spec.hidden_dim,
        )

    # Create transformer block
    return EnergyTransformer(
        layer_norm=layer_norm,
        attention=attention,
        hopfield=hopfield,
        steps=spec.steps,
        alpha=spec.alpha,
    )


@register_realiser(NormSpec)
def _realise_norm_spec(spec: NormSpec, info: SpecInfo) -> nn.Module:
    """Realise a normalization layer specification.

    Parameters
    ----------
    spec : NormSpec
        The normalization specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The normalization layer.

    Raises
    ------
    ValueError
        If embedding dimension is not available from context.
    """
    # Embedding dimension is guaranteed to be available by validation
    embed_dim = info.embedding_dim

    if embed_dim is None:
        raise ValueError("embed_dim must be available from context")

    return nn.LayerNorm(embed_dim, eps=spec.eps)


# Initialize module registry with known classes
try:
    from energy_transformer.models.base import EnergyTransformer

    register_module("EnergyTransformer", EnergyTransformer)
except ImportError:
    pass
