"""Energy-based Hopfield Network implementations."""

import math
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseHopfieldNetwork


class SimplicialHopfieldNetwork(BaseHopfieldNetwork):
    """Simplicial Hopfield Network with feature-wise higher-order connectivity.

    Parameters
    ----------
    in_dim : int
        Input dimension D of token vectors
    hidden_dim : int
        Number of memory patterns P
    max_dim : int, default 3
        Highest simplex dimension k to include
    budget_scale : float, default 1.0
        Total simplex budget as multiple of binom(D,2)
    dim_probs : Sequence[float] | None, default None
        Probability distribution for simplex allocation across dimensions
    temperature : float, default 1.0
        Temperature parameter T^(-1) for log-sum-exp energy function
    trainable_patterns : bool, default True
        Whether memory patterns are trainable
    use_autocast : bool, default False
        Use autocast context in memory score computation for mixed precision training

    Notes
    -----
    This implements a Simplicial Hopfield Network based on the continuous
    modern formulation with log-sum-exp energy function (Eq. 6):

    E(S) = -lse(T^{-1}, Ξ^T S, K) + (1/2)S^T S

    where lse(T^{-1}, Ξ^T S, K) = T·log(∑_{μ=1}^{P} ∑_{σ∈K} exp(T^{-1}·Ξ_{σ}^{μ}·S_{σ}))

    In this implementation, simplices σ are defined over feature dimensions (σ ⊂ {1...D}),
    not over tokens. This means:

    - A k-simplex represents a higher-order conjunction of k+1 feature dimensions
    - The operation is applied to each token independently
    - The implementation parallels the MLP-like structure of standard Hopfield modules

    A simplicial complex K is a collection of simplices closed under taking subsets:
    - 0-simplices: vertices (single features)
    - 1-simplices: edges (pairs of features)
    - 2-simplices: triangles (triples of features)
    - k-simplices: (k+1)-tuples of features

    Due to computational constraints (since the number of possible k-simplices is
    binom(D, k+1), which grows exponentially), we use a mixed-diluted approach where:
    - Only a subset of possible simplices are included
    - The total simplex budget is controlled by budget_scale parameter
    - Simplices are distributed across dimensions according to dim_probs
    """

    # Type annotation for the simplices buffer
    simplices: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        *,
        max_dim: int = 3,
        budget_scale: float = 1.0,
        dim_probs: Sequence[float] | None = None,
        temperature: float = 1.0,
        trainable_patterns: bool = True,
        exact_split: bool = False,
        use_autocast: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        """Initialize the Simplicial Hopfield Network.

        Parameters
        ----------
        in_dim : int
            Input dimension D of token vectors
        hidden_dim : int
            Number of memory patterns P
        max_dim : int, default 3
            Highest simplex dimension k to include (2=edges, 3=triangles, etc.)
        budget_scale : float, default 1.0
            Total simplex budget as multiple of binom(D,2)
        dim_probs : Sequence[float] | None, default None
            Probability distribution for simplex allocation across dimensions.
            Must have length max_dim-1 (for dims 2...max_dim) and sum to 1.
        temperature : float, default 1.0
            Temperature parameter T for log-sum-exp function
        trainable_patterns : bool, default True
            Whether memory patterns are trainable during optimization
        exact_split : bool, default False
            If True, use multinomial sampling for precise dimension proportions
        use_autocast : bool, default False
            Use autocast context in memory score computation for mixed precision training
        device : Any, optional
            Device to place tensors on
        dtype : Any, optional
            Data type for tensors
        """
        super().__init__()

        # Validate and initialize parameters
        self._validate_parameters(max_dim, budget_scale)

        # Store model parameters
        self.in_dim = in_dim  # Feature dimension D
        self.hidden_dim = hidden_dim  # Number of patterns P
        self.max_dim = max_dim  # Maximum simplex dimension k
        self.T = float(temperature)  # Temperature parameter T from Eq. (5)
        self.exact_split = exact_split
        self.use_autocast = use_autocast

        # Initialize dimension probabilities for mixed-diluted approach
        self.dim_probs = self._initialize_dim_probs(dim_probs, max_dim)

        # Calculate parameter budget for mixed diluted network
        # Budget proportional to number of edges between features (D choose 2)
        n_edges = in_dim * (in_dim - 1) // 2
        self.param_budget = max(1, math.ceil(budget_scale * n_edges))

        # Allocate budget per dimension (for mixed diluted approach)
        self.simplex_counts = self._allocate_simplex_budget()
        self._validate_simplex_counts(in_dim)

        # Memory patterns Ξ ∈ ℝᴾˣᴰ (P:hidden_dim, D:in_dim)
        # This corresponds to the pattern matrix Ξ in Eq. (5-6)
        self.Ξ = nn.Parameter(
            torch.empty(hidden_dim, in_dim, device=device, dtype=dtype),
            requires_grad=trainable_patterns,
        )  # shape: [P, D]

        # Simplices buffer - initialized on first use
        # This represents the simplicial complex K in Eq. (5-6)
        self.register_buffer(
            "simplices", torch.zeros(0, 0, dtype=torch.long, device=device)
        )

        self.reset_parameters()

    def __repr__(self) -> str:
        """Return a string representation of the module."""
        # Ensure simplices are initialized
        self._initialize_simplices()
        num_simplices = (
            0 if self.simplices.numel() == 0 else int(len(self.simplices))
        )
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"max_dim={self.max_dim}, "
            f"param_budget={self.param_budget}, "
            f"num_simplices={num_simplices}, "
            f"temperature={self.T})"
        )

    def reset_parameters(self) -> None:
        """Initialize learnable parameters.

        The pattern matrix Ξ (xi) is initialized with normal distribution.
        """
        nn.init.normal_(self.Ξ, std=0.02)

    def _validate_parameters(self, max_dim: int, budget_scale: float) -> None:
        """Validate input parameters."""
        if max_dim < 2:
            raise ValueError("max_dim must be ≥ 2 (need at least edges)")
        if budget_scale <= 0:
            raise ValueError("budget_scale must be positive")

    def _validate_simplex_counts(self, in_dim: int) -> None:
        """Check if requested simplices exceed mathematical limits."""
        for k, num in enumerate(self.simplex_counts, start=2):
            max_possible = math.comb(in_dim, k)
            if num > max_possible:
                raise ValueError(
                    f"Requested {num} {k}-simplices, but only {max_possible} exist."
                )

    def _initialize_dim_probs(
        self, dim_probs: Sequence[float] | None, max_dim: int
    ) -> torch.Tensor:
        """Initialize dimension probabilities for mixed-diluted network.

        These probabilities determine how the parameter budget is
        allocated across different simplex dimensions.
        """
        if dim_probs is None:
            # Uniform distribution across dimensions
            dim_probs = [1.0 / (max_dim - 1)] * (max_dim - 1)
        else:
            if len(dim_probs) != max_dim - 1:
                raise ValueError(
                    f"dim_probs must have length max_dim-1 (got {len(dim_probs)})"
                )
            if not math.isclose(sum(dim_probs), 1.0, rel_tol=1e-6):
                raise ValueError("dim_probs must sum to 1.0")
            if any(p < 0 for p in dim_probs):
                raise ValueError("dim_probs entries must be non-negative")

        return torch.as_tensor(
            dim_probs, dtype=torch.float
        )  # shape: [max_dim-1]

    def _allocate_simplex_budget(self) -> torch.Tensor:
        """Allocate simplices budget across dimensions.

        This implements the mixed-diluted approach described in the paper,
        where we constrain the total number of weighted simplices to
        manage computational complexity.
        """
        if self.exact_split:
            # Sample dimension for each simplex using multinomial
            dim_choice = torch.multinomial(
                self.dim_probs, self.param_budget, replacement=True
            )
            counts = torch.bincount(dim_choice, minlength=self.max_dim - 1)
        else:
            # Allocate budget proportionally to dimension probabilities
            raw_counts = self.dim_probs * self.param_budget
            counts = raw_counts.floor().to(torch.long)

            # Distribute any remaining budget to highest probability dimensions
            remainder = self.param_budget - counts.sum()
            if remainder > 0:
                order = torch.argsort(self.dim_probs, descending=True)
                for idx in order[:remainder]:
                    counts[idx] += 1

        return (
            counts  # shape: [max_dim-1], counts[k-2] is number of k-simplices
        )

    @staticmethod
    def _reservoir_sample(
        n: int, k: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Perform reservoir sampling when n is too large for randperm.

        Memory-efficient implementation for sampling k elements from [0...n-1]
        without replacement, even when n is extremely large.

        Parameters
        ----------
        n : int
            Size of the population
        k : int
            Number of samples to draw
        device : torch.device, optional
            Device for the output tensor

        Returns
        -------
        torch.Tensor
            Tensor of k unique integers in range [0, n-1]
        """
        # For small n, just use randperm which is faster
        if n < 1e7 or k > n // 10:
            return torch.randperm(n, device=device)[:k]

        # Use reservoir sampling on CPU with int32 (more memory efficient)
        reservoir = torch.arange(k, dtype=torch.int32)
        rng = torch.Generator()
        rng.manual_seed(torch.initial_seed())

        for i in range(k, n):
            j = torch.randint(0, i + 1, (1,), generator=rng).item()
            if j < k:
                reservoir[j] = i

        # Transfer to the requested device and convert to long
        return reservoir.to(device=device, dtype=torch.long)

    @staticmethod
    def _unrank(r: int, n: int, k: int) -> tuple[int, ...]:
        """Convert rank to k-subset in lexicographic order.

        Efficiently computes the k-subset of {0,1,...,n-1} with rank r
        in lexicographic ordering. This is used to convert between
        integer ranks and actual simplices.
        """
        comb = math.comb
        if not (0 <= r < comb(n, k)):
            raise ValueError("rank out of range")

        subset = []
        cur, rem_rank, rem_k = 0, r, k
        while rem_k:
            skip = comb(n - (cur + 1), rem_k - 1)
            if rem_rank < skip:
                subset.append(cur)
                rem_k -= 1
            else:
                rem_rank -= skip
            cur += 1
        return tuple(subset)

    @staticmethod
    def _sample_simplices(n: int, counts: Tensor, device: Any = None) -> Tensor:
        """Sample simplices using rank-based sampling.

        For each dimension k, choose counts[k-2] distinct k-simplices
        uniformly without replacement using unranking procedure.

        Parameters
        ----------
        n : int
            Number of vertices (in_dim) - here these are feature dimensions
        counts : Tensor
            counts[k-2] is the number of k-simplices to sample
        device : Any, optional
            Device for the output tensor

        Returns
        -------
        Tensor
            Tensor of shape [M, k_max] where M is total number of
            simplices and k_max is the maximum simplex dimension
        """
        simplices: list[tuple[int, ...]] = []

        # Sample simplices for each dimension
        for k, num in enumerate(counts, start=2):
            if num == 0:
                continue

            # Total number of possible k-simplices
            total = math.comb(n, k)

            # Use reservoir sampling for large combinations
            if total > 1e7:
                # Move to CPU for memory efficiency with large permutations
                ranks = SimplicialHopfieldNetwork._reservoir_sample(
                    total, num, device="cpu"
                )
            else:
                # Use regular randperm for smaller combinations
                gen = torch.Generator(device=device)
                gen.manual_seed(torch.initial_seed())
                ranks = torch.randperm(total, generator=gen, device=device)[
                    :num
                ]

            # Convert ranks to k-subsets
            simplices.extend(
                SimplicialHopfieldNetwork._unrank(int(r), n, k)
                for r in ranks.tolist()
            )

        # Handle edge case of empty simplices
        if not simplices:
            return torch.empty(0, 0, dtype=torch.long, device=device)

        # Determine maximum simplex size and create output tensor
        k_max = max(len(s) for s in simplices)
        out = -torch.ones(
            len(simplices), k_max, dtype=torch.long, device=device
        )

        # Fill output tensor with simplices
        for i, s in enumerate(simplices):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        return out  # shape: [M, k_max]

    def _initialize_simplices(self) -> None:
        """Lazily initialize simplicial complex K on the correct device.

        This initializes the simplicial complex K used in Eq. (5-6).
        Each simplex connects multiple feature dimensions within a token.
        """
        if self.simplices.numel() == 0:
            self.simplices = self._sample_simplices(
                n=self.in_dim,  # Sampling from feature dimensions D
                counts=self.simplex_counts,
                device=self.Ξ.device,
            )  # shape: [M, k_max]

    def _compute_memory_scores(self, g: Tensor) -> Tensor:
        """
        Return scores[..., μ, σ] = (∏_{j∈σ} ξ^μ_j) · (∏_{j∈σ} S_j).

        Shape of output is [..., N, P, M] where:
        - N is sequence length
        - P is hidden_dim (memory patterns)
        - M is number of simplices
        """
        # Use autocast context for mixed precision if enabled
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_autocast and g.is_cuda
            else nullcontext()
        )

        with context:
            # lazy initialisation (device correct)
            self._initialize_simplices()

            # ------------------------------------------------------------------
            # 0.  Shapes and helpers
            # ------------------------------------------------------------------
            batch_shape, seq_len = g.shape[:-2], g.shape[-2]
            p, d = self.hidden_dim, self.in_dim
            simplices = self.simplices  # (m, k_max)
            m, k_max = simplices.shape

            # Create valid mask and safe gather indices
            valid = simplices >= 0  # (m, k_max)
            gather_idx = simplices.clamp(min=0)  # (m, k_max)

            # ------------------------------------------------------------------
            # 1.  Prepare token tensor in log-space for numerical stability
            # ------------------------------------------------------------------
            # Avoid reshaping assumption about contiguous by using flatten+unflatten
            batch_size = 1
            for dim in batch_shape:
                batch_size *= dim

            # Safely reshape input
            g_flat = g.reshape(batch_size * seq_len, d)  # (b·n, d)

            # Convert to log-space with safe sign handling for gradients
            g_abs = g_flat.abs().clamp_min_(1e-12)
            g_log = g_abs.log()  # (b·n, d)
            one_g = torch.ones((), dtype=g.dtype, device=g.device)
            g_sgn = torch.where(g_flat >= 0, one_g, -one_g)  # (b·n, d)

            # Compute parameter log/sign with proper gradient handling
            xi_abs = self.Ξ.abs().clamp_min_(1e-12)
            xi_log = xi_abs.log()  # (p, d)
            one_xi = torch.ones((), dtype=self.Ξ.dtype, device=self.Ξ.device)
            xi_sgn = torch.where(self.Ξ >= 0, one_xi, -one_xi)  # (p, d)

            # ------------------------------------------------------------------
            # 2.  Memory-efficient gather using take_along_dim
            # ------------------------------------------------------------------
            # Use take_along_dim for more efficient gathering with proper broadcasting
            g_log_exp = g_log.unsqueeze(1).expand(-1, m, -1)  # (b·n, m, d)
            g_log_sub = torch.take_along_dim(
                g_log_exp,
                gather_idx.unsqueeze(0).expand(g_log_exp.size(0), -1, -1),
                dim=-1,
            )  # shape: (b·n, m, k_max)

            g_sgn_exp = g_sgn.unsqueeze(1).expand(-1, m, -1)  # (b·n, m, d)
            g_sgn_sub = torch.take_along_dim(
                g_sgn_exp,
                gather_idx.unsqueeze(0).expand(g_sgn_exp.size(0), -1, -1),
                dim=-1,
            )  # shape: (b·n, m, k_max)

            xi_log_exp = xi_log.unsqueeze(1).expand(-1, m, -1)  # (p, m, d)
            xi_log_sub = torch.take_along_dim(
                xi_log_exp,
                gather_idx.unsqueeze(0).expand(xi_log_exp.size(0), -1, -1),
                dim=-1,
            )  # shape: (p, m, k_max)

            xi_sgn_exp = xi_sgn.unsqueeze(1).expand(-1, m, -1)  # (p, m, d)
            xi_sgn_sub = torch.take_along_dim(
                xi_sgn_exp,
                gather_idx.unsqueeze(0).expand(xi_sgn_exp.size(0), -1, -1),
                dim=-1,
            )  # shape: (p, m, k_max)

            # ------------------------------------------------------------------
            # 3.  Broadcast for computation
            # ------------------------------------------------------------------
            g_log_gathered = g_log_sub.unsqueeze(1)  # (b·n, 1, m, k_max)
            g_sgn_gathered = g_sgn_sub.unsqueeze(1)  # (b·n, 1, m, k_max)
            xi_log_gathered = xi_log_sub.unsqueeze(0)  # (1, p, m, k_max)
            xi_sgn_gathered = xi_sgn_sub.unsqueeze(0)  # (1, p, m, k_max)

            # Get valid mask efficiently (avoid mask broadcasting)
            valid_mask = valid.unsqueeze(0).unsqueeze(0)  # (1, 1, m, k_max)

            # ------------------------------------------------------------------
            # 4.  Compute products in log-space for numerical stability
            # ------------------------------------------------------------------
            # Add logs (multiply values) and multiply signs (maintain sign)
            log_sum = g_log_gathered + xi_log_gathered  # (b·n, p, m, k_max)
            sign_prod = g_sgn_gathered * xi_sgn_gathered  # (b·n, p, m, k_max)

            # For invalid entries, use 0 (log 1) in log-space with correct device/dtype
            zero = torch.zeros((), dtype=log_sum.dtype, device=log_sum.device)
            log_sum = torch.where(valid_mask, log_sum, zero)

            # For invalid signs, use 1 so they don't affect the product
            one = torch.ones((), dtype=sign_prod.dtype, device=sign_prod.device)
            sign_prod = torch.where(valid_mask, sign_prod, one)

            # Sum logs within each simplex (product in original space)
            log_term = log_sum.sum(dim=-1)  # (b·n, p, m)

            # Product of signs - only needed for valid positions
            sign_term = sign_prod.prod(dim=-1)  # (b·n, p, m)

            # Apply sign and exp to get final scores
            scores = (sign_term * torch.exp(log_term)).to(
                g.dtype
            )  # (b·n, p, m)

            # ------------------------------------------------------------------
            # 5.  Reshape back to [..., N, P, M] using contiguous for safe view
            # ------------------------------------------------------------------
            out = scores.contiguous().view(*batch_shape, seq_len, p, m)

            return out

    def forward(self, g: Tensor) -> Tensor:
        """Compute Simplicial Hopfield Network energy.

        This implements the complete energy function from Eq. (6):
        E = -lse(T^{-1}, Ξ^T S, K) + (1/2)S^T S

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension

        Returns
        -------
        Tensor
            Scalar energy value representing the total Hopfield energy
        """
        # Use autocast context for mixed precision if enabled
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_autocast and g.is_cuda
            else nullcontext()
        )

        with context:
            # Ensure simplices are initialized (lazy initialization)
            self._initialize_simplices()

            # Compute memory scores (Ξ_{σ}^{μ}·S_{σ})
            mem_scores = self._compute_memory_scores(g)  # shape: [..., N, P, M]

            # Flatten the P×M dimensions for the ∑_{μ=1}^{P} ∑_{σ∈K} term
            flat = mem_scores.flatten(
                start_dim=-2
            )  # Flattening last two dims [P, M]

            # Use stable logsumexp implementation with temperature scaling
            # lse(T^{-1}, Ξ^T S, K) = T·log(∑_{μ=1}^{P} ∑_{σ∈K} exp(T^{-1}·Ξ_{σ}^{μ}·S_{σ}))
            temperature = torch.as_tensor(
                self.T, dtype=g.dtype, device=g.device
            )
            lse = (
                torch.logsumexp(flat / temperature, dim=-1) * temperature
            )  # shape: [..., N]

            # Sum -lse across ALL dimensions to get a true scalar
            lse_sum = lse.sum()  # scalar

            # Calculate (1/2)S^T S - already summing over all dimensions
            squares_sum = 0.5 * (g**2).sum()  # scalar

            # E(S) = -lse(T^{-1}, Ξ^T S, K) + (1/2)S^T S
            energy = (-lse_sum + squares_sum).to(g.dtype)  # scalar

            return energy
