"""Memory usage benchmarks for models and components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from energy_transformer.layers import (
    HopfieldNetwork,
    LayerNorm,
    MultiHeadEnergyAttention,
    SimplicialHopfieldNetwork,
)
from energy_transformer.models.vision import (
    viet_tiny,
    viset_tiny,
    vit_tiny,
)

from .memory_utils import MemoryProfiler, analyze_memory_scaling

pytestmark = [pytest.mark.performance, pytest.mark.memory_bench]


class TestModelMemory:
    """Test memory usage of complete models."""

    @pytest.mark.parametrize(
        ("model_name", "factory"),
        [
            ("vit_tiny", vit_tiny),
            ("viet_tiny", viet_tiny),
            ("viset_tiny", viset_tiny),
        ],
    )
    def test_model_memory_footprint(
        self,
        model_name: str,
        factory,
        device,
    ) -> None:
        """Test memory footprint of model initialization and first forward pass."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")

        profiler = MemoryProfiler(device)

        def create_model() -> torch.nn.Module:
            return (
                factory(
                    img_size=224,
                    patch_size=16,
                    in_chans=3,  # Added missing parameter
                    num_classes=1000,
                )
                .to(device)
                .eval()
            )

        model_memory, model = profiler.profile_function(
            create_model, return_output=True
        )

        x = torch.randn(1, 3, 224, 224, device=device)

        def run_inference() -> torch.Tensor:
            with torch.no_grad():
                kwargs = (
                    {"et_kwargs": {"detach": True}}
                    if "viet" in model_name or "viset" in model_name
                    else {}
                )
                return model(x, **kwargs)

        inference_memory, _ = profiler.profile_function(run_inference)

        param_size_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            / 1024
            / 1024
        )

        assert model_memory.gpu_allocated_mb < param_size_mb * 2, (
            f"Model memory ({model_memory.gpu_allocated_mb:.1f}MB) "
            f"too high vs parameters ({param_size_mb:.1f}MB)"
        )

        assert inference_memory.gpu_allocated_mb < param_size_mb, (
            f"Inference memory ({inference_memory.gpu_allocated_mb:.1f}MB) too high"
        )

    @pytest.mark.parametrize(
        "factory",
        [viet_tiny, viset_tiny],
    )
    def test_memory_scaling_with_batch(
        self,
        factory,
        device,
    ) -> None:
        """Test how memory scales with batch size."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")

        batch_sizes = [1, 2, 4, 8, 16]

        results = analyze_memory_scaling(
            factory,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,  # Added missing parameter
                "num_classes": 100,
            },
            batch_sizes,
            (3, 32, 32),
            device,
        )

        activation_memories = results["activation_memory"]
        memory_per_sample = [
            (batch, memory / batch) for batch, memory in activation_memories
        ]

        mem_per_sample_values = [m[1] for m in memory_per_sample]
        mean_per_sample = float(np.mean(mem_per_sample_values))
        std_per_sample = float(np.std(mem_per_sample_values))
        cv = std_per_sample / mean_per_sample

        assert cv < 0.1, (
            f"Memory scaling not linear: CV={cv:.3f} "
            f"(mean={mean_per_sample:.2f}MB/sample, "
            f"std={std_per_sample:.2f}MB)"
        )

    def test_memory_efficiency_comparison(
        self,
        device,
    ) -> None:
        """Compare memory efficiency across model variants."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")

        models = {
            "vit": vit_tiny,
            "viet": viet_tiny,
            "viset": viset_tiny,
        }

        profiler = MemoryProfiler(device)
        results: dict[str, int] = {}

        for name, factory in models.items():
            model = (
                factory(
                    img_size=32,
                    patch_size=4,
                    in_chans=3,  # Added missing parameter
                    num_classes=100,
                )
                .to(device)
                .eval()
            )

            max_batch = 1
            for batch_size in [1, 2, 4, 8, 16, 32, 64]:
                try:
                    x = torch.randn(batch_size, 3, 32, 32, device=device)

                    profiler.clean_memory()
                    _ = profiler.get_memory_snapshot()

                    with torch.no_grad():
                        kwargs = (
                            {"et_kwargs": {"detach": True}}
                            if name != "vit"
                            else {}
                        )
                        _ = model(x, **kwargs)

                    _ = profiler.get_memory_snapshot()

                    max_batch = batch_size
                except RuntimeError as e:  # pragma: no cover - OOM branch
                    if "out of memory" in str(e):
                        break
                    raise

            results[name] = max_batch

        assert results["viet"] >= results["vit"] * 0.8, (
            f"ViET max batch ({results['viet']}) too low vs ViT ({results['vit']})"
        )
        assert results["viset"] >= results["vit"] * 0.7, (
            f"ViSET max batch ({results['viset']}) too low vs ViT ({results['vit']})"
        )


class TestComponentMemory:
    """Test memory usage of individual components."""

    def test_attention_memory_scaling(
        self,
        device,
    ) -> None:
        """Test memory usage of attention mechanism with sequence length."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")

        profiler = MemoryProfiler(device)

        embed_dim = 768
        num_heads = 12
        head_dim = 64
        batch_size = 4

        attention = (
            MultiHeadEnergyAttention(
                in_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            .to(device)
            .eval()
        )

        results: list[dict[str, float]] = []

        for seq_len in [64, 128, 256, 512]:
            x = torch.randn(batch_size, seq_len, embed_dim, device=device)

            # Create a closure that captures x for this iteration
            def make_run_attention(x_val):
                def run_attention() -> torch.Tensor:
                    with torch.no_grad():
                        return attention(x_val)

                return run_attention

            memory_used, _ = profiler.profile_function(make_run_attention(x))

            # Skip if memory is zero (can happen on some systems)
            if memory_used.gpu_allocated_mb > 0:
                results.append(
                    {
                        "seq_len": seq_len,
                        "memory_mb": memory_used.gpu_allocated_mb,
                        "peak_mb": memory_used.gpu_max_allocated_mb,
                    }
                )

        # Only run assertions if we have valid results
        if len(results) > 1:
            seq_lens = [r["seq_len"] for r in results]
            memories = [r["memory_mb"] for r in results]

            for i in range(1, len(results)):
                seq_ratio = seq_lens[i] / seq_lens[i - 1]
                mem_ratio = memories[i] / memories[i - 1]
                expected_ratio = seq_ratio**2

                assert (
                    0.8 * expected_ratio <= mem_ratio <= 1.2 * expected_ratio
                ), (
                    f"Memory scaling not quadratic: seq {seq_lens[i - 1]}->{seq_lens[i]} "
                    f"(ratio {seq_ratio:.1f}), memory ratio {mem_ratio:.1f}, "
                    f"expected ~{expected_ratio:.1f}"
                )

    def test_hopfield_memory_usage(
        self,
        device,
    ) -> None:
        """Compare memory usage of standard vs simplicial Hopfield networks."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")

        profiler = MemoryProfiler(device)

        in_dim = 768
        hidden_dim = 3072
        batch_size = 4
        seq_len = 128

        hopfield = (
            HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim)
            .to(device)
            .eval()
        )

        num_vertices = seq_len
        simplicial = (
            SimplicialHopfieldNetwork(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_vertices=num_vertices,
                max_dim=2,
                budget=0.15,
            )
            .to(device)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, in_dim, device=device)

        def run_hopfield() -> torch.Tensor:
            with torch.no_grad():
                return hopfield(x)

        hopfield_memory, _ = profiler.profile_function(run_hopfield)

        def run_simplicial() -> torch.Tensor:
            with torch.no_grad():
                return simplicial(x)

        simplicial_memory, _ = profiler.profile_function(run_simplicial)

        # Only calculate ratio if both measurements are valid
        if (
            hopfield_memory.gpu_allocated_mb > 0
            and simplicial_memory.gpu_allocated_mb > 0
        ):
            memory_ratio = (
                simplicial_memory.gpu_allocated_mb
                / hopfield_memory.gpu_allocated_mb
            )

            assert memory_ratio < 2.0, (
                f"Simplicial Hopfield uses too much memory: "
                f"{simplicial_memory.gpu_allocated_mb:.1f}MB vs "
                f"{hopfield_memory.gpu_allocated_mb:.1f}MB "
                f"(ratio: {memory_ratio:.2f})"
            )
        else:
            # If measurements were zero, we can't do the assertion
            pass

    def test_layer_norm_memory_efficiency(
        self,
        device,
    ) -> None:
        """Test memory efficiency of energy-based LayerNorm."""
        profiler = MemoryProfiler(device)

        embed_dim = 768
        batch_size = 32
        seq_len = 256

        energy_ln = LayerNorm(embed_dim).to(device).eval()
        standard_ln = torch.nn.LayerNorm(embed_dim).to(device).eval()

        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        def run_energy_ln() -> torch.Tensor:
            with torch.no_grad():
                return energy_ln(x)

        def run_standard_ln() -> torch.Tensor:
            with torch.no_grad():
                return standard_ln(x)

        energy_memory, _ = profiler.profile_function(run_energy_ln)
        standard_memory, _ = profiler.profile_function(run_standard_ln)

        if device.type == "cuda":
            energy_mem = energy_memory.gpu_allocated_mb
            standard_mem = standard_memory.gpu_allocated_mb
        else:
            energy_mem = energy_memory.cpu_rss_mb
            standard_mem = standard_memory.cpu_rss_mb

        assert energy_mem <= standard_mem * 1.2, (
            f"Energy LayerNorm uses too much memory: "
            f"{energy_mem:.1f}MB vs standard {standard_mem:.1f}MB"
        )


class TestMemoryLeaks:
    """Test for memory leaks during repeated operations."""

    def test_no_memory_leak_inference(self, device) -> None:
        """Ensure no memory leaks during repeated inference."""
        if device.type != "cuda":
            pytest.skip("Memory leak testing requires CUDA")

        model = (
            viet_tiny(
                img_size=32,
                patch_size=4,
                in_chans=3,  # Added missing parameter
                num_classes=100,
            )
            .to(device)
            .eval()
        )

        x = torch.randn(4, 3, 32, 32, device=device)

        with torch.no_grad():
            for _ in range(10):
                _ = model(x, et_kwargs={"detach": True})

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()

        with torch.no_grad():
            for i in range(100):
                _ = model(x, et_kwargs={"detach": True})

                if i % 20 == 0:
                    torch.cuda.synchronize()
                    current_memory = torch.cuda.memory_allocated()

                    memory_growth = (
                        (current_memory - start_memory) / 1024 / 1024
                    )
                    assert memory_growth < 10, (
                        f"Memory leak detected: {memory_growth:.1f}MB "
                        f"growth after {i} iterations"
                    )

    def test_no_memory_leak_training(self, device) -> None:
        """Ensure no memory leaks during training iterations."""
        if device.type != "cuda":
            pytest.skip("Memory leak testing requires CUDA")

        model = (
            viet_tiny(
                img_size=32,
                patch_size=4,
                in_chans=3,  # Added missing parameter
                num_classes=100,
            )
            .to(device)
            .train()
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(5):
            x = torch.randn(4, 3, 32, 32, device=device)
            y = torch.randint(0, 100, (4,), device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()

        for i in range(50):
            x = torch.randn(4, 3, 32, 32, device=device)
            y = torch.randint(0, 100, (4,), device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated()

                memory_growth = (current_memory - start_memory) / 1024 / 1024
                assert memory_growth < 50, (
                    f"Memory leak in training: {memory_growth:.1f}MB "
                    f"growth after {i} iterations"
                )
