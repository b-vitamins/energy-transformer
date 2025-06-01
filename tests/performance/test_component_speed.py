"""Component-level performance benchmarks."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch

from energy_transformer.layers import (
    MultiHeadEnergyAttention,
    HopfieldNetwork,
    SimplicialHopfieldNetwork,
    LayerNorm,
    PatchEmbedding,
)
from energy_transformer.models.base import EnergyTransformer

from .conftest import assert_performance

pytestmark = [pytest.mark.performance, pytest.mark.component_bench]


class TestLayerPerformance:
    """Benchmark individual layer performance."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    @pytest.mark.parametrize("num_heads", [8, 12])
    def test_attention_layer_speed(
        self,
        benchmark,
        seq_len: int,
        num_heads: int,
        device,
        clean_state,
        performance_tracker,
    ) -> None:
        """Benchmark MultiHeadEnergyAttention performance."""
        embed_dim = 768
        head_dim = embed_dim // num_heads
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

        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        with torch.no_grad():
            for _ in range(10):
                _ = attention(x)

        def run_attention() -> torch.Tensor:
            with torch.no_grad():
                return attention(x)

        benchmark.pedantic(
            run_attention,
            rounds=50,
            iterations=5,
            warmup_rounds=5,
        )

        test_name = f"attention_s{seq_len}_h{num_heads}"
        thresholds = {
            "attention_s64_h8": {"cpu": 0.01, "cuda": 0.002},
            "attention_s128_h8": {"cpu": 0.04, "cuda": 0.005},
            "attention_s256_h8": {"cpu": 0.15, "cuda": 0.02},
            "attention_s64_h12": {"cpu": 0.015, "cuda": 0.003},
            "attention_s128_h12": {"cpu": 0.06, "cuda": 0.008},
            "attention_s256_h12": {"cpu": 0.2, "cuda": 0.03},
        }

        assert_performance(
            benchmark,
            performance_tracker,
            test_name,
            device.type,
            thresholds.get(test_name, {}),
        )

        flops = 2 * seq_len * seq_len * embed_dim * batch_size
        time_sec = benchmark.stats.stats.mean
        gflops = flops / time_sec / 1e9

        benchmark.extra_info.update(
            {
                "seq_len": seq_len,
                "num_heads": num_heads,
                "gflops": gflops,
            }
        )

    @pytest.mark.parametrize("hidden_multiplier", [2, 4, 8])
    def test_hopfield_performance(
        self,
        benchmark,
        hidden_multiplier: int,
        device,
        clean_state,
    ) -> None:
        """Benchmark Hopfield network performance."""
        in_dim = 768
        hidden_dim = in_dim * hidden_multiplier
        batch_size = 4
        seq_len = 128

        hopfield = (
            HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim)
            .to(device)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, in_dim, device=device)

        def run_hopfield() -> torch.Tensor:
            with torch.no_grad():
                return hopfield(x)

        benchmark.pedantic(
            run_hopfield,
            rounds=50,
            iterations=5,
            warmup_rounds=5,
        )

        elements_processed = batch_size * seq_len
        throughput = elements_processed / benchmark.stats.stats.mean

        benchmark.extra_info.update(
            {
                "hidden_multiplier": hidden_multiplier,
                "hidden_dim": hidden_dim,
                "throughput_tokens_per_sec": throughput,
            }
        )

    def test_simplicial_vs_standard_hopfield(
        self,
        benchmark,
        device,
        clean_state,
    ) -> None:
        """Compare performance of simplicial vs standard Hopfield."""
        in_dim = 768
        hidden_dim = 3072
        batch_size = 4
        seq_len = 64

        standard = (
            HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim)
            .to(device)
            .eval()
        )

        simplicial = (
            SimplicialHopfieldNetwork(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_vertices=seq_len,
                max_dim=2,
                budget=0.15,
                dim_weights={1: 0.5, 2: 0.5},
            )
            .to(device)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, in_dim, device=device)

        def run_standard() -> torch.Tensor:
            with torch.no_grad():
                return standard(x)

        def run_simplicial() -> torch.Tensor:
            with torch.no_grad():
                return simplicial(x)

        standard_result = benchmark.pedantic(
            run_standard,
            rounds=50,
            iterations=5,
            warmup_rounds=5,
        )

        simplicial_result = benchmark.pedantic(
            run_simplicial,
            rounds=50,
            iterations=5,
            warmup_rounds=5,
        )

        standard_time = standard_result.stats.mean
        simplicial_time = simplicial_result.stats.mean
        overhead = (simplicial_time / standard_time - 1) * 100

        assert overhead < 200, (
            f"Simplicial Hopfield overhead too high: {overhead:.1f}% "
            f"({simplicial_time:.4f}s vs {standard_time:.4f}s)"
        )

        benchmark.extra_info.update(
            {
                "standard_time": standard_time,
                "simplicial_time": simplicial_time,
                "overhead_percent": overhead,
            }
        )

    def test_patch_embedding_speed(
        self,
        benchmark,
        device,
        clean_state,
    ) -> None:
        """Benchmark patch embedding performance."""
        batch_sizes = [1, 4, 16]
        image_sizes = [224, 384]
        patch_sizes = [16, 32]

        results: list[Dict[str, float]] = []

        for img_size in image_sizes:
            for patch_size in patch_sizes:
                patch_embed = (
                    PatchEmbedding(
                        img_size=img_size,
                        patch_size=patch_size,
                        in_chans=3,
                        embed_dim=768,
                    )
                    .to(device)
                    .eval()
                )

                for batch_size in batch_sizes:
                    x = torch.randn(
                        batch_size, 3, img_size, img_size, device=device
                    )

                    def run_patch_embed() -> torch.Tensor:
                        with torch.no_grad():
                            return patch_embed(x)

                    result = benchmark.pedantic(
                        run_patch_embed,
                        rounds=20,
                        iterations=5,
                        warmup_rounds=5,
                    )

                    time_ms = result.stats.mean * 1000
                    num_patches = (img_size // patch_size) ** 2

                    results.append(
                        {
                            "img_size": img_size,
                            "patch_size": patch_size,
                            "batch_size": batch_size,
                            "time_ms": time_ms,
                            "num_patches": num_patches,
                            "throughput_img_per_sec": batch_size
                            / result.stats.mean,
                        }
                    )

        benchmark.extra_info["patch_embedding_results"] = results

        for r in results:
            max_time = 10 if device.type == "cuda" else 50
            assert r["time_ms"] < max_time, (
                f"Patch embedding too slow: {r['time_ms']:.1f}ms for "
                f"{r['img_size']}x{r['img_size']} image with {r['patch_size']}x{r['patch_size']} patches"
            )


class TestEnergyTransformerBlock:
    """Test complete Energy Transformer block performance."""

    @pytest.mark.parametrize("et_steps", [1, 2, 4, 8])
    def test_et_block_scaling(
        self,
        benchmark,
        et_steps: int,
        device,
        clean_state,
    ) -> None:
        """Test how ET block performance scales with optimization steps."""
        embed_dim = 768
        num_heads = 12
        head_dim = 64
        hidden_dim = 3072
        batch_size = 4
        seq_len = 128

        et_block = (
            EnergyTransformer(
                layer_norm=LayerNorm(embed_dim),
                attention=MultiHeadEnergyAttention(
                    in_dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                ),
                hopfield=HopfieldNetwork(
                    in_dim=embed_dim, hidden_dim=hidden_dim
                ),
                steps=et_steps,
                alpha=0.125,
            )
            .to(device)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        def run_et_block() -> torch.Tensor:
            with torch.no_grad():
                return et_block(x, detach=True)

        result = benchmark.pedantic(
            run_et_block,
            rounds=20,
            iterations=3,
            warmup_rounds=5,
        )

        time_sec = result.stats.mean
        expected_time = time_sec / et_steps

        benchmark.extra_info.update(
            {
                "et_steps": et_steps,
                "time_sec": time_sec,
                "time_per_step": expected_time,
            }
        )

    def test_et_vs_standard_transformer(
        self,
        benchmark,
        device,
        clean_state,
    ) -> None:
        """Compare ET block with standard transformer block."""
        from torch.nn import TransformerEncoderLayer

        embed_dim = 768
        num_heads = 12
        hidden_dim = 3072
        batch_size = 4
        seq_len = 128

        et_block = (
            EnergyTransformer(
                layer_norm=LayerNorm(embed_dim),
                attention=MultiHeadEnergyAttention(
                    in_dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=embed_dim // num_heads,
                ),
                hopfield=HopfieldNetwork(
                    in_dim=embed_dim, hidden_dim=hidden_dim
                ),
                steps=4,
                alpha=0.125,
            )
            .to(device)
            .eval()
        )

        standard_block = (
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
            )
            .to(device)
            .eval()
        )

        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        def run_et() -> torch.Tensor:
            with torch.no_grad():
                return et_block(x, detach=True)

        et_result = benchmark.pedantic(
            run_et,
            rounds=20,
            iterations=3,
            warmup_rounds=5,
        )

        def run_standard() -> torch.Tensor:
            with torch.no_grad():
                return standard_block(x)

        standard_result = benchmark.pedantic(
            run_standard,
            rounds=20,
            iterations=3,
            warmup_rounds=5,
        )

        et_time = et_result.stats.mean
        standard_time = standard_result.stats.mean
        overhead = (et_time / standard_time - 1) * 100

        benchmark.extra_info.update(
            {
                "et_time": et_time,
                "standard_time": standard_time,
                "overhead_percent": overhead,
            }
        )

        assert overhead < 500, (
            f"ET overhead too high: {overhead:.1f}% "
            f"({et_time:.4f}s vs {standard_time:.4f}s)"
        )
