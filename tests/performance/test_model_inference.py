"""Inference performance benchmarks for vision models."""

import pytest
import torch

from energy_transformer.models.vision import (
    viet_2l_cifar,
    viet_base,
    viet_small,
    viet_tiny,
    viset_2l_e50_t50_cifar,
    viset_base,
    viset_small,
    viset_tiny,
    vit_base,
    vit_small,
    vit_tiny,
    vit_tiny_cifar,
)

from .conftest import assert_performance

pytestmark = [pytest.mark.performance, pytest.mark.inference_bench]

MODEL_FACTORIES = {
    "vit_tiny": (
        vit_tiny,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "vit_small": (
        vit_small,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "vit_base": (
        vit_base,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "viet_tiny": (
        viet_tiny,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "viet_small": (
        viet_small,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "viet_base": (
        viet_base,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "viset_tiny": (
        viset_tiny,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "viset_small": (
        viset_small,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "viset_base": (
        viset_base,
        {"img_size": 224, "patch_size": 16, "num_classes": 1000},
    ),
    "vit_tiny_cifar": (vit_tiny_cifar, {"num_classes": 100}),
    "viet_2l_cifar": (viet_2l_cifar, {"num_classes": 100}),
    "viset_2l_cifar": (viset_2l_e50_t50_cifar, {"num_classes": 100}),
}

INFERENCE_THRESHOLDS = {
    "vit_tiny_224": {"cpu": 0.5, "cuda": 0.05},
    "vit_small_224": {"cpu": 1.0, "cuda": 0.08},
    "vit_base_224": {"cpu": 2.0, "cuda": 0.15},
    "viet_tiny_224": {"cpu": 0.6, "cuda": 0.06},
    "viet_small_224": {"cpu": 1.2, "cuda": 0.10},
    "viet_base_224": {"cpu": 2.5, "cuda": 0.20},
    "viset_tiny_224": {"cpu": 0.7, "cuda": 0.07},
    "viset_small_224": {"cpu": 1.4, "cuda": 0.12},
    "viset_base_224": {"cpu": 3.0, "cuda": 0.25},
    "vit_tiny_cifar_32": {"cpu": 0.1, "cuda": 0.02},
    "viet_2l_cifar_32": {"cpu": 0.15, "cuda": 0.03},
    "viset_2l_cifar_32": {"cpu": 0.2, "cuda": 0.04},
}


class TestModelInference:
    """Test inference performance of vision models."""

    @pytest.mark.parametrize(
        ("model_name", "model_config"),
        tuple(MODEL_FACTORIES.items()),
    )
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.usefixtures("clean_state")
    def test_single_forward_pass(
        self,
        benchmark,
        model_name: str,
        model_config: tuple,
        batch_size: int,
        device,
        performance_tracker,
    ) -> None:
        """Benchmark single forward pass for each model."""
        factory, kwargs = model_config
        model = factory(**kwargs).to(device).eval()

        img_size = 32 if "cifar" in model_name else kwargs.get("img_size", 224)

        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        et_kwargs = {}
        if "viet" in model_name or "viset" in model_name:
            et_kwargs = {"et_kwargs": {"detach": True}}

        with torch.no_grad():
            for _ in range(10):
                _ = model(x, **et_kwargs)

        def run_inference() -> torch.Tensor:
            with torch.no_grad():
                return model(x, **et_kwargs)

        benchmark.pedantic(
            run_inference, rounds=20, iterations=5, warmup_rounds=5
        )

        test_key = f"{model_name}_{img_size}"
        assert_performance(
            benchmark,
            performance_tracker,
            f"{test_key}_b{batch_size}",
            device.type,
            INFERENCE_THRESHOLDS.get(test_key, {}),
        )

        benchmark.extra_info.update(
            {
                "model": model_name,
                "batch_size": batch_size,
                "image_size": img_size,
                "device": device.type,
                "params_millions": sum(p.numel() for p in model.parameters())
                / 1e6,
            }
        )

    @pytest.mark.parametrize("model_name", ["viet_tiny", "viset_tiny"])
    @pytest.mark.usefixtures("clean_state")
    def test_et_steps_scaling(
        self,
        benchmark,
        model_name: str,
        device,
    ) -> None:
        """Test how inference time scales with ET steps."""
        factory = MODEL_FACTORIES[model_name][0]

        results = []
        for et_steps in [1, 2, 4, 8]:
            model = (
                factory(
                    img_size=32,
                    patch_size=4,
                    num_classes=100,
                    et_steps=et_steps,
                    et_alpha=0.125,
                )
                .to(device)
                .eval()
            )

            x = torch.randn(4, 3, 32, 32, device=device)

            with torch.no_grad():
                for _ in range(5):
                    _ = model(x, et_kwargs={"detach": True})

                import time

                start = time.perf_counter()
                for _ in range(20):
                    _ = model(x, et_kwargs={"detach": True})
                elapsed = (time.perf_counter() - start) / 20

            results.append((et_steps, elapsed))

        base_time = results[0][1]
        for steps, elapsed in results[1:]:
            expected = base_time * steps
            assert elapsed < expected * 1.5, (
                f"ET steps scaling exceeded: {elapsed:.3f}s > {expected * 1.5:.3f}s for {steps} steps"
            )

        benchmark.extra_info["et_steps_scaling"] = results

    @pytest.mark.slow
    @pytest.mark.usefixtures("clean_state")
    def test_throughput(  # noqa: C901 - complex benchmark logic
        self,
        benchmark,
        device,
    ) -> None:
        """Test maximum throughput (images/second) for each model type."""
        models_to_test = ["vit_tiny", "viet_tiny", "viset_tiny"]
        batch_sizes = [1, 4, 16, 32] if device.type == "cuda" else [1, 4, 8]

        throughput_results = {}

        for model_name in models_to_test:
            factory, kwargs = MODEL_FACTORIES[model_name]
            model = factory(**kwargs).to(device).eval()

            et_kwargs = (
                {"et_kwargs": {"detach": True}} if "vi" in model_name else {}
            )

            best_throughput = 0.0
            best_batch = 0

            for batch_size in batch_sizes:
                try:
                    x = torch.randn(batch_size, 3, 224, 224, device=device)

                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(x, **et_kwargs)

                        import time

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        start = time.perf_counter()
                        for _ in range(100):
                            _ = model(x, **et_kwargs)

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        elapsed = time.perf_counter() - start
                        throughput = (batch_size * 100) / elapsed

                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_batch = batch_size

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    raise

            throughput_results[model_name] = {
                "best_throughput": best_throughput,
                "best_batch_size": best_batch,
            }

        min_throughput = {"cpu": 10, "cuda": 100}
        for model_name, result in throughput_results.items():
            assert result["best_throughput"] > min_throughput.get(
                device.type, 1
            ), (
                f"{model_name} throughput too low: {result['best_throughput']:.1f} img/s"
            )

        benchmark.extra_info["throughput_results"] = throughput_results


class TestBatchScaling:
    """Test how performance scales with batch size."""

    @pytest.mark.parametrize(
        "model_name", ["vit_tiny", "viet_tiny", "viset_tiny"]
    )
    @pytest.mark.usefixtures("clean_state")
    def test_batch_scaling_efficiency(
        self,
        benchmark,
        model_name: str,
        device,
    ) -> None:
        """Test that larger batches are more efficient per sample."""
        factory, kwargs = MODEL_FACTORIES[model_name]

        if "_cifar" not in model_name:
            kwargs = {"img_size": 32, "patch_size": 4, "num_classes": 100}

        model = factory(**kwargs).to(device).eval()
        et_kwargs = (
            {"et_kwargs": {"detach": True}} if "vi" in model_name else {}
        )

        times_per_sample = []
        batch_sizes = [1, 2, 4, 8, 16] if device.type == "cuda" else [1, 2, 4]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 32, 32, device=device)

            def run(x=x) -> torch.Tensor:
                with torch.no_grad():
                    return model(x, **et_kwargs)

            benchmark.pedantic(run, rounds=10, iterations=5, warmup_rounds=3)

            time_per_sample = benchmark.stats.stats.mean / batch_size
            times_per_sample.append((batch_size, time_per_sample))

        for i in range(1, len(times_per_sample)):
            prev_batch, prev_time = times_per_sample[i - 1]
            curr_batch, curr_time = times_per_sample[i]
            assert curr_time < prev_time * 0.9, (
                f"Batch scaling not efficient: batch {curr_batch} ({curr_time:.4f}s/sample) "
                f"not faster than batch {prev_batch} ({prev_time:.4f}s/sample)"
            )

        benchmark.extra_info["batch_scaling"] = times_per_sample
