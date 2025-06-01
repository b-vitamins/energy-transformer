"""Test inference latency distribution and consistency."""

import numpy as np
import pytest
import torch

from energy_transformer.models.vision import viet_tiny, viset_tiny

pytestmark = [pytest.mark.performance, pytest.mark.inference_bench]


class TestLatencyConsistency:
    """Test that inference latency is consistent and predictable."""

    @pytest.mark.parametrize("model_factory", [viet_tiny, viset_tiny])
    @pytest.mark.usefixtures("clean_state")
    def test_latency_distribution(self, model_factory, device) -> None:
        """Test that inference latency has low variance."""
        model = (
            model_factory(img_size=32, patch_size=4, num_classes=100)
            .to(device)
            .eval()
        )
        x = torch.randn(1, 3, 32, 32, device=device)

        with torch.no_grad():
            for _ in range(50):
                _ = model(x, et_kwargs={"detach": True})

        latencies = []
        with torch.no_grad():
            for _ in range(100):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                import time

                start = time.perf_counter()
                _ = model(x, et_kwargs={"detach": True})
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                latencies.append(latency)

        latencies = np.array(latencies)
        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))

        cv = std_latency / mean_latency
        assert cv < 0.1, (
            f"Latency variance too high: CV={cv:.3f} (std={std_latency:.4f}, mean={mean_latency:.4f})"
        )
        assert p95_latency < mean_latency * 1.2, (
            f"P95 latency too high: {p95_latency:.4f}s vs mean {mean_latency:.4f}s"
        )
        assert p99_latency < mean_latency * 1.5, (
            f"P99 latency too high: {p99_latency:.4f}s vs mean {mean_latency:.4f}s"
        )

        print(f"\nLatency stats for {model_factory.__name__}:")
        print(f"  Mean: {mean_latency * 1000:.2f}ms")
        print(f"  Std:  {std_latency * 1000:.2f}ms")
        print(f"  P95:  {p95_latency * 1000:.2f}ms")
        print(f"  P99:  {p99_latency * 1000:.2f}ms")

    @pytest.mark.usefixtures("clean_state")
    def test_first_inference_penalty(self, device) -> None:
        """Test that first inference after model creation isn't too slow."""
        first_run_ratios = []

        for _ in range(5):
            model = (
                viet_tiny(img_size=32, patch_size=4, num_classes=100)
                .to(device)
                .eval()
            )
            x = torch.randn(1, 3, 32, 32, device=device)

            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                import time

                start = time.perf_counter()
                _ = model(x, et_kwargs={"detach": True})
                if device.type == "cuda":
                    torch.cuda.synchronize()
                first_time = time.perf_counter() - start

                subsequent_times = []
                for _ in range(10):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = model(x, et_kwargs={"detach": True})
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    subsequent_times.append(time.perf_counter() - start)

            avg_subsequent = float(np.mean(subsequent_times))
            ratio = first_time / avg_subsequent
            first_run_ratios.append(ratio)

        avg_ratio = float(np.mean(first_run_ratios))
        assert avg_ratio < 5.0, (
            f"First inference too slow: {avg_ratio:.1f}x slower than subsequent runs"
        )
