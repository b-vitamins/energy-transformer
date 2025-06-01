"""Test thread safety of realisation system."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from energy_transformer.spec import configure_realisation, realise, Context
from dataclasses import dataclass

from torch import nn

from energy_transformer.spec import ModuleCache, param, register, Spec


@dataclass(frozen=True)
class ThreadSpec(Spec):
    hidden: int = param(default=1)


@register(ThreadSpec)
def realise_simple(spec: ThreadSpec, _context: Context) -> nn.Linear:
    return nn.Linear(spec.hidden, spec.hidden)


class TestThreadSafety:
    """Test that the realisation system is thread-safe."""

    def test_concurrent_realisation(self):
        num_threads = 10
        num_specs_per_thread = 50

        def worker(thread_id: int):
            results = []
            configure_realisation(
                cache=ModuleCache(enabled=True), strict=bool(thread_id % 2)
            )
            for i in range(num_specs_per_thread):
                spec = ThreadSpec(hidden=thread_id * 10 + i + 1)
                module = realise(spec)
                results.append((thread_id, i, module))
            return results

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        assert len(all_results) == num_threads * num_specs_per_thread
        for thread_id, spec_id, module in all_results:
            expected_in = thread_id * 10 + spec_id + 1
            assert isinstance(module, nn.Linear)
            assert module.in_features == expected_in

    def test_config_isolation(self):
        barrier = threading.Barrier(2)
        results = {}

        def worker1():
            configure_realisation(strict=True, warnings=False)
            barrier.wait()
            time.sleep(0.1)
            from energy_transformer.spec.realise import _get_config

            config = _get_config()
            results["worker1"] = (config.strict, config.warnings)

        def worker2():
            configure_realisation(strict=False, warnings=True)
            barrier.wait()
            from energy_transformer.spec.realise import _get_config

            config = _get_config()
            results["worker2"] = (config.strict, config.warnings)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["worker1"] == (True, False)
        assert results["worker2"] == (False, True)
