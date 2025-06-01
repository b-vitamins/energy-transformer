"""Debug utilities for the realisation system."""

import logging
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from torch.nn import Module

from .primitives import Context, Spec
from .realise import _get_config


@dataclass
class DebugEvent:
    """A single debug event during realisation."""

    timestamp: float
    event_type: str  # "enter", "exit", "cache_hit", "error"
    spec_type: str
    spec_id: int
    depth: int
    duration: Optional[float] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebugTracer:
    """Advanced debug tracer for spec realisation."""

    def __init__(self, max_events: int = 10000) -> None:
        self.events: Deque[DebugEvent] = deque(maxlen=max_events)
        self.spec_stack: List[tuple[Spec, float]] = []
        self.enabled = False

    def trace_enter(self, spec: Spec, depth: int, context: Context) -> None:
        """Record spec realisation entry."""
        if not self.enabled:
            return

        event = DebugEvent(
            timestamp=time.perf_counter(),
            event_type="enter",
            spec_type=spec.__class__.__name__,
            spec_id=id(spec),
            depth=depth,
            metadata={"context_dims": dict(context.dimensions)},
        )
        self.events.append(event)
        self.spec_stack.append((spec, event.timestamp))

    def trace_exit(
        self, spec: Spec, depth: int, module: Optional[Module] = None
    ) -> None:
        """Record spec realisation exit."""
        if not self.enabled:
            return

        start_time = None
        for i in range(len(self.spec_stack) - 1, -1, -1):
            if self.spec_stack[i][0] is spec:
                _, start_time = self.spec_stack.pop(i)
                break

        duration = time.perf_counter() - start_time if start_time else None

        event = DebugEvent(
            timestamp=time.perf_counter(),
            event_type="exit",
            spec_type=spec.__class__.__name__,
            spec_id=id(spec),
            depth=depth,
            duration=duration,
            metadata={"module_type": module.__class__.__name__ if module else None},
        )
        self.events.append(event)

    def trace_cache_hit(self, spec: Spec, depth: int) -> None:
        """Record cache hit."""
        if not self.enabled:
            return

        event = DebugEvent(
            timestamp=time.perf_counter(),
            event_type="cache_hit",
            spec_type=spec.__class__.__name__,
            spec_id=id(spec),
            depth=depth,
        )
        self.events.append(event)

    def trace_error(self, spec: Spec, depth: int, error: Exception) -> None:
        """Record error during realisation."""
        if not self.enabled:
            return

        event = DebugEvent(
            timestamp=time.perf_counter(),
            event_type="error",
            spec_type=spec.__class__.__name__,
            spec_id=id(spec),
            depth=depth,
            error=error,
        )
        self.events.append(event)

    def print_summary(self) -> None:
        """Print a summary of traced events."""
        if not self.events:
            print("No events traced")
            return

        spec_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "errors": 0}
        )
        for event in self.events:
            if event.event_type == "exit" and event.duration:
                spec_stats[event.spec_type]["count"] += 1
                spec_stats[event.spec_type]["total_time"] += event.duration
            elif event.event_type == "error":
                spec_stats[event.spec_type]["errors"] += 1

        print("\n=== Realisation Trace Summary ===")
        print(f"Total events: {len(self.events)}")
        print(f"Unique specs: {len(set(e.spec_id for e in self.events))}")
        print(f"Max depth: {max(e.depth for e in self.events)}")

        if spec_stats:
            print("\n--- Per Spec Type Statistics ---")
            for spec_type, stats in sorted(
                spec_stats.items(),
                key=lambda x: x[1]["total_time"],
                reverse=True,
            ):
                if stats["count"] > 0:
                    avg_time = stats["total_time"] / stats["count"]
                    print(
                        f"{spec_type:30} | Count: {stats['count']:5} | Total: {stats['total_time']:8.3f}s | Avg: {avg_time:8.5f}s | Errors: {stats['errors']}"
                    )

    def get_trace_events(self) -> List[DebugEvent]:
        """Get copy of trace events."""
        return list(self.events)


@contextmanager
def debug_realisation(
    log_level: int = logging.DEBUG,
    break_on_error: bool = False,
    trace_cache: bool = True,
    trace_realisation: bool = False,
    _trace_imports: bool = True,
) -> Iterator[None]:
    """Context manager for debugging realisation issues."""
    logger = logging.getLogger("energy_transformer.spec")
    old_level = logger.level
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        ),
    )
    logger.addHandler(handler)

    config = _get_config()
    old_warnings = config.warnings
    config.warnings = True

    tracer = None
    old_tracer = None
    if trace_realisation:
        tracer = DebugTracer()
        tracer.enabled = True
        old_tracer = getattr(config, "debug_tracer", None)
        config.debug_tracer = tracer

    if trace_cache:
        original_get = config.cache.get
        original_put = config.cache.put

        def traced_get(spec: Spec, context: Context) -> Module | None:
            result = original_get(spec, context)
            status = "HIT" if result else "MISS"
            logger.debug(
                "Cache %s: %s (hit rate: %.1f%%)",
                status,
                spec.__class__.__name__,
                config.cache.hit_rate * 100,
            )
            return result

        def traced_put(spec: Spec, context: Context, module: Module) -> None:
            logger.debug("Cache PUT: %s", spec.__class__.__name__)
            original_put(spec, context, module)

        config.cache.get = traced_get  # type: ignore[method-assign]
        config.cache.put = traced_put  # type: ignore[method-assign]

    try:
        yield tracer
    except Exception:
        if break_on_error:
            import pdb

            pdb.post_mortem()
        raise
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(handler)
        config.warnings = old_warnings

        if trace_realisation and tracer:
            tracer.print_summary()
            config.debug_tracer = old_tracer

        if trace_cache:
            config.cache.get = original_get  # type: ignore[method-assign]
            config.cache.put = original_put  # type: ignore[method-assign]


def inspect_cache_stats() -> None:
    """Print current cache statistics."""
    config = _get_config()
    cache = config.cache
    print("Cache Statistics:")
    print(f"  Enabled: {cache.enabled}")
    print(f"  Size: {len(cache._cache)}/{cache.max_size}")
    print(f"  Hit rate: {cache.hit_rate:.1%}")
    print(f"  Hits: {cache._hit_count}")
    print(f"  Misses: {cache._miss_count}")

    if cache._cache:
        print("\nCached specs:")
        for key in list(cache._cache.keys())[:10]:
            if isinstance(key, tuple) and len(key) > 1:
                spec_info = key[1]
                if isinstance(spec_info, tuple) and len(spec_info) > 1:
                    print(f"  - {spec_info[1]}")


def clear_cache() -> None:
    """Clear the realisation cache."""
    _get_config().cache.clear()
    print("Cache cleared")
