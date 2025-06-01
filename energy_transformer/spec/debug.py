"""Debug utilities for the realisation system."""

import logging
from contextlib import contextmanager
from typing import Any

from .realise import _config


@contextmanager
def debug_realisation(
    log_level: int = logging.DEBUG,
    break_on_error: bool = False,
    trace_cache: bool = True,
    trace_imports: bool = True,
):
    """Context manager for debugging realisation issues."""
    logger = logging.getLogger("energy_transformer.spec")
    old_level = logger.level
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)

    old_warnings = _config.warnings
    _config.warnings = True

    if trace_cache:
        original_get = _config.cache.get
        original_put = _config.cache.put

        def traced_get(spec: Any, context: Any) -> Any:
            result = original_get(spec, context)
            status = "HIT" if result else "MISS"
            logger.debug(
                f"Cache {status}: {spec.__class__.__name__} (hit rate: {_config.cache.hit_rate:.1%})"
            )
            return result

        def traced_put(spec: Any, context: Any, module: Any) -> None:
            logger.debug(f"Cache PUT: {spec.__class__.__name__}")
            original_put(spec, context, module)

        _config.cache.get = traced_get
        _config.cache.put = traced_put

    try:
        yield
    except Exception:
        if break_on_error:
            import pdb

            pdb.post_mortem()
        raise
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(handler)
        _config.warnings = old_warnings

        if trace_cache:
            _config.cache.get = original_get
            _config.cache.put = original_put


def inspect_cache_stats() -> None:
    """Print current cache statistics."""
    cache = _config.cache
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
    _config.cache.clear()
    print("Cache cleared")
