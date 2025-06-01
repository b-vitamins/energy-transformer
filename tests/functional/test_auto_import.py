"""Test realisation system robustness."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from energy_transformer.spec import (
    Spec,
    configure_realisation,
    param,
)
from energy_transformer.spec.realise import (
    Realiser,
)

pytestmark = pytest.mark.functional


@dataclass(frozen=True)
class SimpleSpec(Spec):
    """Simple spec for testing."""

    value: int = param(default=1)


@dataclass(frozen=True)
class DeepSpec(Spec):
    """Spec that creates deep nesting."""

    depth: int = param(default=10)

    def children(self) -> list[Spec]:
        if self.depth > 0:
            return [DeepSpec(depth=self.depth - 1)]
        return []


class TestAutoImportLogging:
    """Test auto-import with proper logging."""

    def test_import_failure_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.DEBUG,
            logger="energy_transformer.spec.realise",
        )

        @dataclass(frozen=True)
        class UnknownSpec(Spec):
            pass

        realiser = Realiser()
        result = realiser._try_auto_import(UnknownSpec())
        assert result is None
        assert len(caplog.records) > 0
        assert "No auto-import mapping" in caplog.text

    def test_missing_module_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.WARNING,
            logger="energy_transformer.spec.realise",
        )
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("non_existent_module", "TestClass")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is None
            assert "Failed to import non_existent_module" in caplog.text
            assert "pip install" in caplog.text

    def test_missing_class_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.WARNING,
            logger="energy_transformer.spec.realise",
        )
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("torch.nn", "NonExistentClass")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is None
            assert "has no attribute NonExistentClass" in caplog.text
            assert "Available attributes:" in caplog.text

    def test_instantiation_failure_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.WARNING,
            logger="energy_transformer.spec.realise",
        )
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("torch.nn", "Linear")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is None
            assert "Failed to instantiate Linear" in caplog.text
            assert (
                "missing" in caplog.text.lower()
                or "required" in caplog.text.lower()
            )

    def test_successful_import_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(logging.INFO, logger="energy_transformer.spec.realise")
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("torch.nn", "Identity")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is not None
            assert isinstance(result, nn.Identity)
            assert "Successfully auto-imported" in caplog.text
