"""Test auto-import functionality."""

import pytest
from unittest.mock import Mock, patch
from torch import nn

from energy_transformer.spec import Context
from energy_transformer.spec.realise import AutoImporter
from energy_transformer.spec.library import HNSpec, MHEASpec, SHNSpec


pytestmark = pytest.mark.unit


class TestAutoImporter:
    """Test the AutoImporter class."""

    def test_successful_import(self):
        """Test successful auto-import."""
        spec = MHEASpec(num_heads=12, head_dim=64)
        context = Context(dimensions={"embed_dim": 768})
        importer = AutoImporter(context, warnings_enabled=False)

        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_class = Mock(return_value=Mock(spec=nn.Module))
            mock_module.MultiHeadEnergyAttention = mock_class
            mock_import.return_value = mock_module

            result = importer.try_import(spec)
            assert result is not None
            mock_class.assert_called_once()

    def test_import_failure(self):
        """Test handling of import failure."""
        spec = MHEASpec(num_heads=12, head_dim=64)
        context = Context(dimensions={"embed_dim": 768})
        importer = AutoImporter(context, warnings_enabled=False)

        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            result = importer.try_import(spec)
            assert result is None

    def test_spec_specific_handlers(self):
        """Test spec-specific parameter handling."""
        context = Context(dimensions={"embed_dim": 768})
        importer = AutoImporter(context, warnings_enabled=False)

        spec = MHEASpec(num_heads=12, head_dim=64)
        kwargs = importer._get_base_kwargs(spec)
        importer._apply_spec_specific_logic(spec, "MHEASpec", kwargs)
        assert kwargs["in_dim"] == 768

        spec = HNSpec(multiplier=4.0)
        kwargs = importer._get_base_kwargs(spec)
        importer._apply_spec_specific_logic(spec, "HNSpec", kwargs)
        assert kwargs["in_dim"] == 768
        assert kwargs["hidden_dim"] == 3072
        assert "multiplier" not in kwargs

        spec = SHNSpec(num_vertices=None, max_dim=2)
        context.set_dim("simplicial_vertices", 64)
        importer = AutoImporter(context, warnings_enabled=False)
        kwargs = importer._get_base_kwargs(spec)
        importer._apply_spec_specific_logic(spec, "SHNSpec", kwargs)
        assert kwargs["num_vertices"] == 64

    def test_kwargs_cleaning(self):
        """Test kwargs cleaning removes None and internal values."""
        context = Context()
        importer = AutoImporter(context, warnings_enabled=False)

        kwargs = {
            "valid": 123,
            "none_value": None,
            "_type": "SomeType",
            "_version": "1.0",
            "another_valid": "test",
        }

        cleaned = importer._clean_kwargs(kwargs)
        assert cleaned == {"valid": 123, "another_valid": "test"}
        assert "none_value" not in cleaned
        assert "_type" not in cleaned
        assert "_version" not in cleaned
