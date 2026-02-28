"""
Unit tests for graph analysis functionality.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from deepx_dock.analyze.graph.analyze_graph import (
    GraphAnalyzer,
    suggest_batch_size_from_edges,
    _convert_orbital_types_to_string,
    _load_graph_info_from_npz,
    analyze_graph,
)


class TestSuggestBatchSize:
    """Tests for suggest_batch_size_from_edges function."""

    def test_small_edges(self):
        """Test with small edge count (should recommend larger batch)."""
        batch_size, valid = suggest_batch_size_from_edges(10)
        assert valid is True
        assert batch_size >= 20  # min_batch = ceil(200/10) = 20

    def test_medium_edges(self):
        """Test with medium edge count."""
        batch_size, valid = suggest_batch_size_from_edges(100)
        assert valid is True
        assert batch_size >= 2  # min_batch = ceil(200/100) = 2

    def test_large_edges(self):
        """Test with large edge count."""
        batch_size, valid = suggest_batch_size_from_edges(1000)
        assert valid is True
        assert batch_size == 1  # min_batch = 1

    def test_very_large_edges(self):
        """Test with very large edge count (> 5000)."""
        batch_size, valid = suggest_batch_size_from_edges(10000)
        assert valid is True
        assert batch_size == 1  # For avg_edges > 5000, batch=1 is acceptable

    def test_zero_edges(self):
        """Test with zero edge count."""
        batch_size, valid = suggest_batch_size_from_edges(0)
        assert valid is False
        assert batch_size == 1

    def test_negative_edges(self):
        """Test with negative edge count."""
        batch_size, valid = suggest_batch_size_from_edges(-10)
        assert valid is False
        assert batch_size == 1


class TestConvertOrbitalTypes:
    """Tests for orbital type conversion functions."""

    def test_sp_orbitals(self):
        """Test s and p orbital conversion."""
        result = _convert_orbital_types_to_string([0, 1])
        assert result == "sp"

    def test_spd_orbitals(self):
        """Test s, p, d orbital conversion."""
        result = _convert_orbital_types_to_string([0, 1, 2])
        assert result == "spd"

    def test_spdf_orbitals(self):
        """Test s, p, d, f orbital conversion."""
        result = _convert_orbital_types_to_string([0, 1, 2, 3])
        assert result == "spdf"

    def test_empty_orbitals(self):
        """Test empty orbital list."""
        result = _convert_orbital_types_to_string([])
        assert result == ""

    def test_none_orbitals(self):
        """Test None orbital list."""
        result = _convert_orbital_types_to_string(None)
        assert result == ""

    def test_numpy_array(self):
        """Test numpy array input."""
        result = _convert_orbital_types_to_string(np.array([0, 1, 2]))
        assert result == "spd"


class TestGraphAnalyzer:
    """Tests for GraphAnalyzer class."""

    @pytest.fixture
    def temp_graph_file(self):
        """Create a temporary graph file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "test.memory.npz"

            info = {
                "dataset_name": "test-dataset",
                "graph_type": "train-H",
                "dtype": np.dtype(np.float32).str,
                "elements_orbital_map": {
                    1: [0],  # H: s
                    8: [0, 1],  # O: sp
                },
                "common_orbital_types": np.array([0, 1, 2]),
                "common_orbital_num": 3,
                "spinful": False,
                "node_num_list": np.array([5, 6, 4, 5, 5]),
                "edge_num_list": np.array([50, 60, 40, 50, 50]),
                "entries_num_list": np.array([100, 120, 80, 100, 100]),
                "structure_num": 5,
                "all_structure_id": np.array(["0", "1", "2", "3", "4"]),
            }

            save_data = {"__num_graphs__": np.array(5)}
            for key, value in info.items():
                save_data[f"__info__:{key}"] = value

            np.savez_compressed(graph_path, **save_data)
            yield graph_path

    def test_load_memory_graph(self, temp_graph_file):
        """Test loading memory graph file."""
        analyzer = GraphAnalyzer(temp_graph_file)
        assert analyzer.storage_type == "memory"

        info = analyzer.info
        assert info["dataset_name"] == "test-dataset"
        assert info["graph_type"] == "train-H"
        assert info["structure_num"] == 5

    def test_analyze_graph_features(self, temp_graph_file):
        """Test analyzing graph features."""
        analyzer = GraphAnalyzer(temp_graph_file)
        analysis = analyzer.analyze_graph_features()

        assert analysis["storage_type"] == "memory"
        assert analysis["graph_type"] == "train-H"
        assert analysis["num_elements"] == 2
        assert analysis["model_type"] == "dedicated"
        assert analysis["spinful"] is False
        assert analysis["structure_num"] == 5
        assert analysis["avg_nodes"] == 5.0
        assert analysis["avg_edges"] == 50.0
        assert analysis["batch_size_valid"] is True
        assert "H" in analysis["elements"]
        assert "O" in analysis["elements"]

    def test_analyze_with_output(self, temp_graph_file):
        """Test analyzing graph and saving to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "analysis.json"

            analysis = analyze_graph(
                graph_path=temp_graph_file,
                output=output_path,
                quiet=True,
            )

            assert output_path.exists()
            with open(output_path) as f:
                saved = json.load(f)
            assert saved["dataset_name"] == "test-dataset"
            assert "toml_hints" in saved


class TestLoadGraphInfoFromNpz:
    """Tests for _load_graph_info_from_npz function."""

    def test_load_simple_npz(self):
        """Test loading a simple npz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"

            data = {
                "scalar": np.array(42),
                "array": np.array([1, 2, 3]),
                "string": np.array("hello"),
            }
            np.savez(npz_path, **data)

            info = _load_graph_info_from_npz(npz_path)

            assert info["scalar"] == 42
            assert list(info["array"]) == [1, 2, 3]


class TestBatchSizeFormula:
    """Tests for batch size formula constraint."""

    def test_constraint_200_to_5000(self):
        """Verify batch_size formula: 200 <= avg_edges * batch_size <= 5000."""
        test_cases = [
            (10, 20),  # 200 <= 10*20 = 200 <= 5000 ✓
            (50, 4),  # 200 <= 50*4 = 200 <= 5000 ✓
            (100, 2),  # 200 <= 100*2 = 200 <= 5000 ✓
            (200, 1),  # 200 <= 200*1 = 200 <= 5000 ✓
            (1000, 1),  # 200 <= 1000*1 = 1000 <= 5000 ✓
        ]

        for avg_edges, expected_batch in test_cases:
            batch_size, valid = suggest_batch_size_from_edges(avg_edges)
            if valid:
                product = avg_edges * batch_size
                assert 200 <= product <= 5000, (
                    f"Failed for avg_edges={avg_edges}, batch={batch_size}, product={product}"
                )
