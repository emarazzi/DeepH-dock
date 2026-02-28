"""
Analyze DeepH graph files to extract training-relevant features.

Supports both memory (.npz) and disk (.db + .npz) storage formats.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from deepx_dock.CONSTANT import PERIODIC_TABLE_INDEX_TO_SYMBOL
from deepx_dock.analyze.dataset.e3nn_irreps import Irreps


GRAPH_INFO_KEYS = [
    "dataset_name",
    "graph_type",
    "dtype",
    "elements_orbital_map",
    "common_orbital_types",
    "common_orbital_num",
    "elements_fitting_map",
    "common_fitting_types",
    "common_fitting_num",
    "spinful",
    "node_num_list",
    "edge_num_list",
    "entries_num_list",
    "structure_num",
    "all_structure_id",
]


def suggest_batch_size_from_edges(avg_edges: float) -> tuple[int, bool]:
    """
    Suggest batch size based on constraint: 200 <= avg_edges * batch_size <= 5000.

    For very large graphs (avg_edges > 5000), batch_size=1 is acceptable.

    Prefers smaller batch sizes for higher accuracy.

    Args:
        avg_edges: Average number of edges per structure.

    Returns:
        (recommended_batch_size, is_valid) tuple.
        is_valid is False if avg_edges <= 0, indicating manual calculation needed.
    """
    if avg_edges <= 0:
        return 1, False

    min_batch = max(1, math.ceil(200 / avg_edges))
    max_batch = math.floor(5000 / avg_edges) if avg_edges <= 5000 else 1
    candidates = [1, 2, 4, 10, 20, 50, 100]

    valid = [b for b in candidates if min_batch <= b <= max_batch]
    return (valid[0] if valid else 1), True


def _load_graph_info_from_npz(npz_path: Path) -> dict[str, Any]:
    """Load graph info from memory storage (.npz) file."""
    with np.load(npz_path, allow_pickle=True) as data:
        info = {}
        for key in data.files:
            val = data[key]
            if val.ndim == 0:
                val = val.item()
            info[key] = val
        return info


def _parse_info_from_raw_keys(info: dict[str, Any]) -> dict[str, Any]:
    """Parse info dict with __info__: prefix (from DeepH-pack format)."""
    parsed = {}
    for key, val in info.items():
        if key.startswith("__info__:"):
            real_key = key[9:]
            parsed[real_key] = val
        elif key == "__num_graphs__":
            continue
        else:
            parsed[key] = val
    return parsed


def _convert_list_to_orbital_string(ls: list[int] | np.ndarray | None) -> str:
    """Convert orbital type list to string format using bincount.

    Example:
        [0, 0, 0, 1, 1, 2, 3] -> "s3p2d1f1"
    """
    if ls is None or (hasattr(ls, "__len__") and len(ls) == 0):
        return ""

    orbital_map = ["s", "p", "d", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    ls_arr = np.array(ls) if not isinstance(ls, np.ndarray) else ls
    ls_counts = np.bincount(ls_arr)

    string = ""
    for ll, n in enumerate(ls_counts):
        if n > 0:
            string += f"{orbital_map[ll]}{n}"

    return string


def _suggest_bs3b_orbital_types(ls: list[int] | np.ndarray | None) -> str:
    """Suggest BS3B orbital types based on common orbital types.

    Uses orbi_factor to determine the number of channels for each orbital type.
    """
    if ls is None or (hasattr(ls, "__len__") and len(ls) == 0):
        return ""

    orbital_map = ["s", "p", "d", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    orbi_factor = [5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ls_arr = np.array(ls) if not isinstance(ls, np.ndarray) else ls
    ls_counts = np.bincount(ls_arr)

    string = ""
    for ll, n in enumerate(ls_counts):
        if n > 0 and ll < len(orbi_factor):
            string += f"{orbital_map[ll]}{n * orbi_factor[ll]}"

    return string


def _gen_suggest_irreps_in(irreps_comm_orb, spinful):
    """Generate suggested irreps with minimum channel count of 2.

    This matches the logic from DeepH-pack DatasetAnalyzer._gen_suggest_irreps_in.
    """

    def suggest_mul(mul):
        _exp_suggest = 2 ** int(np.ceil(np.log2(mul)))
        _8_suggest = 8 * ((mul - 1) // 8 + 1)
        if mul <= 2:
            return 2
        elif mul <= 8:
            return _exp_suggest
        else:
            return _8_suggest

    return Irreps([(suggest_mul(mul), (ll, p)) for mul, (ll, p) in irreps_comm_orb // (1 + spinful)]).regroup()


def _common_orbital_types_to_irreps(
    common_orbital_types: list[int],
    spinful: bool,
    consider_parity: bool = False,
) -> str:
    """Convert common orbital types to irreps string.

    Returns suggested irreps with channel counts >= 2.
    Matches the logic from DeepH-pack.
    """
    from deepx_dock.analyze.dataset.e3nn_irreps import Irreps

    irreps_l_list = []
    for orb_l1 in common_orbital_types:
        for orb_l2 in common_orbital_types:
            use_odd_parity = consider_parity and (((orb_l1 + orb_l2) % 2) == 1)
            p = -1 if use_odd_parity else 1
            irreps = [(1, (orb_l, p)) for orb_l in range(abs(orb_l1 - orb_l2), orb_l1 + orb_l2 + 1)]
            if spinful:
                irreps_x1 = []
                for _, ir in irreps:
                    ir_x1 = [(1, (orb_l, p)) for orb_l in range(abs(ir[0] - 1), ir[0] + 2)]
                    irreps_x1.extend(ir_x1)
                irreps.extend(irreps_x1)
            irreps_l_list.extend(irreps)

    if spinful:
        irreps_l_list += irreps_l_list

    irreps_comm_orb = Irreps(irreps_l_list).regroup()

    # Generate suggested irreps with minimum channel count of 2
    irreps_suggested = _gen_suggest_irreps_in(irreps_comm_orb, spinful)

    return str(irreps_suggested)


class GraphAnalyzer:
    """Analyze DeepH graph files to extract training-relevant features.

    Supports both memory (.npz) and disk (.db + .npz) storage formats.
    """

    def __init__(self, graph_path: str | Path):
        """Initialize GraphAnalyzer.

        Args:
            graph_path: Path to graph file (.npz for memory, or directory containing
                       disk storage files).
        """
        self.graph_path = Path(graph_path)
        self._info: dict[str, Any] | None = None
        self._storage_type: str | None = None

    @property
    def info(self) -> dict[str, Any]:
        """Load and return graph info."""
        if self._info is None:
            self._load_graph_info()
            assert self._info is not None
        return self._info

    @property
    def storage_type(self) -> str:
        """Return storage type: 'memory' or 'disk'."""
        if self._storage_type is None:
            self._detect_storage_type()
            assert self._storage_type is not None
        return self._storage_type

    def _detect_storage_type(self) -> None:
        """Detect whether graph is memory or disk storage.

        Supports multiple input formats:
        - *.memory.npz -> memory storage
        - *.disk.npz -> disk storage (main metadata)
        - *.disk.part*.db -> disk storage (find main .disk.npz)
        - *.disk.part*.npz -> disk storage (find main .disk.npz)
        - directory -> auto-detect
        """
        if self.graph_path.is_file():
            name = self.graph_path.name.lower()
            if ".memory." in name and self.graph_path.suffix == ".npz":
                self._storage_type = "memory"
            elif ".disk." in name:
                self._storage_type = "disk"
                # If input is a .db or part .npz file, resolve to main .disk.npz
                if ".part" in name or self.graph_path.suffix == ".db":
                    self._resolve_disk_main_npz()
            elif self.graph_path.suffix == ".npz":
                # Default to memory for .npz without .disk. or .memory.
                self._storage_type = "memory"
            else:
                raise ValueError(f"Unknown file format: {self.graph_path}")
        elif self.graph_path.is_dir():
            # Auto-detect from directory contents
            disk_npz = list(self.graph_path.glob("*.disk.npz"))
            memory_npz = list(self.graph_path.glob("*.memory.npz"))
            if disk_npz:
                self._storage_type = "disk"
                self.graph_path = disk_npz[0]
            elif memory_npz:
                self._storage_type = "memory"
                self.graph_path = memory_npz[0]
            else:
                raise ValueError(f"No valid graph files found in {self.graph_path}")
        else:
            raise ValueError(f"Invalid graph path: {self.graph_path}")

    def _resolve_disk_main_npz(self) -> None:
        """Resolve disk storage main .disk.npz from part file or .db file.

        Example:
            data.disk.part1-of-3.db -> data.disk.npz
            data.disk.part2-of-3.npz -> data.disk.npz
        """
        name = self.graph_path.name
        # Remove .part* suffix and change extension to .npz
        import re

        # Match pattern: name.disk.part{N}-of-{M}.{ext}
        match = re.match(r"(.+\.disk)\.part\d+-of-\d+\.(db|npz)$", name)
        if match:
            base_name = match.group(1) + ".npz"
            main_npz = self.graph_path.parent / base_name
            if main_npz.exists():
                self.graph_path = main_npz
            else:
                raise ValueError(f"Main disk npz file not found: {main_npz}")
        # Handle .db file without .part
        elif name.endswith(".db") and ".disk." in name:
            main_npz = self.graph_path.with_suffix(".npz")
            if main_npz.exists():
                self.graph_path = main_npz
            else:
                raise ValueError(f"Main disk npz file not found: {main_npz}")

    def _load_graph_info(self) -> None:
        """Load graph info from file."""
        self._detect_storage_type()

        if self.storage_type == "memory":
            self._info = self._load_memory_graph_info()
        else:
            self._info = self._load_disk_graph_info()

    def _load_memory_graph_info(self) -> dict[str, Any]:
        """Load info from memory storage (.npz) file."""
        if self.graph_path.is_file():
            npz_path = self.graph_path
        else:
            npz_files = list(self.graph_path.glob("*.memory.npz"))
            if not npz_files:
                raise ValueError(f"No memory graph files found in {self.graph_path}")
            npz_path = npz_files[0]

        info = _load_graph_info_from_npz(npz_path)

        if any(k.startswith("__info__:") for k in info.keys()):
            info = _parse_info_from_raw_keys(info)

        return self._normalize_info(info)

    def _load_disk_graph_info(self) -> dict[str, Any]:
        """Load info from disk storage (.db + .npz) files."""
        if self.graph_path.is_file():
            npz_path = self.graph_path
        else:
            npz_files = list(self.graph_path.glob("*.disk.npz"))
            if not npz_files:
                raise ValueError(f"No disk graph files found in {self.graph_path}")
            npz_path = npz_files[0]

        info = _load_graph_info_from_npz(npz_path)
        info = _parse_info_from_raw_keys(info) if any(k.startswith("__info__:") for k in info.keys()) else info

        return self._normalize_info(info)

    def _normalize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Normalize info dict to consistent format."""
        normalized = {}

        for key in GRAPH_INFO_KEYS:
            val = info.get(key)
            if val is not None:
                if isinstance(val, np.ndarray):
                    if val.ndim == 0:
                        val = val.item()
                    elif key in ["node_num_list", "edge_num_list", "entries_num_list"]:
                        val = val.tolist()
                    elif key == "common_orbital_types":
                        val = val.tolist()
                    elif key == "all_structure_id":
                        val = val.tolist()
                normalized[key] = val

        return normalized

    def analyze_graph_features(
        self,
        consider_parity: bool = False,
    ) -> dict[str, Any]:
        """Extract training-relevant features from graph.

        Args:
            consider_parity: Whether to consider parity in irreps calculation.

        Returns:
            Dictionary containing:
                - storage_type: "memory" or "disk"
                - graph_type: Type of graph (e.g., "train-H", "train-HS")
                - spinful: Whether spin-polarized
                - elements: List of element symbols
                - num_elements: Number of unique elements
                - model_type: "dedicated" (<=3 elements) or "general" (>3 elements)
                - common_orbital_types: String representation (e.g., "spd")
                - structure_num: Total number of structures
                - node_num_list: List of node counts per structure
                - edge_num_list: List of edge counts per structure
                - avg_nodes: Average nodes per structure
                - avg_edges: Average edges per structure
                - recommended_batch_size: Based on edge constraint
                - batch_size_valid: Whether batch_size calculation was valid
                - irreps: Suggested irreps for model
        """
        info = self.info

        elements_orbital_map = info.get("elements_orbital_map", {})
        if isinstance(elements_orbital_map, dict):
            elements = [PERIODIC_TABLE_INDEX_TO_SYMBOL.get(int(z), f"Z{z}") for z in elements_orbital_map.keys()]
        else:
            elements = []

        num_elements = len(elements)
        model_type = "dedicated" if num_elements <= 3 else "general"

        common_orbital_types = info.get("common_orbital_types", [])
        if isinstance(common_orbital_types, np.ndarray):
            common_orbital_types = common_orbital_types.tolist()

        common_orbital_str = _convert_list_to_orbital_string(common_orbital_types)

        node_num_list = info.get("node_num_list", [])
        edge_num_list = info.get("edge_num_list", [])

        avg_nodes = sum(node_num_list) / len(node_num_list) if node_num_list else 0
        avg_edges = sum(edge_num_list) / len(edge_num_list) if edge_num_list else 0

        recommended_batch_size, batch_size_valid = suggest_batch_size_from_edges(avg_edges)

        spinful = info.get("spinful", False)

        irreps_suggested = ""
        irreps_dim = 0
        if common_orbital_types:
            try:
                irreps_suggested = _common_orbital_types_to_irreps(common_orbital_types, spinful, consider_parity)
                irreps_dim = Irreps(irreps_suggested).dim
            except Exception:
                pass

        bs3b_orbital_str = _suggest_bs3b_orbital_types(common_orbital_types)

        return {
            "storage_type": self.storage_type,
            "graph_type": info.get("graph_type", "unknown"),
            "dataset_name": info.get("dataset_name", "unknown"),
            "spinful": spinful,
            "elements": elements,
            "num_elements": num_elements,
            "model_type": model_type,
            "common_orbital_types": common_orbital_str,
            "structure_num": info.get("structure_num", 0),
            "node_num_list": node_num_list,
            "edge_num_list": edge_num_list,
            "entries_num_list": info.get("entries_num_list", []),
            "avg_nodes": round(avg_nodes, 1) if avg_nodes > 0 else "unknown",
            "avg_edges": round(avg_edges, 1) if avg_edges > 0 else "unknown",
            "recommended_batch_size": recommended_batch_size,
            "batch_size_valid": batch_size_valid,
            "irreps": {
                "suggested": irreps_suggested,
                "dim": irreps_dim,
            },
            "suggested_bs3b_orbital_types": bs3b_orbital_str,
        }

    def print_summary(self, analysis: dict[str, Any] | None = None) -> None:
        """Print a human-readable summary of graph analysis."""
        if analysis is None:
            analysis = self.analyze_graph_features()

        lines = []
        lines.append("=" * 60)
        lines.append("GRAPH ANALYSIS RESULTS")
        lines.append("=" * 60)

        lines.append("\nBASIC INFO")
        lines.append(f"   Storage type:     {analysis['storage_type']}")
        lines.append(f"   Graph type:       {analysis['graph_type']}")
        lines.append(f"   Dataset name:     {analysis['dataset_name']}")
        lines.append(f"   Model type:       {analysis['model_type']} ({analysis['num_elements']} elements)")
        lines.append(f"   Spinful:          {analysis['spinful']}")
        lines.append(f"   Total structures: {analysis['structure_num']:,}")
        lines.append(f"   Elements:         {', '.join(analysis['elements'])}")

        lines.append("\nSTRUCTURE STATISTICS")
        lines.append(f"   Avg nodes:        {analysis['avg_nodes']} atoms/structure")
        lines.append(f"   Avg edges:        {analysis['avg_edges']} edges/structure")

        lines.append("\nRECOMMENDED SETTINGS")
        if analysis["batch_size_valid"]:
            lines.append(f"   batch_size:       {analysis['recommended_batch_size']}")
        else:
            lines.append("   batch_size:       WARNING - MANUAL CALCULATION REQUIRED")
            lines.append("   ! Could not determine avg_edges")
            lines.append("   ! Calculate: batch_size >= ceil(200 / avg_edges)")

        lines.append("\nORBITAL INFO")
        lines.append(f"   Common orbital types: {analysis['common_orbital_types']}")
        lines.append(f"   BS3B orbital types:   {analysis['suggested_bs3b_orbital_types']}")

        if analysis["irreps"]["suggested"]:
            lines.append("\nIRREPS (for model.advanced.net_irreps)")
            lines.append(f"   suggested: {analysis['irreps']['suggested']}")
            lines.append(f"   dimension: {analysis['irreps']['dim']}")

        lines.append("\n" + "=" * 60)

        print("\n".join(lines))


def analyze_graph(
    graph_path: Path,
    output: Path | None = None,
    consider_parity: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Analyze graph and optionally save results.

    Args:
        graph_path: Path to graph file.
        output: Optional output JSON file path.
        consider_parity: Whether to consider parity in irreps calculation.
        quiet: If True, suppress console output.

    Returns:
        Analysis results dictionary.
    """
    analyzer = GraphAnalyzer(graph_path)
    analysis = analyzer.analyze_graph_features(consider_parity=consider_parity)

    if not quiet:
        analyzer.print_summary(analysis)

    if output:
        result = {**analysis, "toml_hints": _generate_toml_hints(analysis)}
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if not quiet:
            print(f"\nJSON output saved to: {output}")

    return analysis


def _generate_toml_hints(data: dict[str, Any]) -> list[dict[str, str]]:
    """Generate TOML configuration hints from analysis data."""
    hints = []

    num_elements = data.get("num_elements", 0)
    model_type = data.get("model_type", "dedicated")
    recommended_batch_size = data.get("recommended_batch_size", 1)
    batch_size_valid = data.get("batch_size_valid", False)
    avg_edges = data.get("avg_edges", "unknown")

    if data["irreps"].get("suggested"):
        hints.append(
            {
                "param": "model.advanced.net_irreps",
                "value": data["irreps"]["suggested"],
                "source": "graph analysis (suggested irreps)",
                "note": "All channel counts should be >= 2",
            }
        )

    if data["common_orbital_types"]:
        hints.append(
            {
                "param": "data.graph.common_orbital_types",
                "value": data["common_orbital_types"],
                "source": "graph analysis (common orbital types)",
                "note": "Must match dataset orbital configuration",
            }
        )

    if batch_size_valid:
        hints.append(
            {
                "param": "process.train.dataloader.batch_size",
                "value": str(recommended_batch_size),
                "source": f"Avg {avg_edges} edges/structure, edges*batch in [200, 5000]",
                "note": "Smaller batch size favors higher accuracy. Increase if GPU memory allows.",
            }
        )
    else:
        hints.append(
            {
                "param": "process.train.dataloader.batch_size",
                "value": "MANUAL_CALCULATION_REQUIRED",
                "source": "WARNING: Could not determine avg_edges",
                "note": "Calculate: batch_size >= ceil(200 / avg_edges). DO NOT use batch_size=1 for small molecules.",
            }
        )

    if model_type == "general":
        if data.get("suggested_bs3b_orbital_types"):
            hints.append(
                {
                    "param": "data.graph.bs3b_orbital_types",
                    "value": data["suggested_bs3b_orbital_types"],
                    "source": "graph analysis (suggested BS3B)",
                    "note": "Required when enable_bs3b_layer = true",
                }
            )
        hints.append(
            {
                "param": "model.advanced.enable_bs3b_layer",
                "value": "true",
                "source": f"General model detected ({num_elements} elements)",
                "note": "BS3B layer recommended for 4+ element datasets",
            }
        )
        hints.append(
            {
                "param": "model.advanced.standardize_gauge",
                "value": "true",
                "source": f"General model detected ({num_elements} elements)",
                "note": "Required for general models with inconsistent chemical potential",
            }
        )
        hints.append(
            {
                "param": "process.train.scheduler.type",
                "value": "warmup_cosine_decay",
                "source": f"General model detected ({num_elements} elements)",
                "note": "Warmup cosine decay recommended for large-scale general model training",
            }
        )
    else:
        hints.append(
            {
                "param": "process.train.scheduler.type",
                "value": "reduce_on_plateau",
                "source": f"Dedicated model detected ({num_elements} elements)",
                "note": "Reduce on plateau recommended for dedicated models (1-3 elements)",
            }
        )
        hints.append(
            {
                "param": "model.advanced.enable_bs3b_layer",
                "value": "false",
                "source": f"Dedicated model detected ({num_elements} elements)",
                "note": "BS3B not needed for dedicated models",
            }
        )

    return hints
