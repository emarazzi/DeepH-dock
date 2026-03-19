"""
Example pipeline: transform Hamiltonian/overlap in k-space, then dump back to R-space.

This script is intentionally minimal and includes a placeholder function where
users can implement their own k-dependent transformation T(k).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import h5py
import numpy as np

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import (
    DEEPX_POSCAR_FILENAME,
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_INFO_FILENAME,
    DEEPX_OVERLAP_FILENAME,
)


L_TO_LABEL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}


def _parse_shell_selector(token: str) -> tuple[int, int]:
    """Parse shell selector strings like '1s', '2p' into (n, l)."""
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z])\s*", token)
    if m is None:
        raise ValueError(f"Invalid shell selector '{token}'. Use forms like '1s', '2p'.")
    n = int(m.group(1))
    orb = m.group(2).lower()
    l_candidates = [ll for ll, label in L_TO_LABEL.items() if label == orb]
    if not l_candidates:
        raise ValueError(f"Unsupported orbital label in selector '{token}'.")
    return n, l_candidates[0]


def _normalize_orbital_labels(labels: list[str]) -> set[int]:
    """Map labels like ['s', 'p'] to a set of angular momentum integers l."""
    out: set[int] = set()
    inv_map = {v: k for k, v in L_TO_LABEL.items()}
    for lb in labels:
        key = lb.strip().lower()
        if key == "":
            continue
        if key not in inv_map:
            raise ValueError(f"Unsupported orbital family '{lb}'. Supported: {sorted(inv_map)}")
        out.add(inv_map[key])
    return out


def resolve_indices_from_orbital_selectors(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    target_elements: list[str],
    target_atom_indices: list[int],
    remove_orbitals: list[str],
    remove_shells: list[str],
) -> tuple[list[int], dict[str, object]]:
    """
    Resolve semantic orbital selectors into global AO indices.

    The selector semantics are:
    - remove_orbitals=['p'] removes all p channels for selected atoms (all m components).
    - remove_shells=['1s', '2p'] removes those shell orders per selected atom.
    """
    n_atoms = len(elements)
    atom_filter = set(range(n_atoms))

    if target_elements:
        target_set = {x.strip() for x in target_elements if x.strip()}
        atom_filter = {ia for ia, el in enumerate(elements) if el in target_set}

    if target_atom_indices:
        atom_index_set = set(int(i) for i in target_atom_indices)
        bad_idx = [i for i in atom_index_set if i < 0 or i >= n_atoms]
        if bad_idx:
            raise ValueError(f"target_atom_indices out of range [0, {n_atoms - 1}]: {sorted(bad_idx)}")
        atom_filter = atom_filter.intersection(atom_index_set)

    rm_l_set = _normalize_orbital_labels(remove_orbitals)
    rm_shell_set = {_parse_shell_selector(tok) for tok in remove_shells}

    if not rm_l_set and not rm_shell_set:
        return [], {
            "selected_atoms": sorted(atom_filter),
            "remove_orbital_families": [],
            "remove_shells": [],
            "resolved_shells": [],
        }

    rm_indices: list[int] = []
    resolved_shells: list[dict[str, object]] = []
    g0 = 0
    for ia, el in enumerate(elements):
        shell_ls = [int(v) for v in elements_orbital_map[el]]
        shell_count_by_l: dict[int, int] = {}

        for shell_pos, l in enumerate(shell_ls):
            shell_n = shell_count_by_l.get(l, 0) + 1
            shell_count_by_l[l] = shell_n
            shell_dim = 2 * l + 1
            shell_start = g0
            shell_stop = g0 + shell_dim

            if ia in atom_filter and (l in rm_l_set or (shell_n, l) in rm_shell_set):
                rm_indices.extend(range(shell_start, shell_stop))
                resolved_shells.append(
                    {
                        "atom_index": ia,
                        "element": el,
                        "shell_position": shell_pos,
                        "shell": f"{shell_n}{L_TO_LABEL.get(l, f'l{l}')}",
                        "global_index_start": shell_start,
                        "global_index_stop": shell_stop,
                    }
                )

            g0 = shell_stop

    meta = {
        "selected_atoms": sorted(atom_filter),
        "remove_orbital_families": [L_TO_LABEL[l] for l in sorted(rm_l_set)],
        "remove_shells": sorted([f"{n}{L_TO_LABEL[l]}" for n, l in rm_shell_set]),
        "resolved_shells": resolved_shells,
    }
    return sorted(set(rm_indices)), meta


def resolve_indices_from_rules(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    rules: list[dict[str, object]],
) -> tuple[list[int], list[dict[str, object]]]:
    """Resolve a list of selector rules into global indices and per-rule metadata."""

    def _get_list_field(rule_obj: dict[str, object], key: str) -> list[object]:
        value = rule_obj.get(key, [])
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"Rule field '{key}' must be a list, got {type(value).__name__}")
        return value

    def _to_int_list(values: list[object], key: str) -> list[int]:
        out: list[int] = []
        for v in values:
            if isinstance(v, bool):
                raise ValueError(f"Rule field '{key}' contains bool value, expected integer index")
            if isinstance(v, (int, np.integer, str)):
                out.append(int(v))
            else:
                raise ValueError(f"Rule field '{key}' contains unsupported value type: {type(v).__name__}")
        return out

    all_indices: list[int] = []
    per_rule_meta: list[dict[str, object]] = []

    for i_rule, rule in enumerate(rules):
        target_elements = [str(x) for x in _get_list_field(rule, "target_elements")]
        target_atom_indices = _to_int_list(_get_list_field(rule, "target_atom_indices"), "target_atom_indices")
        remove_orbitals = [str(x) for x in _get_list_field(rule, "remove_orbitals")]
        remove_shells = [str(x) for x in _get_list_field(rule, "remove_shells")]

        rm_i, meta_i = resolve_indices_from_orbital_selectors(
            elements=elements,
            elements_orbital_map=elements_orbital_map,
            target_elements=target_elements,
            target_atom_indices=target_atom_indices,
            remove_orbitals=remove_orbitals,
            remove_shells=remove_shells,
        )
        all_indices.extend(rm_i)
        per_rule_meta.append(
            {
                "rule_index": i_rule,
                "input_rule": {
                    "target_elements": target_elements,
                    "target_atom_indices": target_atom_indices,
                    "remove_orbitals": remove_orbitals,
                    "remove_shells": remove_shells,
                },
                "resolved": meta_i,
            }
        )

    return sorted(set(all_indices)), per_rule_meta


def build_reduced_elements_orbital_map(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    removed_indices: list[int],
) -> dict[str, list[int]]:
    """
    Build updated elements_orbital_map after shell/channel removal.

    This requires that all atoms of the same element keep the same shell list,
    otherwise info.json cannot represent the reduced basis consistently.
    """
    rm_set = set(int(i) for i in removed_indices)
    per_element_kept_shells: dict[str, list[int]] = {}
    g0 = 0

    for _, el in enumerate(elements):
        shell_ls = [int(v) for v in elements_orbital_map[el]]
        kept_shells_for_atom: list[int] = []
        for l in shell_ls:
            dim = 2 * l + 1
            rng = set(range(g0, g0 + dim))
            kept_count = len(rng - rm_set)
            if kept_count not in (0, dim):
                raise ValueError(
                    f"Partial removal inside shell is not supported for info.json rewrite: "
                    f"element={el}, l={l}, global_range=[{g0}, {g0 + dim})"
                )
            if kept_count == dim:
                kept_shells_for_atom.append(l)
            g0 += dim

        if el in per_element_kept_shells:
            if per_element_kept_shells[el] != kept_shells_for_atom:
                raise ValueError(
                    f"Inconsistent reduced shells among atoms of element {el}. "
                    "Use element-wide consistent selectors for info.json compatibility."
                )
        else:
            per_element_kept_shells[el] = kept_shells_for_atom

    return per_element_kept_shells


def write_reduced_info_json(
    input_dir: Path,
    output_dir: Path,
    elements: list[str],
    removed_indices: list[int],
) -> None:
    """Write updated info.json matching the reduced basis."""
    with open(input_dir / DEEPX_INFO_FILENAME, "r", encoding="utf-8") as fr:
        raw_info = json.load(fr)

    raw_map = raw_info["elements_orbital_map"]
    new_map = build_reduced_elements_orbital_map(elements, raw_map, removed_indices)

    new_orbits = int(sum(np.sum(2 * np.array(new_map[el], dtype=int) + 1) for el in elements))
    raw_info["elements_orbital_map"] = new_map
    raw_info["orbits_quantity"] = new_orbits

    with open(output_dir / DEEPX_INFO_FILENAME, "w", encoding="utf-8") as fw:
        json.dump(raw_info, fw, indent=2)
        fw.write("\n")

def build_orbital_labels(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
) -> list[dict[str, object]]:
    """
    Build a per-global-index label table.

    Returns a list (one entry per AO index) with keys:
        global_idx, atom_idx, element, shell_n, shell_l, shell_label, m
    Example row: {global_idx: 5, atom_idx: 0, element: 'Mo', shell_n: 2,
                  shell_l: 1, shell_label: '2p', m: 0}
    """
    labels: list[dict[str, object]] = []
    g = 0
    for ia, el in enumerate(elements):
        shell_ls = [int(v) for v in elements_orbital_map[el]]
        shell_count_by_l: dict[int, int] = {}
        for l in shell_ls:
            shell_n = shell_count_by_l.get(l, 0) + 1
            shell_count_by_l[l] = shell_n
            shell_label = f"{shell_n}{L_TO_LABEL.get(l, f'l{l}')}"
            for m in range(-l, l + 1):
                labels.append({
                    "global_idx": g,
                    "atom_idx": ia,
                    "element": el,
                    "shell_n": shell_n,
                    "shell_l": l,
                    "shell_label": shell_label,
                    "m": m,
                })
                g += 1
    return labels


def build_shell_groups(orbital_labels: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group orbital labels by atom and shell label."""
    groups: dict[tuple[int, str, str], list[int]] = {}
    for lbl in orbital_labels:
        key = (int(str(lbl["atom_idx"])), str(lbl["element"]), str(lbl["shell_label"]))
        groups.setdefault(key, []).append(int(str(lbl["global_idx"])))

    out: list[dict[str, object]] = []
    for (atom_idx, element, shell_label), indices in groups.items():
        out.append(
            {
                "key": (atom_idx, element, shell_label),
                "atom_idx": atom_idx,
                "element": element,
                "shell_label": shell_label,
                "indices": sorted(indices),
            }
        )
    out.sort(key=lambda g: (int(str(g["atom_idx"])), str(g["element"]), str(g["shell_label"])))
    return out


def seed_groups_from_indices(seed_indices: list[int], shell_groups: list[dict[str, object]]) -> list[tuple[int, str, str]]:
    """Expand seed orbital indices to full shell groups containing them."""
    seed_set = set(int(i) for i in seed_indices)
    out: list[tuple[int, str, str]] = []
    for group in shell_groups:
        indices = group["indices"]
        if isinstance(indices, list) and seed_set.intersection(int(i) for i in indices):
            key = group["key"]
            if isinstance(key, tuple):
                out.append((int(key[0]), str(key[1]), str(key[2])))
    return sorted(set(out))


def evaluate_removal_set(
    Hk: np.ndarray,
    Sk: np.ndarray,
    shell_groups: list[dict[str, object]],
    removed_keys: list[tuple[int, str, str]],
) -> dict[str, object]:
    """Evaluate Schur perturbation for a combined removal set made of shell groups."""
    removed_set = set(removed_keys)
    M_idx: list[int] = []
    K_idx: list[int] = []
    for group in shell_groups:
        indices = group["indices"]
        if not isinstance(indices, list):
            continue
        key = group["key"]
        key_t = (int(key[0]), str(key[1]), str(key[2])) if isinstance(key, tuple) else None
        if key_t in removed_set:
            M_idx.extend(int(i) for i in indices)
        else:
            K_idx.extend(int(i) for i in indices)

    if not M_idx:
        return {
            "removed_indices": [],
            "kept_indices": K_idx,
            "schur_s_max": 0.0,
            "schur_h_max": 0.0,
            "safe": True,
        }
    if not K_idx:
        return {
            "removed_indices": M_idx,
            "kept_indices": [],
            "schur_s_max": float("inf"),
            "schur_h_max": float("inf"),
            "safe": False,
        }

    M = np.array(sorted(M_idx), dtype=int)
    K = np.array(sorted(K_idx), dtype=int)

    schur_s_vals: list[float] = []
    schur_h_vals: list[float] = []
    for ik in range(len(Sk)):
        S_MM = Sk[ik][np.ix_(M, M)]
        S_MK = Sk[ik][np.ix_(M, K)]
        S_KK = Sk[ik][np.ix_(K, K)]
        H_MK = Hk[ik][np.ix_(M, K)]
        H_KK = Hk[ik][np.ix_(K, K)]
        try:
            SMMi_SMK = np.linalg.solve(S_MM, S_MK)
        except np.linalg.LinAlgError:
            schur_s_vals.append(float("inf"))
            schur_h_vals.append(float("inf"))
            continue

        schur_s = S_MK.conj().T @ SMMi_SMK
        denom_s = float(np.linalg.norm(S_KK)) or 1.0
        schur_s_vals.append(float(np.linalg.norm(schur_s)) / denom_s)

        H_KM = H_MK.conj().T
        schur_h = H_KM @ SMMi_SMK
        schur_h = schur_h + schur_h.conj().T
        denom_h = float(np.linalg.norm(H_KK)) or 1.0
        schur_h_vals.append(float(np.linalg.norm(schur_h)) / denom_h)

    return {
        "removed_indices": sorted(M_idx),
        "kept_indices": sorted(K_idx),
        "schur_s_max": float(np.max(schur_s_vals)),
        "schur_h_max": float(np.max(schur_h_vals)),
        "safe": False,
    }


def rank_group_coupling_to_removed_set(
    Hk: np.ndarray,
    Sk: np.ndarray,
    shell_groups: list[dict[str, object]],
    removed_keys: list[tuple[int, str, str]],
) -> list[dict[str, object]]:
    """Rank remaining shell groups by coupling strength to the current removed set."""
    removed_set = set(removed_keys)
    removed_idx: list[int] = []
    for group in shell_groups:
        key = group["key"]
        key_t = (int(key[0]), str(key[1]), str(key[2])) if isinstance(key, tuple) else None
        if key_t in removed_set:
            indices = group["indices"]
            if isinstance(indices, list):
                removed_idx.extend(int(i) for i in indices)

    if not removed_idx:
        return []

    M = np.array(sorted(removed_idx), dtype=int)
    ranked: list[dict[str, object]] = []
    for group in shell_groups:
        key = group["key"]
        key_t = (int(key[0]), str(key[1]), str(key[2])) if isinstance(key, tuple) else None
        if key_t in removed_set:
            continue
        indices = group["indices"]
        if not isinstance(indices, list):
            continue
        G = np.array(sorted(int(i) for i in indices), dtype=int)
        scores_s: list[float] = []
        scores_h: list[float] = []
        for ik in range(len(Sk)):
            S_GM = Sk[ik][np.ix_(G, M)]
            S_GG = Sk[ik][np.ix_(G, G)]
            S_MM = Sk[ik][np.ix_(M, M)]
            H_GM = Hk[ik][np.ix_(G, M)]
            H_GG = Hk[ik][np.ix_(G, G)]
            H_MM = Hk[ik][np.ix_(M, M)]

            denom_s = (float(np.linalg.norm(S_GG)) * float(np.linalg.norm(S_MM))) ** 0.5 or 1.0
            denom_h = (float(np.linalg.norm(H_GG)) * float(np.linalg.norm(H_MM))) ** 0.5 or 1.0
            scores_s.append(float(np.linalg.norm(S_GM)) / denom_s)
            scores_h.append(float(np.linalg.norm(H_GM)) / denom_h)

        ranked.append(
            {
                "key": key_t,
                "atom_idx": int(str(group["atom_idx"])),
                "element": str(group["element"]),
                "shell_label": str(group["shell_label"]),
                "indices": list(indices),
                "cross_s_max": float(np.max(scores_s)),
                "cross_h_max": float(np.max(scores_h)),
                "score": float(np.max(scores_s)) + 0.3 * float(np.max(scores_h)),
            }
        )

    ranked.sort(key=lambda row: float(str(row["score"])), reverse=True)
    return ranked


def suggest_companion_removals(
    Hk: np.ndarray,
    Sk: np.ndarray,
    shell_groups: list[dict[str, object]],
    seed_indices: list[int],
    safe_threshold: float,
    max_steps: int,
) -> dict[str, object]:
    """Greedily grow a shell-removal set until the combined set becomes safe or max_steps is reached."""
    seed_keys = seed_groups_from_indices(seed_indices, shell_groups)
    if not seed_keys:
        raise ValueError("No shell groups found for the requested seed indices")

    removed_keys = list(seed_keys)
    history: list[dict[str, object]] = []
    final_ranked: list[dict[str, object]] = []

    for step in range(max_steps + 1):
        metrics = evaluate_removal_set(Hk, Sk, shell_groups, removed_keys)
        schur_s_max = float(str(metrics["schur_s_max"]))
        schur_h_max = float(str(metrics["schur_h_max"]))
        metrics["safe"] = schur_s_max < safe_threshold
        history.append(
            {
                "step": step,
                "removed_keys": list(removed_keys),
                "schur_s_max": schur_s_max,
                "schur_h_max": schur_h_max,
                "safe": bool(metrics["safe"]),
            }
        )
        if metrics["safe"]:
            return {
                "seed_keys": seed_keys,
                "removed_keys": removed_keys,
                "history": history,
                "terminated": "safe",
                "ranked_candidates": final_ranked,
            }

        if step == max_steps:
            break

        ranked = rank_group_coupling_to_removed_set(Hk, Sk, shell_groups, removed_keys)
        final_ranked = ranked
        if not ranked:
            break
        best = ranked[0]
        best_key = best["key"]
        if isinstance(best_key, tuple):
            removed_keys.append((int(best_key[0]), str(best_key[1]), str(best_key[2])))
        else:
            break

    return {
        "seed_keys": seed_keys,
        "removed_keys": removed_keys,
        "history": history,
        "terminated": "max_steps",
        "ranked_candidates": final_ranked,
    }


def print_companion_removal_suggestion(report: dict[str, object]) -> None:
    """Pretty-print the greedy companion-removal closure suggestion."""
    seed_keys = report.get("seed_keys", [])
    removed_keys = report.get("removed_keys", [])
    history = report.get("history", [])
    terminated = str(report.get("terminated", "unknown"))
    ranked = report.get("ranked_candidates", [])

    print("\n" + "=" * 80)
    print("Companion Removal Suggestion")
    print(f"  Seed shell groups: {seed_keys}")
    print("  Greedy closure history:")
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict):
                continue
            print(
                f"    step={row.get('step')}  schur_s_max={float(str(row.get('schur_s_max', 0.0))):.4e}  "
                f"schur_h_max={float(str(row.get('schur_h_max', 0.0))):.4e}  safe={bool(row.get('safe', False))}"
            )

    print(f"  Termination: {terminated}")
    print(f"  Suggested shell-removal set: {removed_keys}")
    if isinstance(ranked, list) and ranked:
        print("  Top coupled remaining shells at stop point:")
        for row in ranked[:5]:
            if not isinstance(row, dict):
                continue
            print(
                f"    atom={row['atom_idx']} elem={row['element']} shell={row['shell_label']} "
                f"cross_s_max={float(str(row['cross_s_max'])):.4e} cross_h_max={float(str(row['cross_h_max'])):.4e}"
            )
    print("=" * 80 + "\n")


def analyze_orbital_coupling(
    Hk: np.ndarray,
    Sk: np.ndarray,
    orbital_labels: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    For each shell group (atom × shell_label), compute how strongly it couples
    to the rest of the basis via the Schur complement.

    Key metric:
        schur_s  =  max_k  ||S_KM(k) S_MM(k)^{-1} S_MK(k)||_F / ||S_KK(k)||_F
        schur_h  =  max_k  ||H_KM(k) S_MM(k)^{-1} S_MK(k) + h.c.||_F / ||H_KK(k)||_F

    These measure how much the remaining S(k) and H(k) blocks change when shell M
    is eliminated via the T-matrix projection.  Values < 1% mean the remaining
    bands are barely affected — the shell is safe to remove.

    Results are returned sorted safest-first (lowest schur_s_max).
    """
    nb = Sk.shape[-1]

    # Group local indices by (atom_idx, shell_label)
    groups: dict[tuple[int, str], list[int]] = {}
    for lbl in orbital_labels:
        key = (int(str(lbl["atom_idx"])), str(lbl["shell_label"]))
        groups.setdefault(key, []).append(int(str(lbl["global_idx"])))

    results: list[dict[str, object]] = []

    for (atom_idx, shell_label), M_local in groups.items():
        K_local = [i for i in range(nb) if i not in set(M_local)]
        M = np.array(M_local)
        K = np.array(K_local)
        if len(K) == 0:
            continue

        S_MM = Sk[:, M, :][:, :, M]
        S_MK = Sk[:, M, :][:, :, K]
        S_KK = Sk[:, K, :][:, :, K]
        H_MK = Hk[:, M, :][:, :, K]
        H_KK = Hk[:, K, :][:, :, K]

        schur_s_vals: list[float] = []
        schur_h_vals: list[float] = []
        cross_s_vals: list[float] = []
        cross_h_vals: list[float] = []

        for ik in range(len(Sk)):
            try:
                SMMi_SMK = np.linalg.solve(S_MM[ik], S_MK[ik])
            except np.linalg.LinAlgError:
                schur_s_vals.append(np.inf)
                schur_h_vals.append(np.inf)
                cross_s_vals.append(float(np.linalg.norm(S_MK[ik])))
                cross_h_vals.append(float(np.linalg.norm(H_MK[ik])))
                continue

            schur_s = S_MK[ik].conj().T @ SMMi_SMK
            denom_s = float(np.linalg.norm(S_KK[ik])) or 1.0
            schur_s_vals.append(float(np.linalg.norm(schur_s)) / denom_s)

            H_KM = H_MK[ik].conj().T
            schur_h = H_KM @ SMMi_SMK
            schur_h = schur_h + schur_h.conj().T
            denom_h = float(np.linalg.norm(H_KK[ik])) or 1.0
            schur_h_vals.append(float(np.linalg.norm(schur_h)) / denom_h)

            cross_s_vals.append(float(np.linalg.norm(S_MK[ik])))
            cross_h_vals.append(float(np.linalg.norm(H_MK[ik])))

        elem = str(orbital_labels[M_local[0]]["element"])
        results.append({
            "atom_idx": atom_idx,
            "element": elem,
            "shell_label": shell_label,
            "global_indices": sorted(M_local),
            "schur_s_max": float(np.max(schur_s_vals)),
            "schur_s_mean": float(np.mean(schur_s_vals)),
            "schur_h_max": float(np.max(schur_h_vals)),
            "schur_h_mean": float(np.mean(schur_h_vals)),
            "cross_s_max": float(np.max(cross_s_vals)),
            "cross_h_max": float(np.max(cross_h_vals)),
        })

    results.sort(key=lambda x: float(str(x["schur_s_max"])))
    return results


def print_coupling_report(
    results: list[dict[str, object]],
    elements: list[str],
    label: str = "",
    safe_threshold: float = 0.01,
    marginal_threshold: float = 0.05,
) -> None:
    """
    Pretty-print per-shell Schur coupling table, sorted safest-first.

    Safety flags (based on schur_s_max):
      ✓ SAFE     < safe_threshold     (default 1%)  — remove freely
      ~ MARG     < marginal_threshold (default 5%)  — check Schur-H too
      ✗ UNSAFE   otherwise            — removal will corrupt remaining bands
    """
    tag = f" ({label})" if label else ""
    W = 80
    print(f"\n{'='*W}")
    print(f"Orbital coupling / removal safety analysis{tag}")
    print(f"  Schur-S: relative perturbation to S(k) of remaining basis after shell removal.")
    print(f"  Schur-H: relative perturbation to H(k) of remaining basis (first order).")
    print(f"  SAFE = Schur-S < {safe_threshold*100:.1f}%  |  MARGINAL < {marginal_threshold*100:.1f}%  |  UNSAFE otherwise")
    print()
    hdr = f"  {'atom':>4}  {'elem':>5}  {'shell':>6}  {'indices':>20}  {'Schur-S(max)':>14}  {'Schur-H(max)':>14}  status"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # Track per-element+shell safety across all atoms
    elem_shell_safety: dict[tuple[str, str], list[bool]] = {}

    for r in results:
        ss = float(str(r["schur_s_max"]))
        sh = float(str(r["schur_h_max"]))
        if ss < safe_threshold:
            flag = "✓ SAFE"
        elif ss < marginal_threshold:
            flag = "~ MARG"
        else:
            flag = "✗ UNSAFE"
        idxs_str = str(r["global_indices"])
        if len(idxs_str) > 20:
            idxs_str = idxs_str[:17] + "..."
        print(
            f"  {r['atom_idx']:>4}  {r['element']:>5}  {r['shell_label']:>6}  "
            f"{idxs_str:>20}  {ss:>14.4e}  {sh:>14.4e}  {flag}"
        )
        key = (str(r["element"]), str(r["shell_label"]))
        elem_shell_safety.setdefault(key, []).append(ss < safe_threshold)

    # Only suggest shells that are safe on ALL atoms of the same element
    # (required for consistent info.json update)
    print()
    print("  Suggested removal commands (safe on ALL atoms of element — info.json compatible):")
    found_any = False
    seen: set[tuple[str, str]] = set()
    for r in results:
        key = (str(r["element"]), str(r["shell_label"]))
        if key in seen:
            continue
        seen.add(key)
        if all(elem_shell_safety.get(key, [False])):
            found_any = True
            elem = str(r["element"])
            shell = str(r["shell_label"])
            all_idxs: list[int] = []
            for r2 in results:
                if str(r2["element"]) == elem and str(r2["shell_label"]) == shell:
                    gi = r2["global_indices"]
                    if isinstance(gi, list):
                        all_idxs.extend(int(i) for i in gi)
            all_idxs = sorted(all_idxs)
            print(f"    {elem} {shell}: --remove-index {' '.join(str(i) for i in all_idxs)}")
            print(f"         or: --target-element {elem} --remove-shell {shell}")
    if not found_any:
        print("    None found with uniform element-wide safety.")
        print("    Use --remove-index with per-atom indices for asymmetric cases,")
        print("    but note info.json will NOT be auto-updated in that case.")
    print(f"{'='*W}\n")


def build_uniform_kmesh(nk: tuple[int, int, int]) -> np.ndarray:
    """Build a uniform fractional k-mesh in [0, 1)."""
    nx, ny, nz = nk
    xs = np.arange(nx, dtype=float) / nx
    ys = np.arange(ny, dtype=float) / ny
    zs = np.arange(nz, dtype=float) / nz
    kx, ky, kz = np.meshgrid(xs, ys, zs, indexing="ij")
    ks = np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
    return ks


def apply_custom_kspace_transform(
    Hk: np.ndarray,
    Sk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for your custom transformation in k-space.

    Parameters
    ----------
    Hk, Sk : np.ndarray, shape (Nk, Nb, Nb)
        Original Hamiltonian and overlap matrices in k-space.
    remove_indices : list[int]
        Global orbital indices to remove from the basis.

    Returns
    -------
    Hk_new, Sk_new : np.ndarray, shape (Nk, Nb, Nb)
        Transformed k-space matrices.

    Notes
    -----
    Implement your T-matrix logic here, for example:
    - Hk_new[k] = T(k)^dagger @ Hk[k] @ T(k)
    - Sk_new[k] = T(k)^dagger @ Sk[k] @ T(k)
    """
    if Hk.shape != Sk.shape:
        raise ValueError(f"Hk/Sk shape mismatch: {Hk.shape} vs {Sk.shape}")
    if Hk.ndim != 3:
        raise ValueError(f"Hk/Sk must have shape (Nk, Nb, Nb), got {Hk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        return Hk, Sk

    Tk, _, _ = build_elimination_tk(Sk, rm)
    Hk_new, Sk_new = apply_tk_projection(Hk, Sk, Tk)
    return Hk_new, Sk_new


def build_elimination_tk(
    Sk: np.ndarray,
    remove_indices: list[int],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Build elimination transform T(k) for removing a set of orbitals."""
    if Sk.ndim != 3:
        raise ValueError(f"Sk must have shape (Nk, Nb, Nb), got {Sk.shape}")

    nk, nb, nb2 = Sk.shape
    if nb != nb2:
        raise ValueError(f"Sk must be square in last two dims, got {Sk.shape}")

    rm = sorted(set(int(i) for i in remove_indices))
    if len(rm) == 0:
        eye = np.eye(nb, dtype=np.complex128)
        return np.broadcast_to(eye, (nk, nb, nb)).copy(), list(range(nb)), []
    if rm[0] < 0 or rm[-1] >= nb:
        raise ValueError(f"remove_indices out of range for Nb={nb}: {rm}")

    keep = [i for i in range(nb) if i not in rm]
    nkp = len(keep)
    if nkp == 0:
        raise ValueError("Cannot remove all orbitals")

    # Generalized elimination transform for a removed set M:
    # T[M, :] = -S_MM^{-1} S_MK, T[K, :] = I
    S_mm = Sk[:, rm, :][:, :, rm]
    S_mk = Sk[:, rm, :][:, :, keep]
    coeff = -np.linalg.solve(S_mm, S_mk)

    Tk = np.zeros((nk, nb, nkp), dtype=np.complex128)
    Tk[:, keep, :] = np.eye(nkp, dtype=np.complex128)[None, :, :]
    Tk[:, rm, :] = coeff
    return Tk, keep, rm


def apply_tk_projection(
    Hk: np.ndarray,
    Sk: np.ndarray,
    Tk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply H'(k)=T(k)^dagger H(k) T(k), S'(k)=T(k)^dagger S(k) T(k)."""
    Tc = np.conjugate(np.swapaxes(Tk, 1, 2))
    Hk_new = np.matmul(np.matmul(Tc, Hk), Tk)
    Sk_new = np.matmul(np.matmul(Tc, Sk), Tk)
    return Hk_new, Sk_new


def k_to_r_operator(
    ks: np.ndarray,
    Rijk_list: np.ndarray,
    Mk: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse transform operator blocks from k-space to real-space on a target R list."""
    ks = np.asarray(ks, dtype=float)
    Rs = np.asarray(Rijk_list, dtype=float)
    Mk = np.asarray(Mk)

    if ks.ndim != 2 or ks.shape[1] != 3:
        raise ValueError(f"ks must have shape (Nk, 3), got {ks.shape}")
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (NR, 3), got {Rs.shape}")
    if Mk.ndim != 3:
        raise ValueError(f"Mk must have shape (Nk, Nrow, Ncol), got {Mk.shape}")
    if Mk.shape[0] != ks.shape[0]:
        raise ValueError(f"Nk mismatch between ks and Mk: {ks.shape[0]} vs {Mk.shape[0]}")

    nk = ks.shape[0]
    if weights is None:
        w = np.full(nk, 1.0 / nk, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != nk:
            raise ValueError(f"weights must have shape (Nk,), got {w.shape}")

    phase = np.exp(-2j * np.pi * np.matmul(Rs, ks.T))
    wr = phase * w[None, :]

    Mk_flat = Mk.reshape(nk, -1)
    MR_flat = np.matmul(wr, Mk_flat)
    return MR_flat.reshape(len(Rs), Mk.shape[1], Mk.shape[2])


def project_real_space_via_convolution(
    mats_R: np.ndarray,
    T_R: np.ndarray,
    Rijk_list: np.ndarray,
) -> np.ndarray:
    """Project real-space blocks via convolution with T(R)."""
    mats_R = np.asarray(mats_R)
    T_R = np.asarray(T_R)
    Rijk_list = np.asarray(Rijk_list)

    if mats_R.ndim != 3:
        raise ValueError(f"mats_R must have shape (NR, Nb, Nb), got {mats_R.shape}")
    if T_R.ndim != 3:
        raise ValueError(f"T_R must have shape (NR, Nb, Nkp), got {T_R.shape}")
    if mats_R.shape[0] != T_R.shape[0]:
        raise ValueError(f"NR mismatch between mats_R and T_R: {mats_R.shape[0]} vs {T_R.shape[0]}")
    if mats_R.shape[1] != T_R.shape[1]:
        raise ValueError(f"Nb mismatch between mats_R and T_R: {mats_R.shape[1]} vs {T_R.shape[1]}")
    if Rijk_list.shape[0] != mats_R.shape[0]:
        raise ValueError(f"Rijk_list size mismatch: {Rijk_list.shape[0]} vs {mats_R.shape[0]}")

    nr, _, nkp = T_R.shape
    out = np.zeros((nr, nkp, nkp), dtype=np.complex128)

    r_keys = [tuple(int(v) for v in row) for row in Rijk_list]
    r_to_idx = {r: i for i, r in enumerate(r_keys)}

    for i_rp, rp in enumerate(r_keys):
        acc = np.zeros((nkp, nkp), dtype=np.complex128)
        for i_r1, r1 in enumerate(r_keys):
            Tl = np.conjugate(T_R[i_r1].T)
            r_mid_offset = (rp[0] + r1[0], rp[1] + r1[1], rp[2] + r1[2])
            for i_r2, r2 in enumerate(r_keys):
                r_mid = (
                    r_mid_offset[0] - r2[0],
                    r_mid_offset[1] - r2[1],
                    r_mid_offset[2] - r2[2],
                )
                i_mid = r_to_idx.get(r_mid)
                if i_mid is None:
                    continue
                acc += Tl @ mats_R[i_mid] @ T_R[i_r2]
        out[i_rp] = acc

    return out


def apply_custom_realspace_convolution_projection(
    ks: np.ndarray,
    Rijk_list: np.ndarray,
    HR: np.ndarray,
    SR: np.ndarray,
    Sk: np.ndarray,
    remove_indices: list[int],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Build T(k), transform to T(R), then project H(R)/S(R) via real-space convolution."""
    Tk, keep, _ = build_elimination_tk(Sk, remove_indices)
    TR = k_to_r_operator(ks=ks, Rijk_list=Rijk_list, Mk=Tk, weights=weights)

    HR_proj = project_real_space_via_convolution(HR, TR, Rijk_list)
    SR_proj = project_real_space_via_convolution(SR, TR, Rijk_list)
    return HR_proj, SR_proj, Tk, TR, keep


def save_t_matrices(
    output_dir: Path,
    ks: np.ndarray,
    Rijk_list: np.ndarray,
    Tk: np.ndarray,
    TR: np.ndarray,
) -> None:
    """Save T matrices in reciprocal and real space for comparison."""
    np.savez_compressed(
        output_dir / "T_matrices_kspace.npz",
        ks=np.asarray(ks, dtype=float),
        T_k=np.asarray(Tk),
    )
    np.savez_compressed(
        output_dir / "T_matrices_rspace.npz",
        Rijk_list=np.asarray(Rijk_list, dtype=int),
        T_R=np.asarray(TR),
    )


def dump_reduced_matrix_h5(
    out_path: Path,
    mats_R: np.ndarray,
    Rijk_list: np.ndarray | None,
    atom_pairs: np.ndarray,
    atom_num_orbits_cumsum: np.ndarray,
    keep_global: list[int],
) -> None:
    """
    Dump reduced-basis real-space matrices to DeepH-like h5.

    Unlike the default serializer, this writer recomputes chunk shapes using
    the kept orbital indices per atom, so reduced basis dimensions are encoded
    correctly for each atom pair block.
    """
    if Rijk_list is None:
        raise ValueError("Rijk_list is None")

    keep_arr = np.array(sorted(set(int(i) for i in keep_global)), dtype=int)

    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    atom_pairs = np.asarray(atom_pairs, dtype=np.int64)

    # For each atom, store indices in the reduced basis.
    per_atom_reduced_indices = []
    n_atoms = len(atom_num_orbits_cumsum) - 1
    for ia in range(n_atoms):
        a0 = int(atom_num_orbits_cumsum[ia])
        a1 = int(atom_num_orbits_cumsum[ia + 1])
        idx = np.where((keep_arr >= a0) & (keep_arr < a1))[0]
        per_atom_reduced_indices.append(idx)

    entries_chunks = []
    chunk_shapes = np.zeros((len(atom_pairs), 2), dtype=np.int64)
    chunk_boundaries = np.zeros(len(atom_pairs) + 1, dtype=np.int64)

    for i_ap, ap in enumerate(atom_pairs):
        Rijk = (int(ap[0]), int(ap[1]), int(ap[2]))
        ia = int(ap[3])
        ja = int(ap[4])

        mat_R = mats_R[r_to_idx[Rijk]]
        ii = per_atom_reduced_indices[ia]
        jj = per_atom_reduced_indices[ja]
        block = mat_R[np.ix_(ii, jj)]

        chunk_shapes[i_ap] = np.array(block.shape, dtype=np.int64)
        chunk_boundaries[i_ap + 1] = chunk_boundaries[i_ap] + block.size
        entries_chunks.append(block.reshape(-1))

    if entries_chunks:
        entries = np.concatenate(entries_chunks)
    else:
        entries = np.array([], dtype=mats_R.dtype)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("chunk_shapes", data=chunk_shapes)
        f.create_dataset("chunk_boundaries", data=chunk_boundaries)
        f.create_dataset("entries", data=entries)


def hermitize_real_space_blocks(mats_R: np.ndarray, Rijk_list: np.ndarray) -> np.ndarray:
    """
    Enforce Hermiticity in real space by symmetrizing R and -R pairs.

    For each displacement R, this applies:
        M(R) <- 0.5 * (M(R) + M(-R)^dagger)
        M(-R) <- M(R)^dagger
    """
    mats_out = np.array(mats_R, copy=True)
    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    visited: set[tuple[int, ...]] = set()

    for r, i_r in r_to_idx.items():
        if r in visited:
            continue

        r_neg = (-r[0], -r[1], -r[2])
        i_neg = r_to_idx.get(r_neg)

        A = mats_out[i_r]
        if i_neg is None or i_neg == i_r:
            mats_out[i_r] = 0.5 * (A + np.conjugate(A.T))
            visited.add(r)
            continue

        B = mats_out[i_neg]
        A_sym = 0.5 * (A + np.conjugate(B.T))
        mats_out[i_r] = A_sym
        mats_out[i_neg] = np.conjugate(A_sym.T)

        visited.add(r)
        visited.add(r_neg)

    return mats_out


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    nk: tuple[int, int, int],
    remove_indices: list[int],
    remove_first_orbital_each_atom: bool,
    remove_orbitals: list[str],
    remove_shells: list[str],
    target_elements: list[str],
    target_atom_indices: list[int],
    removal_plan_json: Path | None,
    analyze_coupling: bool = False,
    suggest_companion_removals_flag: bool = False,
    safe_threshold: float = 0.01,
    marginal_threshold: float = 0.05,
    suggest_max_steps: int = 8,
    project_via_rspace_convolution: bool = False,
) -> None:
    """Load H/S, apply k-space transform, map back to R-space, and dump files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep geometry metadata file.
    shutil.copy2(input_dir / DEEPX_POSCAR_FILENAME, output_dir / DEEPX_POSCAR_FILENAME)
    obj = HamiltonianObj(input_dir)

    rm = list(remove_indices)

    selector_indices, selector_meta = resolve_indices_from_orbital_selectors(
        elements=[str(el) for el in obj.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
        target_elements=target_elements,
        target_atom_indices=target_atom_indices,
        remove_orbitals=remove_orbitals,
        remove_shells=remove_shells,
    )
    rm.extend(selector_indices)

    plan_meta = []
    if removal_plan_json is not None:
        with open(removal_plan_json, "r", encoding="utf-8") as fr:
            payload = json.load(fr)
        rules = payload.get("rules", payload)
        if not isinstance(rules, list):
            raise ValueError("removal plan must be a list or an object with key 'rules' (list)")
        plan_indices, plan_meta = resolve_indices_from_rules(
            elements=[str(el) for el in obj.elements],
            elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
            rules=rules,
        )
        rm.extend(plan_indices)

    if remove_first_orbital_each_atom:
        # First AO index for each atom in the global AO basis.
        atom_first = [int(v) for v in obj.atom_num_orbits_cumsum[:-1]]
        rm.extend(atom_first)

    rm = sorted(set(rm))
    print(f"Removing global orbital indices: {rm}")

    ks = build_uniform_kmesh(nk)
    Sk, Hk = obj.Sk_and_Hk(ks)
    nb = Sk.shape[-1]
    keep_global = [i for i in range(nb) if i not in rm]

    orbital_labels = build_orbital_labels(
        elements=[str(el) for el in obj.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
    )
    shell_groups = build_shell_groups(orbital_labels)

    if analyze_coupling:
        print("\n-- Orbital coupling analysis: original basis --")
        coupling_orig = analyze_orbital_coupling(Hk, Sk, orbital_labels)
        print_coupling_report(
            coupling_orig,
            elements=[str(el) for el in obj.elements],
            label="original basis",
            safe_threshold=safe_threshold,
            marginal_threshold=marginal_threshold,
        )
        if len(rm) > 0:
            print("-- Orbital coupling analysis: after your requested removals --")
            _Hk_tmp, _Sk_tmp = apply_custom_kspace_transform(Hk, Sk, remove_indices=rm)
            keep_labels = [orbital_labels[i] for i in keep_global]
            coupling_post = analyze_orbital_coupling(_Hk_tmp, _Sk_tmp, keep_labels)
            print_coupling_report(
                coupling_post,
                elements=[str(el) for el in obj.elements],
                label="after user removals",
                safe_threshold=safe_threshold,
                marginal_threshold=marginal_threshold,
            )

    if suggest_companion_removals_flag:
        if len(rm) == 0:
            raise ValueError(
                "Companion-removal suggestion requires a seed orbital/shell. "
                "Use --removal-plan-json or selector flags like --target-element/--remove-shell."
            )
        suggestion = suggest_companion_removals(
            Hk=Hk,
            Sk=Sk,
            shell_groups=shell_groups,
            seed_indices=rm,
            safe_threshold=safe_threshold,
            max_steps=suggest_max_steps,
        )
        print_companion_removal_suggestion(suggestion)
        return

    if project_via_rspace_convolution:
        if obj.Rijk_list is None:
            raise ValueError("Rijk_list is None")
        HR0 = obj.HR
        SR0 = obj.SR
        if HR0 is None or SR0 is None:
            raise ValueError("HR/SR is None")

        Hk_new, Sk_new = apply_custom_kspace_transform(Hk, Sk, remove_indices=rm)
        HR_ref, SR_ref = obj.Hk_and_Sk_to_real(ks=ks, Hk=Hk_new, Sk=Sk_new)

        HR_new, SR_new, Tk, TR, _ = apply_custom_realspace_convolution_projection(
            ks=ks,
            Rijk_list=obj.Rijk_list,
            HR=HR0,
            SR=SR0,
            Sk=Sk,
            remove_indices=rm,
            weights=None,
        )
        save_t_matrices(output_dir, ks, obj.Rijk_list, Tk, TR)

        hr_diff = float(np.max(np.abs(HR_new - HR_ref))) if HR_new.size > 0 else 0.0
        sr_diff = float(np.max(np.abs(SR_new - SR_ref))) if SR_new.size > 0 else 0.0
        print(
            "\n-- Real-space convolution projection check against k-space route --"
        )
        print(f"   max|HR_conv - HR_kft| = {hr_diff:.4e}")
        print(f"   max|SR_conv - SR_kft| = {sr_diff:.4e}")
        print("   Saved T(k) and T(R) as T_matrices_kspace.npz / T_matrices_rspace.npz")
    else:
        Hk_new, Sk_new = apply_custom_kspace_transform(Hk, Sk, remove_indices=rm)
        HR_new, SR_new = obj.Hk_and_Sk_to_real(ks=ks, Hk=Hk_new, Sk=Sk_new)

    if obj.Rijk_list is None:
        raise ValueError("Rijk_list is None")
    HR_new = hermitize_real_space_blocks(HR_new, obj.Rijk_list)
    SR_new = hermitize_real_space_blocks(SR_new, obj.Rijk_list)

    overlap_imag_max = float(np.max(np.abs(np.imag(SR_new)))) if SR_new.size > 0 else 0.0
    if overlap_imag_max > 0.0:
        print(f"\n-- Forcing overlap to real after Hermitization (max |Im S(R)| = {overlap_imag_max:.3e}) --")
    SR_new = np.asarray(np.real(SR_new), dtype=np.float64)

    dump_reduced_matrix_h5(
        output_dir / DEEPX_HAMILTONIAN_FILENAME,
        HR_new,
        obj.Rijk_list,
        obj.atom_pairs,
        obj.atom_num_orbits_cumsum,
        keep_global,
    )
    dump_reduced_matrix_h5(
        output_dir / DEEPX_OVERLAP_FILENAME,
        SR_new,
        obj.Rijk_list,
        obj.atom_pairs,
        obj.atom_num_orbits_cumsum,
        keep_global,
    )

    reduced_orbital_counts = []
    csum = obj.atom_num_orbits_cumsum
    for ia in range(len(csum) - 1):
        a0 = int(csum[ia])
        a1 = int(csum[ia + 1])
        reduced_orbital_counts.append(int(np.sum((np.array(keep_global) >= a0) & (np.array(keep_global) < a1))))

    meta = {
        "removed_global_indices": rm,
        "kept_global_indices": keep_global,
        "original_orbits_quantity": int(nb),
        "reduced_orbits_quantity": int(len(keep_global)),
        "reduced_orbitals_per_atom": reduced_orbital_counts,
        "selector_resolution": selector_meta,
        "rule_plan_resolution": plan_meta,
        "overlap_imag_max_before_real_cast": overlap_imag_max,
    }

    write_reduced_info_json(
        input_dir=input_dir,
        output_dir=output_dir,
        elements=[str(el) for el in obj.elements],
        removed_indices=rm,
    )

    with open(output_dir / "reduced_basis_meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-space transform -> real-space dump pipeline")
    parser.add_argument("input_dir", type=Path, help="Directory containing POSCAR/info.json/hamiltonian.h5/overlap.h5")
    parser.add_argument("output_dir", type=Path, help="Directory to write transformed files")
    parser.add_argument(
        "--kgrid",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("NKX", "NKY", "NKZ"),
        help="Uniform k-grid dimensions (default: 4 4 4)",
    )
    parser.add_argument(
        "--remove-index",
        type=int,
        nargs="+",
        default=[],
        help="Global orbital indices to remove (default: none)",
    )
    parser.add_argument(
        "--remove-first-orbital-each-atom",
        action="store_true",
        help="Also remove the first orbital of each atom based on atom ordering.",
    )
    parser.add_argument(
        "--remove-orbital",
        type=str,
        nargs="+",
        default=[],
        help="Orbital families to remove for selected atoms (e.g. p or s p d).",
    )
    parser.add_argument(
        "--remove-shell",
        type=str,
        nargs="+",
        default=[],
        help="Specific shells to remove for selected atoms (e.g. 1s 2p).",
    )
    parser.add_argument(
        "--target-element",
        type=str,
        nargs="+",
        default=[],
        help="Restrict semantic orbital removal to these elements (e.g. C O).",
    )
    parser.add_argument(
        "--target-atom-index",
        type=int,
        nargs="+",
        default=[],
        help="Restrict semantic orbital removal to these 0-based atom indices.",
    )
    parser.add_argument(
        "--removal-plan-json",
        type=Path,
        default=None,
        help=(
            "JSON file defining multiple removal rules. "
            "Each rule may include: target_elements, target_atom_indices, remove_orbitals, remove_shells."
        ),
    )
    parser.add_argument(
        "--analyze-coupling",
        action="store_true",
        help=(
            "For every shell of every atom, compute the Schur-complement coupling "
            "to the rest of the basis and rank shells from safest to most coupled. "
            "SAFE shells (Schur-S < --safe-threshold) can be removed without corrupting "
            "the remaining bands. Use with removal flags to also check residual coupling "
            "after planned removals. Does NOT write output files."
        ),
    )
    parser.add_argument(
        "--suggest-companion-removals",
        action="store_true",
        help=(
            "Use the requested removal shell(s) as a seed and greedily suggest which additional "
            "shells should also be removed so that the combined removal set becomes safe under "
            "the Schur-coupling criterion. This is an analysis-only mode and does not write output files."
        ),
    )
    parser.add_argument(
        "--safe-threshold",
        type=float,
        default=0.01,
        help="Schur-S max below which a shell is flagged SAFE to remove (default: 0.01 = 1%%).",
    )
    parser.add_argument(
        "--marginal-threshold",
        type=float,
        default=0.05,
        help="Schur-S max below which a shell is flagged MARGINAL (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--suggest-max-steps",
        type=int,
        default=8,
        help="Maximum number of greedy companion-removal steps to take (default: 8).",
    )
    parser.add_argument(
        "--project-via-rspace-convolution",
        action="store_true",
        help=(
            "Build T(k), inverse-transform it to T(R), then compute projected H'(R)/S'(R) "
            "via full real-space convolution. Also saves T matrices in k-space and real-space "
            "as NPZ files and prints the numerical mismatch against the k-space-then-FT route."
        ),
    )

    args = parser.parse_args()
    run_pipeline(
        args.input_dir,
        args.output_dir,
        tuple(args.kgrid),
        remove_indices=list(args.remove_index),
        remove_first_orbital_each_atom=args.remove_first_orbital_each_atom,
        remove_orbitals=list(args.remove_orbital),
        remove_shells=list(args.remove_shell),
        target_elements=list(args.target_element),
        target_atom_indices=list(args.target_atom_index),
        removal_plan_json=args.removal_plan_json,
        analyze_coupling=args.analyze_coupling,
        suggest_companion_removals_flag=args.suggest_companion_removals,
        safe_threshold=args.safe_threshold,
        marginal_threshold=args.marginal_threshold,
        suggest_max_steps=args.suggest_max_steps,
        project_via_rspace_convolution=args.project_via_rspace_convolution,
    )
