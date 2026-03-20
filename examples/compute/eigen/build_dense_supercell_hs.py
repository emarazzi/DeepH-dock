"""
Build dense supercell Hamiltonian/overlap matrices from DeepH real-space blocks.

This example reconstructs full supercell matrices H and S as single dense NumPy
arrays by placing each non-zero real-space block H(R), S(R) into the correct
cell-cell block position.

Usage
-----
python examples/compute/eigen/build_dense_supercell_hs.py \
    --input-dir examples/compute/eigen/eigen.clean/Si_bulk \
    --output-npz supercell_hs.npz

If --supercell is omitted, it is inferred from Rijk_list as
(max(R) - min(R) + 1) along each lattice direction.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re
import shutil

import h5py

import numpy as np
from scipy import sparse

from deepx_dock.CONSTANT import (
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_INFO_FILENAME,
    DEEPX_OVERLAP_FILENAME,
    DEEPX_POSCAR_FILENAME,
)
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj


L_TO_LABEL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}


def _validate_supercell(supercell: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(supercell) != 3:
        raise ValueError(f"supercell must be length-3, got {supercell}")
    sx, sy, sz = (int(v) for v in supercell)
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"supercell entries must be positive, got {supercell}")
    return sx, sy, sz


def _cell_to_linear(ix: int, iy: int, iz: int, sx: int, sy: int, sz: int) -> int:
    if not (0 <= ix < sx and 0 <= iy < sy and 0 <= iz < sz):
        raise ValueError(f"Cell index out of range: ({ix}, {iy}, {iz}) for ({sx}, {sy}, {sz})")
    return ix + sx * (iy + sy * iz)


def infer_supercell_from_rijk(Rijk_list: np.ndarray | None) -> tuple[int, int, int]:
    """
    Infer a supercell size from available non-zero R blocks.

    The inferred size is the span of available R vectors in each direction:
    supercell[d] = max(R[:, d]) - min(R[:, d]) + 1.

    Parameters
    ----------
    Rijk_list : np.ndarray, shape (NR, 3)
        Integer translation vectors where matrix blocks are available.

    Returns
    -------
    tuple[int, int, int]
        Inferred supercell size (sx, sy, sz).
    """
    if Rijk_list is None:
        raise ValueError("Rijk_list is None; cannot infer supercell")

    Rs = np.asarray(Rijk_list, dtype=int)
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (NR, 3), got {Rs.shape}")
    if Rs.shape[0] == 0:
        return (1, 1, 1)

    mins = Rs.min(axis=0)
    maxs = Rs.max(axis=0)
    spans = maxs - mins + 1
    return (int(spans[0]), int(spans[1]), int(spans[2]))


def build_dense_supercell_hs(
    hamiltonian_obj: HamiltonianObj,
    supercell: tuple[int, int, int],
    periodic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dense supercell Hamiltonian and overlap matrices from non-zero R blocks.

    Parameters
    ----------
    hamiltonian_obj : HamiltonianObj
        Loaded Hamiltonian object containing HR/SR in shape (NR, Norb, Norb).
    supercell : tuple[int, int, int]
        Supercell size (sx, sy, sz) in lattice-vector units.
    periodic : bool, optional
        If True, apply periodic wrapping when target cell goes out of range.
        If False, use open boundaries (discard out-of-range couplings).

    Returns
    -------
    H_dense : np.ndarray, shape (Ncell*Norb, Ncell*Norb)
        Dense supercell Hamiltonian.
    S_dense : np.ndarray, shape (Ncell*Norb, Ncell*Norb)
        Dense supercell overlap matrix.
    """
    sx, sy, sz = _validate_supercell(supercell)

    HR = np.asarray(hamiltonian_obj.HR)
    SR = np.asarray(hamiltonian_obj.SR)
    Rs = np.asarray(hamiltonian_obj.Rijk_list, dtype=int)

    if HR.shape != SR.shape:
        raise ValueError(f"HR/SR shape mismatch: {HR.shape} vs {SR.shape}")
    if HR.ndim != 3:
        raise ValueError(f"Expected HR/SR with shape (NR, Norb, Norb), got {HR.shape}")
    if Rs.shape[0] != HR.shape[0] or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list shape mismatch: {Rs.shape} vs NR={HR.shape[0]}")

    nr, norb, _ = HR.shape
    ncell = sx * sy * sz
    dim = ncell * norb

    out_dtype = np.result_type(HR.dtype, SR.dtype)
    H_dense = np.zeros((dim, dim), dtype=out_dtype)
    S_dense = np.zeros((dim, dim), dtype=out_dtype)

    for iz in range(sz):
        for iy in range(sy):
            for ix in range(sx):
                src_cell = _cell_to_linear(ix, iy, iz, sx, sy, sz)
                row = slice(src_cell * norb, (src_cell + 1) * norb)

                for i_r in range(nr):
                    rx, ry, rz = Rs[i_r]
                    jx = ix + int(rx)
                    jy = iy + int(ry)
                    jz = iz + int(rz)

                    if periodic:
                        jx %= sx
                        jy %= sy
                        jz %= sz
                    elif not (0 <= jx < sx and 0 <= jy < sy and 0 <= jz < sz):
                        continue

                    dst_cell = _cell_to_linear(jx, jy, jz, sx, sy, sz)
                    col = slice(dst_cell * norb, (dst_cell + 1) * norb)

                    # Accumulate in case multiple R vectors fold to the same pair.
                    H_dense[row, col] += HR[i_r]
                    S_dense[row, col] += SR[i_r]

    return H_dense, S_dense


def build_sparse_supercell_hs(
    hamiltonian_obj: HamiltonianObj,
    supercell: tuple[int, int, int],
    periodic: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build full supercell H/S as sparse CSR matrices (not block containers).

    This keeps the same global basis ordering as the dense path:
    global_index = cell_index * Norb + orbital_index.
    """
    sx, sy, sz = _validate_supercell(supercell)

    HR = np.asarray(hamiltonian_obj.HR)
    SR = np.asarray(hamiltonian_obj.SR)
    Rs = np.asarray(hamiltonian_obj.Rijk_list, dtype=int)

    if HR.shape != SR.shape:
        raise ValueError(f"HR/SR shape mismatch: {HR.shape} vs {SR.shape}")
    if HR.ndim != 3:
        raise ValueError(f"Expected HR/SR with shape (NR, Norb, Norb), got {HR.shape}")
    if Rs.shape[0] != HR.shape[0] or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list shape mismatch: {Rs.shape} vs NR={HR.shape[0]}")

    nr, norb, _ = HR.shape
    ncell = sx * sy * sz
    dim = ncell * norb

    # Precompute local orbital index grids once.
    local_i = np.repeat(np.arange(norb, dtype=np.int64), norb)
    local_j = np.tile(np.arange(norb, dtype=np.int64), norb)

    h_rows: list[np.ndarray] = []
    h_cols: list[np.ndarray] = []
    h_vals: list[np.ndarray] = []
    s_rows: list[np.ndarray] = []
    s_cols: list[np.ndarray] = []
    s_vals: list[np.ndarray] = []

    for iz in range(sz):
        for iy in range(sy):
            for ix in range(sx):
                src_cell = _cell_to_linear(ix, iy, iz, sx, sy, sz)
                row_base = src_cell * norb
                row_idx = row_base + local_i

                for i_r in range(nr):
                    rx, ry, rz = Rs[i_r]
                    jx = ix + int(rx)
                    jy = iy + int(ry)
                    jz = iz + int(rz)

                    if periodic:
                        jx %= sx
                        jy %= sy
                        jz %= sz
                    elif not (0 <= jx < sx and 0 <= jy < sy and 0 <= jz < sz):
                        continue

                    dst_cell = _cell_to_linear(jx, jy, jz, sx, sy, sz)
                    col_base = dst_cell * norb
                    col_idx = col_base + local_j

                    h_block = HR[i_r].reshape(-1)
                    s_block = SR[i_r].reshape(-1)

                    h_nz = np.abs(h_block) > 0.0
                    s_nz = np.abs(s_block) > 0.0

                    if np.any(h_nz):
                        h_rows.append(row_idx[h_nz])
                        h_cols.append(col_idx[h_nz])
                        h_vals.append(h_block[h_nz])
                    if np.any(s_nz):
                        s_rows.append(row_idx[s_nz])
                        s_cols.append(col_idx[s_nz])
                        s_vals.append(s_block[s_nz])

    if h_vals:
        h_row = np.concatenate(h_rows)
        h_col = np.concatenate(h_cols)
        h_val = np.concatenate(h_vals)
    else:
        h_row = np.array([], dtype=np.int64)
        h_col = np.array([], dtype=np.int64)
        h_val = np.array([], dtype=HR.dtype)

    if s_vals:
        s_row = np.concatenate(s_rows)
        s_col = np.concatenate(s_cols)
        s_val = np.concatenate(s_vals)
    else:
        s_row = np.array([], dtype=np.int64)
        s_col = np.array([], dtype=np.int64)
        s_val = np.array([], dtype=SR.dtype)

    H_sparse = sparse.csr_matrix((h_val, (h_row, h_col)), shape=(dim, dim))
    S_sparse = sparse.csr_matrix((s_val, (s_row, s_col)), shape=(dim, dim))
    H_sparse.sum_duplicates()
    S_sparse.sum_duplicates()
    return H_sparse, S_sparse


def _resolve_global_removed_indices(
    ncell: int,
    norb_per_cell: int,
    remove_orbitals_unit: list[int],
) -> tuple[list[int], list[int]]:
    """Expand unit-cell orbital removals to global dense-matrix indices."""
    rm_unit = sorted(set(int(i) for i in remove_orbitals_unit))
    if len(rm_unit) == 0:
        return [], list(range(norb_per_cell))
    if rm_unit[0] < 0 or rm_unit[-1] >= norb_per_cell:
        raise ValueError(
            f"remove_orbitals_unit out of range [0, {norb_per_cell - 1}]: {rm_unit}"
        )

    keep_unit = [i for i in range(norb_per_cell) if i not in rm_unit]
    rm_global: list[int] = []
    for icell in range(ncell):
        base = icell * norb_per_cell
        rm_global.extend(base + i for i in rm_unit)
    return rm_global, keep_unit


def apply_sparse_orbital_mask(
    H_sparse: sparse.csr_matrix,
    S_sparse: sparse.csr_matrix,
    norb_per_cell: int,
    remove_orbitals_unit: list[int],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, object]]:
    """
    Remove selected unit-cell orbitals from sparse full supercell matrices.

    This is a structural basis reduction (principal submatrix extraction):
    rows/columns of the removed orbitals are dropped consistently across all cells.
    """
    h_nrow, h_ncol = H_sparse.get_shape()
    s_nrow, s_ncol = S_sparse.get_shape()
    if h_nrow != h_ncol:
        raise ValueError(f"H_sparse must be square, got {(h_nrow, h_ncol)}")
    if (s_nrow, s_ncol) != (h_nrow, h_ncol):
        raise ValueError(f"S_sparse shape mismatch: {(s_nrow, s_ncol)} vs {(h_nrow, h_ncol)}")
    if norb_per_cell <= 0:
        raise ValueError(f"norb_per_cell must be positive, got {norb_per_cell}")

    dim = int(h_nrow)
    if dim % norb_per_cell != 0:
        raise ValueError(
            f"Sparse size {dim} is not divisible by norb_per_cell={norb_per_cell}"
        )
    ncell = dim // norb_per_cell

    rm_global, keep_unit = _resolve_global_removed_indices(
        ncell=ncell,
        norb_per_cell=norb_per_cell,
        remove_orbitals_unit=remove_orbitals_unit,
    )

    if len(rm_global) == 0:
        return H_sparse.copy(), S_sparse.copy(), {
            "ncell": int(ncell),
            "norb_per_cell_original": int(norb_per_cell),
            "norb_per_cell_reduced": int(norb_per_cell),
            "remove_orbitals_unit": [],
            "keep_orbitals_unit": list(range(norb_per_cell)),
            "removed_global_indices": [],
            "kept_global_indices": list(range(dim)),
            "mode": "sparse-mask",
        }

    rm_set = set(rm_global)
    keep_global = np.array([i for i in range(dim) if i not in rm_set], dtype=np.int64)
    if keep_global.size == 0:
        raise ValueError("Cannot remove all orbitals in sparse basis")

    H_new = sparse.csr_matrix(H_sparse[keep_global, :][:, keep_global])
    S_new = sparse.csr_matrix(S_sparse[keep_global, :][:, keep_global])

    meta = {
        "ncell": int(ncell),
        "norb_per_cell_original": int(norb_per_cell),
        "norb_per_cell_reduced": int(len(keep_unit)),
        "remove_orbitals_unit": sorted(set(int(i) for i in remove_orbitals_unit)),
        "keep_orbitals_unit": keep_unit,
        "removed_global_indices": rm_global,
        "kept_global_indices": [int(v) for v in keep_global],
        "mode": "sparse-mask",
    }
    return H_new, S_new, meta


def apply_realspace_dense_t_transform(
    H_dense: np.ndarray,
    S_dense: np.ndarray,
    norb_per_cell: int,
    remove_orbitals_unit: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    Apply elimination T-transform directly in real-space dense supercell basis.

    This mirrors the k-space logic used in transform_k_to_r_pipeline.py:
      T[M, :] = -S_MM^{-1} S_MK
      T[K, :] = I
    but here M/K are chosen on the full dense supercell matrix.

    Orbital-consistent removal is enforced by `remove_orbitals_unit`: if local
    orbital index `p` is removed in one cell, it is removed in all cells.

    Parameters
    ----------
    H_dense : np.ndarray, shape (N, N)
        Dense supercell Hamiltonian matrix.
    S_dense : np.ndarray, shape (N, N)
        Dense supercell overlap matrix.
    norb_per_cell : int
        Number of orbitals per unit cell in the dense basis.
    remove_orbitals_unit : list[int]
        Local orbital indices (0-based, within a unit cell) to eliminate.

    Returns
    -------
    H_new : np.ndarray, shape (N_keep, N_keep)
        Transformed Hamiltonian in reduced basis.
    S_new : np.ndarray, shape (N_keep, N_keep)
        Transformed overlap in reduced basis.
    T_dense : np.ndarray, shape (N, N_keep)
        Dense elimination transform matrix.
    meta : dict[str, object]
        Mapping metadata for reconstruction and bookkeeping.
    """
    H = np.asarray(H_dense)
    S = np.asarray(S_dense)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H_dense must be square 2D, got {H.shape}")
    if S.ndim != 2 or S.shape != H.shape:
        raise ValueError(f"S_dense shape mismatch: {S.shape} vs {H.shape}")
    if norb_per_cell <= 0:
        raise ValueError(f"norb_per_cell must be positive, got {norb_per_cell}")

    dim = H.shape[0]
    if dim % norb_per_cell != 0:
        raise ValueError(
            f"Dense size {dim} is not divisible by norb_per_cell={norb_per_cell}"
        )
    ncell = dim // norb_per_cell

    rm_global, keep_unit = _resolve_global_removed_indices(
        ncell=ncell,
        norb_per_cell=norb_per_cell,
        remove_orbitals_unit=remove_orbitals_unit,
    )

    if len(rm_global) == 0:
        eye = np.eye(dim, dtype=np.complex128)
        H_new = np.array(H, copy=True)
        S_new = np.array(S, copy=True)
        meta = {
            "ncell": int(ncell),
            "norb_per_cell_original": int(norb_per_cell),
            "norb_per_cell_reduced": int(norb_per_cell),
            "remove_orbitals_unit": [],
            "keep_orbitals_unit": list(range(norb_per_cell)),
            "removed_global_indices": [],
            "kept_global_indices": list(range(dim)),
            "reduced_index_to_cell_orbital": [
                {
                    "reduced_idx": int(i),
                    "cell_idx": int(i // norb_per_cell),
                    "orbital_unit_idx": int(i % norb_per_cell),
                    "original_global_idx": int(i),
                }
                for i in range(dim)
            ],
        }
        return H_new, S_new, eye, meta

    keep_global = [i for i in range(dim) if i not in set(rm_global)]
    if len(keep_global) == 0:
        raise ValueError("Cannot remove all orbitals in dense basis")

    S_mm = S[np.ix_(rm_global, rm_global)]
    S_mk = S[np.ix_(rm_global, keep_global)]
    coeff = -np.linalg.solve(S_mm, S_mk)

    nkeep = len(keep_global)
    T_dense = np.zeros((dim, nkeep), dtype=np.complex128)
    for j_keep, i_global in enumerate(keep_global):
        T_dense[i_global, j_keep] = 1.0
    T_dense[rm_global, :] = coeff

    Tc = np.conjugate(T_dense.T)
    H_new = Tc @ H @ T_dense
    S_new = Tc @ S @ T_dense

    norb_reduced = len(keep_unit)
    reduced_index_to_cell_orbital: list[dict[str, int]] = []
    for red_i, gidx in enumerate(keep_global):
        cell_idx = gidx // norb_per_cell
        orb_idx = gidx % norb_per_cell
        reduced_index_to_cell_orbital.append(
            {
                "reduced_idx": int(red_i),
                "cell_idx": int(cell_idx),
                "orbital_unit_idx": int(orb_idx),
                "original_global_idx": int(gidx),
            }
        )

    meta = {
        "ncell": int(ncell),
        "norb_per_cell_original": int(norb_per_cell),
        "norb_per_cell_reduced": int(norb_reduced),
        "remove_orbitals_unit": sorted(set(int(i) for i in remove_orbitals_unit)),
        "keep_orbitals_unit": keep_unit,
        "removed_global_indices": rm_global,
        "kept_global_indices": keep_global,
        "reduced_index_to_cell_orbital": reduced_index_to_cell_orbital,
    }
    return H_new, S_new, T_dense, meta


def _evaluate_schur_set_safety(
    S_dense: np.ndarray,
    norb_per_cell: int,
    remove_orbitals_unit: list[int],
    eig_tol: float,
    cond_max: float,
    reduced_cond_max: float,
    herm_rel_tol: float,
    removal_map: dict[str, object] | None = None,
) -> tuple[bool, dict[str, object]]:
    """Evaluate numerical safety of a Schur elimination set on dense overlap."""
    S = np.asarray(S_dense)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S_dense must be square 2D, got {S.shape}")
    if norb_per_cell <= 0:
        raise ValueError(f"norb_per_cell must be positive, got {norb_per_cell}")

    dim = int(S.shape[0])
    if dim % norb_per_cell != 0:
        raise ValueError(
            f"Dense size {dim} is not divisible by norb_per_cell={norb_per_cell}"
        )
    ncell = dim // norb_per_cell

    rm_global, _ = _resolve_global_removed_indices(
        ncell=ncell,
        norb_per_cell=norb_per_cell,
        remove_orbitals_unit=remove_orbitals_unit,
    )
    keep_global = [i for i in range(dim) if i not in set(rm_global)]
    if len(keep_global) == 0:
        return False, {
            "reason": "all-orbitals-removed",
            "safe": False,
            "removal_plan_like": removal_map if removal_map is not None else {},
        }

    S_mm = S[np.ix_(rm_global, rm_global)]
    S_mk = S[np.ix_(rm_global, keep_global)]
    S_kk = S[np.ix_(keep_global, keep_global)]

    S_mm_h = 0.5 * (S_mm + np.conjugate(S_mm.T))
    mm_norm = float(np.linalg.norm(S_mm_h))
    mm_anti = float(np.linalg.norm(S_mm - np.conjugate(S_mm.T)))
    mm_herm_rel = mm_anti / max(mm_norm, 1.0e-30)
    if mm_herm_rel > herm_rel_tol:
        return False, {
            "reason": "s_mm-non-hermitian",
            "safe": False,
            "removal_plan_like": removal_map if removal_map is not None else {},
            "s_mm_hermiticity_rel": mm_herm_rel,
            "herm_rel_tol": float(herm_rel_tol),
        }

    eval_mm = np.linalg.eigvalsh(S_mm_h)
    mm_min = float(np.min(eval_mm)) if eval_mm.size > 0 else float("inf")
    if mm_min <= eig_tol:
        return False, {
            "reason": "s_mm-not-spd",
            "safe": False,
            "removal_plan_like": removal_map if removal_map is not None else {},
            "s_mm_min_eig": mm_min,
            "eig_tol": float(eig_tol),
        }

    cond_mm = float(np.linalg.cond(S_mm_h))
    if not np.isfinite(cond_mm) or cond_mm > cond_max:
        return False, {
            "reason": "s_mm-ill-conditioned",
            "safe": False,
            "removal_plan_like": removal_map if removal_map is not None else {},
            "s_mm_cond": cond_mm,
            "cond_max": float(cond_max),
            "s_mm_min_eig": mm_min,
        }

    coeff = np.linalg.solve(S_mm, S_mk)
    S_red = S_kk - np.conjugate(S_mk.T) @ coeff
    S_red_h = 0.5 * (S_red + np.conjugate(S_red.T))
    red_norm = float(np.linalg.norm(S_red_h))
    red_anti = float(np.linalg.norm(S_red - np.conjugate(S_red.T)))
    red_herm_rel = red_anti / max(red_norm, 1.0e-30)
    if red_herm_rel > herm_rel_tol:
        return False, {
            "reason": "s_reduced-non-hermitian",
            "safe": False,
            "removal_plan_like": removal_map if removal_map is not None else {},
            "s_mm_min_eig": mm_min,
            "s_mm_cond": cond_mm,
            "s_reduced_hermiticity_rel": red_herm_rel,
            "herm_rel_tol": float(herm_rel_tol),
        }

    eval_red = np.linalg.eigvalsh(S_red_h)
    red_min = float(np.min(eval_red)) if eval_red.size > 0 else float("inf")
    cond_red = float(np.linalg.cond(S_red_h))
    if not np.isfinite(cond_red) or cond_red > reduced_cond_max:
        return False, {
            "reason": "s_reduced-ill-conditioned",
            "safe": False,
            "removal_plan_like": removal_map if removal_map is not None else {},
            "s_mm_min_eig": mm_min,
            "s_mm_cond": cond_mm,
            "s_reduced_min_eig": red_min,
            "s_reduced_cond": cond_red,
            "reduced_cond_max": float(reduced_cond_max),
        }

    red_safe = bool(red_min > eig_tol)

    return red_safe, {
        "reason": "ok" if red_safe else "s_reduced-not-spd",
        "safe": red_safe,
        "removal_plan_like": removal_map if removal_map is not None else {},
        "s_mm_min_eig": mm_min,
        "s_mm_cond": cond_mm,
        "s_reduced_min_eig": red_min,
        "s_reduced_cond": cond_red,
        "s_reduced_hermiticity_rel": red_herm_rel,
        "eig_tol": float(eig_tol),
        "cond_max": float(cond_max),
        "reduced_cond_max": float(reduced_cond_max),
        "herm_rel_tol": float(herm_rel_tol),
    }


def _build_removal_plan_like_map(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    remove_orbitals_unit: list[int],
) -> dict[str, object]:
    """Build a species/orbital map for removed unit-cell orbital indices."""
    rm_set = set(int(i) for i in remove_orbitals_unit)

    by_atom: list[dict[str, object]] = []
    by_element_acc: dict[str, dict[str, set[str]]] = {}

    g0 = 0
    for ia, el in enumerate(elements):
        shell_ls = [int(v) for v in elements_orbital_map[el]]
        shell_count_by_l: dict[int, int] = {}

        atom_orbital_families: set[str] = set()
        atom_full_shells: set[str] = set()
        atom_partial_shells: list[dict[str, object]] = []

        for l in shell_ls:
            shell_n = shell_count_by_l.get(l, 0) + 1
            shell_count_by_l[l] = shell_n

            shell_dim = 2 * l + 1
            shell_start = g0
            shell_stop = g0 + shell_dim

            removed_here = [idx for idx in range(shell_start, shell_stop) if idx in rm_set]
            if removed_here:
                label = L_TO_LABEL.get(l, f"l{l}")
                shell_name = f"{shell_n}{label}"
                atom_orbital_families.add(label)
                if len(removed_here) == shell_dim:
                    atom_full_shells.add(shell_name)
                else:
                    atom_partial_shells.append(
                        {
                            "shell": shell_name,
                            "removed_components": int(len(removed_here)),
                            "shell_dim": int(shell_dim),
                        }
                    )
            g0 = shell_stop

        if atom_orbital_families or atom_full_shells or atom_partial_shells:
            by_atom.append(
                {
                    "target_elements": [str(el)],
                    "target_atom_indices": [int(ia)],
                    "remove_orbitals": sorted(atom_orbital_families),
                    "remove_shells": sorted(atom_full_shells),
                    "partial_shells": atom_partial_shells,
                }
            )

            acc = by_element_acc.setdefault(str(el), {"remove_orbitals": set(), "remove_shells": set()})
            acc["remove_orbitals"].update(atom_orbital_families)
            acc["remove_shells"].update(atom_full_shells)

    by_element: list[dict[str, object]] = []
    for el in sorted(by_element_acc):
        by_element.append(
            {
                "target_elements": [el],
                "remove_orbitals": sorted(by_element_acc[el]["remove_orbitals"]),
                "remove_shells": sorted(by_element_acc[el]["remove_shells"]),
            }
        )

    return {
        "rules_by_atom": by_atom,
        "rules_by_element": by_element,
    }


def analyze_schur_elimination_safety(
    S_dense: np.ndarray,
    norb_per_cell: int,
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    candidate_orbitals_unit: list[int] | None = None,
    max_group_size: int = 2,
    eig_tol: float = 1.0e-8,
    cond_max: float = 1.0e12,
    reduced_cond_max: float = 1.0e10,
    herm_rel_tol: float = 1.0e-10,
    max_combinations: int = 5000,
) -> dict[str, object]:
    """
    Analyze which orbital-removal sets are numerically safe for Schur elimination.

    The analysis first checks single-orbital removals, then optional grouped removals
    up to `max_group_size`, and reports combinations that are safe despite containing
    one or more single-orbital unsafe members.
    """
    if max_group_size < 1:
        raise ValueError(f"max_group_size must be >= 1, got {max_group_size}")
    if max_combinations <= 0:
        raise ValueError(f"max_combinations must be positive, got {max_combinations}")

    S = np.asarray(S_dense)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S_dense must be square 2D, got {S.shape}")
    if norb_per_cell <= 0:
        raise ValueError(f"norb_per_cell must be positive, got {norb_per_cell}")

    if candidate_orbitals_unit is None:
        candidates = list(range(norb_per_cell))
    else:
        candidates = sorted(set(int(i) for i in candidate_orbitals_unit))
    bad = [i for i in candidates if i < 0 or i >= norb_per_cell]
    if bad:
        raise ValueError(
            f"candidate_orbitals_unit out of range [0, {norb_per_cell - 1}]: {bad}"
        )

    single_safe: list[int] = []
    single_safe_details: list[dict[str, object]] = []
    single_unsafe: list[dict[str, object]] = []
    single_safe_set: set[int] = set()

    for i in candidates:
        is_safe, detail = _evaluate_schur_set_safety(
            S_dense=S,
            norb_per_cell=norb_per_cell,
            remove_orbitals_unit=[i],
            eig_tol=eig_tol,
            cond_max=cond_max,
            reduced_cond_max=reduced_cond_max,
            herm_rel_tol=herm_rel_tol,
            removal_map=_build_removal_plan_like_map(
                elements=elements,
                elements_orbital_map=elements_orbital_map,
                remove_orbitals_unit=[i],
            ),
        )
        if is_safe:
            single_safe.append(int(i))
            single_safe_set.add(int(i))
            single_safe_details.append(detail)
        else:
            single_unsafe.append(detail)

    group_safe_with_unsafe_members: list[dict[str, object]] = []
    tested_combinations = len(candidates)

    for k in range(2, max_group_size + 1):
        for group in itertools.combinations(candidates, k):
            if tested_combinations >= max_combinations:
                break
            tested_combinations += 1
            is_safe, detail = _evaluate_schur_set_safety(
                S_dense=S,
                norb_per_cell=norb_per_cell,
                remove_orbitals_unit=list(group),
                eig_tol=eig_tol,
                cond_max=cond_max,
                reduced_cond_max=reduced_cond_max,
                herm_rel_tol=herm_rel_tol,
                removal_map=_build_removal_plan_like_map(
                    elements=elements,
                    elements_orbital_map=elements_orbital_map,
                    remove_orbitals_unit=list(group),
                ),
            )
            if is_safe:
                has_any_single_unsafe = any(int(i) not in single_safe_set for i in group)
                if has_any_single_unsafe:
                    group_safe_with_unsafe_members.append(detail)
        if tested_combinations >= max_combinations:
            break

    return {
        "norb_per_cell": int(norb_per_cell),
        "candidate_count": int(len(candidates)),
        "single_safe_count": int(len(single_safe)),
        "single_safe_details": single_safe_details,
        "single_unsafe_details": single_unsafe,
        "safe_groups_with_single_unsafe_members": group_safe_with_unsafe_members,
        "analysis_params": {
            "max_group_size": int(max_group_size),
            "eig_tol": float(eig_tol),
            "cond_max": float(cond_max),
            "reduced_cond_max": float(reduced_cond_max),
            "herm_rel_tol": float(herm_rel_tol),
            "max_combinations": int(max_combinations),
            "tested_combinations": int(tested_combinations),
        },
    }


def dense_to_cell_block_matrix(
    M_dense: np.ndarray,
    ncell: int,
    norb_per_cell: int,
) -> np.ndarray:
    """
    Reshape a dense supercell matrix into explicit cell-cell block form.

    Returns
    -------
    np.ndarray, shape (Ncell, Ncell, Norb, Norb)
        Block matrix where out[i, j] is the (i, j) cell block.
    """
    M = np.asarray(M_dense)
    dim_expected = ncell * norb_per_cell
    if M.shape != (dim_expected, dim_expected):
        raise ValueError(
            f"Expected M_dense shape {(dim_expected, dim_expected)}, got {M.shape}"
        )

    out = M.reshape(ncell, norb_per_cell, ncell, norb_per_cell)
    return np.transpose(out, (0, 2, 1, 3)).copy()


def aggregate_dense_to_rblocks(
    M_dense: np.ndarray,
    supercell: tuple[int, int, int],
    Rijk_list: np.ndarray,
    norb_per_cell: int,
    periodic: bool = False,
) -> np.ndarray:
    """
    Aggregate a dense supercell matrix back to R-space blocks on a target R list.

    For each R in Rijk_list, this computes the average cell-cell block M(i, i+R)
    over all valid source cells i (or wrapped cells if periodic=True).
    """
    sx, sy, sz = _validate_supercell(supercell)
    M = np.asarray(M_dense)
    Rs = np.asarray(Rijk_list, dtype=int)

    ncell = sx * sy * sz
    dim = ncell * norb_per_cell
    if M.shape != (dim, dim):
        raise ValueError(f"M_dense shape mismatch: expected {(dim, dim)}, got {M.shape}")
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (NR, 3), got {Rs.shape}")

    nr = Rs.shape[0]
    out = np.zeros((nr, norb_per_cell, norb_per_cell), dtype=M.dtype)
    counts = np.zeros(nr, dtype=np.int64)

    for iz in range(sz):
        for iy in range(sy):
            for ix in range(sx):
                src_cell = _cell_to_linear(ix, iy, iz, sx, sy, sz)
                row = slice(src_cell * norb_per_cell, (src_cell + 1) * norb_per_cell)

                for i_r, (rx, ry, rz) in enumerate(Rs):
                    jx = ix + int(rx)
                    jy = iy + int(ry)
                    jz = iz + int(rz)

                    if periodic:
                        jx %= sx
                        jy %= sy
                        jz %= sz
                    elif not (0 <= jx < sx and 0 <= jy < sy and 0 <= jz < sz):
                        continue

                    dst_cell = _cell_to_linear(jx, jy, jz, sx, sy, sz)
                    col = slice(dst_cell * norb_per_cell, (dst_cell + 1) * norb_per_cell)
                    out[i_r] += M[row, col]
                    counts[i_r] += 1

    for i_r in range(nr):
        if counts[i_r] > 0:
            out[i_r] /= counts[i_r]
    return out


def aggregate_sparse_to_rblocks(
    M_sparse: sparse.csr_matrix,
    supercell: tuple[int, int, int],
    Rijk_list: np.ndarray,
    norb_per_cell: int,
    periodic: bool = False,
) -> np.ndarray:
    """
    Aggregate a sparse full supercell matrix back to R-space blocks on a target R list.

    For each R in Rijk_list, this computes the average cell-cell block M(i, i+R)
    over all valid source cells i (or wrapped cells if periodic=True).
    """
    sx, sy, sz = _validate_supercell(supercell)
    M = M_sparse.tocsr()
    Rs = np.asarray(Rijk_list, dtype=int)

    ncell = sx * sy * sz
    dim = ncell * norb_per_cell
    if M.shape != (dim, dim):
        raise ValueError(f"M_sparse shape mismatch: expected {(dim, dim)}, got {M.shape}")
    if Rs.ndim != 2 or Rs.shape[1] != 3:
        raise ValueError(f"Rijk_list must have shape (NR, 3), got {Rs.shape}")

    nr = Rs.shape[0]
    out = np.zeros((nr, norb_per_cell, norb_per_cell), dtype=M.dtype)
    counts = np.zeros(nr, dtype=np.int64)

    for iz in range(sz):
        for iy in range(sy):
            for ix in range(sx):
                src_cell = _cell_to_linear(ix, iy, iz, sx, sy, sz)
                row0 = src_cell * norb_per_cell
                row1 = (src_cell + 1) * norb_per_cell

                for i_r, (rx, ry, rz) in enumerate(Rs):
                    jx = ix + int(rx)
                    jy = iy + int(ry)
                    jz = iz + int(rz)

                    if periodic:
                        jx %= sx
                        jy %= sy
                        jz %= sz
                    elif not (0 <= jx < sx and 0 <= jy < sy and 0 <= jz < sz):
                        continue

                    dst_cell = _cell_to_linear(jx, jy, jz, sx, sy, sz)
                    col0 = dst_cell * norb_per_cell
                    col1 = (dst_cell + 1) * norb_per_cell

                    out[i_r] += M[row0:row1, col0:col1].toarray()
                    counts[i_r] += 1

    for i_r in range(nr):
        if counts[i_r] > 0:
            out[i_r] /= counts[i_r]
    return out


def _parse_shell_selector(token: str) -> tuple[int, int]:
    """Parse shell selector strings like '1s', '2p', '3d' into (n, l)."""
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z])\s*", token)
    if m is None:
        raise ValueError(f"Invalid shell selector '{token}'. Use forms like '1s', '2p', '3d'.")
    n = int(m.group(1))
    label = m.group(2).lower()
    inv = {v: k for k, v in L_TO_LABEL.items()}
    if label not in inv:
        raise ValueError(f"Unsupported shell label '{label}'.")
    return n, inv[label]


def _normalize_orbital_labels(labels: list[str]) -> set[int]:
    """Map labels like ['s', 'p'] to angular momentum integers l."""
    inv = {v: k for k, v in L_TO_LABEL.items()}
    out: set[int] = set()
    for lb in labels:
        key = lb.strip().lower()
        if key == "":
            continue
        if key not in inv:
            raise ValueError(f"Unsupported orbital family '{lb}'. Supported: {sorted(inv)}")
        out.add(inv[key])
    return out


def resolve_indices_from_orbital_selectors(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    target_elements: list[str],
    target_atom_indices: list[int],
    remove_orbitals: list[str],
    remove_shells: list[str],
) -> tuple[list[int], dict[str, object]]:
    """Resolve semantic shell/orbital selectors into unit-cell global AO indices."""
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
    """Resolve a list of removal rules into unit-cell global AO indices."""

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
    """Build updated elements_orbital_map after shell removal."""
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
                    "Partial removal inside shell is not supported for info.json rewrite: "
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


def dump_reduced_matrix_h5(
    out_path: Path,
    mats_R: np.ndarray,
    Rijk_list: np.ndarray,
    atom_pairs: np.ndarray,
    atom_num_orbits_cumsum: np.ndarray,
    keep_global: list[int],
) -> None:
    """Dump reduced-basis real-space matrices to DeepH-format h5."""
    keep_arr = np.array(sorted(set(int(i) for i in keep_global)), dtype=int)

    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    atom_pairs = np.asarray(atom_pairs, dtype=np.int64)

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


def save_transformed_deeph_dataset(
    input_dir: Path,
    output_dir: Path,
    obj: HamiltonianObj,
    supercell: tuple[int, int, int],
    H_dense_new: np.ndarray,
    S_dense_new: np.ndarray,
    keep_orbitals_unit: list[int],
    periodic: bool,
    selector_meta: dict[str, object],
    rule_plan_meta: list[dict[str, object]],
) -> None:
    """Save transformed reduced H/S and info.json into a new folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_dir / DEEPX_POSCAR_FILENAME, output_dir / DEEPX_POSCAR_FILENAME)

    norb_reduced = len(keep_orbitals_unit)
    HR_new = aggregate_dense_to_rblocks(
        M_dense=H_dense_new,
        supercell=supercell,
        Rijk_list=np.asarray(obj.Rijk_list, dtype=int),
        norb_per_cell=norb_reduced,
        periodic=periodic,
    )
    SR_new = aggregate_dense_to_rblocks(
        M_dense=S_dense_new,
        supercell=supercell,
        Rijk_list=np.asarray(obj.Rijk_list, dtype=int),
        norb_per_cell=norb_reduced,
        periodic=periodic,
    )
    SR_new = np.asarray(np.real(SR_new), dtype=np.float64)

    dump_reduced_matrix_h5(
        output_dir / DEEPX_HAMILTONIAN_FILENAME,
        HR_new,
        np.asarray(obj.Rijk_list, dtype=int),
        np.asarray(obj.atom_pairs, dtype=np.int64),
        np.asarray(obj.atom_num_orbits_cumsum, dtype=np.int64),
        keep_orbitals_unit,
    )
    dump_reduced_matrix_h5(
        output_dir / DEEPX_OVERLAP_FILENAME,
        SR_new,
        np.asarray(obj.Rijk_list, dtype=int),
        np.asarray(obj.atom_pairs, dtype=np.int64),
        np.asarray(obj.atom_num_orbits_cumsum, dtype=np.int64),
        keep_orbitals_unit,
    )

    removed_unit = [
        i for i in range(int(obj.orbits_quantity))
        if i not in set(int(v) for v in keep_orbitals_unit)
    ]
    write_reduced_info_json(
        input_dir=input_dir,
        output_dir=output_dir,
        elements=[str(el) for el in obj.elements],
        removed_indices=removed_unit,
    )

    meta = {
        "supercell": list(supercell),
        "removed_unit_indices": removed_unit,
        "kept_unit_indices": list(keep_orbitals_unit),
        "selector_resolution": selector_meta,
        "rule_plan_resolution": rule_plan_meta,
    }
    with open(output_dir / "reduced_basis_meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2)
        fw.write("\n")


def save_transformed_deeph_dataset_sparse(
    input_dir: Path,
    output_dir: Path,
    obj: HamiltonianObj,
    supercell: tuple[int, int, int],
    H_sparse_new: sparse.csr_matrix,
    S_sparse_new: sparse.csr_matrix,
    keep_orbitals_unit: list[int],
    periodic: bool,
    selector_meta: dict[str, object],
    rule_plan_meta: list[dict[str, object]],
) -> None:
    """Save transformed reduced H/S and info.json from sparse full matrices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_dir / DEEPX_POSCAR_FILENAME, output_dir / DEEPX_POSCAR_FILENAME)

    norb_reduced = len(keep_orbitals_unit)
    HR_new = aggregate_sparse_to_rblocks(
        M_sparse=H_sparse_new,
        supercell=supercell,
        Rijk_list=np.asarray(obj.Rijk_list, dtype=int),
        norb_per_cell=norb_reduced,
        periodic=periodic,
    )
    SR_new = aggregate_sparse_to_rblocks(
        M_sparse=S_sparse_new,
        supercell=supercell,
        Rijk_list=np.asarray(obj.Rijk_list, dtype=int),
        norb_per_cell=norb_reduced,
        periodic=periodic,
    )
    SR_new = np.asarray(np.real(SR_new), dtype=np.float64)

    dump_reduced_matrix_h5(
        output_dir / DEEPX_HAMILTONIAN_FILENAME,
        HR_new,
        np.asarray(obj.Rijk_list, dtype=int),
        np.asarray(obj.atom_pairs, dtype=np.int64),
        np.asarray(obj.atom_num_orbits_cumsum, dtype=np.int64),
        keep_orbitals_unit,
    )
    dump_reduced_matrix_h5(
        output_dir / DEEPX_OVERLAP_FILENAME,
        SR_new,
        np.asarray(obj.Rijk_list, dtype=int),
        np.asarray(obj.atom_pairs, dtype=np.int64),
        np.asarray(obj.atom_num_orbits_cumsum, dtype=np.int64),
        keep_orbitals_unit,
    )

    removed_unit = [
        i for i in range(int(obj.orbits_quantity))
        if i not in set(int(v) for v in keep_orbitals_unit)
    ]
    write_reduced_info_json(
        input_dir=input_dir,
        output_dir=output_dir,
        elements=[str(el) for el in obj.elements],
        removed_indices=removed_unit,
    )

    h_nrow, h_ncol = H_sparse_new.get_shape()
    meta = {
        "supercell": list(supercell),
        "removed_unit_indices": removed_unit,
        "kept_unit_indices": list(keep_orbitals_unit),
        "selector_resolution": selector_meta,
        "rule_plan_resolution": rule_plan_meta,
        "mode": "sparse",
        "shape": [int(h_nrow), int(h_ncol)],
        "H_nnz": int(H_sparse_new.nnz),
        "S_nnz": int(S_sparse_new.nnz),
    }
    with open(output_dir / "reduced_basis_meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2)
        fw.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense supercell H/S from DeepH real-space blocks")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing DeepH files")
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=None,
        help=(
            "Optional supercell size in lattice vectors, e.g. --supercell 2 2 2. "
            "If omitted, infer from Rijk_list span."
        ),
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic wrapping for out-of-range target cells",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Build and keep full supercell matrices in sparse CSR format.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional output .npz path to store dense H and S",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory to write transformed hamiltonian.h5/overlap.h5/info.json",
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
        help="Specific shells to remove for selected atoms (e.g. 1p 3d).",
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
        "--analyze-schur-safety",
        action="store_true",
        help="Analyze which orbital removals are numerically safe for Schur elimination (dense mode only).",
    )
    parser.add_argument(
        "--safety-max-group-size",
        type=int,
        default=2,
        help="Maximum unit-cell removal group size to test in Schur safety analysis.",
    )
    parser.add_argument(
        "--safety-eig-tol",
        type=float,
        default=1.0e-8,
        help="Eigenvalue tolerance used to classify SPD safety in Schur analysis.",
    )
    parser.add_argument(
        "--safety-cond-max",
        type=float,
        default=1.0e12,
        help="Maximum allowed condition number for S_MM in Schur safety analysis.",
    )
    parser.add_argument(
        "--safety-reduced-cond-max",
        type=float,
        default=1.0e10,
        help="Maximum allowed condition number for reduced overlap S' in Schur safety analysis.",
    )
    parser.add_argument(
        "--safety-herm-rel-tol",
        type=float,
        default=1.0e-10,
        help="Relative Hermiticity tolerance for S_MM and reduced overlap S' in Schur safety analysis.",
    )
    parser.add_argument(
        "--safety-max-combinations",
        type=int,
        default=5000,
        help="Hard cap on tested removal combinations in Schur safety analysis.",
    )
    parser.add_argument(
        "--safety-report-json",
        type=Path,
        default=None,
        help="Optional path to write Schur safety analysis report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    ham = HamiltonianObj(args.input_dir)
    if args.supercell is None:
        supercell = infer_supercell_from_rijk(ham.Rijk_list)
        print(f"Supercell inferred from Rijk_list: {supercell}")
    else:
        supercell = tuple(args.supercell)

    use_sparse = bool(args.sparse)
    if use_sparse:
        H_sparse, S_sparse = build_sparse_supercell_hs(
            hamiltonian_obj=ham,
            supercell=supercell,
            periodic=bool(args.periodic),
        )
    else:
        H_dense, S_dense = build_dense_supercell_hs(
            hamiltonian_obj=ham,
            supercell=supercell,
            periodic=bool(args.periodic),
        )

    selector_indices, selector_meta = resolve_indices_from_orbital_selectors(
        elements=[str(el) for el in ham.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in ham.elements_orbital_map.items()},
        target_elements=list(args.target_element),
        target_atom_indices=[int(v) for v in args.target_atom_index],
        remove_orbitals=list(args.remove_orbital),
        remove_shells=list(args.remove_shell),
    )

    plan_meta: list[dict[str, object]] = []
    rm_unit = list(selector_indices)
    if args.removal_plan_json is not None:
        with open(args.removal_plan_json, "r", encoding="utf-8") as fr:
            payload = json.load(fr)
        rules = payload.get("rules", payload)
        if not isinstance(rules, list):
            raise ValueError("removal plan must be a list or an object with key 'rules' (list)")
        plan_indices, plan_meta = resolve_indices_from_rules(
            elements=[str(el) for el in ham.elements],
            elements_orbital_map={k: [int(v) for v in vals] for k, vals in ham.elements_orbital_map.items()},
            rules=rules,
        )
        rm_unit.extend(plan_indices)

    rm_unit = sorted(set(int(i) for i in rm_unit))

    if args.analyze_schur_safety:
        if use_sparse:
            raise ValueError("--analyze-schur-safety requires dense mode; do not pass --sparse")
        analysis_candidates = rm_unit if rm_unit else list(range(int(ham.orbits_quantity)))
        safety_report = analyze_schur_elimination_safety(
            S_dense=S_dense,
            norb_per_cell=int(ham.orbits_quantity),
            elements=[str(el) for el in ham.elements],
            elements_orbital_map={k: [int(v) for v in vals] for k, vals in ham.elements_orbital_map.items()},
            candidate_orbitals_unit=analysis_candidates,
            max_group_size=int(args.safety_max_group_size),
            eig_tol=float(args.safety_eig_tol),
            cond_max=float(args.safety_cond_max),
            reduced_cond_max=float(args.safety_reduced_cond_max),
            herm_rel_tol=float(args.safety_herm_rel_tol),
            max_combinations=int(args.safety_max_combinations),
        )
        single_safe_obj = safety_report.get("single_safe_details", [])
        candidates_obj = safety_report.get("candidate_count", 0)
        rescued_groups_obj = safety_report.get("safe_groups_with_single_unsafe_members", [])
        if not isinstance(single_safe_obj, list):
            raise ValueError("Internal error: single_safe_details is not a list")
        if not isinstance(candidates_obj, int):
            raise ValueError("Internal error: candidate_count is not an int")
        if not isinstance(rescued_groups_obj, list):
            raise ValueError("Internal error: safe_groups_with_single_unsafe_members is not a list")
        single_safe = single_safe_obj
        candidates = candidates_obj
        rescued_groups = rescued_groups_obj
        print("Schur safety analysis completed.")
        print(
            "Single safe orbitals: "
            f"{len(single_safe)}/"
            f"{candidates}"
        )
        print(
            "Safe groups containing single-unsafe orbitals: "
            f"{len(rescued_groups)}"
        )
        if args.safety_report_json is not None:
            args.safety_report_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.safety_report_json, "w", encoding="utf-8") as fw:
                json.dump(safety_report, fw, indent=2)
                fw.write("\n")
            print(f"Saved Schur safety report: {args.safety_report_json}")

    if use_sparse:
        if rm_unit:
            H_sparse_out, S_sparse_out, t_meta = apply_sparse_orbital_mask(
                H_sparse=H_sparse,
                S_sparse=S_sparse,
                norb_per_cell=int(ham.orbits_quantity),
                remove_orbitals_unit=rm_unit,
            )
            print("Applied sparse orbital masking (row/column removal across all cells).")
            print(f"Removed unit-cell orbital indices: {rm_unit}")
        else:
            H_sparse_out = H_sparse
            S_sparse_out = S_sparse
            t_meta = {
                "keep_orbitals_unit": list(range(int(ham.orbits_quantity))),
                "remove_orbitals_unit": [],
                "mode": "sparse-mask",
            }
        h_nnz = int(H_sparse_out.nnz)
        s_nnz = int(S_sparse_out.nnz)
    else:
        if rm_unit:
            H_dense_out, S_dense_out, T_dense, t_meta = apply_realspace_dense_t_transform(
                H_dense=H_dense,
                S_dense=S_dense,
                norb_per_cell=int(ham.orbits_quantity),
                remove_orbitals_unit=rm_unit,
            )
            print(f"Removed unit-cell orbital indices: {rm_unit}")
            print(f"Dense T shape: {T_dense.shape}")
        else:
            H_dense_out = H_dense
            S_dense_out = S_dense
            t_meta = {
                "keep_orbitals_unit": list(range(int(ham.orbits_quantity))),
                "remove_orbitals_unit": [],
            }

        h_nnz = int(np.count_nonzero(np.abs(H_dense_out) > 0.0))
        s_nnz = int(np.count_nonzero(np.abs(S_dense_out) > 0.0))

    print(f"Supercell: {supercell}")
    if use_sparse:
        print(f"Sparse H shape: {H_sparse_out.shape}, nnz={h_nnz}")
        print(f"Sparse S shape: {S_sparse_out.shape}, nnz={s_nnz}")
    else:
        print(f"Dense H shape: {H_dense_out.shape}, nnz={h_nnz}")
        print(f"Dense S shape: {S_dense_out.shape}, nnz={s_nnz}")

    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        if use_sparse:
            stem = args.output_npz.stem
            H_path = args.output_npz.with_name(f"{stem}_H_sparse.npz")
            S_path = args.output_npz.with_name(f"{stem}_S_sparse.npz")
            sparse.save_npz(H_path, H_sparse_out)
            sparse.save_npz(S_path, S_sparse_out)
            meta_path = args.output_npz.with_name(f"{stem}_sparse_meta.json")
            h_nrow, h_ncol = H_sparse_out.get_shape()
            with open(meta_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {
                        "shape": [int(h_nrow), int(h_ncol)],
                        "H_nnz": int(H_sparse_out.nnz),
                        "S_nnz": int(S_sparse_out.nnz),
                        "supercell": list(supercell),
                        "remove_orbitals_unit": t_meta.get("remove_orbitals_unit", []),
                        "keep_orbitals_unit": t_meta.get("keep_orbitals_unit", []),
                        "mode": t_meta.get("mode", "sparse"),
                    },
                    fw,
                    indent=2,
                )
                fw.write("\n")
            print(f"Saved: {H_path}")
            print(f"Saved: {S_path}")
            print(f"Saved: {meta_path}")
        else:
            np.savez_compressed(
                args.output_npz,
                H=H_dense_out,
                S=S_dense_out,
                remove_orbitals_unit=np.array(t_meta.get("remove_orbitals_unit", []), dtype=np.int64),
                keep_orbitals_unit=np.array(t_meta.get("keep_orbitals_unit", []), dtype=np.int64),
            )
            print(f"Saved: {args.output_npz}")

    if args.output_dir is not None:
        keep_raw = t_meta.get("keep_orbitals_unit", [])
        if not isinstance(keep_raw, list):
            raise ValueError("Internal error: keep_orbitals_unit metadata is not a list")
        keep_orbitals_unit = [int(v) for v in keep_raw]
        if use_sparse:
            save_transformed_deeph_dataset_sparse(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                obj=ham,
                supercell=supercell,
                H_sparse_new=H_sparse_out,
                S_sparse_new=S_sparse_out,
                keep_orbitals_unit=keep_orbitals_unit,
                periodic=bool(args.periodic),
                selector_meta=selector_meta,
                rule_plan_meta=plan_meta,
            )
        else:
            save_transformed_deeph_dataset(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                obj=ham,
                supercell=supercell,
                H_dense_new=H_dense_out,
                S_dense_new=S_dense_out,
                keep_orbitals_unit=keep_orbitals_unit,
                periodic=bool(args.periodic),
                selector_meta=selector_meta,
                rule_plan_meta=plan_meta,
            )
        print(f"Saved transformed DeepH dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
