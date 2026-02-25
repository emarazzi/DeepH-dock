"""
Hamiltonian Object

This module provides the HamiltonianObj class for handling Hamiltonian and
overlap matrices, with support for diagonalization and ill-conditioned
eigenvalue handling.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from joblib import Parallel, delayed
import threadpoolctl

from deepx_dock.compute.eigen.matrix_obj import AOMatrixObj, AOMatrixK


class HamiltonianObj(AOMatrixObj):
    """
    Hamiltonian object containing both H and S matrices.

    This class extends AOMatrixObj to include the overlap matrix and provides
    methods for diagonalization with support for ill-conditioned eigenvalue
    handling.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing DeepH format files.
    H_file_path : str or Path, optional
        Path to the Hamiltonian file. Default: hamiltonian.h5 under data_path.

    Properties
    ----------
    HR : np.ndarray
        Hamiltonian matrix in real space.
    SR : np.ndarray
        Overlap matrix in real space.
    """

    def __init__(self, data_path, H_file_path=None):
        super().__init__(data_path, H_file_path)
        overlap_obj = AOMatrixObj(data_path, matrix_type="overlap")
        self.assert_compatible(overlap_obj)
        self.SR = overlap_obj.mats

    @property
    def HR(self):
        """Hamiltonian matrix in real space."""
        return self.mats

    @staticmethod
    def _r2k(ks, Rijk_list, mats):
        """
        Fourier transform from real space to reciprocal space.

        Parameters
        ----------
        ks : np.ndarray, shape (Nk, 3) or (3,)
            k-points in fractional coordinates.
        Rijk_list : np.ndarray, shape (NR, 3)
            R-vectors in fractional coordinates.
        mats : np.ndarray, shape (NR, Nb, Nb)
            Matrices in real space.

        Returns
        -------
        MKs : np.ndarray, shape (Nk, Nb, Nb)
            Matrices in reciprocal space.
        """
        phase = np.exp(2j * np.pi * np.matmul(ks, Rijk_list.T))
        MRs_flat = mats.reshape(len(Rijk_list), -1)
        Mks_flat = np.matmul(phase, MRs_flat)
        return Mks_flat.reshape(len(ks), *mats.shape[1:])

    def Sk_and_Hk(self, k):
        """
        Get overlap and Hamiltonian matrices at given k-point(s).

        Parameters
        ----------
        k : np.ndarray
            k-point(s) in fractional coordinates. Shape (3,) for single k-point
            or (Nk, 3) for multiple k-points.

        Returns
        -------
        Sk, Hk : np.ndarray
            Overlap and Hamiltonian matrices at the k-point(s).
            Shape (Nb, Nb) for single k-point or (Nk, Nb, Nb) for multiple.
        """
        if k.ndim == 1:
            ks = k[None, :]
            squeeze = True
        else:
            ks = k
            squeeze = False

        Sk = self._r2k(ks, self.Rijk_list, self.SR)
        Hk = self._r2k(ks, self.Rijk_list, self.HR)

        if squeeze:
            return Sk[0], Hk[0]
        return Sk, Hk

    def diag(
        self,
        ks,
        k_process_num: int = 1,
        thread_num: Optional[int] = None,
        sparse_calc: bool = False,
        bands_only: bool = True,
        ill_handler=None,
        kept_orbitals: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Diagonalize the Hamiltonian at specified k-points.

        Parameters
        ----------
        ks : array_like, shape (Nk, 3)
            k-points in reduced coordinates (fractional).
        k_process_num : int, optional
            Number of parallel processes. Default is 1.
        thread_num : int, optional
            Number of BLAS threads per process. Default is from environment.
        sparse_calc : bool, optional
            If True, use sparse solver (eigsh). Default is False.
        bands_only : bool, optional
            If True, only compute eigenvalues. Default is True.
        ill_handler : IllConditionedHandler, optional
            Handler for ill-conditioned eigenvalues.
        kept_orbitals : list of int, optional
            Indices of orbitals to keep (for orbital removal mode).
        **kwargs : dict
            Additional arguments passed to the solver.

        Returns
        -------
        eigvals : np.ndarray, shape (Nband, Nk)
            Eigenvalues (band energies).
        eigvecs : np.ndarray, shape (Norb, Nband, Nk), optional
            Eigenvectors (only if bands_only is False).
        """
        HR = self.HR
        SR = self.SR

        def process_k(k):
            Sk = self._r2k(k[None, :], self.Rijk_list, SR)[0]
            Hk = self._r2k(k[None, :], self.Rijk_list, HR)[0]

            # Handle ill-conditioned eigenvalues
            if ill_handler is not None:
                return ill_handler.process_k(Hk, Sk, return_vecs=not bands_only)

            # Handle orbital removal mode (direct)
            if kept_orbitals is not None:
                from deepx_dock.compute.eigen.ill_conditioned import eig_with_orbital_mask

                return eig_with_orbital_mask(Hk, Sk, kept_orbitals, return_vecs=not bands_only)

            # Standard diagonalization
            if sparse_calc:
                if bands_only:
                    vals = eigsh(Hk, M=Sk, return_eigenvectors=False, **kwargs)
                    return np.sort(vals)
                else:
                    vals, vecs = eigsh(Hk, M=Sk, **kwargs)
                    idx = np.argsort(vals)
                    return vals[idx], vecs[:, idx]
            else:
                if bands_only:
                    vals = eigh(Hk, Sk, eigvals_only=True)
                    return vals
                else:
                    vals, vecs = eigh(Hk, Sk)
                    return vals, vecs

        if thread_num is None:
            thread_num = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))

        with threadpoolctl.threadpool_limits(limits=thread_num, user_api="blas"):
            if k_process_num == 1:
                results = [process_k(k) for k in tqdm(ks, leave=False)]
            else:
                results = Parallel(n_jobs=k_process_num)(delayed(process_k)(k) for k in tqdm(ks, leave=False))

        if bands_only:
            return np.stack(results, axis=1)
        else:
            eigvals = np.stack([res[0] for res in results], axis=1)
            eigvecs = np.stack([res[1] for res in results], axis=2)
            return eigvals, eigvecs

    def get_all_Sk(self, ks, k_process_num: int = 1, thread_num: Optional[int] = None):
        """
        Get overlap matrices for all k-points.

        This method is useful for the orbital removal algorithm which needs
        all Sk matrices to determine which orbitals to remove globally.

        Parameters
        ----------
        ks : np.ndarray, shape (Nk, 3)
            k-points in fractional coordinates.
        k_process_num : int, optional
            Number of parallel processes.
        thread_num : int, optional
            Number of BLAS threads per process.

        Returns
        -------
        Sk_list : list of np.ndarray
            List of overlap matrices, each with shape (Nb, Nb).
        """
        if thread_num is None:
            thread_num = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))

        def get_Sk(k):
            return self._r2k(k[None, :], self.Rijk_list, self.SR)[0]

        with threadpoolctl.threadpool_limits(limits=thread_num, user_api="blas"):
            if k_process_num == 1:
                Sk_list = [get_Sk(k) for k in tqdm(ks, leave=False, desc="Getting Sk")]
            else:
                Sk_list = Parallel(n_jobs=k_process_num)(
                    delayed(get_Sk)(k) for k in tqdm(ks, leave=False, desc="Getting Sk")
                )

        return Sk_list
