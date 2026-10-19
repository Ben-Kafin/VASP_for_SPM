# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:33:07 2025

@author: Benjamin Kafin
"""

import os
import csv
import numpy as np

from wavecar_parser_force import WavecarParser
from gpoint_checker_gpts import GpointChecker


class WavefunctionOrthonormality:
    """
    Build |<ψ_m | ψ_n>|^2 overlap matrices from WAVECAR coefficients,
    but first weed out any periodic‐facet duplicates in the G‐point grid.
    """

    def __init__(self,
                 directory,
                 vasp_version="s",
                 cache=True,
                 use_delta=False,
                 normalize=False,
                 out_npz=None):
        self.directory    = directory
        self.vasp_version = vasp_version
        self.cache        = cache
        self.use_delta    = use_delta
        self.normalize    = normalize
        self.out_npz      = out_npz or os.path.join(directory, "overlaps.npz")

        # 1) Parse & cache WAVECAR
        self.parser = WavecarParser(directory, vasp_version)
        # you can still pass force=True here if you patched the parser
        self.parser.run()

        if self.use_delta:
            delta_npz = WavecarParser.make_delta_coeffs_npz(
                real_npz_path=self.parser.cache_pth,
                out_path=None,
                compressed=False,
                verbose=True
            )
            self.parser.cache_pth = delta_npz
            self.parser.load_cache()

        # 2) Run GpointChecker and filter coefficients
        gpt_path = os.path.join(directory, "Gpoints.txt")
        self.gpc = GpointChecker(gpt_path)
        self.gpc.remove_facet_duplicates()
        self.removed_indices   = self.gpc.removed_indices   # dict spin -> removed row indices (the gpts checker returns masks)
        self.filtered_gpoints  = self.gpc.filtered_blocks
        self._filter_coeffs()

    @staticmethod
    def _wavefunction_norm(vec):
        """
        Normalize a vector by the absolute value squared of a dot product with itself
        """
        flat = vec.ravel()
        norm = np.vdot(flat,flat)
        nvec = flat / np.sqrt(norm)
        return nvec


    def _filter_coeffs(self):
        """
        Delete the duplicated‐facet G‐points from every band-array in each block
        according to self.removed_indices.
        """
        new_coeffs = []
        for blk_idx, bands in enumerate(self.parser.coeffs):
            to_remove = self.removed_indices.get(blk_idx, np.array([],int))
            if to_remove.size == 0:
                new_coeffs.append(bands)
                continue

            filtered_bands = []
            for arr in bands:
                filtered = np.delete(arr, to_remove, axis=0)
                filtered_bands.append(filtered)
            new_coeffs.append(filtered_bands)

        self.parser.coeffs = new_coeffs

    def _block_coeff_matrix(self, block_idx):
        bands = self.parser.coeffs[block_idx]
        first = bands[0]

        if first.ndim == 1:
            C = np.vstack(bands).astype(np.complex128, copy=False)
        elif first.ndim == 2 and first.shape[1] == 2:
            C = np.vstack([b.reshape(-1) for b in bands])\
                  .astype(np.complex128, copy=False)
        else:
            raise ValueError(f"Block {block_idx}: unsupported shape {first.shape}")

        return C

    def compute(self, return_S=False):
        """
        Compute per-block overlap matrices, optionally normalizing each
        wavefunction first.

        Parameters
        ----------
        return_S : bool
            If True, also return the complex overlap matrices S.

        Returns
        -------
        overlap2 : ndarray, shape (n_blocks, nb, nb)
            Squared magnitudes |<ψ_m|ψ_n>|^2 for each block.
        S_all : ndarray, shape (n_blocks, nb, nb), optional
            Complex overlap matrices <ψ_m|ψ_n> (only if return_S=True).
        """
        n_blocks = len(self.parser.coeffs)
        nb       = self.parser.nb

        overlap2 = np.empty((n_blocks, nb, nb), dtype=np.float64)
        S_all    = np.empty((n_blocks, nb, nb), dtype=np.complex128) if return_S else None

        for blk in range(n_blocks):
            # Build the coefficient matrix for this block
            C = self._block_coeff_matrix(blk)

            # Optionally normalize each band-vector to unit Hermitian norm
            if self.normalize:
                for i in range(C.shape[0]):
                    C[i, :] = self._wavefunction_norm(C[i, :])

            # Compute the Hermitian overlap matrix
            S = C @ C.conj().T

            if return_S:
                S_all[blk] = S

            # Store squared magnitudes
            overlap2[blk] = np.abs(S)**2

        return (overlap2, S_all) if return_S else overlap2

    def save(self, include_S=False, write_tsv=False):
        """
        Save overlaps to an .npz, and optionally output TSVs of
        both squared magnitudes |<ψ|ψ>|^2 and absolute values |<ψ|ψ>|.
        
        Parameters
        ----------
        include_S : bool
            If True, also store the complex overlap matrix S in the NPZ.
        write_tsv : bool
            If True, write TSV files for both squared (overlap2) and
            absolute (abs(S)) values per block.
        """
        # Always compute both S and overlap2
        overlap2, S_all = self.compute(return_S=True)

        # 1) Save to NPZ
        arrays = {"overlap2": overlap2}
        if include_S:
            arrays["S"] = S_all

        # metadata
        shp = np.shape(self.parser.energies)
        if len(shp) == 3:
            nspins, nkpts, _ = shp
        elif len(shp) == 2:
            nspins, nkpts = 1, shp[0]
        else:
            nspins = getattr(self.parser, "nspin", 1)
            nkpts  = getattr(self.parser, "nk", self.parser.nk)

        meta = {
            "nb":             np.int32(self.parser.nb),
            "n_blocks":       np.int32(len(self.parser.coeffs)),
            "nspins":         np.int32(nspins),
            "nkpts_physical": np.int32(nkpts),
            "vasp_version":   np.array(self.vasp_version)
        }
        arrays.update(meta)

        np.savez(self.out_npz, **arrays)

        # 2) Optionally write TSVs
        if write_tsv:
            headers = [""] + [str(i) for i in range(self.parser.nb)]

            # squared overlaps (|<ψ|ψ>|^2)
            for blk in range(overlap2.shape[0]):
                tsv_sq = os.path.join(
                    self.directory, f"overlap_block{blk}_sq.tsv"
                )
                with open(tsv_sq, "w", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(headers)
                    for i in range(self.parser.nb):
                        row = [str(i)] + [
                            f"{overlap2[blk, i, j]:.8e}"
                            for j in range(self.parser.nb)
                        ]
                        w.writerow(row)

            # absolute overlaps (|<ψ|ψ>|)
            for blk in range(S_all.shape[0]):
                tsv_abs = os.path.join(
                    self.directory, f"overlap_block{blk}_abs.tsv"
                )
                with open(tsv_abs, "w", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(headers)
                    for i in range(self.parser.nb):
                        row = [str(i)] + [
                            f"{abs(S_all[blk, i, j]):.8e}"
                            for j in range(self.parser.nb)
                        ]
                        w.writerow(row)

        return self.out_npz

    def summary_filtered(self):
        """
        Print a summary of original vs. filtered G-point & coefficient sizes.
        """
        print("===== G-point Filtering Summary =====")

        for i, Gf in enumerate(self.filtered_gpoints):
            removed = self.removed_indices.get(i, np.array([],int))
            orig = Gf.shape[0] + removed.size
            print(
                f" Block {i}: "
                f"original G-points = {orig}, "
                f"removed = {removed.size}, "
                f"remaining = {Gf.shape[0]}"
            )

        print("Now running orthonormality on filtered coefficients...\n")

        overlap2 = self.compute(return_S=False)
        print("Overlap2 shape per block:", overlap2.shape)
        print("Maximum |<m|n>|^2 (off-diagonal):")
        for blk in range(overlap2.shape[0]):
            # ignore diagonal
            offdiag = overlap2[blk] - np.eye(overlap2.shape[1])
            print(f"  Block {blk}: {offdiag.max():.3e}")

        print("======================================\n")


if __name__ == "__main__":
    directory = r"dir/run"
    wf = WavefunctionOrthonormality(
        directory,
        vasp_version="s",
        cache=True,
        use_delta=False,
        normalize=True,
        out_npz=None
    )
    wf.summary_filtered()
    wf.save(write_tsv=True)