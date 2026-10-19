#!/usr/bin/env python3
"""
wavedat_npz_writer.py

Parses WAVECAR + EIGENVAL, caches to .npz, and optionally
generates synthetic orthonormal coefficient sets for unit tests.

This patched version adds a `force` flag to always re-parse and
overwrite any existing cache file when desired.
"""

import os
import numpy as np
from pymatgen.io.vasp.outputs import Wavecar


class WavecarParser:
    """
    Parses WAVECAR, extracts:
      • Fermi Energy
      • Number of k-points
      • Number of bands
      • kpts:     float ndarray (nk_actual, 3)
      • energies: list of eigen energies and occupancy for each k-point
      • Gpoints:  ragged list of integer G-vectors for each block
      • coeffs:   ragged list of coefficient arrays per block

    Can save a flat-buffer cache (.npz), replace coefficients with all-ones,
    or generate per-k-point delta-function orthonormal coefficients.

    Added options:
      • parse(..., force=False)  – skip loading cache if force=True
      • run(...,   force=False)  – propagate `force` into parse()
    """

    def __init__(self, directory: str, vasp_version="s"):
        self.directory    = os.path.abspath(directory)
        self.vasp_version = vasp_version
        self.wavecar_pth  = os.path.join(self.directory, "WAVECAR")
        self.cache_pth    = os.path.join(self.directory, "wavecar_cache.npz")

        # placeholders, set in parse() or load_cache()
        self.ef          = None
        self.nk          = None     # total blocks (spin × k)
        self.nb          = None
        self.kpoints     = None     # (nk_actual, 3)
        self.energies    = None     # (nspins, nk_actual, nb) or (nk_actual, nb)
        self.occupancies = None
        self.Gpoints     = None     # list of length nk; each array is (nG_k, 3)
        self.coeffs      = None     # list of length nk; each is list of nb arrays

    def parse(self, vasp_version, force=False):
        """
        Parse WAVECAR into internal buffers or load from cache if present.

        Parameters
        ----------
        vasp_version : str
            's' or 'n' style precision for parsing spinor data.
        force : bool
            If True, ignore any existing cache and re-parse WAVECAR.
        """
        # Only load existing cache if it exists _and_ we're not forcing a rewrite
        if os.path.exists(self.cache_pth) and not force:
            self.load_cache()
            return

        if not os.path.isfile(self.wavecar_pth):
            raise FileNotFoundError(f"WAVECAR not found at {self.wavecar_pth}")

        wave = Wavecar(self.wavecar_pth, precision="a", vasp_type=vasp_version)

        # --- Basic scalars and energies ---
        self.ef          = wave.efermi
        self.nb          = wave.nb
        energies_all     = np.array(wave.band_energy)
        self.energies    = energies_all[..., 0]
        self.occupancies = energies_all[..., 1]
        self.kpoints     = np.array(wave.kpoints)

        # --- Detect ISPIN and flatten coeffs / Gpoints across (spin × k) ---
        coeffs_obj = wave.coeffs
        if isinstance(coeffs_obj[0][0], np.ndarray):
            # ISPIN = 1
            nspin        = 1
            nk_per_spin  = len(coeffs_obj)
            spin_blocks  = [coeffs_obj]
        else:
            # ISPIN = 2
            nspin        = len(coeffs_obj)
            nk_per_spin  = len(coeffs_obj[0])
            spin_blocks  = coeffs_obj

        self.nk = nk_per_spin * nspin

        self.Gpoints = []
        self.coeffs  = []
        for s in range(nspin):
            for k in range(nk_per_spin):
                Gpts_k = np.array(wave.Gpoints[k])
                nG_k   = Gpts_k.shape[0]
                self.Gpoints.append(Gpts_k)

                coeffs_k = []
                for b in range(self.nb):
                    arr = np.array(spin_blocks[s][k][b])
                    if vasp_version == "n":
                        expected = nG_k * 2
                        if arr.size != expected:
                            raise ValueError(
                                f"(k={k}, spin={s}, band={b}): size {arr.size}, expected {expected}"
                            )
                        arr = arr.reshape(nG_k, 2)
                    else:
                        if arr.size != nG_k:
                            raise ValueError(
                                f"(k={k}, spin={s}, band={b}): size {arr.size}, expected {nG_k}"
                            )
                        arr = arr.reshape(nG_k,)
                    coeffs_k.append(arr)

                self.coeffs.append(coeffs_k)

        # Write out new cache
        self.save_cache()

    def run(self, force=False):
        """
        One-shot parser + summary + cache writer.

        Parameters
        ----------
        force : bool
            If True, ignore any existing cache and re-parse WAVECAR.
        """
        # propagate the force flag into parse()
        self.parse(self.vasp_version, force=force)
        self.save_gpoints_txt()
        self.summary()
        return (
            self.ef,
            self.nk,
            self.kpoints,
            self.nb,
            self.energies,
            self.occupancies,
            self.Gpoints,
            self.coeffs,
        )

    def save_gpoints_txt(self, filename=None):
        """
        Write all G-points to a tab-separated text file.
        One block at a time, with a header.
        """
        if filename is None:
            filename = os.path.join(self.directory, "Gpoints.txt")

        with open(filename, "w") as f:
            for blk, gpts in enumerate(self.Gpoints):
                f.write(f"# Block {blk}  nG={gpts.shape[0]}\n")
                np.savetxt(f, gpts, fmt="%d", delimiter="\t")
                f.write("\n")
        print(f"Wrote G-points for {len(self.Gpoints)} blocks to {filename}")

    def summary(self):
        """
        Print out dimensions and consistency checks of the parsed data.
        """
        print("\nParsed Files Summary")
        print("--------------------")
        print(f" Directory      : {self.directory}")
        print(f" Blocks (spin×k): {self.nk}")
        print(f" Bands per block: {self.nb}")

        if self.Gpoints:
            gcounts = [g.shape[0] for g in self.Gpoints]
            if min(gcounts) == max(gcounts):
                print(f" G-vectors      : {gcounts[0]}")
            else:
                print(f" G-vectors      : min={min(gcounts)}, max={max(gcounts)}")
        else:
            print(" G-vectors      : (none)")

        if self.energies is not None:
            print(f" Energies shape : {np.shape(self.energies)}")
        else:
            print(" Energies shape : (none)")
        print()

    def save_cache(self):
        """
        Save WAVECAR-derived data to a flat-buffer NPZ for quick reload.
        """
        # Flatten Gpoints
        g_counts = np.array([g.shape[0] for g in self.Gpoints], dtype=np.int32)
        g_off    = np.concatenate(([0], np.cumsum(g_counts)))
        G_flat   = np.vstack(self.Gpoints).astype(np.int32)

        # Flatten coeffs
        coeff_blocks = [np.stack(self.coeffs[k], axis=0) for k in range(self.nk)]
        c_counts     = np.array([blk.size for blk in coeff_blocks], dtype=np.int64)
        c_off        = np.concatenate(([0], np.cumsum(c_counts)))
        C_flat       = np.concatenate([blk.ravel() for blk in coeff_blocks])

        arrays = {
            "ef":            np.float64(self.ef),
            "nk":            np.int32(self.nk),
            "nb":            np.int32(self.nb),
            "kpoints":       self.kpoints,
            "energies":      self.energies,
            "occupancies":   self.occupancies,
            "Gpoints_flat":  G_flat,
            "Gpoints_off":   g_off,
            "Gpoints_cnt":   g_counts,
            "coeffs_flat":   C_flat,
            "coeffs_off":    c_off,
            "coeffs_cnt":    c_counts,
            "coeffs_dtype":  np.array(str(C_flat.dtype)),
        }

        # record spinor columns
        arrays["spinor_cols"] = np.int32(
            coeff_blocks[0].shape[-1] if coeff_blocks and coeff_blocks[0].ndim == 3 else 1
        )

        np.savez(self.cache_pth, **arrays)
        print(f"Flat-buffer cache saved to {self.cache_pth}")

    def load_cache(self):
        """
        Load previously saved flat-buffer NPZ and rebuild ragged arrays.
        """
        data = np.load(self.cache_pth, allow_pickle=True)

        self.ef          = float(data["ef"])
        self.nk          = int(data["nk"])
        self.nb          = int(data["nb"])
        self.kpoints     = data["kpoints"]
        self.energies    = data["energies"]
        self.occupancies = data["occupancies"]

        # restore ragged Gpoints
        G_flat = data["Gpoints_flat"]
        g_off  = data["Gpoints_off"]
        self.Gpoints = [G_flat[g_off[k]:g_off[k+1]] for k in range(self.nk)]

        # restore ragged coeffs
        C_flat = data["coeffs_flat"]
        c_off  = data["coeffs_off"]
        ncols  = int(data.get("spinor_cols", 1))

        self.coeffs = []
        for k in range(self.nk):
            blk = C_flat[c_off[k]:c_off[k+1]]
            nGk = self.Gpoints[k].shape[0]
            if ncols > 1:
                coeffs_k = blk.reshape(self.nb, nGk, ncols)
            else:
                coeffs_k = blk.reshape(self.nb, nGk)
            self.coeffs.append([coeffs_k[b] for b in range(self.nb)])

        print(f"Cache loaded from {self.cache_pth} ({self.nk} blocks)")

    @staticmethod
    def make_allones_coeffs_npz(real_npz_path,
                                out_path=None,
                                key_match=('coeffs',),
                                compressed=False,
                                verbose=True):
        """
        Copy an NPZ and replace complex coefficient arrays with ones.
        """
        if not os.path.isfile(real_npz_path):
            raise FileNotFoundError(f"No such file: {real_npz_path}")

        with np.load(real_npz_path, allow_pickle=True) as data:
            arrays = {k: data[k] for k in data.files}

        def _matches(k):
            return any(k.startswith(p) or (p in k) for p in key_match)

        replaced = []
        for k, arr in arrays.items():
            if _matches(k) and np.iscomplexobj(arr):
                arrays[k] = np.ones_like(arr)
                replaced.append(k)

        if verbose:
            print(f"[allones] replaced {len(replaced)} arrays: {replaced}")

        if out_path is None:
            base, ext = os.path.splitext(real_npz_path)
            out_path = f"{base}_allones{ext if ext == '.npz' else '.npz'}"

        save = np.savez_compressed if compressed else np.savez
        save(out_path, **arrays)
        if verbose:
            print(f"[allones] wrote {out_path}")
        return out_path, replaced

    @staticmethod
    def make_delta_coeffs_npz(real_npz_path: str,
                              out_path: str = None,
                              key_off='coeffs_off',
                              key_cnt='coeffs_cnt',
                              key_dtype='coeffs_dtype',
                              key_spinor='spinor_cols',
                              compressed: bool = False,
                              verbose: bool = True):
        """
        Create a copy of an NPZ where, within each k-block,
        the nb bands form a perfect delta-function orthonormal set.

        Gram matrix → identity per k-point with exact 1's on diag, 0 off-diag.
        """
        if not os.path.isfile(real_npz_path):
            raise FileNotFoundError(f"No such file: {real_npz_path}")

        with np.load(real_npz_path, allow_pickle=True) as data:
            arrays = {k: data[k] for k in data.files}

        c_off  = arrays[key_off]
        c_cnt  = arrays[key_cnt]
        dtype  = np.dtype(arrays[key_dtype].item())
        ncols  = int(arrays.get(key_spinor, 1))
        nb     = int(arrays["nb"])
        nk_blk = len(c_cnt)

        new_blocks = []
        for k in range(nk_blk):
            block_elems    = c_cnt[k]
            elems_per_band = block_elems // nb
            nGk            = elems_per_band // ncols

            for b in range(nb):
                v   = np.zeros(elems_per_band, dtype=dtype)
                idx = b % nGk
                if ncols == 1:
                    v[idx] = 1.0
                else:
                    v[idx * ncols] = 1.0
                new_blocks.append(v)

        new_flat       = np.concatenate(new_blocks).astype(dtype, copy=False)
        arrays["coeffs_flat"] = new_flat

        if out_path is None:
            base, ext = os.path.splitext(real_npz_path)
            out_path = f"{base}_delta{ext if ext == '.npz' else '.npz'}"

        save = np.savez_compressed if compressed else np.savez
        save(out_path, **arrays)
        if verbose:
            print(f"[delta] wrote per-k delta basis NPZ to {out_path}")
        return out_path


if __name__ == "__main__":
    # Example one-click run (Spyder-friendly)
    directory    = r'dir/run'
    vasp_version = 's'
    parser       = WavecarParser(directory, vasp_version)

    # 1) parse & write the real cache + G-points txt + summary (force overwrite)
    parser.run(force=True)

    # 2) generate a per-k delta-basis NPZ that yields identity overlaps
    WavecarParser.make_delta_coeffs_npz(
        real_npz_path=parser.cache_pth,
        out_path=None,
        compressed=False,
        verbose=True
    )