# -*- coding: utf-8 -*-
"""
synthetic_doscar.py — σ-smeared DOSCAR writer + σ-tagged working dirs
(Module B).

Builds, for a committed σ, a synthetic DOSCAR that the UNMODIFIED
SpinAwareDosParser auto-detects as NCL and parses identically to a real
one, mirroring the example's real DOSCAR structure: ion count, per-site
column count (NCL stride-4 orbital groups), the five header lines (copied
from the real file), and the line-6 fields EMAX EMIN NEDOS EFERMI 1.0000 —
with our grid's span/length and the REAL E_F.

Energy grid: uniform absolute energies over the real DOSCAR's [EMIN, EMAX],
built by the shared _curve_grid rule (spacing min(0.005, max(σ/6, 0.001)),
point count capped at 40000); NEDOS = grid length. Energies are printed
with 5 decimals (the real file prints 3) so sub-5-meV grids at small σ
remain single-valued; the whitespace-splitting parser is unaffected.

Block 0 (total DOS): Σ_bands g_σ(E−E_band) from the PROCAR eigenvalues —
unit weight per state, NOT a sum of sphere projections — plus the
integrated-DOS column (cumulative trapezoid). Site blocks: per ion, per
orbital column, the σ-smeared sticks of the parsed PROCAR weights, the
total, mx, my, and mz column groups each filled from their own parsed
block, so the simulator's charge and magnetization displays both remain
meaningful.

Every Gaussian is evaluated by the ONE shared _gauss_sum implementation
(the per-band kernel rows below are literal _gauss_sum calls); site curves
are the weight-matrix product with those kernel rows — the same combine
rule, never a reimplementation.

σ-tagged working directory per commit: <run>/mo_ldos_work/sigma_<σ>/ holds
the synthetic DOSCAR plus every other file simulator.py reads from its
filepath — enumerated from the simulator source: CONTCAR, POSCAR, LOCPOT,
and the LOCPOT.npy cache (hardlink where available, silent fallback to
copy). The simulator's global_topo_*/ldos_topo_* caches are OUTPUTS keyed
by the directory and are deliberately NOT provisioned: a fresh directory
per σ is what guarantees the directory-keyed caches can never serve
stale-σ data (caches regenerated inside a σ-dir remain valid for that σ).
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import numpy as np

if __name__ != "__main__":
    from ._imports import _gauss_sum, _curve_grid
    from .procar_dos import ProcarData

# Files simulator.py reads from its filepath (besides DOSCAR), enumerated
# from the source: CONTCAR/POSCAR (geometry; CONTCAR preferred when
# non-empty), LOCPOT (barrier), LOCPOT.npy (LocpotManager cache). Optional
# entries are provisioned only when present in the run directory.
SIM_INPUT_FILES = ("CONTCAR", "POSCAR", "LOCPOT.npy", "LOCPOT")


def read_doscar_header(path: str) -> Dict[str, Any]:
    """
    Read ONLY the six header lines of the real DOSCAR (no full parse —
    the file may be hundreds of MB). Returns the five verbatim header
    lines plus the parsed line-6 fields; EFERMI here is the one Fermi
    energy Module B uses anywhere (same field SpinAwareDosParser reads).
    """
    lines = []
    with open(path, "r") as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: fewer than 6 header lines.")
            lines.append(line.rstrip("\n"))
    fields = lines[5].split()
    if len(fields) < 5:
        raise ValueError(f"{path}: line 6 has {len(fields)} fields, "
                         f"expected 'EMAX EMIN NEDOS EFERMI 1.0': {lines[5]!r}")
    natoms = int(lines[0].split()[0])
    return dict(
        header_lines=lines[:5],
        emax=float(fields[0]),
        emin=float(fields[1]),
        nedos=int(fields[2]),
        efermi=float(fields[3]),
        natoms=natoms,
    )


def build_synthetic_doscar(
    procar: "ProcarData",
    header: Dict[str, Any],
    sigma: float,
    out_path: str,
) -> Dict[str, Any]:
    """
    Write the σ-smeared synthetic DOSCAR to out_path. Returns the grid
    info (energies, nedos). The written file:
      line 1–5   copied verbatim from the real DOSCAR
      line 6     EMAX EMIN NEDOS EFERMI 1.0000 (our NEDOS, real E_F)
      block 0    NEDOS rows: E, total DOS, integrated DOS
      per ion    line-6 repeat + NEDOS rows: E + norb×4 columns
                 (total, mx, my, mz per orbital — stride-4, NCL layout)
    """
    if procar.nions != header["natoms"]:
        raise ValueError(
            f"PROCAR has {procar.nions} ions but the real DOSCAR header "
            f"says {header['natoms']} — refusing to build a mismatched "
            "synthetic DOSCAR.")

    emin, emax, efermi = header["emin"], header["emax"], header["efermi"]
    x = _curve_grid(emin, emax, float(sigma))
    nedos = int(x.size)
    dx = float(x[1] - x[0]) if nedos > 1 else 1.0

    energies = procar.energies.astype(float)
    nbands, nions, norb = procar.nbands, procar.nions, procar.norb

    # Per-band Gaussian kernel rows: literal _gauss_sum evaluations (unit
    # weight), so every downstream curve is a weighted sum of _gauss_sum
    # outputs — one Gaussian convention everywhere.
    G = np.empty((nbands, nedos), dtype=np.float64)
    for b in range(nbands):
        G[b, :] = _gauss_sum(x, [(float(energies[b]), 1.0)], float(sigma))

    # Block 0: total DOS (unit weight per state) + integrated column.
    total_dos = G.sum(axis=0)
    integrated = np.empty_like(total_dos)
    if nedos > 1:
        mids = 0.5 * (total_dos[1:] + total_dos[:-1]) * dx
        integrated[0] = 0.0
        integrated[1:] = np.cumsum(mids)
    else:
        integrated[:] = 0.0

    line6 = f"{emax:16.8f}{emin:16.8f}{nedos:5d}{efermi:16.8f}{1.0:16.8f}"

    blocks = (procar.total, procar.mx, procar.my, procar.mz)
    with open(out_path, "w", newline="\n") as f:
        for line in header["header_lines"]:
            f.write(line + "\n")
        f.write(line6 + "\n")
        block0 = np.column_stack([x, total_dos, integrated])
        np.savetxt(f, block0, fmt=["%11.5f", "%12.4E", "%12.4E"])
        for i in range(nions):
            # (nbands, norb*4) weight matrix, stride-4 orbital groups.
            W = np.empty((nbands, norb * 4), dtype=np.float64)
            for k, blk in enumerate(blocks):
                W[:, k::4] = blk[:, i, :]
            site = W.T @ G                      # (norb*4, nedos)
            f.write(line6 + "\n")
            rows = np.column_stack([x, site.T])
            np.savetxt(f, rows,
                       fmt=["%11.5f"] + ["%12.4E"] * (norb * 4))

    print(f"[SYNTH-DOSCAR] wrote {out_path}: NEDOS={nedos}, "
          f"sigma={sigma:.4f}, {nions} ions × {norb * 4} columns.")
    return dict(energies=x, nedos=nedos, sigma=float(sigma))


def _link_or_copy(src: str, dst: str) -> None:
    """Hardlink src to dst where available; always fall back to copy on
    failure (never assume symlinks; Windows-safe)."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def make_sigma_workdir(
    run_dir: str,
    sigma: float,
    procar: "ProcarData",
    header: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> str:
    """
    Ensure the σ-tagged working directory <run>/mo_ldos_work/sigma_<σ>/
    exists and is fully provisioned (synthetic DOSCAR + every other file
    the simulator reads). An existing directory for the SAME σ is reused
    as-is (its regenerated topo caches are valid for that σ) unless
    force=True rebuilds the synthetic DOSCAR.
    """
    if header is None:
        header = read_doscar_header(os.path.join(run_dir, "DOSCAR"))
    workdir = os.path.join(run_dir, "mo_ldos_work", f"sigma_{sigma:.4f}")
    os.makedirs(workdir, exist_ok=True)

    doscar_path = os.path.join(workdir, "DOSCAR")
    if force or not os.path.isfile(doscar_path):
        build_synthetic_doscar(procar, header, sigma, doscar_path)

    for name in SIM_INPUT_FILES:
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(workdir, name))
    return workdir
