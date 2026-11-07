# -*- coding: utf-8 -*-
"""
procar_dos.py — LORBIT=14 noncollinear PROCAR parser (Module B).

Parses the full system's PROCAR (VASP 6.5.2, SOC/noncollinear, LORBIT=14,
single Γ k-point) in the phase-augmented "new format" layout, verified
against the example calculation before this parser was written:

    line 1   title: 'PROCAR lm decomposed + phase; opt. projector ...'
    line 2   '# of k-points:  N   # of bands:  NB   # of ions:  NI'
    line 3   blank
    line 4   k-point line (coords + weight)
    line 5   blank
    then per band, a block of exactly
        1 line   'band  N # energy  E # occ.  O'
        1 blank
        1 line   orbital header 'ion  s py pz px ... tot'
        4×(NI+1) projection rows: the total, mx, my, and mz blocks, each
                 NI ion rows + one 'tot' row, with NO separator lines
        1 line   phase header 'ion  s py ... dx2-y2' (no tot column)
        NI       phase rows: ion + (Re, Im) per orbital + per-ion Σ|c|²
        1 line   'charge' row: per-orbital Σ_ions |c|² + total
        1 blank

Extracted quantities per (band, ion, orbital-column, block): the real
projection values of the total/mx/my/mz blocks — the same kind of quantity
as the MOPM ov values; these are what all downstream consumers use.

Validation (hard, all asserted against the file being read):
  * header counts vs actual block structure and file length;
  * k-point count == 1 (Γ-only scope);
  * per band: Σ_{ions} (total block) ≈ the file's own 'tot' row, and each
    ion row's 'tot' column ≈ the row's orbital sum (print-precision
    tolerances);
  * phase sub-block internal consistency (approved amendment): the file's
    per-ion trailing column ≈ Σ_orb |c|² formed from the Re/Im pairs, and
    the 'charge' row ≈ Σ_ions |c|² per orbital. The phase block uses
    VASP's interval-optimized projector, so |c|² does NOT equal the
    total-block values (band-dependent scale); that ratio is reported as
    informational, never asserted.

Energies are raw eigenvalues; the Fermi energy comes from the real DOSCAR
header only (see synthetic_doscar.read_doscar_header) — no other alignment.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Print-precision tolerances of the PROCAR fixed-point columns: projection
# values carry 3 decimals (rounding error ≤ 5e-4 per value), so column sums
# over NI values may deviate by up to NI·5e-4 from the printed row of sums;
# the factor 1.2 is slack for the sums' own rounding.
_VAL_ROUND = 5e-4


@dataclass
class ProcarData:
    """Parsed LORBIT=14 NCL PROCAR content (single Γ k-point)."""
    path: str
    nkpts: int
    nbands: int
    nions: int
    orbital_names: List[str]
    energies: np.ndarray            # (nbands,) raw eigenvalues (eV)
    occupancies: np.ndarray         # (nbands,)
    # Projection blocks, each (nbands, nions, norb) float64:
    total: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    mz: np.ndarray
    # Per-band totals from the file's own 'tot' rows, (nbands, norb + 1):
    tot_rows: np.ndarray
    phase_scale_range: tuple = (float("nan"), float("nan"))
    validation_report: Dict[str, float] = field(default_factory=dict)

    @property
    def norb(self) -> int:
        return len(self.orbital_names)


def parse_procar(path: str, spot_check_bands: int = 12) -> ProcarData:
    """
    Parse and validate the PROCAR at `path`. All structural assumptions are
    asserted at runtime against the file; any mismatch fails loudly.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()

    if len(lines) < 8:
        raise ValueError(f"{path}: too short to be a PROCAR ({len(lines)} lines).")
    title = lines[0]
    if "phase" not in title:
        raise ValueError(
            f"{path}: title line lacks 'phase' — not the LORBIT>=12 "
            f"phase-augmented format: {title!r}")

    m = re.match(
        r"#\s*of\s*k-points:\s*(\d+)\s+#\s*of\s*bands:\s*(\d+)\s+"
        r"#\s*of\s*ions:\s*(\d+)", lines[1].strip())
    if not m:
        raise ValueError(f"{path}: unrecognized counts line: {lines[1]!r}")
    nkpts, nbands, nions = (int(g) for g in m.groups())
    if nkpts != 1:
        raise ValueError(
            f"{path}: # of k-points = {nkpts}; only single Γ-point PROCARs "
            "are in scope — multi-k inputs are not supported.")

    if not lines[3].lstrip().startswith("k-point"):
        raise ValueError(f"{path}: expected k-point line at line 4, got "
                         f"{lines[3]!r}")

    # Orbital header of band 1 fixes the orbital-column layout.
    header_line = lines[7]
    header_fields = header_line.split()
    if header_fields[0] != "ion" or header_fields[-1] != "tot":
        raise ValueError(f"{path}: unexpected orbital header: {header_line!r}")
    orbital_names = header_fields[1:-1]
    norb = len(orbital_names)

    # Exact per-band block length, then verify the total line count.
    lines_per_band = 1 + 1 + 1 + 4 * (nions + 1) + 1 + nions + 1 + 1
    expected = 5 + nbands * lines_per_band
    if len(lines) not in (expected, expected + 1):  # optional trailing blank
        raise ValueError(
            f"{path}: file has {len(lines)} lines; expected "
            f"{expected} (= 5 + {nbands} bands × {lines_per_band}) for "
            f"{nions} ions and {norb} orbitals.")

    energies = np.empty(nbands)
    occupancies = np.empty(nbands)
    blocks = {name: np.empty((nbands, nions, norb))
              for name in ("total", "mx", "my", "mz")}
    block_names = ("total", "mx", "my", "mz")
    tot_rows = np.empty((nbands, norb + 1))

    band_re = re.compile(
        r"band\s+(\d+)\s*#\s*energy\s+([-\d.]+)\s*#\s*occ\.\s+([-\d.]+)")

    tot_dev_max = 0.0
    row_dev_max = 0.0
    phase_ion_dev_max = 0.0
    phase_charge_dev_max = 0.0
    scale_lo, scale_hi = np.inf, -np.inf

    spot_every = max(1, nbands // max(spot_check_bands, 1))

    for b in range(nbands):
        base = 5 + b * lines_per_band
        mb = band_re.match(lines[base].strip())
        if not mb or int(mb.group(1)) != b + 1:
            raise ValueError(
                f"{path}: band header mismatch at line {base + 1}: "
                f"{lines[base]!r} (expected band {b + 1}).")
        energies[b] = float(mb.group(2))
        occupancies[b] = float(mb.group(3))
        if lines[base + 2].split() != header_fields:
            raise ValueError(
                f"{path}: orbital header changed at band {b + 1}.")

        # ---- four projection blocks: NI ion rows + one 'tot' row each ----
        row = base + 3
        for k, name in enumerate(block_names):
            body = lines[row:row + nions]
            tokens = " ".join(body).split()
            if len(tokens) != nions * (norb + 2):
                raise ValueError(
                    f"{path}: band {b + 1} {name} block: expected "
                    f"{nions * (norb + 2)} fields, got {len(tokens)}.")
            arr = np.asarray(tokens, dtype=np.float64).reshape(
                nions, norb + 2)
            if not np.array_equal(arr[:, 0].astype(int),
                                  np.arange(1, nions + 1)):
                raise ValueError(
                    f"{path}: band {b + 1} {name} block: ion indices are "
                    "not 1..NI in order.")
            vals = arr[:, 1:norb + 1]
            blocks[name][b] = vals
            trow = lines[row + nions].split()
            if trow[0] != "tot":
                raise ValueError(
                    f"{path}: band {b + 1} {name} block: expected 'tot' row "
                    f"at line {row + nions + 1}, got {lines[row + nions]!r}")
            tvals = np.asarray(trow[1:], dtype=np.float64)
            if name == "total":
                tot_rows[b] = tvals
                # Hard validation: per-orbital column sums and the grand
                # total against the file's own 'tot' row.
                col_sums = vals.sum(axis=0)
                atol_cols = 1.2 * _VAL_ROUND * nions
                dev = float(np.max(np.abs(col_sums - tvals[:norb])))
                grand = float(abs(arr[:, norb + 1].sum() - tvals[norb]))
                tot_dev_max = max(tot_dev_max, dev, grand)
                if dev > atol_cols or grand > atol_cols:
                    raise ValueError(
                        f"{path}: band {b + 1}: Σ_ions(total block) deviates "
                        f"from the file 'tot' row by {max(dev, grand):.4g} "
                        f"(allowed {atol_cols:.4g}).")
                # Spot-check: each ion row's own 'tot' column vs its
                # orbital sum (print precision).
                if b % spot_every == 0:
                    rd = float(np.max(np.abs(
                        vals.sum(axis=1) - arr[:, norb + 1])))
                    row_dev_max = max(row_dev_max, rd)
                    if rd > 1.2 * _VAL_ROUND * norb + _VAL_ROUND:
                        raise ValueError(
                            f"{path}: band {b + 1}: ion-row 'tot' column "
                            f"deviates from its orbital sum by {rd:.4g}.")
            row += nions + 1

        # ---- phase sub-block: header + NI rows + charge row ----
        ph_header = lines[row].split()
        if ph_header[0] != "ion" or len(ph_header) != norb + 1:
            raise ValueError(
                f"{path}: band {b + 1}: unexpected phase header "
                f"{lines[row]!r} (want 'ion' + {norb} orbitals).")
        if b % spot_every == 0:
            body = lines[row + 1:row + 1 + nions]
            tokens = " ".join(body).split()
            if len(tokens) != nions * (2 * norb + 2):
                raise ValueError(
                    f"{path}: band {b + 1} phase block: expected "
                    f"{nions * (2 * norb + 2)} fields, got {len(tokens)}.")
            ph = np.asarray(tokens, dtype=np.float64).reshape(
                nions, 2 * norb + 2)
            if not np.array_equal(ph[:, 0].astype(int),
                                  np.arange(1, nions + 1)):
                raise ValueError(
                    f"{path}: band {b + 1} phase block: ion indices are "
                    "not 1..NI in order.")
            re_part = ph[:, 1:2 * norb:2]
            im_part = ph[:, 2:2 * norb + 1:2]
            c2 = re_part ** 2 + im_part ** 2          # (nions, norb)
            trailing = ph[:, 2 * norb + 1]
            # Internal phase consistency (approved amendment): |c|² vs the
            # phase block's OWN per-ion totals and 'charge' row.
            dev_ion = float(np.max(np.abs(c2.sum(axis=1) - trailing)))
            phase_ion_dev_max = max(phase_ion_dev_max, dev_ion)
            if dev_ion > 1.2 * _VAL_ROUND * (2 * norb) + _VAL_ROUND:
                raise ValueError(
                    f"{path}: band {b + 1}: phase per-ion Σ|c|² deviates "
                    f"from the trailing column by {dev_ion:.4g}.")
            crow = lines[row + 1 + nions].split()
            if crow[0] != "charge":
                raise ValueError(
                    f"{path}: band {b + 1}: expected 'charge' row, got "
                    f"{lines[row + 1 + nions]!r}")
            cvals = np.asarray(crow[1:], dtype=np.float64)
            dev_charge = float(np.max(np.abs(
                c2.sum(axis=0) - cvals[:norb])))
            phase_charge_dev_max = max(phase_charge_dev_max, dev_charge)
            if dev_charge > 1.2 * _VAL_ROUND * nions + _VAL_ROUND:
                raise ValueError(
                    f"{path}: band {b + 1}: phase 'charge' row deviates "
                    f"from Σ_ions |c|² by {dev_charge:.4g}.")
            # Informational: interval-projector scale vs the total block.
            band_tot = float(tot_rows[b][norb])
            band_c2 = float(cvals[norb])
            if band_c2 > 1e-3 and band_tot > 1e-3:
                ratio = band_tot / band_c2
                scale_lo = min(scale_lo, ratio)
                scale_hi = max(scale_hi, ratio)

    report = dict(
        tot_dev_max=tot_dev_max,
        row_dev_max=row_dev_max,
        phase_ion_dev_max=phase_ion_dev_max,
        phase_charge_dev_max=phase_charge_dev_max,
    )
    data = ProcarData(
        path=path, nkpts=nkpts, nbands=nbands, nions=nions,
        orbital_names=orbital_names,
        energies=energies, occupancies=occupancies,
        total=blocks["total"], mx=blocks["mx"], my=blocks["my"],
        mz=blocks["mz"], tot_rows=tot_rows,
        phase_scale_range=(float(scale_lo), float(scale_hi)),
        validation_report=report,
    )
    print(f"[PROCAR] {os.path.basename(path)}: {nbands} bands, {nions} ions, "
          f"{norb} orbitals, 1 k-point. Validation: "
          f"tot dev {tot_dev_max:.2e}, row dev {row_dev_max:.2e}, "
          f"phase ion dev {phase_ion_dev_max:.2e}, "
          f"phase charge dev {phase_charge_dev_max:.2e}. "
          f"Interval-projector scale (total/|c|², informational): "
          f"[{scale_lo:.3f}, {scale_hi:.3f}].")
    return data
