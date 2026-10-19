# -*- coding: utf-8 -*-
"""Section 10.1 — PROCAR parser: header/ion/band count assertions, per-band
tot-consistency, |c|² internal phase consistency, spot-checks vs the raw
file."""
import os

import numpy as np


def test_counts_and_shapes(procar):
    assert procar.nkpts == 1                      # Γ-only scope asserted
    assert procar.nbands >= 1 and procar.nions >= 1
    assert procar.total.shape == (procar.nbands, procar.nions, procar.norb)
    for blk in (procar.mx, procar.my, procar.mz):
        assert blk.shape == procar.total.shape
    assert procar.energies.shape == (procar.nbands,)
    assert procar.occupancies.shape == (procar.nbands,)


def test_tot_consistency_reported(procar):
    # The parser hard-asserts per band; here we check the recorded maxima
    # sit inside the print-precision budget.
    rep = procar.validation_report
    assert rep["tot_dev_max"] <= 1.2 * 5e-4 * procar.nions
    assert rep["row_dev_max"] <= 1.2 * 5e-4 * procar.norb + 5e-4


def test_phase_internal_consistency(procar):
    # Approved amendment: |c|² is checked against the phase block's OWN
    # totals (trailing per-ion column and the charge row) — exact at print
    # precision. The interval-projector scale vs the total block is
    # informational only and must NOT be ~1 in general.
    rep = procar.validation_report
    assert rep["phase_ion_dev_max"] <= 1.2 * 5e-4 * (2 * procar.norb) + 5e-4
    assert rep["phase_charge_dev_max"] <= 1.2 * 5e-4 * procar.nions + 5e-4
    lo, hi = procar.phase_scale_range
    assert np.isfinite(lo) and np.isfinite(hi) and 0 < lo <= hi


def test_spot_checks_vs_raw_file(procar, example_dir):
    # Direct line-level comparison against the raw file for band 1 and a
    # mid-file band (values transcribed by an independent read).
    path = os.path.join(example_dir, "PROCAR")
    with open(path) as f:
        lines = f.read().splitlines()
    lines_per_band = 1 + 1 + 1 + 4 * (procar.nions + 1) + 1 + procar.nions + 2
    for b in (0, procar.nbands // 2, procar.nbands - 1):
        base = 5 + b * lines_per_band
        fields = lines[base].split()
        assert float(fields[4]) == procar.energies[b]
        assert float(fields[7]) == procar.occupancies[b]
        # first ion row of the total block
        row = lines[base + 3].split()
        np.testing.assert_array_equal(
            np.asarray(row[1:procar.norb + 1], dtype=float),
            procar.total[b, 0, :])
        # 'tot' row
        trow = lines[base + 3 + procar.nions].split()
        np.testing.assert_array_equal(
            np.asarray(trow[1:], dtype=float), procar.tot_rows[b])


def test_energies_match_eigenval(procar, example_dir):
    path = os.path.join(example_dir, "EIGENVAL")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        lines = f.read().splitlines()
    nb = int(lines[5].split()[2])
    assert nb == procar.nbands
    first = float(lines[8].split()[1])
    assert abs(first - procar.energies[0]) < 5e-6
