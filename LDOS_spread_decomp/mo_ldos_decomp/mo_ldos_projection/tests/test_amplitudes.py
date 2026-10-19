# -*- coding: utf-8 -*-
"""Section 10.3 — amplitudes: |grid_amp|² == grid (rtol 1e-6) per label;
S_MO within-component blocks ≈ I; diagnostics present and finite."""
import numpy as np


def test_grid_amp_squares_to_grid(results):
    assert results["extended"], "extended matcher record required"
    for label in results["comp_labels"]:
        g = results["grid"][label]
        a = results["grid_amp"][label]
        np.testing.assert_allclose(np.abs(a) ** 2, g, rtol=1e-6, atol=1e-30)


def test_s_mo_within_component_blocks_near_identity(results):
    S = results["S_MO"]
    mo_axis = results["mo_axis"]
    labels = [e[0] for e in mo_axis]
    # block boundaries per component (mo_axis is grouped by label)
    bounds = {}
    for i, l in enumerate(labels):
        lo, hi = bounds.get(l, (i, i))
        bounds[l] = (min(lo, i), max(hi, i))
    for l, (lo, hi) in bounds.items():
        blk = S[lo:hi + 1, lo:hi + 1]
        dev = np.max(np.abs(blk - np.eye(blk.shape[0])))
        # measured 1.34e-2 on the example; 5e-2 is the acceptance bound
        assert dev < 5e-2, (l, dev)
    # Hermitian to machine precision
    assert np.max(np.abs(S - S.conj().T)) < 1e-12


def test_s_mm_gram_properties(results):
    S_mm = results["S_mm"]
    n = S_mm.shape[0]
    assert S_mm.shape == (n, n)
    assert np.max(np.abs(np.diag(S_mm).real - 1.0)) < 1e-10
    assert np.max(np.abs(S_mm - S_mm.conj().T)) < 1e-12


def test_diagnostics_present_and_finite(results):
    d = results["stored_diagnostics"]
    for key in ("S_MO_dev_max", "S_MO_dev_fro", "S_mm_dev_max",
                "S_mm_dev_fro", "min_eig_S_mt"):
        assert key in d and np.isfinite(d[key]), key
    assert d["min_eig_S_mt"] > 0.0     # linear-dependence guard holds


def test_b_shape_and_mo_axis(results):
    B = results["B"]
    mo_axis = results["mo_axis"]
    n_metal = results["grid_amp"]["metal"].shape[0]
    assert B.shape == (n_metal, len(mo_axis))
    # mo_axis: molecule components in comp_labels order, bands ascending
    labels = [l for l in results["comp_labels"] if l != "metal"]
    expect = [[l, i + 1]
              for l in labels
              for i in range(results["grid_amp"][l].shape[0])]
    assert mo_axis == expect
