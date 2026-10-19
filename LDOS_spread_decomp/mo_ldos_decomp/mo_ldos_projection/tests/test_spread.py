# -*- coding: utf-8 -*-
"""Section 10.6 — spread codes: combined curve equals the sum of the two
previously separate curves (pre-normalization), sticks sum per full state,
HOMO-relative selection maps correctly, non-SOC component ⇒ singles."""
import os
import sys

import numpy as np
import pytest

# The spread codes live in the MOPmatcher_mem matcher package (two levels up,
# next to the wrapper folder); the _imports shim has already made them importable.
from mo_ldos_projection import _imports  # noqa: F401  (path setup)

try:
    from mo_ldos_projection.mopm.matcher import spread_collapse as sc
    from mo_ldos_projection.mopm.matcher import spread_plot as spl
except ImportError:
    from matcher import spread_collapse as sc
    from matcher import spread_plot as spl

from mo_ldos_projection._imports import PlotConfig, pair_entities


@pytest.fixture(scope="module")
def spread_plotter(example_dir):
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    _, sp = sc.plot_spread(
        output_dir=example_dir,
        components=["NHC_c1m1", "NHC_c2m1"],
        comp_indices=[-1, 0, 1],
        cfg=cfg,
    )
    return sp


def test_combined_curve_equals_sum_of_member_curves(spread_plotter):
    """Pre-normalization: g(combined weights) == g(w_a) + g(w_b)."""
    sp = spread_plotter
    ent = sp._entity_for("NHC_c1m1", 0)
    assert ent is not None and len(ent.indices) == 2
    a, b = ent.indices
    per_member = {a: [], b: []}
    combined = []
    for full_idx, comps in sp.by_full.items():
        s = {a: 0.0, b: 0.0}
        E = None
        found = False
        for rec in comps.get("NHC_c1m1", []):
            if rec["comp_idx"] in (a, b):
                E = rec["E"] + rec["dE"]
                s[rec["comp_idx"]] += rec["ov"]
                found = True
        if found:
            per_member[a].append((E, s[a]))
            per_member[b].append((E, s[b]))
            combined.append((E, s[a] + s[b]))
    x = sc._curve_grid(-25.0, 8.0, 0.2)
    y_comb = sc._gauss_sum(x, combined, 0.2)
    y_sum = sc._gauss_sum(x, per_member[a], 0.2) + \
        sc._gauss_sum(x, per_member[b], 0.2)
    np.testing.assert_allclose(y_comb, y_sum, rtol=1e-12, atol=1e-12)


def test_sticks_sum_per_full_state(spread_plotter, results):
    """The assembled bar weight per full state equals the sum of the two
    member grid rows (at txt print precision)."""
    sp = spread_plotter
    ent = sp._entity_for("NHC_c1m1", 0)
    rows = sum(results["grid"]["NHC_c1m1"][i - 1, :] for i in ent.indices)
    for full_idx, comps in sp.by_full.items():
        s = None
        for rec in comps.get("NHC_c1m1", []):
            if rec["comp_idx"] in ent.indices:
                s = (s or 0.0) + rec["ov"]
        if s is not None:
            assert abs(s - rows[full_idx - 1]) < 2.5e-5  # 2 × txt rounding


def test_homo_relative_selection_maps_correctly(spread_plotter, prepared):
    P, cfg, _ = prepared
    sp = spread_plotter
    for label in ("NHC_c1m1", "NHC_c2m1"):
        homo = P.known_homos[label]
        ent0 = sp._entity_for(label, 0)
        assert homo in ent0.indices
        ent_lumo = sp._entity_for(label, 1)
        assert min(ent_lumo.indices) > max(ent0.indices)


def test_not_found_offset_titles(example_dir):
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    fig, axes = spl.plot_spread(
        output_dir=example_dir, comp_label="NHC_c1m1",
        comp_indices=[0, 999], cfg=cfg)
    titles = [ax.get_title() for ax in axes]
    assert any("not found" in t for t in titles)
    assert any("HOMO (" in t for t in titles)


def test_non_soc_component_singles_unchanged():
    """Synthetic non-SOC input: entities are singles, and the combined
    weight of a single equals the raw weight (behavior unchanged)."""
    pairs = [(1, -2.0), (2, -1.0), (3, 0.5)]
    ents = pair_entities(pairs, False, 0.001)
    assert all(len(e.indices) == 1 for e in ents)
    G = np.array([[0.3, 0.1], [0.2, 0.4], [0.05, 0.15]])
    from mo_ldos_projection._imports import combine_pair_weights
    W = combine_pair_weights(G, ents)
    np.testing.assert_array_equal(W, G)


def test_multi_run_mo_row_one_stick_per_entity(example_dir):
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    _, msp = sc.plot_spread_multi(
        runs=[(example_dir, ["NHC_c1m1", "NHC_c2m1"])],
        comp_indices=[-1, 0, 1],
        cfg=cfg,
    )
    # one averaged energy per requested entity offset
    assert set(msp.mo_energies.keys()) == {-1, 0, 1}
    sp0 = msp.plotters[0]
    for off, E in msp.mo_energies.items():
        vals = []
        for comp in ("NHC_c1m1", "NHC_c2m1"):
            ent = sp0._entity_for(comp, off)
            vals.append(ent.E_mean + sp0.fermi_shift)
        assert abs(E - float(np.mean(vals))) < 1e-9
