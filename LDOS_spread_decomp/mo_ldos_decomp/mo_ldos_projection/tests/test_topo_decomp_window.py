# -*- coding: utf-8 -*-
"""Topo-decomposition-window invariants: per-grid-point projections are
independent normalized overlaps of the topo LDOS curves; the input
offsets list (spread_collapse convention) alone decides the MO set (no
threshold); every cell is ONE single-color layer (the MO's color,
opacity ramped from transparent at the reference minimum intensity to
opaque at the reference maximum — no black, nothing beneath);
Amplitude/Norm are display-only; the ramp anchors per-MO (Norm off) or
on the shared range (Norm on); every drawn cell carries a colorbar in
the same single-color fade graduated over the reference range of the
plotted quantity; export saves raw p; refresh follows snapshot
semantics; unknown offsets draw "not found"."""
import numpy as np
import pytest

from mo_ldos_projection.topo_decomp_window import TopoDecompWindow
from mo_ldos_projection.decomp_window import _offset_name
from mo_ldos_projection._imports import PlotConfig

# An offset no molecule component can have (far beyond the band window):
# draws as a "not found" cell in every column.
_MISSING = 500


def _snapshot(sigma=0.2, bias=0.2, n=5):
    """Synthetic topo snapshot: an n×n grid over an orthogonal 10×10 Å
    cell with smooth positive per-point LDOS curves over [0, bias]."""
    eg = np.arange(0.0, bias + 1e-9, 0.005)
    xs, ys = np.meshgrid(np.linspace(0.0, 10.0, n),
                         np.linspace(0.0, 10.0, n))
    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    t = np.linspace(0.0, 1.0, n * n)
    ldos = (0.05
            + np.outer(t, np.exp(-((eg - 0.05) / 0.03) ** 2))
            + np.outer(1.0 - t, np.exp(-((eg - 0.15) / 0.04) ** 2)))
    lv = np.diag([10.0, 10.0, 20.0])
    return dict(sigma=sigma, eg=eg, grid_xy=grid_xy, ldos=ldos,
                topo_bias=bias, topo_height=2.5, lv=lv)


@pytest.fixture(scope="module")
def window(example_dir):
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    return TopoDecompWindow(example_dir, _snapshot(),
                            offsets=[-1, 0, 1, _MISSING], cfg=cfg)


def _data_axes(win):
    return [a for a in win.fig.axes if a.get_subplotspec() is not None]


def _found(win):
    return [c for c in win.cells if c["p"] is not None]


def test_default_toggle_states(window):
    assert not window.amplitude
    assert not window.norm_shared


def test_cell_grid_is_offsets_by_components(window):
    R = 4                                  # [-1, 0, 1, _MISSING]
    C = len(window.mol_labels)
    assert len(window.cells) == R * C
    assert len(_data_axes(window)) == R * C
    # input order preserved, row-major (offset rows × component columns)
    expect = [(o, c) for o in [-1, 0, 1, _MISSING]
              for c in window.mol_labels]
    assert [(c["offset"], c["comp"]) for c in window.cells] == expect


def test_missing_offset_draws_not_found(window):
    C = len(window.mol_labels)
    assert all(c["p"] is None for c in window.cells[-C:])
    annos = [t.get_text() for a in _data_axes(window)[-C:] for t in a.texts]
    assert annos and all("not found" in t for t in annos)


def test_projection_is_normalized_overlap_per_grid_point(window):
    """p_a(x,y) = ⟨L̂(x,y), Ŵ_a⟩ — every curve unit-normalized first,
    each entity projected independently (LDOS-decomp scaling rule)."""
    snap = window.snapshot
    L = np.asarray(snap["ldos"], dtype=np.float64)
    L_hat = L / np.linalg.norm(L, axis=1)[:, None]
    for cell in _found(window):
        W = cell["W"]
        ref = L_hat @ (W / np.linalg.norm(W))
        np.testing.assert_allclose(cell["p"], ref, rtol=1e-12, atol=1e-15)
        assert cell["p"].shape == (window.grid_xy.shape[0],)


def test_toggles_are_display_only(window):
    p_ref = [c["p"].copy() for c in _found(window)]
    W_ref = [c["W"].copy() for c in _found(window)]
    for amp, norm in ((True, False), (True, True),
                      (False, True), (False, False)):
        window.amplitude, window.norm_shared = amp, norm
        window._build_ui()
        for c, p0, W0 in zip(_found(window), p_ref, W_ref):
            np.testing.assert_array_equal(c["p"], p0)
            np.testing.assert_array_equal(c["W"], W0)


def test_intensity_matches_amplitude_toggle(window):
    cell = _found(window)[0]
    window.amplitude = False
    np.testing.assert_allclose(window._intensity(cell), cell["p"] ** 2)
    window.amplitude = True
    np.testing.assert_allclose(window._intensity(cell),
                               np.abs(cell["p"]))
    window.amplitude = False


def test_alpha_ramp_anchors_per_mo_and_shared(window):
    """Norm OFF: every (non-flat) map's alpha field spans exactly [0, 1]
    — full per-panel contrast, transparent at its OWN minimum, opaque at
    its OWN maximum. Norm ON: one shared [min, max] range — every field
    is the same affine map of its intensities."""
    window.amplitude = False
    window.norm_shared = False
    ranges = {}
    for cell in _found(window):
        I = window._intensity(cell)
        ranges[id(cell)] = (float(np.min(I)), float(np.max(I)))
        if ranges[id(cell)][1] > ranges[id(cell)][0]:
            f = window._alpha_field(cell)
            assert abs(float(np.min(f)) - 0.0) < 1e-12
            assert abs(float(np.max(f)) - 1.0) < 1e-12
    g_lo = min(r[0] for r in ranges.values())
    g_hi = max(r[1] for r in ranges.values())
    assert g_hi > g_lo
    window.norm_shared = True
    for cell in _found(window):
        I = window._intensity(cell)
        f = window._alpha_field(cell)
        np.testing.assert_allclose(
            f, np.clip((I - g_lo) / (g_hi - g_lo), 0.0, 1.0),
            rtol=0, atol=1e-12)
    window.norm_shared = False


def test_single_layer_mo_color_alpha_ramp(window):
    """Each found cell holds ONE tricontourf layer whose bands all share
    the MO's exact fill color and differ only in opacity (ascending with
    the field level — the Illustrator-recolorable construction), with no
    background patch beneath and a transparent axes background; not-found
    cells hold nothing."""
    window._build_ui()
    # transparent figure patch: exports carry no opaque background rect
    assert window.fig.patch.get_facecolor()[3] == 0.0
    axes = _data_axes(window)
    for cell, ax in zip(window.cells, axes):
        if cell["p"] is None:
            assert len(ax.patches) == 0
            assert len(ax.collections) == 0
            continue
        assert len(ax.patches) == 0          # no background layer
        assert len(ax.collections) >= 1      # the single color layer
        assert ax.get_facecolor()[3] == 0.0  # transparent axes patch
        fc = np.asarray(ax.collections[0].get_facecolor())
        assert fc.shape[0] >= 2
        np.testing.assert_allclose(
            fc[:, :3], np.tile(cell["row"]["color"], (fc.shape[0], 1)),
            rtol=0, atol=1e-12)
        alphas = fc[:, 3]
        assert np.all(np.diff(alphas) >= -1e-12)   # ascending opacity
        assert alphas[0] < 0.05 and alphas[-1] > 0.95


def test_save_data_exports_raw_p(window, tmp_path):
    # Redirect the export into the isolated temp dir (save_data always
    # writes <run_dir>/topo_decomp_data.txt) so no residue lands in the
    # shared example run directory.
    run_dir_orig = window.run_dir
    window.run_dir = str(tmp_path)
    try:
        path = window.save_data()
    finally:
        window.run_dir = run_dir_orig
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    header = [l for l in lines if l.startswith("#")]
    body = np.array([[float(x) for x in l.split()]
                     for l in lines if not l.startswith("#")])
    found = _found(window)
    assert body.shape == (window.grid_xy.shape[0], 2 + len(found))
    np.testing.assert_allclose(body[:, 0], window.grid_xy[:, 0])
    for k, cell in enumerate(found):
        np.testing.assert_allclose(body[:, 2 + k], cell["p"], atol=1e-9)
    # not-found cells are named in the header but get no column
    assert any("not found" in l for l in header)
    assert any(_offset_name(_MISSING) in l for l in header)


def test_refresh_snapshot_semantics(window):
    """refresh() recomputes the projections from the new press."""
    p_before = [c["p"].copy() for c in _found(window)]
    snap2 = _snapshot()
    snap2["ldos"] = snap2["ldos"][::-1].copy()   # different curves
    window.refresh(snap2)
    changed = any(not np.allclose(c["p"], p0)
                  for c, p0 in zip(_found(window), p_before))
    assert changed
    window.refresh(_snapshot())


def test_empty_offsets_rejected(example_dir):
    with pytest.raises(ValueError):
        TopoDecompWindow(example_dir, _snapshot(), offsets=[])


# ======================================================================
# Per-cell colorbars (the same single-color fade, graduated over the
# reference range the opacity ramp anchors on)
# ======================================================================
def test_each_drawn_cell_has_matching_colorbar(window):
    """Every drawn cell gets a colorbar in ITS MO's single-color fade,
    graduated over exactly the reference range the opacity ramp anchors
    on; "not found" cells get none; data axes stay the R×C grid
    (colorbar axes are subplotspec-free)."""
    window.amplitude = False
    window.norm_shared = False
    window._build_ui()
    assert len(window.cbars) == len(window.cells)
    assert len(_data_axes(window)) == len(window.cells)
    for cell, cbar in zip(window.cells, window.cbars):
        if cell["p"] is None:
            assert cbar is None
            continue
        assert cbar is not None
        lo, hi = window._reference_range(cell)
        assert cbar.mappable.norm.vmin == lo
        assert cbar.mappable.norm.vmax == hi
        top = np.asarray(cbar.mappable.cmap(1.0), dtype=float)
        bot = np.asarray(cbar.mappable.cmap(0.0), dtype=float)
        np.testing.assert_allclose(top[:3], cell["row"]["color"],
                                   atol=1e-12)
        np.testing.assert_allclose(bot[:3], cell["row"]["color"],
                                   atol=1e-12)
        assert top[3] == 1.0 and bot[3] == 0.0
        assert cbar.ax.get_subplotspec() is None


def test_colorbars_share_range_under_norm(window):
    """Norm ON: every colorbar graduates the ONE shared range."""
    window.norm_shared = True
    window._build_ui()
    found = _found(window)
    intensities = [window._intensity(c) for c in found]
    g_lo = min(float(np.min(I)) for I in intensities)
    g_hi = max(float(np.max(I)) for I in intensities)
    bars = [b for b in window.cbars if b is not None]
    assert bars
    for cbar in bars:
        assert cbar.mappable.norm.vmin == g_lo
        assert cbar.mappable.norm.vmax == g_hi
    window.norm_shared = False
    window._build_ui()


def test_colorbar_label_tracks_view(window):
    window.amplitude = False
    assert window._scale_label() == "p²"
    window.amplitude = True
    assert window._scale_label() == "|p|"
    window.amplitude = False
