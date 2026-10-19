# -*- coding: utf-8 -*-
"""Sections 10.7 and 10.9 — decomposition-window invariants and layout
toggles: Σ kept drawn areas == A_LDOS (rtol 1e-10); common-α check;
threshold filtering; empty-set annotation; export/reload reconstructs the
drawn curves via the header α; all four layout modes render with identical
underlying data; Amplitude is display-only; LDOS-only empty cells and
dropped components; lightness offsets on multi-component axes."""
import numpy as np
import pytest

from mo_ldos_projection.decomp_window import (
    DecompWindow, MO_AREA_NOISE_THRESHOLD)
from mo_ldos_projection._imports import PlotConfig

_TRAPZ = getattr(np, "trapezoid", np.trapz)


def _snapshot(sigma=0.2):
    eg = np.arange(-3.0, 0.5 + 1e-9, 0.005)
    ld1 = (0.6 * np.exp(-((eg + 1.0) / 0.5) ** 2)
           + 0.2 * np.exp(-((eg + 2.2) / 0.3) ** 2) + 0.01)
    ld2 = 0.9 * np.exp(-((eg + 0.4) / 0.4) ** 2) + 0.02
    return dict(sigma=sigma, eg=eg, markers=[(1.0, 2.0), (3.0, 4.0)],
                ldos=[ld1, ld2], marker_colors=["#1f77b4", "#2ca02c"])


@pytest.fixture(scope="module")
def window(example_dir):
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    return DecompWindow(example_dir, _snapshot(), cfg=cfg)


def test_default_toggle_states(window):
    assert window.collapse_vertical and window.collapse_horizontal
    assert not window.amplitude
    assert window.threshold == MO_AREA_NOISE_THRESHOLD


def test_sum_kept_areas_equals_a_ldos_probability_view(window):
    """β normalization: the kept curves' summed drawn area equals A_LDOS
    in the probability view, with relative weights set by the projection
    (the magnitude view is a pointwise sqrt — its areas deliberately do
    not obey the identity)."""
    for m in range(len(window.markers)):
        c = window.proj_coeffs[m]
        a = np.array([r["a_ent"] for r in window.kept])
        drawn_prob = window.beta_prob[m] * float(np.sum(c ** 2 * a))
        assert abs(drawn_prob - window.A_ldos[m]) <= 1e-10 * window.A_ldos[m]


def test_projection_is_normalized_overlap_vs_full_ldos(window):
    """The coefficients are independent normalized overlaps against the
    FULL unit LDOS: p_a = ⟨L̂, Ŵ_a⟩. The metal curve is the per-marker
    composite of ALL metal-state spreads, each q²-weighted by its own
    overlap with THIS LDOS — so its shape differs between markers — and
    is drawn as the LDOS's projection component ‖L‖·p_met·Ĉ_m."""
    design = np.column_stack([r["W"] for r in window.kept])
    w_norms = np.linalg.norm(design, axis=0)
    shapes = []
    for m, y in enumerate(window.ldos_curves):
        l_norm = np.linalg.norm(y)
        L_hat = y / l_norm
        q = (window.metal_spreads @ L_hat) / window.metal_norms
        composite = (q ** 2) @ window.metal_spreads
        C_hat = composite / np.linalg.norm(composite)
        shapes.append(C_hat)
        pm = float(L_hat @ C_hat)
        assert abs(pm - window.p_met[m]) < 1e-12
        np.testing.assert_allclose(window.metal_curves[m],
                                   l_norm * pm * C_hat,
                                   rtol=1e-12, atol=1e-15)
        ref = (design.T @ L_hat) / w_norms
        np.testing.assert_allclose(window.proj_coeffs[m], ref,
                                   rtol=1e-12, atol=1e-15)
        assert np.all(window.proj_coeffs[m] > 0.0)
    assert not np.allclose(window.proj_coeffs[0], window.proj_coeffs[1])
    # the metal composite SHAPE follows the LDOS: markers differ
    assert not np.allclose(shapes[0], shapes[1])


def test_drawn_ratios_follow_projection(window):
    """Probability view: drawn-area ratios equal (p²·a_ent) ratios."""
    eg = window.eg
    c = window.proj_coeffs[0]
    nz = [k for k in range(len(window.kept)) if c[k] > 0]
    if len(nz) < 2:
        pytest.skip("need two nonzero coefficients")
    k0, k1 = nz[0], nz[1]
    r0, r1 = window.kept[k0], window.kept[k1]
    window.amplitude = False
    prob_ratio = (_TRAPZ(window._row_factor(0, k0) * r0["W"], eg)
                  / _TRAPZ(window._row_factor(0, k1) * r1["W"], eg))
    expect = (c[k0] ** 2 * r0["a_ent"]) / (c[k1] ** 2 * r1["a_ent"])
    assert abs(prob_ratio - expect) < 1e-9 * abs(expect)


def test_layouts_render_without_recompute(window):
    kept_ids = [(r["comp"], r["offset"]) for r in window.kept]
    W_ref = {kid: window.kept[i]["W"].copy()
             for i, kid in enumerate(kept_ids)}
    c_ref = [c.copy() for c in window.proj_coeffs]
    b_ref = list(window.beta_prob)
    axes_counts = {}
    for v, h in ((True, True), (True, False), (False, True), (False, False)):
        window.collapse_vertical, window.collapse_horizontal = v, h
        window._build_ui()
        axes_counts[(v, h)] = len(
            [a for a in window.fig.axes if a.get_subplotspec() is not None])
        for i, kid in enumerate(kept_ids):
            np.testing.assert_array_equal(window.kept[i]["W"], W_ref[kid])
        for m in range(len(window.markers)):
            np.testing.assert_array_equal(window.proj_coeffs[m], c_ref[m])
        assert window.beta_prob == b_ref
    n_mark = len(window.markers)
    R, C = len(window.kept_offsets), len(window.kept_comps)
    assert axes_counts[(True, True)] == n_mark
    assert axes_counts[(True, False)] == n_mark * C
    assert axes_counts[(False, True)] == n_mark * R
    assert axes_counts[(False, False)] == n_mark * R * C
    window.collapse_vertical = window.collapse_horizontal = True
    window._build_ui()


def test_block_axis_areas_sum_to_a_ldos_in_every_mode(window):
    """Per block, Σ of MO-curve areas over ALL of the block's axes equals
    A_LDOS in every non-Amplitude layout (each kept entity drawn once;
    the sqrt view deliberately breaks the identity)."""
    window.amplitude = False
    for v, h in ((True, True), (True, False),
                 (False, True), (False, False)):
        window.collapse_vertical, window.collapse_horizontal = v, h
        window._build_ui()
        n_mark = len(window.markers)
        data_axes = [a for a in window.fig.axes
                     if a.get_subplotspec() is not None]
        per_block = len(data_axes) // n_mark
        for m in range(n_mark):
            block_axes = data_axes[m * per_block:(m + 1) * per_block]
            total = 0.0
            for ax in block_axes:
                for line in ax.get_lines():
                    lbl = line.get_label()
                    if lbl.startswith("LDOS") or lbl.startswith("metal"):
                        continue
                    total += _TRAPZ(line.get_ydata(), line.get_xdata())
            assert abs(total - window.A_ldos[m]) <= \
                1e-9 * window.A_ldos[m]
    window.collapse_vertical = window.collapse_horizontal = True
    window._build_ui()


def test_ldos_on_every_axis_same_scale(window):
    window._build_ui()
    data_axes = [a for a in window.fig.axes
                 if a.get_subplotspec() is not None]
    for m, ax in enumerate(data_axes):
        ldos_lines = [l for l in ax.get_lines()
                      if l.get_label().startswith("LDOS")]
        assert ldos_lines, "LDOS missing from an axis"
        np.testing.assert_array_equal(
            ldos_lines[0].get_ydata(), window.ldos_curves[m])


def test_amplitude_is_pointwise_sqrt_of_probability_view(window):
    """Amplitude ON: every drawn curve — the LDOS AND each MO spread
    curve — is the pointwise sqrt of its probability-view values, so
    each MO curve is scaled by |p| times the shared sqrt shape."""
    window.collapse_vertical = window.collapse_horizontal = True
    window.amplitude = True
    window._build_ui()
    data_axes = [a for a in window.fig.axes
                 if a.get_subplotspec() is not None]
    ldos_line = [l for l in data_axes[0].get_lines()
                 if l.get_label().startswith("LDOS")][0]
    np.testing.assert_allclose(
        ldos_line.get_ydata(), np.sqrt(np.abs(window.ldos_curves[0])),
        rtol=1e-12)
    mo_lines = [l for l in data_axes[0].get_lines()
                if not l.get_label().startswith("LDOS")
                and not l.get_label().startswith("metal")]
    # In the collapsed view the kept entities draw in kept order, so the
    # first MO line is kept[0] (legend labels now carry p²/p, not the
    # band composition, so a label lookup no longer applies).
    line0 = mo_lines[0]
    k0 = 0
    prob = (window.beta_prob[0] * window.proj_coeffs[0][k0] ** 2
            * window.kept[k0]["W"])
    np.testing.assert_allclose(line0.get_ydata(), np.sqrt(prob),
                               rtol=1e-12)
    # equivalently: scaled by |p| times the shared sqrt shape
    p = window.proj_coeffs[0][k0]
    np.testing.assert_allclose(
        line0.get_ydata(),
        abs(p) * np.sqrt(window.beta_prob[0] * window.kept[k0]["W"]),
        rtol=1e-12)
    window.amplitude = False
    window._build_ui()


def test_export_probability_scale_regardless_of_toggle(window):
    window.amplitude = True
    window._build_ui()
    p_amp = window.save_data()
    with open(p_amp, encoding="utf-8") as f:
        amp_txt = f.read()
    window.amplitude = False
    window._build_ui()
    p_prob = window.save_data()
    with open(p_prob, encoding="utf-8") as f:
        prob_txt = f.read()
    # exports identical except the toggle-state header line
    strip = lambda t: [l for l in t.splitlines()
                       if not l.startswith("# TOGGLES")]
    assert strip(amp_txt) == strip(prob_txt)


def test_threshold_filter_and_empty_annotation(window):
    old = window.threshold
    window.threshold = 1e9
    window._recompute_kept()
    window._build_ui()
    assert window.kept == []
    annos = [t.get_text() for a in window.fig.axes for t in a.texts]
    assert any("no MOs above threshold" in t for t in annos)
    window.threshold = old
    window._recompute_kept()
    window._build_ui()
    assert len(window.kept) > 0


def test_export_reconstructs_both_drawn_views(window):
    path = window.save_data()
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    beta_prob = None
    coeffs0 = None
    for ln in lines:
        if ln.startswith("# MARKER 0:"):
            beta_prob = float(
                ln.split("beta_prob=")[1].split(";")[0])
        if ln.startswith("# PROJ marker 0:"):
            coeffs0 = [float(part.rsplit("=", 1)[1])
                       for part in ln.split(":", 1)[1].split(";")]
    assert beta_prob is not None and coeffs0 is not None
    i0 = lines.index("# BLOCK marker 0\n")
    body = []
    for ln in lines[i0 + 2:]:
        if ln.startswith("#"):
            break
        body.append([float(x) for x in ln.split()])
    arr = np.array(body)
    # first kept entity, unscaled (after E, LDOS, metal component)
    W0 = arr[:, 3]
    recon_prob = beta_prob * coeffs0[0] ** 2 * W0
    np.testing.assert_allclose(
        recon_prob, window._row_factor(0, 0) * window.kept[0]["W"],
        atol=1e-9)
    # magnitude view = pointwise sqrt of the reconstructed probability view
    np.testing.assert_allclose(
        np.sqrt(recon_prob),
        np.sqrt(window._row_factor(0, 0) * window.kept[0]["W"]),
        atol=1e-9)


def test_lightness_offsets_on_multi_component_axes(window):
    """Horizontal-only mode: axes hold both components' entities at one
    offset — their drawn colors must differ by the lightness offset."""
    window.collapse_vertical, window.collapse_horizontal = False, True
    window._build_ui()
    # find an offset kept for BOTH components
    both = [off for off in window.kept_offsets
            if sum(1 for r in window.kept if r["offset"] == off) >= 2]
    if not both:
        pytest.skip("no offset kept for both components in this run")
    off = both[0]
    rows = [r for r in window.kept if r["offset"] == off]
    c0 = window._comp_color(rows[0], True)
    c1 = window._comp_color(rows[1], True)
    assert c0 != c1
    window.collapse_vertical = window.collapse_horizontal = True
    window._build_ui()
