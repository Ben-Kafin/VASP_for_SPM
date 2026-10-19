# -*- coding: utf-8 -*-
"""
state_decomposition_plot.py

@author: Benjamin Kafin

For a given MOPMatcher run, plots how each full-system Bloch state decomposes
into its contributing component states, flanked by "original states" reference
panels showing every component eigenstate at its own energy.

Kramers combining is always on, gated only by the SOC detection exactly like
the color pairing (kramers.pair_entities — the same walk the colors use).
Every consumer operates on combined entities: the original-states panels draw
one line per entity at its E_mean in the rep member's color, the spread
stacks/segments carry one segment per entity with the member overlaps
pair-summed per full state, the tail threshold competes entity weights, and
the curve groups pool entity contributions. Non-SOC runs: entities are
singles and behavior is unchanged.

LAYOUT -- the figure is ONE ROW of panels with ENERGY VERTICAL (shared energy
axis: tick marks on every panel, tick VALUES and the energy label only on the
leftmost). Left -> right, for the systems present:

    [metal states] [metal spreads] [molecule spreads] [molecule states]

* "states" panels (outer): every original component eigenstate (from
  comp_pairs -- ALL states, not just the thresholded contributors) drawn as a
  full-width horizontal line at its own energy, in the canonical MOPMatcher
  color of that state. These panels have NO intensity tick marks or values,
  but still carry the intensity axis label.
* "spreads" panels (inner): each full state appears as a horizontal stack at
  its full-system energy. Segment length = that match's individual overlap
  probability (ov); segment color = the canonical MOPMatcher color of the
  contributing component state. Segments in one state's stack are sorted by
  comp_idx ascending (with comp_label as the tiebreaker for the
  multi-molecule case), the literal key into component_colors[comp_label] --
  invariant under any energy shift. Intensity axis has tick marks AND values.
  ORIENTATION: each spreads panel's intensity ZERO sits on the edge FACING
  its own states panel -- metal: zero at the LEFT edge growing right;
  molecule: zero at the RIGHT edge growing left (reversed axis direction,
  tick values stay positive) -- so the two spreads panels grow toward each
  other.

The plotted set is selected by a per-full-state cumulative-tail threshold
on the pool across BOTH systems: for each full state, sort all contributions
ascending by ov and drop from the smallest upward while the cumulative
dropped probability mass remains <= sum_threshold; stop the first time
including the next ov would push the dropped mass past sum_threshold.

A Curve CheckButton (lower-left) replaces each SPREADS panel (the outer
states panels stay put) with that system's single total curve: the system's
summed Gaussian-broadened intensity over all its contributing component
states (VASP-style smearing,
g = exp(-((E-E0)/sigma)^2) / (sigma*sqrt(pi))), drawn in the same
orientation as the stick panel it replaces (intensity zero on the edge
facing its states panel). Both curve panels share ONE intensity maximum so
joint normalization keeps the metal-vs-molecule relative intensity readable.
Each curve's COLOR varies along its length: at energy E it is the
intensity-weighted blend of the contributing component states' canonical
colors,

    color(E) = sum_i c_i w_i(E) / sum_i w_i(E),
    w_i(E)   = sum_j ov_ij * g(E - E_j)   (state i's broadened weight at E)

rendered as a per-segment-colored LineCollection (consecutive grid points
become tiny segments, each colored by the blend at that energy). The color
weights come from the RAW broadened intensities, so the blend is unaffected
by the normalization and amplitude transforms below. The in-figure slider
sets sigma in eV live.

CURVE PIPELINE -- ORDER OF OPERATIONS (keep this order in any future edit):

  1. combine    Each system's total probability curve sum_i w_i(E). This is
                the ONLY combination rule; amplitudes are never turned into
                Gaussians.
  2. normalize  The Norm sub-toggle (visible in curve mode) picks the rule:
                  off (default) -> JOINT: one shared factor so the SUM of
                                   the system curves has unit area,
                                   preserving metal-vs-molecule relative
                                   intensity
                  on            -> each system curve divided by its OWN
                                   integral (unit area each)
                Any future toggle that reshapes the curve must slot in
                BEFORE this step.
  3. amplitude  THE FINAL STEP, after normalization: with Amplitude on the
                displayed curve is sqrt(normalized curve). Nothing is
                re-normalized afterward.

In stick mode (Curve off) the stacked-segment drawing is unchanged apart
from the rotation; Amplitude there keeps the stack additive in PROBABILITY:
the cumulative segment boundaries are passed through the square root, so the
end of each stack is sqrt(total) rather than a sum of square roots.
"""
from __future__ import annotations
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import CheckButtons, Slider

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

if __name__ != "__main__":
    from .vacupot_plotter import RectAEPAWColorPlotter, PlotConfig
    from .kramers import pair_entities

# Curve-mode smearing slider range (eV), VASP SIGMA convention
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0


def _bands_str(indices) -> str:
    """'band 74' for a single, 'bands 74+75' for a Kramers pair."""
    idx = list(indices)
    if len(idx) == 1:
        return f"band {idx[0]}"
    return "bands " + "+".join(str(i) for i in idx)

# np.trapz was renamed np.trapezoid in NumPy 2.0
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _gauss_sum(x: np.ndarray, states: List[Tuple[float, float]],
               sigma: float) -> np.ndarray:
    """
    VASP-style Gaussian smearing of a set of (E0, w) sticks onto the grid x:
    each stick contributes w * exp(-((x - E0)/sigma)^2) / (sigma * sqrt(pi)),
    so the area under each Gaussian equals its weight w and sigma is the
    smearing width exactly as VASP's SIGMA. Each Gaussian is only evaluated
    within +-8 sigma of its center (exp(-64) ~ 1e-28, so the truncation is
    exact to machine precision) to keep slider drags responsive.
    """
    y = np.zeros_like(x)
    norm = 1.0 / (sigma * np.sqrt(np.pi))
    for E0, w in states:
        lo = np.searchsorted(x, E0 - 8.0 * sigma)
        hi = np.searchsorted(x, E0 + 8.0 * sigma)
        if hi <= lo:
            continue
        y[lo:hi] += (w * norm) * np.exp(-(((x[lo:hi] - E0) / sigma) ** 2))
    return y


def _system_curve(x: np.ndarray,
                  groups: List[Tuple[Tuple[float, ...], List[Tuple[float, float]]]],
                  sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 1 (combine) plus the color field for ONE system's total curve.

    groups is [(rgba, states), ...]: one entry per contributing component
    state, rgba its canonical color and states its (E_full, ov) pairs.
    Returns (total, colors): the RAW summed probability curve
    sum_i w_i(x), with w_i(x) = sum_j ov_ij g(x - E_j), and the per-point
    blended RGB color(x) = sum_i rgb_i w_i(x) / sum_i w_i(x).

    The blend uses the RAW broadened weights, so the colors are independent
    of the normalization and amplitude transforms applied to the heights
    afterward (steps 2 and 3 of the pipeline in the module docstring).
    """
    total = np.zeros_like(x)
    num = np.zeros((x.size, 3))
    for rgba, sts in groups:
        w = _gauss_sum(x, sts, sigma)
        total += w
        num += w[:, None] * np.asarray(rgba[:3])[None, :]
    colors = num / np.maximum(total, 1e-300)[:, None]
    return total, np.clip(colors, 0.0, 1.0)


def _gradient_segments(xs: np.ndarray, ys: np.ndarray,
                       colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment array + per-segment RGBA for a LineCollection rendering of a
    per-point-colored curve: consecutive grid points become tiny segments,
    each colored by the mean of its endpoints' blended colors. Orientation
    agnostic: xs/ys are the PLOTTED horizontal/vertical coordinates (in the
    rotated layout the intensity is horizontal and the energy vertical).
    """
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_rgb = 0.5 * (colors[:-1] + colors[1:])
    seg_rgba = np.concatenate(
        [seg_rgb, np.ones((seg_rgb.shape[0], 1))], axis=1)
    return segs, seg_rgba


def _curve_grid(lo: float, hi: float, sigma: float) -> np.ndarray:
    """
    Ascending energy grid for the broadened curves over [lo, hi]. Spacing
    adapts to the current sigma (<= sigma/6, never coarser than 5 meV, never
    finer than 1 meV) so the sampled peak height stays well within ~1% of
    the true peak even at the slider minimum; rebuilt on every sigma change.
    """
    spacing = min(0.005, max(sigma / 6.0, 0.001))
    npts = int(np.clip(np.ceil((hi - lo) / spacing) + 1, 200, 40000))
    return np.linspace(lo, hi, npts)


def plot_state_decomposition(
    output_dir: str,
    sum_threshold: float = 0.1,
    cfg: Optional[PlotConfig] = None,
    curve: bool = False,
    sigma: float = 0.2,
    amplitude: bool = False,
    norm_each: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Parameters
    ----------
    output_dir : str
        Directory containing band_matches_rectangular.txt and
        band_matches_rectangular_all.txt (the full_dir from the matcher).
    sum_threshold : float
        Per-full-state cumulative-tail threshold (see module docstring).
    cfg : PlotConfig or None
        Plot configuration. Uses defaults if None.
    curve : bool
        Initial state of the Curve toggle: replace each system's stacked
        spreads panel with its gradient-colored total curve (the outer
        original-states panels stay), with an in-figure sigma slider (eV).
        See the module docstring for the color blend and pipeline.
    sigma : float
        Initial Gaussian smearing width in eV (VASP SIGMA convention).
    amplitude : bool
        Initial state of the Amplitude toggle: sticks plot the square root
        of the overlap (stacks stay additive in probability); curves show
        the sqrt of the normalized curve (the final pipeline step, applied
        after normalization).
    norm_each : bool
        Initial state of the Norm sub-toggle (curve mode only): True gives
        each system curve unit area; False (default) applies one shared
        factor so the SUM of the system curves has unit area, preserving
        their relative intensities.

    Returns
    -------
    fig, axes
        axes is a LIVE list, left -> right: [metal states, metal
        spreads/curve, molecule spreads/curve, molecule states] for the
        systems present. Toggling Curve rebuilds the subplot row and the
        list is mutated in place to match.
    """
    if cfg is None:
        cfg = PlotConfig()

    plotter = RectAEPAWColorPlotter(cfg)

    # ------------------------------------------------------------------
    # 1. Load via the main plotter's _prepare so colors / Fermis /
    #    vacuum / SOC pairing / shared_molecule_color all come out
    #    identical to the main MOPMatcher plot.
    # ------------------------------------------------------------------
    main_path = os.path.join(output_dir, "band_matches_rectangular.txt")
    P = plotter._prepare(main_path, bonding=False, write_summaries=False)

    by_full          = P.by_full
    comp_pairs       = P.comp_pairs
    known_fermis     = P.known_fermis
    v_aligned        = P.v_aligned
    component_colors = P.component_colors
    mol_labels       = P.mol_labels
    metal_present    = P.metal_present

    # ------------------------------------------------------------------
    # 1b. Always-on Kramers combining (SOC-gated like the colors, via
    #     P.is_soc and the same pairing walk the color code uses). Every
    #     downstream consumer — original-states panels, threshold pool,
    #     spread stacks/segments, and curve groups — operates on combined
    #     entities: one line/segment per entity at E_mean, member weights
    #     pair-summed per full state, rep-member colors (both members of a
    #     pair share that color by construction). Non-SOC runs: every
    #     entity is a single and behavior is unchanged.
    # ------------------------------------------------------------------
    entities_by_lbl = {
        lbl: pair_entities(
            comp_pairs.get(lbl, []), P.is_soc, cfg.soc_pair_thresh_eV,
            center_idx=P.known_homos.get(lbl))
        for lbl in comp_pairs
    }
    member_to_ent = {
        lbl: {i: ent for ent in ents for i in ent.indices}
        for lbl, ents in entities_by_lbl.items()
    }
    comp_energy_map = {lbl: dict(pairs) for lbl, pairs in comp_pairs.items()}

    # Entity-level original-states pairs: (indices, rep, E_mean) per label.
    comp_pairs_ent = {
        lbl: [(ent.indices, ent.rep, ent.E_mean) for ent in ents]
        for lbl, ents in entities_by_lbl.items()
    }

    # Entity-level by_full: pair-sum the member weights per full state; dE
    # is recomputed against E_mean so E + dE still reconstructs E_full.
    by_full_ent: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for full_idx, comps in by_full.items():
        out: Dict[str, List[Dict[str, Any]]] = {}
        for lbl, recs in comps.items():
            merged: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for r in recs:
                ent = member_to_ent.get(lbl, {}).get(int(r["comp_idx"]))
                if ent is None:
                    continue
                m = merged.get(ent.indices)
                if m is None:
                    E_full_rec = float(r["E"]) + float(r["dE"])
                    merged[ent.indices] = dict(
                        indices=ent.indices, rep=ent.rep,
                        ov=float(r["ov"]), E=float(ent.E_mean),
                        dE=E_full_rec - float(ent.E_mean))
                else:
                    m["ov"] += float(r["ov"])
            out[lbl] = list(merged.values())
        by_full_ent[full_idx] = out

    # ------------------------------------------------------------------
    # 2. Energy shift: put full-system Fermi at zero (same convention
    #    as spread_plot.py).
    # ------------------------------------------------------------------
    fermi_shift = -float(known_fermis.get("full", 0.0))
    v_shifted = (v_aligned + fermi_shift) if v_aligned is not None else None

    # ------------------------------------------------------------------
    # 3. System definitions, then the column layout. Column order left ->
    #    right: metal states, metal spreads, molecule spreads, molecule
    #    states -- each system's ORIGINAL-STATES panel sits OUTSIDE its
    #    spreads panel. flip=True marks a spreads/curve panel whose
    #    intensity zero sits on its RIGHT edge (the side facing its
    #    states panel); the metal spreads' zero is on its LEFT edge.
    # ------------------------------------------------------------------
    panels: List[Tuple[str, List[str]]] = []
    if metal_present:
        panels.append(("metal", ["metal"]))
    if mol_labels:
        panels.append(("molecule", list(mol_labels)))

    if not panels:
        raise RuntimeError(
            "No components found in matcher output; cannot plot."
        )

    n_panels = len(panels)

    columns: List[Dict[str, Any]] = []
    for panel_idx, (kind, lbls) in enumerate(panels):
        if kind == "metal":
            columns.append(dict(role="og", kind=kind, lbls=lbls))
            columns.append(dict(role="sys", kind=kind, lbls=lbls,
                                panel_idx=panel_idx, flip=False))
        else:
            columns.append(dict(role="sys", kind=kind, lbls=lbls,
                                panel_idx=panel_idx, flip=True))
            columns.append(dict(role="og", kind=kind, lbls=lbls))
    n_cols = len(columns)

    label_to_grp: Dict[str, int] = {}
    for gi, (_, lbls) in enumerate(panels):
        for lbl in lbls:
            label_to_grp[lbl] = gi
    figure_labels = list(label_to_grp.keys())

    # ------------------------------------------------------------------
    # 4. Per-full-state thresholding on the pool across ALL systems, at
    #    the ENTITY level (Kramers-combined weights are what compete
    #    against the tail threshold). kept[grp_idx][full_idx] -> list of
    #    (comp_label, indices, rep, ov, E_mean, dE) for that system.
    # ------------------------------------------------------------------
    kept: List[Dict[int, List[
        Tuple[str, Tuple[int, ...], int, float, float, float]]]] = [
        defaultdict(list) for _ in panels
    ]

    for full_idx, comps in by_full_ent.items():
        # Pool every entity contribution across systems for this full
        # state. tuple = (grp_idx, comp_label, indices, rep, ov, E_mean, dE)
        pool: List[Tuple[int, str, Tuple[int, ...], int,
                         float, float, float]] = []
        for lbl in figure_labels:
            gi = label_to_grp[lbl]
            for r in comps.get(lbl, []):
                pool.append((
                    gi, lbl, r["indices"], int(r["rep"]),
                    float(r["ov"]), float(r["E"]), float(r["dE"]),
                ))

        # Drop from smallest ov upward while cumulative dropped mass
        # stays at or below sum_threshold.
        pool_sorted = sorted(pool, key=lambda t: t[4])
        dropped_sum = 0.0
        keep_start = 0
        for i, rec in enumerate(pool_sorted):
            ov_i = rec[4]
            if dropped_sum + ov_i <= sum_threshold:
                dropped_sum += ov_i
                keep_start = i + 1
            else:
                break

        for gi, lbl, indices, rep, ov, E_c, dE in pool_sorted[keep_start:]:
            kept[gi][full_idx].append((lbl, indices, rep, ov, E_c, dE))

    # ------------------------------------------------------------------
    # 5. Per-system curve groups: every contribution of one ENTITY pooled
    #    over the full states it appears in. Keyed by (lbl, rep) so the
    #    existing color lookup component_colors[lbl][rep] applies (both
    #    pair members share the rep's color by construction).
    #    panel_groups[panel_idx][(lbl, rep)] = [(E_pos, ov), ...]
    # ------------------------------------------------------------------
    panel_groups: List[Dict[Tuple[str, int], List[Tuple[float, float]]]] = []
    for panel_idx in range(n_panels):
        g: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
        for full_idx, recs in kept[panel_idx].items():
            for lbl, indices, rep, ov, E_c, dE in recs:
                E_pos = E_c + dE + fermi_shift   # = E_full shifted
                g[(lbl, rep)].append((E_pos, ov))
        panel_groups.append(dict(g))

    # ------------------------------------------------------------------
    # 6. Figure. Rotated layout: ENERGY IS VERTICAL, so the figure height
    #    carries the spectrum and the width carries the row of columns.
    #    The data axes are built by _rebuild_axes inside _draw_all and
    #    `axes` is mutated in place. Bottom strip: reserve a fixed ~1.05
    #    inches for the widgets regardless of column count.
    # ------------------------------------------------------------------
    fig_h = max(5.0, 2.0 * cfg.figsize[1])
    fig = plt.figure(figsize=(cfg.figsize[0], fig_h))
    bfrac = min(0.35, 1.05 / fig_h)
    fig.subplots_adjust(bottom=bfrac, top=0.93, wspace=0.35)
    axes: List[plt.Axes] = []

    state = {"curve": bool(curve), "sigma": float(sigma),
             "amplitude": bool(amplitude), "norm_each": bool(norm_each)}
    # one entry per system curve: ax, lc (LineCollection), groups, lo, hi,
    # name, flip, and the latest (x=energy grid, y=intensity) arrays
    curve_entries: List[Dict[str, Any]] = []

    def _yval(v: float) -> float:
        # stick magnitude: raw overlap, or its square root in amplitude mode.
        # Stacked segment boundaries pass their CUMULATIVE probability
        # through here, so a stack's end is sqrt(total), not a sum of roots.
        return float(np.sqrt(max(v, 0.0))) if state["amplitude"] else float(v)

    def _ylabel() -> str:
        # Intensity-axis label (the HORIZONTAL axis in the rotated layout).
        # Bare math only -- no units or descriptive text. Curve mode plots a
        # SUM of broadened overlaps, so the label gains a large \sum; with
        # Amplitude on, one radical extends over the entire sum (the sqrt is
        # taken after the sum and normalization). Stick mode is unchanged:
        # Amplitude just drops the squared bracket.
        lab = cfg.ylabel
        if state["curve"]:
            inner = lab.strip("$")
            if state["amplitude"]:
                return rf"$\sqrt{{\sum {inner}}}$"
            return rf"$\sum {inner}$"
        if state["amplitude"] and lab.endswith("^2$"):
            lab = lab[:-3] + "$"
        return lab

    # Per-axis bookkeeping for the click-armed cursors (stick mode), the
    # per-axis curve cursors (curve mode) and the original-states panels'
    # cursors (both modes); all rebuilt on every redraw.
    state_artists: Dict[Any, Dict[int, List[Any]]] = {}
    hover_maps:    Dict[Any, Dict[int, Dict[Any, str]]] = {}
    state_extent:  Dict[Any, Dict[int, Tuple[float, float]]] = {}
    state_cursors: Dict[Any, Dict[int, Any]] = {}
    curve_cursors: Dict[Any, Any] = {}
    og_cursors:    Dict[Any, Any] = {}

    def _erange(energies: List[float]) -> Tuple[float, float]:
        # ascending (lo, hi) energy window for the curves; a reversed
        # energy_range (flipped axis) is normalized so the grid stays valid
        # for the searchsorted windowing in _gauss_sum
        if cfg.energy_range:
            lo, hi = cfg.energy_range
        else:
            lo, hi = min(energies) - 3.0, max(energies) + 3.0
        lo, hi = (min(lo, hi), max(lo, hi))
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        return float(lo), float(hi)

    def _remove_all_cursors():
        curs = [c for by_state in state_cursors.values()
                for c in by_state.values()]
        curs.extend(curve_cursors.values())
        curs.extend(og_cursors.values())
        for cur in curs:
            try:
                cur.remove()
            except Exception:
                try:
                    for sel in list(cur.selections):
                        cur.remove_selection(sel)
                    cur.enabled = False
                except Exception:
                    pass
        state_cursors.clear()
        curve_cursors.clear()
        og_cursors.clear()

    def _draw_reflines(ax, kind, lbls):
        # energy runs vertically, so every reference line is horizontal
        if cfg.show_fermi_line:
            ax.axhline(
                0.0,
                color=cfg.fermi_line_color,
                linestyle=cfg.fermi_line_style,
                alpha=0.7,
            )

        local_fermi_label = "metal" if kind == "metal" else (
            lbls[0] if lbls else None
        )
        if (cfg.show_local_fermi
                and local_fermi_label is not None
                and local_fermi_label in known_fermis):
            ax.axhline(
                known_fermis[local_fermi_label] + fermi_shift,
                color=cfg.local_fermi_color,
                linestyle=cfg.local_fermi_style,
                alpha=0.9,
                zorder=10,
            )

        if cfg.show_vacuum_line and v_shifted is not None:
            ax.axhline(
                v_shifted,
                color=cfg.vacuum_line_color,
                linewidth=cfg.vacuum_line_width,
                linestyle=cfg.vacuum_line_style,
                zorder=10,
            )

    # ------------------------------------------------------------------
    # 7. Draw closures. The subplot row always has one column per layout
    #    entry; the Curve toggle only swaps the CONTENT of the two system
    #    columns (stacked sticks <-> gradient curve). The original-states
    #    columns are identical in both modes.
    # ------------------------------------------------------------------
    def _rebuild_axes(n: int):
        for a in list(axes):
            a.remove()
        axes.clear()
        arr = fig.subplots(1, n, sharey=True, squeeze=False)
        axes.extend(arr[0, :])

    def _update_curve_artists() -> float:
        """
        Recompute every system curve for the current sigma / Norm /
        Amplitude state and push intensities + gradient colors into the
        existing LineCollections. Pipeline order (module docstring):
        1. combine (raw system totals + color blend), 2. normalize (joint
        by default, per-curve when Norm is on), 3. amplitude sqrt LAST.
        Segments are laid down as (intensity, energy) pairs -- intensity
        horizontal, energy vertical. Returns the tallest plotted peak.
        """
        if not curve_entries:
            return 0.0
        lo, hi = curve_entries[0]["lo"], curve_entries[0]["hi"]
        x = _curve_grid(lo, hi, state["sigma"])
        raws, cols = [], []
        for ent in curve_entries:
            raw, col = _system_curve(x, ent["groups"], state["sigma"])
            raws.append(raw)
            cols.append(col)
        # 2. normalize: per-curve (Norm on) or joint (Norm off, default)
        areas = [max(float(_TRAPZ(r, x)), 0.0) for r in raws]
        joint = sum(areas)
        peak = 0.0
        for ent, raw, col, area in zip(curve_entries, raws, cols, areas):
            denom = area if state["norm_each"] else joint
            y = raw / denom if denom > 0 else raw
            # 3. amplitude: sqrt of the NORMALIZED curve, the final step
            if state["amplitude"]:
                y = np.sqrt(y)
            segs, seg_rgba = _gradient_segments(y, x, col)
            ent["lc"].set_segments(segs)
            ent["lc"].set_color(seg_rgba)
            ent["x"], ent["y"] = x, y
            if y.size and float(y.max()) > peak:
                peak = float(y.max())
        return peak

    def _apply_curve_xlims(peak: float):
        # ONE shared intensity maximum across both curve panels so joint
        # normalization keeps metal-vs-molecule relative intensity readable;
        # flipped panels get a reversed (still positive) intensity axis
        hi = peak * 1.05 if peak > 0 else 1.0
        for ent in curve_entries:
            ent["ax"].set_xlim((hi, 0.0) if ent["flip"] else (0.0, hi))

    def _draw_og_panel(ax, kind, lbls):
        # ---- original-states panel: EVERY combined Kramers entity as a
        # full-width horizontal line at its E_mean (shifted), in the rep
        # member's canonical color (both members share it). No intensity
        # ticks or values; label kept. ----
        og_hover: Dict[Any, str] = {}
        for lbl in lbls:
            emap = comp_energy_map.get(lbl, {})
            for indices, rep, E_m in comp_pairs_ent.get(lbl, []):
                color = component_colors.get(lbl, {}).get(
                    rep, (0.4, 0.4, 0.4, 1.0)
                )
                ln = ax.axhline(
                    E_m + fermi_shift, color=color, linewidth=cfg.lw_stick,
                )
                member_lines = "".join(
                    f"\nband {i}: E {emap[i] + fermi_shift:+.4f} eV"
                    for i in indices if i in emap
                ) if len(indices) > 1 else ""
                og_hover[ln] = (
                    f"{lbl} {_bands_str(indices)}\n"
                    f"E {E_m + fermi_shift:+.4f} eV"
                    + member_lines
                )
        _draw_reflines(ax, kind, lbls)
        ax.set_xticks([])
        ax.set_xlabel(_ylabel())
        name = "metal" if kind == "metal" else ", ".join(lbls)
        # small titles: four side-by-side columns leave little room and long
        # molecule names collide with their neighbors at the default size
        ax.set_title(f"{name} states", fontsize="small")
        if cfg.annotate_on_hover and HAS_MPLCURSORS and og_hover:
            cur = mplcursors.cursor(list(og_hover), hover=True)
            cur.enabled = False
            og_cursors[ax] = cur

            @cur.connect("add")
            def _on_add(sel, hmap=og_hover):
                txt = hmap.get(sel.artist)
                if txt:
                    sel.annotation.set_text(txt)

    def _draw_curve_panel(ax, panel_idx, kind, lbls, flip, lo, hi):
        name = "metal" if kind == "metal" else ", ".join(lbls)
        groups = [
            (component_colors.get(lbl, {}).get(
                comp_idx, (0.4, 0.4, 0.4, 1.0)), sts)
            for (lbl, comp_idx), sts in panel_groups[panel_idx].items()
        ]
        _draw_reflines(ax, kind, lbls)
        ax.set_xlabel(_ylabel())
        if not groups:
            ax.set_title(f"{name} curve\n(no contributions kept)",
                         fontsize="small")
            return
        lc = LineCollection([], linewidths=1.8, zorder=2)
        ax.add_collection(lc)
        n_contrib = sum(len(sts) for _, sts in groups)
        hover_txt = (
            f"{name}\n"
            f"total decomposition curve\n"
            f"{len(groups)} component states, "
            f"{n_contrib} contributions\n"
            f"color = intensity-weighted blend of state colors")
        curve_entries.append(dict(ax=ax, lc=lc, groups=groups,
                                  lo=lo, hi=hi, name=name, flip=flip))
        ax.set_title(f"{name} curve", fontsize="small")
        if cfg.annotate_on_hover and HAS_MPLCURSORS:
            cur = mplcursors.cursor([lc], hover=True)
            cur.enabled = False
            curve_cursors[ax] = cur

            @cur.connect("add")
            def _on_add(sel, txt=hover_txt):
                sel.annotation.set_text(txt)

    def _draw_stick_panel(ax, panel_idx, kind, lbls, flip):
        # ---- stick mode: stacked-segment drawing, rotated -- each stack
        # extends along the horizontal INTENSITY axis at its full state's
        # (vertical) energy ----
        state_artists[ax] = defaultdict(list)
        hover_maps[ax]    = defaultdict(dict)
        state_extent[ax]  = {}
        max_total_h = 0.0

        for full_idx, recs in kept[panel_idx].items():
            if not recs:
                continue
            # Sort by entity indices ascending (lexicographic over the
            # member tuple, equivalent to comp_idx order for singles);
            # comp_label as deterministic tiebreaker for the
            # multi-molecule case.
            recs_sorted = sorted(recs, key=lambda t: (t[1], t[0]))

            # Recover E_full from any one record: E_full = E_mean + dE.
            # All recs for this full_idx share the same E_full.
            E_full = recs_sorted[0][4] + recs_sorted[0][5]
            y_pos  = E_full + fermi_shift

            # Cumulative boundaries are tracked in PROBABILITY and
            # each is passed through _yval, so in amplitude mode the
            # stack end is sqrt(total) rather than a sum of roots.
            x_bot = 0.0
            for lbl, indices, rep, ov, E_c, dE in recs_sorted:
                x_top = x_bot + ov
                color = component_colors.get(lbl, {}).get(
                    rep, (0.4, 0.4, 0.4, 1.0)
                )
                line = ax.hlines(
                    y_pos, _yval(x_bot), _yval(x_top),
                    color=color, linewidth=cfg.lw_stick,
                )
                state_artists[ax][full_idx].append(line)
                hover_maps[ax][full_idx][line] = (
                    f"{lbl} {_bands_str(indices)}\n"
                    f"E_comp {E_c + fermi_shift:+.4f} eV\n"
                    f"dE     {dE:+.4f} eV\n"
                    f"ov     {ov:.5f}\n"
                    f"full_idx {full_idx}"
                )
                x_bot = x_top

            x_bot = _yval(x_bot)
            state_extent[ax][full_idx] = (y_pos, x_bot)
            if x_bot > max_total_h:
                max_total_h = x_bot

        # intensity zero on the edge FACING this system's states panel;
        # flipped panels reverse direction, tick values stay positive
        hi_x = max_total_h * 1.05 if max_total_h > 0 else 1.0
        ax.set_xlim((hi_x, 0.0) if flip else (0.0, hi_x))

        _draw_reflines(ax, kind, lbls)
        ax.set_xlabel(_ylabel())
        name = "metal" if kind == "metal" else ", ".join(lbls)
        ax.set_title(f"{name} spreads", fontsize="small")

    def _draw_all():
        curve_entries.clear()
        _remove_all_cursors()
        state_artists.clear()
        hover_maps.clear()
        state_extent.clear()

        _rebuild_axes(n_cols)

        # both curve panels share one energy window (and thus one grid)
        lo = hi = None
        if state["curve"]:
            all_E = [E for pg in panel_groups
                     for sts in pg.values() for E, _ in sts]
            if all_E:
                lo, hi = _erange(all_E)

        for ax, col in zip(axes, columns):
            if col["role"] == "og":
                _draw_og_panel(ax, col["kind"], col["lbls"])
            elif state["curve"]:
                if lo is None:
                    ax.set_title("no contributions kept", fontsize="small")
                else:
                    _draw_curve_panel(ax, col["panel_idx"], col["kind"],
                                      col["lbls"], col["flip"], lo, hi)
            else:
                _draw_stick_panel(ax, col["panel_idx"], col["kind"],
                                  col["lbls"], col["flip"])

        if state["curve"] and curve_entries:
            _apply_curve_xlims(_update_curve_artists())

        # shared vertical energy axis: tick marks everywhere, tick values
        # and the energy label only on the leftmost panel (sharey hides
        # the inner tick labels)
        axes[0].set_ylabel(cfg.xlabel)
        if cfg.energy_range:
            axes[0].set_ylim(cfg.energy_range)
        elif state["curve"] and lo is not None:
            axes[0].set_ylim(lo, hi)

        # --- Click-armed per-full-state cursors (stick mode only) ---
        if not state["curve"] and cfg.annotate_on_hover and HAS_MPLCURSORS:
            for ax in axes:
                if ax not in state_artists:
                    continue
                state_cursors[ax] = {}
                by_state = state_artists.get(ax, {})
                for full_idx, arts in by_state.items():
                    if not arts:
                        continue
                    cur = mplcursors.cursor(arts, hover=True)
                    cur.enabled = False
                    hm = hover_maps[ax][full_idx]
                    @cur.connect("add")
                    def _on_add(sel, _hm=hm):
                        txt = _hm.get(sel.artist)
                        if txt:
                            sel.annotation.set_text(txt)
                    state_cursors[ax][full_idx] = cur

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 8. Widgets: Curve + Amplitude toggles, a Norm sub-toggle and the
    #    sigma slider shown only in curve mode.
    # ------------------------------------------------------------------
    # Widget axes are placed at inch-stable heights (fractions scaled by
    # 1/fig_h) so they stay inside the reserved bottom strip and clear of
    # the intensity labels at every column count. Hiding a widget's axes is
    # what blocks its events: matplotlib skips invisible axes when resolving
    # event.inaxes, so a hidden widget cannot be clicked.
    chk = CheckButtons(
        fig.add_axes([0.01, 0.06 / fig_h, 0.13, 0.62 / fig_h]),
        ["Curve", "Amplitude"], [state["curve"], state["amplitude"]],
    )
    norm_ax = fig.add_axes([0.16, 0.06 / fig_h, 0.10, 0.32 / fig_h])
    chk_norm = CheckButtons(norm_ax, ["Norm"], [state["norm_each"]])
    sld_ax = fig.add_axes([0.35, 0.16 / fig_h, 0.40, 0.20 / fig_h])
    sld = Slider(
        sld_ax, r"$\sigma$ (eV)", SIGMA_MIN, SIGMA_MAX,
        valinit=state["sigma"], valfmt="%.3f",
    )

    def _sync_widget_visibility():
        sld_ax.set_visible(state["curve"])
        sld.set_active(state["curve"])
        # Norm picks the normalization scheme (step 2 of the pipeline), so
        # it only means anything in curve mode. Never call
        # CheckButtons.set_active here -- that toggles a box rather than
        # enabling the widget.
        norm_ax.set_visible(state["curve"])

    _sync_widget_visibility()

    def _on_chk(_label):
        state["curve"], state["amplitude"] = chk.get_status()
        _sync_widget_visibility()
        _draw_all()

    def _on_norm(_label):
        # normalization scheme change: recompute in place (step 2 of the
        # pipeline changes, steps 1 and 3 rerun around it)
        state["norm_each"] = chk_norm.get_status()[0]
        if curve_entries:
            _apply_curve_xlims(_update_curve_artists())
        fig.canvas.draw_idle()

    def _on_sigma(val):
        # in-place update through the full pipeline: combine + colors,
        # normalize per the Norm toggle, then the final amplitude sqrt
        state["sigma"] = float(val)
        if curve_entries:
            _apply_curve_xlims(_update_curve_artists())
        fig.canvas.draw_idle()

    chk.on_clicked(_on_chk)
    chk_norm.on_clicked(_on_norm)
    sld.on_changed(_on_sigma)

    _draw_all()

    # ------------------------------------------------------------------
    # 9. Click handling. Stick mode: left click on a state's stack ->
    #    disable every other state's cursor on that axis, enable this
    #    state's cursor; right click -> clear/disable every state cursor.
    #    Curve and original-states panels: left click arms the axis'
    #    cursor, right click clears it.
    # ------------------------------------------------------------------
    def _hit_full_idx(ax, ex, ey):
        info = state_extent.get(ax, {})
        if not info or ex is None or ey is None:
            return None
        # ~5-pixel tolerance along the (vertical) energy axis, in data
        # units, derived per-axis.
        inv = ax.transData.inverted()
        p0 = inv.transform((0.0, 0.0))
        p1 = inv.transform((0.0, 5.0))
        y_tol = abs(p1[1] - p0[1])

        best = None
        best_dy = float("inf")
        for full_idx, (y_pos, x_top) in info.items():
            if 0.0 <= ex <= x_top and abs(ey - y_pos) <= y_tol:
                dy = abs(ey - y_pos)
                if dy < best_dy:
                    best_dy = dy
                    best = full_idx
        return best

    def _arm_or_clear(cur, button):
        if button == 1:
            if not cur.enabled:
                cur.enabled = True
        elif button == 3:
            for sel in list(cur.selections):
                cur.remove_selection(sel)
            cur.enabled = False

    def _on_click(event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        cur = og_cursors.get(ax)
        if cur is not None:
            _arm_or_clear(cur, event.button)
            return
        if state["curve"]:
            cur = curve_cursors.get(ax)
            if cur is not None:
                _arm_or_clear(cur, event.button)
            return
        cursors = state_cursors.get(ax, {})
        if not cursors:
            return
        if event.button == 1:
            hit = _hit_full_idx(ax, event.xdata, event.ydata)
            if hit is None:
                return
            for fi, cur in cursors.items():
                if fi != hit:
                    for sel in list(cur.selections):
                        cur.remove_selection(sel)
                    cur.enabled = False
            cursors[hit].enabled = True
        elif event.button == 3:
            for cur in cursors.values():
                for sel in list(cur.selections):
                    cur.remove_selection(sel)
                cur.enabled = False

    if HAS_MPLCURSORS:
        fig.canvas.mpl_connect("button_press_event", _on_click)

    # keep the widgets (and the closures wired to them) alive with the figure
    fig._decomp_widgets = (chk, sld, chk_norm)
    # registry of the per-system gradient curves, exposed for tests/tooling
    fig._decomp_curve_entries = curve_entries
    return fig, axes


# ======================================================================
if __name__ == "__main__":

    import sys, types
    _pkg_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_pkg_dir, ".."))
    # Stub package so relative imports resolve without __init__.py
    _pkg = types.ModuleType("matcher")
    _pkg.__path__ = [_pkg_dir]
    sys.modules.setdefault("matcher", _pkg)
    from matcher.vacupot_plotter import RectAEPAWColorPlotter, PlotConfig
    from matcher.kramers import pair_entities

    # ---- Directory containing matcher output files ----
    output_dir = r'dir/run'

    # ---- Per-full-state cumulative-tail threshold ----
    sum_threshold = 0.01

    # ---- Curve (Gaussian broadening) initial state ----
    curve = False            # True = one gradient curve per system (slider)
    sigma = 0.1              # initial smearing width in eV (VASP SIGMA style)
    amplitude = True        # sticks: sqrt(ov); curves: sqrt of the
                             # normalized curve (applied last)
    norm_each = False        # curve mode: False = joint (sum of curves has
                             # unit area), True = each curve unit area

    cfg = PlotConfig(
        cmap_name_simple="managua_r",
        cmap_name_metal="vanimo_r",
        energy_range=None,
        shared_molecule_color=True,
        min_total_mol_wspan=0.02,

        power_simple_neg=0.25,
        power_simple_pos=0.75,
        power_metal_neg=0.075,
        power_metal_pos=0.075,
        power_residual=0.5,

        show_fermi_line=True,
        fermi_line_style=":",
        fermi_line_color="k",
        show_local_fermi=True,

        show_vacuum_line=True,
        vacuum_line_color="black",
        vacuum_line_width=3.5,
        vacuum_line_style="-",

        annotate_on_hover=True,
        soc_pair_thresh_eV=0.005,
    )

    fig, axes = plot_state_decomposition(
        output_dir=output_dir,
        sum_threshold=sum_threshold,
        cfg=cfg,
        curve=curve,
        sigma=sigma,
        amplitude=amplitude,
        norm_each=norm_each,
    )
    plt.show()
