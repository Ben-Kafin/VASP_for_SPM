# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 2026

@author: Benjamin Kafin

Standalone spread plotter.
For a given component label and list of HOMO-relative entity offsets, produces a
vertically-stacked figure showing how each combined Kramers entity distributes
its squared overlap across the full-system Bloch states.

Kramers combining is always on. It is gated by P.is_soc from _prepare
(INCAR via detect_ncl_from_incar when present, else inferred from Kramers
doubling via kramers.infer_soc_from_spectra) exactly like the color pairing
(non-SOC runs: entities are singles, behavior otherwise unchanged), and it is
a data-level transform applied at weight assembly, strictly upstream of the
curve pipeline below: stick mode draws
summed stick heights per full state, curve mode broadens the combined weights.

comp_indices are combined-entity offsets: 0 = the HOMO entity, -1 = HOMO-1,
+1 = LUMO, etc. (kramers.homo_entity_offsets). Absolute band indices are no
longer accepted.

A Curve CheckButton (lower-left) replaces the per-state sticks with a
Gaussian-broadened curve: each matched full state becomes a Gaussian centered
at its energy (VASP-style smearing,
g = exp(-((E-E0)/sigma)^2) / (sigma*sqrt(pi))), combined into one continuous
curve per panel. The in-figure slider sets the smearing width sigma in eV
live (curves are updated in place, no rebuild).

An Amplitude CheckButton plots sqrt(ov) instead of ov for the sticks; its
curve-mode meaning is defined by the pipeline below.

CURVE PIPELINE -- ORDER OF OPERATIONS (keep this order in any future edit):

  1. combine    The probability curve: sum_i p_i g_i, with p the raw
                overlap probability. This is the ONLY combination rule --
                amplitudes are never turned into Gaussians.
  2. normalize  Each probability curve is divided by its OWN integral, so
                every probability curve individually encloses unit area.
                Any future toggle that reshapes the curve must slot in
                BEFORE this step.
  3. amplitude  THE FINAL STEP, after normalization: with Amplitude on the
                displayed curve is sqrt(normalized probability curve).
                Nothing is re-normalized afterward -- the sqrt curve's area
                is deliberately not 1.

So the two curve forms are:

  Amplitude off   (sum_i p_i g_i) / integral    [unit area]
  Amplitude on    sqrt of that                  [area != 1 by design]
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

if __name__ != "__main__":
    from .vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )
    from .kramers import pair_entities, homo_entity_offsets

# Curve-mode smearing slider range (eV), VASP SIGMA convention
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0


def _offset_name(offset: int) -> str:
    """Display name of a HOMO-relative entity offset: 0 = HOMO, -1 = HOMO-1,
    +1 = LUMO, +2 = LUMO+1, ... (unicode minus, matching the marker labels)."""
    if offset == 0:
        return "HOMO"
    if offset < 0:
        return f"HOMO−{-offset}"
    if offset == 1:
        return "LUMO"
    return f"LUMO+{offset - 1}"


def _entity_label(offset: int, ent) -> str:
    """Label showing both forms, e.g. 'HOMO−1 (74+75)'."""
    if ent is None:
        return _offset_name(offset)
    return f"{_offset_name(offset)} ({'+'.join(str(i) for i in ent.indices)})"

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


def _curve_values(x: np.ndarray, states: List[Tuple[float, float]],
                  sigma: float, amplitude: bool) -> np.ndarray:
    """
    The curve actually plotted for one MO, from its (E0, p) states with p
    the raw overlap probability.

    ===================== ORDER OF OPERATIONS =====================
    The curve pipeline is strictly ordered, and every future edit must keep
    it in this order:

        1. combine    -- the probability curve sum_i p_i g_i. This is the
                         ONLY combination rule; amplitudes are never turned
                         into Gaussians.
        2. normalize  -- divide the curve by its OWN integral over x, so
                         every probability curve individually encloses unit
                         area. Any future toggle that reshapes the curve
                         must be applied BEFORE this step.
        3. amplitude  -- THE FINAL STEP, after normalization: sqrt of the
                         normalized probability curve. Nothing is
                         re-normalized afterward, so the amplitude curve's
                         area is deliberately not 1.
    ===============================================================

    The integral is taken numerically over the displayed grid, so the drawn
    probability curve encloses exactly 1 even where Gaussian tails fall
    outside the axis range.
    """
    # 1. combine
    y = _gauss_sum(x, states, sigma)
    # 2. normalize: unit area per probability curve
    area = float(_TRAPZ(y, x))
    if area > 0:
        y = y / area
    # 3. amplitude: sqrt of the NORMALIZED curve, the final step
    if amplitude:
        y = np.sqrt(y)
    return y


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


def plot_spread(
    output_dir: str,
    comp_label: str,
    comp_indices: List[int],
    darken_factor: float = 0.4,
    cfg: Optional[PlotConfig] = None,
    curve: bool = False,
    sigma: float = 0.2,
    amplitude: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Parameters
    ----------
    output_dir : str
        Directory containing band_matches_rectangular.txt and
        band_matches_rectangular_all.txt (the full_dir from the matcher).
    comp_label : str
        Component label (e.g. a molecule folder name or "metal").
    comp_indices : list of int
        Combined-entity HOMO-relative offsets within that component: 0 = the
        HOMO entity, -1 = HOMO-1, +1 = LUMO (kramers.homo_entity_offsets,
        computed with that component's own HOMO index). Each gets its own
        subplot. Absolute band indices are no longer accepted.
    darken_factor : float
        Fraction to blend the MO marker bar toward black (0 = base color, 1 = black).
    cfg : PlotConfig or None
        Plot configuration. Uses defaults if None.
    curve : bool
        Initial state of the Curve toggle: replace the per-state sticks with a
        Gaussian-broadened curve per panel, normalized to unit area, with an
        in-figure sigma slider (eV).
    sigma : float
        Initial Gaussian smearing width in eV (VASP SIGMA convention).
    amplitude : bool
        Initial state of the Amplitude toggle: sticks plot sqrt(ov); curves
        show the sqrt of the unit-area probability curve (the final pipeline
        step, applied after normalization).

    Returns
    -------
    fig, axes
    """
    if cfg is None:
        cfg = PlotConfig()

    plotter = RectAEPAWColorPlotter(cfg)

    # ------------------------------------------------------------------
    # 1. Load data + build colors through _prepare (same SOC-correct path
    #    as the main vacupot_plotter and spread_collapse). Reads from
    #    matcher_results.h5 when present, else the .txt outputs.
    #    write_summaries=False -> no behavior files written here.
    # ------------------------------------------------------------------
    main_path = os.path.join(output_dir, "band_matches_rectangular.txt")
    P = plotter._prepare(main_path, bonding=False, write_summaries=False)

    by_full = P.by_full
    comp_pairs = P.comp_pairs
    known_fermis = P.known_fermis
    v_aligned = P.v_aligned
    component_colors = P.component_colors

    # ------------------------------------------------------------------
    # 2. Energy shift: put full-system Fermi at zero
    # ------------------------------------------------------------------
    fermi_full = known_fermis.get("full", 0.0)
    fermi_shift = -fermi_full
    mol_fermi_shifted = known_fermis.get(comp_label, 0.0) + fermi_shift

    v_shifted = (v_aligned + fermi_shift) if v_aligned is not None else None

    # Default x-min: lowest energy present in both systems
    comp_energies = [E for idx, E in comp_pairs.get(comp_label, [])]
    full_energies = [rec["E"] + rec["dE"]
                     for comps in by_full.values()
                     for recs in comps.values()
                     for rec in recs]
    if comp_energies and full_energies:
        shared_e_min = max(min(comp_energies), min(full_energies)) + fermi_shift
    else:
        shared_e_min = None

    # ------------------------------------------------------------------
    # 3. Component-state energies from comp_pairs, and the always-on
    #    Kramers entities (SOC-gated like the colors) with their
    #    HOMO-relative offset map for this component.
    # ------------------------------------------------------------------
    comp_energy_map: Dict[int, float] = {}
    for idx, E in comp_pairs.get(comp_label, []):
        comp_energy_map[idx] = E

    entities = pair_entities(
        comp_pairs.get(comp_label, []), P.is_soc, cfg.soc_pair_thresh_eV,
        center_idx=P.known_homos.get(comp_label),
    )
    try:
        offset_map = homo_entity_offsets(entities, P.known_homos.get(comp_label))
    except ValueError:
        offset_map = {}

    # ------------------------------------------------------------------
    # 4. Per-panel static data (colors, entity energy, combined match
    #    bars) — computed once; the draw closure below re-renders from
    #    this on every toggle. Kramers combining happens HERE, at weight
    #    assembly: one bar per full state carrying the pair-summed weight,
    #    strictly upstream of the curve pipeline.
    # ------------------------------------------------------------------
    panel_data: List[Dict[str, Any]] = []
    for offset in comp_indices:
        ent = offset_map.get(offset)
        base_color = component_colors.get(comp_label, {}).get(
            ent.rep if ent is not None else -1, (0.4, 0.4, 0.4, 1.0)
        )
        base_rgb = np.array(base_color[:3])
        half = darken_factor / 2.0
        light_rgb = base_rgb + (1.0 - base_rgb) * half   # toward white
        dark_rgb = base_rgb * (1.0 - half)                # toward black

        E_comp = None if ent is None else ent.E_mean + fermi_shift

        match_bars: List[Dict[str, Any]] = []
        if ent is not None:
            members = set(ent.indices)
            for full_idx, comps in by_full.items():
                bar = None
                for rec in comps.get(comp_label, []):
                    if rec["comp_idx"] in members:
                        if bar is None:
                            bar = {
                                "E_full": rec["E"] + rec["dE"] + fermi_shift,
                                "ov": rec["ov"],
                                "full_idx": full_idx,
                            }
                        else:
                            bar["ov"] += rec["ov"]
                if bar is not None:
                    match_bars.append(bar)

        member_energy_lines = ""
        if ent is not None:
            member_energy_lines = "".join(
                f"\nband {i}: E = {comp_energy_map[i] + fermi_shift:+.4f} eV"
                for i in ent.indices if i in comp_energy_map
            )

        panel_data.append(dict(
            offset=offset, ent=ent, label=_entity_label(offset, ent),
            light_rgb=light_rgb, dark_rgb=dark_rgb,
            E_comp=E_comp, match_bars=match_bars,
            member_energy_lines=member_energy_lines,
        ))

    # ------------------------------------------------------------------
    # 5. Build figure: one subplot per requested band index, bottom strip
    #    reserved for the Curve checkbox + sigma slider.
    # ------------------------------------------------------------------
    n_panels = len(comp_indices)
    fig_h = max(3.0, 1.8 * n_panels)
    fig, axes_arr = plt.subplots(
        n_panels, 1,
        sharex=True,
        figsize=(cfg.figsize[0], fig_h),
        squeeze=False,
    )
    axes = [axes_arr[i, 0] for i in range(n_panels)]
    # Bottom strip: reserve a fixed ~1.05 inches for the widgets plus the
    # shared x-label regardless of panel count, so short (1-2 panel) figures
    # do not let the x-label descend into the slider band.
    bfrac = min(0.35, 1.05 / fig_h)
    fig.subplots_adjust(bottom=bfrac, top=0.93, hspace=0.65)

    state = {"curve": bool(curve), "sigma": float(sigma),
             "amplitude": bool(amplitude)}
    cursor_by_axes: Dict[Any, Any] = {}
    curve_entries: List[Dict[str, Any]] = []   # ax, line, lo, hi, states
    marker_entries: List[Dict[str, Any]] = []  # ax, line, E

    def _yval(v: float) -> float:
        # stick magnitude: raw overlap, or its square root in amplitude mode
        return float(np.sqrt(max(v, 0.0))) if state["amplitude"] else float(v)

    def _ylabel() -> str:
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

    def _xrange(energies: List[float]) -> Tuple[float, float]:
        # ascending (lo, hi): the same range the axis shows; a reversed
        # energy_range (flipped axis) is normalized so the grid stays valid
        # for the searchsorted windowing in _gauss_sum
        if cfg.energy_range:
            lo, hi = cfg.energy_range
        else:
            lo = shared_e_min if shared_e_min is not None else min(energies) - 3.0
            hi = v_shifted if v_shifted is not None else max(energies) + 3.0
        lo, hi = (min(lo, hi), max(lo, hi))
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        return float(lo), float(hi)

    # ------------------------------------------------------------------
    # 6. Draw closure: renders every panel from panel_data in the current
    #    mode (sticks or Gaussian-broadened curve), rebuilding the hover
    #    cursors each time.
    # ------------------------------------------------------------------
    def _draw_all():
        curve_entries.clear()
        marker_entries.clear()
        for cur in cursor_by_axes.values():
            try:
                cur.remove()
            except Exception:
                try:
                    for sel in list(cur.selections):
                        cur.remove_selection(sel)
                    cur.enabled = False
                except Exception:
                    pass
        cursor_by_axes.clear()

        for pd, ax in zip(panel_data, axes):
            ax.clear()
            offset = pd["offset"]

            if pd["E_comp"] is None:
                ax.set_title(f"{comp_label} {_offset_name(offset)} — not found")
                continue
            E_comp = pd["E_comp"]
            match_bars = pd["match_bars"]

            artists = []
            hover_map = {}

            if state["curve"]:
                # --- Gaussian-broadened curve from the RAW probabilities,
                # through the full pipeline: combine, per-curve unit-area
                # normalization, then the final amplitude sqrt ---
                sts = [(mb["E_full"], mb["ov"]) for mb in match_bars]
                peak = 0.0
                if sts:
                    lo, hi = _xrange([E for E, _ in sts])
                    x = _curve_grid(lo, hi, state["sigma"])
                    y = _curve_values(x, sts, state["sigma"],
                                      state["amplitude"])
                    (line,) = ax.plot(
                        x, y, color=tuple(pd["light_rgb"]), linewidth=1.8,
                    )
                    artists.append(line)
                    total = sum(p for _, p in sts)
                    hover_map[line] = (
                        f"{comp_label} {pd['label']}\n"
                        f"Gaussian-broadened spread\n"
                        f"total ov {total:.5f} over {len(sts)} states\n"
                        f"normalized to unit area"
                        + ("\nsqrt (amplitude) of the normalized curve"
                           if state["amplitude"] else "")
                    )
                    curve_entries.append(dict(ax=ax, line=line, lo=lo, hi=hi,
                                              states=sts))
                    peak = float(y.max()) if y.size else 0.0
                mo_height = 1.1 * peak if peak > 0 else 1.0
            else:
                # --- Draw match bars (lightened) ---
                max_h = max((_yval(mb["ov"]) for mb in match_bars), default=0.0)
                mo_height = 1.1 * max_h if max_h > 0 else 1.0
                for mb in match_bars:
                    ov = mb["ov"]
                    h = _yval(ov)
                    line = ax.vlines(
                        mb["E_full"], 0, h,
                        color=tuple(pd["light_rgb"]), linewidth=cfg.lw_stick,
                    )
                    artists.append(line)
                    hover_map[line] = (
                        f"full_idx {mb['full_idx']}\n"
                        f"E_full {mb['E_full']:+.4f} eV\n"
                        f"{'amp' if state['amplitude'] else 'ov'} {h:.5f}"
                    )

            # --- The MO marker bar: 1.1x the tallest match/peak, on top ---
            mo_line = ax.vlines(
                E_comp, 0, mo_height,
                color=tuple(pd["dark_rgb"]), linewidth=cfg.lw_stick + 1.0,
            )
            artists.append(mo_line)
            hover_map[mo_line] = (
                f"{comp_label} {pd['label']}\n"
                f"E {E_comp:+.4f} eV"
                + pd["member_energy_lines"]
            )
            if state["curve"]:
                marker_entries.append(dict(ax=ax, line=mo_line, E=E_comp))

            # --- Reference lines ---
            if cfg.show_fermi_line:
                ax.axvline(
                    0.0, color=cfg.fermi_line_color,
                    linestyle=cfg.fermi_line_style, alpha=0.7,
                )
            ax.axvline(
                mol_fermi_shifted, color="red",
                linestyle="--", alpha=0.9,
            )
            if v_shifted is not None:
                ax.axvline(
                    v_shifted, color="black",
                    linestyle="-", linewidth=3.5,
                )

            # --- Axis limits & labels ---
            ax.set_ylim(0, mo_height)
            if cfg.energy_range:
                ax.set_xlim(cfg.energy_range)
            else:
                if shared_e_min is not None:
                    ax.set_xlim(left=shared_e_min)
                if v_shifted is not None:
                    ax.set_xlim(right=v_shifted)
            ax.set_ylabel(_ylabel())
            ax.set_title(
                f"{comp_label} {pd['label']},  E = {E_comp:+.4f} eV"
            )

            # --- Hover annotations ---
            if cfg.annotate_on_hover and HAS_MPLCURSORS and artists:
                cur = mplcursors.cursor(artists, hover=True); cur.enabled = False
                cursor_by_axes[ax] = cur

                @cur.connect("add")
                def _on_add(sel, hmap=hover_map):
                    txt = hmap.get(sel.artist)
                    if txt:
                        sel.annotation.set_text(txt)

        # --- Shared x-label on bottom axis ---
        axes[-1].set_xlabel(cfg.xlabel)
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 7. Widgets: Curve + Amplitude toggles, and the sigma slider shown
    #    only in curve mode.
    # ------------------------------------------------------------------
    # Widget axes are placed at inch-stable heights (fractions scaled by
    # 1/fig_h) so they stay inside the reserved bottom strip and clear of
    # the x-label at every panel count. Hiding a widget's axes is what
    # blocks its events: matplotlib skips invisible axes when resolving
    # event.inaxes, so a hidden widget cannot be clicked.
    chk = CheckButtons(
        fig.add_axes([0.01, 0.06 / fig_h, 0.13, 0.62 / fig_h]),
        ["Curve", "Amplitude"], [state["curve"], state["amplitude"]],
    )
    sld_ax = fig.add_axes([0.35, 0.16 / fig_h, 0.40, 0.20 / fig_h])
    sld = Slider(
        sld_ax, r"$\sigma$ (eV)", SIGMA_MIN, SIGMA_MAX,
        valinit=state["sigma"], valfmt="%.3f",
    )

    def _sync_widget_visibility():
        sld_ax.set_visible(state["curve"])
        sld.set_active(state["curve"])

    _sync_widget_visibility()

    def _on_chk(_label):
        state["curve"], state["amplitude"] = chk.get_status()
        _sync_widget_visibility()
        _draw_all()

    def _on_sigma(val):
        # in-place update: regenerate the grid so its resolution tracks
        # sigma, recompute every curve through the full pipeline (combine,
        # per-curve unit-area normalization, then the final amplitude
        # sqrt), rescale, stretch MO markers
        state["sigma"] = float(val)
        ax_max: Dict[Any, float] = {}
        grids: Dict[Tuple[float, float], np.ndarray] = {}
        for ent in curve_entries:
            key = (ent["lo"], ent["hi"])
            x = grids.get(key)
            if x is None:
                x = _curve_grid(key[0], key[1], state["sigma"])
                grids[key] = x
            y = _curve_values(x, ent["states"], state["sigma"],
                              state["amplitude"])
            ent["line"].set_data(x, y)
            peak = float(y.max()) if y.size else 0.0
            ax_max[ent["ax"]] = max(ax_max.get(ent["ax"], 0.0), peak)
        for ent in marker_entries:
            peak = ax_max.get(ent["ax"], 0.0)
            top = 1.1 * peak if peak > 0 else 1.0
            E = ent["E"]
            ent["line"].set_segments([np.array([[E, 0.0], [E, top]])])
            ent["ax"].set_ylim(0, top)
        fig.canvas.draw_idle()

    chk.on_clicked(_on_chk)
    sld.on_changed(_on_sigma)

    _draw_all()

    if HAS_MPLCURSORS:
        def _on_click(event, _map=cursor_by_axes):
            if event.inaxes is None: return
            cur = _map.get(event.inaxes)
            if cur is None: return
            if event.button == 1:
                if not cur.enabled:
                    cur.enabled = True
            elif event.button == 3:
                for sel in list(cur.selections):
                    cur.remove_selection(sel)
                cur.enabled = False
        fig.canvas.mpl_connect("button_press_event", _on_click)

    # keep the widgets (and the closures wired to them) alive with the figure
    fig._spread_widgets = (chk, sld)
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
    from matcher.vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )
    from matcher.kramers import pair_entities, homo_entity_offsets

    # ---- Directory containing matcher output files ----
    output_dir = r'dir/run'

    # ---- Which component and which entities to plot ----
    comp_label = "clean_kp551"
    comp_indices = [-2, -1, 0, 1]        # HOMO-relative entity offsets:
                                         # 0 = HOMO entity, -1 = HOMO-1,
                                         # +1 = LUMO

    # ---- Visual tuning ----
    darken_factor = 0.4                  # 0 = base color, 1 = black

    # ---- Curve (Gaussian broadening) initial state ----
    curve = False            # True = Gaussian-broadened curves (sigma slider)
    sigma = 0.2              # initial smearing width in eV (VASP SIGMA style)
    amplitude = False        # sticks: sqrt(ov); curves: sqrt of the
                             # unit-area probability curve (applied last)

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

        annotate_on_hover=True,
    )

    fig, axes = plot_spread(
        output_dir=output_dir,
        comp_label=comp_label,
        comp_indices=comp_indices,
        darken_factor=darken_factor,
        cfg=cfg,
        curve=curve,
        sigma=sigma,
        amplitude=amplitude,
    )
    plt.show()
