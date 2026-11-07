# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 2026

@author: Benjamin Kafin

Standalone spread plotter.
For a list of component labels and a shared list of HOMO-relative entity
offsets, produces a figure showing how each combined Kramers entity
distributes its squared overlap across the full-system Bloch states.

Kramers combining is always on. It is gated by P.is_soc from _prepare
(INCAR via detect_ncl_from_incar when present, else inferred from Kramers
doubling via kramers.infer_soc_from_spectra) exactly like the color pairing
(non-SOC runs: entities are singles, behavior otherwise unchanged), and it is
a data-level transform applied at weight
assembly, strictly upstream of the curve pipeline below: stick mode draws
summed stick heights per full state, curve mode broadens the combined
weights. In Horizontal mode the order is: pair-sum within component ->
component-sum -> combine -> normalize -> amplitude.

comp_indices are combined-entity offsets: 0 = the HOMO entity, -1 = HOMO-1,
+1 = LUMO, etc. (kramers.homo_entity_offsets, computed per component
including "metal" via its own HOMO index). Absolute band indices are no
longer accepted.

Two checks in the controls under the panels toggle the layout live:

  Vertical OFF, Horizontal OFF (base grid):
      one subplot per (entity-offset row x component column).

  Vertical ON only:
      one column per component; that component's entity spreads are
      overlaid onto a single axis.

  Horizontal ON only:
      one axis per entity offset; every component's spread of that entity is
      overlaid onto the single axis, with each component offset in lightness
      (same hue) so they remain distinguishable.

  both ON:
      a single axis holds every component's spread of every entity offset.

A further Curve toggle replaces the per-state sticks with Gaussian-broadened
curves: each matched full state becomes a Gaussian centered at its energy
(VASP-style smearing, g = exp(-((E-E0)/sigma)^2) / (sigma*sqrt(pi))), and
each MO's spread is drawn as one continuous curve in the MO's stick color.
The σ (eV) number box (controls.SigmaBox, shown in curve mode) sets sigma in
eV live (in-place curve update, no rebuild): Enter, leaving the box after an
edit, or an arrow-key digit step applies it. Segmentation is a stick-mode-only
feature and is ignored while Curve is on.

CURVE PIPELINE -- ORDER OF OPERATIONS (keep this order in any future edit):

  1. combine    The probability curve: sum_i p_i g_i, with p the state's
                raw overlap probability (summed over components when
                Horizontal is on). This is the ONLY combination rule --
                amplitudes are never turned into Gaussians.
  2. normalize  Each probability curve is divided by its OWN integral, so
                every probability curve individually encloses unit area.
                Any future toggle that reshapes the curve must slot in
                BEFORE this step.
  3. amplitude  THE FINAL STEP, after normalization: with Amplitude on the
                displayed curve is sqrt(normalized probability curve).
                Nothing is re-normalized afterward -- the sqrt curve's area
                is deliberately not 1, because Amplitude is defined to act
                on the already-normalized probability curve.

So the two curve forms are:

  Amplitude off   (sum_i p_i g_i) / integral    [unit area]
  Amplitude on    sqrt of that                  [area != 1 by design]

In stick mode Amplitude keeps its original meaning (stick height sqrt(ov),
unnormalized); the pipeline above applies to curves only.

The window is a qtgui figure (the shared Qt layer, this package's own copy): a
persistent figure whose _build_ui() does fig.clf() and rebuilds the per-mode
subplot layout in the window's grid of cells (automatic layout); the controls
(the checks, the Segmentation check shown while Horizontal is on, the σ box
shown in curve mode) are real Qt widgets made once in a strip under the
panels, and their callbacks update the state and rebuild. Hover labels are
widgets.HoverLabels on every panel: a left click on a panel arms them, a
right click clears and disarms them.

Color/data are obtained through RectAEPAWColorPlotter._prepare so that the
per-component colors (including SOC Kramers pairing, shared-molecule, and metal
handling) are identical to the main vacupot_plotter rendering.

Multi-run mode (MultiSpreadPlotter / plot_spread_multi): several run
directories, each with its own component labels but a shared band-index list,
stacked vertically in one figure with a Horizontal toggle in its controls. With
Horizontal ON each run is drawn fully collapsed onto a single axis; with it
OFF each run splits into one equal-width subplot per component on a
left-aligned column grid. Bars are plain base colors with no MO markers; the
original molecule MOs instead occupy their own row at the very top, sticks at
the averaged molecule energies (global average when collapsed, per-column
average when split — the red dashed molecule-Fermi line follows the same
rule). All subplots share one energy axis on which the runs' aligned vacuum
potentials coincide; the whole stack is then shifted so the LAST run's
full-system Fermi sits at 0 eV (dashed black line), with each row's own
system Fermi marked by a dotted black line.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from math import sqrt

if __name__ != "__main__":
    from .qtgui import qtplot as plt
    from .qtgui import controls, widgets

# y-axis label of the original-MO row in the multi-run figure: same overlap
# form as PlotConfig.ylabel but with psi_MO on both sides of the bracket.
MO_YLABEL = r"$|\langle\psi_{\mathrm{MO}}|\psi_{\mathrm{MO}}\rangle|^2$"

# Curve-mode smearing width limits / default (eV) of the σ box, VASP SIGMA convention.
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
SIGMA_DEFAULT = 0.2

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

if __name__ != "__main__":
    from .vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )
    from .kramers import pair_entities, homo_entity_offsets


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


def _attach_hover(cfg, ax, artists, hover_map):
    """
    The hover labels of one panel (widgets.HoverLabels, the one function behind every hover
    in the suite): `hover_map` gives each artist its text; a left click on the panel arms them,
    a right click clears and disarms them.  Nothing when cfg.annotate_on_hover is off or the
    panel holds no artist.  Shared by SpreadPlotter and MultiSpreadPlotter.
    """
    if cfg.annotate_on_hover and artists:
        widgets.HoverLabels(ax, artists, hover_map)


class SpreadPlotter:
    """
    Builds and maintains a spread-plot figure whose layout is controlled by the
    Vertical / Horizontal checks in its controls. All data loading and color
    construction happen once in __init__; a toggle only rebuilds the drawing
    via _build_ui().  DESIGN (inches), the window's design size and the text
    zoom's reference: cfg.figsize[0] wide, 1.8 inches per entity row (at least 3).
    """

    def __init__(
        self,
        output_dir: str,
        components: List[str],
        comp_indices: List[int],
        collapse_vertical: bool = False,
        collapse_horizontal: bool = False,
        darken_factor: float = 0.4,
        cfg: Optional[PlotConfig] = None,
        build_figure: bool = True,
        amplitude: bool = False,
        curve: bool = False,
        sigma: float = SIGMA_DEFAULT,
    ):
        self.output_dir = output_dir
        self.components = list(components)
        self.comp_indices = list(comp_indices)
        self.collapse_vertical = bool(collapse_vertical)
        self.collapse_horizontal = bool(collapse_horizontal)
        self.collapse_segmentation = False
        self.amplitude = bool(amplitude)
        self.curve_mode = bool(curve)
        self.sigma = float(sigma)
        # Registries for in-place slider updates (curve mode only):
        # _curve_entries: dict(ax, line, x, states=[(E, w), ...])
        # _marker_entries: dict(ax, line, E) — MO markers stretched to the top
        self._curve_entries: List[Dict[str, Any]] = []
        self._marker_entries: List[Dict[str, Any]] = []
        self.darken_factor = float(darken_factor)
        self.cfg = cfg if cfg is not None else PlotConfig()

        self.plotter = RectAEPAWColorPlotter(self.cfg)

        # --------------------------------------------------------------
        # Load data + build colors through _prepare (SOC-correct path,
        # identical to vacupot_plotter). write_summaries=False -> no files.
        # --------------------------------------------------------------
        main_path = os.path.join(self.output_dir, "band_matches_rectangular.txt")
        P = self.plotter._prepare(main_path, bonding=False, write_summaries=False)

        self.component_colors = P.component_colors
        self.comp_pairs = P.comp_pairs
        self.by_full = P.by_full
        self.known_fermis = P.known_fermis
        self.v_aligned = P.v_aligned

        # --------------------------------------------------------------
        # Raw x-min: lowest energy present in both systems (pooled over the
        # plotted components, which are identical molecules so share states).
        # Kept unshifted so set_energy_shift can re-derive shared_e_min.
        # --------------------------------------------------------------
        comp_energies = [E
                         for component in self.components
                         for idx, E in self.comp_pairs.get(component, [])]
        full_energies = [rec["E"] + rec["dE"]
                         for comps in self.by_full.values()
                         for recs in comps.values()
                         for rec in recs]
        if comp_energies and full_energies:
            self._raw_e_min = max(min(comp_energies), min(full_energies))
        else:
            self._raw_e_min = None

        # Energy shift: put full-system Fermi at zero. MultiSpreadPlotter
        # overrides this via set_energy_shift for vacuum-aligned stacking.
        self.set_energy_shift(-self.known_fermis.get("full", 0.0))

        # --------------------------------------------------------------
        # Per-component component-state energy maps
        # --------------------------------------------------------------
        self.comp_energy_map: Dict[str, Dict[int, float]] = {}
        for component in self.components:
            m: Dict[int, float] = {}
            for idx, E in self.comp_pairs.get(component, []):
                m[idx] = E
            self.comp_energy_map[component] = m

        # --------------------------------------------------------------
        # Always-on Kramers entities (SOC-gated like the colors, via
        # P.is_soc: INCAR when present, else Kramers-doubling inference)
        # and the HOMO-relative
        # offset map per component ("metal" anchors on its own HOMO).
        # comp_indices are combined-entity offsets from here on.
        # --------------------------------------------------------------
        self.is_soc = P.is_soc
        self.known_homos = P.known_homos
        self.entities: Dict[str, List[Any]] = {}
        self.entity_offsets: Dict[str, Dict[int, Any]] = {}
        for component in self.components:
            ents = pair_entities(
                self.comp_pairs.get(component, []), self.is_soc,
                self.cfg.soc_pair_thresh_eV,
                center_idx=self.known_homos.get(component),
            )
            self.entities[component] = ents
            try:
                self.entity_offsets[component] = homo_entity_offsets(
                    ents, self.known_homos.get(component))
            except ValueError:
                self.entity_offsets[component] = {}

        # --------------------------------------------------------------
        # Band-index ranking (marker z-order and figure sizing)
        # --------------------------------------------------------------
        self.sorted_idx = sorted(self.comp_indices)
        self.R_count = len(self.comp_indices)

        # Persistent figure (created once) with its controls (made once, in a
        # strip under the panels). Skipped when this instance only supplies
        # data/drawing to MultiSpreadPlotter.
        self.fig = None
        self.chk = self.chk_seg = self.sld_sigma = None
        if build_figure:
            self.DESIGN = (self.cfg.figsize[0], max(3.0, 1.8 * max(1, self.R_count)))
            self.fig = plt.figure(figsize=self.DESIGN)
            self._make_controls()
            self._build_ui()

    # ==================================================================
    # Energy shift
    # ==================================================================
    def set_energy_shift(self, shift: float) -> None:
        """
        Set the additive shift applied to every plotted energy and recompute
        the shift-dependent derived values. __init__ defaults to -fermi(full)
        (own full-system Fermi at zero); MultiSpreadPlotter calls this again
        to place each run on the common vacuum-aligned scale.
        """
        self.fermi_shift = float(shift)
        self.v_shifted = (self.v_aligned + self.fermi_shift) if self.v_aligned is not None else None
        self.shared_e_min = (self._raw_e_min + self.fermi_shift) if self._raw_e_min is not None else None

    # ==================================================================
    # Amplitude / probability transform
    # ==================================================================
    def _yval(self, v: float) -> float:
        """
        Magnitude actually plotted: the raw overlap (probability, default)
        or its square root (Amplitude toggle on). Summed bars pass their SUM
        through this, so collapsed amplitude bars are sqrt(sum ov), not
        sum sqrt(ov).
        """
        return sqrt(max(v, 0.0)) if self.amplitude else v

    def _mag_name(self) -> str:
        return "amp" if self.amplitude else "ov"

    def _ylabel(self) -> str:
        # Bare math only -- no units or descriptive text. Curve mode plots a
        # SUM of broadened overlaps, so the label gains a large \sum; with
        # Amplitude on, one radical extends over the entire sum (the sqrt is
        # taken after the sum and normalization). Stick mode is unchanged:
        # Amplitude just drops the squared bracket.
        lab = self.cfg.ylabel
        if self.curve_mode:
            inner = lab.strip("$")
            if self.amplitude:
                return rf"$\sqrt{{\sum {inner}}}$"
            return rf"$\sum {inner}$"
        if self.amplitude and lab.endswith("^2$"):
            lab = lab[:-3] + "$"
        return lab

    # ==================================================================
    # Color helpers
    # ==================================================================
    def _base_rgb_for(self, component: str, comp_idx: int) -> np.ndarray:
        base_color = self.component_colors.get(component, {}).get(
            comp_idx, (0.4, 0.4, 0.4, 1.0)
        )
        return np.array(base_color[:3])

    def _light_rgb_from(self, base_rgb: np.ndarray, f: float) -> np.ndarray:
        # toward white by blend fraction f
        return base_rgb + (1.0 - base_rgb) * f

    def _dark_rgb_from(self, base_rgb: np.ndarray) -> np.ndarray:
        # toward black by darken_factor/2
        return base_rgb * (1.0 - self.darken_factor / 2.0)

    def _light_f_for(self, j: int, n: int) -> float:
        """
        Per-component toward-white blend fraction. For n components spread
        evenly across [0, darken_factor]: component 0 -> 0 (base, darkest),
        component n-1 -> darken_factor (lightest), midpoint darken_factor/2
        coincides with the non-offset light bar color. For n == 1, no offset
        (darken_factor/2).
        """
        if n <= 1:
            return self.darken_factor / 2.0
        return self.darken_factor * j / (n - 1)

    # ==================================================================
    # Kramers-entity helpers (comp_indices are HOMO-relative offsets)
    # ==================================================================
    def _entity_for(self, component: str, offset: int):
        """The component's combined entity at the given HOMO-relative
        offset, or None when that offset does not exist ("not found")."""
        return self.entity_offsets.get(component, {}).get(offset)

    def _entity_label(self, component: str, offset: int) -> str:
        """Label showing both forms, e.g. 'HOMO−1 (74+75)'."""
        ent = self._entity_for(component, offset)
        if ent is None:
            return _offset_name(offset)
        return (f"{_offset_name(offset)} "
                f"({'+'.join(str(i) for i in ent.indices)})")

    def _entity_member_lines(self, component: str, offset: int) -> str:
        """Hover suffix listing both member energies of the entity."""
        ent = self._entity_for(component, offset)
        if ent is None:
            return ""
        emap = self.comp_energy_map.get(component, {})
        return "".join(
            f"\nband {i}: E = {emap[i] + self.fermi_shift:+.4f} eV"
            for i in ent.indices if i in emap
        )

    def _lead_component(self, comps: List[str], offset: int) -> Optional[str]:
        """First component that has an entity at this offset (label and rep
        color source for axes holding several components' curves)."""
        for c in comps:
            if self._entity_for(c, offset) is not None:
                return c
        return None

    def _entity_rgb(self, component: Optional[str], offset: int) -> np.ndarray:
        """Base color of the entity's rep member (the member the color walk
        assigned the pair's slot to)."""
        if component is not None:
            ent = self._entity_for(component, offset)
            if ent is not None:
                return self._base_rgb_for(component, ent.rep)
        return np.array([0.4, 0.4, 0.4])

    # ==================================================================
    # Energy / fermi averaging helpers (components are identical molecules)
    # ==================================================================
    def _comp_fermi_shifted(self, component: str) -> float:
        return self.known_fermis.get(component, 0.0) + self.fermi_shift

    def _avg_fermi_shifted(self, comps: List[str]) -> float:
        vals = [self.known_fermis.get(c, 0.0) for c in comps]
        return (sum(vals) / len(vals) + self.fermi_shift) if vals else self.fermi_shift

    def _avg_marker_E(self, comps: List[str], offset: int) -> Optional[float]:
        es = []
        for c in comps:
            ent = self._entity_for(c, offset)
            if ent is not None:
                es.append(ent.E_mean)
        if not es:
            return None
        return sum(es) / len(es) + self.fermi_shift

    # ==================================================================
    # Z-order (markers only — spread bars are z-sorted per energy point
    # at draw time: tallest in back, shortest in front)
    # ==================================================================
    def _rank_of(self, comp_idx: int) -> int:
        return self.sorted_idx.index(comp_idx)

    def _marker_z(self, comp_idx: int) -> float:
        return 1.0 + (self._rank_of(comp_idx) / max(self.R_count, 1)) * 0.9

    # ==================================================================
    # Drawing primitives
    # ==================================================================
    def _draw_marker(self, ax, comp_idx, E_shifted, base_rgb, zorder, label_text,
                     height: float = 1.0):
        col = tuple(self._dark_rgb_from(base_rgb))
        line = ax.vlines(
            E_shifted, 0, height, color=col,
            linewidth=self.cfg.lw_stick + 1.0, zorder=zorder,
        )
        return line, label_text

    def _draw_reflines(self, ax, fermi_shifted_val):
        if self.cfg.show_fermi_line:
            ax.axvline(
                0.0, color=self.cfg.fermi_line_color,
                linestyle=self.cfg.fermi_line_style, alpha=0.7, zorder=10,
            )
        ax.axvline(
            fermi_shifted_val, color="red",
            linestyle="--", alpha=0.9, zorder=10,
        )
        if self.v_shifted is not None:
            ax.axvline(
                self.v_shifted, color="black",
                linestyle="-", linewidth=3.5, zorder=10,
            )

    def _finalize_axis(self, ax, max_ov, title, set_xlabel, set_ylabel_flag):
        top = max_ov if max_ov > 0 else 1.0
        ax.set_ylim(0, top)
        if self.cfg.energy_range:
            ax.set_xlim(self.cfg.energy_range)
        else:
            if self.shared_e_min is not None:
                ax.set_xlim(left=self.shared_e_min)
            if self.v_shifted is not None:
                ax.set_xlim(right=self.v_shifted)
        if set_ylabel_flag:
            ax.set_ylabel(self._ylabel())
        if title is not None:
            ax.set_title(title)
        if set_xlabel:
            ax.set_xlabel(self.cfg.xlabel)

    # ==================================================================
    # Curve (Gaussian-broadening) helpers
    # ==================================================================
    def _curve_xrange(self, energies: List[float]) -> Tuple[float, float]:
        """
        Ascending (lo, hi) energy window the broadened curves are evaluated
        over: the same range the axis will show (energy_range override, else
        shared_e_min to the vacuum level), padded from the data when either
        edge is unknown. A reversed energy_range (flipped axis) is
        normalized to ascending order so the grid stays valid for the
        searchsorted windowing in _gauss_sum.
        """
        if self.cfg.energy_range:
            lo, hi = self.cfg.energy_range
        else:
            lo = (self.shared_e_min if self.shared_e_min is not None
                  else min(energies) - 3.0)
            hi = (self.v_shifted if self.v_shifted is not None
                  else max(energies) + 3.0)
        lo, hi = (min(lo, hi), max(lo, hi))
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        return float(lo), float(hi)

    def _update_curves(self):
        """
        Recompute every registered broadened curve for the current sigma
        in place (no UI rebuild) — regenerating the grid so its resolution
        tracks sigma — through the full _curve_values pipeline (combine,
        per-curve unit-area normalization, then the final amplitude sqrt),
        rescale each affected axis (5% headroom over the tallest peak,
        matching the build path), and stretch the curve-mode MO markers to
        the new top. Called by the σ box on every applied value.
        """
        if not self._curve_entries:
            return
        ax_max: Dict[Any, float] = {}
        grids: Dict[Tuple[float, float], np.ndarray] = {}
        for ent in self._curve_entries:
            key = (ent["lo"], ent["hi"])
            x = grids.get(key)
            if x is None:
                x = _curve_grid(key[0], key[1], self.sigma)
                grids[key] = x
            y = _curve_values(x, ent["states"], self.sigma, self.amplitude)
            ent["line"].set_data(x, y)
            peak = float(y.max()) if y.size else 0.0
            ax_max[ent["ax"]] = max(ax_max.get(ent["ax"], 0.0), peak)
        for ax, peak in ax_max.items():
            ax.set_ylim(0, peak * 1.05 if peak > 0 else 1.0)
        for ent in self._marker_entries:
            peak = ax_max.get(ent["ax"])
            if peak and peak > 0:
                E = ent["E"]
                ent["line"].set_segments(
                    [np.array([[E, 0.0], [E, peak * 1.05]])])
        for fig in {ent["ax"].figure for ent in self._curve_entries}:
            fig.canvas.draw_idle()

    def _on_sigma(self, val):
        self.sigma = float(val)
        self._update_curves()

    # ==================================================================
    # The controls (made once) and the UI build (clf, layout, hover)
    # ==================================================================
    def _make_controls(self):
        """
        The window's controls, real Qt widgets made once (controls.py, the one control set of
        every window): the 'collapse' checks (Vertical, Horizontal, Amplitude, Curve), the
        Segmentation check (shown while Horizontal is on) and the σ (eV) number box (shown in
        curve mode), each in a box frame, in a strip that wraps under the panels.
        automatic layout: the one rule for every window (nothing placed by hand).
        """
        self.chk = controls.make(
            self.fig, 'collapse',
            labels=['Vertical', 'Horizontal', 'Amplitude', 'Curve'],
            values=[self.collapse_vertical, self.collapse_horizontal,
                    self.amplitude, self.curve_mode])
        self.chk.on_change(self._on_toggle)
        self.chk_seg = controls.make(self.fig, 'Segmentation', value=self.collapse_segmentation)
        self.chk_seg.on_change(self._on_seg_toggle)
        self._seg_box = controls.box(self.chk_seg)
        self.sld_sigma = controls.SigmaBox(self.fig, self.sigma, SIGMA_MIN, SIGMA_MAX)
        self.sld_sigma.on_change(self._on_sigma)
        self._sigma_box = controls.box(self.sld_sigma)
        strip = controls.FlowLayout()
        strip.addWidget(self.chk)
        strip.addWidget(self._seg_box)
        strip.addWidget(self._sigma_box)
        self.fig.layout_root().addLayout(strip)

    def _sync_control_visibility(self):
        """Segmentation is shown (and takes input) while Horizontal is on; the σ box in curve mode."""
        self._seg_box.setVisible(self.collapse_horizontal)
        self.chk_seg.enable(self.collapse_horizontal)
        self._sigma_box.setVisible(self.curve_mode)
        self.sld_sigma.enable(self.curve_mode)

    def _build_ui(self):
        """
        The panels rebuilt for the current state: the figure cleared, the per-mode subplot grid
        drawn again (automatic layout: fig.subplots fills the window's grid of cells), every
        panel's hover labels made anew.  The controls are never rebuilt.
        """
        self.fig.clf()
        widgets.HoverLabels.prune(self.fig)
        self._curve_entries = []
        self._marker_entries = []
        self._sync_control_visibility()

        self._draw_layout()

        self.fig.canvas.draw_idle()

    def _on_toggle(self, _label, states):
        """The 'collapse' checks reported a click: the window takes the four states and rebuilds."""
        self.collapse_vertical, self.collapse_horizontal = states[0], states[1]
        self.amplitude = states[2]
        self.curve_mode = states[3]
        self._build_ui()

    def _on_seg_toggle(self, flag):
        """The Segmentation check reported a click: the window takes the state and rebuilds."""
        self.collapse_segmentation = bool(flag)
        self._build_ui()

    # ==================================================================
    # Per-mode layout drawing
    #
    # Two independent collapses:
    #   vertical   -> collapses the ROW (band-index) dimension: n_rows = 1
    #   horizontal -> collapses the COLUMN (component) dimension: n_cols = 1
    # Neither toggle is aware of the other. The grid is
    # n_rows x n_cols; each cell is drawn by _draw_cell, which receives the
    # band indices and components that fall into that cell. Segmentation only
    # affects the horizontal-on drawing path and is independent of vertical.
    # ==================================================================
    def _draw_layout(self):
        R = len(self.comp_indices)
        C = len(self.components)
        components = self.components
        comp_indices = self.comp_indices

        horizontal_on = self.collapse_horizontal
        vertical_on = self.collapse_vertical
        segmentation_on = self.collapse_segmentation

        n_rows = 1 if vertical_on else R
        n_cols = 1 if horizontal_on else C
        axes = self.fig.subplots(n_rows, n_cols, sharex=True, squeeze=False)

        for row_slot in range(n_rows):
            cell_band_indices = (comp_indices if vertical_on
                                 else [comp_indices[row_slot]])
            for col_slot in range(n_cols):
                cell_components = (components if horizontal_on
                                   else [components[col_slot]])
                ax = axes[row_slot][col_slot]
                self._draw_cell(
                    ax, cell_band_indices, cell_components,
                    horizontal_on, vertical_on, segmentation_on,
                    set_xlabel=(row_slot == n_rows - 1),
                    set_ylabel_flag=(col_slot == 0),
                )

    # ==================================================================
    # Single-cell drawing. cell_band_indices x cell_components are the band
    # indices / components that land on this one axis. Colors go through the
    # frozen helpers; bar z-order is decided per plotted energy (tallest in
    # back), markers keep their band-index ranking behind all bars.
    # ==================================================================
    def _draw_cell_curves(self, ax, bars_by_x, cell_components, horizontal_on,
                          plain_color, artists, hover):
        """
        Curve-mode drawing for one axis: broadens the bars collected in
        pass 1 into one Gaussian-sum curve per band index (colors identical
        to the stick branches) and registers each curve for in-place
        sigma-slider updates. Returns the tallest curve peak (curve peaks
        only — stick heights must not leak into the curve-mode axis top).
        """
        axis_max = 0.0
        # Curves are built from the RAW probability p, not the stick height:
        # the amplitude/sqrt transform belongs inside _curve_values (square
        # each amplitude Gaussian, then sum), not on the per-state weight.
        groups: Dict[int, List[Tuple[float, float]]] = {}
        for recs in bars_by_x.values():
            for rec in recs:
                groups.setdefault(rec["comp_idx"], []).append(
                    (rec["E_full"], rec["p"]))
        if not groups:
            return axis_max
        all_E = [E for states in groups.values() for E, _ in states]
        lo, hi = self._curve_xrange(all_E)
        x = _curve_grid(lo, hi, self.sigma)
        curves = []
        for comp_idx, states in groups.items():
            if horizontal_on:
                lead = self._lead_component(cell_components, comp_idx)
                base = self._entity_rgb(lead, comp_idx)
                col = (tuple(base) if plain_color
                       else tuple(self._light_rgb_from(
                           base, self.darken_factor / 2.0)))
                name = (f"{self._entity_label(lead, comp_idx)} (Σ components)"
                        if lead is not None
                        else f"{_offset_name(comp_idx)} (Σ components)")
            else:
                component = cell_components[0]
                base = self._entity_rgb(component, comp_idx)
                col = tuple(self._light_rgb_from(
                    base, 0.0 if plain_color else self.darken_factor / 2.0))
                name = f"{component} {self._entity_label(component, comp_idx)}"
            # full pipeline: combine, per-curve unit-area normalization,
            # then the final amplitude sqrt
            y = _curve_values(x, states, self.sigma, self.amplitude)
            curves.append((comp_idx, name, states, col, y))
        # tallest curve in back, shortest in front (same rule as the sticks)
        curves.sort(key=lambda t: -float(t[4].max()))
        n = len(curves)
        for rank, (comp_idx, name, states, col, y) in enumerate(curves):
            zorder = 2.0 + 0.9 * rank / n
            (line,) = ax.plot(x, y, color=col, linewidth=1.8, zorder=zorder)
            artists.append(line)
            total = sum(p for _, p in states)
            hover[line] = (
                f"{name}\n"
                f"Gaussian-broadened spread\n"
                f"total ov {total:.5f} over {len(states)} states\n"
                f"normalized to unit area"
                + ("\nsqrt (amplitude) of the normalized curve"
                   if self.amplitude else ""))
            self._curve_entries.append(dict(ax=ax, line=line, lo=lo, hi=hi,
                                            states=states))
            peak = float(y.max()) if y.size else 0.0
            if peak > axis_max:
                axis_max = peak
        return axis_max

    def _draw_cell_content(self, ax, cell_band_indices, cell_components,
                           horizontal_on, segmentation_on,
                           draw_markers=True, plain_color=False):
        """
        Spread bars (+ MO marker lines unless draw_markers=False) for one
        axis — no reflines, title, or axis finalization. Returns (artists,
        hover, axis_max) so the single-run cell drawing and MultiSpreadPlotter
        rows share one path. plain_color=True draws the summed bars in the
        unmodified base color (multi-run mode, where the original MOs live on
        their own subplot instead of being shade-separated on this one).

        Bar z-order is per plotted energy: at each energy the tallest bar is
        drawn in the back and shorter bars stack toward the front, keeping
        every state visible at every energy point.
        """
        C = len(self.components)
        artists: List[Any] = []
        hover: Dict[Any, str] = {}
        axis_max = 0.0

        # ---- pass 1: collect every bar, grouped by plotted energy ----
        # bars_by_x[x] = [dict(comp_idx, full_idx, E_full, h, parts), ...]
        bars_by_x: Dict[float, List[Dict[str, Any]]] = {}

        # Kramers combining happens HERE, at weight assembly (data-level,
        # strictly upstream of the curve pipeline): per full state, each
        # component contributes the PAIR-SUMMED weight of its entity at this
        # offset. In Horizontal mode the order is: pair-sum within component
        # -> component-sum -> (curve pipeline: combine -> normalize ->
        # amplitude). The bar dicts keep the key name 'comp_idx' but carry
        # the entity OFFSET from here on.
        for comp_idx in cell_band_indices:
            if horizontal_on:
                # ---- sum components mapping to the same full_idx ----
                # contrib[full_idx] = (E_full, [(j, component, ov), ...]);
                # each ov is that component's pair-summed entity weight.
                contrib: Dict[int, Tuple[float, List[Tuple[int, str, float]]]] = {}
                for full_idx, comps in self.by_full.items():
                    for j, component in enumerate(cell_components):
                        ent = self._entity_for(component, comp_idx)
                        if ent is None:
                            continue
                        ov_c = None
                        E_full = None
                        for rec in comps.get(component, []):
                            if rec["comp_idx"] in ent.indices:
                                if ov_c is None:
                                    ov_c = 0.0
                                    E_full = rec["E"] + rec["dE"] + self.fermi_shift
                                ov_c += rec["ov"]
                        if ov_c is None:
                            continue
                        if full_idx not in contrib:
                            contrib[full_idx] = (E_full, [])
                        contrib[full_idx][1].append((j, component, ov_c))

                for full_idx, (E_full, parts) in contrib.items():
                    total = sum(ov for _, _, ov in parts)
                    h_total = self._yval(total)
                    bars_by_x.setdefault(round(E_full, 6), []).append(dict(
                        comp_idx=comp_idx, full_idx=full_idx,
                        E_full=E_full, h=h_total, p=total, parts=parts))
                    if h_total > axis_max:
                        axis_max = h_total
            else:
                # ---- horizontal off: cell holds exactly one component ----
                component = cell_components[0]
                ent = self._entity_for(component, comp_idx)
                if ent is None:
                    continue
                for full_idx, comps in self.by_full.items():
                    ov_c = None
                    E_full = None
                    for rec in comps.get(component, []):
                        if rec["comp_idx"] in ent.indices:
                            if ov_c is None:
                                ov_c = 0.0
                                E_full = rec["E"] + rec["dE"] + self.fermi_shift
                            ov_c += rec["ov"]
                    if ov_c is None:
                        continue
                    h = self._yval(ov_c)
                    bars_by_x.setdefault(round(E_full, 6), []).append(dict(
                        comp_idx=comp_idx, full_idx=full_idx,
                        E_full=E_full, h=h, p=ov_c, parts=None))
                    if h > axis_max:
                        axis_max = h

        # ---- pass 2: draw ----
        if self.curve_mode:
            # Curve mode: each MO (band index) becomes one continuous curve,
            # the sum of one area-normalized Gaussian per matched full state,
            # so the area under each Gaussian is that state's plotted
            # magnitude. Segmentation is a stick-mode-only feature and is
            # ignored here. The axis top comes from the curve peaks alone
            # (never the pass-1 stick heights) with 5% headroom so the
            # tallest apex is not clipped by the top spine; _update_curves
            # applies the identical rule on slider drags.
            axis_max = self._draw_cell_curves(
                ax, bars_by_x, cell_components, horizontal_on, plain_color,
                artists, hover)
            if axis_max > 0:
                axis_max *= 1.05
        else:
            # Within each energy group the tallest bar goes to the back and
            # shorter bars stack toward the front, so every state stays
            # visible at every energy point (exact ties in both energy and
            # intensity excepted).
            for recs in bars_by_x.values():
                recs.sort(key=lambda r: -r["h"])
                n = len(recs)
                for rank, rec in enumerate(recs):
                    zorder = 2.0 + 0.9 * rank / n
                    comp_idx = rec["comp_idx"]
                    if horizontal_on and segmentation_on:
                        # darkest (smallest j) on bottom, lightest on top.
                        # Cumulative boundaries go through _yval so the stack
                        # top stays at _yval(total) in amplitude mode.
                        parts_sorted = sorted(rec["parts"], key=lambda t: t[0])
                        y_bot = 0.0
                        for j, component, ov in parts_sorted:
                            y_top = y_bot + ov
                            base = self._entity_rgb(component, comp_idx)
                            col = tuple(self._light_rgb_from(
                                base, self._light_f_for(j, C)))
                            line = ax.vlines(
                                rec["E_full"],
                                self._yval(y_bot), self._yval(y_top),
                                color=col,
                                linewidth=self.cfg.lw_stick, zorder=zorder,
                            )
                            artists.append(line)
                            hover[line] = (
                                f"{component} "
                                f"{self._entity_label(component, comp_idx)}\n"
                                f"full_idx {rec['full_idx']}\n"
                                f"E_full {rec['E_full']:+.4f} eV\n"
                                f"{self._mag_name()} "
                                f"{self._yval(y_top) - self._yval(y_bot):.5f}"
                            )
                            y_bot = y_top
                    elif horizontal_on:
                        base = self._entity_rgb(
                            self._lead_component(cell_components, comp_idx),
                            comp_idx)
                        col = (tuple(base) if plain_color
                               else tuple(self._light_rgb_from(
                                   base, self.darken_factor / 2.0)))
                        line = ax.vlines(
                            rec["E_full"], 0, rec["h"], color=col,
                            linewidth=self.cfg.lw_stick, zorder=zorder,
                        )
                        artists.append(line)
                        hover[line] = (
                            f"full_idx {rec['full_idx']}\n"
                            f"E_full {rec['E_full']:+.4f} eV\n"
                            f"{self._mag_name()} {rec['h']:.5f}"
                        )
                    else:
                        component = cell_components[0]
                        base = self._entity_rgb(component, comp_idx)
                        col = tuple(self._light_rgb_from(
                            base, 0.0 if plain_color else self.darken_factor / 2.0))
                        line = ax.vlines(
                            rec["E_full"], 0, rec["h"], color=col,
                            linewidth=self.cfg.lw_stick, zorder=zorder,
                        )
                        artists.append(line)
                        hover[line] = (
                            f"full_idx {rec['full_idx']}\n"
                            f"E_full {rec['E_full']:+.4f} eV\n"
                            f"{self._mag_name()} {rec['h']:.5f}"
                        )

        # ---- markers (one per entity, at E_mean, behind all bars) ----
        for comp_idx in cell_band_indices:
            if not draw_markers:
                break
            if horizontal_on:
                E_m = self._avg_marker_E(cell_components, comp_idx)
                lead = self._lead_component(cell_components, comp_idx)
                base = self._entity_rgb(lead, comp_idx)
                marker_label = (
                    f"{self._entity_label(lead, comp_idx)}\n"
                    f"E {E_m:+.4f} eV (mean of {C})"
                    + self._entity_member_lines(lead, comp_idx)
                    if E_m is not None and lead is not None else None)
            else:
                component = cell_components[0]
                E_m = self._avg_marker_E([component], comp_idx)
                base = self._entity_rgb(component, comp_idx)
                marker_label = (
                    f"{component} {self._entity_label(component, comp_idx)}\n"
                    f"E {E_m:+.4f} eV"
                    + self._entity_member_lines(component, comp_idx)
                    if E_m is not None else None)
            if E_m is not None:
                # curve mode: marker spans the full axis height (the ylim is
                # the tallest curve peak, which the sigma slider rescales)
                marker_h = (axis_max if (self.curve_mode and axis_max > 0)
                            else 1.0)
                mline, mh = self._draw_marker(
                    ax, comp_idx, E_m, base, self._marker_z(comp_idx),
                    marker_label, height=marker_h)
                artists.append(mline)
                hover[mline] = mh
                if self.curve_mode:
                    self._marker_entries.append(dict(ax=ax, line=mline, E=E_m))

        return artists, hover, axis_max

    def _draw_cell(self, ax, cell_band_indices, cell_components,
                   horizontal_on, vertical_on, segmentation_on,
                   set_xlabel, set_ylabel_flag):
        artists, hover, axis_max = self._draw_cell_content(
            ax, cell_band_indices, cell_components,
            horizontal_on, segmentation_on)

        # ---- reflines: averaged when components are combined, else single ----
        if horizontal_on:
            self._draw_reflines(ax, self._avg_fermi_shifted(cell_components))
        else:
            self._draw_reflines(ax, self._comp_fermi_shifted(cell_components[0]))

        # ---- title: the text for this collapse combination, "Short: details" (the
        # layer shows the short name, the details go to the information panel) ----
        if vertical_on and horizontal_on:
            title = "Spread: all components, all entities"
        elif horizontal_on:
            comp_idx = cell_band_indices[0]
            E_m = self._avg_marker_E(cell_components, comp_idx)
            lead = self._lead_component(cell_components, comp_idx)
            title = (f"{self._entity_label(lead, comp_idx)}, E = {E_m:+.4f} eV"
                     if E_m is not None and lead is not None
                     else f"{_offset_name(comp_idx)} \u2014 not found")
        elif vertical_on:
            title = f"{cell_components[0]}"
        else:
            component = cell_components[0]
            comp_idx = cell_band_indices[0]
            E_m = self._avg_marker_E([component], comp_idx)
            title = (f"{component} {self._entity_label(component, comp_idx)}, "
                     f"E = {E_m:+.4f} eV"
                     if E_m is not None
                     else f"{component} {_offset_name(comp_idx)} \u2014 not found")

        self._finalize_axis(
            ax, axis_max, title,
            set_xlabel=set_xlabel, set_ylabel_flag=set_ylabel_flag)
        _attach_hover(self.cfg, ax, artists, hover)


class MultiSpreadPlotter:
    """
    Stacked figure over several full-system run directories, with a
    Horizontal check in its controls that toggles the layout (and the
    Amplitude and Curve checks, the σ box in curve mode).  DESIGN (inches):
    cfg.figsize[0] wide, 1.8 inches per row including the MO row (at least 3).

    Each run is loaded through its own SpreadPlotter (build_figure=False) so
    data and colors are identical to the single-run figure. Spread bars use
    the unmodified base color and carry no MO markers: the original molecule
    MOs instead get their own row at the very top, full-height sticks at the
    averaged molecule energies in the same base colors.

    Horizontal ON (initial): rows in the order given, each run collapsed
    onto a single full-width axis (components summed); the MO row averages
    every component of every run, and the red dashed line on every row is
    that global molecule-Fermi average.

    Horizontal OFF: every run splits into one equal-width subplot per
    component, left-aligned on a vertically aligned column grid (runs with
    fewer components leave blank slots to the right). The MO row splits the
    same way: column k averages only the k-th components of the runs below
    it, and the red dashed line of every subplot in column k is that
    column's averaged molecule Fermi.

    Energy alignment: all subplots share one x axis on which the runs'
    aligned vacuum potentials coincide, and the whole stack is shifted so
    the LAST run's full-system Fermi sits at 0 eV. Per subplot the
    reference lines are:

      dashed black      at 0 eV (global zero: last run's full-system Fermi)
      dotted black      at the subplot's own system Fermi: the run's
                        full-system Fermi (shifted), or on the MO row the
                        molecule Fermi itself (so it coincides with the red
                        line there, just as it coincides with the dashed
                        zero on the last run's row)
      dashed red        at the averaged original-molecule Fermi (global or
                        per-column, per the toggle)
      thick solid black at the aligned vacuum potential (same x everywhere)
    """

    def __init__(
        self,
        runs: List[Tuple[str, List[str]]],
        comp_indices: List[int],
        darken_factor: float = 0.4,
        cfg: Optional[PlotConfig] = None,
        collapse_horizontal: bool = True,
        amplitude: bool = False,
        curve: bool = False,
        sigma: float = SIGMA_DEFAULT,
    ):
        if not runs:
            raise ValueError("plot_spread_multi requires at least one (output_dir, components) entry")
        # Validate the shape of every entry up front: unpacking a stray
        # element below would otherwise fail with a bare "too many values to
        # unpack". The usual cause is a triple-quoted block used to comment
        # entries out INSIDE the list, which Python keeps as a string element
        # and which silently pushes a single-run setup down this path.
        for i, entry in enumerate(runs):
            ok = (not isinstance(entry, (str, bytes))
                  and isinstance(entry, (tuple, list))
                  and len(entry) == 2
                  and isinstance(entry[0], str))
            if not ok:
                hint = ""
                if isinstance(entry, (str, bytes)):
                    hint = ("  A triple-quoted block inside the list is a "
                            "string ELEMENT, not a comment - use '#' to "
                            "comment entries out.")
                raise TypeError(
                    f"runs[{i}] is not an (output_dir, [components]) pair: "
                    f"got {type(entry).__name__} {entry!r:.80}.{hint}")
        self.amplitude = bool(amplitude)
        self.curve_mode = bool(curve)
        self.sigma = float(sigma)
        self.runs = [(d, list(comps)) for d, comps in runs]
        self.comp_indices = list(comp_indices)
        self.darken_factor = float(darken_factor)
        self.cfg = cfg if cfg is not None else PlotConfig()

        # ---- load every run, figure-less ----
        self.plotters: List[SpreadPlotter] = []
        for output_dir, components in self.runs:
            sp = SpreadPlotter(
                output_dir=output_dir,
                components=components,
                comp_indices=self.comp_indices,
                collapse_vertical=True,
                collapse_horizontal=True,
                darken_factor=self.darken_factor,
                cfg=self.cfg,
                build_figure=False,
                amplitude=self.amplitude,
                curve=self.curve_mode,
                sigma=self.sigma,
            )
            if sp.v_aligned is None:
                raise ValueError(
                    f"{output_dir}: no aligned vacuum potential in the matcher "
                    "output; cannot vacuum-align this run in the stacked figure")
            self.plotters.append(sp)

        # ---- common energy scale: vacuums coincide, last run's Fermi at 0 ----
        sp_final = self.plotters[-1]
        fermi_final = sp_final.known_fermis.get("full", 0.0)
        self.v_common = sp_final.v_aligned - fermi_final
        for sp in self.plotters:
            sp.set_energy_shift(self.v_common - sp.v_aligned)

        # ---- averaged original-molecule energies / Fermi ----
        # Global averages (Horizontal ON: pool every component of every run)
        # and per-column averages (Horizontal OFF: column k pools the k-th
        # component of each run that has one), each component first placed on
        # its own shifted scale.
        self.max_C = max(len(comps) for _, comps in self.runs)
        self.col_sources: List[List[Tuple[SpreadPlotter, str]]] = [
            [(sp, comps[k]) for sp, (_, comps)
             in zip(self.plotters, self.runs) if k < len(comps)]
            for k in range(self.max_C)]

        def _avg_mo(sources):
            # b is a HOMO-relative entity offset; each source contributes
            # its entity's E_mean (one stick per entity, existing averaging
            # rules otherwise unchanged).
            vals: Dict[int, List[float]] = {b: [] for b in self.comp_indices}
            fvals: List[float] = []
            for sp, c in sources:
                fvals.append(sp.known_fermis.get(c, 0.0) + sp.fermi_shift)
                for b in self.comp_indices:
                    ent = sp._entity_for(c, b)
                    if ent is not None:
                        vals[b].append(ent.E_mean + sp.fermi_shift)
            energies = {b: sum(v) / len(v) for b, v in vals.items() if v}
            fermi = (sum(fvals) / len(fvals)) if fvals else 0.0
            return energies, fermi, len(fvals)

        all_sources = [(sp, c) for sp, (_, comps)
                       in zip(self.plotters, self.runs) for c in comps]
        self.mo_energies, self.mol_fermi_avg, self.n_mo_sources = _avg_mo(all_sources)
        per_col = [_avg_mo(sources) for sources in self.col_sources]
        self.col_mo_energies = [t[0] for t in per_col]
        self.col_fermi = [t[1] for t in per_col]

        # ---- persistent figure + the controls (made once; a toggle rebuilds) ----
        self.collapse_horizontal = bool(collapse_horizontal)
        R = len(self.plotters)
        self.DESIGN = (self.cfg.figsize[0], max(3.0, 1.8 * (R + 1)))
        self.fig = plt.figure(figsize=self.DESIGN)
        self._make_controls()
        self._build_ui()

    # ==================================================================
    # The controls (made once) and the UI build (clf, layout, hover)
    # ==================================================================
    def _make_controls(self):
        """
        The window's controls, real Qt widgets made once: the 'collapse' checks (Horizontal,
        Amplitude, Curve) and the σ (eV) number box (shown in curve mode), in a strip under the
        panels.  automatic layout: the one rule for every window (nothing placed by hand).
        """
        self.chk = controls.make(
            self.fig, 'collapse', labels=['Horizontal', 'Amplitude', 'Curve'],
            values=[self.collapse_horizontal, self.amplitude, self.curve_mode])
        self.chk.on_change(self._on_toggle)
        self.sld_sigma = controls.SigmaBox(self.fig, self.sigma, SIGMA_MIN, SIGMA_MAX)
        self.sld_sigma.on_change(self._on_sigma)
        self._sigma_box = controls.box(self.sld_sigma)
        strip = controls.FlowLayout()
        strip.addWidget(self.chk)
        strip.addWidget(self._sigma_box)
        self.fig.layout_root().addLayout(strip)

    def _build_ui(self):
        """
        The rows rebuilt for the current state: the figure cleared, the collapsed or split grid
        drawn again (automatic layout: fig.subplots fills the window's grid of cells), every
        panel's hover labels made anew.  The controls are never rebuilt.
        """
        self.fig.clf()
        widgets.HoverLabels.prune(self.fig)

        # Sync the shared drawing state onto every figure-less sub-plotter
        # and clear their curve registries before the rows redraw into them.
        for sp in self.plotters:
            sp.curve_mode = self.curve_mode
            sp.sigma = self.sigma
            sp._curve_entries = []
            sp._marker_entries = []

        # The σ box is shown, and takes input, only in curve mode.
        self._sigma_box.setVisible(self.curve_mode)
        self.sld_sigma.enable(self.curve_mode)

        if self.collapse_horizontal:
            self._draw_collapsed()
        else:
            self._draw_split()

        self._set_shared_xlims()
        self.fig.canvas.draw_idle()

    def _on_toggle(self, _label, states):
        """The checks reported a click: the window takes the three states and rebuilds."""
        self.collapse_horizontal, self.amplitude = states[0], states[1]
        self.curve_mode = states[2]
        for sp in self.plotters:
            sp.amplitude = self.amplitude
        self._build_ui()

    def _on_sigma(self, val):
        self.sigma = float(val)
        for sp in self.plotters:
            sp.sigma = self.sigma
            sp._update_curves()

    # ==================================================================
    # Horizontal ON: MO row on top, one collapsed row per run
    # ==================================================================
    def _draw_collapsed(self):
        R = len(self.plotters)
        self.axes = self.fig.subplots(R + 1, 1, sharex=True, squeeze=False)

        self._draw_mo_axis(
            self.axes[0][0], self.mo_energies, self.mol_fermi_avg,
            self.n_mo_sources, self.col_sources[0][0],
            set_ylabel_flag=True,
            title="Original MOs: averaged over all runs")

        for i, sp in enumerate(self.plotters):
            ax = self.axes[i + 1][0]
            components = self.runs[i][1]
            artists, hover, axis_max = sp._draw_cell_content(
                ax, self.comp_indices, components,
                horizontal_on=True, segmentation_on=False,
                draw_markers=False, plain_color=True)
            self._draw_reflines(
                ax, sp.known_fermis.get("full", 0.0) + sp.fermi_shift,
                self.mol_fermi_avg)
            ax.set_ylim(0, axis_max if axis_max > 0 else 1.0)
            ax.set_ylabel(sp._ylabel())
            ax.set_title(os.path.basename(os.path.normpath(self.runs[i][0])))
            if i == R - 1:
                ax.set_xlabel(self.cfg.xlabel)
            _attach_hover(self.cfg, ax, artists, hover)

    # ==================================================================
    # Horizontal OFF: every run splits into one equal-width subplot per
    # component (left-aligned; unused grid slots are hidden), and the MO
    # row splits likewise, each column averaging only the components below
    # it and carrying that column's averaged molecule Fermi in red.
    # ==================================================================
    def _draw_split(self):
        R = len(self.plotters)
        n_comps = [len(comps) for _, comps in self.runs]
        self.axes = self.fig.subplots(
            R + 1, self.max_C, sharex=True, squeeze=False)

        for k in range(self.max_C):
            self._draw_mo_axis(
                self.axes[0][k], self.col_mo_energies[k], self.col_fermi[k],
                len(self.col_sources[k]), self.col_sources[k][0],
                set_ylabel_flag=(k == 0),
                title="Original MOs: column average")

        for i, sp in enumerate(self.plotters):
            run_dir, comps = self.runs[i]
            own_fermi = sp.known_fermis.get("full", 0.0) + sp.fermi_shift
            run_name = os.path.basename(os.path.normpath(run_dir))
            for k in range(self.max_C):
                ax = self.axes[i + 1][k]
                if k >= len(comps):
                    ax.axis("off")
                    continue
                artists, hover, axis_max = sp._draw_cell_content(
                    ax, self.comp_indices, [comps[k]],
                    horizontal_on=False, segmentation_on=False,
                    draw_markers=False, plain_color=True)
                self._draw_reflines(ax, own_fermi, self.col_fermi[k])
                ax.set_ylim(0, axis_max if axis_max > 0 else 1.0)
                if k == 0:
                    ax.set_ylabel(sp._ylabel())
                ax.set_title(f"{run_name} — {comps[k]}")
                if all(n_comps[j] <= k for j in range(i + 1, R)):
                    ax.set_xlabel(self.cfg.xlabel)
                _attach_hover(self.cfg, ax, artists, hover)

    # ------------------------------------------------------------------
    def _set_shared_xlims(self):
        # all subplots share x, so setting the first sets all
        ax0 = self.axes[0][0]
        if self.cfg.energy_range:
            ax0.set_xlim(self.cfg.energy_range)
        else:
            mins = [sp.shared_e_min for sp in self.plotters
                    if sp.shared_e_min is not None]
            if mins:
                ax0.set_xlim(left=min(mins))
            ax0.set_xlim(right=self.v_common)

    # ------------------------------------------------------------------
    def _draw_mo_axis(self, ax, mo_energies, mol_fermi, n_sources,
                      color_src, set_ylabel_flag, title):
        """
        One original-MO axis: a full-height stick per band index at the
        averaged molecule energy, in the plain base color. color_src is the
        (SpreadPlotter, component) whose color map supplies the stick colors
        (the pooled components describe the same molecule, so their maps are
        essentially identical).
        """
        artists: List[Any] = []
        hover: Dict[Any, str] = {}
        sp0, comp0 = color_src
        for comp_idx in self.comp_indices:
            E_m = mo_energies.get(comp_idx)
            if E_m is None:
                continue
            col = tuple(sp0._entity_rgb(comp0, comp_idx))
            line = ax.vlines(
                E_m, 0, 1.0, color=col,
                linewidth=self.cfg.lw_stick + 1.0,
                zorder=sp0._marker_z(comp_idx))
            artists.append(line)
            hover[line] = (
                f"{sp0._entity_label(comp0, comp_idx)}\n"
                f"E {E_m:+.4f} eV (mean of {n_sources})"
                + sp0._entity_member_lines(comp0, comp_idx))
        # own system Fermi of the molecule row IS the molecule Fermi, so the
        # dotted black line lands on the red dashed line here by design
        self._draw_reflines(ax, mol_fermi, mol_fermi)
        ax.set_ylim(0, 1.0)
        if set_ylabel_flag:
            ax.set_ylabel(self._mo_ylabel())
        ax.set_title(title)
        _attach_hover(self.cfg, ax, artists, hover)

    # ------------------------------------------------------------------
    def _mo_ylabel(self) -> str:
        # amplitude mode drops the trailing ^2, mirroring SpreadPlotter._ylabel
        return MO_YLABEL[:-3] + "$" if self.amplitude else MO_YLABEL

    # ------------------------------------------------------------------
    def _draw_reflines(self, ax, own_fermi_shifted: float, mol_fermi: float):
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.9, zorder=10)
        ax.axvline(own_fermi_shifted, color="black", linestyle=":", alpha=0.9, zorder=10)
        ax.axvline(
            mol_fermi, color="red",
            linestyle="--", alpha=0.9, zorder=10)
        ax.axvline(
            self.v_common, color="black",
            linestyle="-", linewidth=3.5, zorder=10)



def plot_spread(
    output_dir: str,
    components: List[str],
    comp_indices: List[int],
    collapse_vertical: bool = False,
    collapse_horizontal: bool = False,
    darken_factor: float = 0.4,
    cfg: Optional[PlotConfig] = None,
    amplitude: bool = False,
    curve: bool = False,
    sigma: float = SIGMA_DEFAULT,
) -> Tuple[plt.Figure, "SpreadPlotter"]:
    """
    Build the interactive spread-plot figure. The Vertical / Horizontal checks
    in the controls under the panels toggle the collapse live;
    collapse_vertical / collapse_horizontal set the initial check state.

    Parameters
    ----------
    output_dir : str
        Directory containing band_matches_rectangular.txt and
        band_matches_rectangular_all.txt (the full_dir from the matcher).
    components : list of str
        Component labels. The same comp_indices are used for every component.
    comp_indices : list of int
        Combined-entity HOMO-relative offsets, shared across all components
        and computed per component (0 = HOMO entity, -1 = HOMO-1,
        +1 = LUMO; "metal" anchors on its own HOMO index). Absolute band
        indices are no longer accepted.
    collapse_vertical, collapse_horizontal : bool
        Initial state of the in-figure toggles.
    darken_factor : float
        Controls the light/dark split and the per-component lightness offset
        (see SpreadPlotter).
    cfg : PlotConfig or None
        Plot configuration. Uses defaults if None.
    curve : bool
        Initial state of the Curve toggle: replace the per-state sticks with
        Gaussian-broadened curves (one per MO), with the σ (eV) number box
        shown. Each probability curve is normalized to unit area; with
        Amplitude on the displayed curve is the sqrt of that normalized
        curve (the final step, applied after normalization).
    sigma : float
        Initial Gaussian smearing width in eV (VASP SIGMA convention).

    Returns
    -------
    fig, sp
        The qtplot Figure and the SpreadPlotter instance (kept so the
        controls stay responsive; do not let it be garbage-collected).
    """
    sp = SpreadPlotter(
        output_dir=output_dir,
        components=components,
        comp_indices=comp_indices,
        collapse_vertical=collapse_vertical,
        collapse_horizontal=collapse_horizontal,
        darken_factor=darken_factor,
        cfg=cfg,
        amplitude=amplitude,
        curve=curve,
        sigma=sigma,
    )
    return sp.fig, sp


def plot_spread_multi(
    runs: List[Tuple[str, List[str]]],
    comp_indices: List[int],
    darken_factor: float = 0.4,
    cfg: Optional[PlotConfig] = None,
    collapse_horizontal: bool = True,
    amplitude: bool = False,
    curve: bool = False,
    sigma: float = SIGMA_DEFAULT,
) -> Tuple[plt.Figure, "MultiSpreadPlotter"]:
    """
    Build the stacked multi-run spread figure (see MultiSpreadPlotter). The
    Horizontal check in its controls toggles between the collapsed
    one-axis-per-run stack and the per-component column grid live.

    Parameters
    ----------
    runs : list of (output_dir, components)
        Run directories (each containing the matcher outputs) paired with
        their component labels, in top-to-bottom row order. The LAST entry
        defines the global zero: after the runs are aligned on their vacuum
        potentials, the whole stack is shifted so that run's full-system
        Fermi sits at 0 eV.
    comp_indices : list of int
        Combined-entity HOMO-relative offsets, shared across all runs and
        components and computed per component (0 = HOMO entity, -1 = HOMO-1,
        +1 = LUMO). Absolute band indices are no longer accepted.
    darken_factor : float
        As in plot_spread.
    cfg : PlotConfig or None
        Plot configuration. Uses defaults if None.
    collapse_horizontal : bool
        Initial state of the Horizontal toggle.
    curve : bool
        Initial state of the Curve toggle: replace the per-state sticks with
        Gaussian-broadened curves (area under each Gaussian = that state's
        intensity), with the σ (eV) number box shown. Each probability
        curve is normalized to unit area; with Amplitude on the displayed
        curve is the sqrt of that normalized curve.
    sigma : float
        Initial Gaussian smearing width in eV (VASP SIGMA convention).

    Returns
    -------
    fig, msp
        The qtplot Figure and the MultiSpreadPlotter instance (keep a
        reference so the toggles and hover labels stay responsive).
    """
    msp = MultiSpreadPlotter(
        runs=runs,
        comp_indices=comp_indices,
        darken_factor=darken_factor,
        cfg=cfg,
        collapse_horizontal=collapse_horizontal,
        amplitude=amplitude,
        curve=curve,
        sigma=sigma,
    )
    return msp.fig, msp


# ======================================================================
if __name__ == "__main__":

    import sys, types
    _pkg_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_pkg_dir, ".."))
    # Stub package so relative imports resolve without __init__.py
    _pkg = types.ModuleType("matcher")
    _pkg.__path__ = [_pkg_dir]
    sys.modules.setdefault("matcher", _pkg)
    from matcher.qtgui import qtplot as plt
    from matcher.qtgui import controls, widgets
    from matcher.vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _read_ov_all,
    )
    from matcher.kramers import pair_entities, homo_entity_offsets

    # ---- Runs: (full_dir, [component labels]), in top-to-bottom order ----
    # One entry  -> the interactive single-run figure with layout toggles.
    # Multiple   -> the static stacked figure: averaged original-MO subplot
    #               on top, then every run collapsed horizontally +
    #               vertically onto one axis per run (plain colors, no MO
    #               markers), vacuum potentials aligned across runs, global
    #               zero (dashed black) at the LAST run's full-system Fermi,
    #               each row's own system Fermi dotted black, averaged
    #               molecule Fermi red dashed on every row.
    # NB: comment unused entries out with '#', one per line. A triple-quoted
    # block placed inside the list is a STRING ELEMENT of the list, not a
    # comment, so it silently makes len(runs) larger and sends a single-run
    # setup down the multi-run path.
    '''
    runs = [
        (r'dir/run_1',["NHC"]),
    ]
    '''
    '''
    runs = [
        
        (r'dir/run_complex', ["NHC_c1", "NHC_c2"]),
        (r'dir/run_surface', ["NHC_1", "NHC_2"]),
        (r'dir/run_sam', ["NHC_c1m1", "NHC_c2m1", "NHC_c3m1", "NHC_c4m1", "NHC_c1m2", "NHC_c2m2", "NHC_c3m2", "NHC_c4m2"]),
    ]
    '''
    #'''
    runs = [
        (r'dir/run_1',['NHC_complex']),
        (r'dir/run_2', ['NHC']),
        (r'dir/run_3',['NHC_c1m1', 'NHC_c2m1']),
        (r'dir/run_4',['NHC_c1m1', 'NHC_c2m1', 'NHC_c3m1', 'NHC_c4m1'])
        ]
    #'''
    # ---- Which entities to plot: HOMO-relative offsets, shared by all
    # runs/components (0 = HOMO entity, -1 = HOMO-1, +1 = LUMO) ----
    comp_indices = [-2, -1, 0, 1]

    # ---- Initial toggle state (flip live with the checks under the panels) ----
    collapse_vertical = True
    collapse_horizontal = True
    amplitude = True        # sticks: sqrt(ov); curves: sqrt of the
                            # unit-area probability curve (applied last)
    curve = True            # True = Gaussian-broadened curves (σ box)
    sigma = 0.2              # initial smearing width in eV (VASP SIGMA style)

    # ---- Visual tuning ----
    darken_factor = 0.4                  # 0 = base color, 1 = black

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

    if len(runs) == 1:
        output_dir, components = runs[0]
        fig, sp = plot_spread(
            output_dir=output_dir,
            components=components,
            comp_indices=comp_indices,
            collapse_vertical=collapse_vertical,
            collapse_horizontal=collapse_horizontal,
            darken_factor=darken_factor,
            cfg=cfg,
            amplitude=amplitude,
            curve=curve,
            sigma=sigma,
        )
    else:
        fig, msp = plot_spread_multi(
            runs=runs,
            comp_indices=comp_indices,
            darken_factor=darken_factor,
            cfg=cfg,
            amplitude=amplitude,
            curve=curve,
            sigma=sigma,
        )
    plt.show()