# -*- coding: utf-8 -*-
"""
decomp_window.py — the per-marker LDOS + og-MO decomposition figure
(Module B, §8.4).

Markers are stacked vertically as blocks (spread_collapse multi-run
pattern), marker order preserved, shared energy axis. Within every block
the layout is governed by ONE shared set of three CheckButtons — Vertical,
Horizontal, Amplitude — functioning exactly as spread_collapse's curve-mode
toggles. Vertical and Horizontal DEFAULT ON; Amplitude DEFAULTS OFF: the
window opens fully collapsed, probability-scale. There is no Curve toggle
and no stick mode: always curves at the committed σ.

Grid mapping inside a marker block (spread_collapse semantics):
rows = HOMO-relative offsets (union of the kept entities' offsets),
columns = molecule components (only components with ≥1 kept entity appear).
  V ON + H ON  (default): a single axis — every kept MO curve of every
               component overlaid, plus the LDOS.
  V ON only:   one column per component; that component's kept entities
               overlaid on a single axis, plus the LDOS.
  H ON only:   one axis per offset; every component's entity at that
               offset overlaid, plus the LDOS, with the per-component
               lightness offset (same hue).
  both OFF:    the full offsets × components grid; each cell shows that
               single entity's curve plus the LDOS; a cell whose entity
               was not kept shows the LDOS alone.
The LDOS curve is drawn on EVERY axis in every mode, always at the same
scale. Curve data are identical in every mode: α is computed once per
button press and a layout change never rescales anything.

Energy frame: the simulator's axis (DOSCAR-E_F-referenced). MO stick
energies are meta/e_raw_full − E_F(DOSCAR header) — NOT
E_full − final_fermis["full"] — so a fermi_mid-style matcher Fermi can
never offset the overlay from the simulator axis.

MO set: automatic — all molecule components' MOs, Kramers-combined via the
shared kramers utility, no manual index lists.

Spread curves: W_ent(E) = Σ_j p_ent(j) g_σ(E−E_j) at the committed σ (the
same σ that built the synthetic DOSCAR), evaluated on the snapshot's LDOS
energy grid (clipped to the LDOS window by construction).

Threshold (noise filter): in-window area a_ent = trapz(W_ent, window);
keep iff a_ent ≥ MO_AREA_NOISE_THRESHOLD (module constant, default 1e-4),
adjustable via the small textbox on this window; re-filter and redraw on
commit. The kept set is computed once per press and is common to all
marker blocks.

Display scaling (user-directed revision of the original common-α rule):
each kept MO spread curve is scaled by ITS OWN projection onto the
marker's LDOS curve — the normalized overlap

    p_a^(m) = ⟨W_a, LDOS_m⟩ / (‖W_a‖ · ‖LDOS_m‖)      (window grid)

computed INDEPENDENTLY per (entity, marker): overlapping MOs never
compete for weight (a joint least-squares decomposition was tried first
and rejected — it zeroes near-duplicate curve shapes, so entities that
survived the area threshold could draw flat).

All curves are L2-unit-normalized BEFORE any overlap; every projection
is taken against the FULL unit-normed LDOS: p_a = ⟨L̂, Ŵ_a⟩ per
(entity, marker).

Metal curve (user-directed; display-only toggle via the "Metal curve"
CheckButton in the bottom strip, between the area textbox and the Save
data button — ON by default): a PER-MARKER composite of every
independent metal state's spread curve. Each metal state n has its own
spread W_n(E) = Σ_j grid_metal[n,j]·g_σ(E−E_j) (the matcher's
grid/metal rows, smeared at the committed σ on the window grid) and its
own normalized overlap with the marker's LDOS, q_n = ⟨L̂, Ŵ_n⟩; the
composite is

    C_m(E) = Σ_n q_n² · W_n(E)

(each state weighted by its squared overlap — the probability
weighting, matching how every curve in this window scales), so the
metal curve is SHAPED differently for each LDOS curve. The composite's
own projection p_met = ⟨L̂, Ĉ_m⟩ is computed exactly like an MO
projection, and the drawn curve is the projection component of the
LDOS along it, ‖L‖·p_met·Ĉ_m (thin dashed, in the marker's color),
with p_met²/p_met in its legend entry like the MO curves. The toggle
only shows/hides this curve — nothing is ever subtracted from the LDOS
and no projection changes with the toggle.

Every axis carries the marker's LDOS (thick solid, marker color) and,
when toggled on, the metal curve; every axis gets a legend in every
layout mode.

    probability view (Amplitude OFF):
        MO curve a:  β_m · p_a² · W_a(E)         LDOS: as computed
    magnitude view (Amplitude ON):
        the pointwise SQUARE ROOT of the probability view — every drawn
        curve, the LDOS, the metal curve, AND each MO spread curve, is
        displayed as the sqrt of its probability-scale values, so each
        MO curve is scaled by |p_a| (times the shared sqrt shape)

with one block factor β_m = A_LDOS,m / Σ_kept p_a²·a_ent so the kept
curves' summed area equals the LDOS curve's area in the probability
view; the drawn-area identities deliberately do not hold in the sqrt
view. Projections are computed once per press (Send or threshold
commit); layout, Amplitude, and Metal-curve toggles never recompute
them. If no entity passes the threshold anywhere, each block is a
single axis with the LDOS (and metal curve, if toggled on) plus the
annotation "no MOs above threshold". The extended matcher record
(grid_amp/, cross/ incl. S_mm) is REQUIRED — the metal curve needs the
Module A channel.

Exports are always the UNSCALED probability-scale W_ent columns
regardless of the toggles; the per-marker projections (incl. p_met) and
the block factor ride in the header, so the probability view is exactly
reconstructable and the magnitude view is its pointwise sqrt.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Button, CheckButtons, TextBox

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

if __name__ != "__main__":
    from ._imports import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _gauss_sum,
        pair_entities,
        homo_entity_offsets,
    )
    from .synthetic_doscar import read_doscar_header

# Near-zero numerical-noise filter on the in-window entity area (eV⁻¹·eV).
MO_AREA_NOISE_THRESHOLD = 5e-2

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _offset_name(offset: int) -> str:
    if offset == 0:
        return "HOMO"
    if offset < 0:
        return f"HOMO−{-offset}"
    if offset == 1:
        return "LUMO"
    return f"LUMO+{offset - 1}"


def load_static_mo_data(run_dir: str, cfg: PlotConfig) -> Dict[str, Any]:
    """
    Static (per run) MO data shared by the decomposition windows: the
    prepared plotter bundle, per-component Kramers entities with their
    full-precision grid weights and colors, the metal grid rows, and the
    stick energies on the simulator's axis (DOSCAR E_F at zero).

    Single load path for DecompWindow and TopoDecompWindow — the entity
    set, weights, colors, and energy frame can never diverge between the
    two figures.
    """
    results_path = os.path.join(run_dir, "matcher_results.h5")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(
            f"{results_path} is required for the decomposition window "
            "(meta/e_raw_full and the full-precision grid).")

    plotter = RectAEPAWColorPlotter(cfg)
    main_path = os.path.join(run_dir, "band_matches_rectangular.txt")
    P = plotter._prepare(main_path, bonding=False, write_summaries=False)

    with h5py.File(results_path, "r") as hf:
        comp_labels = json.loads(hf["meta"]["comp_labels_json"][()])
        e_raw_full = hf["meta"]["e_raw_full"][...].astype(np.float64)
        grid = {l: hf["grid"][l][...] for l in comp_labels}
    if "metal" not in grid:
        raise ValueError(
            f"{results_path} has no grid/metal — the metal composite "
            "curve needs the metal spread rows.")
    # The grid columns span the matcher's full-band window (bands below
    # the vacuum level); e_raw_full spans ALL bands. Slice e_raw_full to
    # the window length from the front (the window is a prefix of the
    # band list because it is where(e_raw < v_vac), which is contiguous
    # from band 1 for physical spectra; assert to be safe).
    n_full = grid[comp_labels[0]].shape[1]
    if n_full > e_raw_full.shape[0]:
        raise ValueError("grid has more columns than e_raw_full entries")
    e_raw_win = e_raw_full[:n_full]

    header = read_doscar_header(os.path.join(run_dir, "DOSCAR"))
    ef_doscar = float(header["efermi"])
    # MO stick energies on the simulator's axis (DOSCAR E_F at zero).
    E_sticks = e_raw_win - ef_doscar

    # Automatic MO set: every molecule component's entities.
    mol_labels = [l for l in comp_labels if l.lower() != "metal"]
    entities_by_comp: Dict[str, Dict[int, Any]] = {}
    entity_rows: List[Dict[str, Any]] = []
    for label in mol_labels:
        ents = pair_entities(
            P.comp_pairs.get(label, []), P.is_soc,
            cfg.soc_pair_thresh_eV,
            center_idx=P.known_homos.get(label))
        try:
            offmap = homo_entity_offsets(ents, P.known_homos.get(label))
        except ValueError:
            offmap = {}
        entities_by_comp[label] = offmap
        G = grid[label]
        for offset, ent in sorted(offmap.items()):
            w = np.zeros(n_full, dtype=np.float64)
            for i in ent.indices:
                w += G[i - 1, :]
            color = P.component_colors.get(label, {}).get(
                ent.rep, (0.4, 0.4, 0.4, 1.0))
            entity_rows.append(dict(
                comp=label, offset=offset, ent=ent, weights=w,
                color=np.array(color[:3]),
                label=f"{label} {_offset_name(offset)} "
                      f"({'+'.join(str(i) for i in ent.indices)})",
            ))
    return dict(
        P=P, metal_grid=grid["metal"], e_raw_win=e_raw_win,
        ef_doscar=ef_doscar, E_sticks=E_sticks, mol_labels=mol_labels,
        entities_by_comp=entities_by_comp, entity_rows=entity_rows,
    )


class DecompWindow:
    """
    Snapshot-semantics decomposition window: reflects the moment of the
    "Send to decomposition" press (markers, σ, window, threshold); it
    never live-tracks the simulator; pressing again regenerates it via
    refresh().
    """

    def __init__(
        self,
        run_dir: str,
        snapshot: Dict[str, Any],
        cfg: Optional[PlotConfig] = None,
        threshold: float = MO_AREA_NOISE_THRESHOLD,
        darken_factor: float = 0.4,
        show_metal: bool = True,
    ):
        self.run_dir = run_dir
        self.cfg = cfg if cfg is not None else PlotConfig()
        self.threshold = float(threshold)
        self.darken_factor = float(darken_factor)
        # Metal-curve toggle (also on the figure, bottom strip): purely a
        # display switch for the metal-(sans-molecule) projection curve —
        # no projection ever changes with it.
        self.show_metal = bool(show_metal)

        # Layout toggles (spread_collapse curve-mode semantics): Vertical
        # and Horizontal default ON, Amplitude default OFF.
        self.collapse_vertical = True
        self.collapse_horizontal = True
        self.amplitude = False

        self._load_static()
        self._apply_snapshot(snapshot)

        self.cursor_by_axes: Dict[Any, Any] = {}
        self.fig = plt.figure(
            figsize=(self.cfg.figsize[0],
                     max(4.0, 2.2 * max(1, len(self.markers)))))
        self._build_ui()

    # ------------------------------------------------------------------
    # Static (per run) data: entities, colors, raw stick energies/weights.
    # ------------------------------------------------------------------
    def _load_static(self):
        d = load_static_mo_data(self.run_dir, self.cfg)
        self.P = d["P"]
        # Every independent metal state's per-full-state weights (rows of
        # grid/metal, full precision): the composite metal curve is built
        # from their σ-smeared spreads in _apply_snapshot.
        self.metal_grid = d["metal_grid"]
        self._e_raw_win = d["e_raw_win"]
        self.ef_doscar = d["ef_doscar"]
        self.E_sticks = d["E_sticks"]
        self.mol_labels = d["mol_labels"]
        self.entities_by_comp = d["entities_by_comp"]
        self.entity_rows = d["entity_rows"]

    # ------------------------------------------------------------------
    # Snapshot: markers, LDOS curves, σ, energy window.
    # ------------------------------------------------------------------
    def _apply_snapshot(self, snapshot: Dict[str, Any]):
        self.snapshot = snapshot
        self.sigma = float(snapshot["sigma"])
        self.eg = np.asarray(snapshot["eg"], dtype=np.float64)
        self.markers = list(snapshot["markers"])
        self.ldos_curves = [np.asarray(y, dtype=np.float64)
                            for y in snapshot["ldos"]]
        self.marker_colors = list(snapshot.get(
            "marker_colors", ["black"] * len(self.markers)))
        if len(self.ldos_curves) != len(self.markers):
            raise ValueError("snapshot markers and LDOS curves differ in count")

        # Spread curves at the committed σ on the LDOS window grid, and
        # their in-window areas (threshold statistic; marker-independent).
        for row in self.entity_rows:
            states = list(zip(self.E_sticks.tolist(),
                              row["weights"].tolist()))
            W = _gauss_sum(self.eg, states, self.sigma)
            row["W"] = W
            row["a_ent"] = float(_TRAPZ(W, self.eg))

        # Every metal state's spread curve at the committed σ: one kernel
        # row per full state (each a literal _gauss_sum evaluation), then
        # W_n = grid_metal[n] · G for all states in one matrix product.
        G = np.empty((self.E_sticks.size, self.eg.size), dtype=np.float64)
        for j, E0 in enumerate(self.E_sticks):
            G[j] = _gauss_sum(self.eg, [(float(E0), 1.0)], self.sigma)
        self.metal_spreads = self.metal_grid @ G      # (n_metal, nE)
        norms = np.linalg.norm(self.metal_spreads, axis=1)
        self.metal_norms = np.where(norms > 0, norms, 1.0)
        self._metal_zero = norms <= 0
        self._recompute_kept()

    def _recompute_kept(self):
        """Kept set + per-marker projection coefficients and block factors:
        computed once per press (Send press or threshold commit); the kept
        set is common to all marker blocks, the coefficients are per
        marker."""
        self.kept = [row for row in self.entity_rows
                     if row["a_ent"] >= self.threshold]
        self.kept_offsets = sorted({row["offset"] for row in self.kept})
        self.kept_comps = [c for c in self.mol_labels
                           if any(r["comp"] == c for r in self.kept)]
        self.A_ldos = [float(_TRAPZ(y, self.eg)) for y in self.ldos_curves]

        # Projections, all curves L2-unit-normalized BEFORE any overlap
        # (on the uniform window grid the sampled-value inner products
        # equal the dE-weighted ones up to a common factor that cancels
        # in every p). Per marker m, every projection is taken against
        # the FULL unit-normed LDOS: p_a = ⟨L̂, Ŵ_a⟩ per kept entity, and
        # p_met = ⟨L̂, Ŵ_met⊥⟩ for the (display-only) metal curve — each
        # curve projected INDEPENDENTLY, so overlapping curves never
        # compete for weight.
        self.proj_coeffs = []      # per marker: ndarray over kept rows
        self.beta_prob = []        # per marker: A_LDOS / Σ p²·a_ent
        self.p_met = []            # per marker: composite's LDOS share
        self.metal_curves = []     # per marker: ‖L‖·p_met·Ĉ_m
        if self.kept:
            design = np.column_stack([row["W"] for row in self.kept])
            w_norms = np.linalg.norm(design, axis=0)
            w_safe = np.where(w_norms > 0, w_norms, 1.0)
            a_ent = np.array([row["a_ent"] for row in self.kept])
        for m, y in enumerate(self.ldos_curves):
            y = np.asarray(y, dtype=np.float64)
            l_norm = float(np.linalg.norm(y))
            L_hat = y / l_norm if l_norm > 0 else np.zeros_like(y)
            # Per-marker metal composite: every metal state's spread,
            # weighted by its squared normalized overlap with THIS LDOS —
            # C_m = Σ_n q_n²·W_n, so the shape follows the LDOS.
            q = (self.metal_spreads @ L_hat) / self.metal_norms
            q[self._metal_zero] = 0.0
            composite = (q ** 2) @ self.metal_spreads
            c_norm = float(np.linalg.norm(composite))
            C_hat = (composite / c_norm if c_norm > 0
                     else np.zeros_like(composite))
            pm = float(L_hat @ C_hat)
            self.p_met.append(pm)
            self.metal_curves.append(l_norm * pm * C_hat)
            if not self.kept:
                self.proj_coeffs.append(np.zeros(0))
                self.beta_prob.append(0.0)
                continue
            c = (design.T @ L_hat) / w_safe
            c[w_norms <= 0] = 0.0
            self.proj_coeffs.append(c)
            s_prob = float(np.sum(c ** 2 * a_ent))
            self.beta_prob.append(self.A_ldos[m] / s_prob if s_prob > 0 else 0.0)

    def _row_factor(self, m: int, k: int) -> float:
        """Probability-scale factor for kept row k in marker block m:
        β_m·p². The Amplitude view is the pointwise sqrt of the resulting
        probability curve (applied in _disp), so the drawn magnitude curve
        is scaled by |p|. Projections are never recomputed here."""
        p = float(self.proj_coeffs[m][k])
        return self.beta_prob[m] * (p ** 2)

    def _disp(self, y: np.ndarray) -> np.ndarray:
        """Amplitude is the FINAL display step: the pointwise square root
        of the ABSOLUTE probability-scale values, applied to the LDOS and
        every MO spread curve alike; exports never use it. (abs, not a
        clamp-to-zero: a real DOSCAR's Methfessel–Paxton smearing dips
        slightly negative, and clamping drew those stretches flat at 0.)"""
        if self.amplitude:
            return np.sqrt(np.abs(y))
        return y

    # ------------------------------------------------------------------
    def refresh(self, snapshot: Dict[str, Any]):
        """Regenerate from a new press (snapshot semantics)."""
        self._apply_snapshot(snapshot)
        self._build_ui()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _light_f_for(self, j: int, n: int) -> float:
        # Per-component toward-white blend fraction (spread_collapse
        # convention): n components spread evenly across [0, darken_factor].
        if n <= 1:
            return 0.0
        return self.darken_factor * j / (n - 1)

    def _comp_color(self, row: Dict[str, Any], multi_comp_axis: bool) -> Tuple:
        base = row["color"]
        if multi_comp_axis and len(self.kept_comps) > 1:
            j = self.kept_comps.index(row["comp"])
            f = self._light_f_for(j, len(self.kept_comps))
            base = base + (1.0 - base) * f
        return tuple(base)

    # ------------------------------------------------------------------
    # UI build (persistent figure; fig.clf() rebuild; toggles re-layout
    # from the snapshot data only — LDOS and W_ent are never recomputed
    # by a toggle, and α is never rescaled by one)
    # ------------------------------------------------------------------
    def _build_ui(self):
        for w in (getattr(self, "chk", None), getattr(self, "tb_thresh", None),
                  getattr(self, "chk_metal", None),
                  getattr(self, "btn_save", None)):
            if w is not None:
                w.disconnect_events()
        self.fig.clf()
        self.cursor_by_axes = {}

        self.chk = CheckButtons(
            self.fig.add_axes([0.01, 0.02, 0.11, 0.12]),
            ["Vertical", "Horizontal", "Amplitude"],
            [self.collapse_vertical, self.collapse_horizontal,
             self.amplitude],
        )
        self.chk.on_clicked(self._on_toggle)

        ax_tb = self.fig.add_axes([0.30, 0.03, 0.10, 0.035])
        self.tb_thresh = TextBox(ax_tb, "area ≥ ",
                                 initial=f"{self.threshold:.2e}")
        self.tb_thresh.on_submit(self._on_threshold)

        # Metal-curve display toggle, between the area textbox and Save data.
        self.chk_metal = CheckButtons(
            self.fig.add_axes([0.50, 0.02, 0.20, 0.06]),
            ["Metal curve"], [self.show_metal])
        self.chk_metal.on_clicked(self._on_metal_toggle)

        self.btn_save = Button(self.fig.add_axes([0.84, 0.025, 0.12, 0.045]),
                               "Save data")
        self.btn_save.on_clicked(lambda _e: self.save_data())

        self.fig.subplots_adjust(bottom=0.18, hspace=0.9)
        self._draw_blocks()

        # Figure-level click handler for hover enable/disable (per axis;
        # spread_collapse pattern: left click enables, right click clears
        # and disables).
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.draw_idle()

    def _attach_cursor(self, ax, artists, hover_map):
        if HAS_MPLCURSORS and artists:
            cur = mplcursors.cursor(artists, hover=True)
            cur.enabled = False
            self.cursor_by_axes[ax] = cur

            @cur.connect("add")
            def _on_add(sel, hmap=hover_map):
                txt = hmap.get(sel.artist)
                if txt:
                    sel.annotation.set_text(txt)

    def _on_click(self, event):
        if not HAS_MPLCURSORS:
            return
        if event.inaxes is None:
            return
        cur = self.cursor_by_axes.get(event.inaxes)
        if cur is None:
            return
        if event.button == 1:
            if not cur.enabled:
                cur.enabled = True
        elif event.button == 3:
            for sel in list(cur.selections):
                cur.remove_selection(sel)
            cur.enabled = False

    def _on_toggle(self, _label):
        states = self.chk.get_status()
        self.collapse_vertical, self.collapse_horizontal = states[0], states[1]
        self.amplitude = states[2]
        self._build_ui()

    def _on_threshold(self, text):
        try:
            v = float(text)
        except ValueError:
            self.tb_thresh.set_val(f"{self.threshold:.2e}")
            return
        self.threshold = v
        self._recompute_kept()
        self._build_ui()

    def _on_metal_toggle(self, _label):
        # Display-only: shows/hides the metal curve. No projection ever
        # changes with this toggle.
        self.show_metal = self.chk_metal.get_status()[0]
        self._build_ui()

    # ------------------------------------------------------------------
    def _draw_blocks(self):
        n_mark = len(self.markers)
        kept = self.kept

        if not kept:
            # No entity passes anywhere: one axis per marker block, LDOS
            # alone, with the annotation.
            gs = gridspec.GridSpec(n_mark, 1, figure=self.fig,
                                   bottom=0.20, hspace=0.7)
            share = None
            for m in range(n_mark):
                ax = self.fig.add_subplot(gs[m, 0], sharex=share)
                share = share or ax
                ldos_lines = self._draw_ldos(ax, m)
                ax.annotate("no MOs above threshold", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center",
                            fontsize=10)
                self._finish_axis(ax, m, title=self._marker_title(m),
                                  last=(m == n_mark - 1))
                ax.legend(fontsize=6, loc="upper right")
                self._attach_cursor(ax, list(ldos_lines), dict(ldos_lines))
            return

        offsets = self.kept_offsets
        comps = self.kept_comps
        R = 1 if self.collapse_vertical else len(offsets)
        C = 1 if self.collapse_horizontal else len(comps)

        gs = gridspec.GridSpec(n_mark * R, C, figure=self.fig,
                               bottom=0.20, hspace=0.9, wspace=0.25)
        share = None
        for m in range(n_mark):
            for rs in range(R):
                cell_offsets = (offsets if self.collapse_vertical
                                else [offsets[rs]])
                for cs in range(C):
                    cell_comps = (comps if self.collapse_horizontal
                                  else [comps[cs]])
                    ax = self.fig.add_subplot(gs[m * R + rs, cs],
                                              sharex=share)
                    share = share or ax
                    multi = len(cell_comps) > 1
                    artists = []
                    hover = {}
                    for k, row in enumerate(kept):
                        if (row["offset"] in cell_offsets
                                and row["comp"] in cell_comps):
                            y = self._disp(self._row_factor(m, k) * row["W"])
                            (line,) = ax.plot(
                                self.eg, y,
                                color=self._comp_color(row, multi),
                                lw=1.5,
                                label=self._legend_label(m, k, row))
                            artists.append(line)
                            hover[line] = self._hover_text(m, k, row)
                    ldos_lines = self._draw_ldos(ax, m)
                    artists.extend(ldos_lines)
                    hover.update(ldos_lines)
                    # A legend on EVERY axis, whatever the toggle states.
                    ax.legend(fontsize=6, loc="upper right")
                    self._attach_cursor(ax, artists, hover)

                    # Titles / row labels (grid modes: column titles =
                    # component labels, row labels = offsets).
                    title = None
                    if rs == 0 and m == 0 and not self.collapse_horizontal:
                        title = cell_comps[0]
                    block_title = (self._marker_title(m)
                                   if (rs == 0 and cs == 0) else None)
                    if block_title is not None:
                        title = (f"{block_title}" if title is None
                                 else f"{block_title} — {title}")
                    if title:
                        ax.set_title(title, fontsize=9)
                    if cs == 0 and not self.collapse_vertical:
                        ax.set_ylabel(_offset_name(cell_offsets[0]),
                                      fontsize=9)
                    last = (m == n_mark - 1 and rs == R - 1)
                    self._finish_axis(ax, m, title=None, last=last)
    def _legend_label(self, m: int, k: int, row) -> str:
        """Legend entry for an MO curve: component + offset name plus the
        projection value in the CURRENT view (p² probability / p
        amplitude) — the band composition (#+#) lives in the hover and
        the export columns instead."""
        p = float(self.proj_coeffs[m][k])
        name = f"{row['comp']} {_offset_name(row['offset'])}"
        if self.amplitude:
            return f"{name} p={abs(p):.3g}"
        return f"{name} p²={p * p:.3g}"

    def _hover_text(self, m: int, k: int, row) -> str:
        """Hover body for an MO curve: that MO's probability (p²) or
        amplitude (p), matching the current Amplitude toggle state."""
        p = float(self.proj_coeffs[m][k])
        if self.amplitude:
            return (f"{row['label']}\nmarker {m}\n"
                    f"amplitude p = {abs(p):.5f}")
        return (f"{row['label']}\nmarker {m}\n"
                f"probability p² = {p * p:.5f}")

    def _metal_hover(self, m: int) -> str:
        """Hover body for the metal composite curve, matching the current
        Amplitude toggle state."""
        p = float(self.p_met[m])
        if self.amplitude:
            return (f"metal composite\nmarker {m}\n"
                    f"amplitude p_met = {abs(p):.5f}")
        return (f"metal composite\nmarker {m}\n"
                f"probability p_met² = {p * p:.5f}")

    def _draw_ldos(self, ax, m: int) -> Dict[Any, str]:
        # The marker's LDOS (thick solid, marker color) appears on every
        # axis in every mode, always at the same scale; with the
        # Metal-curve toggle ON, the projection component of the LDOS
        # along the per-marker metal composite Ĉ_m (built from every
        # metal state's spread, q²-weighted by its overlap with THIS
        # LDOS), ‖L‖·p_met·Ĉ_m, is drawn too (thin dashed, same color)
        # with p_met²/p_met in its legend entry. The Amplitude toggle
        # displays both (like every spread curve) as the sqrt of their
        # probability-scale values. Returns {line: hover_text}.
        color = self.marker_colors[m % len(self.marker_colors)]
        lines: Dict[Any, str] = {}
        (line,) = ax.plot(
            self.eg, self._disp(self.ldos_curves[m]),
            color=color, lw=2.6, alpha=0.9, zorder=1,
            label=f"LDOS marker {m}")
        lines[line] = (f"LDOS marker {m}\n"
                       f"A = {self.A_ldos[m]:.5f}")
        if self.show_metal:
            pm = float(self.p_met[m])
            leg = (f"metal p={abs(pm):.3g}" if self.amplitude
                   else f"metal p²={pm * pm:.3g}")
            (mline,) = ax.plot(
                self.eg, self._disp(self.metal_curves[m]),
                color=color, lw=1.2, ls="--", alpha=0.9, zorder=1,
                label=leg)
            lines[mline] = self._metal_hover(m)
        return lines

    def _marker_title(self, m: int) -> str:
        x, y = self.markers[m]
        return f"marker {m} ({x:.3f}, {y:.3f}) Å"

    def _finish_axis(self, ax, m: int, title, last: bool):
        if title:
            ax.set_title(title, fontsize=9)
        if last:
            ax.set_xlabel("Energy − E_F (eV)")
        ax.set_xlim(float(self.eg[0]), float(self.eg[-1]))

    # ------------------------------------------------------------------
    def save_data(self) -> str:
        """
        Commented-header txt per press: header = σ, threshold, toggle
        states, per-marker coordinates with A_LDOS and the block factor,
        the per-marker projection coefficients of every kept entity, kept
        entities with a_ent, entity/pairing map; body = per-marker blocks
        of columns E, LDOS, and each kept entity's UNSCALED W_ent (the
        coefficients and block factor in the header make the probability
        view exactly reconstructable, and the magnitude view is its
        pointwise sqrt; no display scaling is baked into saved numbers;
        always probability-scale).
        """
        out_path = os.path.join(self.run_dir, "decomp_window_data.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# SIGMA: {self.sigma:.6f}\n")
            f.write(f"# THRESHOLD: {self.threshold:.6e}\n")
            f.write(f"# TOGGLES: vertical={self.collapse_vertical}; "
                    f"horizontal={self.collapse_horizontal}; "
                    f"amplitude={self.amplitude}; "
                    f"show_metal={self.show_metal}\n")
            for m, xy in enumerate(self.markers):
                f.write(f"# MARKER {m}: x={xy[0]:.6f}; y={xy[1]:.6f}; "
                        f"A_LDOS={self.A_ldos[m]:.8e}; "
                        f"beta_prob={self.beta_prob[m]:.8e}; "
                        f"p_met={self.p_met[m]:.8e}\n")
            for m in range(len(self.markers)):
                proj_str = "; ".join(
                    f"{row['label']}={self.proj_coeffs[m][k]:.8e}"
                    for k, row in enumerate(self.kept))
                f.write(f"# PROJ marker {m}: {proj_str}\n")
            for row in self.kept:
                f.write(f"# KEPT: {row['label']}; "
                        f"a_ent={row['a_ent']:.8e}\n")
            for label in self.mol_labels:
                offmap = self.entities_by_comp.get(label, {})
                pair_str = "; ".join(
                    f"{off:+d}=({'+'.join(str(i) for i in ent.indices)})"
                    f"@{ent.E_mean:.5f}"
                    for off, ent in sorted(offmap.items()))
                f.write(f"# ENTITIES {label}: {pair_str}\n")
            cols = (["E", "LDOS", "LDOS_metal_component"]
                    + [r["label"].replace(" ", "_") for r in self.kept])
            for m in range(len(self.markers)):
                f.write(f"# BLOCK marker {m}\n")
                f.write("# " + " ".join(cols) + "\n")
                arr = np.column_stack(
                    [self.eg, self.ldos_curves[m], self.metal_curves[m]]
                    + [r["W"] for r in self.kept])
                for rowv in arr:
                    f.write(" ".join(f"{v:.8e}" for v in rowv) + "\n")
        print(f"[DECOMP] Wrote decomposition data to '{out_path}'")
        return out_path
