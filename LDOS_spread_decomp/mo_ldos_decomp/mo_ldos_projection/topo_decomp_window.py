# -*- coding: utf-8 -*-
"""
topo_decomp_window.py — the MO topo-decomposition figure (Module B).

Opened by the wrapper's "topo" button in the "Send to decomp" box.
Snapshot semantics exactly like DecompWindow: the figure reflects the
moment of the press and never live-tracks the simulator; pressing again
regenerates it via refresh().

Input (spread_collapse comp_indices convention): a list of HOMO-relative
Kramers-entity offsets (0 = HOMO, −1 = HOMO−1, +1 = LUMO, ...), set in
run_stm_decomp.py as topo_decomp_offsets. The figure is an
offsets × molecule-components grid — one MO topo-decomp map per offset
per component, in the input order; an offset a component does not have
draws as a "not found" cell. No area threshold applies here: the input
list alone decides the MO set.

Data: the topo's own LDOS. The snapshot carries the total LDOS curve of
EVERY topography grid point, evaluated at that point's converged
constant-current tip height (the global topo surface) over the topo's
own stabilization window sorted([0, topo_bias]) — the very curves whose
integral defines the constant-current condition. Markers play no role.

Projection: identical mathematics to the LDOS decomposition window's
per-curve scaling — every curve L2-unit-normalized first, then the
independent normalized overlap per (entity, grid point)

    p_a(x, y) = ⟨L̂(x,y), Ŵ_a⟩            (topo-window grid)

with W_a(E) = Σ_j w_a(j) g_σ(E−E_j) at the committed σ, evaluated on the
snapshot's topo energy grid. Projections are computed once per press;
the toggles below never recompute them.

Rendering (user-directed, for Adobe-Illustrator recoloring): each cell
is ONE single-color layer — the MO's color (the entity color the
package already assigns via P.component_colors) drawn as a tricontourf
whose bands all share that IDENTICAL fill color and differ only in
OPACITY: fully opaque (the color at its brightest) at the reference
MAXIMUM intensity, fully transparent at the reference MINIMUM, fading
linearly in between. No black anywhere, and nothing is drawn beneath
the layer (the cell's axes background is transparent), so
low-intensity regions show whatever sits behind the figure.

In Illustrator the bands share one exact fill color while opacity is a
separate appearance attribute: select the layer group (or one band →
Select → Same → Fill Color) and change the fill once — every band
recolors and the transparency ramp is untouched.

Every drawn cell carries its OWN colorbar built from the identical
single-color fade (transparent at the reference minimum, opaque at the
reference maximum), graduated in the PLOTTED quantity's real values —
p² (probability) or |p| (amplitude) — so the scale of each panel's
opacity ramp is readable directly. The bar tracks the toggles exactly
like the maps: Amplitude changes the graduated quantity, Norm makes
every bar show the one shared range. A flat map (no range — drawn
fully transparent) gets no bar, and "not found" cells never do.

Toggles (both display-only; CheckButtons, both default OFF):
  Amplitude   OFF: intensity = p² (probability, the LDOS-decomp scaling)
              ON:  intensity = |p| (amplitude)
  Norm        OFF: the opacity ramp anchors on that MO's OWN
                   [min, max] intensity range (full per-panel contrast)
              ON:  one shared [min, max] range across all maps (panels
                   directly intensity-comparable; weak MOs stay mostly
                   transparent)

Exports (Save data) are always the RAW projection coefficients p per
grid point per found entity — probability (p²) and amplitude (|p|) views
are exactly reconstructable and no display normalization is baked in.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import Button, CheckButtons

if __name__ != "__main__":
    from ._imports import PlotConfig, _gauss_sum
    from .decomp_window import _offset_name, load_static_mo_data


class TopoDecompWindow:
    """
    Snapshot-semantics MO topo-decomposition window: an
    offsets × components grid of single-color alpha-ramp maps (the MO's
    color, opaque at the reference maximum intensity, transparent at the
    reference minimum) of the per-grid-point projection of the topo LDOS
    curves onto each input MO's spread curve.
    """

    def __init__(
        self,
        run_dir: str,
        snapshot: Dict[str, Any],
        offsets: List[int],
        cfg: Optional[PlotConfig] = None,
        mask_levels: int = 61,
    ):
        self.run_dir = run_dir
        self.cfg = cfg if cfg is not None else PlotConfig()
        self.offsets = [int(o) for o in offsets]
        if not self.offsets:
            raise ValueError(
                "topo_decomp_offsets must name at least one HOMO-relative "
                "entity offset (spread_collapse convention: 0 = HOMO, "
                "-1 = HOMO-1, +1 = LUMO, ...)")
        self.mask_levels = int(mask_levels)

        # Display toggles (both display-only; both default OFF).
        self.amplitude = False     # OFF: p² (probability)  ON: |p|
        self.norm_shared = False   # OFF: per-MO max        ON: shared max

        d = load_static_mo_data(self.run_dir, self.cfg)
        self.E_sticks = d["E_sticks"]
        self.mol_labels = d["mol_labels"]
        self.entity_rows = d["entity_rows"]
        self._row_map = {(r["comp"], r["offset"]): r
                         for r in self.entity_rows}

        self._apply_snapshot(snapshot)

        R, C = len(self.offsets), len(self.mol_labels)
        self.fig = plt.figure(
            figsize=(max(6.0, 4.2 * C), max(4.2, 3.4 * R)))
        # Transparent figure patch: with savefig.facecolor='auto' (the
        # default) a toolbar SVG/PNG save then carries NO opaque
        # background rectangle beneath the color layers, so the
        # low-intensity regions are genuinely transparent in the export.
        self.fig.patch.set_facecolor("none")
        self._build_ui()

    # ------------------------------------------------------------------
    # Snapshot: topo grid, per-point LDOS curves, σ, topo bias window.
    # ------------------------------------------------------------------
    def _apply_snapshot(self, snapshot: Dict[str, Any]):
        self.snapshot = snapshot
        self.sigma = float(snapshot["sigma"])
        self.eg = np.asarray(snapshot["eg"], dtype=np.float64)
        self.grid_xy = np.asarray(snapshot["grid_xy"], dtype=np.float64)
        self.topo_bias = float(snapshot["topo_bias"])
        self.topo_height = float(snapshot["topo_height"])
        self.lv = np.asarray(snapshot["lv"], dtype=np.float64)
        L = np.asarray(snapshot["ldos"], dtype=np.float64)
        if L.shape[0] != self.grid_xy.shape[0]:
            raise ValueError("snapshot grid points and LDOS curves differ "
                             "in count")
        if L.shape[1] != self.eg.size:
            raise ValueError("snapshot LDOS curves and energy grid differ "
                             "in length")

        # Unit-normalize every grid point's LDOS curve (zero-safe).
        norms = np.linalg.norm(L, axis=1)
        safe = np.where(norms > 0, norms, 1.0)
        L_hat = L / safe[:, None]
        L_hat[norms <= 0] = 0.0

        # One cell per (offset, component), input order preserved: the
        # entity's spread curve at the committed σ on the topo energy
        # grid, and the per-grid-point independent normalized overlap
        # p = ⟨L̂, Ŵ⟩ (computed ONCE here; toggles never recompute).
        self.cells: List[Dict[str, Any]] = []
        for offset in self.offsets:
            for comp in self.mol_labels:
                row = self._row_map.get((comp, offset))
                if row is None:
                    self.cells.append(dict(offset=offset, comp=comp,
                                           row=None, W=None, p=None))
                    continue
                states = list(zip(self.E_sticks.tolist(),
                                  row["weights"].tolist()))
                W = _gauss_sum(self.eg, states, self.sigma)
                w_norm = float(np.linalg.norm(W))
                if w_norm > 0:
                    p = L_hat @ (W / w_norm)
                else:
                    p = np.zeros(self.grid_xy.shape[0], dtype=np.float64)
                self.cells.append(dict(offset=offset, comp=comp, row=row,
                                       W=W, p=p))

    # ------------------------------------------------------------------
    def refresh(self, snapshot: Dict[str, Any]):
        """Regenerate from a new press (snapshot semantics)."""
        self._apply_snapshot(snapshot)
        self._build_ui()

    # ------------------------------------------------------------------
    # Display quantities (toggle-dependent, projections untouched)
    # ------------------------------------------------------------------
    def _intensity(self, cell: Dict[str, Any]) -> Optional[np.ndarray]:
        """The cell's intensity map in the CURRENT view: p² (probability,
        Amplitude off) or |p| (amplitude, Amplitude on)."""
        if cell["p"] is None:
            return None
        p = cell["p"]
        return np.abs(p) if self.amplitude else p ** 2

    def _reference_range(self, cell: Dict[str, Any]) -> Tuple[float, float]:
        """The (min, max) intensity the opacity ramp anchors on: the
        cell's own range (Norm off) or the shared range over every found
        cell (Norm on)."""
        if self.norm_shared:
            fields = [self._intensity(c)
                      for c in self.cells if c["p"] is not None]
            lo = min(float(np.min(I)) for I in fields)
            hi = max(float(np.max(I)) for I in fields)
        else:
            I = self._intensity(cell)
            lo, hi = float(np.min(I)), float(np.max(I))
        return lo, hi

    def _alpha_field(self, cell: Dict[str, Any]) -> Optional[np.ndarray]:
        """Normalized [0, 1] field driving the layer's opacity: 1 = fully
        opaque MO color (reference maximum intensity), 0 = fully
        transparent (reference minimum). A flat map (no range) draws
        fully transparent."""
        I = self._intensity(cell)
        if I is None:
            return None
        lo, hi = self._reference_range(cell)
        if hi <= lo:
            return np.zeros_like(I)
        return np.clip((I - lo) / (hi - lo), 0.0, 1.0)

    def _cell_cmap(self, cell: Dict[str, Any]) -> LinearSegmentedColormap:
        """Single-color transparent→opaque colormap in the MO's own
        color: every band shares the identical RGB fill and differs only
        in opacity (the Illustrator-recolorable construction)."""
        r, g, b = (float(v) for v in cell["row"]["color"])
        return LinearSegmentedColormap.from_list(
            "mo_alpha", [(r, g, b, 0.0), (r, g, b, 1.0)])

    def _scale_label(self) -> str:
        """The colorbar's quantity label in the CURRENT view: the
        projection under the active Amplitude transform."""
        return "|p|" if self.amplitude else "p²"

    # ------------------------------------------------------------------
    # UI build (persistent figure; fig.clf() rebuild on every toggle /
    # refresh — projections are never recomputed by a toggle)
    # ------------------------------------------------------------------
    def _cell_corners(self) -> np.ndarray:
        """The topo cell parallelogram (xy corners, closed order)."""
        a, b = self.lv[0, :2], self.lv[1, :2]
        return np.array([[0.0, 0.0], a, a + b, b])

    def _build_ui(self):
        for w in (getattr(self, "chk", None), getattr(self, "btn_save", None)):
            if w is not None:
                w.disconnect_events()
        self.fig.clf()

        self.chk = CheckButtons(
            self.fig.add_axes([0.01, 0.02, 0.10, 0.10]),
            ["Amplitude", "Norm"],
            [self.amplitude, self.norm_shared],
        )
        self.chk.on_clicked(self._on_toggle)

        self.btn_save = Button(self.fig.add_axes([0.84, 0.025, 0.12, 0.045]),
                               "Save data")
        self.btn_save.on_clicked(lambda _e: self.save_data())

        lo, hi = sorted([0.0, self.topo_bias])
        self.fig.suptitle(
            f"MO topo decomposition — σ = {self.sigma:.3f} eV, "
            f"window {lo:+.3f} → {hi:+.3f} eV, "
            f"topo {self.topo_bias:g} V @ {self.topo_height:g} Å",
            fontsize=11)

        R, C = len(self.offsets), len(self.mol_labels)
        gs = gridspec.GridSpec(R, C, figure=self.fig,
                               bottom=0.16, top=0.90,
                               hspace=0.35, wspace=0.25)
        corners = self._cell_corners()
        levels = np.linspace(0.0, 1.0, self.mask_levels)
        share = None
        self.cbars: List[Any] = []   # one entry per cell (None = no bar)
        for r, offset in enumerate(self.offsets):
            for c, comp in enumerate(self.mol_labels):
                cell = self.cells[r * C + c]
                ax = self.fig.add_subplot(gs[r, c], sharex=share,
                                          sharey=share)
                share = share or ax
                cbar = None
                if cell["p"] is None:
                    ax.annotate("not found", xy=(0.5, 0.5),
                                xycoords="axes fraction", ha="center",
                                fontsize=10)
                    ax.set_title(f"{comp} {_offset_name(offset)} — "
                                 "not found", fontsize=9)
                else:
                    # ONE single-color layer: the MO's color, opacity
                    # ramped by the intensity field (transparent at the
                    # reference minimum, opaque at the reference
                    # maximum). Every band shares the identical fill
                    # color, so recoloring in Illustrator is one fill
                    # change; nothing is drawn beneath the layer.
                    ax.set_facecolor("none")
                    ax.tricontourf(
                        self.grid_xy[:, 0], self.grid_xy[:, 1],
                        self._alpha_field(cell),
                        levels=levels, cmap=self._cell_cmap(cell),
                        zorder=2)
                    ax.set_title(cell["row"]["label"], fontsize=9)
                    # The cell's OWN colorbar: the identical single-
                    # color fade graduated in the plotted quantity's
                    # real values over the reference range the opacity
                    # ramp anchors on (per-MO or shared per the Norm
                    # toggle). use_gridspec=False keeps the stolen bar
                    # axes subplotspec-free — data axes stay the R×C
                    # grid. Flat maps (drawn fully transparent) get no
                    # bar: there is no range to graduate.
                    lo_i, hi_i = self._reference_range(cell)
                    if hi_i > lo_i:
                        sm = ScalarMappable(
                            norm=Normalize(vmin=lo_i, vmax=hi_i),
                            cmap=self._cell_cmap(cell))
                        sm.set_array([])
                        cbar = self.fig.colorbar(
                            sm, ax=ax, use_gridspec=False,
                            fraction=0.046, pad=0.04)
                        cbar.set_label(self._scale_label(), fontsize=7)
                        cbar.ax.tick_params(labelsize=6)
                        cbar.ax.set_facecolor("none")
                ax.set_aspect("equal")
                ax.set_xlim(float(np.min(corners[:, 0])),
                            float(np.max(corners[:, 0])))
                ax.set_ylim(float(np.min(corners[:, 1])),
                            float(np.max(corners[:, 1])))
                if r == R - 1:
                    ax.set_xlabel("x (Å)")
                if c == 0:
                    ax.set_ylabel("y (Å)")
                self.cbars.append(cbar)

        self.fig.canvas.draw_idle()

    def _on_toggle(self, _label):
        states = self.chk.get_status()
        self.amplitude, self.norm_shared = states[0], states[1]
        self._build_ui()

    # ------------------------------------------------------------------
    def save_data(self) -> str:
        """
        Commented-header txt per press: header = σ, topo bias window and
        height, toggle states, the input offsets, and every cell's entity
        (or "not found"); body = columns x, y, then the RAW projection
        coefficient p of every found cell (probability = p², amplitude =
        |p| — no display normalization is baked into saved numbers).
        """
        out_path = os.path.join(self.run_dir, "topo_decomp_data.txt")
        lo, hi = sorted([0.0, self.topo_bias])
        found = [c for c in self.cells if c["p"] is not None]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# SIGMA: {self.sigma:.6f}\n")
            f.write(f"# TOPO_BIAS: {self.topo_bias:.6f}\n")
            f.write(f"# WINDOW: {lo:.6f} {hi:.6f}\n")
            f.write(f"# TOPO_HEIGHT: {self.topo_height:.6f}\n")
            f.write(f"# TOGGLES: amplitude={self.amplitude}; "
                    f"norm_shared={self.norm_shared}\n")
            f.write("# OFFSETS: "
                    + " ".join(f"{o:+d}" for o in self.offsets) + "\n")
            for i, cell in enumerate(self.cells):
                name = (cell["row"]["label"] if cell["row"] is not None
                        else f"{cell['comp']} "
                             f"{_offset_name(cell['offset'])} — not found")
                f.write(f"# CELL {i}: {name}\n")
            cols = (["x", "y"]
                    + [f"p_{c['row']['label'].replace(' ', '_')}"
                       for c in found])
            f.write("# " + " ".join(cols) + "\n")
            arr = np.column_stack(
                [self.grid_xy[:, 0], self.grid_xy[:, 1]]
                + [c["p"] for c in found])
            for rowv in arr:
                f.write(" ".join(f"{v:.8e}" for v in rowv) + "\n")
        print(f"[TOPO-DECOMP] Wrote topo decomposition data to "
              f"'{out_path}'")
        return out_path
