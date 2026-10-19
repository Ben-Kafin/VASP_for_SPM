# -*- coding: utf-8 -*-
"""
sim_wrapper.py — launcher around the UNMODIFIED stm_from_dft simulator
(Module B, §8.3).

Instantiates the existing Interactive_STM_Simulator against the current
σ-tagged working directory (synthetic DOSCAR + provisioned inputs). Same
figures, same controls, identical behavior — plus two wrapper-added
widgets placed clear of the existing ones (left column, below the
broadening block):

1. σ slider (0.01–1.0, default 0.2) with a paired two-way-synced textbox.
   σ COMMIT = slider mouse-release or textbox Enter (no continuous-drag
   recompute). On commit: re-smear → write the new σ-tagged dir → capture
   UI state → tear down the simulator instance → re-instantiate on the
   new dir → restore UI state (energy-range values, LDOS height, bias
   sign, every toggle/checkbox state, γ and the other slider values,
   colormap, and all marker positions).

DOSCAR source option (session INPUT, decided at launch — deliberately not
a toggle on the figure): use_synthetic_doscar=True (default) is the
behavior above. use_synthetic_doscar=False runs the simulator directly on
the run directory's REAL DOSCAR — no PROCAR parse, no σ-tagged working
dirs, and the directory's existing topo caches stay in play. The σ widget
remains, but a commit then only sets the smearing width of the MO spread
curves for the NEXT "Send to decomposition" press (no teardown — the real
DOSCAR does not depend on σ). NOTE: the real DOSCAR carries VASP's own
smearing (ISMEAR/SIGMA), so the one-σ consistency between the site DOS
and the spread curves that the synthetic route guarantees does not hold
in this mode; that trade-off is the point of the option.

2. A boxed "Send to decomp" widget group (below the σ widgets, boxed like
   the broadening checkbox cluster) holding TWO buttons:

   "ldos" — the original send-to-decomposition behavior, unchanged. If
   the Decay toggle is ON: prints a clear warning ("projection LDOS
   requires energy decay off") and does nothing — never auto-toggles,
   never recomputes. Otherwise it collects ALL current markers'
   single-point LDOS spectra exactly as the simulator computed them for
   its spectra panel (reusing its cached per-marker arrays; nothing
   re-derived) over the current energy window, and opens/refreshes the
   decomposition window. Snapshot semantics: the window reflects the
   moment of the press; it never live-tracks; pressing again regenerates
   it.

   "topo" — the MO topo decomposition (same Decay-off guard, same
   snapshot semantics, its own separate window). No markers involved:
   it evaluates the total LDOS curve of EVERY topography grid point at
   its converged constant-current tip height (the global topo surface)
   over the topo's own stabilization window sorted([0, topo_bias]) — the
   same single evaluation the convergence engine's final iteration made,
   nothing re-converged — and opens/refreshes the TopoDecompWindow: one
   single-color alpha-ramp map (the MO's color, opaque at the reference
   maximum intensity, transparent at the reference minimum) per entry of
   the session's topo_decomp_offsets input per molecule component, each
   map showing that MO's per-point projection probability/amplitude in
   place of tip height.

simulator.py itself is NEVER modified: the wrapper re-attaches its
widgets after every simulator _build_ui() (mode switches clf() the
figure) by wrapping the INSTANCE attribute only.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from scipy.integrate import simpson

if __name__ != "__main__":
    from ._imports import PlotConfig, resolve_stm
    from .procar_dos import parse_procar
    from .synthetic_doscar import read_doscar_header, make_sigma_workdir
    from .decomp_window import DecompWindow, MO_AREA_NOISE_THRESHOLD
    from .topo_decomp_window import TopoDecompWindow

SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
SIGMA_DEFAULT = 0.2

# Wrapper-widget rects (figure coords): left column below the broadening
# block, above the RUN/checkbox cluster — clear of every simulator widget
# in every mode (rects enumerated from simulator._build_ui). The two send
# buttons sit INSIDE the titled "Send to decomp" box. The box/button
# rects also stay LEFT of x = 0.125 (the default gridspec left margin):
# an axes overlapping a plot axes steals canvas.inaxes in the overlap, so
# markers there become unpickable and clicks can misfire a send (the
# Line/Path tip-height profile panel reaches down to y ≈ 0.23).
_RECT_SIGMA_SLIDER = [0.035, 0.62, 0.10, 0.018]
_RECT_SIGMA_TEXT = [0.035, 0.575, 0.055, 0.028]
_RECT_DECOMP_BOX = [0.022, 0.435, 0.101, 0.095]
_RECT_SEND_TOPO = [0.030, 0.462, 0.042, 0.042]
_RECT_SEND_LDOS = [0.078, 0.462, 0.042, 0.042]


class StmDecompSession:
    """
    One interactive session: PROCAR parsed once, real-DOSCAR header read
    once, then simulator instances come and go per σ commit.
    """

    def __init__(
        self,
        run_dir: str,
        erange=(-3.0, 0.5),
        ldos_height: float = 1.25,
        cmap_topo=None,
        unit_cell_num: int = 1,
        sigma: float = SIGMA_DEFAULT,
        sim_kwargs: Optional[Dict[str, Any]] = None,
        cfg: Optional[PlotConfig] = None,
        threshold: float = MO_AREA_NOISE_THRESHOLD,
        use_synthetic_doscar: bool = True,
        topo_decomp_offsets=(0,),
    ):
        """
        use_synthetic_doscar : bool (session input, not a figure toggle)
            True (default): σ-smeared synthetic DOSCAR in σ-tagged working
            dirs; a σ commit re-smears and re-instantiates the simulator.
            False: the simulator runs on the run directory's REAL DOSCAR;
            σ then only smears the MO spread curves of the decomposition
            window (committed value applies at the next Send press).
        topo_decomp_offsets : list of int (session input)
            HOMO-relative Kramers-entity offsets (spread_collapse
            comp_indices convention: 0 = HOMO, -1 = HOMO-1, +1 = LUMO,
            ...) that get an MO topo-decomposition map per molecule
            component when the "topo" send button is pressed.
        """
        self.run_dir = os.path.abspath(run_dir)
        self.sigma = float(sigma)
        self.cfg = cfg if cfg is not None else PlotConfig()
        self.threshold = float(threshold)
        self.use_synthetic_doscar = bool(use_synthetic_doscar)
        self.topo_decomp_offsets = [int(o) for o in topo_decomp_offsets]

        if cmap_topo is None:
            from matplotlib.colors import LinearSegmentedColormap
            cmap_topo = LinearSegmentedColormap.from_list(
                "t", ["black", "firebrick", "yellow"])
        self._ctor_kwargs = dict(
            erange=list(erange), ldos_height=float(ldos_height),
            cmap_topo=cmap_topo, unit_cell_num=int(unit_cell_num))

        # run_interactive kwargs (simulator defaults unless overridden).
        self.sim_kwargs: Dict[str, Any] = dict(
            grid_res=64, topo_bias=0.2, topo_height=2.5,
            ldos_bias_sign="neg", use_decay=True, show_decay_toggle=True,
            reuse_cache=True, line_endpoints=None, marker_positions=None,
            extra_broadening=None, path=None, path_extend=False,
        )
        if sim_kwargs:
            self.sim_kwargs.update(sim_kwargs)

        # The header carries the ONE Fermi energy used anywhere in Module B
        # (read up front in both modes — cheap, and it validates the DOSCAR
        # early). The PROCAR is only needed to build synthetic DOSCARs; the
        # Γ-only scope it asserts is skipped in real-DOSCAR mode, where the
        # matcher record already comes from the same single-k run.
        self.header = read_doscar_header(os.path.join(self.run_dir, "DOSCAR"))
        if self.use_synthetic_doscar:
            self.procar = parse_procar(os.path.join(self.run_dir, "PROCAR"))
        else:
            self.procar = None
            print("[WRAPPER] use_synthetic_doscar=False: the simulator runs "
                  "on the run directory's real DOSCAR; σ commits only set "
                  "the MO spread smearing for the decomposition window.")

        self.sim = None
        self.decomp_window: Optional[DecompWindow] = None
        self.topo_window: Optional[TopoDecompWindow] = None
        self._widgets: Dict[str, Any] = {}
        self._sigma_pending = self.sigma
        self._orig_build_ui = None
        self._pending_prerun_state: Dict[str, Any] = {}
        self._pending_postrun_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _prepare_workdir(self) -> str:
        """The directory the simulator runs on: the σ-tagged synthetic
        working dir (built/reused here), or the run directory itself in
        real-DOSCAR mode."""
        if not self.use_synthetic_doscar:
            return self.run_dir
        return make_sigma_workdir(
            self.run_dir, self.sigma, self.procar, self.header)

    def launch(self):
        """Prepare the working directory and start the simulator
        (non-blocking: interactive mode; call matplotlib's event loop /
        plt.show() from the run script to hold the process)."""
        plt.ion()
        self._instantiate(self._prepare_workdir())
        return self.sim

    # ------------------------------------------------------------------
    def _instantiate(self, workdir: str):
        simulator_mod, _ = resolve_stm()
        sim = simulator_mod.Interactive_STM_Simulator(
            filepath=workdir, **self._ctor_kwargs)
        self.sim = sim
        # Pre-run state restore hooks (attributes consumed by _build_ui).
        pre = self._pending_prerun_state
        if pre:
            for attr in ("mode", "normalize", "show_mag", "show_atoms",
                         "show_unit_cell", "t2_clamp", "display_cells",
                         "ldos_gamma", "active_marker_idx", "plot_level",
                         "cached_nepts"):
                if attr in pre:
                    setattr(sim, attr, pre[attr])
        sim.run_interactive(**self.sim_kwargs)      # returns under plt.ion()
        post = self._pending_postrun_state
        if post:
            if "marker_ratios" in post and post["marker_ratios"] is not None:
                sim.marker_ratios = list(post["marker_ratios"])
            if post.get("is_running"):
                sim._toggle_run(None)
            sim._update_all()
        self._pending_prerun_state = {}
        self._pending_postrun_state = {}

        # Re-attach the wrapper widgets after EVERY simulator _build_ui
        # (mode switches clf() the whole figure). Instance-attribute wrap
        # only — simulator.py itself is never modified.
        self._orig_build_ui = sim._build_ui

        def _wrapped_build_ui(*a, **k):
            self._orig_build_ui(*a, **k)
            self._attach_widgets()
        sim._build_ui = _wrapped_build_ui

        self._attach_widgets()

    # ------------------------------------------------------------------
    def _attach_widgets(self):
        sim = self.sim
        for w in self._widgets.values():
            try:
                w.disconnect_events()
            except Exception:
                pass
        self._widgets = {}
        if getattr(self, "_release_cid", None) is not None:
            try:
                sim.fig.canvas.mpl_disconnect(self._release_cid)
            except Exception:
                pass
            self._release_cid = None

        ax_sld = sim.fig.add_axes(_RECT_SIGMA_SLIDER)
        sld = Slider(ax_sld, "σ", SIGMA_MIN, SIGMA_MAX,
                     valinit=self.sigma, valfmt="%.3f")
        ax_tb = sim.fig.add_axes(_RECT_SIGMA_TEXT)
        tb = TextBox(ax_tb, "", initial=f"{self.sigma:.3f}")
        # Titled "Send to decomp" box (boxed like the broadening checkbox
        # cluster) under the σ widgets, holding the two send buttons.
        ax_box = sim.fig.add_axes(_RECT_DECOMP_BOX)
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.set_facecolor("none")
        ax_box.set_navigate(False)
        ax_box.set_title("Send to decomp", fontsize=9)
        btn_topo = Button(sim.fig.add_axes(_RECT_SEND_TOPO), "topo",
                          color="lightsteelblue", hovercolor="lightblue")
        btn = Button(sim.fig.add_axes(_RECT_SEND_LDOS), "ldos",
                     color="lightsteelblue", hovercolor="lightblue")

        guard = {"busy": False}

        def _on_slider(val):
            # Drag: track + mirror into the textbox only. Commit happens
            # on mouse RELEASE (no continuous-drag recompute).
            if guard["busy"]:
                return
            guard["busy"] = True
            try:
                self._sigma_pending = float(val)
                tb.set_val(f"{self._sigma_pending:.3f}")
            finally:
                guard["busy"] = False

        def _on_release(event):
            if event.inaxes is ax_sld and \
                    abs(self._sigma_pending - self.sigma) > 1e-12:
                self.commit_sigma(self._sigma_pending)

        def _on_text(text):
            if guard["busy"]:
                return
            try:
                v = float(text)
            except ValueError:
                tb.set_val(f"{self.sigma:.3f}")
                return
            v = min(max(v, SIGMA_MIN), SIGMA_MAX)
            guard["busy"] = True
            try:
                sld.set_val(v)
                tb.set_val(f"{v:.3f}")
            finally:
                guard["busy"] = False
            self._sigma_pending = v
            if abs(v - self.sigma) > 1e-12:
                self.commit_sigma(v)

        sld.on_changed(_on_slider)
        tb.on_submit(_on_text)
        btn.on_clicked(lambda _e: self.send_to_decomposition())
        btn_topo.on_clicked(lambda _e: self.send_topo_to_decomposition())
        cid = sim.fig.canvas.mpl_connect("button_release_event", _on_release)

        self._widgets = dict(sigma_slider=sld, sigma_text=tb,
                             send_ldos_btn=btn, send_topo_btn=btn_topo)
        self._release_cid = cid
        sim.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # σ commit: re-smear → new σ-tagged dir → capture → tear down →
    # re-instantiate → restore.
    # ------------------------------------------------------------------
    def capture_state(self) -> Dict[str, Any]:
        sim = self.sim
        state: Dict[str, Any] = {}
        state["mode"] = sim.mode
        state["normalize"] = sim.normalize
        state["show_mag"] = sim.show_mag
        state["show_atoms"] = sim.show_atoms
        state["show_unit_cell"] = sim.show_unit_cell
        state["t2_clamp"] = sim.t2_clamp
        state["display_cells"] = int(sim.s_cell.val)
        state["ldos_gamma"] = float(sim.s_ldos_gamma.val)
        state["active_marker_idx"] = sim.active_marker_idx
        state["plot_level"] = sim.plot_level
        state["is_running"] = bool(sim.is_running)
        state["marker_ratios"] = list(sim.marker_ratios)
        state["cached_nepts"] = (int(sim.s_nepts.val)
                                 if hasattr(sim, "s_nepts")
                                 else sim.cached_nepts)
        # run_interactive kwargs for the next instance:
        kw = dict(self.sim_kwargs)
        kw["use_decay"] = bool(sim.use_decay)
        kw["ldos_bias_sign"] = sim.ldos_bias_sign
        kw["topo_height"] = float(sim.s_topo_h.val) \
            if hasattr(sim, "s_topo_h") else sim.topo_height
        kw["topo_bias"] = sim.global_topo_bias
        kw["marker_positions"] = [list(pt) for pt in sim.marker_coords]
        kw["line_endpoints"] = ([list(sim.p1), list(sim.p2)]
                                if sim.p1 is not None else None)
        kw["extra_broadening"] = dict(sim.extra_broadening)
        kw["path"] = ([list(map(float, n)) for n in sim.path_nodes]
                      if sim.path_nodes is not None else None)
        kw["path_extend"] = bool(sim.path_extend)
        kw["reuse_cache"] = bool(sim.reuse_cache)
        kw["show_decay_toggle"] = bool(sim.show_decay_toggle)
        state["sim_kwargs"] = kw
        # constructor kwargs:
        ctor = dict(self._ctor_kwargs)
        ctor["erange"] = [float(sim.s_emin.val), float(sim.s_emax.val)]
        ctor["ldos_height"] = (float(sim.s_ldos_h.val)
                               if hasattr(sim, "s_ldos_h")
                               else float(sim.ldos_height))
        state["ctor_kwargs"] = ctor
        return state

    def commit_sigma(self, sigma: float):
        sigma = float(min(max(sigma, SIGMA_MIN), SIGMA_MAX))
        if not self.use_synthetic_doscar:
            # Real-DOSCAR mode: the site DOS does not depend on σ, so there
            # is nothing to re-smear and no reason to tear the simulator
            # down. The committed σ only smears the MO spread curves of the
            # decomposition window, starting with the next Send press.
            print(f"[WRAPPER] σ commit: {self.sigma:.4f} → {sigma:.4f} "
                  "(real-DOSCAR mode: applies to the MO spread curves on "
                  "the next 'Send to decomposition' press).")
            self.sigma = sigma
            self._sigma_pending = sigma
            return
        print(f"[WRAPPER] σ commit: {self.sigma:.4f} → {sigma:.4f}")
        state = self.capture_state()
        self.sim_kwargs = state["sim_kwargs"]
        self._ctor_kwargs = state["ctor_kwargs"]
        self._pending_prerun_state = state
        self._pending_postrun_state = dict(
            marker_ratios=state["marker_ratios"],
            is_running=state["is_running"])
        # Tear down the old instance before the new one chdirs elsewhere.
        old_fig = self.sim.fig
        plt.close(old_fig)
        self.sim = None
        self.sigma = sigma
        self._sigma_pending = sigma
        workdir = make_sigma_workdir(
            self.run_dir, sigma, self.procar, self.header)
        self._instantiate(workdir)

    # ------------------------------------------------------------------
    # "Send to decomposition"
    # ------------------------------------------------------------------
    def send_to_decomposition(self):
        sim = self.sim
        if sim.use_decay:
            print("[WRAPPER] projection LDOS requires energy decay off — "
                  "turn the Decay toggle off, rerun, and press again "
                  "(nothing was sent).")
            return None
        if sim.cached_eg is None:
            print("[WRAPPER] no computed spectra yet — press RUN first "
                  "(nothing was sent).")
            return None

        eg = np.asarray(sim.cached_eg, dtype=np.float64)
        curves = []
        markers = []
        colors = []
        if sim.mode in ("Single Point", "Map"):
            if sim.cached_spec_ldos is None:
                print("[WRAPPER] no per-marker spectra cached — press RUN "
                      "first (nothing was sent).")
                return None
            for i, pt in enumerate(sim.marker_coords):
                c_ldos = sim.cached_spec_ldos[i].copy()
                t_y = np.sum(c_ldos, axis=(1, 2))
                if sim.normalize:
                    t_y = t_y / (simpson(t_y, x=eg) + 1e-15)
                curves.append(t_y)
                markers.append((float(pt[0]), float(pt[1])))
                colors.append(sim.m_colors[i % len(sim.m_colors)])
        else:  # Line / Path: the spectra panel samples the line LDOS
            if sim.cached_ld_up is None:
                print("[WRAPPER] no line LDOS cached — press RUN first "
                      "(nothing was sent).")
                return None
            f_raw = sim._combine_ldos_display(
                sim.cached_ld_up.copy(),
                (sim.cached_ld_dn.copy()
                 if sim.cached_ld_dn is not None else None),
                sim.show_mag)
            offsets = np.asarray(sim.marks.get_offsets())
            for i, r in enumerate(sim.marker_ratios):
                idx = int(r * (sim.npts - 1))
                c_ldos = f_raw[idx].copy()
                t_y = np.sum(c_ldos, axis=(1, 2))
                if sim.normalize:
                    t_y = t_y / (simpson(t_y, x=eg) + 1e-15)
                curves.append(t_y)
                if i < len(offsets):
                    markers.append((float(offsets[i][0]),
                                    float(offsets[i][1])))
                else:
                    markers.append((float("nan"), float("nan")))
                colors.append(sim.m_colors[i % len(sim.m_colors)])

        snapshot = dict(sigma=self.sigma, eg=eg, markers=markers,
                        ldos=curves, marker_colors=colors)
        if self.decomp_window is None or not plt.fignum_exists(
                self.decomp_window.fig.number):
            self.decomp_window = DecompWindow(
                self.run_dir, snapshot, cfg=self.cfg,
                threshold=self.threshold)
        else:
            self.decomp_window.refresh(snapshot)
        self.decomp_window.fig.show()
        return self.decomp_window

    # ------------------------------------------------------------------
    # "Send Topo to decomp" (the "topo" button)
    # ------------------------------------------------------------------
    def send_topo_to_decomposition(self):
        """
        MO topo decomposition from the topo's own LDOS: evaluate the
        total LDOS curve of EVERY topography grid point at its stored
        converged constant-current tip height (global topo surface) over
        the topo's own stabilization window sorted([0, topo_bias]) — the
        identical single evaluation the convergence engine's final
        iteration made (the curves are not cached, and a disk-cached topo
        never computed them this session; nothing is re-converged) — and
        open/refresh the TopoDecompWindow on the per-point projections.
        Markers play no role. Snapshot semantics like the ldos button.
        """
        sim = self.sim
        if sim.use_decay:
            print("[WRAPPER] projection LDOS requires energy decay off — "
                  "turn the Decay toggle off, rerun, and press again "
                  "(nothing was sent).")
            return None
        if getattr(sim, "_broadening_topo_dirty", False):
            print("[WRAPPER] the global topography is stale (decay/"
                  "broadening changed since it was converged) — press RUN "
                  "to refresh it, then press again (nothing was sent).")
            return None
        if getattr(sim, "global_z_map", None) is None:
            print("[WRAPPER] no converged global topography available "
                  "(nothing was sent).")
            return None

        bias = float(sim.global_topo_bias)
        t_emin, t_emax = sorted([0.0, bias])
        pts = np.hstack([np.asarray(sim.grid_xy, dtype=np.float64),
                         np.asarray(sim.global_z_map,
                                    dtype=np.float64)[:, None]])
        # topo_only=True: the combined total-ρ curve per point, exactly
        # what the constant-current condition integrates. Decay is off
        # (guarded above), so clamp_occupied is moot.
        ld_gpu, _, eg_gpu = sim._calculate_ldos_at_points_gpu(
            pts, t_emin, t_emax, use_energy_decay=False,
            preserve_orbitals=False, topo_only=True, clamp_occupied=False)
        ld = np.asarray(ld_gpu.get(), dtype=np.float64)
        eg = np.asarray(eg_gpu.get(), dtype=np.float64)

        snapshot = dict(
            sigma=self.sigma, eg=eg,
            grid_xy=np.asarray(sim.grid_xy, dtype=np.float64),
            ldos=ld, topo_bias=bias,
            topo_height=float(sim.topo_height),
            lv=np.asarray(sim.lv, dtype=np.float64))
        if self.topo_window is None or not plt.fignum_exists(
                self.topo_window.fig.number):
            self.topo_window = TopoDecompWindow(
                self.run_dir, snapshot, self.topo_decomp_offsets,
                cfg=self.cfg)
        else:
            self.topo_window.refresh(snapshot)
        self.topo_window.fig.show()
        return self.topo_window
