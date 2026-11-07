# -*- coding: utf-8 -*-
"""
sim_wrapper.py — launcher around the stm_from_dft simulator (Module B, §8.3).

Instantiates the Interactive_STM_Simulator on the run directory's real
DOSCAR. Same window, same controls, identical behavior — plus two control
boxes the wrapper adds to the simulator window's left column
(Interactive_STM_Simulator.add_control_box, under the broadening box; the
window's automatic layout places them, nothing is placed by hand):

1. The σ box (controls.SigmaBox, 0.01–1.0, default 0.2): a number box
   whose digits step with the arrow keys.  σ COMMIT = a value applied by
   the box (Enter, leaving the box after an edit, a digit step) that
   differs from the current σ.  A commit only sets the smearing width of
   the MO spread curves for the NEXT "Send to decomposition" press (no
   teardown — the real DOSCAR does not depend on σ).  The commit runs on
   the next turn of the event loop, after the box's own signal has
   returned.  NOTE: the real DOSCAR carries VASP's own smearing
   (ISMEAR/SIGMA), so the site DOS and the spread curves are not smeared
   with the same σ.

2. A boxed "Send to decomp" control group (below the σ box, a box frame
   like the broadening box) holding TWO buttons:

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

The wrapper's boxes are made once per simulator instance (_attach_widgets)
and live in the simulator window's left column, which the window keeps
across view changes.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import simpson

if __name__ != "__main__":
    from ._imports import PlotConfig, resolve_stm, resolve_qt, resolve_doscar_parser
    from .decomp_window import DecompWindow, MO_AREA_NOISE_THRESHOLD
    from .topo_decomp_window import TopoDecompWindow
    read_doscar_header = resolve_doscar_parser().read_doscar_header
    plt, controls, widgets, mplexport = resolve_qt()
    QtCore = plt.QtCore

SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
SIGMA_DEFAULT = 0.2


class StmDecompSession:
    """
    One interactive session: real-DOSCAR header read once, then simulator
    instances come and go per σ commit.
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
        topo_decomp_offsets=(0,),
    ):
        """
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
        self.topo_decomp_offsets = [int(o) for o in topo_decomp_offsets]

        if cmap_topo is None:
            cmap_topo = plt.LinearSegmentedColormap.from_list(
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

        # The header is read early to check the DOSCAR; it carries the ONE
        # Fermi energy used anywhere in Module B.
        self.header = read_doscar_header(os.path.join(self.run_dir, "DOSCAR"))

        self.sim = None
        self.decomp_window: Optional[DecompWindow] = None
        self.topo_window: Optional[TopoDecompWindow] = None
        self._widgets: Dict[str, Any] = {}
        self._sigma_pending = self.sigma

    # ------------------------------------------------------------------
    def launch(self):
        """Prepare the working directory and start the simulator
        (non-blocking: interactive mode, qtplot.ion; the run script holds
        the process with plt.ioff() and plt.show())."""
        plt.ion()
        self._instantiate(self.run_dir)
        return self.sim

    # ------------------------------------------------------------------
    def _instantiate(self, workdir: str):
        simulator_mod, _ = resolve_stm()
        sim = simulator_mod.Interactive_STM_Simulator(
            filepath=workdir, **self._ctor_kwargs)
        self.sim = sim
        sim.run_interactive(**self.sim_kwargs)      # returns under plt.ion()

        # The wrapper's boxes in the simulator window's left column, kept by
        # the window across its view changes (add_control_box).
        self._attach_widgets()

    # ------------------------------------------------------------------
    def _attach_widgets(self):
        """
        The wrapper's controls on the current simulator window: the σ box
        (controls.SigmaBox) and the 'Send to decomp' box with its 'topo' and
        'ldos' buttons, placed by the window (add_control_box; automatic
        layout: the one rule for every window, nothing placed by hand).
        """
        sim = self.sim
        fig = sim.fig
        sigma_box = controls.SigmaBox(fig, self.sigma, SIGMA_MIN, SIGMA_MAX)
        btn_topo = controls.make(fig, 'topo')
        btn = controls.make(fig, 'ldos')

        def _on_sigma(v):
            # A value the box applied (Enter, leaving after an edit, a digit
            # step): committed when it differs from the current σ.  The
            # commit tears the window down, so it runs after the box's own
            # signal has returned (the next turn of the event loop).
            v = float(min(max(v, SIGMA_MIN), SIGMA_MAX))
            self._sigma_pending = v
            if abs(v - self.sigma) > 1e-12:
                QtCore.QTimer.singleShot(0, lambda v=v: self.commit_sigma(v))

        sigma_box.on_change(_on_sigma)
        btn.on_change(self.send_to_decomposition)
        btn_topo.on_change(self.send_topo_to_decomposition)
        sim.add_control_box(controls.box(sigma_box))
        sim.add_control_box(controls.box(btn_topo, btn, title='Send to decomp'))

        self._widgets = dict(sigma_box=sigma_box, send_ldos_btn=btn,
                             send_topo_btn=btn_topo)
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # σ commit: re-smear → new σ-tagged dir → capture → tear down →
    # re-instantiate → restore.
    # ------------------------------------------------------------------
    def commit_sigma(self, sigma: float):
        sigma = float(min(max(sigma, SIGMA_MIN), SIGMA_MAX))
        # Real-DOSCAR mode: the site DOS does not depend on σ, so there
        # is nothing to re-smear and no reason to tear the simulator
        # down. The committed σ only smears the MO spread curves of the
        # decomposition window, starting with the next Send press.
        print(f"[WRAPPER] σ commit: {self.sigma:.4f} → {sigma:.4f} "
              "(real-DOSCAR mode: applies to the MO spread curves on "
              "the next 'Send to decomposition' press).")
        self.sigma = sigma
        self._sigma_pending = sigma

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
        # what the constant-current condition integrates.
        ld_gpu, _, eg_gpu = sim._calculate_ldos_at_points_gpu(
            pts, t_emin, t_emax, use_energy_decay=False,
            preserve_orbitals=False, topo_only=True)
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
