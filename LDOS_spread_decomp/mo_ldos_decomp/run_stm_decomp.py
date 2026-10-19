# -*- coding: utf-8 -*-
"""
run_stm_decomp.py — single entry point for the MO→LDOS projection package.

Flow:
  1. Check whether the builder and the EXTENDED matcher have already run
     for the configured directories (a matcher_results.h5 in full_dir
     carrying grid_amp/, cross/ incl. S_mm).
  2. If they have not, run the full pipeline via run_match: the builder
     constructs true_blochstates.h5 for every directory that lacks a valid
     one (checksum/provenance-gated), then the matcher writes the extended
     run record and the txt outputs. A virgin set of VASP directories is
     fully processed here — no prior MOPMatcher run is required.
  3. Launch the STM decomposition workflow: the unmodified simulator on a
     σ-tagged working directory (synthetic DOSCAR), with the wrapper's σ
     slider + textbox and the "Send to decomp" box (an "ldos" button for
     the per-marker LDOS decomposition and a "topo" button for the MO
     topo decomposition over topo_decomp_offsets). Set
     use_synthetic_doscar = False below to run the simulator on the run
     directory's REAL DOSCAR instead (σ then only smears the MO spread
     curves of the decomposition window).

Module A (the global tri-partition figure) stays standalone-runnable:
    python -m mo_ldos_projection.global_decomposition
(or edit its __main__ config), as do the spread plotters inside
mo_ldos_projection/mopm/matcher/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import matplotlib.pyplot as plt

from mo_ldos_projection._imports import PlotConfig, resolve_run_match
from mo_ldos_projection.sim_wrapper import StmDecompSession
from mo_ldos_projection.decomp_window import MO_AREA_NOISE_THRESHOLD

# ======================================================================
# Directories: one isolated VASP directory per molecule, the metal
# subsystem, and the full combined system (matcher outputs land there).
# ======================================================================
molecule_dirs = [#r'dir/molecule_1',
                 r'dir/molecule_1', 
                 r'dir/molecule_2', 
                 r'dir/molecule_3',
                 r'dir/molecule_4',
    #r"dir/molecule_1",
    #r"dir/molecule_2",
    
    ###r'dir/molecule_1',
]
#metal_dir = r'dir/metal'
metal_dir = r'dir/metal'

###r'dir/metal'
#"dir/metal"
#full_dir = r'dir/full'
full_dir = r'dir/full'

###r'dir/full'
#"dir/full"

# ======================================================================
# Builder + matcher settings (run_MOPMatcher.py conventions).
# ======================================================================
run_kwargs = {
    # --- Builder settings ---
    "k_index": 1,                  # which k-point (1-based)
    "tol_map": 1e-3,               # atom-matching tolerance (Å)
    "check_species": True,         # enforce species matching
    "reuse_cached": True,          # skip builder if valid .h5 files exist

    # --- Band window: slice object or None per molecule ---
    "band_window_molecules": [slice(0, 82), slice(0, 82)],

    # --- Vacuum alignment ---
    "reuse_vac_cache": True,
    "curvature_tol": 5e-8,
    "dipole_threshold": 0.05,

    # --- Global zero reference ---
    "zero_reference": "full",      # "metal", "full", or a molecule folder name

    # --- Local Fermi-line mode per directory (see run_MOPMatcher.py) ---
    "fermi_mid_molecules": True,
    "fermi_mid_metal": False,
    "fermi_mid_full": False,
}

# ======================================================================
# STM / decomposition settings.
# ======================================================================
sigma = 0.156                        # committed smearing width (eV, VASP SIGMA)
mo_area_threshold = MO_AREA_NOISE_THRESHOLD   # decomp-window noise filter

# MO topo decomposition input (the "topo" button in the Send-to-decomp
# box): HOMO-relative Kramers-entity offsets, spread_collapse comp_indices
# convention (0 = HOMO, -1 = HOMO-1, +1 = LUMO, ...). One single-color
# alpha-ramp map (MO color, opaque at max intensity, transparent at min)
# per offset per molecule component, built from the topo's own LDOS
# curves over sorted([0, topo_bias]).
topo_decomp_offsets = [-2, -1, 0, 1]

# DOSCAR source (input option, decided at launch — not a figure toggle):
#   True  = σ-smeared synthetic DOSCAR built from the PROCAR, in σ-tagged
#           working dirs; σ commits re-smear and re-instantiate (the one-σ
#           consistency between site DOS and MO spread curves holds).
#   False = the simulator runs on the run directory's REAL DOSCAR (its own
#           VASP smearing); σ commits then only set the MO spread smearing
#           used by the decomposition window at the next Send press.
use_synthetic_doscar = False

erange = [-2.425, -1.225]               # initial energy slider range
ldos_height = 1.0                 # Å above highest atom for LDOS sampling
unit_cell_num = 2

sim_kwargs = dict(
    grid_res=128,                   # NxN topography grid
    topo_bias=-2.0,                 # V for constant-current topography
    topo_height=5,               # initial tip height above highest atom (Å)
    ldos_bias_sign="neg",          # 'pos' or 'neg' — which slider sets the bias
    use_decay=True,                # Chen barrier model (turn OFF before sending
                                   # to decomposition — the button enforces it)
    show_decay_toggle=True,
    reuse_cache=True,              # per-σ-dir topo caches
    #line_endpoints=None,
    #line_endpoints=([11.935324, -5.631926], [4.789904, 23.200468]),
    
    #marker_positions=[[4.784411, 7.921674], [7.675310, 4.182948]],
    #marker_positions=[[4.536844, 7.713255], [13.012102, 2.932340]],
    #marker_positions=[[4.784411, 7.921674], [7.675310, 4.182948], [13.211896, 10.788004]],
    #marker_positions=[[8.668339, 7.550643], [10.411960, 0.514981], [6.837538, 14.938088]],
    #LINE1_60deg of long axis of void
    line_endpoints=([12.060682, -8.640524], [5.542054, 22.573677]),
    marker_positions=[[11.475490, -5.838353], [7.171550, 14.770895], [8.722939, 7.342131]],
    #LINE2_90deg of long axis of void
    #line_endpoints=([2.533456, -5.882643], [14.191772, 19.439721]),
    #marker_positions=[[2.837332, -5.222612], [10.911140, 12.314046], [8.861003, 7.861061]],
    extra_broadening=None,
    path=None,
    path_extend=False,
)

cfg = PlotConfig(
    cmap_name_simple="managua_r",
    cmap_name_metal="vanimo_r",
    shared_molecule_color=True,
    min_total_mol_wspan=0.01,
    power_simple_neg=0.25,
    power_simple_pos=0.75,
    power_metal_neg=0.075,
    power_metal_pos=0.075,
    power_residual=0.5,
    soc_pair_thresh_eV=0.005,
)


# ======================================================================
def extended_matcher_ready(full_dir: str):
    """True when full_dir holds a matcher_results.h5 with the extended
    datasets (grid_amp/, cross/ incl. S_mm) the projection needs."""
    path = os.path.join(full_dir, "matcher_results.h5")
    if not os.path.isfile(path):
        return False, "matcher_results.h5 missing"
    try:
        with h5py.File(path, "r") as hf:
            if "grid_amp" not in hf or "cross" not in hf:
                return False, "run record lacks grid_amp/cross (pre-extension run)"
            if "S_mm" not in hf["cross"]:
                return False, "run record lacks cross/S_mm (pre-amendment run)"
    except Exception as e:
        return False, f"matcher_results.h5 unreadable ({type(e).__name__})"
    return True, "extended run record present"


if __name__ == "__main__":
    ready, reason = extended_matcher_ready(full_dir)
    if ready:
        print(f"[RUN] Matcher outputs ready: {reason} — skipping the "
              "builder/matcher phase.")
    else:
        print(f"[RUN] {reason} — running the builder/matcher pipeline "
              "(builder only rebuilds directories without a valid "
              "true_blochstates.h5).")
        run_match = resolve_run_match()
        run_match(
            molecule_dirs=molecule_dirs,
            metal_dir=metal_dir,
            full_dir=full_dir,
            **run_kwargs,
        )
        ready, reason = extended_matcher_ready(full_dir)
        if not ready:
            raise RuntimeError(
                f"matcher run finished but the record is not ready: {reason}")

    print("[RUN] Launching the STM decomposition workflow "
          f"(σ = {sigma:.4f}, "
          f"{'synthetic' if use_synthetic_doscar else 'real'} DOSCAR).")
    session = StmDecompSession(
        full_dir,
        erange=erange,
        ldos_height=ldos_height,
        unit_cell_num=unit_cell_num,
        sigma=sigma,
        sim_kwargs=sim_kwargs,
        cfg=cfg,
        threshold=mo_area_threshold,
        use_synthetic_doscar=use_synthetic_doscar,
        topo_decomp_offsets=topo_decomp_offsets,
    )
    session.launch()

    # Hold the app: the wrapper runs under interactive mode so the
    # simulator's plt.show() returns; block here until every window closes.
    plt.ioff()
    plt.show()
