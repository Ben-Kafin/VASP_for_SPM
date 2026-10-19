# stm_from_dft_opt

GPU-accelerated STM / LDOS simulator from DFT output (package `stm_from_dft/`), Tersoff-Hamann
style with the Chen tunneling barrier from the local potential. Auto-detects ISPIN=1, ISPIN=2 and
noncollinear / SOC calculations from DOSCAR and LOCPOT.

## Run

`run_DFT_STM.py`: set `v_dir` to the VASP directory containing DOSCAR, LOCPOT and POSCAR/CONTCAR,
then adjust the `Interactive_STM_Simulator(...)` arguments (energy range, LDOS sampling height,
colormap, number of unit cells) and the `run_interactive(...)` arguments (topography grid, bias,
tip height, decay model on/off, cache reuse, line-scan endpoints, marker positions, extra
broadening per element, path mode seed). The interactive figure gives constant-current topography,
dI/dV at markers, line scans and path scans with sliders and toggles.

## Package `stm_from_dft/`

| module | role |
|---|---|
| `simulator.py` | `Unified_STM_Simulator` (GPU LDOS, Simpson integration, Chen barrier, constant-current convergence, caches) and `Interactive_STM_Simulator` (the figure and its controls). Decay model: Tier 2 with the optional occupied-side 0-clamp |
| `doscar_parser.py` | `SpinAwareDosParser`, `SpinMode`: DOSCAR reader for all three spin modes |
| `locpot_manager.py` | `LocpotManager`: LOCPOT loading and `.npy` caching for all three spin modes |
| `plot_dos_cursors_levels.py` | `DosPlotter`: DOS plot with energy cursors and level markers (run block at the bottom, `v_dir`) |
| `tier3_decay.py` | UNUSED, fully commented: the Tier 3 decay model, kept as reference for future work on the tunneling barrier |
| `sim_decay_full.py` | UNUSED, fully commented: the full-model frozen-bias decay build of the simulator, kept as reference for the same work |

The two UNUSED files are not imported by `__init__.py`; removing the leading `# ` from every line
restores them exactly.

Inputs: DOSCAR, LOCPOT, POSCAR or CONTCAR (and INCAR for the spin-mode check). Outputs: figures and
the `global_topo_*` / `ldos_topo_*` `.npy` caches in the run directory.

Dependencies: numpy, scipy, matplotlib, cupy (CUDA GPU), pymatgen, mplcursors.
