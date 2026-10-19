# MOPmatcher_mem (MOPMatcher)

PAW-correct molecular-orbital projection matching. Builds true (all-electron) Bloch states from
WAVECAR + POTCAR for each component calculation (isolated molecules, bare metal / adatom surface)
and for the full system, whitens the component states, and matches every full-system Bloch state
against the components by squared overlap. Energies are put on one vacuum-aligned axis from each
run's LOCPOT work function. Collinear (ISPIN=1, 2) and noncollinear / SOC runs are auto-detected;
under SOC, Kramers pairs are combined everywhere a weight is shown.

## Run scripts

| script | does | settings to edit |
|---|---|---|
| `run_MOPMatcher.py` | the full pipeline: `run_match` (builder + matcher) for the configured molecule, metal and full directories, then the rectangular color plot through `RectAEPAWColorPlotter` | `run_kwargs`, `molecule_dirs`, `metal_dir`, `full_dir`, the `PlotConfig` fields |
| `run_combined_MOPMatcher.py` | plots two finished runs (a "complex" run and a "full" run sharing the same molecules) on one vacuum-aligned energy axis | the two run directories and the `PlotConfig` |
| `run_vacupot_plotter.py` | re-plots an existing `band_matches_rectangular.txt` without re-running the matcher | `MATCH_FILE`, output figure path, `PlotConfig` |
| `stamp_w_uid.py` | one-time provenance stamper: writes the whitener id into an existing, consistent set of `whitener.h5` + `true_blochstates.h5` files | the directories in its `__main__` |
| `mark_bands_done.py` | one-time patch: adds the `bands_done` completion marker to `true_blochstates.h5` files written before resumable builds existed | the directory in its `__main__` |
| `profile_proj.py` | profiles where the build spends time and captures reference betas for the batched projection kernel | the directories in its `__main__` |

## Package `matcher/`

| module | role |
|---|---|
| `builder.py` | `TrueBlochStateBuilder`: WAVECAR/POTCAR to true Bloch states, whitening, resumable `true_blochstates.h5` writing, NCL/SOC detection (`detect_ncl_from_incar`) |
| `matcher_vacupot.py` | `run_match`: vacuum alignment (work function, dipole, vacuum potential from LOCPOT), builder, classifier, the rectangular overlap tables and the extended record (`matcher_results.h5` with grid amplitudes and cross overlaps) |
| `classifier.py` | state behaviour classification of the matches |
| `kramers.py` | the one Kramers-pairing walk and HOMO-relative offsets used by every plot |
| `vacupot_plotter.py` | `PlotConfig` and `RectAEPAWColorPlotter`, the rectangular color plot with SOC pairing |
| `combined_vacupot_plotter.py` | two runs on one global energy axis |
| `spread_plot.py`, `spread_collapse.py` | how each component entity spreads its squared overlap across the full-system states (single component / several components collapsed) |
| `state_decomposition_plot.py` | how each full-system state decomposes into component states |
| `wf_dpl_vacupot.py` | work function, dipole step and vacuum level from LOCPOT and DOSCAR/OUTCAR |

## Physics modules (bundled, from QijingZheng's VaspBandUnfolding)

`vaspwfc.py` (WAVECAR reader), `paw.py` (PAW projectors and POTCAR), `aewfc.py` (all-electron
wavefunctions; carries the SOC spinor patch), `sph_harm.py` (real/complex spherical harmonics, with
the scipy >= 1.17 `sph_harm_y` shim), `vasp_constant.py`. `proj_batched.py` is the batched (GEMM)
reimplementation of the PAW projection, kept separate from `paw.py` for head-to-head comparison.

## Inputs and outputs

Reads WAVECAR, POTCAR, POSCAR/CONTCAR, DOSCAR, OUTCAR, LOCPOT and INCAR of every run. Writes
`whitener.h5`, `true_blochstates.h5`, `matcher_results.h5`, `band_matches_rectangular*.txt` and
the figures into the full-system directory (or the configured `output_path`).

`directories.txt` is a notes file of directory sets used for past runs; no code reads it.

Dependencies: numpy, scipy, matplotlib, h5py, pymatgen, pySBT.
