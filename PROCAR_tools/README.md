# PROCAR_tools

One PROCAR reader (`procar_parser.py`) and the tools built on it. Every tool imports
`ProcarParser`; none parses PROCAR itself.

## `procar_parser.py`

`ProcarParser(directory)` then `.parse()` reads PROCAR (and DOSCAR for the Fermi energy, POSCAR
for the ion-to-element map, INCAR only as a cross-check). Handled from the file structure alone:
ISPIN=1, ISPIN=2 (second projection set), noncollinear / SOC (total plus the three magnetization
blocks), LORBIT=10 (s p d), 11 (lm), 12/13/14 (lm plus the complex phase block), any number of
k-points, bands and ions, any orbital column set (spd, spdf), single-ion files without `tot` rows.
Unrecognized layouts raise with the line number. Arrays: `energies`, `occupations`, `total`,
`ion_tot`, `tot_rows`, `mag`, `phase`, `phase_ion_sum`, `charge_rows` (spin axis always first);
views: `get_states()` (one dict per spin, k-point, band), `states_by_index()`. The complex-block
tools of the former `complex_procar_parser.py` live here (`build_state_types`, `filter_states`,
`combine_kpoint_bands`, `export_combined_bands`, `run`). `orbital_groups()` maps the file's column
names to s/p/d/f groups.

## Tools

| script | does | reads | run block settings |
|---|---|---|---|
| `Band_structure_plot.py` | interactive fatband plot: element mode (RGB = named elements' total projections, all orbital columns) and orbital mode (s/p/d hue per element); SOC Kramers doublets merged along the path; hover labels with mplcursors | PROCAR, POSCAR, DOSCAR, VASPKIT `BAND.dat` and `KLABELS` | `directory`, `filter_types` (up to three elements, `None` skips a channel) |
| `procar_filter_lorbit14.py` | `ProcarStatePlotter`: tags each state with its contributing elements, drops states made only of `filter_types`, plots per-ion totals as sticks, combines k-points per band (per spin and band for ISPIN=2) and exports the normalized composition matrices | PROCAR, POSCAR | `directory`, `fermi_level`, `filter_types` |
| `phase_DOS.py` | phase-sensitive (coherent) site- and orbital-projected DOS from the complex phase block, written to `PHASE_SENSITIVE_DOSCAR.dat`; needs LORBIT=12/13/14 | PROCAR, POSCAR, DOSCAR | `target_dir`, `energy_range`, `n_points`, `sigma` |
| `procar_spin.py` | `GPUComplexProcarParser`: GPU workflow that pairs the two spin channels per k-point and band, applies per-element square-root Gaussian broadening and k-weighting, and saves `Broadened_Orbital_Curves.npy`; needs LORBIT=12/13/14 and cupy | PROCAR, POSCAR, DOSCAR | `target_dir`, `my_sigmas` |
| `unfold_bands.py` | band-structure plot from VASPKIT `BAND.dat` / `KLABELS`, Fermi energy from `BAND_GAP` when present, with the gap evaluated at the central k-point (Γ if present) (does not read PROCAR) | BAND.dat, KLABELS, BAND_GAP | `filepath` |

Dependencies: numpy, matplotlib; cupy for `procar_spin.py`; mplcursors (optional) for
`Band_structure_plot.py`.
