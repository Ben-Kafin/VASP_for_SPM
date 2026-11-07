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
| `run_vacupot_plotter.py` | re-plots an existing `band_matches_rectangular.txt` without re-running the matcher; `SAVE_PATH` writes a publication file (PDF, EPS, SVG, or PNG / TIFF at `SAVE_DPI`) | `MATCH_FILE`, `SAVE_PATH`, `PlotConfig` |
| `stamp_w_uid.py` | one-time provenance stamper: writes the whitener id into an existing, consistent set of `whitener.h5` + `true_blochstates.h5` files | the directories in its `__main__` |
| `mark_bands_done.py` | one-time patch: adds the `bands_done` completion marker to `true_blochstates.h5` files written before resumable builds existed | the directory in its `__main__` |
| `profile_proj.py` | profiles where the build spends time and captures reference betas for the batched projection kernel | the directories in its `__main__` |

## The plot windows

Every plot of the package is a Qt window (PySide6 + pyqtgraph) drawn with the package's own copy
of the `qtgui` layer, `matcher/qtgui/`, which each module imports from its package (no folder
walking).
The rules are the layer's: the panels fill an automatic grid of cells and the controls are real
widgets in a strip under them (nothing is placed by hand, nothing overlaps at any window size, the
window never scrolls, a row of panels is as tall as its spatial panels ask and a window is never
trimmed to its panels); every text scales with the window's size and is never below 10 pt; a panel's title shows its short
name and the details after the first `: ` or `, ` (the entity's energy, the component names) go to
the information panel docked at the bottom of the window. The toolbar has Home, Pan, Zoom and
Save; Save writes PDF, EPS, SVG, PNG or TIFF through matplotlib (publication files), a PNG or SVG
as on screen, a screenshot, or the curves as CSV.

**Hover labels.** Every panel's hover labels (the band, energy, overlap and classification texts
of a stick, a segment or a curve) start off: a left click on the panel arms them and labels the
artist under the pointer, a right click clears the label and disarms them. In the state
decomposition's spreads panels a left click on one full state's stack arms the labels of that
state's segments alone. The combined plot's molecule panels show their two-column hover text in a
monospace label.

**Controls.** The checks are the old toggles: `Vertical`, `Horizontal`, `Amplitude`, `Curve` and
`Segmentation` (shown while Horizontal is on) in the collapsed spread plot; `Horizontal`,
`Amplitude`, `Curve` in the multi-run spread plot; `Curve`, `Amplitude` in the single-component
spread plot; `Curve`, `Amplitude` and `Norm` (shown in curve mode) in the state decomposition. The
smearing width is the `σ (eV)` number box, shown in curve mode, limited to `SIGMA_MIN`
.. `SIGMA_MAX` (0.01 to 1 eV): type a value and press Enter, or select a digit and press Up or Down
to step it (each step applies at once); a value beyond the limits is clamped to them, anything
that is not a number is reported on the console and the box put back. A toggle rebuilds the panels
from the loaded data (nothing is recomputed by it); σ recomputes the curves in place.

The energy range of a plot without an `energy_range` is the one matplotlib's autoscale gave the
old figures: every stick and reference line (the Fermi level at 0, the local Fermis, the vacuum)
with a 5 % margin at each end.

## Package `matcher/`

| module | role |
|---|---|
| `builder.py` | `TrueBlochStateBuilder`: WAVECAR/POTCAR to true Bloch states, whitening, resumable `true_blochstates.h5` writing, NCL/SOC detection (`detect_ncl_from_incar`) |
| `matcher_vacupot.py` | `run_match`: vacuum alignment (work function, dipole, vacuum potential from LOCPOT), builder, classifier, the rectangular overlap tables and the extended record (`matcher_results.h5` with grid amplitudes and cross overlaps) |
| `classifier.py` | state behaviour classification of the matches |
| `kramers.py` | the one Kramers-pairing walk and HOMO-relative offsets used by every plot |
| `vacupot_plotter.py` | `PlotConfig` and `RectAEPAWColorPlotter`, the rectangular color plot with SOC pairing: one panel per component and the full-system panel, hover labels per panel |
| `combined_vacupot_plotter.py` | two runs on one global energy axis (horizontal rows or vertical columns) |
| `spread_plot.py`, `spread_collapse.py` | how each component entity spreads its squared overlap across the full-system states (single component / several components collapsed; `MultiSpreadPlotter` stacks several runs), sticks or Gaussian-broadened curves with the σ box |
| `state_decomposition_plot.py` | how each full-system state decomposes into component states: original-states panels and stacked spreads (or gradient-coloured total curves) with energy vertical |
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

Dependencies: numpy, scipy, h5py, pymatgen, pySBT, PySide6 and pyqtgraph (the package's own
`qtgui` copy, `matcher/qtgui/`), matplotlib (the publication files written by the Save button).
