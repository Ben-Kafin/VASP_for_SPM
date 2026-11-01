# mo_ldos_decomp (MO to LDOS projection)

Attributes features of simulated STM LDOS spectra and topographies to the original gas-phase
molecular orbitals, using MOPMatcher's PAW-correct full-wavefunction overlaps (no atom/orbital PDOS
partitioning). Package `mo_ldos_projection/` plus the run scripts next to it.

The package reaches the matcher and the simulator through `mo_ldos_projection/_imports.py`, which
expects `../MOPmatcher_mem/` and `../stm_from_dft_opt/` next to this folder (two levels above the
package). Keep the three folders side by side. The windows draw with the shared Qt layer `qtgui`
at the repository root (PySide6 + pyqtgraph; every window finds it by walking up from its own
file), see [The windows](#the-windows).

## Run scripts (edit the settings at the top of each `__main__` block)

| script | does |
|---|---|
| `run_stm_decomp.py` | single entry point: checks for an extended `matcher_results.h5` in the full directory (runs `run_match` if needed), then launches the simulator wrapper, which puts the σ box and the "Send to decomp" box (`topo` and `ldos` buttons) in the simulator window's control strip; the buttons open the LDOS and topo decomposition windows. Settings: `molecule_dirs`, `metal_dir`, `full_dir`, `run_kwargs`, `sigma`, `mo_area_threshold`, `topo_decomp_offsets` |
| `run_mo_band_lists.py` | classifies full-system bands by one MO's standing among its component's MOs and writes the vaspkit job that generates the qualifying real-space squared wavefunctions (the SLURM account and mail lines in the job template are placeholders in the clean copy) |
| `run_wfn_mo_sum.py` | MO-weighted sum of the vaspkit real-space squared wavefunctions, written as one CHGCAR-format probability map |
| `run_wfn_plane_slice.py` | vertical plane slice through that sum (or any CHGCAR-format file): one panel per input on one shared colour scale with one colour bar, the PNG written first, then a live `γ` box under the panels that recolours the maps (any γ > 0) |

## Package modules

| module | role |
|---|---|
| `global_decomposition.py` | Module A: the global tri-partition figure (MO subspace, MO-orthogonalized metal, residual) with the `σ (eV)` box (0.01 to 1 eV, three decimals; the curves are recomputed on every change) and `Save data` (`global_decomposition.txt`) under the panel |
| `procar_dos.py` | Module B: strict LORBIT=14 noncollinear single-Γ PROCAR reader with validation (used only here; the general reader lives in `PROCAR_tools`) |
| `synthetic_doscar.py` | σ-smeared synthetic DOSCAR writer and σ-tagged working directories |
| `sim_wrapper.py` | launcher around the simulator: the σ box (`controls.SigmaBox`) and the "Send to decomp" box (`topo`, `ldos`) in the simulator window's layout |
| `decomp_window.py` | `DecompWindow`: the per-marker LDOS + MO decomposition window (one block of panels per marker, rebuilt on every toggle). Controls under the panels: the layout toggles `Vertical`, `Horizontal`, `Amplitude`; the `area ≥` box (the MO area threshold; Enter or leaving the box applies it, a non-number is put back); the `Metal curve` toggle; `Save data` (`decomp_window_data.txt` in the run directory). Hover labels on every panel (see below) |
| `topo_decomp_window.py` | `TopoDecompWindow`: the offsets × components grid of single-colour alpha-ramp maps, one colour bar beside every drawn map, the `Amplitude` and `Norm` toggles and `Save data` (`topo_decomp_data.txt`) under the grid; the figure background is transparent in the exports |
| `_imports.py` | import resolution of the matcher and simulator packages |
| `tests/` | pytest suite, run offscreen: `QT_QPA_PLATFORM=offscreen python -m pytest mo_ldos_projection/tests` from this folder (`conftest.py` sets the offscreen platform itself when none is set and takes the example directory from `MO_LDOS_EXAMPLE_DIR`; the tests that need the example run or a GPU skip without them) |

## The windows

Every window of this package is a Qt window of the shared layer (`qtgui/` at the repository
root), laid out automatically: the panels sit in a grid of cells and the controls in one strip
under them, nothing is placed by hand, and a window never scrolls. A row of panels is as tall as
its spatial panels ask (no empty band above or below them), and a window whose panels are all
spatial (the topography decomposition, the plane slice) takes the size they give it at its first
show and after every resize. Its toolbar has Home, Pan, Zoom
and Save; Save writes PDF, EPS, SVG, PNG or TIFF through matplotlib (every curve and text an
editable object, the controls left out), a PNG or SVG as drawn on screen, a screenshot, or the
curves as CSV. A panel title shows its short name; the details after the first `": "` go to the
information panel docked at the bottom (the topo window's σ, window, bias and height; a marker's
position). Every text scales with the window's size relative to its design size, and in a window
taller than wide the controls show their short labels.

Number boxes (`area ≥`, `σ (eV)`, `γ`): type a value and press Enter, or leave the box after an
edit; or select one or more digits and press Up or Down to step them in the place of the last
selected digit, which applies at once. A σ outside 0.01 to 1 eV is clamped to the limits; a γ must
be a finite number above 0 (above 2 included); an entry that is not a number is reported on the
console and the box put back.

Hover labels (the decomposition window): a left click on a panel arms its labels and labels the
curve under the click (an MO curve's projection, the LDOS area, the metal composite's projection);
moving over another curve moves the label; a right click clears the label and disarms the panel.

`checks/check_decomp.py` at the repository root opens every window of this package offscreen on
synthetic inputs, scripts every control and measures the layout at four sizes.

Inputs: the MOPMatcher outputs (`matcher_results.h5`), PROCAR (LORBIT=14), DOSCAR, LOCPOT,
POSCAR/CONTCAR of the full run, vaspkit real-space wavefunction files. Outputs: decomposition
figures, synthetic DOSCARs in σ-tagged working directories, CHGCAR-format maps.

Dependencies: those of `MOPmatcher_mem` and `stm_from_dft_opt`, PySide6 and pyqtgraph (the shared
`qtgui` layer at the repository root; matplotlib stays for the publication files), plus pytest for
the tests.
