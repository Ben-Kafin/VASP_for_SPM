# VASP_for_SPM

VASP post-processing codes for scanning probe work (STM/STS on molecule/metal systems), collected
as a suite of independent packages. Each top-level folder is its own package with its own README:
run a package's scripts from inside that folder (or with that folder on `sys.path`). Nothing in one
package imports another, with two exceptions: the MO to LDOS decomposition uses the matcher and the
simulator through fixed sibling paths (see `LDOS_spread_decomp/README.md`), and every interactive
window draws through the shared Qt layer `qtgui/` at the root of the repository, which each window
finds by walking up from its own file (see [The interactive windows](#the-interactive-windows)).

| Package | Purpose |
|---|---|
| `LDOS_spread_decomp/MOPmatcher_mem` | MOPMatcher: PAW-correct Bloch-state overlap matching of molecule and metal states against a full system, with the vacuum-aligned, Kramers-paired plots |
| `LDOS_spread_decomp/stm_from_dft_opt` | GPU (CuPy) STM / LDOS simulator from DFT output (constant-current topography, dI/dV maps, line scans) |
| `LDOS_spread_decomp/mo_ldos_decomp` | MO to LDOS projection: attributes simulated LDOS features to the gas-phase molecular orbitals using the two packages above |
| `dIdV_tip_sample` | dI/dV simulation from a sample LDOS and a tip LDOS |
| `PROCAR_tools` | one shared PROCAR reader and the tools built on it: fatband plot, state filter/export, phase-sensitive DOS, GPU broadened orbital curves, band unfolding plot |
| `PROJCAR_tools` | PROJCAR parser and LOCPROJ line generator |
| `WAVECAR_tools` | WAVECAR coefficient parser, G-point facet-duplicate checker, wavefunction orthonormality test |
| `charge_density` | CHGCAR / PARCHG / LOCPOT slicing and plotting, PARCHG naming and projections, planar averages |
| `POSCAR_tools` | POSCAR creation and manipulation (layers, molecules from xyz, dimers, pyramids and tips, subsets, selective dynamics, supercells, unit-cell reshaping) |
| `convergence` | relaxation monitoring from OUTCAR (forces, convergence plot, lowest-force structure) |
| `LOCPOT_tools` | electric field from LOCPOT as volumetric files |
| `qtgui` | the shared Qt layer of every interactive window: `qtplot` (figures, axes, artists, events, the toolbar, the information panel, the automatic layout, the publication export), `controls` (the one control set), `widgets` (the control table, the hover labels), `mplexport` |
| `checks` | the offscreen render-and-measure checks of the windows (`render_windows.py` and one `check_*.py` per window family) |

## Dependencies

Python 3 with numpy, scipy and matplotlib everywhere. Every interactive window needs PySide6 and
pyqtgraph (the `qtgui` layer; `pip install "PySide6<6.12" pyqtgraph`, or Python 3.12 or later with
any PySide6: PySide6 6.12.0 has a reference-counting bug on older Pythons that `qtplot` warns
about at import). matplotlib is not used to draw on screen: it writes the publication files the
Save button offers (PDF, EPS, SVG, PNG, TIFF) and draws the 3D panel of `convergence/forces.py`,
the one panel the layer has no counterpart for. In addition: pymatgen (`dIdV_tip_sample`,
`LOCPOT_tools`, `WAVECAR_tools`, `MOPmatcher_mem`, `stm_from_dft_opt`), cupy with a CUDA GPU
(`stm_from_dft_opt`, `mo_ldos_decomp`, `PROCAR_tools/procar_spin.py`), h5py (`MOPmatcher_mem`,
`mo_ldos_decomp`), pySBT (`MOPmatcher_mem`, from github.com/QijingZheng/pySBT), pytest
(`mo_ldos_decomp/mo_ldos_projection/tests`). VASPKIT output (BAND.dat, KLABELS) is read by the
band-structure plotters.

## The interactive windows

Every window that has controls or answers the mouse (the STM simulator and its decomposition
windows, the matcher's spread and decomposition plots, the DOS plotter, the fatband plot, the
forces viewer, the two-system slice) is a Qt window drawn by PySide6 and pyqtgraph through
`qtgui/`, converted from the matplotlib-widget figures by the rules of the guide "From a
matplotlib-wired GUI to a real GUI" (the sm4_Browser conversion). The rules, as they show in the
windows:

- **Automatic layout.** Nothing is placed by hand: every panel is a cell of the window's Qt layout
  and every control a Qt widget in it, so nothing overlaps at any window size and a window never
  scrolls (a panel that holds only text scrolls inside itself). A spatial panel (a topography, a
  map) keeps its x:y ratio; the panels of one row share one height.
- **One control set.** A control of a given name is the same in every window (`qtgui/widgets.py`,
  `SPECS`): the checks, the radio, the buttons, and number boxes instead of sliders. In a number
  box, select one or more digits and press Up or Down: the selected digits step by one in the place
  of the last selected digit, carrying as arithmetic does, and the same digits stay selected; with
  nothing selected the digit left of the cursor steps; a box with limits stops at them. Each step
  applies at once; Enter applies a typed value, leaving the box applies it after an edit. The one
  slider left is the window-start slider of `convergence/forces.py`, which scrubs through the
  ionic steps.
- **Hover labels.** Where a window labelled its curves or markers on hover (the former mplcursors
  cursors), a left click on the panel arms the labels and labels the item under the click, the
  label follows the item under the pointer and stays until another is hovered, and a right click
  removes it and disarms the panel (`qtgui/widgets.py`, `HoverLabels`). The simulator's and the
  DOS plotter's orbital curves are labelled without arming, as before.
- **Short titles, an information panel.** A panel title shows its short name; the details (bias,
  height, window, counts) are one line each in the information panel docked at the bottom of the
  window, selectable, where they can be copied from.
- **One text-size rule.** Every window has a design size (its `DESIGN` constant, the old figure
  size); all of its text scales with the window's size relative to it, from 0.5x to 4x, and no
  text is ever cut: when the controls would not fit, their text shrinks together to at most 75 % of
  that, and below that the window stops shrinking. A window taller than it is wide shows the
  short labels (`SHORT_LABELS`).
- **Saving.** The toolbar's Save button writes PDF, EPS, SVG, PNG or TIFF through matplotlib
  (`qtgui/mplexport.py`: the figure at its design size, every line and text an editable object,
  the controls left out), PNG / SVG as drawn on screen, a screenshot of the window, or the curves
  as CSV; `fig.savefig(path, dpi=300)` from code does the same.
- **The mathematics is unchanged.** The windows call the same functions as before; only the
  drawing, the events and the controls are Qt.

Running a window's module as a script works as before (`python run_DFT_STM.py`, `python
Band_structure_plot.py`, ...); the layer is found from the window's own location, so the
repository's folders must stay together. The checks under `checks/` open every window offscreen
on synthetic inputs, resize it to its design size, half, double and taller than wide, and measure
that no text is cut, no controls overlap and the labels follow the rules (`QT_QPA_PLATFORM=offscreen
python checks/check_simulator.py`, `check_decomp.py`, `check_matcher_plots.py`,
`check_misc_windows.py`).

## Third-party code

`MOPmatcher_mem` bundles the physics modules of QijingZheng's VaspBandUnfolding (vaspwfc.py,
paw.py, aewfc.py, sph_harm.py, vasp_constant.py; aewfc.py carries a spinor patch for SOC and
sph_harm.py a shim for scipy >= 1.17). `charge_density/lib.py` and the parsers inside the charge
density scripts, `convergence/plot_forces.py` and `dIdV_tip_sample/lib.py` come from Ben McDowell's
repositories (github.com/benwmcdowell) with Benjamin Kafin's edits. Everything else is Benjamin
Kafin's own code.

## Paths in the run blocks

Every script keeps its settings in the `if __name__ == "__main__":` block at the bottom. In the
clean distribution the directory strings there are placeholders that say what the path must point at:

| placeholder | points at |
|---|---|
| `dir/run` | one VASP calculation directory (the files named in the package README) |
| `dir/run/<file>` | a specific file inside such a directory |
| `dir/run_1`, `dir/run_2`, ... | several calculation directories of the same kind |
| `dir/run_complex`, `dir/run_surface`, `dir/run_sam` | the complex, bare-surface and SAM runs of a combined plot |
| `dir/molecule_N` | the N-th isolated-molecule (component) calculation |
| `dir/metal` | the bare metal / adatom-surface calculation |
| `dir/full` | the full (molecule + surface) calculation |
| `dir/sample`, `dir/tip` | the sample and the tip calculation of the dI/dV simulation |
| `dir/system_1`, `dir/system_2` | the two systems of a side-by-side comparison |
| `dir/output`, `dir/output/<file>` | where a script writes its result |
| `dir/package` | the folder of this package when a script adds it to `sys.path` |
