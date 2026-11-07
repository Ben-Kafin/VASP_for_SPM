# stm_from_dft_opt

GPU-accelerated STM / LDOS simulator from DFT output (package `stm_from_dft/`): the site- and
orbital-projected DOS weighted by Chen's energy-dependent tunneling matrix element, with the tip DOS
constant as in Tersoff and Hamann and the barrier taken from the local potential at each tip point
(the `Decay` row below). Auto-detects ISPIN=1, ISPIN=2 and noncollinear / SOC calculations from
DOSCAR and LOCPOT.

## Run

`run_DFT_STM.py`: set `v_dir` to the VASP directory containing DOSCAR, LOCPOT and POSCAR/CONTCAR,
then adjust the `Interactive_STM_Simulator(...)` arguments (energy range, LDOS sampling height,
colormap, number of unit cells) and the `run_interactive(...)` arguments (topography grid, bias,
tip height, decay model on/off, cache reuse, line-scan endpoints, marker positions, extra
broadening per element, path mode seed). The window gives constant-current topography, dI/dV at
markers, line scans, path scans and energy maps.

## The window

The window is a Qt window (PySide6 + pyqtgraph) drawn with the package's own copy of the `qtgui`
layer, `stm_from_dft/qtgui/` (imported from the package, no folder walking): its panels and
controls sit in one automatic layout (nothing is placed by hand, nothing overlaps at any window
size, the window never scrolls), every text follows the window's size and is never below 10 pt,
and a panel's title shows its short name with the details (bias, height, the marker) in the
information panel docked at the bottom. The panels are arranged by one rule (the guide's rule 11):
every plotted panel is a first-class citizen, the plots fill the window, spatial panels keep their
ratio, nothing plotted is shrunk to make room, and blank space is what is minimized. The shaped
groups of a view, the placement topo (the Global Topo, where the markers, the line ends and the
path nodes are dragged, with its boxes above), the heatmap row of Line and Path (its label over
it), the Global Topo and Map Topo pair and the block of energy maps of Map (the maps wrap into the
rows that make them largest), and the curve panels (Tip Height and Partitioned LDOS, side by side
or stacked by their room) are placed by the arrangement that scores best, the smallest group's size
against its size alone less the blank fraction, chosen again only when a view's panels are built or
their number changes (the drill-down level, the per-heatmap colour bars), when a crop changes a
panel's shape and when the window is resized (maximizing included); the window is never trimmed to
its panels, so a maximized window stays full screen. The line ends, the path nodes and the Single
Point and Map markers recompute on release: the tip heights, the LDOS and the spectra a drag
changes are computed once, when the button is let go. The Topo h box, Crop and Uncrop sit
above the Global Topo, the LDOS h box above the panel its height belongs to (the Line view's Topo
panel, the Path view's tip-height stripe, the Map Topo). The Global Topo can be cropped as the
sm4_Browser Line window crops its heatmap: two draggable cursors per axis start at the full extent
of the drawn cells, the area outside each pair is shaded, a box riding with each cursor above (x)
and right of (y) the panel shows its value and takes a typed one, Crop applies the ranges between
the cursors as the panel's limits (in Map the Map Topo takes the same) and Uncrop restores the
extent; a crop changes the panel's shape and the arrangement follows (a tall crop, a tall layout).
The window opens on the screen's available area. The toolbar has Home, Pan, Zoom and Save; Save writes PDF,
EPS, SVG, PNG or TIFF through matplotlib (publication files), a PNG or SVG as on screen, a
screenshot, or the curves as CSV.

| control | where | what it does |
|---|---|---|
| `mode` radio | left column, top | the view: `Single Point`, `Line`, `Path`, `Map` (the panels of the view are rebuilt; the controls stay) |
| `broadening` box | left column | one check per atom type with its `σ <type>` box (eV, 0 to 1, two decimals), shown while the check is on: extra Gaussian broadening of that type's DOS |
| `RUN` / `STOP` | left column, bottom | starts and stops the computation; on STOP the line endpoints / path nodes and marker positions are printed in the form `run_interactive` takes |
| toggles | beside RUN | `Atoms`, `Decay`, `Norm`, `Feenstra`, `broadened` (under Feenstra), `Mag`, `Cell`, `Extend` |
| `Decay` | toggles | the tunneling matrix element that weights the DOS of every atom image at a tip point, with the tip DOS constant (the s-wave tip of J. Tersoff and D. R. Hamann, Phys. Rev. B 31, 805 (1985)). On (the default): Chen's energy-dependent element (C. J. Chen, *Introduction to Scanning Tunneling Microscopy*, 3rd ed., Oxford 2021, Sec. 2.2.3-2.2.4): a state at E (eV from E_F) at distance d from the tip at bias V carries exp(-2 κ0 d + κ0 (E - V/2) d / φ), κ0 = √(2 m φ)/ħ (Eqs. 2.34, 2.37-2.39: κ to first order in the state's energy E - V/2 from the middle of the bias window), φ the LOCPOT potential at the tip point minus E_F (Chen's φ̄, one average work function for the junction, taken here at each tip point). As in a measurement, the topography is converged on the current at one frozen bias (Eq. 2.40: the window [0, V_b], every state at E - V_b/2), and the displayed LDOS is dI/dV with the bias swept (the derivative of Eq. 2.40 at each V: the state entering at the window edge plus the change of every state already in [0, V]; that integral starts at 0 V whether or not the window reaches it). Off: exp(-2 κ0 d), the same element without its energy dependence. κ0 = 0.512317 Å⁻¹ eV^-1/2 × √φ, from m_e, h and e, in every use (the tip-height convergence step included). The topography caches carry `_decay_chen` (on) or `_decay_k0` (off) |
| `Feenstra` | toggles | Feenstra's normalized conductance (dI/dV)/(I/V) on top of the calculated LDOS (`feenstra.py`): at the same tip positions as the display, I(V) from 0 V with the σ broadening in force (trapezoid rule, I(0) = 0 exactly; a window that does not reach 0 V is extended to it for the integral): with Decay on, Chen's Eq. 2.40 at each V (the window [0, V], every state weighted at that frozen bias, as the topography's current); with Decay off, the total LDOS integrated. I/V is formed, and the spectra, heatmaps and maps are divided by it along their energy axis; every partition, atom and orbital of a point is divided by the same total I/V (with `Mag` on, the magnetization over the total I/V). Display only: no LDOS recompute; `Norm`, when on, acts on the ratio. A point with no states between 0 V and V (0/0) shows 0. The spectrum's y axis and the label over the heatmaps and maps read (dI/dV)/(I/V) |
| `broadened`, `width (V)` | under Feenstra; the width box under the toggles, shown while `broadened` is on | the I/V broadened by the sm4_Browser's rule: at each energy the exponentially weighted mean of I/V with the given width (V, off at open, 0.2 V, at least 0.001 V) over the samples the current was integrated on; off, the plain I/V (a sample exactly at 0 V given its neighbours' I/V) |
| `cmap norm` | bottom strip, Map only | off: every energy map on one colour scale, the largest value over all of them, with one colour bar; on: each map its own scale with its own bar beside it (as the sm4_Browser Map window's `cmap norm`). The maps' colour map is the constructor's `cmap_map`: `MAP_CMAP` by default, the sm4_Browser's (`custom_map`, midnight blue to light sky blue); any matplotlib colour-map name or Colormap |
| contour slice | the topo colour bar beside the Map Topo | the sm4_Browser Map window's: a cyan level line on the bar (half the range at open) is dragged by a press on it, following the cursor's height while the button is down even off the narrow bar sideways; the Map Topo's contour at that height, the perimeters of everything above it (WSxM's flooding, as outlines), is drawn on every energy map as white 1 pt traces, at the 2 × 2 images the maps show, never on the topography panels; the traces go to the publication files with the maps (`contour.py`: marching squares on the height map, placed on the unit cell's lattice) |
| `Topo h`, `LDOS h` | columns beside the topography panels | the tip heights (Å, 0 to the cell top above the highest atom) with a `Rerun` button each: the height applies when Rerun is pressed while running |
| bottom strip | under the panels | `Cells` (0 to 4 periodic images each way), `Path Pts` (Path view, 2 to 12 nodes), `Markers` (1 to 10), `E Pts` (Map view, 1 to 20 energy maps), `E Min`, `E Max` (eV, -5 to 5), `LDOS γ` (the heatmaps' power-law colour scale, re-colours only) |

Every count and value is a number box: type a value and press Enter, or select a digit and press
Up or Down to step it (a Value stops at its limits; a step applies at once). Markers, line ends,
path nodes and the energy markers of the Map view are dragged with the mouse; a click on a curve
of the Partitioned LDOS panel descends a level (partition, atom types, atoms, orbitals) and a click
beside the curves ascends. At the orbital level the curves carry hover labels. The MO to LDOS
decomposition wrapper (`mo_ldos_decomp`) adds its σ box and its `Send to decomp` buttons to the
left column through `Interactive_STM_Simulator.add_control_box`.

## Package `stm_from_dft/`

| module | role |
|---|---|
| `simulator.py` | `Unified_STM_Simulator` (GPU LDOS, Simpson integration, Chen's tunneling matrix element `gpu_chen_tunneling_factor`, constant-current convergence, caches) and `Interactive_STM_Simulator` (the window: its controls made once, its panels per view, on the package's own `qtgui` copy, `stm_from_dft/qtgui/`). The decay model is described in the `Decay` row above |
| `feenstra.py` | the `Feenstra` toggle's arithmetic (numpy only): `current_from_ldos` (I(V), the LDOS integrated from 0 V), `iv_ratio` (I/V, plain or broadened: a port of the sm4_Browser's `feenstra.iv_broadened` without its current filters and 0Vsub, which have no counterpart for a computed current), `divide` |
| `doscar_parser.py` | `SpinAwareDosParser`, `SpinMode`: DOSCAR reader for all three spin modes |
| `locpot_manager.py` | `LocpotManager`: LOCPOT loading and `.npy` caching for all three spin modes |
| `plot_dos_cursors_levels.py` | `DosPlotter`: DOS window with levels (total, Au / molecule, atom types, atoms, one atom's orbitals with hover labels; a click on a curve descends, a click beside the curves ascends; `m` toggles the \|m\| overlay in SOC mode) (run block at the bottom, `v_dir`) |
| `wrong_decays/` | UNUSED, not imported, fully commented (removing the leading `# ` from every line restores each file): `tilted_barrier_decay.py`, `tilted_barrier_decay_github.py`, `tier3_decay.py` and `sim_decay_full.py`. Their decay factor is the WKB average of 2κ over a barrier rising linearly from φ - E to φ - E + V, (4√(2m)/(3ħV)) [(φ - E + V)^(3/2) - (φ - E)^(3/2)] (the trapezoidal-barrier WKB integral; e.g. the TU Wien dissertation at iue.tuwien.ac.at/phd/goes, App. A.3, Eq. A.28; R. Rejali et al., Phys. Rev. B 107, 035406 (2023), Eq. 5), named after Chen in that code but not the model of Chen's book. `tilted_barrier_decay.py` holds the simulator's decay code with that factor: every state at V = \|E\| (the edge of its own bias window) in the topography and the display alike, E taken as max(E, 0) for occupied states with its '0-clamp' toggle, and κ = 0.512 √φ Å⁻¹ with Decay off and in the tip-height convergence step. `tilted_barrier_decay_github.py` holds GitHub's earlier version of the same model (float32 arrays, the `1e-6` V floor, `max(0.1, phi)`, no Feenstra). `tier3_decay.py` holds a frozen-bias topography and signed I(V)-column display built on the factor, `sim_decay_full.py` a full build of the simulator with it on the matplotlib GUI |

Inputs: DOSCAR, LOCPOT, POSCAR or CONTCAR (and INCAR for the spin-mode check). Outputs: figures and
the `global_topo_*` / `ldos_topo_*` `.npy` caches in the run directory.

Dependencies: numpy, scipy, cupy (CUDA GPU), pymatgen, PySide6 and pyqtgraph (the package's own
`qtgui` copy, `stm_from_dft/qtgui/`), matplotlib (the publication files written by the toolbar's
Save button).

The window sizes its panels by rule 11 of `matplotlib_to_real_gui.md`, but is never trimmed to
them: the topographies and the heatmaps are sized first and fill the window (a topography never
more than twice a heatmap; the Line view's Topo panel is 2:3, width to height, L/3 either side of
the line by its length L; the Line and Path heatmaps are sized as squares, then fill their row's
whole slot, stretched in x or y, never taking room from another panel); the curve panels (the Tip
Height profile and the Partitioned LDOS) take the room those leave, never less
than a readable size; in Map the spectrum is one cell of the map block, a map's size, in the free
slot of the last row; and in Single Point the spectrum stands beside the placement topo as its
equal. The panels drawn in the unit cell's frame (the Global Topo, the Map
Topo and the energy maps; the placement topo of the other views) are shown as drawn or turned 90
degrees anticlockwise (x up, y to the left: a rotation, never a mirror), whichever fits the window
better; that is decided when a view is built and again on Crop / Uncrop, never on a resize, and the
console says which way and why. The heatmaps of energy against position never turn. Everything on a
turned panel keeps its data coordinates: the cursors, the markers, the line and path endpoints, the
limits and the publication files (which show the turn as on screen).
