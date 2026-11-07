# stm_from_dft_opt

GPU-accelerated STM / LDOS simulator from DFT output (package `stm_from_dft/`): the site- and
orbital-projected DOS weighted by the tunneling matrix element of the exact tilted barrier (to first
order in (E - V/2)/φ and V/φ, Chen's energy-dependent element), with the tip DOS constant as in
Tersoff and Hamann and the barrier taken from the local potential at each tip point (the `Decay` row
below). Auto-detects ISPIN=1, ISPIN=2 and noncollinear / SOC calculations from
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
rows that make them largest), and the curve panels (Tip Height and Partitioned LDOS with its legend
panel at its right, side by side in Line, side by side or stacked by their room in Path) are placed by the arrangement that
scores best, the smallest group's size against its size alone less the blank fraction, chosen
again only when a view's panels are built or their number changes (the drill-down level, the
per-heatmap colour bars), when a crop changes a
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
EPS or SVG through matplotlib (publication files), a PNG or TIFF (the window at 720 dpi) or an SVG
as on screen, or the curves as CSV.

| control | where | what it does |
|---|---|---|
| `mode` radio | left column, top | the view: `Single Point`, `Line`, `Path`, `Map` (the panels of the view are rebuilt; the controls stay) |
| `cmap set` radio | left column, under `mode` | the colour-map set of every panel (`cmap_shading.py`, its row below; the constructor's `cmap_set` is the one the window opens with): `default 1`, the topographies, the Line / Path stripe and the topo colour bar black to firebrick to yellow, the Line and Path heatmaps `nipy_spectral`, the energy maps `MAP_CMAP`, and under `Mag` the signed spin density `bwr` and \|m\| `hot`, in the maps and the heatmaps; `default 2`, `CET-L3`, `CET-R1`, `CET-L6`, `CET-D1A` and `CET-L19` in the same places. A change recolours every panel in place, running or not (nothing recomputed or arranged again); a shaded energy map keeps its shading map, the same in both sets |
| `broadening` box | left column | one check per atom type with its `σ <type>` box (eV, 0 to 1, two decimals), shown while the check is on: extra Gaussian broadening of that type's DOS |
| `RUN` / `STOP` | left column, bottom | starts and stops the computation; on STOP the line endpoints / path nodes and marker positions are printed in the form `run_interactive` takes |
| toggles | beside RUN | `Atoms`, `Decay` (`exact` and `linearized` indented under it), `Norm`, `Feenstra`, `Mag`, `Cell`, `Extend` |
| `Decay` | toggles | **Physics:** the tunneling matrix element that weights the DOS of every atom image at a tip point, with the tip DOS constant (the s-wave tip of J. Tersoff and D. R. Hamann, Phys. Rev. B 31, 805 (1985)). As in a measurement (C. J. Chen, *Introduction to Scanning Tunneling Microscopy*, 3rd ed., Oxford 2021, Sec. 2.2.3-2.2.4), the topography is converged on the current at one frozen bias (the current of Eq. 2.40 with the weight below: the window [0, V_b], every state weighted at that frozen bias, w(E, V_b)), and the displayed LDOS is dI/dV with the bias swept (that current's derivative at each V: the state entering at the window edge plus the change of every state already in [0, V]; that integral starts at 0 V whether or not the window reaches it). A state at E (eV from E_F) of an atom image at distance d from the tip at bias V carries the weight w(E, V), φ the LOCPOT potential at the tip point minus E_F (Chen's φ̄, one average work function for the junction, taken here at each tip point). On (the default): one of two models, the rows `exact` and `linearized` under it (their row below). `exact` (the default): the exact tilted barrier, w = exp(-S), S = (4 k d / (3 V)) [a₊^(3/2) - b₊^(3/2)], a = φ - E + V, b = φ - E, x₊ = max(x, 0): the WKB exponent 2∫κ dz across a barrier rising linearly from φ - E at the sample to φ - E + V at the tip, κ = 0 wherever the state lies above the barrier (a state above one end sees only the part of the barrier above it; above both ends, w = 1). `linearized`: its first order in (E - V/2)/φ and V/φ, the energy-dependent element of Chen's Eq. 2.40, w = exp(-2 κ0 d + κ0 (E - V/2) d / φ), dw/dV = -(κ0 d / (2 φ)) w (Eqs. 2.34, 2.37-2.39: κ to first order in the state's energy E - V/2 from the middle of the bias window). Off: exp(-2 κ0 d), the same weight without its energy and bias dependence. κ0 = k √φ = √(2 m φ)/ħ, k = 0.512317 Å⁻¹ eV^-1/2 from m_e, h and e, in every use (the tip-height convergence step included). **Machinery:** V_b is the bias end's snapped DOS sample (the bottom strip row): the topography's window is [0, V_b] from exactly 0 V, with the weight at V_b. All integrals use the rule of `quadrature.py` (its row below). The topography caches carry `_decay_exact` (on, `exact`), `_decay_linearized` (on, `linearized`) or `_decay_k0` (off), followed by `_fold` and `_f32` |
| `exact`, `linearized` | toggles, indented under `Decay` | the Decay-on model (the physics of both in the `Decay` row): exactly one is on, `exact` at start (`run_interactive`'s `decay_linearized=False`; `True` starts on `linearized`); a click on one turns it on and the other off, a click on the one already on leaves it on. While Decay is off both are ignored (a click then changes no model and recomputes nothing); while it is on, a change of model recomputes what a `Decay` change recomputes. `show_decay_toggle=False` hides both with `Decay`. The information panel lists both rows. **Machinery:** both models use the same steps, the rule of `quadrature.py` and float32: the topography from the frozen-bias weight w(E, V_b), integrated from 0 V; the swept dI/dV and the current (`Feenstra`) in one pass of a CUDA kernel per launch of tip points, with the console line `x/n spectra calculated`. `exact`'s weight does not separate in E and V, so its kernel (`exact_sweep`) integrates each requested V on its own; `linearized`'s weight is one factor per image in V, w(E, V) = w(E, 0) w(0, V) / w(0, 0), so its kernel (`linearized_sweep`) gives dI/dV and the current at every requested V in one walk outward from 0 V on each side. Each model has its own topography caches (the `Decay` row) |
| `Feenstra` | toggles | **Physics:** Feenstra's normalized conductance (dI/dV)/(I/V) on top of the calculated LDOS (`feenstra.py`): at the same tip positions as the display, I(V) from 0 V with the σ broadening in force (I(0) = 0 exactly; a window that does not reach 0 V is extended to it for the integral): with Decay on, the current of the Decay-on model (`exact` or `linearized`) at each V, the integral from 0 to V of ρ(E) w(E, V), as the topography's current at its bias; with Decay off, the total LDOS integrated. I/V is the plain I(V)/V, and the spectra, heatmaps and maps are divided by it along their energy axis; every partition, atom and orbital of a point is divided by the same total I/V (with `Mag` on, the magnetization over the total I/V). **Machinery:** I(V) is integrated by the rule of `quadrature.py` (its row below); with Decay on it comes from the same pass as the displayed dI/dV. The current, I/V and the division are computed on the GPU. Display only: no LDOS recompute; `Norm`, when on, acts on the ratio. A point with no states between 0 V and V (0/0) shows 0. The spectrum's y axis and the label over the heatmaps and maps read (dI/dV)/(I/V) |
| `cmap norm` | bottom strip, Map only | off: every energy map on one colour scale, the largest value over all of them, with one colour bar; on: each map its own scale with its own bar beside it (as the sm4_Browser Map window's `cmap norm`). The maps' colour map is the Map map of the window's colour-map set (the constructor's `cmap_set`; the `cmap set` choice in the left column): `MAP_CMAP` in `default 1`, the sm4_Browser's (`custom_map`, midnight blue to light sky blue), `CET-L6` in `default 2`; while the maps are shaded (`Shading` on), the isoluminant `CET-I1` in both sets (under `Mag`, `CET-D11` for the signed spin density and `CET-I3` for \|m\|) |
| contour slice | the topo colour bar beside the Map Topo | the sm4_Browser Map window's, drawn while `Contour` is on (the Contour / Shading box): a cyan level line on the bar (half the range at open) is dragged by a press on it, following the cursor's height while the button is down even off the narrow bar sideways; the Map Topo's contour at that height, the perimeters of everything above it (WSxM's flooding, as outlines), is drawn on every energy map as white 1 pt traces, at the 2 × 2 images the maps show, never on the topography panels; the traces go to the publication files with the maps (`contour.py`: marching squares on the height map, placed on the unit cell's lattice) |
| `Contour`, `Shading`, `shading by` | bottom strip, Map only, after `cmap norm` (an untitled box) | the Contour / Shading box (`cmap_shading.py`): at most one of the two toggles on, both may be off (turning one on turns the other off); the window opens with `Contour` on, `Shading` off and `Height`. `Contour` on: the contour slice (its row above); off: no traces on the energy maps and the level line hidden. `Shading` on: each energy map drawn as a mesh of one colour per grid point of the cell (the cell's repeated edge left out, each point's quad centred on it, at the 2 × 2 images the maps show), the isoluminant shading map (`CET-I1`; under `Mag`, `CET-D11` for the signed spin density, `CET-I3` for \|m\|) through the map's own colour scale, then shaded through CIELAB L*: each colour's lightness scaled evenly from 0 (black) at the lowest shade over the cell up to its own at the highest. `shading by`: `Height`, the shade is the LDOS topo's height (low dark, high light); `Relief`, the LDOS topo's slopes lit from the upper right as the map is shown (Kovesi's `relief.m` with its defaults: azimuth 45°, elevation 45°, Lambertian; the gradients wrap around the cell). The colour bars show the shading map unshaded; `Shading` off, the set's maps and the filled contours come back. Display only: no LDOS recomputed, nothing arranged again, only the energy maps redrawn |
| `Topo h`, `LDOS h` | columns beside the topography panels | the tip heights (Å, 0 to the cell top above the highest atom) with a `Rerun` button each: the height applies when Rerun is pressed while running |
| bottom strip | under the panels | `Cells` (0 to 4 periodic images each way), `Path Pts` (Path view, 2 to 12 nodes), `Markers` (1 to 10), `E Pts` (Map view, 1 to 20 energy maps), `E Min`, `E Max` (eV, -5 to 5), `LDOS γ` (the heatmaps' power-law colour scale, re-colours only). E Min and E Max snap to the nearest DOS sample (a tie: E Min to the higher sample, E Max to the lower); an end of exactly 0 is exactly 0 V, not a sample: it makes the other end the bias; each Map energy snaps to the nearest displayed sample, a tie going to the sample closer to 0 |

Every count and value is a number box: type a value and press Enter, or select a digit and press
Up or Down to step it (a Value stops at its limits; a step applies at once). Markers, line ends,
path nodes and the energy markers of the Map view are dragged with the mouse; a left click on a
curve of the Partitioned LDOS panel, or on its entry in the legend panel at its right, descends a
level (partition, atom types, orbitals) and a right click in the panel or in its legend panel, or
the legend panel's Back button, ascends. At the orbital level the curves carry hover labels. The
MO to LDOS decomposition wrapper (`mo_ldos_decomp`) adds its σ box and its `Send to decomp`
buttons to the left column through `Interactive_STM_Simulator.add_control_box`.

## Package `stm_from_dft/`

| module | role |
|---|---|
| `simulator.py` | **Physics** (the block THE DECAY MODEL at the top of the file): the tunneling matrix elements of the two Decay-on models, the exact tilted barrier `gpu_exact_factor` (`exact`) and its first order `gpu_linearized_factor` (`linearized`), and the Decay-off weight `gpu_decay_off_factor`. The decay model is described in the `Decay` row above. **Machinery** (NUMERICS AND GPU, the rest of the file): `Unified_STM_Simulator` (GPU LDOS, the integration rule of `quadrature.py`, constant-current convergence, caches; the GPU computes in float32 from the float64 DOS grid: one step, each sample's energy formed from its index, so the samples are exactly equally spaced) and `Interactive_STM_Simulator` (the window: its controls made once, its panels per view, on the package's own `qtgui` copy, `stm_from_dft/qtgui/`) |
| `ldos_decomp.py` | the decomposition of the Partitioned LDOS panel, everything in it that depends on its level (display machinery, no physics): its state (the level, the active marker, the active element), the partitions the maps and heatmaps show at each level, the spectrum's curves and the legend panel's title and entries, the click rules, the hover labels of the orbital curves and its line in the information panel |
| `cmap_shading.py` | the colour maps of the panels, the shading of the Map view's energy maps and the Contour / Shading box, one self-contained module (display machinery, no physics; it imports only the standard library, NumPy and CuPy, nothing from the simulator or `qtgui`, so it can be copied into sm4_Browser as it is): the two colour-map sets by role (`SETS`, `cmap_spec`, `make_cmap`: the `cmap set` choice) and the energy maps' isoluminant shading maps (`SHADING_CMAPS`), the box's state and rules (`BoxState`) and its builder (`build_box`, given the window's figure and the `qtgui` controls module), the shading maths on NumPy or CuPy arrays (the simulator passes CuPy: on the GPU): the Height and Relief shades (`relief.m`'s Lambertian shading), the colours through the maps' own colour scale and the CIELAB L* stretch, and the corners of the mesh of one quad per grid point |
| `feenstra.py` | the `Feenstra` toggle's arithmetic (CuPy): `current_from_ldos` (I(V), the LDOS integrated from 0 V), `iv_ratio` (the plain I/V; a sample exactly at 0 V given its neighbours' I/V), `divide` |
| `quadrature.py` | the integration rule (CuPy, float32): `simpson` (SciPy 1.18.1's composite Simpson, written out for the GPU) and `integral_from_zero` (from 0 V: the cubic through the two samples on each side of 0 V, E₋₂, E₋₁, E₁, E₂ (through E₋₁, 0, E₁, E₂ above and E₋₂, E₋₁, 0, E₁ below when a sample is at 0 V), its value at 0 V used by the unequal pair next to 0 V when the interval count is even, its exact integral over the interval next to 0 V when the count is odd, Simpson pairs over the rest) |
| `doscar_parser.py` | `SpinAwareDosParser`, `SpinMode`: DOSCAR reader for all three spin modes; the energies are rebuilt from line 6 as an even grid, E min + i (E max - E min)/(NEDOS - 1), minus E_F, the printed column only checked against it, within half a unit of its last printed decimal plus 1e-6 eV. `read_doscar_header` reads only the header and gives the decomposition package its E_F |
| `locpot_manager.py` | `LocpotManager`: LOCPOT loading and `.npy` caching for all three spin modes |
| `plot_dos_cursors_levels.py` | `DosPlotter`: DOS window with levels (total, Au / molecule, atom types, atoms, one atom's orbitals with hover labels; a click on a curve descends, a click beside the curves ascends; `m` toggles the \|m\| overlay in SOC mode) (run block at the bottom, `v_dir`) |
| `wrong_decays/` | UNUSED, not imported, fully commented (removing the leading `# ` from every line restores each file): `tilted_barrier_decay.py`, `tilted_barrier_decay_github.py`, `tier3_decay.py` and `sim_decay_full.py`. Their decay factor is the WKB average of 2κ over a barrier rising linearly from φ - E to φ - E + V, (4√(2m)/(3ħV)) [(φ - E + V)^(3/2) - (φ - E)^(3/2)] (the trapezoidal-barrier WKB integral; e.g. the TU Wien dissertation at iue.tuwien.ac.at/phd/goes, App. A.3, Eq. A.28; R. Rejali et al., Phys. Rev. B 107, 035406 (2023), Eq. 5), named after Chen in that code but not the model of Chen's book. `tilted_barrier_decay.py` holds the simulator's decay code with that factor: every state at V = \|E\| (the edge of its own bias window) in the topography and the display alike, E taken as max(E, 0) for occupied states with its '0-clamp' toggle, and κ = 0.512 √φ Å⁻¹ with Decay off and in the tip-height convergence step. `tilted_barrier_decay_github.py` holds GitHub's earlier version of the same model (float32 arrays, the `1e-6` V floor, `max(0.1, phi)`, no Feenstra). `tier3_decay.py` holds a frozen-bias topography and signed I(V)-column display built on the factor, `sim_decay_full.py` a full build of the simulator with it on the matplotlib GUI |

Inputs: DOSCAR, LOCPOT, POSCAR or CONTCAR (and INCAR for the spin-mode check). Outputs: figures and
the `global_topo_*` / `ldos_topo_*` `.npy` caches in the run directory, their names carrying the
snapped bias (the energy of its DOS sample), `_fold` and `_f32`.

Dependencies: numpy, scipy, cupy (CUDA GPU), pymatgen, PySide6 and pyqtgraph (the package's own
`qtgui` copy, `stm_from_dft/qtgui/`), matplotlib (the publication files written by the toolbar's
Save button).

The window sizes its panels by rule 11 of `matplotlib_to_real_gui.md`, but is never trimmed to
them: the topographies and the heatmaps are sized first and fill the window (a topography never
more than twice a heatmap; the Line view's Topo panel is 2:3, width to height, L/3 either side of
the line by its length L; the Line and Path heatmaps are sized as squares, then fill their row's
whole slot, stretched in x or y, never taking room from another panel); the curve panels (the Tip
Height profile and the Partitioned LDOS, its legend panel of one fixed width at its right) take the
room those leave, never less than a readable size; in Map the spectrum is one cell of the map block,
a map's size, in the free slot of the last row, its legend panel at its right in that row, counted
in the block's width; and in Single Point the spectrum, its legend panel at its right, stands beside
the placement topo as its equal. The panels drawn in the unit cell's frame (the Global Topo, the Map
Topo and the energy maps; the placement topo of the other views) are shown as drawn or turned 90
degrees anticlockwise (x up, y to the left: a rotation, never a mirror), whichever fits the window
better; that is decided when a view is built and again on Crop / Uncrop, never on a resize, and the
console says which way and why. The heatmaps of energy against position never turn. Everything on a
turned panel keeps its data coordinates: the cursors, the markers, the line and path endpoints, the
limits and the publication files (which show the turn as on screen).
