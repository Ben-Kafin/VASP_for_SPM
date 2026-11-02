# stm_from_dft_opt

GPU-accelerated STM / LDOS simulator from DFT output (package `stm_from_dft/`), Tersoff-Hamann
style with the Chen tunneling barrier from the local potential. Auto-detects ISPIN=1, ISPIN=2 and
noncollinear / SOC calculations from DOSCAR and LOCPOT.

## Run

`run_DFT_STM.py`: set `v_dir` to the VASP directory containing DOSCAR, LOCPOT and POSCAR/CONTCAR,
then adjust the `Interactive_STM_Simulator(...)` arguments (energy range, LDOS sampling height,
colormap, number of unit cells) and the `run_interactive(...)` arguments (topography grid, bias,
tip height, decay model on/off, cache reuse, line-scan endpoints, marker positions, extra
broadening per element, path mode seed). The window gives constant-current topography, dI/dV at
markers, line scans, path scans and energy maps.

## The window

The window is a Qt window (PySide6 + pyqtgraph) drawn with the shared `qtgui` layer at the
repository root: its panels and controls sit in one automatic layout (nothing is placed by hand,
nothing overlaps at any window size, the window never scrolls), every text follows the window's
size, and a panel's title shows its short name with the details (bias, height, the marker) in the
information panel docked at the bottom. The panels are arranged by one rule (the guide's rule 11):
every plotted panel is a first-class citizen, the plots fill the window, spatial panels keep their
ratio, nothing plotted is shrunk to make room, and blank space is what is minimized. The shaped
groups of a view, the placement topo (the Global Topo, where the markers, the line ends and the
path nodes are dragged, with its boxes above), the heatmap row of Line and Path (its label over
it), the Global Topo and Map Topo pair and the block of energy maps of Map (the maps wrap into the
rows that make them largest), and the curve panels (Tip Height and Partitioned LDOS, side by side
or stacked by their room) are placed by the arrangement that scores best, the smallest group's size
against its size alone less the blank fraction, chosen again on every view change, rebuild, crop
and resize; the window trims the blank strips at its edges. The Topo h box, Crop and Uncrop sit
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
| toggles | beside RUN | `Atoms`, `Decay`, `0-clamp` (under Decay), `Norm`, `Mag`, `Cell`, `Extend` |
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
| `simulator.py` | `Unified_STM_Simulator` (GPU LDOS, Simpson integration, Chen barrier, constant-current convergence, caches) and `Interactive_STM_Simulator` (the window: its controls made once, its panels per view, on the shared `qtgui` layer). Decay model: Tier 2 with the optional occupied-side 0-clamp |
| `doscar_parser.py` | `SpinAwareDosParser`, `SpinMode`: DOSCAR reader for all three spin modes |
| `locpot_manager.py` | `LocpotManager`: LOCPOT loading and `.npy` caching for all three spin modes |
| `plot_dos_cursors_levels.py` | `DosPlotter`: DOS window with levels (total, Au / molecule, atom types, atoms, one atom's orbitals with hover labels; a click on a curve descends, a click beside the curves ascends; `m` toggles the \|m\| overlay in SOC mode) (run block at the bottom, `v_dir`) |
| `tier3_decay.py` | UNUSED, fully commented: the Tier 3 decay model (the frozen-bias topography and the signed I(V)-column display) as an extracted reference, with the note on how the live build carried it (the E_F segment, the I(V) column beside dI/dV); kept for future work on the tunneling barrier |
| `sim_decay_full.py` | UNUSED, fully commented: the full build of the simulator with the 'Tier 3' box and the 'Feenstra' box (the display divided by I/V of the same point and energy, Tier 3 only), on the matplotlib GUI the simulator had before its conversion; kept as reference for the same work. The live simulator runs Tier 2 (with the optional 0-clamp) only |

The two UNUSED files are not imported by `__init__.py`; removing the leading `# ` from every line
restores them exactly.

Inputs: DOSCAR, LOCPOT, POSCAR or CONTCAR (and INCAR for the spin-mode check). Outputs: figures and
the `global_topo_*` / `ldos_topo_*` `.npy` caches in the run directory.

Dependencies: numpy, scipy, cupy (CUDA GPU), pymatgen, PySide6 and pyqtgraph (the shared `qtgui`
layer at the repository root), matplotlib (the publication files written by the toolbar's Save
button). `checks/check_simulator.py` at the repository root opens the window offscreen on synthetic
inputs (numpy stand-ins for cupy) and measures its layout at several sizes.
