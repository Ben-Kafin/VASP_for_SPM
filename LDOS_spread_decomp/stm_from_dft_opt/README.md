# stm_from_dft_opt

GPU-accelerated STM / LDOS simulator from DFT output (package `stm_from_dft/`): the site- and
orbital-projected DOS weighted by the tunneling matrix element of the exact tilted barrier or, chosen
with the `exact` / `linearized` rows under `Decay`, of its first order in (E - V/2)/φ and V/φ, Chen's
energy-dependent element, with the tip DOS constant as in
Tersoff and Hamann and the barrier taken from the local potential at each tip point (the `Decay` row
below). Auto-detects ISPIN=1, ISPIN=2 and noncollinear / SOC calculations from
DOSCAR and LOCPOT.

## Run

`run_DFT_STM.py`: set `v_dir` to the VASP directory containing DOSCAR, LOCPOT and POSCAR/CONTCAR,
then adjust the `Interactive_STM_Simulator(...)` arguments (energy range, the LDOS tip height,
colormap, number of unit cells) and the `run_interactive(...)` arguments (topography grid, the topo
bias and the topo tip height, the `bias` toggles' starting end (`ldos_bias_sign`), decay on/off and
its model (`decay_linearized`: `exact` or `linearized`), cache reuse (the caches in the folder
`caches` inside the VASP folder), line-scan endpoints, marker positions, extra
broadening per element, path mode seed). The window gives constant-current topography, dI/dV at
markers, line scans, path scans and energy maps.

The two topographies are each converged at constant current on one unit cell's grid and tiled for
display: the global topo, at the topo bias (`topo_bias`) and the topo tip height (`Topo h`), and
the LDOS topo, at the LDOS bias V_b (the end the `bias` toggles pick) and the LDOS tip height
(`LDOS h`), the only one at the LDOS tip height. The LDOS topo is shared by every view: taken from
memory, from its `ldos_topo_*` cache, or converged on the whole cell and saved. Single Point's and
Map's marker spectra take their tip heights from it (at the grid point nearest each marker), and
Map's energy maps are computed at its heights. The Line and Path stripes (the tip heights along
the line or path, not a topo) have their own quick line convergence: they start from the LDOS topo
interpolated at the sample points (bilinear, periodic in the cell's fractional coordinates, the
grid's duplicate edge row and column left out of the interpolation), and their setpoint is the LDOS
topo's own, the current at its tallest grid point (ties to the largest current), so they sit on the
LDOS topo's tip height even where the line or path does not pass there. A stripe has no cache of its
own: its heights are saved in its view's LDOS cache, with the LDOS computed at them (the Outputs
paragraph below).

## The window

The window is a Qt window (PySide6 + pyqtgraph) drawn with the package's own copy of the `qtgui`
layer, `stm_from_dft/qtgui/` (imported from the package, no folder walking): its panels and
controls sit in one automatic layout (nothing is placed by hand, nothing overlaps at any window
size, the window never scrolls), every text follows the window's size and is never below 10 pt,
and a panel's title shows its short name (the two topos theirs with their bias and tip height:
`Global Topo: bias … V, height … Å`, `LDOS Topo: bias … V, height … Å`; the stripe `Topo`) with the
other details (the marker) in the information panel docked at the bottom. The panels are arranged
by one rule (the guide's rule 11):
every plotted panel is a first-class citizen, the plots fill the window, spatial panels keep their
ratio, nothing plotted is shrunk to make room, and blank space is what is minimized. A view's panels
are grouped into sub-sections, which keep the same layout relative to each other whatever the
settings (Single Point's tall-window rule the one exception), each view in the one layout its rules
give, and every row of every view has one plot-box height, so the axes line up (titles and axis
labels sit outside the boxes). Every view's first sub-section is the topo sub-section, one row,
from the left: the topo colour bar (its own panel, always far left, clear of every text and tick:
the topo colour map's ramp, no ticks, no numbers, `topo` above it, carrying the contour slice's
level line only in Map), the global topo (the Topo h box, Crop and Uncrop above it) and the
LDOS topo (the LDOS h box above it, in every view). Single Point: the topo sub-section and
Partitioned LDOS with its legend panel at its right in one row, or the spectrum under the topo
sub-section in a tall window (the sizes below). Line and Path: the top row, the topo sub-section,
then the stripe (titled `Topo`), one heatmap per partition and their colour bars, under their
label, so that the LDOS topo's, the stripe's and the heatmaps' distance axes line up, over the
second row, Tip Height and Partitioned LDOS with its legend panel at its right, side by side. Map:
the top row, the topo sub-section, then Partitioned LDOS with its legend panel at its right, over
the block of energy maps (the maps wrap into the rows and columns that make them largest, chosen
again at every resize) and, under the maps however many rows they wrap to, Map's last row, the
shading row: from the left the Contour / Shading box, the `darkness` box and the light's top view
and side view (their rows below), its bottom edge on the bottom edge of the left column's RUN /
toggles box, the height the window leaves over going between the maps and it, never below it. The
placement panel, where the markers, the line ends and the path nodes are placed, is the global topo
in Single Point, Line and Path and the LDOS topo in Map; the markers are drawn on the LDOS topo too.
The arrangement is chosen
again only when a view's panels are built or their number changes (the drill-down level, the
per-heatmap colour bars), when a crop changes a
panel's shape and when the window is resized (maximizing included); the window is never trimmed to
its panels, so a maximized window stays full screen. The line ends, the path nodes and the Single
Point and Map markers recompute on release: the tip heights, the LDOS and the spectra a drag
changes are computed once, when the button is let go. The global topo can be cropped as the
sm4_Browser Line window crops its heatmap: two draggable cursors per axis start at the full extent
of the drawn cells, the area outside each pair is shaded, a box riding with each cursor above (x)
and right of (y) the panel shows its value and takes a typed one, Crop applies the ranges between
the cursors as the panel's limits (in Single Point and Map the LDOS topo takes the same) and Uncrop
restores the extent; a crop changes the panel's shape, the unit-cell panels' orientation is decided
again and the arrangement follows.
The window opens on the screen's available area. The toolbar has Save and, at its right end, the
cursor readout (Home, Pan and Zoom are not in the simulator's toolbar; Save and the readout keep
their places): while the mouse is in a panel, `x=…  y=…`, the mouse's place in the coordinates that
panel's axes are labelled in (on a turned panel too; on Line's and Path's LDOS topo their own
`Perp.` and `Position`), with `value=…` after them on a panel that shows data as colour, the value
that corresponds to the colour under the mouse: the height on the global topo and the LDOS topo
(Å), the tip height of the stripe's segment, the value of the heatmap's cell, the energy map's
value, contoured or shaded (on a filled-contour panel the data interpolated at the mouse's point,
not a band's edge; on a tiled panel the periodic data's value); the curve panels and the light
panels give x and y only; it is blank on no panel and over a colour bar. Save writes PDF,
EPS or SVG through matplotlib (publication files), a PNG or TIFF (the window at 720 dpi) or an SVG
as on screen, or the curves as CSV.

| control | where | what it does |
|---|---|---|
| `mode` radio | left column, top | the view: `Single Point`, `Line`, `Path`, `Map` (the panels of the view are rebuilt; the controls stay) |
| `cmap set` radio | left column, under `mode` | the colour-map set of every panel (`cmap_shading.py`, its row below; the constructor's `cmap_set` is the one the window opens with): `default 1`, the topographies, the Line / Path stripe and the topo colour bar black to firebrick to yellow, the Line and Path heatmaps `nipy_spectral`, the energy maps `MAP_CMAP`, and under `Mag` the signed spin density `bwr` and \|m\| `hot`, in the maps and the heatmaps; `default 2`, `CET-L3`, `CET-R1`, `CET-L6`, `CET-D1A` and `CET-L19` in the same places. A change recolours every panel in place, running or not (nothing recomputed or arranged again); a shaded energy map keeps its shading map, the same in both sets |
| `broadening` box | left column | one check per atom type with its `σ <type>` box (eV, 0 to 1, two decimals), shown while the check is on: extra Gaussian broadening of that type's DOS |
| `RUN` / `STOP` | left column, bottom | starts and stops the computation; on STOP the line endpoints / path nodes and marker positions are printed in the form `run_interactive` takes |
| toggles | beside RUN | `Atoms`, `Decay` (`exact` and `linearized` indented under it), `Norm`, `Feenstra`, `Mag`, `Cell`, `Extend` |
| `Decay` | toggles | **Physics:** the tunneling matrix element that weights the DOS of every atom image at a tip point, with the tip DOS constant (the s-wave tip of J. Tersoff and D. R. Hamann, Phys. Rev. B 31, 805 (1985)). As in a measurement (C. J. Chen, *Introduction to Scanning Tunneling Microscopy*, 3rd ed., Oxford 2021, Sec. 2.2.3-2.2.4), the topography is converged on the current at one frozen bias (the current of Eq. 2.40 with the weight below: the window [0, V_b], every state weighted at that frozen bias, w(E, V_b)), and the displayed LDOS is dI/dV with the bias swept (that current's derivative at each V: the state entering at the window edge plus the change of every state already in [0, V]; that integral starts at 0 V whether or not the window reaches it). A state at E (eV from E_F) of an atom image at distance d from the tip at bias V carries the weight w(E, V), φ the LOCPOT potential at the tip point minus E_F (Chen's φ̄, one average work function for the junction, taken here at each tip point). On (the default): one of two models, the rows `exact` and `linearized` under it (their row below). `exact` (the default): the exact tilted barrier, w = exp(-S), S = (4 k d / (3 V)) [a₊^(3/2) - b₊^(3/2)], a = φ - E + V, b = φ - E, x₊ = max(x, 0): the WKB exponent 2∫κ dz across a barrier rising linearly from φ - E at the sample to φ - E + V at the tip, κ = 0 wherever the state lies above the barrier (a state above one end sees only the part of the barrier above it; above both ends, w = 1). `linearized`: its first order in (E - V/2)/φ and V/φ, the energy-dependent element of Chen's Eq. 2.40, w = exp(-2 κ0 d + κ0 (E - V/2) d / φ), dw/dV = -(κ0 d / (2 φ)) w (Eqs. 2.34, 2.37-2.39: κ to first order in the state's energy E - V/2 from the middle of the bias window). Off: exp(-2 κ0 d), the same weight without its energy and bias dependence. κ0 = k √φ = √(2 m φ)/ħ, k = 0.512317 Å⁻¹ eV^-1/2 from m_e, h and e, in every use (the tip-height convergence step included). **Machinery:** V_b is the bias end's snapped DOS sample (the `bias` row): the topography's window is [0, V_b] from exactly 0 V, with the weight at V_b. All integrals use the rule of `quadrature.py` (its row below). The topography caches carry `_decay_exact` (on, `exact`), `_decay_linearized` (on, `linearized`) or `_decay_k0` (off), followed by `_fold` and `_f32` |
| `exact`, `linearized` | toggles, indented under `Decay` | the Decay-on model (the physics of both in the `Decay` row): exactly one is on, `exact` at start (`run_interactive`'s `decay_linearized=False`; `True` starts on `linearized`); a click on one turns it on and the other off, a click on the one already on leaves it on. While Decay is off both are ignored (a click then changes no model and recomputes nothing); while it is on, a change of model recomputes what a `Decay` change recomputes. `show_decay_toggle=False` hides both with `Decay`. The information panel lists both rows. **Machinery:** both models use the same steps, the rule of `quadrature.py` and float32: the topography from the frozen-bias weight w(E, V_b), integrated from 0 V; the swept dI/dV and the current (`Feenstra`) in one pass of a CUDA kernel per launch of tip points, with the console line `x/n spectra calculated`, one line per channel computed, rewritten in place after each launch, led by what is computed (`Line points`, `Map energy maps`, …) and the channel's label: `total: ` (none for an unpolarized run) or, for the mag channel while `Mag` is on, `|m|: ` (noncollinear / SOC) or `m: ` (collinear polarized). `exact`'s weight does not separate in E and V, so its kernel (`exact_sweep`) integrates each requested V on its own, one thread per (tip point, requested V, atom), each thread doing for its one V the same work over the atom's images, in the same order, into its own output slot; `linearized`'s weight is one factor per image in V, w(E, V) = w(E, 0) w(0, V) / w(0, 0), so its kernel (`linearized_sweep`) gives dI/dV and the current at every requested V in one walk outward from 0 V on each side, one thread per (tip point, atom). Each model has its own topography caches (the `Decay` row) and its own Line, Path and Map LDOS caches, so a change of model loads that model's files when they are there (the Outputs paragraph below) |
| `Feenstra` | toggles | **Physics:** Feenstra's normalized conductance (dI/dV)/(I/V) on top of the calculated LDOS (`feenstra.py`): at the same tip positions as the display, I(V) from 0 V with the σ broadening in force (I(0) = 0 exactly; a window that does not reach 0 V is extended to it for the integral): with Decay on, the current of the Decay-on model (`exact` or `linearized`) at each V, the integral from 0 to V of ρ(E) w(E, V), as the topography's current at its bias; with Decay off, the total LDOS integrated. I/V is the plain I(V)/V, and the spectra, heatmaps and maps are divided by it along their energy axis; every partition, atom and orbital of a point is divided by the same total I/V (with `Mag` on, the magnetization over the total I/V). **Machinery:** I(V) is integrated by the rule of `quadrature.py` (its row below); with Decay on it comes from the same pass as the displayed dI/dV, the total channel's (the mag channel's pass computes no current). The current, I/V and the division are computed on the GPU. Display only: no LDOS recompute; `Norm`, when on, acts on the ratio. A point with no states between 0 V and V (0/0) shows 0. The spectrum's y axis and the label over the heatmaps and maps read (dI/dV)/(I/V) |
| `Mag` | toggles | the magnetization channel shown in place of the total LDOS, in the spectra, the heatmaps and the energy maps, with its colour maps (the `cmap set` row): \|m\| for a noncollinear / SOC run, m = up - down, the signed spin density, for a collinear polarized run, the two treated the same; an unpolarized run has none, and the total stays shown. **Machinery:** the LDOS is computed as its total channel, in one pass (noncollinear / SOC the DOS's total, collinear polarized up + down, unpolarized its one channel); the current (`Feenstra`) comes from that pass. The mag channel is not computed until `Mag` is on, and then only for what is shown (the Line / Path LDOS, Map's maps and a dragged map, the marker spectra), in one pass of its own (collinear polarized on up - down); once computed it is kept and not computed again. With the Line / Path LDOS and Map's maps it is saved in a `_mag` file beside their LDOS cache when it is computed, and loaded instead of computed when that file is there (the Outputs paragraph below); turning `Mag` on while running computes it, or loads it, for what is shown and kept without it, the total not computed again. The marker spectra are not cached: their mag channel is computed with them, on a marker's release or creation, and when `Mag` is turned on |
| `cmap norm` | bottom strip, Map only | off: every energy map on one colour scale, the largest value over all of them, with one colour bar; on: each map its own scale with its own bar beside it (as the sm4_Browser Map window's `cmap norm`). The maps' colour map is the Map map of the window's colour-map set (the constructor's `cmap_set`; the `cmap set` choice in the left column): `MAP_CMAP` in `default 1`, the sm4_Browser's (`custom_map`, midnight blue to light sky blue), `CET-L6` in `default 2`; while the maps are shaded (`Shading` on), the isoluminant `CET-I1` in both sets (under `Mag`, `CET-D11` for the signed spin density and `CET-I3` for \|m\|) |
| contour slice | the topo colour bar, far left of Map's topo sub-section (Map only: the other views' bar has no level line) | the sm4_Browser Map window's, drawn while `Contour` is on (the Contour / Shading box): a cyan level line on the bar (half the range at open) is dragged by a press on it, following the cursor's height while the button is down even off the narrow bar sideways; the LDOS topo's contour at that height (as a fraction of the height range of Map's copy of the LDOS topo, the heights its maps were computed at), the perimeters of everything above it (WSxM's flooding, as outlines), is drawn on every energy map as white 1 pt traces, at the 2 × 2 images the maps show, never on the topography panels; the traces go to the publication files with the maps (`contour.py`: marching squares on the height map, placed on the unit cell's lattice) |
| `Contour`, `Shading`, `shading by` | Map's shading row, its first box (an untitled box) | the Contour / Shading box (`cmap_shading.py`): at most one of the two toggles on, both may be off (turning one on turns the other off); the window opens with `Contour` on, `Shading` off and `Height`. `Contour` on: the contour slice (its row above); off: no traces on the energy maps and the level line hidden. `Shading` on: each energy map drawn as a mesh of one colour per grid point of the cell (the cell's repeated edge left out, each point's quad centred on it, at the 2 × 2 images the maps show), the isoluminant shading map (`CET-I1`; under `Mag`, `CET-D11` for the signed spin density, `CET-I3` for \|m\|) through the map's own colour scale, then shaded through CIELAB L*: each colour's lightness scaled by a factor running evenly from 1 - darkness (the `darkness` row) at the lowest shade over the cell to 1 at the highest, in `Height` and `Relief` alike. `shading by`: `Height`, the shade is the LDOS topo's height (low dark, high light); `Relief`, the LDOS topo's slopes lit by the light the two light panels set (their row below; azimuth 45°, elevation 45° at open, the defaults of Kovesi's `relief.m`, whose Lambertian shading it is; the gradients wrap around the cell; the light keeps its azimuth on screen when the maps are turned). The colour bars show the shading map unshaded; `Shading` off, the set's maps and the filled contours come back. Display only: no LDOS recomputed, nothing arranged again, only the energy maps redrawn and the light panels drawn greyed (`Height`) or in colour (`Relief`) |
| `darkness` | Map's shading row, after the Contour / Shading box (its own untitled box) | the shading's darkness, 0 to 1 (two decimals), 0.5 at open: the floor of the L* stretch is 1 - darkness, in `Height` and `Relief` alike (1: the darkest point black; 0: no shading; 0.5: the darkest point at half its colour's L*). Display only, running or not: the shaded energy maps drawn again at it, nothing recomputed or arranged again; the information panel lists it with the other settings |
| top view, side view | Map's shading row, after the `darkness` box | the Relief light's two panels (`cmap_shading.py`, its row below), each a map panel's size, just the controls with no preview behind them: the top view, the plane from above with its centre point, a guide circle of radius R and the light as a dot at its azimuth (0 up, increasing clockwise), R cos(el) from the centre (the light as seen from above), a drag of the dot setting the azimuth only; the side view, the plane edge-on as a horizontal line through the centre point, seen at right angles to the azimuth so the light lies in the drawn plane, the light a dot on the arc of radius R at the elevation above the line with a line from it to the centre point, a drag up or down setting the elevation only (0 to 90°, the dot staying on the arc). While a dot is dragged the energy maps are shaded again live. In `Height`, which uses no light, both are greyed and take no drags. The information panel shows the light in its `azimuth` and `elevation` lines |
| `Topo h`, `LDOS h` | boxes above the topo panels, in every view: `Topo h` above the global topo (with Crop and Uncrop), `LDOS h` above the LDOS topo | the tip heights (Å, 0 to the cell top above the highest atom) with a `Rerun` button each: `Topo h` the global topo's, `LDOS h` the LDOS topo's (the one the spectra, the stripes and the energy maps take their tip heights from: the Run paragraph); the height applies when Rerun is pressed while running |
| bottom strip | under the panels | `Cells` (0 to 4 periodic images each way), `Path Pts` (Path view, 2 to 12 nodes), `Markers` (1 to 10), `E Pts` (Map view, 1 to 20 energy maps), `E Min`, `E Max` (eV, -5 to 5; each with its `bias` toggle beside it, in its frame), `LDOS γ` (the heatmaps' power-law colour scale, re-colours only). E Min and E Max snap to the nearest DOS sample (a tie: E Min to the higher sample, E Max to the lower); an end of exactly 0 is exactly 0 V, not a sample: the bias is then on the other end (the `bias` row); each Map energy snaps to the nearest displayed sample, a tie going to the sample closer to 0 |
| `bias` | bottom strip, beside `E Min` and beside `E Max`, in every view | which end of the window is the LDOS bias V_b (that end's snapped DOS sample, at which the LDOS topo and the stripes are converged, their topography window [0, V_b]): exactly one of the two toggles is on, and one always is, as `exact` / `linearized` (a click on the one on leaves it on); `run_interactive`'s `ldos_bias_sign` sets only their starting state (`'neg'` E Min, `'pos'` E Max). An end at 0.0, however it got there, has the bias on the other end: at start when the end `ldos_bias_sign` picks is 0.0, after an edit of the bias end to 0.0, and a click on the toggle of an end at 0.0 leaves it off; when that end is nonzero again, the bias stays on the other end until a toggle is clicked. Both ends at 0.0 stay an error. A switch recomputes what a change of that end's value recomputes, with the same caches. The information panel lists both, as `E Min bias` and `E Max bias` |

Every count and value is a number box: type a value and press Enter, or select a digit and press
Up or Down to step it (a Value stops at its limits; a step applies at once). Markers, line ends,
path nodes and the energy markers of the Map view are dragged with the mouse; a left click on a
curve of the Partitioned LDOS panel, or on its entry in the legend panel at its right, descends a
level (partition, atom types, orbitals) and a right click anywhere in the panel (on a curve, on
empty space, on its tick labels, axis labels or title) or in its legend panel, or the legend
panel's Back button, ascends; these clicks always act (the toolbar has no Pan or Zoom). The mouse
wheel scrolls the legend panel's entries from anywhere over the panel (its entries, its title, its
Back row, its scroll bar's strip). At the orbital level the curves carry hover labels. The
information panel lists the settings of the controls (the `bias` toggles and the `darkness` box
among them) and has one line of its own for each setting the mouse makes: `decomp level` (its
number and name), `active marker`, `active element`, one `marker k` line per marker (its position;
in Line and Path its distance along the line), `line` (the Line's ends or the Path's nodes with its
length), `crop`, and in Map `map energies`, `contour level`, `azimuth` and `elevation`. The
MO to LDOS decomposition wrapper (`mo_ldos_decomp`) adds its σ box and its `Send to decomp`
buttons to the left column through `Interactive_STM_Simulator.add_control_box`.

## Package `stm_from_dft/`

| module | role |
|---|---|
| `simulator.py` | **Physics** (the block THE DECAY MODEL at the top of the file): the tunneling matrix elements of the two Decay-on models, the exact tilted barrier `gpu_exact_factor` (`exact`) and its first order `gpu_linearized_factor` (`linearized`), and the Decay-off weight `gpu_decay_off_factor`. The decay model is described in the `Decay` row above. **Machinery** (NUMERICS AND GPU, the rest of the file): `Unified_STM_Simulator` (GPU LDOS, its total channel by default and its mag channel on request, the integration rule of `quadrature.py`, constant-current convergence, caches, every one in the VASP folder's `caches/` (`_cache_path`: the Outputs paragraph below); the GPU computes in float32 from the float64 DOS grid: one step, each sample's energy formed from its index, so the samples are exactly equally spaced) and `Interactive_STM_Simulator` (the window: its controls made once, its panels per view, on the package's own `qtgui` copy, `stm_from_dft/qtgui/`) |
| `ldos_decomp.py` | the decomposition of the Partitioned LDOS panel, everything in it that depends on its level (display machinery, no physics): its state (the level, the active marker, the active element), the partitions the maps and heatmaps show at each level, the spectrum's curves and the legend panel's title and entries, the click rules, the hover labels of the orbital curves and a one-line summary of its state (`Decomp.info`; the simulator's window does not show it: its information panel has one line per setting, `decomp level`, `active marker` and `active element`, the simulator's own) |
| `cmap_shading.py` | the colour maps of the panels, the shading of the Map view's energy maps, the Contour / Shading box and the two panels of the Relief light, one self-contained module (display machinery, no physics; it imports only the standard library, NumPy and CuPy, nothing from the simulator or `qtgui`, so it can be copied into sm4_Browser as it is): the two colour-map sets by role (`SETS`, `cmap_spec`, `make_cmap`: the `cmap set` choice) and the energy maps' isoluminant shading maps (`SHADING_CMAPS`), the box's state and rules (`BoxState`: the `Contour` and `Shading` toggles, the `Height` / `Relief` mode and the Relief light, its azimuth `az` in degrees, 0 up and increasing clockwise as in Kovesi's `relief.m`, kept within 0 to 360, and its elevation `el`, 0 to 90 degrees, both starting at 45, `relief.m`'s defaults, and kept whatever the toggles and the mode; the state's `key` carries them) and its builder (`build_box`, given the window's figure and the `qtgui` controls module; the box has no control of the light), the two light panels on axes the caller makes and places (`draw_light_top`, `draw_light_side`, each giving a `LightPanel` that shares the window's `BoxState` and is sent the caller's mouse events, `press`, `motion` and `release`): the top view, the plane from above with its centre point, a guide circle of radius R and the light as a dot at R cos(el) from the centre at the azimuth, the light as seen from above, a drag of the dot setting the azimuth only; the side view, the plane edge-on as a horizontal line through the centre point, seen at right angles to the azimuth so the light lies in the drawn plane, the light a dot on the arc of radius R at the elevation above the line with a line from it to the centre point, a drag up or down setting the elevation only, 0 to 90 degrees, the dot staying on the arc; just the controls, with no preview behind them, both drawn greyed and taking no drags while the mode is `Height`, which uses no light (`light_active`), the shading maths on NumPy or CuPy arrays (the simulator passes CuPy: on the GPU): the Height and Relief shades (`relief.m`'s Lambertian shading, Relief lit by the state's `az` and `el`, given to `shade_field`), the colours through the maps' own colour scale and the CIELAB L* stretch (`shaded_rgba`, `lstar_stretch`; a caller's darkness d, 0 to 1, is passed as the stretch's floor, 1 - d: d = 1 makes the darkest point black, d = 0 shades nothing), and the corners of the mesh of one quad per grid point |
| `feenstra.py` | the `Feenstra` toggle's arithmetic (CuPy): `current_from_ldos` (I(V), the LDOS integrated from 0 V), `iv_ratio` (the plain I/V; a sample exactly at 0 V given its neighbours' I/V), `divide` |
| `quadrature.py` | the integration rule (CuPy, float32): `simpson` (SciPy 1.18.1's composite Simpson, written out for the GPU) and `integral_from_zero` (from 0 V: the cubic through the two samples on each side of 0 V, E₋₂, E₋₁, E₁, E₂ (through E₋₁, 0, E₁, E₂ above and E₋₂, E₋₁, 0, E₁ below when a sample is at 0 V), its value at 0 V used by the unequal pair next to 0 V when the interval count is even, its exact integral over the interval next to 0 V when the count is odd, Simpson pairs over the rest) |
| `doscar_parser.py` | `SpinAwareDosParser`, `SpinMode`: DOSCAR reader for all three spin modes; the energies are rebuilt from line 6 as an even grid, E min + i (E max - E min)/(NEDOS - 1), minus E_F, the printed column only checked against it, within one tolerance for the whole column: half a unit of the last decimal the column carries (the most decimals any entry carries) plus 1e-6 eV. `read_doscar_header` reads only the header and gives the decomposition package its E_F |
| `locpot_manager.py` | `LocpotManager`: LOCPOT loading and `.npy` caching for all three spin modes; the cache, `LOCPOT.npy`, is written to and looked for at the path the simulator gives (`cache_path`: `LOCPOT.npy` in the folder `caches` inside the VASP folder, the simulator's one caches folder, `_cache_path`, which makes the folder; this class has no folder rule of its own); a `LOCPOT.npy` left directly in the VASP folder is neither moved nor read |
| `plot_dos_cursors_levels.py` | `DosPlotter`: DOS window with levels (total, Au / molecule, atom types, atoms, one atom's orbitals with hover labels; a click on a curve descends, a click beside the curves ascends; `m` toggles the \|m\| overlay in SOC mode), with three lines of its own in the information panel (`set_info_line`), the settings the clicks make, set at open and again on every change: the plot level (its number and name), the active element and the active atom (its label and its place, 1-based, in the POSCAR / CONTCAR order), each of the last two shown as held at every level and `none` only while none is held (run block at the bottom, `v_dir`) |
| `wrong_decays/` | UNUSED, not imported, fully commented (removing the leading `# ` from every line restores each file): `tilted_barrier_decay.py`, `tilted_barrier_decay_github.py`, `tier3_decay.py` and `sim_decay_full.py`. Their decay factor is the WKB average of 2κ over a barrier rising linearly from φ - E to φ - E + V, (4√(2m)/(3ħV)) [(φ - E + V)^(3/2) - (φ - E)^(3/2)] (the trapezoidal-barrier WKB integral; e.g. the TU Wien dissertation at iue.tuwien.ac.at/phd/goes, App. A.3, Eq. A.28; R. Rejali et al., Phys. Rev. B 107, 035406 (2023), Eq. 5), named after Chen in that code but not the model of Chen's book. `tilted_barrier_decay.py` holds the simulator's decay code with that factor: every state at V = \|E\| (the edge of its own bias window) in the topography and the display alike, E taken as max(E, 0) for occupied states with its '0-clamp' toggle, and κ = 0.512 √φ Å⁻¹ with Decay off and in the tip-height convergence step. `tilted_barrier_decay_github.py` holds GitHub's earlier version of the same model (float32 arrays, the `1e-6` V floor, `max(0.1, phi)`, no Feenstra). `tier3_decay.py` holds a frozen-bias topography and signed I(V)-column display built on the factor, `sim_decay_full.py` a full build of the simulator with it on the matplotlib GUI |

Inputs: DOSCAR, LOCPOT, POSCAR or CONTCAR (and INCAR for the spin-mode check). Outputs: figures and
the caches, every one written to and looked for in the folder `caches` inside the VASP folder
(`_cache_path`, which makes the folder when first needed; a cache left directly in the VASP folder
is neither moved nor read, and may be moved there by hand): `LOCPOT.npy` (the `locpot_manager.py`
row); the global topo's and the LDOS topo's `global_topo_*` / `ldos_topo_*` `.npy` caches, their
names carrying the snapped bias (the energy of its DOS sample), `_fold` and `_f32`; and the Line,
Path and Map LDOS caches, `line_ldos_<hash>.npz`, `path_ldos_<hash>.npz` and `map_ldos_<hash>.npz`,
`<hash>` the SHA-1 of the cache's key as text, the key itself stored in the file and checked on
load. A Line / Path file holds the LDOS (its total channel), its energies, its current and the
stripe's tip heights it was computed at, under the key of the view and its geometry (the line ends,
or the path nodes and Extend, and the point count), V_b and LDOS h, the window's snapped ends, the
decay model, the broadening, the unit cells, the precision and the grid size; a Map file holds the
energy maps (their total channel), their current, their energies and the LDOS topo heights they
were computed at, under the key of V_b and LDOS h, the map energies' samples, the window's snapped
ends, the decay model, the broadening, the unit cells, the precision and the grid size. The mag
channel (`Mag`) is in a `_mag.npz` file beside each, saved when it is computed. With `reuse_cache`
on, a matching file is loaded and nothing is computed (no stripe convergence, no LDOS); otherwise
the data are computed and saved under that key. Each decay model has its own files, so a change of
model, a rerun or a return to earlier settings loads them when they are there; in memory only the
shown data are kept. An energy map dragged to a new sample changes the map energies' samples, so the
maps are then saved under the file of the new key. The markers' spectra (Single Point, Map) are not
cached: they are computed on a marker's release or creation; the Line and Path markers' spectra are
rows of the cached Line / Path LDOS. A file's size is about points × energies × atoms × orbitals ×
4 bytes: 72 × 350 × 300 × 9 × 4 B ≈ 270 MB for a Line run, grid points × maps × atoms × orbitals ×
4 bytes for a Map file.

The console says what is converged: each tip-height convergence opens with `--- TIP CONVERGENCE:
<what>, bias … V, height … Å, n points ---` (`<what>`: `global topo`, at the topo bias and Topo h;
`LDOS topo`, at V_b and LDOS h; `Line stripe` or `Path stripe`, at V_b and LDOS h, the LDOS topo's
tip height the stripe starts from), after its setpoint print, `[*] Global Topo Setpoint: …`,
`[*] LDOS Topo Setpoint: …`, `[*] Line Topo Setpoint: …` or `[*] Path Topo Setpoint: …`; its Iter
lines and its closing line follow.

Dependencies: numpy, scipy, cupy (CUDA GPU), pymatgen, PySide6 and pyqtgraph (the package's own
`qtgui` copy, `stm_from_dft/qtgui/`), matplotlib (the publication files written by the toolbar's
Save button).

The window sizes its panels by rule 11 of `matplotlib_to_real_gui.md`, but is never trimmed to
them: each view takes only the layout its rules give (Line and Path their two rows, Map its top row
over the energy maps, Single Point its one row or, by its tall-window rule, two); the topographies,
the heatmaps and the energy maps are sized first and fill the window, every row at one plot-box
height. The limits: a topography's plot box at most twice a heatmap's or an energy map's, and in
Map at least an energy map's (a map no larger than a topo panel, a topo panel no larger than twice a
map); the placement panel's plot box never smaller than the size at which the atom dots, as drawn
on screen (every atom at every image inside the panel's limits, dots that overlap, the cell's
periodic images among them, counted once), cover at most a third of it, whether `Atoms` is on or
off (found by bisection), the 2× cap giving way where both cannot hold; the window's
minimum size follows from these limits (in Map with the shading row, whose light panels are a map
panel's size). Line's LDOS topo is 2:3, width to height, L/3 either side of the line by its length L
(the LDOS topo tiled and turned into the line's frame, with the line, its ends and the markers);
Path's is the LDOS topo in the path's rectangle, the smallest rectangle at any angle that holds
every sample point of the path (the minimum-area rectangle, over the directions of the convex
hull's edges), with a margin of 1/8 of its width added left and right and of its height top and
bottom, turned 90 degrees when it is wider than tall, so the panel is square or taller than wide,
with the path, its nodes and the markers; neither takes the crop or the turn below. The Line and
Path heatmaps are sized as squares, then fill their row's whole slot, stretched in x or y, never
taking room from another panel. The curve panels (the Tip Height profile and the Partitioned LDOS,
its legend panel of one fixed width at its right) take the room those leave, never less than a
readable size: in Line and Path the second row takes all the height the top row leaves, with no
empty band; in Map the spectrum, in the top row, has its plot box held equal to a topo panel's; in
Single Point the spectrum, its legend panel at its right, stands beside the topo sub-section at one
topo panel's size, the row as large as the two allow, and goes under the topo sub-section, filling
the bottom row, exactly when that one row would leave an empty band below it, and back as soon as
it no longer would. The panels drawn in the unit cell's frame (the global topo; in Single Point and
Map the LDOS topo, which takes the global topo's crop; in Map the energy maps) are shown as drawn or
turned 90 degrees anticlockwise (x up, y to the left: a rotation, never a mirror), whichever fits
the window better; that is decided when a view is built and again on Crop / Uncrop, never on a
resize, and the console says which way and why. Line's and Path's LDOS topo and the heatmaps of
energy against position never turn this way. Everything on a
turned panel keeps its data coordinates: the cursors, the markers, the line and path endpoints, the
limits and the publication files (which show the turn as on screen).
