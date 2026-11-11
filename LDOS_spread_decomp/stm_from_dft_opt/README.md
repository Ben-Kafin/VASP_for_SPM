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

`stm_from_dft/plot_decay.py`, run directly, draws the decay figure: the barriers, the weights, the
current and dI/dV of a set of decay models side by side for one LDOS at one tip point (its row in
the package table below). It opens no VASP files: the block at its bottom sets the barrier, the
distance, the biases and a synthetic LDOS.

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
level line only in Map), the global topo (above it its tip-height box, its Crop / Uncrop box and,
under the two, its line of four cursor values) and the LDOS topo (the LDOS h box above it, in every
view; in Single Point and Map with a Crop / Uncrop box and a line of cursor values of its own).
Single Point: the topo sub-section and the spectrum with its legend panel at its right in one row,
or the spectrum under the topo sub-section in a tall window (the sizes below). Line and Path: the
top row, the topo sub-section, then the stripe (titled `Topo`), one heatmap per partition and their
colour bars, under their label, so that the LDOS topo's, the stripe's and the heatmaps' distance
axes line up, over the second row, Tip Height and the spectrum with its legend panel at its right,
side by side. Map: the top row, the topo sub-section, then the spectrum with its legend panel above
it, then the two light panels, the top view over the side view at one width, together as tall as
the row's plot boxes, with the Contour / Shading box above them (their rows
below), over the block of energy maps (the maps wrap into the rows and columns that make them
largest, chosen again at every resize). The spectrum is titled `LDOS` at decomp level 0 and
`Partitioned LDOS` once a level is stepped into. The
placement panel, where the markers, the line ends and the path nodes are placed, is the global topo
in Single Point, Line and Path and the LDOS topo in Map; the markers are drawn on the LDOS topo too.
The arrangement is chosen
again only when a view's panels are built or their number changes (the drill-down level, the
per-heatmap colour bars), when a crop changes a
panel's shape and when the window is resized (maximizing included); the window is never trimmed to
its panels, so a maximized window stays full screen. Where the arranged panels leave room in the panel
area (the window less the left column and the information panel), each row section (Map's top row;
its block of energy maps, whose own rows keep their places in the block) is moved sideways on its
own to the middle of the area's width, and all of them are moved down together to the middle of
its height; the sizes are as arranged. At decomp level 0 the energy maps carry no `Total` label, so every map of the
block has the same size. The energy maps' colour bar is as wide as the topo colour bar while `cmap norm` is off (the shared bar, or a
partition's under `Norm`); with `cmap norm` on each map's own bar is half that width. The line ends, the path nodes and the Single
Point and Map markers recompute on release: the tip heights, the LDOS and the spectra a drag
changes are computed once, when the button is let go. The global topo (every view) and the LDOS
topo (Single Point and Map) are each cropped on their own, as the sm4_Browser Line window crops its
heatmap; the energy maps follow the LDOS topo's crop, and Line's and Path's LDOS topo, sized by the
line or the path, has none. Two draggable cursors per axis start at the ends of the panel's view,
the area outside each pair is shaded, and each cursor's dark line runs on past the panel's frame as
a short stub (over the top edge for an x cursor, at the right for a y cursor) by which it can be
grabbed as well; a held cursor follows the mouse outside the plot box. The four cursor values (x
low, x high, y low, y high) are in the line of boxes under the topo's two control boxes and take a
typed value. Crop applies the ranges between the cursors as the panel's limits and Uncrop restores
the view; a crop changes the panel's shape, that topo's group decides its orientation again (the
global topo alone; the LDOS topo with the energy maps) and the arrangement follows. A crop sets
aside what is drawn on its panel outside it, and Uncrop puts it back where it was: in Single Point
the markers outside either topo's crop (a marker is shown only while inside both), in Map the
markers outside the LDOS topo's crop; in Line the line keeps its place over the data and is
shortened to the part inside the crop, with new ends for the time being, its markers keep their
place over the data and those beyond a new end are set aside; in Path the nodes outside are taken
out, the ones left keep their order and connect directly, and the markers keep their fraction of
the path. A crop that would leave too little (no marker, no part of the line, fewer than two nodes)
sets nothing of that kind aside.
The cells a topo shows are set per side (`Topo Cells`, `LDOS Cells`: their row below), the home cell
with its corner at (0, 0) and the others added toward +a and +b; an axis runs below 0 only by what
a lattice vector's negative component needs. A panel's limits are the bounding box of the cells
shown, the largest the data can be drawn in it, and the corners a skewed cell leaves empty are
filled with the periodic data. When a marker, a line end or a path node drawn on a panel lies
outside that box, the panel keeps its size and shape and its view widens (the data is drawn
smaller) at the box's aspect ratio, only on the sides where such points lie and far enough that
the outermost dot is whole; the whole view is filled with the periodic data and the cells shown are
outlined in white. The energy maps share the LDOS topo's view. The widening is applied after the
orientation and the arrangement are set from the cells or the crop and the window's size, and never
changes them. A marker cannot be dragged past the range shown; when it is dropped the view is set
again.
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
| `cmap set` radio | left column, under `mode` | the colour-map set of every panel (`cmap_shading.py`, its row below; the constructor's `cmap_set` is the one the window opens with): `default 1`, the topographies, the Line / Path stripe and the topo colour bar black to firebrick to yellow, the Line and Path heatmaps `nipy_spectral`, the energy maps `MAP_CMAP`, and under `Mag` the signed spin density `bwr` and \|m\| `hot`, in the maps and the heatmaps; `default 2`, `CET-L3`, `CET-R1`, `CET-L6`, `CET-D1A` and `CET-L19` in the same places; `default 3`, `default 1`'s maps again, differing from `default 1` only in the `Height` shading of the LDOS maps (the `Contour`, `Shading` row below). A change recolours every panel in place, running or not (nothing recomputed or arranged again); a shaded energy map keeps its shading map, the same in every set |
| `broadening` box | left column, under `cmap set` | one row per atom type: its check, its σ box (eV, 0 to 1, two decimals) and `eV`. The boxes are always shown and take a value; a value does nothing while its check is off. With the check on, the extra Gaussian broadening of that type's DOS applies at once while running, and at the next RUN while stopped |
| `RUN` / `STOP` | left column, bottom, under the toggles, as wide as the column | starts and stops the computation; on STOP the line endpoints / path nodes and marker positions are printed in the form `run_interactive` takes |
| toggles | left column, bottom, over RUN | `Atoms`, `Decay` (`exact` and `linearized` indented under it), `Norm`, `cmap norm` (Map only), `Feenstra`, `Mag`, `Cell` (the one unit cell outlined in white), `Extend` (Path only). A row that is not in the view leaves no gap and is left out of the information panel. The left column is as wide as the widest thing shown in it in the view |
| `Decay` | toggles | **Physics:** the tunneling matrix element that weights the DOS of every atom image at a tip point, with the tip DOS constant (the s-wave tip of J. Tersoff and D. R. Hamann, Phys. Rev. B 31, 805 (1985)). As in a measurement (C. J. Chen, *Introduction to Scanning Tunneling Microscopy*, 3rd ed., Oxford 2021, Sec. 2.2.3-2.2.4), the topography is converged on the current at one frozen bias (the current of Eq. 2.40 with the weight below: the window [0, V_b], every state weighted at that frozen bias, w(E, V_b)), and the displayed LDOS is dI/dV with the bias swept (that current's derivative at each V: the state entering at the window edge plus the change of every state already in [0, V]; that integral starts at 0 V whether or not the window reaches it). A state at E (eV from E_F) of an atom image at distance d from the tip at bias V carries the weight w(E, V), φ the LOCPOT potential at the tip point minus E_F (Chen's φ̄, one average work function for the junction, taken here at each tip point). On (the default): one of two models, the rows `exact` and `linearized` under it (their row below). `exact` (the default): the exact tilted barrier, w = exp(-S), S = (4 k d / (3 V)) [a₊^(3/2) - b₊^(3/2)], a = φ - E + V, b = φ - E, x₊ = max(x, 0): the WKB exponent 2∫κ dz across a barrier rising linearly from φ - E at the sample to φ - E + V at the tip, κ = 0 wherever the state lies above the barrier (a state above one end sees only the part of the barrier above it; above both ends, w = 1). `linearized`: its first order in (E - V/2)/φ and V/φ, the energy-dependent element of Chen's Eq. 2.40, w = exp(-2 κ0 d + κ0 (E - V/2) d / φ), dw/dV = -(κ0 d / (2 φ)) w (Eqs. 2.34, 2.37-2.39: κ to first order in the state's energy E - V/2 from the middle of the bias window). Off: exp(-2 κ0 d), the same weight without its energy and bias dependence. κ0 = k √φ = √(2 m φ)/ħ, k = 0.512317 Å⁻¹ eV^-1/2 from m_e, h and e, in every use (the tip-height convergence step included). **Machinery:** V_b is the bias end's snapped DOS sample (the `bias` row): the topography's window is [0, V_b] from exactly 0 V, with the weight at V_b. All integrals use the rule of `quadrature.py` (its row below). The topography caches carry `_decay_exact` (on, `exact`), `_decay_linearized` (on, `linearized`) or `_decay_k0` (off), followed by `_fold` and `_f32` |
| `exact`, `linearized` | toggles, indented under `Decay` | the Decay-on model (the physics of both in the `Decay` row): exactly one is on, `exact` at start (`run_interactive`'s `decay_linearized=False`; `True` starts on `linearized`); a click on one turns it on and the other off, a click on the one already on leaves it on. While Decay is off both are ignored (a click then changes no model and recomputes nothing); while it is on, a change of model recomputes what a `Decay` change recomputes. `show_decay_toggle=False` hides both with `Decay`. The information panel lists both rows. **Machinery:** both models use the same steps, the rule of `quadrature.py` and float32: the topography from the frozen-bias weight w(E, V_b), integrated from 0 V; the swept dI/dV and the current (`Feenstra`) in one pass of a CUDA kernel per launch of tip points, with the console line `x/n spectra calculated`, one line per channel computed, rewritten in place after each launch, led by what is computed (`Line points`, `Map energy maps`, …) and the channel's label: `total: ` (none for an unpolarized run) or, for the mag channel while `Mag` is on, `|m|: ` (noncollinear / SOC) or `m: ` (collinear polarized). `exact`'s weight does not separate in E and V, so its kernel (`exact_sweep`) integrates each requested V on its own, one thread per (tip point, requested V, atom), each thread doing for its one V the same work over the atom's images, in the same order, into its own output slot; `linearized`'s weight is one factor per image in V, w(E, V) = w(E, 0) w(0, V) / w(0, 0), so its kernel (`linearized_sweep`) gives dI/dV and the current at every requested V in one walk outward from 0 V on each side, one thread per (tip point, atom). Each model has its own topography caches (the `Decay` row) and its own Line, Path and Map LDOS caches, so a change of model loads that model's files when they are there (the Outputs paragraph below) |
| `Feenstra` | toggles | **Physics:** Feenstra's normalized conductance (dI/dV)/(I/V) on top of the calculated LDOS (`feenstra.py`): at the same tip positions as the display, I(V) from 0 V with the σ broadening in force (I(0) = 0 exactly; a window that does not reach 0 V is extended to it for the integral): with Decay on, the current of the Decay-on model (`exact` or `linearized`) at each V, the integral from 0 to V of ρ(E) w(E, V), as the topography's current at its bias; with Decay off, the total LDOS integrated. I/V is the plain I(V)/V, and the spectra, heatmaps and maps are divided by it along their energy axis; every partition, atom and orbital of a point is divided by the same total I/V (with `Mag` on, the magnetization over the total I/V). **Machinery:** I(V) is integrated by the rule of `quadrature.py` (its row below); with Decay on it comes from the same pass as the displayed dI/dV, the total channel's (the mag channel's pass computes no current). The current, I/V and the division are computed on the GPU. Display only: no LDOS recompute; `Norm`, when on, acts on the ratio. A point with no states between 0 V and V (0/0) shows 0. The spectrum's y axis and the label over the heatmaps and maps read (dI/dV)/(I/V) |
| `Mag` | toggles | the magnetization channel shown in place of the total LDOS, in the spectra, the heatmaps and the energy maps, with its colour maps (the `cmap set` row): \|m\| for a noncollinear / SOC run, m = up - down, the signed spin density, for a collinear polarized run, the two treated the same; an unpolarized run has none, and the total stays shown. **Machinery:** the LDOS is computed as its total channel, in one pass (noncollinear / SOC the DOS's total, collinear polarized up + down, unpolarized its one channel); the current (`Feenstra`) comes from that pass. The mag channel is not computed until `Mag` is on, and then only for what is shown (the Line / Path LDOS, Map's maps and a dragged map, the marker spectra), in one pass of its own (collinear polarized on up - down); once computed it is kept and not computed again. With the Line / Path LDOS and Map's maps it is saved in a `_mag` file beside their LDOS cache when it is computed, and loaded instead of computed when that file is there (the Outputs paragraph below); turning `Mag` on while running computes it, or loads it, for what is shown and kept without it, the total not computed again. A map dragged while `Mag` is off keeps the other maps' mag columns and marks only its own missing; when `Mag` is on again only the missing columns are computed, and the `_mag` file is saved once the channel is whole. The marker spectra are not cached: their mag channel is computed with them, on a marker's release or creation, and when `Mag` is turned on |
| `cmap norm` | toggles, directly under `Norm`, Map only | off: every energy map on one colour scale, the largest value over all of them, with one colour bar; on: each map its own scale with its own bar beside it (as the sm4_Browser Map window's `cmap norm`). The maps' colour map is the Map map of the window's colour-map set (the constructor's `cmap_set`; the `cmap set` choice in the left column): `MAP_CMAP` in `default 1` and `default 3`, the sm4_Browser's (`custom_map`, midnight blue to light sky blue), `CET-L6` in `default 2`; while the maps are shaded (the `Contour`, `Shading` row below), the isoluminant `CET-I1` (under `Mag`, `CET-D11` for the signed spin density and `CET-I3` for \|m\|), except the LDOS maps under `Height` (`default 1` and `default 2`: the set's topo map and Map map combined, with a colour square panel in front of the maps and, with `cmap norm` on, a legend rectangle under each map in place of its colour bar; `default 3`: the m1ds colour square, whose legend replaces each colour bar) and, in `default 1` and `default 3`, under `Relief` alone (the set's own Map map) |
| contour slice | the topo colour bar, far left of Map's topo sub-section (Map only: the other views' bar has no level line) | the sm4_Browser Map window's, drawn while `Contour` is on (the Contour / Shading box): a cyan level line on the bar (half the range at open) is dragged by a press on it, following the cursor's height while the button is down even off the narrow bar sideways; the LDOS topo's contour at that height (as a fraction of the height range of Map's copy of the LDOS topo, the heights its maps were computed at), the perimeters of everything above it (WSxM's flooding, as outlines), is drawn on every energy map as white 1 pt traces, at the images the maps show, never on the topography panels; the traces go to the publication files with the maps (`contour.py`: marching squares on the height map, placed on the unit cell's lattice) |
| `Contour`, `Shading`, `Height`, `Relief` | over Map's light panels (an untitled box: `Contour` and `Shading` side by side, next to the `Height` line over the `Relief` line; the `Relief` line has a number box at its end, the `Height` line only in `default 3`) | the Contour / Shading box (`cmap_shading.py`): `Contour` and `Shading` are the masters, at most one on, both may be off (turning one on turns the other off); under `Shading`, `Height` and `Relief` are each on or off on their own, so either, both or neither shades the maps, and `Contour` is never on with either. The window opens with `Contour` on, `Shading` off, `Height` on and `Relief` off. `Contour` on: the contour slice (its row above); off: no traces on the energy maps and the level line hidden. The number box at the end of a line (no label; `Height max` and `Relief max` in the information panel) is that shading's strength, 0 to 1 (two decimals), 0.05 for `Height` and 0.5 for `Relief` at open: 1 makes its darkest point black, 0 shades nothing. The `Height` box is shown only in `default 3`: in `default 1` and `default 2` the `Height` shading of the LDOS maps takes no strength (the `Mag` maps' `Height` shading keeps the value the box last had). A shaded energy map is a mesh of one colour per grid point of the cell (the cell's repeated edge left out, each point's quad centred on it, at the images the maps show). `Height` in `default 1` and `default 2`, the LDOS maps: the set's own topo map and Map map, unaltered, combined into one two-dimensional colour scale (`topo_ldos_blend.py`, its row below), in which the map's value through its own colour scale and the LDOS topo's height over its range are looked up together: the lowest height shows the Map map exactly, and the colours turn from it toward the topo map's with the height, every colour keeping the lightness (CIELAB L*) of the Map map's colour for its value, so that a black-and-white print of a shaded map is the print of the unshaded map and the height shows as colour only. A colour square panel stands in front of each partition's maps, of a map's size and shape (the LDOS left to right, the height bottom to top), with the topo map's colour bar left of it (`height (Å)`) and the Map map's colour bar under it; with `cmap norm` off that bar carries the maps' scale (`LDOS (a.u. 1e<exp>)`) and the maps have no bar of their own, and with it on the bar runs from `low` to `high` and every map has under it a legend rectangle an eighth of the map's height tall, the combination's colours with the map's own LDOS scale left to right and the height bottom to top (`Topo` on its left). `Height` in `default 3`, the LDOS maps: the m1ds colour square, a plane of CIELAB in which the map's value through its own colour scale (purple to green) and the LDOS topo's height over its range (blue at the lowest to a light tan at the highest) are looked up together, then darkened toward black by the `Height` strength at the lowest height and not at all at the highest. Each colour bar of the maps is then the square's legend: a colour rectangle as tall as the maps it stands beside (LDOS up, height left to right), a thin LDOS bar right of it with the tick values and the vertical title `LDOS (a.u. 1e<exp>)`, and a thin topo bar under it with the lowest and the highest height in Å, every size a fixed fraction of the legend's height (one legend per map under `cmap norm`, one per partition with the decomposition's own scales, else one for the block); and a topo panel stands in front of each partition's maps, of a map's size and shape, the LDOS topo alone in the square's topo colours with the `Height` shading only, with its own colour bar of the rectangle's shape. `Height` elsewhere (the `Mag` maps): the isoluminant shading map through the map's own colour scale, each colour's CIELAB L* scaled by a factor running evenly from 1 - strength at the lowest height over the cell to 1 at the highest. `Relief`: the LDOS topo's slopes lit by the light the two light panels set (their row below; azimuth 45°, elevation 45° at open, the defaults of Kovesi's `relief.m`, whose Lambertian shading it is; the gradients wrap around the cell; the light keeps its azimuth on screen when the maps are turned), applied by the same L* scaling at the `Relief` strength: alone, on the set's own Map map for the LDOS maps of `default 1` and `default 3` and on the isoluminant shading map elsewhere; with `Height` also on, on top of the `Height` colours. The 1D colour bars show the map being shaded, unshaded. Display only, running or not: nothing is recomputed, and the block of maps is built again only when its kind of `Height` colouring changes (the combination, the colour square, or neither). With `Shading` off, or on with both `Height` and `Relief` off, the set's maps and the filled contours come back. |
| top view, side view | the last item of Map's top row, the top view over the side view at one width, together as tall as the row's plot boxes | the Relief light's two panels (`cmap_shading.py`, its row below), just the controls with no preview behind them: the top view, the surface from above (the panel's frame is its only box) with its centre point, a guide circle of radius R and the light as a dot at its azimuth (0 up, increasing clockwise), R cos(el) from the centre (the light as seen from above), a drag of the dot setting the azimuth only; the side view, the plane edge-on as a horizontal line through the centre point, seen at right angles to the azimuth so the light lies in the drawn plane, the light a dot on the arc of radius R at the elevation above the line with a line from it to the centre point, a drag up or down setting the elevation only (0 to 90°, the dot staying on the arc). While a dot is dragged the energy maps are shaded again live. In `Height`, which uses no light, both are greyed and take no drags. The information panel shows the light in its `azimuth` and `elevation` lines |
| `Topo bias`, `Topo h`, `LDOS h` | boxes above the topo panels, in every view: `Topo bias` over `Topo h` above the global topo, `LDOS h` above the LDOS topo | the global topo's bias (V; snapped as `topo_bias` is, 0 refused and the box put back) and the tip heights (Å, 0 to the cell top above the highest atom), with a `Rerun` button per box: the global topo's Rerun applies `Topo bias` and `Topo h` together and recomputes the global topo only (the LDOS topo and the energy maps are not computed again); `LDOS h` is the LDOS topo's (the one the spectra, the stripes and the energy maps take their tip heights from: the Run paragraph); a value applies when Rerun is pressed while running |
| number boxes | left column, stacked under the `broadening` box | `Topo Cells` x, y (the cells of the global topo, along a and along b, 1 to 9, 2 at open), `LDOS Cells` x, y (the same for the LDOS topo and the energy maps; Single Point and Map only: Line's and Path's LDOS topo is sized by the line or the path), `Path Pts` (Path view, 2 to 12 nodes), `Markers` (1 to 10), `E Pts` (Map view, 1 to 20 energy maps), `E Min`, `E Max` (eV, -5 to 5; each with its `bias` toggle beside it, in its frame), `LDOS γ` (the heatmaps' power-law colour scale, re-colours only). E Min and E Max snap to the nearest DOS sample (a tie: E Min to the higher sample, E Max to the lower); an end of exactly 0 is exactly 0 V, not a sample: the bias is then on the other end (the `bias` row); each Map energy snaps to the nearest displayed sample, a tie going to the sample closer to 0. Every number box is wide enough for its digits and shows at most four of them, with the 0 before the decimal point (0.202, 2.500, 12.35, 127.6); the value itself is kept in full |
| `bias` | left column, beside `E Min` and beside `E Max`, in every view | which end of the window is the LDOS bias V_b (that end's snapped DOS sample, at which the LDOS topo and the stripes are converged, their topography window [0, V_b]): exactly one of the two toggles is on, and one always is, as `exact` / `linearized` (a click on the one on leaves it on); `run_interactive`'s `ldos_bias_sign` sets only their starting state (`'neg'` E Min, `'pos'` E Max). An end at 0.0, however it got there, has the bias on the other end: at start when the end `ldos_bias_sign` picks is 0.0, after an edit of the bias end to 0.0, and a click on the toggle of an end at 0.0 leaves it off; when that end is nonzero again, the bias stays on the other end until a toggle is clicked. Both ends at 0.0 stay an error. A switch recomputes what a change of that end's value recomputes, with the same caches. The information panel lists both, as `E Min bias` and `E Max bias` |

Every count and value is a number box: type a value and press Enter, or select a digit and press
Up or Down to step it (a Value stops at its limits; a step applies at once). Markers, line ends,
path nodes and the energy markers of the Map view are dragged with the mouse (an energy marker dropped with RUN off keeps its snapped energy and nothing is computed; RUN then computes its map and re-sorts the maps); a left click on a
curve of the spectrum, or on its entry in the legend panel, descends a
level (partition, atom types, orbitals) and a right click anywhere in the panel (on a curve, on
empty space, on its tick labels, axis labels or title) or in its legend panel, or the legend
panel's Back button, as wide as the panel, ascends; these clicks always act (the toolbar has no Pan
or Zoom). The legend panel is at the spectrum's right in Single Point, Line and Path, titled
`Markers` at level 0; in Map it is above the spectrum with no title, as tall at most as the global
topo's control boxes, scrolling beyond that. At level 0 its entries name the markers by their place:
`(x, y)` in Single Point and Map, `at d Å` along the line in Line and Path. The mouse
wheel scrolls the legend panel's entries from anywhere over the panel (its entries, its title, its
Back row, its scroll bar's strip). At the orbital level the curves carry hover labels. The
information panel lists the settings of the controls (the `bias` toggles, the cursor values and the
two shading strengths among them) and has one line of its own for each setting the mouse makes: `decomp level`
(its number and name), `active marker`, `active element`, one `marker k` line per marker (its
position; in Line and Path its distance along the line), `line` (the Line's ends or the Path's nodes
with its length), `topo crop`, `ldos crop`, and in Map `map energies`, `contour level`, `azimuth`
and `elevation`. The
MO to LDOS decomposition wrapper (`mo_ldos_decomp`) adds its σ box and its `Send to decomp`
buttons to the left column through `Interactive_STM_Simulator.add_control_box`.

## Package `stm_from_dft/`

| module | role |
|---|---|
| `simulator.py` | **Physics** (the block THE DECAY MODEL at the top of the file): the tunneling matrix elements of the two Decay-on models, the exact tilted barrier `gpu_exact_factor` (`exact`) and its first order `gpu_linearized_factor` (`linearized`), and the Decay-off weight `gpu_decay_off_factor`. The decay model is described in the `Decay` row above. **Machinery** (NUMERICS AND GPU, the rest of the file): `Unified_STM_Simulator` (GPU LDOS, its total channel by default and its mag channel on request, the integration rule of `quadrature.py`, constant-current convergence, caches, every one in the VASP folder's `caches/` (`_cache_path`: the Outputs paragraph below); the GPU computes in float32 from the float64 DOS grid: one step, each sample's energy formed from its index, so the samples are exactly equally spaced) and `Interactive_STM_Simulator` (the window: its controls made once, its panels per view, on the package's own `qtgui` copy, `stm_from_dft/qtgui/`) |
| `ldos_decomp.py` | the decomposition of the spectrum panel (titled `LDOS` at level 0, `Partitioned LDOS` at levels 1 to 3), everything in it that depends on its level (display machinery, no physics): its state (the level, the active marker, the active element), the partitions the maps and heatmaps show at each level, the spectrum's curves and the legend panel's title (`Markers`, `Parts`, `Atoms`, `Orbitals`) and entries (at level 0 one per marker, labelled with the marker's place as the simulator gives it, `Decomp.curves`' `marker_labels`), the click rules, the hover labels of the orbital curves and a one-line summary of its state (`Decomp.info`; the simulator's window does not show it: its information panel has one line per setting, `decomp level`, `active marker` and `active element`, the simulator's own) |
| `cmap_shading.py` | the colour maps of the panels, the shading of the Map view's energy maps, the Contour / Shading box and the two panels of the Relief light, one self-contained module (display machinery, no physics; it imports only the standard library, NumPy and CuPy, nothing from the simulator or `qtgui`, so it can be copied into sm4_Browser as it is): the three colour-map sets by role (`SETS`, `cmap_spec`, `make_cmap`: the `cmap set` choice) and the energy maps' isoluminant shading maps (`SHADING_CMAPS`), which `Height` colouring a set's LDOS maps take (`height_blend`, `height_square`), `default 3`'s Height colour square (`M1DS`, `square_rgba`, `square_topo_rgba`, `square_legend`), the lookup of a map's colours in the topo and LDOS combination (`blend_rgba`, given the simulator's `TopoLdosBlend`), the box's state and rules (`BoxState`: the `Contour` and `Shading` master toggles, the `Height` and `Relief` toggles under `Shading` with a strength each, and the Relief light, its azimuth `az` in degrees, 0 up and increasing clockwise as in Kovesi's `relief.m`, kept within 0 to 360, and its elevation `el`, 0 to 90 degrees, both starting at 45, `relief.m`'s defaults, and kept whatever the toggles; the state's `key` carries them) and its builder (`build_box`, given the window's figure and the `qtgui` controls module; the box has no control of the light), the two light panels on axes the caller makes and places (`draw_light_top`, `draw_light_side`, each giving a `LightPanel` that shares the window's `BoxState` and is sent the caller's mouse events, `press`, `motion` and `release`): the top view, square, its frame the plane from above (no square is drawn inside it), with its centre point, a guide circle of radius R and the light as a dot at R cos(el) from the centre at the azimuth, the light as seen from above, a drag of the dot setting the azimuth only; the side view, at the top view's width and scale and `LIGHT_SIDE_RATIO` of that width in height, the plane edge-on as a horizontal line through the centre point, from -R to R, seen at right angles to the azimuth so the light lies in the drawn plane, the light a dot on the arc of radius R at the elevation above the line with a line from it to the centre point, a drag up or down setting the elevation only, 0 to 90 degrees, the dot staying on the arc; just the controls, with no preview behind them, both drawn greyed and taking no drags while the `Relief` toggle is off (`light_active`: `Relief` is the one shading that uses the light, whatever the `Shading` and `Height` toggles), the shading maths on NumPy or CuPy arrays (the simulator passes CuPy: on the GPU): the Height and Relief shades (`relief.m`'s Lambertian shading, Relief lit by the state's `az` and `el`, given to `shade_field`), the colours through the maps' own colour scale and the CIELAB L* stretch (`shaded_rgba`, `lstar_stretch`; a caller's darkness d, 0 to 1, is passed as the stretch's floor, 1 - d: d = 1 makes the darkest point black, d = 0 shades nothing), and the corners of the mesh of one quad per grid point |
| `topo_ldos_blend.py` | one class, `TopoLdosBlend`, NumPy only and importing nothing of the package, so the file can be taken into the sm4_Browser as it is (display machinery, no physics): a topography colour map and an LDOS colour map, used exactly as they are, combined into one two-dimensional colour scale. `TopoLdosBlend(topo, ldos)` takes the two maps (callables of fractions, or arrays of colours); `blend(s, t)` gives the sRGB colours of LDOS fractions `s` and height fractions `t` from a table of the rule made once, `blend.colours(s, t)` evaluates the rule itself, `blend.square()` is the legend, `blend.topo_bar()` and `blend.ldos_bar()` the two maps alone. The rule, in CIELAB against the D65 white: the topo colour is laid over the LDOS colour as a film whose opacity is the topo colour's own lightness (a black topo colour alters nothing), held at the topo map's yellowest colour above the height where the map is at its yellowest (the saturation rule holds that colour's saturation there too); the result is lightened by the topo colour's grey scale; the LDOS colour's hue is turned toward the topo map's yellowest hue and its saturation toward the topo colour's, a quarter of the way from the lightened film to that colour; the corner of the lowest LDOS and the highest height is darkened by 42%; and last the colour keeps its a* and b* and takes the LDOS colour's own L* (its saturation alone reduced where that falls outside the display's range), so the result is exactly as light as the LDOS map's colour at every height. The simulator makes one per colour-map set (`default 1`, `default 2`) from the set's own topo map and Map map |
| `feenstra.py` | the `Feenstra` toggle's arithmetic (CuPy): `current_from_ldos` (I(V), the LDOS integrated from 0 V), `iv_ratio` (the plain I/V; a sample exactly at 0 V given its neighbours' I/V), `divide` |
| `quadrature.py` | the integration rule (CuPy, float32): `simpson` (SciPy 1.18.1's composite Simpson, written out for the GPU) and `integral_from_zero` (from 0 V: the cubic through the two samples on each side of 0 V, E₋₂, E₋₁, E₁, E₂ (through E₋₁, 0, E₁, E₂ above and E₋₂, E₋₁, 0, E₁ below when a sample is at 0 V), its value at 0 V used by the unequal pair next to 0 V when the interval count is even, its exact integral over the interval next to 0 V when the count is odd, Simpson pairs over the rest) |
| `doscar_parser.py` | `SpinAwareDosParser`, `SpinMode`: DOSCAR reader for all three spin modes; the energies are rebuilt from line 6 as an even grid, E min + i (E max - E min)/(NEDOS - 1), minus E_F, the printed column only checked against it, within one tolerance for the whole column: half a unit of the last decimal the column carries (the most decimals any entry carries) plus 1e-6 eV. `read_doscar_header` reads only the header and gives the decomposition package its E_F |
| `locpot_manager.py` | `LocpotManager`: LOCPOT loading and `.npy` caching for all three spin modes; the cache, `LOCPOT.npy`, is written to and looked for at the path the simulator gives (`cache_path`: `LOCPOT.npy` in the folder `caches` inside the VASP folder, the simulator's one caches folder, `_cache_path`, which makes the folder; this class has no folder rule of its own); a `LOCPOT.npy` left directly in the VASP folder is neither moved nor read |
| `plot_dos_cursors_levels.py` | `DosPlotter`: DOS window with levels (total, Au / molecule, atom types, atoms, one atom's orbitals with hover labels; a click on a curve descends, a click beside the curves ascends; `m` toggles the \|m\| overlay in SOC mode), with three lines of its own in the information panel (`set_info_line`), the settings the clicks make, set at open and again on every change: the plot level (its number and name), the active element and the active atom (its label and its place, 1-based, in the POSCAR / CONTCAR order), each of the last two shown as held at every level and `none` only while none is held (run block at the bottom, `v_dir`) |
| `plot_decay.py` | the decay figure, a standalone matplotlib script (not imported by the package; it needs CuPy and the GPU, as the simulator does): run the file directly for the synthetic LDOS of the block at its bottom, or call `decay_barrier_figure(energies, ldos, models, v_topo, v_max=2.0, save_path=None)` with an LDOS on an evenly spaced grid that has two samples at least on each side of 0 V. For one LDOS at one tip point (the barrier φ and the atom-to-tip distance d each model carries) it draws a set of decay models side by side, one framed section per model. The top block, in black, across the window: the input LDOS, its integrated current (no tunneling decay), that current's I/V, its Feenstra ratio ρ/(I/V), and the colour key of the bias. Each section, top to bottom: the barrier (the topo column at the frozen bias `v_topo`, the LDOS column over the swept biases -`v_max` to `v_max`, φ marked); the model's equations (the barrier, the barrier to the weight, the weights to I and dI/dV); the weights on the same two columns; I(V) and dI/dV(V); I(V)/V and the model's Feenstra ratio (dI/dV)/(I/V). Colour is the bias the model passed to its factor at that point; every panel type shares its y range across the sections. Two sets of models, chosen by `DECAYS` in the bottom block: `'correct'` (`simulator_decays`), the simulator's `exact` and `linearized`, with nothing of the decay computed in this file: the weights are `gpu_exact_factor` / `gpu_linearized_factor` called as the topography calls them, and I(V) and dI/dV(V) come from the simulator's `exact_sweep` / `linearized_sweep` kernels launched on the one LDOS as a one-atom, one-orbital, one-image DOS at one tip point; `'wrong'` (`wrong_decays`), the superseded decays (`oldest`, Tier 2, Tier 2 with the 0-clamp, Tier 3, and McDowell's `VASP_ldos_map`), their factor functions copied verbatim into the class and their usages transcribed from their call sites (the files of `wrong_decays/` are commented out, so nothing is imported from them). The integrated current, I/V and the Feenstra division are `feenstra.py`'s (`current_from_ldos`, `iv_ratio`, `divide`). The one quantity the file forms itself is the barrier height (`barrier_heights`), which the simulator never needs as a number. `SAVE_PATH` in the bottom block (or `save_path`) saves the figure; otherwise it is only shown |
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
maps are then saved under the file of the new key, at RUN for a drag made with RUN off (their `_mag`
file too, while their mag channel is whole). The markers' spectra (Single Point, Map) are not
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
height. The limits: in Line and Path a topography's plot box at most twice one of that view's own
heatmaps; in Map at most 1.5 times an energy map's and at least an energy map's (a map no larger
than a topo panel, a topo panel no larger than 1.5 times a map), with 1.5 also the ratio aimed at
(a map about two thirds of a topo panel: a topo panel above it gives up room to the maps, which
wrap into it, unless the maps would then outgrow the topo panel, where the cap alone decides; the
`group_aim` of `Arrangement` in the package's `qtgui/controls.py`); in Single Point, Line and Path
the placement panel's plot box (the global topo's) never smaller than the size at which the atom
dots, as drawn on screen (every atom at every image that reaches into the panel's limits as
arranged, its crop or the box of the cells shown, dots that overlap, the cell's periodic images
among them, counted once), cover at most a third of it, whether `Atoms` is on or off (found by
bisection), the cap giving way where both cannot hold, and the window's minimum size following
from these limits; Map has no such floor, so its cap always holds and its window's minimum size is
the layout's own fixed one. Line's LDOS topo is
2:3, width to height, L/3 either side of the line by its length L
(the LDOS topo tiled and turned into the line's frame, with the line, its ends and the markers);
Path's is the LDOS topo in the path's rectangle, the smallest rectangle at any angle that holds
every sample point of the path (the minimum-area rectangle, over the directions of the convex
hull's edges), with a margin of 1/8 of its width added left and right and of its height top and
bottom, turned 90 degrees when it is wider than tall, so the panel is square or taller than wide,
with the path, its nodes and the markers (a straight path, every sample point on one line so that no
rectangle has an area, takes Line's 2:3 box centred on the path instead, drawn as Line's corridor
is); neither takes the crop or the turn below. The Line and
Path heatmaps are sized as squares, then fill their row's whole slot, stretched in x or y, never
taking room from another panel. The curve panels (the Tip Height profile and the spectrum with
its legend panel) take the room those leave, never less than a
readable size: in Line and Path the second row takes all the height the top row leaves, with no
empty band; in Map the spectrum, in the top row, has its plot box held equal to a topo panel's; in
Single Point the spectrum, its legend panel at its right, stands beside the topo sub-section at one
topo panel's size, the row as large as the two allow, and goes under the topo sub-section, filling
the bottom row, exactly when that one row would leave an empty band below it, and back as soon as
it no longer would. The panels drawn in the unit cell's frame, in two groups (the global topo; the
LDOS topo of Single Point and Map with Map's energy maps), are shown as drawn or turned 90 degrees
anticlockwise (x up, y to the left: a rotation, never a mirror), whichever fits the window better:
one choice for both groups while neither topo is cropped, and each group its own once its topo is;
that is decided when a view is built and again for a topo's group on its Crop / Uncrop, never on a
resize, and the console says which way and why. Line's and Path's LDOS topo and the heatmaps of
energy against position never turn this way. Everything on a
turned panel keeps its data coordinates: the cursors, the markers, the line and path endpoints, the
limits and the publication files (which show the turn as on screen).
