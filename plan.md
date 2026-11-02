# plan.md: the Qt conversion of every interactive window

The single source of truth for this update. A new session, or one whose context was compacted,
reads this file and `git log`, and continues at the first unchecked step. When every step is done
and committed, the last step deletes this file from the repository (it stays in the history).

## 0. Status and how to resume

- **Where the work stands**: sections 1 to 5 are done and on `main` (the conversion of every
  interactive window to the shared Qt layer `qtgui/`, verified; commits 76e35cc .. 07dcf10).
  Section 6 (the owner's shape rule): 6.1 to 6.6 done; 6.7 (this file's deletion in a commit of
  its own) is the last step.
- **Git**: develop on the branch `claude/festive-mayer-q2ig83`; after every commit push the branch
  and fast-forward `main` to it (the owner's instruction: the commits go to `main` and the history
  is recorded): `git push origin claude/festive-mayer-q2ig83 && git push origin HEAD:main`. Never
  force-push, never rewrite history; if `main` has moved, merge it into the branch first. No pull
  request unless the owner asks. Commit messages end with the session's attribution lines.
- **Environment** (a fresh container): `pip install "PySide6<6.12" pyqtgraph scipy matplotlib
  h5py pytest` and the system libraries `libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0
  libdbus-1-3 libfontconfig1 libglib2.0-0 libxcb-cursor0 libopengl0` (apt); pymatgen does not
  install here (its bibtexparser wheel fails) and is not needed by the checks (the simulator
  check gives the LOCPOT reader a cached `LOCPOT.npy`); cupy is absent (the simulator check
  registers numpy stand-ins for `cupy` and `cupyx.scipy.ndimage`). Everything runs offscreen with
  `QT_QPA_PLATFORM=offscreen`.
- **Verification commands** (from the repository root; every one must end clean):
  `QT_QPA_PLATFORM=offscreen python3 checks/check_simulator.py`, `checks/check_decomp.py`,
  `checks/check_matcher_plots.py`, `checks/check_misc_windows.py` (each prints "n of n checks
  passed"), and `cd LDOS_spread_decomp/mo_ldos_decomp && QT_QPA_PLATFORM=offscreen python3 -m
  pytest mo_ldos_projection/tests -q` (68 pass, 58 skip, the 3 pre-existing failures of
  section 5). Screenshots land in the session's scratch folder under `shots/`.
- **The guide** ("From a matplotlib-wired GUI to a real GUI") was uploaded by the owner; step 6.2
  puts the updated copy at the repository root as `matplotlib_to_real_gui.md`, the one copy from
  then on.
- **How the work was done**: the layer and the checks by the main session; the window
  conversions by four parallel forks of it (one per window family), each reviewed against the
  rules below, with every layer edit kept additive and listed.

Status marks: `[ ]` to do, `[~]` in progress, `[x]` done.

---------------------------------------------------------------------------------------------------

## 1. The request and the guide

The owner asked that every package of VASP_for_SPM switch from hard-coded matplotlib GUIs
(figures with hand-placed `add_axes([...])` widgets: `Slider`, `Button`, `CheckButtons`,
`RadioButtons`, `TextBox`, plus `mplcursors` hover labels and `mpl_connect` handlers) to the Qt
style of the guide "From a matplotlib-wired GUI to a real GUI" (the sm4_Browser conversion). The
guide's rules, in short:

1. automatic layout, nothing placed by hand; 2. everything shared is one function; 3. controls
are real Qt widgets; 4. windows never scroll; 5. one text-size rule; 6. short panel titles, the
details in the information panel; 7. spatial panels keep their x:y ratio, a row shares one
height; 8. number boxes instead of sliders (a slider survives only for scrubbing a list);
9. a window owns its state, controls only report; 10. comments name the rule, the README stays
current.  Section 7: efficient drawing (create once, update data; LUT images; coalesced motion;
draw on idle), computation on change, never on paint.  Section 9: copy `qtplot.py`,
`controls.py`, `widgets.py` into the project and import it as `plt`.

## 2. Working rules from the owner (hard constraints, every step)

- Implement exactly this plan; a deviation is stated in the delivery as an "additional change".
- Every delivery lists every function, gate, filter, constant or branch modified, removed or added.
- Never remove, weaken or bypass a gate, filter, threshold or validation to make inputs pass.
- Verify factual and technical claims against sources (library source or documentation) before
  stating them; otherwise say "I don't know".
- The README of every package touched and the root README are kept up to date.
- Layout code comments name the rule, e.g. `# automatic layout: the one rule for every window`.
- Commit messages end with the session's attribution lines; no model names in commits or code.

## 3. Decisions (made under stated assumptions; the owner was not available mid-task)

### 3.1 One shared layer at the repository root: `qtgui/`
- `qtgui/qtplot.py`, `qtgui/controls.py`, `qtgui/widgets.py`, `qtgui/mplexport.py` are the
  sm4_Browser layer (section 9 of the guide) with its sm4-specific parts removed (the sm4 control
  table and composite groups, the `.sm4` file-stamp rule, the scan-area navigator, the marker
  pairs) and what this suite's windows need added (section 3.4).
- Rule 2 (everything shared is one function) forbids a copy per package, so the layer lives once,
  at the root, and every window finds it by walking up from its own file to the folder that holds
  `qtgui/` (the suite's own "fixed sibling path" convention, `mo_ldos_projection/_imports.py`).
  The root README's rule "nothing in one package imports another" gains this one exception.
- `widgets.SPECS` is the one control table of the suite: every control of every window is named
  there once (kind, label, short label, font size, a box's width).

### 3.2 Scope: every window with widgets or canvas handlers, plus the one hand-placed colour bar
| file | old GUI | new |
|---|---|---|
| `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py` | 11 sliders, 2 buttons, 2 check groups, 1 radio, text boxes, pick/drag handlers, mplcursors, hand-placed `plt.axes` | the busiest window; the worked example |
| `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py` | σ slider + box, 'Send to decomp' buttons added onto the simulator's figure | the same controls in the simulator window's layout |
| `.../mo_ldos_projection/decomp_window.py` | checks, area box, metal check, Save, mplcursors, click | |
| `.../mo_ldos_projection/topo_decomp_window.py` | checks, Save, gridspec grid | |
| `.../mo_ldos_projection/global_decomposition.py` | σ slider + box, Save | |
| `LDOS_spread_decomp/mo_ldos_decomp/run_wfn_plane_slice.py` | γ slider + box | |
| `LDOS_spread_decomp/MOPmatcher_mem/matcher/spread_collapse.py` | checks, σ slider, click, mplcursors (two plotter classes) | |
| `.../matcher/spread_plot.py` | checks, σ slider, click, mplcursors | |
| `.../matcher/state_decomposition_plot.py` | checks, Norm check, σ slider, click, mplcursors | |
| `.../matcher/vacupot_plotter.py` | mplcursors, click | |
| `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/plot_dos_cursors_levels.py` | pick/click/key, mplcursors | |
| `PROCAR_tools/Band_structure_plot.py` | 'Orbital' check, mplcursors, click | |
| `convergence/forces.py` | window-start slider, drag/key/scroll handlers, a 3D panel | see 3.5 |
| `charge_density/lv_slice_w_atoms_2system.py` | a colour bar at a hand-placed `add_axes` | a colour-bar cell |

Out of scope (stated, not converted): the scripts that only draw static matplotlib figures with no
widget and no handler (`charge_density/*` except the file above, `dIdV_tip_sample/dIdV_sim.py`,
`convergence/plot_forces.py`, `PROCAR_tools/procar_filter_lorbit14.py`, `unfold_bands.py`,
`POSCAR_tools/Make_pyramid_any_number_sides.py`, the matcher's `combined_vacupot_plotter.py`,
`matcher_vacupot.py`, `wf_dpl_vacupot.py`, the `run_*.py` scripts' own figures). They are not
"GUIs" in the guide's sense (no hand-placed widget, no interaction) and the guide's conversion
target is the interactive window. `sim_decay_full.py` and `tier3_decay.py` are fully commented
out (zero live lines) and are left as they are.

### 3.3 Controls
- Every slider becomes a number box (`controls.Text` for a value, `controls.Count` for a whole
  number with limits) whose digits step with the arrow keys; the one slider kept is the window-
  start slider of `forces.py`, which scrubs through the list of ionic steps (rule 8).
- Groups shared by several windows are one class in `controls.py`: `SigmaBox` (the σ box of the
  spread, state-decomposition, global-decomposition windows and the simulator wrapper),
  `GammaBox` (the γ box of the simulator and the plane slice), `StepButtons`, `RunControls`.
- `mplcursors` is replaced by one class, `widgets.HoverLabels`: the hover label of the artist
  under the cursor, enabled by a left click on the panel and cleared and disabled by a right
  click, exactly the behaviour the windows implemented around `mplcursors.cursor`. The
  `mplcursors` dependency goes.
- Save buttons stay `controls.Button('Save data')`; the mode radio of the simulator a
  `controls.Radio`.

### 3.4 Additions to the layer the windows need (none of them in the sm4 layer)
- `Axes.tricontourf(x, y, z, levels, cmap, norm, vmin, vmax, zorder)`: the scattered points
  interpolated linearly (scipy `LinearNDInterpolator`, the piecewise-linear model matplotlib's
  tricontourf fills) onto a raster, banded by the levels and drawn as one image through a lookup
  table; `offsets=[(dx, dy), ...]` draws the same raster at several positions (the simulator's
  tiled cells: one raster, moved, never re-imaged).
- `Axes.vlines`, `Axes.hlines`, `Axes.add_line`, `Axes.set_facecolor`, `Axes.set(**kw)`,
  `Axes.get_lines`, `Axes.xaxis` / `yaxis` proxies with `set_major_formatter`; `Axes.scatter`
  returning a `PathCollection` (per-point colours, `set_offsets`, `set_facecolors`, pick with
  `event.ind`); `LineCollection` with per-segment colours (`array` + `cmap` + `norm`,
  `set_array`, `set_segments`, `set_color`); `Line2D` and `Patch` as legend proxies;
  `legend(handles=..., title=...)` and a second legend kept through `add_artist`;
  `pcolormesh` with `shading='auto' | 'nearest'` (edges from centres); `Annotation` in
  `'axes fraction'`; `LogNorm`, `SymLogNorm`; `ticker.FuncFormatter`; a `cm` namespace
  (`plt.cm.tab10(x)`, `ScalarMappable`); `fig.patch`; `fig.colorbar(m, ax=[axes])`;
  `get_subplotspec()` on the automatic grid's axes.
- `mplexport`: the publication export of the new artists (tricontourf, PathCollection, coloured
  LineCollection, Patch proxies, LogNorm / SymLogNorm, a norm with `to_matplotlib()`).

### 3.5 `convergence/forces.py`
Its 3D panel (matplotlib `Axes3D`: atoms as spheres, force arrows as cones, rotate / pan / zoom)
has no counterpart in the layer, which draws 2D panels in one pyqtgraph scene. The window is
converted (its Qt layout, the convergence panel with its draggable time marker, the window-start
slider in a box, the arrow keys) and the 3D panel is matplotlib's `Axes3D` on a matplotlib Qt
canvas placed as one cell of the window's layout, with every 3D interaction unchanged. This is
the one documented exception to "one scene per window" and "matplotlib off the screen".

### 3.6 Verification
- `checks/render_windows.py` (new, at the root): every converted window opened offscreen on
  synthetic inputs at its design size, half, double and taller than wide; no control text cut,
  no two controls overlapping, no control inside a panel box; screenshots written for review.
- The mo_ldos_projection pytest suite runs offscreen (`QT_QPA_PLATFORM=offscreen`); its
  widget tests follow the new controls.
- The mathematics is untouched: every window calls the same pipeline functions as before.

## 4. Steps

- [x] 4.1 Survey: inventory of every window, its controls, handlers and matplotlib calls.
- [x] 4.2 The layer: `qtgui/` built from the sm4 files, sm4 parts removed, section 3.4 added,
  `HoverLabels`, `SigmaBox`, `GammaBox`, the suite's control table; a demo window renders
  offscreen (commit 76e35cc).
- [x] 4.3 `checks/render_windows.py`: the render-and-measure helper (commit 76e35cc).
- [x] 4.4 The simulator and its wrapper (the worked example), `plot_dos_cursors_levels.py`.
- [x] 4.5 The decomposition windows: `decomp_window.py`, `topo_decomp_window.py`,
  `global_decomposition.py`, `run_wfn_plane_slice.py`; their tests (68 pass, 58 skip without the
  example data; 3 plane-slice tests fail exactly as they did at the base commit d9c0ca0: a
  `"power"` norm mode the module does not have, and an atom-tolerance expectation; their
  expectations were left as written, see the delivery report).
- [x] 4.6 The matcher plots: `spread_collapse.py`, `spread_plot.py`,
  `state_decomposition_plot.py`, `vacupot_plotter.py`.
- [x] 4.7 `Band_structure_plot.py`, `forces.py`, `lv_slice_w_atoms_2system.py`.
- [x] 4.8 Verification pass (3.6) on every window; the repeated-logic scan (section 5).
- [x] 4.9 READMEs (root and every package touched), this plan ticked, commit, push.

## 5. Findings of the verification pass (not changed; for the owner)

- **Pre-existing test failures** (`mo_ldos_projection/tests/test_wfn_plane_slice.py`), identical at
  the base commit d9c0ca0 and now: `test_run_end_to_end_saves_png[power]` and
  `test_run_multi_input_compare` ask for a norm mode `"power"` that `run_wfn_plane_slice._shared_norm`
  rejects (it knows `"log"` and `"linear"`); `test_multi_input_heterogeneous_defaults` expects one
  atom marker on the second panel while the module's default tolerance (3 Å) places five. The
  conversion fork had rewritten these expectations; they were put back as written, because
  changing what a test asserts about the module is outside the approved scope.
- **Repeated logic that predates the conversion** (the `ast` scan of section 11.6 of the guide):
  `_offset_name` in `decomp_window`, `global_decomposition`, `spread_collapse`, `spread_plot`;
  `_gauss_sum` and `_curve_grid` in `spread_collapse`, `spread_plot`, `state_decomposition_plot`
  (although `_imports.py` names `spread_plot`'s as the single implementation); `_curve_values` in
  `spread_collapse` and `spread_plot`; `_entity_label` in `global_decomposition` and `spread_plot`;
  `_ylabel` and `_on_chk` in `spread_plot` and `state_decomposition_plot`; `refresh` in
  `decomp_window` and `topo_decomp_window`; `_orbit_base`, `_lighten_color` and the
  element-by-index helper in `simulator` and `plot_dos_cursors_levels`. Each is one function in
  more than one place; making each one function touches the physics modules, so it was left for
  the owner's decision. The duplicates the conversion itself introduced were removed (the number
  boxes' submit logic in `controls`, the check function of the check scripts).
- **Layout under rule 7 in the simulator.** In the Single Point and Map views the Partitioned
  LDOS curve panel follows the height of the topography panels in its row (rule 7: curve panels
  follow the spatial panel), so it is shorter than in the matplotlib figure, where it filled the
  row's height, and the row leaves empty room above and below. Widening the topography columns
  or taking the curve panel out of the row is a layout decision for the owner.
- **The 3D panel of `forces.py`** keeps matplotlib's own text sizes (the documented exception).
- **A hover label** keeps its place on the panel, not its data position, across a pan or zoom
  until the next hit.
- **A long run-script title** in the combined matcher plot's vertical layout ("Bulk surface and
  adatom") overflows its narrow column; written "Bulk: surface and adatom" it would show its short
  name (rule 6). The run scripts' settings were not changed.

## 6. The shape rule (the owner's instruction after the first delivery)

The owner's words: the window must always adjust so that the row of panels is as large as, and
takes as much of the window as, possible; the full window sizes itself; the simulator's placement
panel (where the markers, line ends and path nodes are dragged) is too small in the Line, Path and
Map views and must be as large as possible, oriented by its shape (some unit cells are taller than
wide); the tip-height boxes go above the topography panel they belong to, never between panels;
the rule goes into the guide and is then applied to every converted window.  Also: the owner's
Tier 3 and Feenstra changes go into the two inactive reference files, fully commented.

- [x] 6.1 `tier3_decay.py` and `sim_decay_full.py` replaced by the owner's uploads, every line
  commented (`sim_decay_full.py` restores byte for byte); their README rows.
- [x] 6.2 The guide (`matplotlib_to_real_gui.md`, kept at the repository root from now on): rule 11,
  "panels know their shape; the window arranges itself", its layout tools and its pitfalls.
- [x] 6.3 The layer: a cell's height for its width from its axes' ratio (`PanelCell.height_for_width`,
  Qt's height-for-width); `PanelRow`, the row layout whose shaped items share one height and whose
  free items fill the rest; the automatic grid on `PanelRow`s (gridspecs without row spans);
  `Figure.fit_window` (the window sized to its panels' natural size at open and on request);
  `controls.AbovePanel`, `controls.PanelWithBox`, `controls.OrientedPanes` (the main panel's pane,
  oriented by its shape, the other panels in the second pane).  Checked by a prototype first
  (done: Qt's grid sizes a row from a square cell's width; the curve cell beside it fills that
  height; the row below takes the rest).  Found on the way and fixed in the layer: at a text-zoom
  change a row member was measured (through the row rule) before its own fonts changed and
  `Axes._decor_px` kept that stale room as the zoom's maximum (`Figure._apply_zoom` now clears
  every axes' memory after re-fonting them all); a panel in an aligned row (`align_row`, a colour
  bar beside) now counts the row's tallest title and deepest x axis in its shape
  (`PanelCell._decor`), as `Axes._px_rect` places it.  The render check: `empty_bands` (per shaped
  panel, against the row's room; the main pane of an `OrientedPanes` is exempt, its room follows
  the pane's cap), `window_fit`, and `fitted_to_panels` (a window of shaped panels that answered
  a requested size with its own is checked at the size it took: the text rule and the portrait
  flag).  Results: matcher 66/66, decomp 53/53, misc 64/64, pytest 68 pass / 58 skip / the 3
  pre-existing failures; the simulator reports empty bands in every view until 6.4 rebuilds it.
- [x] 6.4 The simulator's four views arranged by one rule, confirmed by the owner on 2026-11-02
  after the first pane layout was rejected (the heatmaps came out as small as a control box):
  **every plotted panel is a first-class citizen; the layout fills the window with plots, spatial
  panels keep their ratio, no plotted panel is ever shrunk to make room, and what is minimized is
  the blank space and the controls' footprint** (preference among plots: the spectral heatmaps of
  Line / Path over the Map view's constant-energy slices).  Concretely:
  - **The arrangement engine** (layer, `controls.Arrangement`): up to two shaped groups and one
    free widget; the candidates (the two groups in one row across the top with the free widget
    below; the first group at the left with the second over the free widget at its right; the
    first across the top with the second at the left of the free widget below; the two groups
    stacked at the left with the free widget at the right; the three side by side; the three
    stacked) are each sized at the window's size (a shaped group takes the largest size its ratio
    allows in its region; the free widget keeps a readable share, `free_share`) and the one with
    the least blank space is taken (hysteresis against the current one); chosen again on every
    resize, rebuild and shape change; `fit_delta` reports the blank at the window's edge for
    `Figure.fit_window`.  It replaces `OrientedPanes` for the simulator (OrientedPanes stays as the
    one-group case of the same rule).
  - **Groups per view.** Single Point: the placement topo (with its boxes above) and the spectrum.
    Line / Path: the placement topo; the heatmap row (Topo corridor with LDOS h above it in Line,
    the stripe with LDOS h above it in Path, the heatmaps, the colour bar) as a `RowWidget`; the
    Tip Height profile and the Partitioned LDOS as the free widget, side by side when their space
    is wide, stacked when tall (`controls.FlipBox`).  Map: the Global Topo and Map Topo pair (Topo
    h and LDOS h above them) as a `RowWidget`; the energy maps as a wrapping block
    (`controls.MapBlock`: partition by partition, the number of maps per row chosen for the
    largest maps that fit the block's region, the colour bar per partition or one for all at the
    right); the spectrum as the free widget.
  - **The crop of the placement topo** (the topo with the atom overlays where markers, line ends
    and path nodes are placed), as the sm4_Browser Line window crops its heatmap (line.py, the
    crop cursors): two draggable cursor lines per axis starting at the full extent of the drawn
    cells, the area outside each pair shaded grey, a text box riding with each cursor in a band
    above (x) and right of (y) the panel (`controls.RideBand`) showing its value and taking a
    typed one, 'Crop' applying the ranges between the cursors as the panel's limits (the cursors
    then at the new edges) and 'Uncrop' restoring the full extent; the boxes and buttons sit above
    the panel with the Topo h box (`controls.CropPanel`, a `PanelWithBox` with the two bands).  In
    Map the Map Topo takes the same ranges.  A crop changes the panel's shape, so the arrangement
    and the window follow it (tall crop: tall layout; wide crop: wide layout).  The crop stays
    across view changes.
  - **The window** opens on the screen's available area and trims its blank edge on every change.
  - **Checks** (`checks/check_simulator.py --shape`, about 90 s, the topographies pre-seeded in
    `checks/_cache/simulator/`): a tall and a wide cell in the four views: no empty band, the
    blank fraction of the panel area small, the heatmaps and maps not smaller than a set share of
    their region, the boxes above their panels; the crop: a dragged cursor updates its box, a typed
    box moves its cursor, Crop changes the limits and the arrangement, Uncrop restores.  The full
    suite once at the end.
  Done: the layer (`controls.Arrangement` with `explain()`, `FlipBox`, `MapBlock`, `CropPanel`,
  `Titled`, `RideBand` following its items, `_shaped_size`; `Axes.set_xlim` / `set_ylim` announce
  the shape, a one-axis crop counts, the row band counts the difference of the boxes above its
  members); the simulator's four views on the Arrangement with the crop of the placement topo
  (`_placement`, `_full_extent`, `_axis_extent`, `_reset_cursors`, `_cursor_value`, `_set_cursor`,
  `_on_cursor_box`, `_draw_cursors`, `_place_cursor_boxes`, `_crop_ranges`, `_apply_crop`,
  `_on_crop`, `_on_uncrop`, `_rearrange`; the spectrum no longer square in Line / Path); the
  shape check (`checks/check_simulator.py --shape`, 74 checks: the two cells in the four views,
  the arrangement's blank and scales, the boxes above their panels, the crop by drag, by typing,
  Crop, Uncrop); the guide's rule 11, tools, recipe step and pitfalls; the READMEs.
- [x] 6.5 Every other converted window on the rule (the automatic grid's rows are PanelRows; a
  window of spatial panels trims itself); the render check measures the empty band of every row
  and the window's fit: decomp 53/53, matcher 66/66, misc windows 64/64; the READMEs.
- [x] 6.6 Screenshots of the four views for a tall and a wide synthetic cell to the owner (sent in
  the session; `checks/check_simulator.py --shape` writes them under the shots folder); READMEs;
  commit, push (the branch and `main`).  Final results: simulator 158/158 (the 74 shape checks
  included), decomp 53/53, matcher 66/66, misc 64/64, pytest 68 pass / 58 skip / the 3
  pre-existing failures, the layer exercises clean.
- [ ] 6.7 When every step above is done and committed: delete `plan.md` from the repository in a
  commit of its own (the owner's instruction; the file stays in the git history), push the branch
  and `main`.
