# Part 3 (with the part 1 answers and the response document): the implementation brief

Sources, in order of precedence (a later one overrides an earlier one where they differ):
1. `handoff_part3_spectrum_decomp.md` (the plan; its rule 2 is dropped by ruling 9);
2. `PROGRESS.md` rulings 8 to 10;
3. `RESPONSE_PART1_PART3.md` (the owner's answers to `PART1_OPEN.md` and `PART3_QUESTIONS.md`, and his sections 2 to 4
   and 6), which stands over the plan where they differ;
4. this brief, which settles the pre-flight's points (`PART3_PREFLIGHT.md`) the sources above leave to mechanics;
5. `PART3_OPEN.md`: nine points only the owner can settle (K1 to K9). Until he answers, the code they touch is left
   as it is today, as each item says.

Rules for every agent: implement exactly this; where the code does not match, or a step needs a change nothing names,
stop and report it as a blocker instead of choosing. Read files and run `python -m py_compile` only (ruling 6). No
tests, no checks, no windows. Every agent runs on Opus 5.5 or higher. One agent edits a given file at a time.

## A. Part 1's open items (response section 1)

Every R of `PART1_OPEN.md` except P2:
- **P1:** the convention stands (qtgui converts its own size arguments). No change.
- **P2 (a), the Line view:** the shared row height, with the leftover height going to the 1D plots. Top row: the
  global topography, the Line Topo (2:3 box), the stripe and the heatmaps, their distance axes level. Its xy-spatial
  panels grow with the window keeping their ratio, up to the largest size the row's width allows; the top row's
  height is the height they then take. When the window is taller, the second row's 1D plots (the Tip Height and the
  LDOS plots) take all the remaining height. No empty band above, between or below the rows. If something in the
  code keeps this from working, stop and report it.
- **P3:** `_topo_cap` counts a heatmap as square (ratio 1.0) in the box it takes from the row's first panel, so the
  Line cap is today's 2h.
- **P4, P6, P7, P9, P12:** keep / no change.
- **P5:** (i) the cells' label room is kept under an arrangement epoch, bumped only by an arrangement; (ii) in the
  narrow branch of `Axes._cell_px`, the box shrinks to stay inside its cell; (iii) arrange again when the text zoom,
  control zoom or portrait setting changed since the last arrangement.
- **P8:** update the stale texts listed in P8 in the same terms, changing only the stale words (`qtgui/__init__.py`
  line 2: "this package's own copy of the Qt layer of VASP_for_SPM's interactive windows (PySide6 + pyqtgraph)").
- **P10:** stands, except `export_vector`, which is 720 dpi (section D3 below).
- **P11:** `plot_dos_cursors_levels.py` line 29's import in the two branches: `from .doscar_parser import ...` in the
  package branch, `from stm_from_dft.doscar_parser import ...` when run directly.
- **P13 (b):** new `mo_ldos_projection/_resolve.py` with `resolve_stm()`, `resolve_qt()` and `resolve_doscar_parser()`;
  module level imports nothing beyond `os` and `sys`; each resolver imports inside its function; the same
  vendored-copy-first lookup and `stm_from_dft_opt` path insert as today. `_imports.py` keeps its matcher imports,
  `resolve_run_match` and `resolve_blochstates_check`, and re-exports the three
  (`from ._resolve import resolve_stm, resolve_qt, resolve_doscar_parser`). `run_wfn_plane_slice.py` imports
  `resolve_qt` from `mo_ldos_projection._resolve`.

## B. The decomp (plan sections 2 to 4)

- **ldos_decomp.py** holds the state (level 0..3, active marker, active element) in a decomp object; the simulator
  gets three properties, `plot_level`, `active_marker_idx`, `active_element`, forwarding get and set to it (no
  validation, as today's plain attributes). `active_atom` goes.
- **Partitions** for levels 0..3 move from `_get_partitions` (which goes) into ldos_decomp; level 4's branch goes.
  Callers (`_update_all`'s map block and heatmaps, `_redraw_map_slice`) take them from ldos_decomp. The level-3
  column test is the code's own (cp.gradient twice on the element's atom-summed `dos_up_gpu`, all three maxima below
  1e-5), and its result is cached keyed on the element and the `dos_up_gpu` array object (`is`), so no GPU work runs
  during a drag (A8). Level 3's naming stays as the maps have it today until K1 is answered.
- **The spectrum's curves** per level come from ldos_decomp (data, colour, line style, line width, label, pickable),
  as the plan's table. Level 3: the element's colour, one line style per orbital (`linestyle_map` over `_orbit_base`),
  the lighter colour for `_up` (`_lighten_color`), line width 1.2, zorder 10, no picker (today's level-4 orbital
  style); the element's total faded behind (alpha 0.15, zorder 1, as today's level 3); `widgets.HoverLabels` on the
  orbital curves, the old HoverLabels removed on every spectrum build and a new one made only at level 3. Levels 0
  to 2 keep `picker=True` and their `pickradius=` (P9 accepted). The helpers `_orbit_base`, `_lighten_color`, the
  style map and `S` move into ldos_decomp (S as the plan says, though nothing then uses it).
  `vesta_label_map` and `_get_element_by_index_helper` stay where they are.
- **Every `ax_spec.legend(...)` and `add_artist(leg1)`** in `_draw_spectrum` go. `Axes.legend` and every other legend
  in the repository stay unchanged; mplexport keeps its legend export.
- **The redraw key** becomes (mode, level, the curves' labels, feenstra, the energy-marker count); colours are
  applied in place with `set_color` where they differ (2a's re-sort), so no clear. The y-limit block stays as it is.
- **Clicks:** one decision per click, in `_on_press`, from what is drawn at the press (before any redraw), with the
  top item by zorder, ties to the one drawn last (`_children` order). `_on_pick`'s spectrum branch only returns (so
  level-0 `marker_i` curves never reach `mark_dynamic`). Left on a curve: down one level with its option (levels 0 to
  2; nothing at 3). Left on a legend entry: the same. Left on neither: nothing. Right anywhere in the spectrum's plot
  box or anywhere in the legend panel (title, Back row, stack, scroll bar): back one level (levels 1 to 3; nothing at
  0). The scroll bar's own context menu is turned off so a right click on it goes back. Back button: the same as a
  right click. The two ends do nothing and do not redraw. A press with `dblclick=True` is ignored (a double click is
  one left click); the Back button ignores its double-click event. Middle click: nothing. While the toolbar's Pan or
  Zoom is on, the decomp clicks are ignored, as `_on_press` does today. One click changes the level at most once and
  redraws once. RUN off: as today until K7. The click rules live in ldos_decomp (press / back), behind one simulator
  entry point that both the clicks and the Back button call.
- **Map's energy markers** take part in the top test: drawn after the curves, an energy marker under a left click is
  the top item; at level 0 it starts its drag (as today, and with Pan / Zoom on as today), at levels 1 to 2 nothing.
  A right click ignores them and goes back. A drag starts only on a left, non-double press.
- **Level tests left in simulator.py** (the cluster label, the heatmap titles, the map ylabels, `plot_level >= 3`)
  stay; they read the properties.
- **Docstrings** in ldos_decomp.py start with "Machinery:" (ruling 5's tags).
- **The info line** (Q6): one window line, key `decomp`, through `fig.set_info_line`, giving the level's full word and
  the selection: level 0 "markers", level 1 "Partition (Au / Molecule), marker k", level 2 "Atom Types, marker k",
  level 3 "Orbitals of X, marker k". Set whenever the decomp state changes.
- **Not touched:** plot_dos_cursors_levels.py's levels (plan section 3).

## C. Line and Path markers (plan section 2a)

- **Sorted wherever a set is formed** (`__init__`, `run_interactive`, the Markers count box) and on every release in
  `_on_rel`: top to bottom (marker_ratios from largest to smallest), stable; ties are exactly equal ratios. A freshly
  formed set is coloured by index after the sort (index 0, the top marker, gets `m_colors[0]`).
- **Kept colours:** a per-marker colour list for Line and Path, permuted with the markers, kept across view changes
  (only `active_marker_idx` goes to 0). The Markers count box replaces every marker on an increase (today's
  placement, unchanged): all are new, sorted, then coloured by index. A decrease truncates to the first k (the bottom
  markers go once sorted); the rest keep their colours; `active_marker_idx` is left as today. With more than
  `len(m_colors)` markers, a marker with no free colour takes `m_colors[i % len(m_colors)]`.
- **The permutation** is applied together to `marker_ratios`, `active_marker_idx` (the same marker stays active),
  the colours and every per-marker cache (as the pre-flight lists them).
- **Every panel that draws a Line / Path marker** uses its kept colour: the placement (global) topo's marks, the LDOS
  topo panel (Line), the stripe, the heatmaps, the Tip Height plot, the spectrum and the legend panel. Artist i always
  shows marker i: each index's artists are recoloured in place and moved by `_place_line_markers`; the spectrum's
  level-0 curves likewise (`set_color`, `set_data`).
- **Redrawn once** for the new order, no rearrangement or rebuild. The legend recolours its samples in place when only
  colours change, and rebuilds only when its labels or count change.
- **ldos_decomp gets the markers' colours in index order** (the kept colours in Line / Path, `m_colors` by index in
  Single Point / Map) where the plan's shared data lists `m_colors`.
- **Single Point and Map** keep their index and colours by index. sim_wrapper unchanged until K3.
- **The info lines** (Q6): key `markers`: Single Point / Map "k (x, y) Å" per marker, Line / Path "k at d Å", the
  active one starred; key `line`: the Line ends with the length, or the Path nodes with the length.

## D. The legend panel (plan section 5; response Q1 to Q3)

- **Built in simulator.py** (next to `_placement`), from existing qtgui pieces plus the qtgui hooks named in E.
- **A composite:** a title cell holding a figure text (`fig.text(..., cell=)`, saved with the figure), the Back button
  (`controls.Button(fig, 'Back')`, default spec, gray `#c8c8c8` background), the stack (a qtplot axes, `axis('off')`,
  white face) and a Qt vertical scroll bar at the stack's right edge. The title and the Back button keep their places
  at level 0 (empty), so nothing moves between levels.
- **The stack:** each entry a sample (a Line2D in its curve's colour, line style and line width) and a label (a text,
  10 pt as today's legend); a clickable entry (levels 0 to 2) has a box: a PolyCollection quad filled `#f0f0f0` with a
  LineCollection outline `#c8c8c8` (`controls.FRAME_CSS`'s colours); level 3's entries are plain key rows; level 0
  entries are colour only, in marker index order. The title: 10 pt bold (qtplot's legend-title rule): "" at 0,
  "Parts", "Atoms", "Orbitals". Not entries: the faded totals and Map's energy markers.
- **Scrolling** (Q1 a): a QScrollBar configured as QScrollArea configures its own (range 0..content minus visible,
  page step = visible, single step 20), shown only while the entries don't fit; the wheel over the stack is forwarded
  to it; both move the stack's y range. Scroll position: back to the top on a level change, kept (clamped) when the
  same level's list is redrawn.
- **Width** (Q3 a): one fixed width, the largest entry any level can show for the loaded data (from the same
  ldos_decomp code that makes the labels), plus the scroll bar's strip, always reserved (empty when the bar is
  hidden). The width changes only with the text zoom.
- **Where:** right of the spectrum, the panel spanning the spectrum's plot box (top to bottom of its box), in Single
  Point, Line and Path; in Line and Path the spectrum and the legend form one sub-sub-section inside the 1D
  sub-section (FlipBox stretches 25:10 unchanged). Smallest height: the title, Back and one entry row. Map waits on K2.
- **Clicks:** an entry is found from where the click lands among the boxes drawn at the press; the lookup returns the
  entry or None; the rules are B's.
- **Title width:** not measured (Q2 a; the owner checks it).
- **CSV:** the stack axes is marked so `export_csv` skips it (E).
- **In the saves:** the PDF, EPS and vector SVG hold the full list (the panel extends down past the window as far as
  its list needs) and leave out the Back button and the scroll bar with their places empty; the full info panel
  starts below the canvas, moved below the legend panel's bottom when it reaches past the canvas. The Qt-drawn SVG and
  the window images show both panels exactly as on screen, the Back button and the bar included.
- **The info line:** the legend's full words are B's `decomp` line.
- **Pan / Zoom over the stack:** as it is until K8.

## E. The info panel and the saves in every qtgui window (plan section 6; response sections 2 to 4, Q4 to Q8)

The owner's rule (response section 2): where sm4_browser's code covers something settled in parts 1 to 3, the settled
thing stands and that part of the browser's code is adapted; everything else of section 6's pieces is taken as it is.

**In the package copy (stm_from_dft/qtgui; then copied byte for byte over MOPmatcher_mem/matcher/qtgui):**
- **qtplot.py:** sm4_browser's info panel in `FigureManager.__init__` (wrap at width, shown once it has text), its
  height kept at part 1's `QFontMetricsF(self.info.font()).lineSpacing() * 4.2` (in `__init__` and the text zoom) and
  its font `qfont(7.5)` with the 10 pt floor; `Figure.info_text` (panel-title details, the window's own lines, an
  empty line, "settings", `controls.settings_text`), `_settings_changed` (the coalesced push), the window hook
  `sm4_settings_changed` (name kept, Q8); `NavigationToolbar.SAVE_FORMATS` as sm4_browser's, with the Qt-drawn SVG
  labelled "SVG, the whole window as Qt draws it, controls included, as shown (*.svg)" (Q7) and its texts (the
  SAVE_FORMATS comment, savefig's and export_vector's docstrings, the module docstring) worded the same way; the
  dialog always offers the full list (no matplotlib filter); `Figure.export_info`, `export_info_panel`,
  `export_root`, `_export_rect`, `window_image` (`WINDOW_IMAGE_DPI = 720`, `k = dpi / DPI` as sm4_browser), and
  savefig's 'vector' (always mplexport, no fallback), 'qt-svg' and 'image' paths and CSV. `import matplotlib` at
  module level (response section 4). `export_vector` only for the Qt-drawn SVG: the window exactly as on screen,
  nothing appended below it, at 720 dpi (`gen.setResolution`), the same physical size as the window images. Keep the
  copy's local `from . import mplexport` imports inside the functions that use it (no module-end binding). The copy's
  `to_cmap` stays as it is. Do not take sm4_browser's texts that are false here (its `align_column` clause; its
  "full info panel" for the Qt-drawn SVG).
- **The legend panel's hooks:** a generic export hook on an axes giving its rectangle for the vector saves (default its
  on-screen `_px_rect()`); mplexport reads it for the axes' place and limits, makes the canvas part
  `max(H, lowest export bottom)` tall and puts the info block there. `export_csv` skips an axes marked `csv = False`.
- **controls.py:** `settings_text(root)` and `plt._settings_text`, `Control._changed` and its calls, and each kind's
  `setting()` as sm4_browser has them; `Slider._changed` (its valueChanged slot) renamed `_moved`, as sm4_browser did,
  so `Control._changed` is not shadowed.
- **mplexport.py:** sm4_browser's `_to_matplotlib` structure (the canvas at its on-screen size, every visible figure
  text, the full info panel below it, wrapped as on screen, nothing cropped), `_pos(ax, mfig)`, `_info_lines`,
  `_info_geometry`, `_info_block`, `export`, `to_matplotlib`, the docstring; with the settled things standing: the
  figure size through `pt2px` (`figsize = (W / pt2px(72), (H + H_info) / pt2px(72))`), `Figure(..., dpi=q.DPI)`
  (P10), the info block's text size through `pt2px`, and every text at its on-screen size with the 10 pt floor
  (one helper). Because the figure size and the dpi differ, `_pos` and `_info_block` take their fractions from the
  logical pixels W and H + H_info (not from `mfig.get_figwidth() * mfig.dpi`), so every panel and line is in its
  place. Keep every copy-only feature (the rotated-axes swap, LogNorm / SymLogNorm and `norm.to_matplotlib`, legend
  proxies, extra legends and legend titles, TriContourSet, PathCollection, per-segment LineCollection colours,
  ContourSet norm and alpha); do not take sm4_browser's side axes or Circle. Background handling: as it is here until
  K4.
- **Left as they are:** `Figure.at_design_size`, `_size_override` and their reads (unused after the port; listed in
  the report), the checks in `checks/`.

**In the root qtgui/:** the same pieces, with its own DPI sizes (sm4_browser's conversions, as the root has no part 1
size API), its absolute import style, 720 dpi for the window images and the Qt-drawn SVG, matplotlib required.

**Ruling 8 / Q6 in the windows:** settings made with the mouse go in as the window's own lines (`set_info_line`),
updated whenever they change. The simulator's: B's `decomp`, C's `markers` and `line`, `map energies` (Map's snapped
energies), `crop` (the crop rectangle, or "none"), `contour level` (Map's z_frac, set by dragging the level line, a
mouse-made setting by the same rule). The RUN button becomes stateful as sm4_browser's map.py does (`stateful = True`,
`setting_label`), so the settings list shows it. The other windows hold no mouse-made settings outside their controls
and titles: no lines. plot_dos_cursors_levels.py is not edited (plan section 3). Pan / Zoom and forces.py's 3D view:
not listed until K6. Two scripted saves: left until K5.

## F. The two colour-map sets (response section 6)

As the response document gives it: the two sets defined in simulator.py (default 1: today's maps, the topo map black /
firebrick / yellow, `MAP_CMAP`, `'nipy_spectral'`, `'bwr'`, `'hot'`; default 2: `'CET-L3'`, `'CET-L6'`, `'CET-R1'`
shared by Line and Path, `'CET-D1A'`, `'CET-L19'`); a choice in the left column between "default 1" and "default 2"
that recolours every panel in place (no rearrangement, no recomputation; if a panel's map cannot change in place,
stop and report); the constructor's `cmap_topo` and `cmap_map` replaced by `cmap_set` (default `'default 1'`);
`run_DFT_STM.py`'s colour-map lines only (`cmap_set='default 1',`, one comment naming the two sets, the
`LinearSegmentedColormap` import removed); `sim_wrapper.py`'s `cmap_topo=None` becomes `cmap_set='default 1'` passed
on, its own black / firebrick / yellow map removed (the one exception to rule 4). Unchanged: the contour colours, the
marker colours, the decomp curves' colours, every other window's maps.

## G. Not in this pass

Part 2's open items O1 to O18 (`PART2_OPEN.md`) and K1 to K9 wait for the owner.

## H. Finish

`python -m py_compile` every changed .py file; the two package qtgui copies byte-identical; the per-file lists of
plan rule 5 (approved changes; additional changes, which must be empty or reported before the commit).
