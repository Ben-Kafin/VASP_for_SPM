# Part 3: the per-file change list (rule 5)

Every function, constant, branch and text changed, added or removed by part 3 (with the part 1 answers and the
response document) and by the addendum's module, from the implementing agents' and fixers' reports (base adf1e25).
Fixer entries are marked (review fix). The brief section each implements is in brackets.

**Additional changes (rule 5's second list):** the part 3 workflow reported none. The module writer reported five
(listed under `cmap_shading.py` below and in `PROGRESS.md` ruling 11 as Claude's readings): `make_cmap`, `_make_control`,
the CuPy check through `sys.modules`, the readings in the maths, and the step vectors as the geometry input.

The matcher's qtgui copy (`LDOS_spread_decomp/MOPmatcher_mem/matcher/qtgui/`) is the package copy, copied byte for byte.

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/ldos_decomp.py`

- **added** `module ldos_decomp (docstring, imports)`: New file. Module docstring starts with 'Machinery:' and gives the levels, the click rules and what the module holds. [B: ldos_decomp.py; B: Docstrings]
- **added** `TITLES`: ('', 'Parts', 'Atoms', 'Orbitals'): the legend title at each level, '' at level 0. [B: legend entries and titles]
- **added** `LAST_LEVEL`: 3. A left click at this level does nothing (Decomp.down). [B: Clicks]
- **added** `PARTS`: ('Au', 'Molecule'). Level 1's labels, shared by partitions, curves and all_labels. [B: Partitions; B: legend width labels]
- **added** `S`: 0.25, the faded-colour factor, moved here and unused, as the brief says. [B: spectrum's curves (helpers move)]
- **added** `Curve (namedtuple)`: Fields y color linestyle linewidth alpha zorder label picker pickradius clickable option entry hover. [B: spectrum's curves]
- **added** `Entry (namedtuple)`: Fields label color linestyle linewidth clickable option. [B: legend entries (one entry type)]
- **added** `_orbit_base(orb)`: Moved unchanged from _draw_spectrum's local helper: strips '_up' / '_down'. [B: helpers move]
- **added** `_lighten_color(color, amount=0.3)`: Moved unchanged from _draw_spectrum's local helper (colorsys HLS, plt.to_rgb). [B: helpers move]
- **added** `_linestyle_map(orbitals)`: The style map, moved: the same unique_bases / styles / dict(zip(...)) code over every orbital name, returned as a dict. [B: helpers move (the style map)]
- **added** `_orbital_label(element, orb)`: f"{element} {orb}", the maps' level-3 naming (left as K1 has it). [B: Partitions (level 3 naming, K1); B: legend width function]
- **added** `all_labels(atomtypes, orbitals)`: Every label any level can show, in order and each once: PARTS, then the atom types, then '<t> <orb>' for every type and every orbital name (shown or not). [B: a function giving every label any level can show]
- **added** `top_artist(artists)`: Returns the top artist (highest get_zorder(); [B: Clicks (top item by zorder, ties to the one drawn last)]
- **added** `hover_labels(ax, lines, curves, old)`: Removes old (a HoverLabels or None). Makes widgets.HoverLabels(ax, [lines whose curve.hover], enabled=True, arm_by_click=False) only when such curves exist (level 3). [B: spectrum's curves (HoverLabels removed on every build, new only at level 3)]
- **added** `class Decomp, __init__`: Holds the state: level 0, marker 0, element None, plus the level-3 column cache (_columns_key, _columns). [B: ldos_decomp state]
- **added** `Decomp.orbital_columns(atomtypes, atomnums, orbitals, dos_up)`: The code's own level-3 column test, transcribed: cp.sum over the element's atoms of dos_up[e_idx,:,col], cp.gradient twice, the column skipped only if all three maxima are < 1e-5. [B: Partitions (level-3 column test, cached)]
- **added** `Decomp.partitions(f_ldos, atomtypes, atomnums, orbitals, dos_up)`: _get_partitions' levels 0 to 3 transcribed: 'Total'; [B: Partitions for levels 0..3]
- **added** `Decomp.curves(active, marker_spectra, marker_colors, atomtypes, atomnums, orbitals, dos_up, type_colors, energies, integral=None)`: [Curve] per the plan's table. L0: per marker, lw 2.5, 'marker_i', picker True, pickradius 5, Entry('', colour, '-', 2.5, True, i). [B: The spectrum's curves per level]
- **added** `Decomp.legend(curves)`: Returns (TITLES[level], [c.entry for c in curves if c.entry], level > 0). [B: legend entries and titles]
- **added** `Decomp.down(option)`: Level 0 sets marker = option, level 2 sets element = option, then level += 1 and returns True. [B: Clicks (left: down one level with its option)]
- **added** `Decomp.back()`: Levels 1 to 3: level -= 1, True. Level 0: False, no change. [B: Clicks (right / Back: back one level)]
- **added** `Decomp.press(button, dblclick, item)`: The click rules. dblclick gives False. plt.MouseButton.RIGHT calls back(). [B: Clicks (rules live in ldos_decomp: press / back)]
- **added** `Decomp.info()`: 'markers'; 'Partition (Au / Molecule), marker k'; [B: The info line]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py`

- **added** `import `from . import ldos_decomp``: Placed after `from . import quadrature`. [B: ldos_decomp.py and its use by simulator.py]
- **changed** `Interactive_STM_Simulator.__init__ (decomp state)`: The three lines `active_marker_idx = 0`, `plot_level = 0`, `active_element, active_atom = None, None` are replaced by `self.decomp = ldos_decomp.Decomp()` with a two-line comment. [B: state; active_atom goes]
- **added** `properties plot_level, active_marker_idx, active_element (getters and setters)`: Each gets and sets self.decomp.level / .marker / .element with no validation. [B: three forwarding properties; B: info line (set whenever the state changes)]
- **removed** `_get_partitions`: Deleted, level 4 branch included. Callers use self.decomp.partitions. [B: Partitions (_get_partitions removed)]
- **added** `_make_controls: initial decomp line`: `self._decomp_changed()`, with a one-line comment, after the four mpl_connect calls, so each new window's info panel has the 'decomp' line. [B: The info line]
- **added** `_clear_panels: self._spec_curves = []`: Resets the Curve records that run parallel to _spec_lines, next to the _spec_* reset line. [B: spectrum's curves / Clicks (curve to option mapping)]
- **changed** `_update_all: partitions`: `self._get_partitions(f_ldos_raw)` becomes `self.decomp.partitions(f_ldos_raw, self.atomtypes, self.atomnums, self.orbitals, self.dos_up_gpu)`. [B: Partitions (callers)]
- **changed** `_redraw_map_slice: partitions`: Same replacement as in _update_all. [B: Partitions (callers)]
- **changed** `_draw_spectrum: marker_spectra`: Line/Path: [f_ldos_raw[int(r*(npts-1))] for r in marker_ratios], or None when f_ldos_raw is None. [B: The spectrum's curves]
- **changed** `_draw_spectrum: curves from ldos_decomp`: marker_colors = m_colors[i % len] by index; [B: The spectrum's curves per level come from ldos_decomp]
- **changed** `_draw_spectrum: redraw key`: Now (mode, plot_level, tuple(curve labels), bool(feenstra), Map energy-marker count). [B: The redraw key]
- **changed** `_draw_spectrum: build path`: After ax_spec.clear(), each Curve is plotted with color, lw, linestyle, alpha, zorder, label and picker (pickradius when set, so levels 0 to 2 keep theirs). [B: spectrum's curves; HoverLabels rule]
- **changed** `_draw_spectrum: in-place path`: zip(_spec_lines, curves) calls set_data, and set_color only where ln.get_color() != c.color. [B: The redraw key (colours in place with set_color)]
- **removed** `_draw_spectrum: locals and level branches`: Removed: nested _curve (with its docstring), `drawn`, _orbit_base, _lighten_color, unique_bases / styles / linestyle_map, S, active_ldos_norm, the level 0 to 4 branches (old per-atom level 3 and level 4 deleted), the level-4 HoverLabels code, all five ax_spec.legend(...) calls and add_artist(leg1). [B: Every ax_spec.legend(...) and add_artist(leg1) go; helpers move; old levels 3/4 go]
- **changed** `_on_pick: spectrum branch`: The body is now only `return`, with a one-line comment. [B: Clicks (_on_pick's spectrum branch only returns)]
- **changed** `_on_press`: For inaxes == ax_spec: hits = pickable lines with contains(event); [B: Clicks (one decision in _on_press); Map's energy markers]
- **added** `_decomp_click(button, dblclick=False, item=None)`: The one simulator entry point for decomp clicks: if self.decomp.press(...) returns True, call _decomp_changed() and then _update_all() once. [B: one simulator entry point (spectrum, legend, Back)]
- **added** `_decomp_changed()`: When self.fig is not None: self.fig.set_info_line('decomp', f"decomp: {self.decomp.info()}"). [B: The info line (key decomp, set_info_line)]
- **added** `Interactive_STM_Simulator.line_marker_colors (attribute)`: Each Line / Path marker's kept colour, a list of colour strings index-aligned with marker_ratios. [C: Kept colours]
- **changed** `Interactive_STM_Simulator.__init__`: After `self.marker_ratios = [0.25, 0.75]`, a two-line comment and a call to `self._form_marker_set()`. [C: Sorted wherever a set is formed; a freshly formed set is coloured by index]
- **changed** `Interactive_STM_Simulator.run_interactive (marker_positions branch)`: After marker_ratios is formed (by projection or the linspace fallback), a call to `self._form_marker_set()`. [C: Sorted wherever a set is formed]
- **changed** `Interactive_STM_Simulator._on_ui_change (Markers count box, Line / Path)`: On an increase, markers are placed as today (linspace 0.1..0.9), then `_form_marker_set()` sorts and colours them by index. [C: Kept colours (the Markers count box)]
- **changed** `Interactive_STM_Simulator._update_all (Line and Path marker blocks)`: `self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])` becomes `self.marks.set_facecolors(self.line_marker_colors)` in both the Line and the Path branch. [C: Every panel that draws a Line / Path marker (the placement topo's marks)]
- **changed** `Interactive_STM_Simulator._update_all (after the marker blocks)`: Added `self._markers_changed()` after the marker offsets and colours are set and before the RUN test, so every _update_all refreshes the markers and line info lines. [C: The info lines]
- **changed** `Interactive_STM_Simulator._draw_spectrum (marker_colors)`: marker_colors passed to Decomp.curves is now `self.line_marker_colors` in Line / Path, and `[m_colors[i % len] for i in range(n_marks)]` in Single Point / Map, as before. [C: ldos_decomp gets the markers' colours in index order; the spectrum's level-0 curves (set_color, set_data)]
- **changed** `Interactive_STM_Simulator._make_line_markers`: Each marker's counterparts (Line Topo scatter, Tip Height axvline, heatmap and stripe axhlines) now take `self.line_marker_colors[i]` instead of `self.m_colors[i % len(self.m_colors)]`. [C: Every panel that draws a Line / Path marker]
- **changed** `Interactive_STM_Simulator._move_line_markers`: Added `self._markers_changed()` before the draw_idle. [C: The info lines; Redrawn once]
- **added** `Interactive_STM_Simulator._line_marker_order(ratios) (staticmethod)`: Returns the indices of ratios from the largest to the smallest ratio, top to bottom on the stripe: `sorted(range(n), key=lambda i: -float(ratios[i]))`. [C: Sorted ... top to bottom (marker_ratios from largest to smallest), stable; ties are exactly equal ratios]
- **added** `Interactive_STM_Simulator._form_marker_set()`: Sorts self.marker_ratios, as just formed, top to bottom, then sets line_marker_colors by index: `m_colors[i % len(m_colors)]`. [C: Sorted wherever a set is formed; a freshly formed set is coloured by index; more than len(m_colors) markers]
- **added** `Interactive_STM_Simulator._resort_line_markers()`: On a Line / Path marker release it computes the order and returns at once if that is the identity. [C: The permutation; Every panel ... artist i shows marker i; Redrawn once]
- **added** `Interactive_STM_Simulator._markers_changed()`: Sets the window's info lines 'markers' and 'line' (fig.set_info_line) from the live state, once fig exists. [C: The info lines (Q6): keys markers and line]
- **changed** `Interactive_STM_Simulator._on_pick (mark_dynamic branch)`: Added `self._markers_changed()` before the draw_idle, so the star follows the new active marker when a counterpart is pressed. [C: The info lines (the active one starred)]
- **changed** `Interactive_STM_Simulator._on_motion (shared tail)`: Added `self._markers_changed()` before `self._drag_moved = True` at the tail that line-end, path-node and Single Point / Map marker drags reach. [C: The info lines (updated whenever they change)]
- **changed** `Interactive_STM_Simulator._on_rel (new elif branch)`: A new elif, after the existing end / path-node / Single Point-Map branch and leaving the energy-marker re-sort branch untouched: `active_obj is not None and mode in ('Line', 'Path') and active_obj[0] in ('mark_map', 'mark_dynamic')` calls `self._resort_line_markers()`, with a comment. [C: Sorted ... on every release in _on_rel]
- **changed** `import line `from .qtgui.qtplot import QtWidgets, QtCore` (line 41)`: QtGui added. QFontMetricsF needs it for the legend's row height and width. [D (Width; The stack)]
- **added** `class _LegendEvents(QtCore.QObject) (module level, line 948), with __init__(sim) and eventFilter(obj, ev)`: Event filter on the Back button and the scroll bar. [B (Clicks: right on the Back row / scroll bar goes back; Back ignores its double-click event)]
- **added** `_make_controls: self._btn_back, self._legend_events, self._legend_rezoom_pending`: Back button made once: controls.Button(fig, 'Back'), default spec. [D (A composite: the Back button, gray; keeps its place at level 0); B (Back button: the same as a right click)]
- **added** `_make_controls: mpl_connect('scroll_event', self._on_scroll)`: The wheel connection for the legend stack. [D (Scrolling: the wheel over the stack is forwarded to the bar)]
- **added** `class constants LEGEND_SAMPLE_PX=26, LEGEND_PAD_PX=4, LEGEND_GAP_PX=4, LEGEND_ROW_GAP_PX=2, LEGEND_BOX_FILL='#f0f0f0', LEGEND_BOX_EDGE='#c8c8c8', LEGEND_BOX_LW=0.72, LEGEND_BACK_CSS`: Legend geometry, in pixels at the old 100 DPI, converted with /(1000/72)*em_px()*font_zoom where used. [D (The stack: box #f0f0f0 / #c8c8c8; Back #c8c8c8)]
- **added** `_spectrum_and_legend(self, spec_cell)`: Builds the composite and returns the sub-section widget controls.widget_of(controls.row(spec_cell, panel)). [D (Built in simulator.py next to _placement; A composite; Where; CSV; In the saves)]
- **added** `_legend_size(self)`: Runs once per text zoom. Sets the row height: label font height + 2 document margins + 2 pads + the row gap. [D (Width, Q3 a; Smallest height; title 10 pt bold, labels 10 pt)]
- **added** `_legend_draw(self, entries, to_top)`: ax.clear(), then each entry in rows k*row_h..(k+1)*row_h, in pixel data units, y downward. [D (The stack; Scroll position)]
- **added** `_update_legend(self)`: Called at the end of every _draw_spectrum. [D (The stack, live); C (the legend recolours its samples in place when only colours change, rebuilds when labels or count change)]
- **added** `_legend_view(self)`: Configures the bar as QScrollArea does. Range 0..ceil(content - visible), page step = visible, single step 20, value clamped, signals blocked. [D (Scrolling, Q1 a)]
- **added** `_legend_scrolled(self, v)`: The bar's valueChanged slot: ax.set_ylim(v + h, v). [D (Scrolling: both move the stack's y range)]
- **added** `_legend_placed(self)`: The stack's on_placed callback. Sets the view (_legend_view). [D (Width changes only with the text zoom)]
- **added** `_legend_rezoom(self)`: At a new text zoom: _legend_size(), then _legend_draw(self._legend_entries, False), scroll kept. [D (Width changes only with the text zoom)]
- **added** `_legend_follow(self)`: ax_spec's on_placed callback. Sets the composite's top and bottom contents margins from ax_spec's plot box inside its cell, as controls.UnderPanel._follow does, so the panel spans the spectrum's plot box. [D (Where: the panel spanning the spectrum's plot box)]
- **added** `_legend_extent(self)`: The stack axes' export_extent. Returns (content_h_px, (0.0, max(on-screen h, content_h_px))): the full list from the top, one data unit per pixel. [D (In the saves); E (the legend panel's hooks)]
- **added** `_in_legend(self, event)`: True when a canvas press is on the stack axes, or inside the panel's rectangle: from the title cell's top to the stack cell's bottom, across the panel's width. [B (Right anywhere in the legend panel: title, Back row, stack, scroll bar)]
- **added** `_legend_entry_at(self, event)`: The Entry whose drawn box holds (xdata, ydata) of a press on the stack, else None. [D (Clicks: the lookup returns the entry or None)]
- **added** `_legend_back(self)`: Called for a right press on the Back button or the bar (through _LegendEvents). [B (Right in the legend panel: back; Pan / Zoom ignored)]
- **added** `_on_scroll(self, event)`: When the wheel is over the stack axes and |angleDelta.y| >= |angleDelta.x|, QtCore.QCoreApplication.sendEvent(bar, event.guiEvent), as QAbstractScrollArea::wheelEvent does. [D (Scrolling: the wheel over the stack is forwarded to it)]
- **changed** `_kept_widgets(self)`: self._btn_back.qt added to the tuple, so the Back button is detached, not deleted, on a view change. [D (A composite: the Back button, made once and kept)]
- **changed** `_clear_panels(self)`: After the axes are removed: the legend title (a figure text) is removed, and every legend attribute is reset (_legend_ax/_title/_bar/_panel, _legend_title_cell/_stack_cell, _spec_sub, _legend_drawn/_samples/_entries, _legend_key/_state/_zoom/_draw_zoom, _legend_content_h/_h/_w). [D (A composite; per-view build)]
- **changed** `_build_ui, Single Point branch`: free = self._spectrum_and_legend(spec_cell) instead of spec_cell. [D (Where: right of the spectrum in Single Point)]
- **changed** `_build_ui, Line / Path branch`: controls.FlipBox((prof_cell, 25), (self._spectrum_and_legend(spec_cell), 10), spacing=10). [D (Where: Line and Path, one sub-sub-section, FlipBox stretches unchanged)]
- **changed** `_curves_readable(self)`: With a legend (Single Point, Line, Path), the readable width adds _legend_panel.minimumWidth() plus the row's spacing, after _legend_size() at the zoom. [D (Where; Width), the pre-flight's L-6]
- **changed** `_draw_spectrum(self, ...)`: self._update_legend() added after self._spec_key = key, so the legend follows every build and in-place draw (drags, the re-sort, mark_dynamic). [D (The stack: entries live); C (the legend recolours in place)]
- **changed** `_on_press(self, event)`: New elif branch after the ax_spec branch: when _in_legend(event) and toolbar.mode == '', calls self._decomp_click(event.button, event.dblclick, self._legend_entry_at(event)). [B (Clicks: left on a legend entry goes down, right in the panel goes back, Pan / Zoom ignored); D (Clicks)]
- **changed** `MAP_CMAP (its comment)`: Only the stale words changed: 'the default of the constructor's cmap_map' became 'default 1's Map entry (CMAP_SETS)'. [F (MAP_CMAP stays as default 1's Map entry; cmap_map goes)]
- **added** `CMAP_SETS (module constant and its comment)`: {'default 1': {'topo': LinearSegmentedColormap.from_list('t', ['black', 'firebrick', 'yellow']), 'map': MAP_CMAP, 'heatmap': 'nipy_spectral', 'mag': 'bwr', '|m|': 'hot'}, 'default 2': {'topo': 'CET-L3', 'map': 'CET-L6', 'heatmap': 'CET-R1', 'mag': 'CET-D1A', '|m|': 'CET-L19'}}. [F (the two sets defined in simulator.py)]
- **changed** `Interactive_STM_Simulator.__init__ signature`: (filepath, erange, ldos_height, cmap_topo, unit_cell_num=1, cmap_map=MAP_CMAP) became (filepath, erange, ldos_height, cmap_set='default 1', unit_cell_num=1). [F (cmap_topo and cmap_map replaced by cmap_set, default 'default 1')]
- **removed** `Interactive_STM_Simulator.cmap_topo / cmap_map attributes`: Removed self.cmap_topo, which was assigned in the erange/ldos_height tuple line, and self.cmap_map = cmap_map. [F]
- **added** `Interactive_STM_Simulator.cmap_set attribute`: self.cmap_set = cmap_set. This is the name of the set shown now, read by _cmap and changed by _on_cmap_set. [F]
- **added** `_make_controls: self.ui_cmap_set`: controls.Radio(fig, 'cmap set', value=self.cmap_set, labels=list(CMAP_SETS), title='cmap set'), with on_change(self._on_cmap_set) and a one-line comment. [F (a choice in the left column between 'default 1' and 'default 2')]
- **changed** `_make_controls: self._left_column`: controls.box(self.ui_cmap_set) is inserted directly under the view radio's box. [F (in the left column)]
- **changed** `_draw_topo_cbar`: The ramp's imshow uses cmap=self._cmap('topo') instead of self.cmap_topo. [F (topo role: the topo colour bar)]
- **changed** `_draw_map_slice`: The three tricontourf branches use cmap=self._cmap('mag'), self._cmap('|m|') and self._cmap('map') instead of 'bwr', 'hot' and self.cmap_map. [F (Map energy maps; Mag signed and |m|)]
- **changed** `_map_cmap`: It returns self._cmap('mag') / self._cmap('|m|') / self._cmap('map') in the same branch order. [F]
- **added** `_cmap(self, role)`: Machinery: returns CMAP_SETS[self.cmap_set][role] for role 'topo', 'map', 'heatmap', 'mag' or '|m|'. [F]
- **added** `_on_cmap_set(self, label)`: Handler for the radio; it runs whether RUN is on or off. [F (recolours every panel in place, no rearrangement, no recomputation)]
- **changed** `_update_all (full_refresh block and the Map needs_topo branch)`: The three topo tricontourf calls (t_ax, ax_map in Map, ax_map after the Map Topo is recomputed) use cmap=self._cmap('topo') instead of self.cmap_topo. [F (topo role: global topo, Map Topo)]
- **changed** `_fill_heat_row`: The Line Topo's tricontourf and the stripe's LineCollection use cmap=self._cmap('topo'). [F (topo role: Line Topo, stripe; Line and Path heatmaps one map; Mag in the heatmaps)]
- **changed** `Interactive_STM_Simulator._make_controls (the RUN button)`: Right after controls.make(fig, 'RUN') and before set_text: self.btn_run.stateful = True and self.btn_run.setting_label = 'RUN button', with a two-line comment (the label names what a click does, 'RUN' while stopped and 'STOP' while running; [E, ruling 8 / Q6 in the windows (the RUN button stateful as sm4_browser's map.py)]
- **added** `Interactive_STM_Simulator._map_energies_changed`: New writer, same pattern as _decomp_changed / _markers_changed. [E, ruling 8 / Q6 (map energies: Map's snapped energies)]
- **added** `Interactive_STM_Simulator._crop_changed`: New writer. Key 'crop'. Text is 'crop: none' when crop_x and crop_y are both None. [E, ruling 8 / Q6 (crop: the crop rectangle, or 'none')]
- **added** `Interactive_STM_Simulator._contour_level_changed`: New writer. Key 'contour level'. In Map the text is f"contour level: {z_frac:.3f} of the Map Topo's height range"; [E, ruling 8 / Q6 (contour level: Map's z_frac)]
- **changed** `Interactive_STM_Simulator._build_ui`: Calls self._map_energies_changed() and self._contour_level_changed(), with a two-line comment, after self._cluster_label.set_text('LDOS') and before _update_all(full_refresh=True). [E, ruling 8 / Q6 (each line updated whenever it changes: view changes)]
- **changed** `Interactive_STM_Simulator._set_level`: Calls self._contour_level_changed() after _draw_contour_artists() and before draw_idle. [E, ruling 8 / Q6 (contour level, set by dragging the level line)]
- **changed** `Interactive_STM_Simulator._apply_crop`: Calls self._crop_changed() at its end. That one place covers Crop, Uncrop, every full refresh (view builds, the Cells box) and the Map Topo recompute. [E, ruling 8 / Q6 (crop)]
- **changed** `Interactive_STM_Simulator._update_all (the Map branch)`: One call, self._map_energies_changed(), with an end-of-line comment, after the contour-slice block and before the cluster label. [E, ruling 8 / Q6 (map energies)]
- **changed** `Interactive_STM_Simulator._on_rel (the 'emarker' branch)`: One call, self._map_energies_changed(), with an end-of-line comment, after the snap, the stable re-sort and the _redraw_map_slice loop, and before draw_idle. [E, ruling 8 / Q6 (map energies: an energy marker's release)]
- **changed** `module docstring (lines 6-7)`: 'the suite's shared Qt layer (qtgui at the repository root: PySide6 + pyqtgraph)' now reads 'the suite's shared Qt layer (qtgui, this package's own copy: PySide6 + pyqtgraph)'. [A, P8]
- **changed** `Interactive_STM_Simulator._topo_cap, inner box_b (the Line / Path branch)`: The heatmap box height is now kept in a local h (max(rect[3] - heat_group._label_h() - hm_tb, 1.0), the same expression as before). [A, P3]
- **changed** `Interactive_STM_Simulator._topo_cap docstring`: Added the clause 'in Line a heatmap counted as square, as its row sizes it' inside the existing parenthesis on the box's width. [A, P3]
- **changed** `Interactive_STM_Simulator._make_controls: the Back button's on_change` (review fix): `self._btn_back.on_change(lambda: self._decomp_click(plt.MouseButton.RIGHT))` became `self._btn_back.on_change(self._legend_back)`. [B (Clicks: 'Back button: the same as a right click'; 'While the toolbar's Pan or Zoom is on, the decomp clicks are ignored')]
- **changed** `Interactive_STM_Simulator._make_controls: the Back button's comment` (review fix): Only the stale reference changed: '(_decomp_click)' became '(_legend_back: ignored while Pan or Zoom is on)'. [B (Clicks: Back button)]
- **changed** `Interactive_STM_Simulator._legend_back docstring` (review fix): 'the Back button's click, or' was added at the start of what it handles. [B (Clicks: Back button)]
- **changed** `Interactive_STM_Simulator._on_ui_change: the decay / Extend branch` (review fix): `self._markers_changed()` was added, with an end-of-line comment, just before the branch's `return`. [E, ruling 8 / Q6 (the 'line' line, updated whenever it changes); C (the info lines: key line, the Path nodes with the length)]

## `LDOS_spread_decomp/stm_from_dft_opt/run_DFT_STM.py`

- **removed** `from stm_from_dft.qtgui.qtplot import LinearSegmentedColormap`: The import line is removed; nothing else in the file used it. [F (the LinearSegmentedColormap import removed)]
- **changed** `Interactive_STM_Simulator(...) call: cmap_topo line`: cmap_topo=LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]), became cmap_set='default 1', [F (cmap_set='default 1',)]
- **changed** `Interactive_STM_Simulator(...) call: the commented cmap_map line and its comment`: These became one comment naming the two sets, in the same place: "# cmap_set: the colour maps the window opens with, 'default 1' (the topo black / firebrick / yellow, MAP_CMAP, 'nipy_spectral', 'bwr', 'hot') or 'default 2' ('CET-L3', 'CET-L6', 'CET-R1', 'CET-D1A', 'CET-L19')". [F (one comment naming the two sets)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/qtplot.py`

- **changed** `module docstring, Layout paragraph`: Replaced the last sentence 'The publication export lays a window out at its design size (Figure.at_design_size).' with sm4_browser's 'A saved figure is the window as it is on screen: nothing is laid out again for the file.' The align_column clause was not taken. [E (qtplot.py: module docstring; sm4 texts that are false here not taken)]
- **changed** `module docstring, 'Also here' paragraph`: The save sentence is now sm4_browser's: PDF / EPS / SVG through mplexport with the full info panel; [E (qtplot.py: the module docstring worded the same way)]
- **added** `import matplotlib (module level)`: `import matplotlib` after `import numpy as np`, so importing qtgui stops with ModuleNotFoundError when matplotlib is missing. [E (import matplotlib at module level; response section 4)]
- **added** `WINDOW_IMAGE_DPI`: Constant `WINDOW_IMAGE_DPI = 720` in the configuration block, with sm4's comment. [E (window_image, WINDOW_IMAGE_DPI = 720)]
- **added** `Axes.__init__: export_extent, csv`: Two instance attributes with a comment: `self.export_extent = None` (None, or a callable returning (content_h_px, (y_top, y_bottom)) for the vector saves) and `self.csv = True` (False leaves the axes out of export_csv). [E (the legend panel's hooks: export_extent and csv on Axes with their defaults)]
- **changed** `NavigationToolbar.__init__: Save tooltip`: Now sm4's text: 'Save exactly what the window shows: PDF / EPS / SVG (vector, editable) with the full info panel, an SVG of the whole window as Qt draws it, PNG / TIFF of the window at 720 dpi, or the curves as CSV'. [E (save list replaced; pre-flight port 6.5, the Save tooltip becomes sm4's)]
- **changed** `NavigationToolbar.SAVE_FORMATS and its comment`: Seven entries as in sm4_browser: PDF / EPS / SVG 'vector'; [E (SAVE_FORMATS as sm4_browser's, Q7 label, the comment worded the same way)]
- **changed** `NavigationToolbar._save`: Removed the `if not _have_matplotlib(): formats = [...]` filter, so the dialog always offers the full list. [E (the dialog always offers the full list; response section 4)]
- **changed** `FigureManager docstring`: Now sm4's: the info panel holds Figure.info_text (title details, the window's own lines, the settings section), wraps at its width, and is hidden only until its first text, which comes when the window is shown. [E (sm4_browser's info panel in FigureManager)]
- **changed** `FigureManager.__init__: info panel`: The comment is sm4's. `setLineWrapMode(...WidgetWidth)` replaces NoWrap. [E (wrap at width; height and font kept from part 1)]
- **added** `FigureManager.__init__: window hook sm4_settings_changed`: `self.window.sm4_settings_changed = figure._settings_changed`, with sm4's comment, at the end of __init__. [E (the window hook sm4_settings_changed, name kept, Q8)]
- **changed** `FigureManager.show`: Added `self.figure._settings_changed()` after `self.figure._shown = True`, with sm4's comment. [E (info panel shown once it has text: the push on show)]
- **added** `_settings_text (module global)`: `_settings_text = None`, with sm4's comment, just before class Figure. [E (Figure.info_text ... controls.settings_text)]
- **changed** `Figure docstring`: '...is the reference of both and the size of the publication export.' becomes sm4's '...is the reference of both. [E (sm4 texts; the design-size export sentence is false after the port)]
- **added** `Figure.__init__: export_info_panel`: `self.export_info_panel = getattr(self.manager, 'info', None)` at the end of __init__. [E (Figure.export_info_panel)]
- **changed** `Figure._set_info`: Ends with `self._settings_changed()` in place of the inline 50 ms QTimer; [E (_settings_changed, the coalesced push)]
- **added** `Figure._settings_changed`: sm4's coalesced push: when no push is pending, set `_info_pending` and `QTimer.singleShot(50, self._push_info)`. [E (_settings_changed, the coalesced push)]
- **changed** `Figure.info_text`: Now sm4's. In a FigureManager window it returns the title details and the window's own lines, then an empty line, 'settings', and `_settings_text(self.manager.window)` when bound. [E (Figure.info_text)]
- **changed** `Figure.savefig`: Now sm4's routes. Default how: .pdf .eps .ps .svg give 'vector', .csv gives 'csv', anything else 'image'. [E (savefig's 'vector' (always mplexport), 'qt-svg' and 'image' paths and CSV; local mplexport import kept)]
- **added** `Figure.export_root`: As sm4's: fig.export_root_widget if set, else the manager's window, else the canvas. [E (export_root)]
- **added** `Figure._export_rect`: As sm4's: in the figure's own window, the canvas geometry united with the dock's when the dock is shown docked; [E (_export_rect)]
- **added** `Figure.export_info`: As sm4's: fig.export_info_fn() if set, else the info panel's toPlainText(), else ''. [E (Figure.export_info)]
- **added** `Figure.window_image`: As sm4's: `window_image(path, dpi=WINDOW_IMAGE_DPI)`. [E (window_image, k = dpi / DPI as sm4_browser)]
- **changed** `export_csv`: Skips an axes with `ax.csv` False (`if not ax._visible or not ax.csv: continue`). [E (export_csv skips an axes marked csv = False)]
- **changed** `_render_scene`: Signature `_render_scene(fig, painter)` as sm4's: renders fig.export_root() over fig._export_rect() with its children (controls included), item caches off, QuadItem.EXPORTING set. [E (export_vector: the whole window as Qt draws it, controls included; pre-flight port 6.6)]
- **changed** `export_vector`: sm4's structure without its appended info block. [E (export_vector only for the Qt-drawn SVG, exactly as on screen, nothing appended; the 720 dpi step is blocker 1)]
- **changed** `screen_scale (docstring)`: Stale words 'figure windows open this much larger, and their text follows (font_zoom)' replaced by '(nothing calls it now: every size on screen comes from em_px)'; [A P8]
- **changed** `_EPOCH (comment)`: Now says a measurement kept for the pass (pass_cached) is made once per pass. [A P5 (i)]
- **added** `_ARRANGED`: New module-level counter [0], the arrangement epoch, bumped only by new_arrangement(). [A P5 (i)]
- **changed** `new_measure_pass (docstring)`: Only the stale words changed: a new pass re-makes the pass_cached measurements (no longer 'every panel cell measures its room ... [A P5 (i)]
- **added** `new_arrangement()`: Bumps _ARRANGED and calls new_measure_pass(). [A P5 (i)]
- **changed** `PanelCell.__init__ (comment on _decor_epoch)`: 'the room measured in the current layout pass (_decor)' changed to 'the room measured at the current arrangement (_decor)'. [A P5 (i)]
- **changed** `PanelCell._want_w`: The narrow cell's width is now cached in self.__dict__['_want_cache'] as ((_ARRANGED[0], font_zoom), value), no longer through pass_cached. [A P5 (i)]
- **changed** `PanelCell._note_decor (docstring)`: 'follows at the next layout pass (_want_w keeps the current pass's width)' changed to 'follows at the next arrangement (_want_w keeps the current arrangement's width)'. [A P5 (i)]
- **changed** `PanelCell._decor`: Its cache check and stamp now use _ARRANGED[0] in place of _EPOCH[0]. [A P5 (i)]
- **changed** `Axes._cell_px (narrow branch, inner_w is not None)`: bw = min(inner_w / (1000 / 72) * em_px() * font_zoom, cw - left - right), so the box shrinks to stay inside its cell (off is then 0; [A P5 (ii)]
- **changed** `Figure._relayout`: Its first line new_measure_pass() is now new_arrangement(): the window laid out at its show and on a resize, in every window. [A P5 (i)]
- **changed** `Figure._fit_controls (comment in the hasHeightForWidth branch)`: Stale words "rule 11's fit is Figure.fit_window's" changed to 'the window is never trimmed to its panels'. [A P8]
- **changed** `Figure._schedule_fit (docstring)`: '(rule 11: the window always adjusts)' changed to '(fit_window)'; [A P8]
- **changed** `Figure.fit_window (docstring)`: The sentence 'Done at the first show ..., a moment after every resize (_schedule_fit ...), and by a window after it rebuilt its panels.' replaced by 'Nothing calls it: the window is never trimmed to its panels (a maximized window stays full screen).' The rest unchanged. [A P8]
- **changed** `Figure.savefig, 'vector' branch` (review fix): Passes the background keywords on again, as at adf1e25 but without the dpi argument: `mplexport.export(self, path, facecolor=kw.get('facecolor'), transparent=kw.get('transparent', False))`. [E mplexport.py 'Background handling: as it is here until K4'; PART3_OPEN K4 (until answered, the background stays as it is here)]
- **changed** `Figure.savefig docstring, keyword sentence` (review fix): '`dpi` and any other keyword ... are accepted and ignored' now reads: '`facecolor` and `transparent` are passed on to the 'vector' files (mplexport.export: their background); [E (savefig's docstring); K4]
- **changed** `NavigationToolbar.__init__: Save tooltip` (review fix): Restored from adf1e25: 'Save the figure: PDF / EPS / SVG / PNG / TIFF for publication (matplotlib), PNG / SVG as on screen, or the curves as CSV'. [E (the named qtplot texts are only the SAVE_FORMATS comment, savefig's and export_vector's docstrings and the module docstring); rule 4]
- **changed** `FigureManager class docstring` (review fix): Restored from adf1e25: '... the info panel docked at the bottom (set_info: the details cut from the titles; [E (sm4_browser's info panel in FigureManager.__init__; class docstring not named); rule 4]
- **changed** `Figure class docstring` (review fix): Restored from adf1e25: '(_design_px) is the reference of both and the size of the publication export.' Not a named text. [E (named texts only); rule 4]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/controls.py`

- **added** `Control.stateful (class attribute, line 653)`: stateful = False with sm4's comment, but without sm4's example '(the Line window's 'f' / 'r')', since that window does not exist here: '# a Button whose label shows a state: setting() gives its label'. [E controls.py (each kind's setting() as sm4_browser has them: Button.setting reads it)]
- **added** `Control.__init__: self.setting_label (line 663)`: self.setting_label = self.label if self.label.strip() else str(name), with sm4's three-line comment taken word for word. [E controls.py (each kind's setting() as sm4_browser has them)]
- **changed** `Control.show`: Now calls self._changed() after setVisible, as sm4 does. [E controls.py (Control._changed and its calls)]
- **changed** `Control._fire`: Now calls self._changed() first, before the callback test, with sm4's inline comment. [E controls.py (Control._changed and its calls)]
- **added** `Control.setting()`: The base method returns None, with sm4's docstring. [E controls.py (each kind's setting())]
- **added** `Control._changed()`: sm4's code word for word: it calls getattr(self.qt.window(), 'sm4_settings_changed', None)() and never raises. [E controls.py (Control._changed)]
- **changed** `Control.remove`: Now calls self._changed() first, as sm4 does. [E controls.py (Control._changed and its calls)]
- **added** `Check.setting()`: Returns (setting_label, 'on' or 'off'), as sm4. [E controls.py (each kind's setting())]
- **changed** `Check.set`: Now ends with self._changed(), outside the if, as sm4. [E controls.py (Control._changed and its calls)]
- **added** `Checks.setting()`: Returns a list with one (row label, 'on' or 'off') per row, as sm4. [E controls.py (each kind's setting())]
- **changed** `Checks.set`: Now ends with self._changed(), as sm4. set_all therefore calls it once per row. [E controls.py (Control._changed and its calls)]
- **added** `Radio.setting()`: Returns (setting_label, the selected option, or '' when none is selected), as sm4. [E controls.py (each kind's setting())]
- **changed** `Radio.set`: Now ends with self._changed(), as sm4. [E controls.py (Control._changed and its calls)]
- **added** `Button.setting()`: Returns (setting_label, self._text) only when self.stateful, else None, as sm4. [E controls.py (each kind's setting())]
- **changed** `Button.set_text`: Now ends with self._changed(), as sm4. [E controls.py (Control._changed and its calls)]
- **added** `Text.setting()`: Returns (setting_label, the box's text), as sm4. [E controls.py (each kind's setting())]
- **changed** `Text.set`: Now ends with self._changed(), as sm4. Count.set and Value.set (and so Value.set_limits) reach it through super().set. [E controls.py (Control._changed and its calls)]
- **changed** `Slider._changed(self, v) renamed Slider._moved(self, v)`: The valueChanged slot is renamed (body unchanged) so it no longer hides Control._changed(). [E controls.py (Slider._changed renamed _moved)]
- **changed** `Slider.__init__: sl.valueChanged.connect`: Now connects self._moved instead of self._changed. [E controls.py (Slider._changed renamed _moved)]
- **added** `Slider.setting()`: Returns (setting_label, self.readout.text()), as sm4. [E controls.py (each kind's setting())]
- **changed** `Slider.set`: Now ends with self._changed(), as sm4. [E controls.py (Control._changed and its calls)]
- **added** `settings_text(root) (module function after make(), line 1575)`: sm4's code word for word. Its docstring's '(a window, or the browser's column)' becomes '(a window)', since there is no browser here, and the docstring is re-wrapped for that. [E controls.py (settings_text(root))]
- **added** `plt._settings_text = settings_text (module level, line 1611)`: Binds qtplot._settings_text at import, with sm4's four-line comment word for word. [E controls.py (plt._settings_text)]
- **changed** `OrientedPanes.fit_delta (docstring)`: '(rule 11: the window follows the panel; Figure.fit_window applies it to the window)' changed to '(Figure.fit_window, which nothing calls: the window is never trimmed to its panels)'. [A P8]
- **changed** `Arrangement (class docstring)`: 'which the window can lose (Figure.fit_window)' changed to '(Figure.fit_window, which nothing calls: the window is never trimmed to its panels)'. [A P8]
- **changed** `Arrangement._later`: Also arranges when self._window_state() differs from self._last_state, besides the existing shape test. [A P5 (iii)]
- **added** `Arrangement._window_state()`: Returns (font_zoom, control_zoom, portrait) of the figure whose canvas holds the widget, found by walking parentWidget() up to the QGraphicsView and reading its .figure; [A P5 (iii)]
- **changed** `Arrangement.resizeEvent`: After _arrange(), when no check is pending: sets _pending and QTimer.singleShot(0, self._later). [A P5 (iii)]
- **changed** `Arrangement._arrange`: plt.new_measure_pass() is now plt.new_arrangement() (P5 i). [A P5 (i), (iii)]
- **added** `Arrangement._last_state (instance attribute)`: The window state at the last arrangement, set in _arrange and read in _later through getattr(..., None). [A P5 (iii)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/mplexport.py`

- **changed** `module docstring`: sm4_browser's docstring adapted. Vector files only, image formats are window_image. [Brief E mplexport.py (the docstring); response sections 2 and 4; export_extent hook]
- **added** `_screen_pt(size, fig)`: The one text-size helper: max(10.0, q.font_pt(size) * q.FONT_SCALE * fig.font_zoom), which is qfont's on-screen rule, in points. [Brief E mplexport.py (every text at its on-screen size with the 10 pt floor, one helper)]
- **changed** `to_matplotlib(fig)`: No longer wraps the rebuild in fig.at_design_size(); [Brief E mplexport.py (to_matplotlib, sm4 structure)]
- **changed** `_to_matplotlib(fig)`: sm4's structure: W, H = fig._canvas_size(); [Brief E mplexport.py (_to_matplotlib structure, pt2px figsize, dpi=q.DPI, export_extent hook)]
- **removed** `_to_matplotlib figure-text filter (branch)`: The rule that dropped figure texts more than 2 % below the lowest panel (and its _pos(ax)[1] 'bottom' line) is gone: every visible figure text is written, as in sm4. [Brief E mplexport.py (every visible figure text)]
- **removed** `_to_matplotlib design size (branch)`: fig._design_px no longer sets the figure size; [Brief E mplexport.py (canvas at its on-screen size)]
- **added** `_extent(ax)`: Returns ax.export_extent() when the attribute is set (not None), else None. [Brief E export_extent hook]
- **changed** `_pos(ax) -> _pos(ax, mfig)`: sm4's signature. Uses the height max(h, content_h_px) when the axes has an export_extent. [Brief E mplexport.py (_pos(ax, mfig), fractions from logical pixels; export_extent hook)]
- **changed** `_axes(mfig, ax)`: Calls _pos(ax, mfig). With an export_extent, ya, yb = y_bottom, y_top, applied by m.set_ylim(ya, yb). [Brief E mplexport.py (_pos callers, on-screen text sizes, copy-only features kept, background as here; export_extent limits)]
- **added** `_axes legend title_fontsize (branch)`: When a legend has a title, lk['title_fontsize'] = lk['fontsize'], so the title is at the legend's on-screen size, as qtplot draws it. [Brief E mplexport.py (every text at its on-screen size)]
- **changed** `_artist Annotation branch`: fontsize=_screen_pt(p['fontsize'], a.figure) in place of q.font_pt(p['fontsize'], 10.0). [Brief E mplexport.py (every text at its on-screen size)]
- **changed** `_text(t, transform, target, swap_xy)`: fontsize=_screen_pt(t._size, t.figure) in place of t._size. [Brief E mplexport.py (every text at its on-screen size)]
- **changed** `_colorbar(mfig, cax)`: Calls _pos(cax, mfig). Label and tick sizes through _screen_pt(..., cax.figure). [Brief E mplexport.py (_pos callers, on-screen text sizes)]
- **added** `section 'The info block' with _info_lines(fig)`: sm4's code word for word. Docstring: browser references and 'and by qtplot.export_vector' dropped, since this copy's export_vector does not use it. [Brief E mplexport.py (_info_lines)]
- **added** `_info_geometry(fig, lines)`: sm4's code word for word; the docstring no longer names qtplot.export_vector as a user. [Brief E mplexport.py (_info_geometry)]
- **added** `_info_block(mfig, fig, H, H_info)`: sm4's code with the settled conversions. size = _px_to_pt(info.pixelSize()) (pixel size / q.pt2px(1)) in place of pixelSize*72/q.DPI. [Brief E mplexport.py (_info_block; info text size through pt2px; fractions from logical pixels)]
- **changed** `export(fig, path, dpi=300, facecolor=None, transparent=False) -> export(fig, path, facecolor=None, transparent=False)`: sm4's export: vector only (ValueError for anything but .pdf .eps .ps .svg), no crop, no margin (bbox_inches and pad_inches gone), fonttypes 42 / 'none'. [Brief E mplexport.py (export; background handling as here until K4)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/__init__.py`

- **changed** `module docstring, line 2`: 'qtgui: the one shared Qt layer of every interactive window of VASP_for_SPM (PySide6 + pyqtgraph),' changed to 'qtgui: this package's own copy of the Qt layer of VASP_for_SPM's interactive windows (PySide6 + pyqtgraph),'. [A P8]

## `qtgui/qtplot.py`

- **changed** `module docstring, Layout paragraph (last sentence)`: 'The publication export lays a window out at its design size (Figure.at_design_size).' replaced by 'A saved figure is the window as it is on screen: nothing is laid out again for the file.' (the package copy's and sm4_browser's wording) [E (module docstring worded the same way)]
- **changed** `module docstring, 'Also here' paragraph`: The save sentence now describes the new routes: PDF / EPS / SVG through mplexport with the full info panel and controls left out; [E (module docstring; Q7 'as shown'; no 'full info panel' for the Qt SVG)]
- **added** `import matplotlib (module level)`: 'import matplotlib # noqa: F401 required: ...' after 'import numpy as np', so importing qtgui fails without matplotlib [E root 'matplotlib required'; response section 4]
- **added** `WINDOW_IMAGE_DPI`: Constant WINDOW_IMAGE_DPI = 720 with sm4_browser's comment, after DPI in the configuration block [E (WINDOW_IMAGE_DPI = 720)]
- **added** `Axes.__init__: export_extent, csv`: Instance attributes 'self.export_extent = None' and 'self.csv = True' after _rotor_item, with the package copy's comment that describes the hook [E (the legend panel's hooks: export_extent, csv)]
- **changed** `NavigationToolbar.__init__: Save action tooltip`: sm4_browser's tooltip: 'Save exactly what the window shows: PDF / EPS / SVG (vector, editable) with the full info panel, an SVG of the whole window as Qt draws it, PNG / TIFF of the window at 720 dpi, or the curves as CSV' [E (save texts)]
- **changed** `NavigationToolbar.SAVE_FORMATS and its comment`: Now seven entries: 'vector' for PDF, EPS and SVG; [E (SAVE_FORMATS as sm4_browser's, Q7 label)]
- **changed** `NavigationToolbar._save`: Removed the '_have_matplotlib()' filter (the dialog always offers the full list). [E (the dialog always offers the full list); response section 4]
- **changed** `FigureManager class docstring`: Describes the info panel as sm4_browser does: Figure.info_text, the settings section, wrapped at its width, hidden only until its first text [E (info panel)]
- **changed** `FigureManager.__init__: info panel`: Comment as sm4_browser's / the package copy's; [E (info panel wraps at width; root keeps its own DPI sizes)]
- **added** `FigureManager.__init__: window hook sm4_settings_changed`: 'self.window.sm4_settings_changed = figure._settings_changed' at the end of __init__, with sm4_browser's comment [E (window hook sm4_settings_changed, Q8)]
- **changed** `FigureManager.show`: After '_shown = True' it now calls 'self.figure._settings_changed()' (sm4_browser's line and comment), so the panel gets its first text when the window is shown [E (shown once it has text; the coalesced push)]
- **added** `_settings_text (module global)`: '_settings_text = None' with sm4_browser's comment, after '_figures = {}' and before class Figure. [E (Figure.info_text ... controls.settings_text)]
- **changed** `Figure class docstring`: '... is the reference of both and the size of the publication export.' -> '... [E (texts worded the same way)]
- **changed** `Figure.__init__ (end)`: Added 'self.export_info_panel = getattr(self.manager, 'info', None)' with the package copy's comment [E (export_info_panel)]
- **changed** `Figure._set_info`: Its inline 50 ms QTimer is replaced by a call to self._settings_changed(), which runs the same timer, so the behaviour is unchanged [E (_settings_changed, the coalesced push)]
- **added** `Figure._settings_changed`: sm4_browser's coalesced push: one QTimer.singleShot(50, self._push_info) per event, guarded by _info_pending [E (_settings_changed)]
- **changed** `Figure.info_text`: sm4_browser's version: the title details and the window's lines. [E (Figure.info_text)]
- **changed** `Figure.savefig`: sm4_browser's routes and the package copy's docstring. [E (savefig 'vector' always mplexport, 'qt-svg', 'image', CSV; absolute import style); response section 4]
- **added** `Figure.export_root`: sm4_browser's: fig.export_root_widget if set, else the manager's window, else the canvas. [E (export_root)]
- **added** `Figure._export_rect`: sm4_browser's: in the figure's own window, canvas.geometry() united with the dock's geometry when the dock is visible and docked; [E (_export_rect)]
- **added** `Figure.export_info`: sm4_browser's: export_info_fn() if a window set it, else info.toPlainText(), else '' [E (export_info)]
- **added** `Figure.window_image(path, dpi=WINDOW_IMAGE_DPI)`: sm4_browser's high-definition screenshot: k = dpi / DPI (the root's DPI = 100). [E (window_image, k = dpi / DPI, 720 dpi window images)]
- **changed** `export_csv`: Skips an axes whose csv is False ('if not ax._visible or not ax.csv'). [E (export_csv skips an axes marked csv = False)]
- **changed** `_render_scene(fig, painter)`: Was (fig, painter, W, H): scene.render with a WINDOW_BG fill. [E (Qt-drawn SVG: the whole window as on screen, controls included)]
- **changed** `export_vector`: The package copy's port: W, H from fig._export_rect(), the window exactly as shown with nothing added below it, gen.setSize rounded, painters ended in try/finally, and the PDF route recorded into a QPicture then played into QPdfWriter. [E (export_vector only for the Qt-drawn SVG, nothing appended); 720 dpi part blocked]
- **changed** `Figure.savefig: the 'vector' branch` (review fix): The call to mplexport.export(self, path) again passes on the caller's background arguments, as at adf1e25: mplexport.export(self, path, facecolor=kw.get('facecolor'), transparent=kw.get('transparent', False)). [E package item 'Background handling: as it is here until K4' (the root takes the same pieces); PART3_OPEN K4 'Until answered: the background handling stays as it is here']
- **changed** `Figure.savefig docstring, first paragraph` (review fix): '`dpi` and any other keyword ... are accepted and ignored' now reads: '`facecolor` and `transparent` are passed on to the vector files (mplexport.export: the file's background); [E (savefig's docstring worded to match the code); K4]

## `qtgui/controls.py`

- **added** `Control.stateful`: Class attribute, False, with the copy's comment ('a Button whose label shows a state: setting() gives its label'). [E controls.py (root: the same pieces)]
- **added** `Control.__init__: setting_label`: Instance attribute set to the label, or str(name) when the label is blank. [E controls.py]
- **changed** `Control.show`: Calls self._changed() after setVisible. [E controls.py (Control._changed and its calls)]
- **changed** `Control._fire`: Calls self._changed() first, before the callback test, with the copy's comment. [E controls.py (Control._changed and its calls)]
- **added** `Control.setting`: Base setting() returning None. Its docstring is the copy's word for word, and its kind list includes Value. [E controls.py (each kind's setting())]
- **added** `Control._changed`: Takes no arguments. Calls getattr(self.qt.window(), 'sm4_settings_changed', None) when present. [E controls.py (Control._changed)]
- **changed** `Control.remove`: Calls self._changed() first. [E controls.py (Control._changed and its calls)]
- **added** `Check.setting`: Returns (setting_label, 'on' or 'off'). [E controls.py (each kind's setting())]
- **changed** `Check.set`: Calls self._changed() at the end, even when the value is unchanged. [E controls.py (Control._changed and its calls)]
- **added** `Checks.setting`: Returns a list of (row label, 'on' or 'off'), one per row, from self.boxes and spec['labels']. [E controls.py (each kind's setting())]
- **changed** `Checks.set`: Calls self._changed() at the end (so set_all calls it once per row). [E controls.py (Control._changed and its calls)]
- **added** `Radio.setting`: Returns (setting_label, the selected option, or '' when none is selected). [E controls.py (each kind's setting())]
- **changed** `Radio.set`: Calls self._changed() at the end. [E controls.py (Control._changed and its calls)]
- **added** `Button.setting`: Returns (setting_label, _text) when stateful, else None. [E controls.py (each kind's setting())]
- **changed** `Button.set_text`: Calls self._changed() after _apply_labels. [E controls.py (Control._changed and its calls)]
- **added** `Text.setting`: Returns (setting_label, the box's text as shown). [E controls.py (each kind's setting())]
- **changed** `Text.set`: Calls self._changed() at the end. Count.set, Value.set and Value.set_limits reach it through super().set. [E controls.py (Control._changed and its calls)]
- **changed** `Slider._changed -> Slider._moved`: The QSlider.valueChanged slot is renamed _moved, so Control._changed is not shadowed. [E controls.py (Slider._changed renamed _moved)]
- **changed** `Slider.__init__: valueChanged.connect`: sl.valueChanged.connect(self._moved), formerly self._changed. [E controls.py (Slider._changed renamed _moved)]
- **added** `Slider.setting`: Returns (setting_label, readout text). [E controls.py (each kind's setting())]
- **changed** `Slider.set`: Calls self._changed() at the end. [E controls.py (Control._changed and its calls)]
- **added** `settings_text(root)`: Module function placed after make(); the copy's code and docstring word for word. [E controls.py (settings_text(root))]
- **added** `plt._settings_text = settings_text`: Module-level binding after settings_text, with the copy's four-line comment. [E controls.py (plt._settings_text)]

## `qtgui/mplexport.py`

- **changed** `module docstring`: Now covers vector files only (PDF, EPS, SVG), with the image formats left to qtplot.Figure.window_image. [E (root: mplexport.py docstring; K4 background as here)]
- **added** `_screen_pt(size, fig)`: A text's on-screen size in points: q.font_pt(size) * q.FONT_SCALE * float(fig.font_zoom). [E (every text at its on-screen size; root: sm4's sizes)]
- **changed** `to_matplotlib(fig)`: No longer wrapped in fig.at_design_size(); [E (to_matplotlib)]
- **changed** `_to_matplotlib(fig)`: sm4's structure. W, H come from fig._canvas_size(), and H_top = max(H, y0 + max(h, content_h_px)) over axes with an export_extent. [E (_to_matplotlib structure, export_extent hook, info block at max(H, lowest export bottom))]
- **added** `_extent(ax)`: Returns ax.export_extent() when that attribute is set (None by default), else None. [E (export_extent hook)]
- **changed** `_pos(ax) -> _pos(ax, mfig)`: Fractions of mfig.get_figwidth()*mfig.dpi and mfig.get_figheight()*mfig.dpi (sm4's). [E (_pos(ax, mfig); root: sm4's DPI conversions)]
- **changed** `_axes(mfig, ax)`: Calls _pos(ax, mfig), with fig = ax.figure. [E (every text at its on-screen size; export_extent hook)]
- **changed** `_artist: Annotation branch`: fontsize = _screen_pt(p['fontsize'], a.figure), was q.font_pt(p['fontsize'], 10.0). [E (every text at its on-screen size)]
- **changed** `_text(t, transform, target, swap_xy)`: fontsize = _screen_pt(t._size, t.figure); [E (every text at its on-screen size)]
- **changed** `_colorbar(mfig, cax)`: Calls _pos(cax, mfig); the label and tick sizes go through _screen_pt(..., cax.figure). [E (_pos(ax, mfig); on-screen text sizes)]
- **added** `section comment 'The info block'`: Section divider above the info block functions, as sm4 and the package copy have it. [E (_info_lines, _info_geometry, _info_block)]
- **added** `_info_lines(fig)`: sm4's text-wrapping code (QTextLayout in the panel's font, text option and viewport width), word for word as in the package copy. [E (_info_lines)]
- **added** `_info_geometry(fig, lines)`: Returns (font, lineSpacing, ascent, documentMargin + frameWidth, H_info); [E (_info_geometry)]
- **added** `_info_block(mfig, fig, H, H_info)`: sm4's code with H being the canvas part (H_top). [E (_info_block; root: sm4's DPI conversions)]
- **changed** `export(fig, path, dpi=300, facecolor=None, transparent=False) -> export(fig, path, facecolor=None, transparent=False)`: Vector files only: a ValueError for any extension other than .pdf, .eps, .ps or .svg. [E (export; K4 background as here)]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/_resolve.py`

- **added** `module _resolve.py (new file)`: New module. It has a utf-8 coding line, and at module level it imports only `import os` and `import sys`. [A P13 (b)]
- **added** `module docstring`: New text. It says what the module resolves (simulator, doscar_parser, the qtgui copy), that it imports nothing beyond os and sys at module level and each resolver imports inside its function, so a Qt-only caller (run_wfn_plane_slice.py) does not load the matcher. [A P13 (b)]
- **added** `resolve_stm()`: Moved from _imports.py byte for byte (docstring and body; [A P13 (b)]
- **added** `resolve_doscar_parser()`: Moved from _imports.py byte for byte (docstring already starts with 'Machinery:'). [A P13 (b)]
- **added** `resolve_qt()`: Moved from _imports.py byte for byte. It returns (qtplot, controls, widgets, mplexport): `.stm.qtgui` first, else the same path insert and `stm_from_dft.qtgui`. [A P13 (b)]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/_imports.py`

- **added** `re-export import`: `from ._resolve import resolve_stm, resolve_qt, resolve_doscar_parser`, placed after `import sys` and before the matcher try/except block. [A P13 (b)]
- **removed** `resolve_stm()`: Def removed (moved to _resolve.py, re-exported). [A P13 (b)]
- **removed** `resolve_doscar_parser()`: Def removed (moved to _resolve.py, re-exported). [A P13 (b)]
- **removed** `resolve_qt()`: Def removed (moved to _resolve.py, re-exported). [A P13 (b)]

## `LDOS_spread_decomp/mo_ldos_decomp/run_wfn_plane_slice.py`

- **changed** `module-level import of resolve_qt (line 313)`: `from mo_ldos_projection._imports import resolve_qt # noqa: E402` became `from mo_ldos_projection._resolve import resolve_qt # noqa: E402`. [A P13 (b)]
- **changed** `module docstring, lines 37-38`: '(qtgui at the repository root)', wrapped over two lines, now reads '(the simulator's qtgui copy)', wrapped at the same place. [A P8]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/plot_dos_cursors_levels.py`

- **removed** `module-level `from doscar_parser import SpinAwareDosParser, SpinMode` (former line 29) and the blank line above it`: The absolute import that failed when the file is imported as part of the package is gone from module level. [A P11]
- **added** `run-directly branch (`if __package__ in (None, ''):`): `from stm_from_dft.doscar_parser import SpinAwareDosParser, SpinMode``: The run-directly branch imports the parser from stm_from_dft, after its two stm_from_dft.qtgui imports. [A P11]
- **added** `package branch (`else:`): `from .doscar_parser import SpinAwareDosParser, SpinMode``: The package branch imports the parser relatively, after its two .qtgui imports. [A P11]
- **changed** `module docstring, line 6`: '(qtgui at the repository root: PySide6 + pyqtgraph)' now reads '(qtgui, this package's own copy: PySide6 + pyqtgraph)'. [A P8]

## `LDOS_spread_decomp/MOPmatcher_mem/matcher/vacupot_plotter.py`

- **changed** `module docstring, line 3`: '(the shared Qt layer at the repository root)' now reads '(the shared Qt layer, this package's own copy)'. [A P8]

## `LDOS_spread_decomp/MOPmatcher_mem/matcher/spread_plot.py`

- **changed** `module docstring, line 35`: '(the shared Qt layer at the repository root)' now reads '(the shared Qt layer, this package's own copy)'. [A P8]

## `LDOS_spread_decomp/MOPmatcher_mem/matcher/state_decomposition_plot.py`

- **changed** `module docstring, line 72`: '(the shared Qt layer at the repository root)' now reads '(the shared Qt layer, this package's own copy)'. [A P8]

## `LDOS_spread_decomp/MOPmatcher_mem/matcher/spread_collapse.py`

- **changed** `module docstring, line 77`: '(the shared Qt layer at the repository root)' now reads '(the shared Qt layer, this package's own copy)'. [A P8]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/decomp_window.py`

- **changed** `module docstring, line 114`: 'Window (the shared Qt layer, qtgui at the repository root)' now reads 'Window (the shared Qt layer, the simulator's qtgui copy)'. [A P8]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/topo_decomp_window.py`

- **changed** `module docstring, line 71`: 'Window (the shared Qt layer, qtgui at the repository root)' now reads 'Window (the shared Qt layer, the simulator's qtgui copy)'. [A P8]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/global_decomposition.py`

- **changed** `module docstring, line 38`: 'Window (the shared Qt layer, qtgui at the repository root)' now reads 'Window (the shared Qt layer, the simulator's qtgui copy)'. [A P8]

## `LDOS_spread_decomp/MOPmatcher_mem/README.md`

- **changed** `'The plot windows' paragraph, line 29`: 'every text scales with the window's size;' now reads 'every text scales with the window's size and is never below 10 pt;'. [A P8]

## `LDOS_spread_decomp/mo_ldos_decomp/README.md`

- **changed** `'The windows' paragraph, line 45`: 'Every text scales with the window's size relative to its design size, and in a window' now reads '... [A P8]
- **changed** `module table, `_imports.py` row`: The row keeps 'import resolution of the matcher and simulator packages'. [A P8 (the `_imports.py` row: no resolve_qt, and now _resolve.py)]
- **removed** `module table, `_imports.py` row: the `_resolve.py` clause` (review fix): Removed the clause nothing names: '; the simulator package's three resolvers (`resolve_stm()`, `resolve_qt()`, `resolve_doscar_parser()`) are defined in `_resolve.py`, which does not import the matcher, and re-exported here'. [A P8 (the decomp README's `_imports.py` row, no `resolve_qt()`); plan rule 5 / task rule 4 (an unplanned change undone)]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py`

- **changed** `StmDecompSession.__init__ (argument cmap_topo=None)`: The argument `cmap_topo=None` is now `cmap_set='default 1'`, in the same place in the signature. [F (RESPONSE_PART1_PART3 section 6, sim_wrapper.py)]
- **removed** `StmDecompSession.__init__ (branch if cmap_topo is None)`: Removed the wrapper's own black / firebrick / yellow map: the `if cmap_topo is None: cmap_topo = plt.LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])` block (3 lines). [F (its own black / firebrick / yellow map removed)]
- **changed** `StmDecompSession._ctor_kwargs`: The dict now has `cmap_set=cmap_set` where it had `cmap_topo=cmap_topo`, so `_instantiate` passes the set on to Interactive_STM_Simulator. [F (passed on to the simulator in place of cmap_topo)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/cmap_shading.py`

- **added** `module docstring`: Starts with 'Machinery:'. Says what the module holds, that it is self-contained (imports only the standard library and NumPy, nothing from the simulator or qtgui, and CuPy only through the arrays it is given), and that it is written to be copied into sm4_Browser, with each package keeping its own co… [addendum 1 (one portable module), 3 (the combination)]
- **added** `SET_NAMES, DEFAULT_SET, ROLES, SETS`: The two sets by role. Roles: 'topo', 'map', 'heat', 'mag_signed', 'mag_abs'. [response section 6 (the two sets); addendum 1]
- **added** `SHADING_CMAPS`: {'map': 'CET-I1', 'mag_signed': 'CET-D11', 'mag_abs': 'CET-I3'}, the same in both sets [addendum 3 (the colour map while Shading is on)]
- **added** `cmap_spec, map_role, make_cmap`: cmap_spec(set_name, role, shading=False) returns the entry, or the shading map when shading=True. [response section 6; addendum 1]
- **added** `CONTOUR, SHADING, SHADE_BY, HEIGHT, RELIEF, SHADE_MODES, BOX_TITLE`: The control names 'Contour', 'Shading', 'Shading by'; [addendum 2]
- **added** `BoxState`: State and rules: at most one toggle on, both may be off. [addendum 2]
- **added** `ShadeBox, _make_control, build_box`: build_box(fig, controls, on_change=None, state=None, title=BOX_TITLE, orient='h') makes two Checks and a Radio(labels Height/Relief) and frames them with controls.box. [addendum 2 (the box); ruling 11 (builder takes the figure and the controls module)]
- **added** `array_module`: Returns cupy.get_array_module(*arrays) when 'cupy' is already in sys.modules, else numpy. [addendum 1 (its maths and the GPU); ruling 11]
- **added** `unique_points`: Takes (N*N,) in grid_xy order or (N,N) and returns a view of shape (N-1,N-1): row i along b, column j along a, with the duplicate last row and column dropped [addendum 3 (drawing, resolution); ruling 11]
- **added** `height_shade`: Height shade: the heights themselves as float64 in their own array module [addendum 3 (Height)]
- **added** `_D5_P, _D5_D1, _along, _conv5, _set_border_replicate`: Transcriptions of Kovesi's derivative5.m first-derivative taps, of MATLAB conv2 'same' along one axis (zero outside, or wrapped when periodic), and of imsetborder 'replicate' in his loop order [addendum 3 (Relief: relief.m)]
- **added** `light_vector`: light_vector(az=45.0, el=45.0, *, panel_turn) builds relief.m's I = [cos(el)sin(az), cos(el)cos(az), sin(el)] / norm in screen coordinates (right, up, out), then turns it into data x, y by -panel_turn degrees. [addendum 3 (Relief: the light vector); the turned-maps question]
- **added** `relief_shade`: relief_shade(heights, step_a, step_b, *, panel_turn, az=45.0, el=45.0, gradscale=1.0, minval=0.0, periodic=True). [addendum 3 (Relief, the normals)]
- **added** `shade_field`: Dispatches on mode to height_shade ('Height') or relief_shade ('Relief'). [addendum 3]
- **added** `cmap_table, _norm_xp, _lookup_xp, colormap_rgba`: The colour map is applied first, with the caller's norm. [addendum 3 (the other display options keep working)]
- **added** `_Y_R/_Y_G/_Y_B, _LSTAR_EPS, _KAPPA, _srgb_decode, _srgb_encode, _y_to_lstar, _lstar_to_y`: The CSS Color 4 sRGB decode / encode and its D65 Y row (87098/409605, 175762/245763, 12673/175545); [addendum 3 (exact even steps)]
- **added** `lstar_stretch, shaded_rgba`: f = floor + (1-floor)(s - s_min)/(s_max - s_min) over the finite shade values given. [addendum 3 (the strength, the floor 0, exact even steps)]
- **added** `mesh_corners`: mesh_corners(step_a, step_b, shape, offsets=((0,0),), origin=(0,0)) returns X, Y as NumPy float64 of shape (len(offsets), rows+1, cols+1). [addendum 3 (drawing); ruling 11 (quads centred, tiled by cell offsets)]
- **changed** `ROLES, SETS, SHADING_CMAPS (and the ROLES comment)` (review fix): The role keys now match the working-tree simulator.py's CMAP_SETS: ROLES = ('topo', 'map', 'heatmap', 'mag', '|m|'). [addendum 1 (the sets move into the module); response section 6 (finding 5)]
- **changed** `cmap_spec, map_role docstrings and return values` (review fix): cmap_spec's docstring names 'map', 'mag' and '|m|' as the roles with a shading map. [response section 6 (finding 5)]
- **changed** `make_cmap docstring` (review fix): Added: each from_list call makes a new map, and qtplot's Colormap has no __eq__, so two calls give maps that compare unequal. [addendum 1 (finding 5)]
- **changed** `relief_shade signature` (review fix): `periodic` is now a required keyword with no default: (heights, step_a, step_b, *, panel_turn, periodic, az=45.0, el=45.0, gradscale=1.0, minval=0.0). [addendum 3, Relief (Kovesi's defaults) (finding 1)]
- **changed** `relief_shade docstring` (review fix): (1) A new 'The edges' item. relief.m has no edge option: conv2 adds nothing from outside the array, then imsetborder's 2-point replicate border. [addendum 3, Relief (the normals; gradient scale 1); addendum 1 (portable to sm4_Browser) (findings 1, 6, 7, 8)]
- **changed** `shade_field signature and docstring` (review fix): `periodic` is now a required keyword: (heights, mode, step_a, step_b, *, panel_turn, periodic, az=45.0, el=45.0, gradscale=1.0, minval=0.0). [addendum 3 (finding 1)]
- **changed** `lstar_stretch docstring` (review fix): A point with a non-finite shade becomes (0,0,0,0) for RGBA input, so it is not drawn. [addendum 3, the strength (finding 9)]

