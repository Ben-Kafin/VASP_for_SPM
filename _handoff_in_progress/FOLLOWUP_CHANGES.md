# Follow-up: the per-file change list (rule 5)

Every function, constant, branch and text changed, added or removed by the follow-up (the owner's answers O1-O18,
K1-K11 and the addendum's integration), from the implementing agents' and fixers' reports (base cfe6242). Fixer
entries are marked (review fix). Additional changes: none reported. Not part of the workflow: ruling 14's
cmap_shading.py edit (cfcba7d). The matcher's qtgui copy is the package copy, copied byte for byte.

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py`

- **added** `gpu_decay_off_factor(phi, d)`: New module-level def in the physics block, after gpu_chen_tunneling_factor and before '# ==== NUMERICS AND GPU'. [The simulator: O7]
- **changed** `Unified_STM_Simulator._calculate_ldos_at_points_gpu._compute_channel, Decay-off branch (not preserve_orbitals, last else)`: The two inline lines (kappa0 = ...; sf = cp.exp(...)) replaced by sf = gpu_decay_off_factor(phi_local, dists); [The simulator: O7]
- **changed** `Unified_STM_Simulator._calculate_ldos_at_points_gpu._compute_channel, Decay-off branch (preserve_orbitals, last else)`: The two inline lines replaced by sf = gpu_decay_off_factor(phi_local, dists); [The simulator: O7]
- **changed** `THE DECAY MODEL (physics) block comment, the 'Decay off:' entry`: Stale words only. The sentence 'It is inline code, not a def: the two lines ... [The simulator: O7; Texts (K9's rule: text the change makes false)]
- **added** `Unified_STM_Simulator._display_indices(self, emin, emax)`: Moved here unchanged (same signature, docstring and body), placed between _precision_cache_suffix and _calculate_ldos_at_points_gpu, which calls it. [The simulator: O3]
- **removed** `Interactive_STM_Simulator._display_indices`: Removed from the interactive class, which inherits it now. [The simulator: O3]
- **added** `Interactive_STM_Simulator._line_norm (attribute)`: The Line / Path Norm factors, (npts,) float32 NumPy: for each line / path point, quadrature.simpson(cp.sum(f_ldos_raw, axis=(2, 3)), dx=self.e_step32, axis=1) computed on the GPU and copied with cp.asnumpy. [The simulator: O1]
- **changed** `Interactive_STM_Simulator._update_all, the Line / Path drag-cache block`: Right after self._line_disp = cp.asnumpy(f_ldos_raw), it now also sets self._line_norm, with a comment. [The simulator: O1]
- **changed** `Interactive_STM_Simulator._clear_panels`: Adds self._line_norm = None next to self._line_disp = None. [The simulator: O1]
- **changed** `Interactive_STM_Simulator._draw_spectrum signature`: New keyword line_norm=None, added last. [The simulator: O1]
- **added** `Interactive_STM_Simulator._draw_spectrum, Line / Path branch: the 'Norm' in a drag`: When line_norm is not None, self.normalize is on and f_ldos_raw is not None: active_ldos = active_ldos / (line_norm[active_idx] + 1e-15), and each marker spectrum is divided the same way by line_norm[int(r * (npts - 1))] + 1e-15. [The simulator: O1]
- **changed** `Interactive_STM_Simulator._draw_spectrum, the integral passed to Decomp.curves`: The condition is now 'if self.normalize and line_norm is None else None'. [The simulator: O1]
- **changed** `Interactive_STM_Simulator._draw_spectrum docstring`: Adds a description of line_norm: the Norm factors kept beside the drag cache, divided by here under 'Norm' (factor + 1e-15) with no GPU work; [The simulator: O1]
- **changed** `Interactive_STM_Simulator._move_line_markers`: Calls self._draw_spectrum(self._line_disp, None, self._line_geom[1], line_norm=self._line_norm). [The simulator: O1]
- **changed** `Interactive_STM_Simulator._on_pick, the mark_dynamic branch`: Its _draw_spectrum call now passes line_norm=self._line_norm. [The simulator: O1]
- **added** `Interactive_STM_Simulator.__init__: Map's own state (map_e_idx, map_e_targets, map_typed, map_ld_up, map_ld_dn, map_ld_up_gpu, map_ld_dn_gpu, map_current, map_current_E, map_eg, map_window, map_d_ldos), all None, with a comment block`: Map's energies, data and caches get their own attributes, apart from Line / Path's, initialised in the constructor. [The simulator, O2]
- **added** `Interactive_STM_Simulator.__init__: cached_spec_window = None`: The window's snapped ends (e_lo, e_hi) that the marker spectra (shared by Single Point and Map) were last computed over. [The simulator, O2]
- **changed** `Interactive_STM_Simulator.__init__: comment on cached_current / cached_current_E`: Was 'Line / Path points or the Map grid'; [The simulator, O2]
- **changed** `Interactive_STM_Simulator._feenstra_iv`: New which='map' returns feenstra.iv_ratio(map_current_E, map_current). [The simulator, O2]
- **changed** `Interactive_STM_Simulator._on_ui_change, the Decay / Extend invalidation branch`: cached_d_topo_map, cached_bias_energy_map, cached_marker_coords, cached_spec_ldos and cached_spec_ldos_gpu are now reset only when the decay model changes. [The simulator, O2 (Map's caches recomputed only for the listed changes)]
- **changed** `Interactive_STM_Simulator._update_all: signature and docstring`: New keyword last_run=False. The docstring now describes the RUN-off decomp redraw and Map's kept energies, data and caches. [The simulator, O2 and K7 (d)]
- **changed** `Interactive_STM_Simulator._update_all: RUN-off early return`: With RUN off it still returns after the geometry, unless last_run is set and _has_last_run() is True; [The simulator, K7 (d)]
- **changed** `Interactive_STM_Simulator._update_all: _broadening_topo_dirty branch`: Now runs only when computing (RUN on). [The simulator, K7 (d)]
- **changed** `Interactive_STM_Simulator._update_all: needs_topo / needs_ldos / needs_spec`: All three are now False when not computing. [The simulator, O2]
- **added** `Interactive_STM_Simulator._update_all: reform block (after the needs_topo block)`: After an edit of E Min / E Max / E Pts, the Map energies are formed again from the typed ends (map_e_idx, map_e_targets). [The simulator, O2]
- **changed** `Interactive_STM_Simulator._update_all: Map LDOS compute`: Split out as `if needs_ldos and self.mode == 'Map'`. [The simulator, O2]
- **changed** `Interactive_STM_Simulator._update_all: Line / Path / Single Point LDOS compute`: Now in the `elif needs_ldos` branch. The Map sub-branch is removed, and cached_nepts is no longer written there; [The simulator, O2]
- **changed** `Interactive_STM_Simulator._update_all: marker-spectra compute`: Also records cached_spec_window = (e_lo, e_hi). [The simulator, O2]
- **changed** `Interactive_STM_Simulator._update_all: displayed LDOS block`: Line / Path draw from cached_ld_*_gpu with Feenstra 'grid'. [The simulator, O2]
- **added** `Interactive_STM_Simulator._update_all: cached_eg = map_eg in Map`: Map's spectrum (and sim_wrapper's cached_eg) uses Map's own displayed energies. [The simulator, O2]
- **changed** `Interactive_STM_Simulator._update_all: _draw_spectrum call`: No longer passes nepts or full_refresh. [The simulator, O2]
- **changed** `Interactive_STM_Simulator._update_all: Map display branch`: nepts = len(self.map_e_idx), the kept energies' count. [The simulator, O2]
- **changed** `Interactive_STM_Simulator._draw_spectrum`: Signature is now (f_ldos_raw, spec_disp, p_dist, line_norm=None): nepts and full_refresh are removed. [The simulator, O2]
- **changed** `Interactive_STM_Simulator._move_line_markers`: New keyword last_run=False. With it, the marker spectra are also set from the drag cache (_line_disp, _line_norm) when RUN is off. [The simulator, K7 (d) (RUN-off re-sort)]
- **changed** `Interactive_STM_Simulator._resort_line_markers`: Calls _move_line_markers(last_run=True). With RUN off, the level-0 curves, the spectrum title and the legend are redrawn in place from the last run's data. [The simulator, K7 (d) (RUN-off re-sort)]
- **changed** `Interactive_STM_Simulator._redraw_map_slice`: Reads map_ld_up_gpu / map_ld_dn_gpu and _feenstra_iv('map'). [The simulator, O2]
- **changed** `Interactive_STM_Simulator._decomp_click`: Calls _update_all(last_run=True), so a decomp step with RUN off redraws once from the last run's data. [The simulator, K7 (d)]
- **added** `Interactive_STM_Simulator._has_last_run`: Machinery def: True when the view has the last run's data. [The simulator, K7 (d)]
- **changed** `Interactive_STM_Simulator._on_rel, the energy-marker branch`: The dragged energy's column and the sort permutation now write map_ld_up/_dn, map_ld_up_gpu/_dn_gpu, map_current and map_current_E instead of the shared cached_*. [The simulator, O2 (the drag as part 2 has it)]
- **changed** `Interactive_STM_Simulator._make_controls (comment on _btn_back)`: '(Single Point, Line, Path; kept across views ...' becomes '(Single Point, Line, Path, Map; [The simulator, K2 (a)]
- **changed** `section comment '# --- the legend panel (right of the spectrum: ...) ---'`: 'Single Point, Line, Path' becomes 'Single Point, Line, Path, Map'. [The simulator, K2 (a)]
- **changed** `Interactive_STM_Simulator._spectrum_and_legend(spec_cell)`: Now only builds the row: self._spec_sub = controls.widget_of(controls.row(spec_cell, self._legend_build())), then returns it. [The simulator, K2 (a)]
- **added** `Interactive_STM_Simulator._legend_build()`: The legend panel builder, which is the former body of _spectrum_and_legend without the row line (the same widgets, connections, _legend_size, _legend_view, on_placed hooks). [The simulator, K2 (a)]
- **added** `Interactive_STM_Simulator._legend_build: ax.set_navigate(False)`: The legend stack axes calls set_navigate(False) after axis('off') / set_facecolor('w'), with an inline comment. [The simulator, K8 (a)]
- **changed** `Interactive_STM_Simulator._update_legend (docstring)`: The sentence 'Nothing in Map (no legend panel there yet).' is removed. [The simulator, K2 (a)]
- **changed** `Interactive_STM_Simulator._legend_rezoom`: Records w0 = self._legend_panel.minimumWidth() before _legend_size(). [The simulator, K2 (a) (counted in the block's width; the qtgui stage's note that a width change alone does not place the block)]
- **changed** `Interactive_STM_Simulator._clear_map_rows`: Its set_groups([], None, extra=...) call also passes extra_side=getattr(self, '_legend_panel', None). [The simulator, K2 (a)]
- **changed** `Interactive_STM_Simulator._clear_panels (comment)`: '# The legend panel (Single Point, Line, Path): ...' becomes '(Single Point, Line, Path, Map)'. [The simulator, K2 (a)]
- **changed** `Interactive_STM_Simulator._build_ui (Map branch)`: The first set_groups call becomes set_groups([], None, extra=spec_cell, extra_side=self._legend_build()): Map now has the legend panel as the block's side item. [The simulator, K2 (a)]
- **changed** `Interactive_STM_Simulator._build_map_block`: Its set_groups call also passes extra_side=self._legend_panel. [The simulator, K2 (a)]
- **added** `import cmap_shading`: `from . import cmap_shading` (before `from . [The simulator: the addendum's integration (the sets from cmap_shading.py)]
- **removed** `CMAP_SETS (module constant) and its comment block`: The simulator's own colour-map set table is gone. [The simulator: the addendum's integration (simulator.py's own set table removed)]
- **changed** `MAP_CMAP (module constant) and its comment`: Kept, now built as cmap_shading.make_cmap(cmap_shading.cmap_spec('default 1', 'map'), plt.to_cmap, plt.LinearSegmentedColormap.from_list, "custom_map"): the same midnightblue to lightskyblue ramp, made from the module's list. [The simulator: the addendum's integration (MAP_CMAP kept as default 1's Map entry built from the module's list)]
- **added** `_CMAPS (module dict) with comment`: A store of each colour map, made once and kept so _on_cmap_set can still tell a panel's role by identity, as make_cmap's docstring asks. [The simulator: the addendum's integration (cmap_spec / make_cmap)]
- **added** `_window_cmap(set_name, role, shading=False) (module-level def)`: Machinery docstring. Returns the qtplot Colormap for cmap_spec(set_name, role, shading). [The simulator: the addendum's integration (cmap_spec / make_cmap)]
- **changed** `Interactive_STM_Simulator.__init__: cmap_set comment`: 'CMAP_SETS' becomes 'cmap_shading.SETS'. [The simulator: the addendum's integration (set table removed)]
- **added** `Interactive_STM_Simulator.__init__: shade_state, _shade_cache (+ comment)`: shade_state = cmap_shading.BoxState(), which starts with Contour on, Shading off, Height and is kept through view changes. [The simulator: the addendum's integration (start Contour on, Shading off, Height)]
- **changed** `_make_controls: the 'cmap set' radio and its comment`: labels=list(cmap_shading.SET_NAMES) instead of list(CMAP_SETS); [The simulator: the addendum's integration (the sets from cmap_shading.py)]
- **added** `_make_controls: self.shade_box (+ comment)`: cmap_shading.build_box(fig, controls, on_change=self._on_shade_box, state=self.shade_state, title=None). [The simulator: the addendum's integration (the Contour / Shading box, cmap_shading.build_box with the figure and the controls module, untitled)]
- **added** `_make_controls: the strip entry 'contour / shading' (+ comment)`: _strip_boxes['contour / shading'] = self.shade_box.qt, added to the bottom strip after 'cmap norm'. [The simulator: the addendum's integration (the box in Map only, like 'cmap norm')]
- **added** `_build_ui: the strip box's visibility`: _strip_boxes['contour / shading'].setVisible(mode == 'Map'). [The simulator: the addendum's integration (Contour / Shading box in Map only)]
- **changed** `_draw_level_line (+ docstring)`: After the line is made, it is shown when shade_state.contour is on and hidden when it is off. [The simulator: the addendum's integration (Contour off: the topo colour bar's level line hidden)]
- **changed** `_draw_contour_artists (+ docstring)`: The old traces are still removed. With shade_state.contour off, no new traces are drawn. [The simulator: the addendum's integration (Contour on: the slice as today; off: no traces on the energy maps)]
- **changed** `_choose_orientation (+ docstring)`: When the orientation changed (best != cur), it calls _reshade_maps() and then fig.canvas.draw_idle(), so a Relief shade is made again for the new panel_turn. [The simulator: the addendum's integration, K11 (a) (panel_turn 90 when the cell is drawn turned, else 0)]
- **changed** `_draw_map_slice(ax, slice_data, diverging, abs_m, v_min, v_max, gamma) (+ docstring)`: The signature now takes the kept signature: diverging, |m| and the LDOS γ, in place of use_diverging/has_mag and self.show_mag/self.ldos_gamma. [The simulator: the addendum's integration (display only; Shading off: the set's maps and filled contours come back)]
- **added** `_draw_shaded_slice(ax, slice_data, diverging, abs_m, v_min, v_max, gamma, shade)`: Machinery. Draws the energy map as an RGBA mesh. [The simulator: the addendum's integration (Shading on: RGBA mesh through qtplot's pcolormesh, one colour per unique grid point, quads centred, tiled by the cell offsets; the shading map through the maps' own norm, then the shading with the L* stretch)]
- **added** `_map_shade()`: Machinery. Returns None while Shading is off. [The simulator: the addendum's integration (Height: the LDOS topo height; Relief: relief.m defaults, periodic=True, panel_turn per K11 (a))]
- **changed** `_map_cmap(use_diverging, has_mag) (+ docstring)`: Now returns _map_role_cmap(use_diverging, show_mag and has_mag). [The simulator: the addendum's integration (the colour bars show the shading map unshaded)]
- **added** `_map_role_cmap(diverging, abs_m)`: Machinery. Returns self._cmap(cmap_shading.map_role(diverging, abs_m), shading=shade_state.shading): the set's map, or while Shading is on CET-I1 / CET-D11 / CET-I3. [The simulator: the addendum's integration (the shading map: CET-I1, Mag signed CET-D11, Mag |m| CET-I3)]
- **changed** `_cmap(role, shading=False) (+ docstring)`: Returns _window_cmap(self.cmap_set, role, shading=shading) instead of CMAP_SETS[self.cmap_set][role]. [The simulator: the addendum's integration (the sets from cmap_shading.py)]
- **changed** `_on_cmap_set (+ docstring)`: same_role now searches cmap_shading.ROLES through _window_cmap(old, role) instead of CMAP_SETS[old]. [The simulator: the addendum's integration (set table removed)]
- **changed** `_set_map_slice (+ docstring)`: Now only works out sig = (diverging, show_mag and has_mag, v_min, v_max, ldos_gamma) and calls _show_map_slice. [The simulator: the addendum's integration (Shading on / off drawing)]
- **added** `_show_map_slice(k, ax, slice_data, sig)`: Machinery. Redraws map k when the data, sig or the shade (identity of _map_shade()) differ, else does nothing. [The simulator: the addendum's integration (Shading on: RGBA mesh; Shading off: filled contours back)]
- **added** `_reshade_maps()`: Machinery. In Map, every map drawn now is shown again from its kept _map_state data and sig under the current shading. [The simulator: the addendum's integration (display only: only the energy maps are redrawn)]
- **added** `_on_shade_box(state)`: Machinery; the box's on_change. It runs whether RUN is on or not. [The simulator: the addendum's integration (contour toggle; at most one on, both may be off; display only)]
- **changed** `Interactive_STM_Simulator (class docstring), the left column's list`: 'a left column with the view radio, the broadening box, ...' now reads 'a left column with the view radio, the 'cmap set' choice, the broadening box, ...', in the code's order (_make_controls: the view radio, cmap set, broadening, the hook boxes, RUN with the toggles). [The simulator: Texts (K9's rule): the class docstring's list of the left column (the cmap set choice)]
- **changed** `Interactive_STM_Simulator (class docstring), the bottom strip's list`: 'the bottom strip of number boxes (Cells, Path Pts, Markers, E Pts, E Min, E Max, LDOS γ)' now continues '... [The simulator: Texts (K9's rule): the class docstring's list of Map's boxes (the Contour / Shading box)]
- **changed** `Interactive_STM_Simulator._draw_spectrum (docstring)`: The legends are gone: 'the legends and the hover labels made' is now 'the hover labels made'. [The simulator: Texts (K9's rule): _draw_spectrum's docstring (no legends, no atom levels)]
- **changed** `Interactive_STM_Simulator._fill_map (docstring)`: 'its contour set drawn again only when its slice or its colour scale changed' was false with Shading on. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (the addendum's integration)]
- **changed** `Interactive_STM_Simulator._update_all (docstring, last sentence)`: 'the periodic images of a map are one raster drawn at every cell offset (tricontourf offsets=: moved, never re-imaged)' now continues ', or with the box's Shading on one RGBA mesh at each cell offset (_draw_shaded_slice)'. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (the addendum's integration)]
- **changed** `Interactive_STM_Simulator._build_ui, Map branch comment on the topo colour bar`: 'the Map Topo's iso-line at that level goes on every energy map.' was false with Contour off. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (Contour off: no traces)]
- **changed** `Interactive_STM_Simulator._update_all, Map branch comment above the _draw_contour_artists call`: '# The contour slice on the maps: drawn again when the maps are new or the Map Topo's heights changed.' now has a second line: '(nothing drawn while the box's Contour is off: _draw_contour_artists).' The code is unchanged. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (Contour off: no traces)]
- **changed** `Unified_STM_Simulator._mag_is_signed (docstring)`: '(needs bwr)' is now '(needs a diverging colour map, the 'mag' role's)'. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (the colour-map sets and the shading maps)]
- **changed** `Interactive_STM_Simulator._on_topo_height_rerun` (review fix): After _recompute_global_topo() and _broadening_topo_dirty = False, it now sets self.cached_bias_energy_map = self.cached_d_topo_map = None, with a one-line comment. [The simulator: O2 (a rerun recomputes at the kept energies; the drag's same tip heights, the LDOS topo)]
- **changed** `Interactive_STM_Simulator._update_all, the comment above Map's typed / reform / needs_ldos tests` (review fix): The needs_topo list '(needs_topo: the bias, the decay model, LDOS h, a broadening)' now ends '..., a broadening, a Topo h rerun)'. [The simulator: Texts (K9's rule), for the O2 fix above]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/ldos_decomp.py`

- **added** `_column_orbitals(orbitals)`: New module-level def. It returns the names of the displayed LDOS columns in column order, each once: list(dict.fromkeys(_orbit_base(o) for o in orbitals)). [The simulator, K1 (a)]
- **changed** `all_labels`: Level 3's width labels are now '<element> <orbital>' over _column_orbitals(orbitals) instead of every name in orbitals, so they hold no '_up' / '_down' names. [The simulator, K1 (a) (the legend's width labels follow)]
- **changed** `Decomp.orbital_columns`: Loops over enumerate(_column_orbitals(orbitals)) instead of enumerate(orbitals). [The simulator, K1 (a)]
- **changed** `Decomp.curves (level-3 branch)`: The line p_color = _lighten_color(color, 0.3) if orb.endswith('_up') else color is removed. [The simulator, K1 (a) (no _up / _down entries, in the spectrum)]
- **changed** `_lighten_color (docstring)`: '(the '_up' orbital curves)' was false after K1, since no '_up' curves remain. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (K1 (a))]
- **changed** `Decomp.curves (docstring, the `integral` parameter)`: '... + 1e-15), else None.' is now '... + 1e-15), else None (also under 'Norm' when the spectra given are already divided: in a Line / Path drag the simulator divides them by its kept Norm factors).' Since O1, _draw_spectrum passes integral=None under Norm when line_norm is given. [The simulator: Texts (K9's rule): any other text the changes make false, stale words only (O1)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/controls.py`

- **changed** `MapBlock (class docstring)`: The set_groups signature now shows extra_side=None. [The package qtgui, controls.py, K2 (a)]
- **added** `MapBlock.__init__: self._extra_side`: New attribute, None at start: the extra cell's side item, which set_groups sets. [The package qtgui, controls.py, K2 (a)]
- **changed** `MapBlock._metrics`: New metrics key 'side_w' from _side_w(), measured once per layout pass like the bar widths. [The package qtgui, controls.py, K2 (a)]
- **changed** `MapBlock.set_groups`: New keyword extra_side=None. It is stored only when extra is not None, otherwise None. [The package qtgui, controls.py, K2 (a)]
- **added** `MapBlock._side_w`: Returns the side item's fixed width as float(minimumWidth()), or None when there is no side item. [The package qtgui, controls.py, K2 (a)]
- **added** `MapBlock._extra_col`: Returns the extra cell's column at n columns: len(last partition's maps) % n, or 0 with no partitions. [The package qtgui, controls.py, K2 (a)]
- **added** `MapBlock._right`: Returns the right edge of the block's cells: the larger of the map columns' edge (n*(cw+cbw)+(n-1)*sp) and the side item's right edge (extra_col*(cw+cbw+sp)+cw+sp+side_w). [The package qtgui, controls.py, K2 (a)]
- **changed** `MapBlock._size_at`: With a side item, the cell width is also capped so the side item and the bars fit inside W: cw <= (W - bar_w - sp - side_w - c*(cbw+sp))/(c+1), with c = _extra_col(n). [The package qtgui, controls.py, K2 (a)]
- **changed** `MapBlock._arrange`: Places the side item at (x_extra + cw + sp, the row's top, its fixed width, the row's height) and does not add it to the align rows. [The package qtgui, controls.py, K2 (a)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/qtplot.py`

- **changed** `module docstring, Colour maps paragraph`: "matplotlib, when installed, is used only to read ..." now reads "matplotlib is used only to read ...". [The package qtgui, qtplot.py: K9 ('when installed' in the module docstring's Colour maps paragraph)]
- **added** `_NAVIGATING (module constant, before class _ViewBox)`: A counter [0], with a comment. It is non-zero while a view box handles a drag or the wheel. [The package qtgui, qtplot.py: K6 (b)]
- **changed** `_ViewBox class docstring`: Now adds that the views its drags and its wheel change are marked as changed with the toolbar (_NAVIGATING). [The package qtgui, qtplot.py: K6 (b)]
- **added** `_ViewBox.mouseDragEvent(self, ev, axis=None)`: Calls pyqtgraph's mouseDragEvent with _NAVIGATING raised, through try/finally. [The package qtgui, qtplot.py: K6 (b)]
- **added** `_ViewBox.wheelEvent(self, ev, axis=None)`: Calls pyqtgraph's wheelEvent with _NAVIGATING raised, through try/finally. [The package qtgui, qtplot.py: K6 (b)]
- **changed** `QuadMesh class docstring`: Now says that an (M, N, 3) RGB or (M, N, 4) RGBA C is painted with its colours as given, with no norm or colour map (_take_c; [The package qtgui, qtplot.py: K10 (a)]
- **changed** `QuadMesh.__init__`: C is taken by _take_c, giving (_C, _rgba). [The package qtgui, qtplot.py: K10 (a)]
- **added** `QuadMesh._rgba (instance attribute)`: None for a scalar mesh. For an RGB(A) mesh: the (M, N, 4) float64 RGBA 0-1 the mesh is painted with; [The package qtgui, qtplot.py: K10 (a)]
- **added** `QuadMesh._take_c(C, alpha=None) (staticmethod)`: Takes C as matplotlib's pcolormesh does. ndim < 3: float64 values, masked cells NaN, with None. [The package qtgui, qtplot.py: K10 (a)]
- **changed** `QuadMesh._paint`: New RGBA branch: colours = (_rgba.reshape(-1, 4) * 255 + 0.5).astype(uint8), with no norm or colour map (rounded as Colormap rounds bytes). [The package qtgui, qtplot.py: K10 (a)]
- **changed** `QuadMesh.set_array`: Gained a docstring. C goes through _take_c. [The package qtgui, qtplot.py: K10 (a)]
- **changed** `QuadMesh.get_array docstring`: Now says it also returns the given RGB(A) array (masked_invalid of _C); [The package qtgui, qtplot.py: K10 (a)]
- **changed** `PanelCell.bind, comment on new_measure_pass()`: "the cell's shape and room may be another's now" now reads "the cell's shape may be another's now (its room follows at the next arrangement)". [The package qtgui, qtplot.py: K9 (PanelCell.bind's comment)]
- **added** `Axes.__init__: _navigate, _nav_moved, _nav_pending, _nav_key and the sigRangeChanged connection`: Set next to _home: _navigate = True, _nav_moved = False, _nav_pending = False, _nav_key = f"view {id(self)}", then self.vb.sigRangeChanged.connect(self._nav_follow), with comments. [The package qtgui, qtplot.py: K8 (a), K6 (b)]
- **changed** `Axes.set_navigate(b)`: Was a no-op. Now stores bool(b) in _navigate and applies the toolbar's current mode again (_apply_toolbar_mode); [The package qtgui, qtplot.py: K8 (a)]
- **changed** `Axes.get_navigate`: Returns _navigate (was always True). [The package qtgui, qtplot.py: K8 (a)]
- **added** `Axes._nav_follow(self, *args)`: The slot for sigRangeChanged. While _NAVIGATING is set it marks _nav_moved, unless the axes is a colour bar or set_navigate(False). [The package qtgui, qtplot.py: K6 (b)]
- **added** `Axes._nav_line`: While _nav_moved, writes the window line fig.set_info_line(_nav_key, "view of <name>: x a to b, y c to d"): limits from get_xlim / get_ylim, :.5g. [The package qtgui, qtplot.py: K6 (b)]
- **added** `Axes._nav_clear`: When _nav_moved: clears it and removes the window line (set_info_line(_nav_key, '')). [The package qtgui, qtplot.py: K6 (b)]
- **changed** `Axes.remove`: Calls self._nav_clear() after fig._set_info(self, '') (with a comment), so a removed panel's view line goes too. [The package qtgui, qtplot.py: K6 (b)]
- **changed** `Axes.clear`: Calls self._nav_clear() after self._home = None (with a comment): the window resets the limits. [The package qtgui, qtplot.py: K6 (b) (removed when the window sets the limits again)]
- **changed** `Axes.set_title`: After _title_full is set: if _nav_moved, calls _nav_follow(), so the view line names the panel by its new title. [The package qtgui, qtplot.py: K6 (b) (giving the panel: its title)]
- **changed** `Axes._store_home`: Ends with self._nav_clear() (with a comment). [The package qtgui, qtplot.py: K6 (b) (removed when the window sets the limits again)]
- **changed** `Axes.autoscale`: With enable, calls self._nav_clear() after enableAutoRange (with a comment). [The package qtgui, qtplot.py: K6 (b) (removed when the window sets the limits again)]
- **changed** `Axes._home_view`: Returns at once also for set_navigate(False) (Home leaves it alone). [The package qtgui, qtplot.py: K8 (a) (Home leaves it alone), K6 (b) (removed on Home)]
- **changed** `Axes._apply_toolbar_mode`: New branch after the colour-bar one: when not _navigate, sets vb.setMouseEnabled(False, False) and vb.setMouseMode(PanMode), then returns (with a comment: pyqtgraph's rectangle zoom ignores mouseEnabled). [The package qtgui, qtplot.py: K8 (a) (the wheel and drags leave it alone)]
- **changed** `Axes.pcolormesh`: Gained a docstring (scalar C through norm and colour map, or RGB / RGBA of 0-1 floats or 0-255 ints painted as given). [The package qtgui, qtplot.py: K10 (a)]
- **changed** `NavigationToolbar.__init__: Save tooltip`: Now 'Save the figure: PDF / EPS / SVG for publication (matplotlib), PNG / TIFF / SVG as on screen, or the curves as CSV' (was '... [The package qtgui, qtplot.py: K9 (the Save button's tooltip)]
- **changed** `FigureManager class docstring`: '(set_info: the details cut from the titles; [The package qtgui, qtplot.py: K9 (the FigureManager docstring)]
- **changed** `Figure class docstring`: '(_design_px) is the reference of both and the size of the publication export.' now reads '(_design_px) is the reference of both.' [The package qtgui, qtplot.py: K9 (the Figure docstring)]
- **changed** `QuadMesh.__init__, the shading='auto' centres test` (review fix): For 2-D X the test is now `X.shape == C.shape[:2]` (was `X.shape == C.shape`). [The package qtgui, qtplot.py: K10 (a) (QuadMesh and Axes.pcolormesh take an (M, N, 3) RGB or (M, N, 4) RGBA C, as matplotlib's)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/mplexport.py`

- **added** `_artist, QuadMesh branch: `if a._rgba is not None` (RGB(A) mesh)`: An RGB(A) QuadMesh is rebuilt as `_turned(m.pcolormesh(a._X, a._Y, a._rgba, shading='flat', zorder=a.zorder), swap, m)`: the colours as painted (RGBA 0..1), with no cmap, norm or alpha. [The package qtgui, mplexport.py: K10 (a) (an RGBA QuadMesh rebuilt with matplotlib's pcolormesh taking the RGBA array)]
- **changed** `_artist docstring, the QuadMesh entry`: 'QuadMesh -> pcolormesh (NaN cells masked)' now reads 'QuadMesh -> pcolormesh (NaN cells masked; [The package qtgui, mplexport.py: K10 (a)]
- **changed** `_axes, legend branch `if kw.get('title')``: `lk['title_fontsize'] = lk['fontsize']` is replaced by `lk['title_fontproperties'] = {'size': lk['fontsize'], 'weight': 'bold'}`, so the title keeps the legend's on-screen size and is now bold, as qtplot's Axes.legend draws it (tf.setBold(True)). [The package qtgui, mplexport.py: the legend title written bold, as on screen]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/__init__.py`

- **changed** `module docstring, the mplexport.py line of the module list`: '(PDF, EPS, SVG, PNG, TIFF)' becomes '(PDF, EPS, editable SVG;' with a continuation line ' the PNG and TIFF are the window image)'. [The package qtgui, __init__.py: K9 (the mplexport line: mplexport writes the PDF, EPS and editable SVG; the window image writes the PNG and TIFF; an exception to ruling 1)]

## `qtgui/qtplot.py`

- **added** `_NAVIGATING (module constant, before class _ViewBox, line 1996)`: A one-item counter list with its two-line comment, raised while a view box handles a drag or the wheel. [The root qtgui, qtplot.py: K6 (b) as in the package copy]
- **changed** `_ViewBox docstring`: Now a three-line docstring that adds 'the views its drags and its wheel change are marked as changed with the toolbar (_NAVIGATING)'. [The root qtgui, qtplot.py: K6 (b)]
- **added** `_ViewBox.mouseDragEvent(self, ev, axis=None)`: Calls pyqtgraph's mouseDragEvent with _NAVIGATING raised for the call (try/finally). [The root qtgui, qtplot.py: K6 (b)]
- **added** `_ViewBox.wheelEvent(self, ev, axis=None)`: Calls pyqtgraph's wheelEvent with _NAVIGATING raised for the call (try/finally). [The root qtgui, qtplot.py: K6 (b)]
- **added** `Axes.__init__: _nav_moved, _nav_pending, _nav_key, and the vb.sigRangeChanged -> _nav_follow connection`: New attributes, set after self._home = None with the package's two-line comment: _nav_moved=False, _nav_pending=False, _nav_key=f"view {id(self)}". [The root qtgui, qtplot.py: K6 (b)]
- **added** `Axes._nav_follow(self, *args)`: The package's method. During _NAVIGATING it marks the axes as moved unless it is a colour bar or get_navigate() is False. [The root qtgui, qtplot.py: K6 (b)]
- **added** `Axes._nav_line(self)`: The package's method, identical. Sets fig.set_info_line(_nav_key, "view of <title or 'panel k'>: x a to b, y c to d") (:.5g), or '' when the axes is not marked. [The root qtgui, qtplot.py: K6 (b)]
- **added** `Axes._nav_clear(self)`: The package's method, identical. When the axes is marked, it unmarks it and removes its view line. [The root qtgui, qtplot.py: K6 (b)]
- **added** `comment block '# --- the view changed with the toolbar's Pan or Zoom: a line in the window's info panel ---'`: The package's section comment, placed before _nav_follow, after get_navigate. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `Axes.remove`: Calls self._nav_clear() after fig._set_info(self, ''), with the package's comment. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `Axes.clear (cla)`: Calls self._nav_clear() after self._home = None, with the package's comment. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `Axes.set_title`: After self._title_full = full, it calls self._nav_follow() when _nav_moved is set, so the view line follows the new title. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `Axes._store_home`: Calls self._nav_clear() after storing _home. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `Axes.autoscale`: In the enable branch, calls self._nav_clear() after enableAutoRange. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `Axes._home_view`: Calls self._nav_clear() after the colour-bar return and before home_callback / _home. [The root qtgui, qtplot.py: K6 (b)]
- **changed** `module docstring, Colour maps paragraph`: 'matplotlib, when installed, is used' becomes 'matplotlib is used', as in the package copy. [The root qtgui, qtplot.py: "when installed"]
- **changed** `NavigationToolbar.__init__: the Save action's tooltip`: Replaces sm4_browser's two-line wording with the adf1e25 text plus the package copy's stale-words edit: 'Save the figure: PDF / EPS / SVG for publication (matplotlib), PNG / TIFF / SVG as on screen, or the curves as CSV'. [The root qtgui, qtplot.py: K9 texts]
- **changed** `FigureManager docstring`: Replaces sm4_browser's wording ('settings section', 'wrapped at its width', 'hidden only until its first text ...') with the adf1e25 text plus the package copy's stale-words edit: '(set_info: Figure.info_text, the details cut from the titles, the window's own lines and the settings of its controls; [The root qtgui, qtplot.py: K9 texts]
- **changed** `Figure docstring`: Removes sm4_browser's sentence 'A saved file is the window at its size on screen (savefig).' The sentence now reads '(_design_px) is the reference of both.', the adf1e25 text with 'and the size of the publication export' dropped, as in the package copy. [The root qtgui, qtplot.py: K9 texts]

## `qtgui/mplexport.py`

- **changed** `_axes: the legend-title keywords (lk) in the legends loop`: The line lk['title_fontsize'] = lk['fontsize'] is removed (matplotlib refuses it together with title_fontproperties). [The root qtgui, mplexport.py: the legend title bold]

## `qtgui/__init__.py`

- **changed** `module docstring: the mplexport.py line`: '(PDF, EPS, SVG, PNG, TIFF)' becomes two lines: '(PDF, EPS, editable SVG;' and, indented 16 spaces, 'the PNG and TIFF are the window image)'. [The root qtgui, __init__.py: K9's mplexport line, as the package copy's]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py`

- **changed** `StmDecompSession class docstring`: 'then simulator instances come and go per σ commit' now reads 'then one simulator instance; [The other files: sim_wrapper.py, O14 (class docstring clause)]
- **changed** `StmDecompSession.launch docstring`: 'Prepare the working directory and start the simulator' now reads 'Start the simulator on the run directory'. [The other files: sim_wrapper.py, O14 (launch's docstring)]
- **changed** `StmDecompSession._attach_widgets._on_sigma comment`: The stated reason 'The commit tears the window down, so it runs after ...' now reads 'The commit only sets σ for the next Send press; [The other files: sim_wrapper.py, O14 (_on_sigma's reason)]
- **changed** `section comment above commit_sigma`: '# σ commit: re-smear → new σ-tagged dir → capture → tear down → re-instantiate → restore.' (two lines) is now one line: '# σ commit: sets the MO spread smearing for the next Send press.' commit_sigma's 'Real-DOSCAR mode' comment and its print are unchanged. [The other files: sim_wrapper.py, O14 (the σ-commit section comment)]
- **changed** `StmDecompSession.send_to_decomposition, Line / Path branch`: colors.append(sim.m_colors[i % len(sim.m_colors)]) is now colors.append(sim.line_marker_colors[i]), with the inline comment '# the marker's kept colour'. [The other files: sim_wrapper.py, K3 (a)]

## `LDOS_spread_decomp/mo_ldos_decomp/run_stm_decomp.py`

- **changed** `sim_kwargs use_decay comment`: '# Chen barrier model (turn OFF before sending to decomposition — the button enforces it)' now reads '# the exact tilted barrier for topography convergence and LDOS together (turn OFF before sending to decomposition — the button enforces it)'. [The other files: run_stm_decomp.py, O15]
- **changed** `sim_kwargs reuse_cache comment`: '# per-σ-dir topo caches' now reads '# the run directory's topo caches'. [The other files: run_stm_decomp.py, O15]

## `LDOS_spread_decomp/stm_from_dft_opt/run_DFT_STM.py`

- **changed** `Interactive_STM_Simulator(...) erange comment`: '# initial energy slider range' now reads '# initial energy box range'. [The other files: run_DFT_STM.py, O16]
- **changed** `run_interactive(...) path comment`: 'the "Path Pts" slider adds/removes nodes' now reads 'the "Path Pts" box adds/removes nodes'. [The other files: run_DFT_STM.py, O16]

## `LDOS_spread_decomp/mo_ldos_decomp/run_wfn_plane_slice.py`

- **added** `run(), the `if save_png:` branch: plt.show(block=False)`: Added right before fig.savefig, inside the branch, so the window is shown and laid out first. [The other files: K5 (a)]
- **changed** `run(), the fig.savefig(png, dpi=300) comment`: '# the publication export (qtgui.mplexport: matplotlib writes the file)' now reads '# the window image (qtgui.qtplot: the window at 720 dpi; [The other files: K5 (a), the save comments (K9, stale words only)]
- **changed** `module docstring, the Qt layer paragraph`: 'the PNG is written through the layer's publication export (matplotlib) before the box is attached' now reads 'the PNG is written as the layer's window image (the window at 720 dpi) before the box is attached'. [The other files: K5 (a), the save comments (K9, stale words only)]

## `LDOS_spread_decomp/MOPmatcher_mem/run_vacupot_plotter.py`

- **added** `main(), the `if SAVE_PATH:` branch: plt.show(block=False)`: Added before fig.savefig(SAVE_PATH, dpi=SAVE_DPI), which is unchanged, as in sm4_browser's tools/rebuild_panels.py. [The other files: K5 (a)]
- **changed** `comment above SAVE_PATH / SAVE_DPI`: 'a publication file through matplotlib (PDF, EPS, SVG, or PNG / TIFF at SAVE_DPI), as the window's Save button.' now reads 'a publication file through matplotlib (PDF, EPS, SVG), or the window at 720 dpi (PNG / TIFF), as the window's Save button; [The other files: K5 (a), the save comments (K9, stale words only)]

## `LDOS_spread_decomp/stm_from_dft_opt/README.md`

- **changed** `'The window' paragraph, the curve-panels clause`: 'the curve panels (Tip Height and Partitioned LDOS, side by side or stacked by their room)' now reads 'the curve panels (Tip Height and Partitioned LDOS with its legend panel at its right, side by side or stacked by their room)'. [READMEs: stm README, K9 (the layout text on the curve panels)]
- **changed** `'The window' paragraph, the save line`: Now reads 'Save writes PDF, EPS or SVG through matplotlib (publication files), a PNG or TIFF (the window at 720 dpi) or an SVG as on screen, or the curves as CSV'. [READMEs: stm README, K9 (the save lines)]
- **changed** `control table, 'cmap norm' row, its colour-map sentence`: Now reads 'The maps' colour map is the Map map of the window's colour-map set (the constructor's `cmap_set`): `MAP_CMAP` in `default 1`, the sm4_Browser's (`custom_map`, midnight blue to light sky blue), `CET-L6` in `default 2`'. [READMEs: stm README, K9 (the 'cmap norm' row's cmap_map)]
- **changed** `paragraph after the control table, the click text`: Now reads 'a left click on a curve of the Partitioned LDOS panel, or on its entry in the legend panel at its right, descends a level (partition, atom types, orbitals) and a right click in the panel or in its legend panel, or the legend panel's Back button, ascends'. [READMEs: stm README, K9 (the click text)]
- **changed** `module table, `simulator.py` row`: Split into a **Physics** part and a **Machinery** part. [READMEs: stm README, O10 (b)]
- **added** `module table, `ldos_decomp.py` row`: New row: the decomposition of the Partitioned LDOS panel (display machinery, no physics). [READMEs: stm README, K9 (the module table's ldos_decomp.py row)]
- **changed** `closing layout paragraph, the curve-panel sentence`: The legend panel is added in each view. Line / Path: 'the Partitioned LDOS, its legend panel of one fixed width at its right'. [READMEs: stm README, K9 (the layout text on the curve panels; K2 (a) as the brief states it)]
- **changed** `control table, 'cmap norm' row, the colour-map sentence's bracket` (review fix): '(the constructor's `cmap_set`)' now reads '(the constructor's `cmap_set`; [READMEs: stm README, K9 (the 'cmap norm' row's cmap_map)]

## `LDOS_spread_decomp/mo_ldos_decomp/README.md`

- **changed** `'The windows' section, the save line`: Now reads 'Save writes PDF, EPS or SVG through matplotlib (every curve and text an editable object, the controls left out), a PNG or TIFF (the window at 720 dpi) or an SVG as drawn on screen, or the curves as CSV'. [READMEs: mo_ldos_decomp README save line (K9)]

## `LDOS_spread_decomp/MOPmatcher_mem/README.md`

- **changed** `Run scripts table, `run_vacupot_plotter.py` row`: '`SAVE_PATH` writes a publication file (PDF, EPS, SVG, or PNG / TIFF at `SAVE_DPI`)' now reads '... [READMEs: MOPmatcher_mem README save lines (K9)]
- **changed** `'The plot windows' section, the save line`: Now reads 'Save writes PDF, EPS or SVG through matplotlib (publication files), a PNG or TIFF (the window at 720 dpi) or an SVG as on screen, or the curves as CSV'. [READMEs: MOPmatcher_mem README save lines (K9)]

## `README.md`

- **changed** `Dependencies, the matplotlib sentence`: 'the publication files the Save button offers (PDF, EPS, SVG, PNG, TIFF)' now reads '(PDF, EPS, SVG)'. [READMEs: root README save lines (K9)]
- **changed** `'The interactive windows', **Saving.** bullet`: Now reads 'writes PDF, EPS or SVG through matplotlib (`qtgui/mplexport.py`: the plots as on screen and the full information panel, every line and text an editable object, the controls left out), PNG / TIFF (the window at 720 dpi) / SVG as drawn on screen, or the curves as CSV; [READMEs: root README save lines (K9)]

## `PROCAR_tools/README.md`

- **changed** ``procar_dos.py` section, the validation sentence`: Every band: the ion sum of the total block against its `tot` row. [READMEs: PROCAR_tools README, O17 (b)]
- **changed** ``procar_dos.py` section, the spot-check stride` (review fix): The stride now reads every `max(1, nbands // max(spot_check_bands, 1))`-th band, as procar_dos.py:152 computes `spot_every`. [READMEs: PROCAR_tools README, O17 (b)]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/topo_decomp_window.py`

- **changed** `TopoDecompWindow.__init__, the transparent-patch comment`: 'the publication export (the toolbar's Save, qtgui.mplexport)' now reads '(the toolbar's Save as PDF, EPS or editable SVG, qtgui.mplexport)'. [READMEs / other files: topo_decomp_window.py's export comment (K9)]

## `convergence/forces.py`

- **added** `forces.<locals>.view_line`: Nested def. It sets the window line through fig.set_info_line('3d view', ...). [convergence/forces.py, K6 (b)]
- **changed** `forces.<locals>.apply_limits`: Calls view_line() after it sets the limits and the box aspect. [convergence/forces.py, K6 (b)]
- **changed** `forces.<locals>.on_release3d`: Calls view_line() after it stores the panned limits in state['lims']. [convergence/forces.py, K6 (b)]
- **added** `forces.<locals>.on_motion3d`: New canvas3d motion handler that calls view_line(). [convergence/forces.py, K6 (b)]
- **added** `canvas3d.mpl_connect('motion_notify_event', on_motion3d)`: Connects the new handler, next to the existing canvas3d connections. [convergence/forces.py, K6 (b)]

