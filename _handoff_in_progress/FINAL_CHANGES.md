# Final and close-out passes: the per-file change list (rule 5)

From the agents' and fixers' reports: the final pass (base df9baf2: F2, F3, F4, Line's two rows, the fill-row cache,
texts) and the close-out (base f9123a3: the consequences of F4). Additional changes: none reported.

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py`

- **changed** `Interactive_STM_Simulator.__init__: the cached_ld_up / cached_ld_dn / cached_eg line (inline comment)` (final): Adds a comment: the Line / Path points' LDOS and its energies, kept together (sim_wrapper reads them). [F4 (a): Line / Path keep their data with their energies]
- **changed** `Interactive_STM_Simulator.__init__: the cached_marker_coords / cached_spec_ldos line (inline comment)` (final): Adds a comment: Single Point's marker spectra (Map keeps its own, map_spec_*). [F4 (a): Map keeps its own marker spectra]
- **changed** `Interactive_STM_Simulator.__init__: the cached_spec_current comment` (final): Stale words: 'the Single Point / Map markers' becomes 'the Single Point markers'. [F4 (a) (stale words)]
- **changed** `Interactive_STM_Simulator.__init__: the comment block over Map's own state` (final): 'apart from Line / Path's' becomes 'apart from the other views''; [F4 (a): Map keeps its own copies]
- **added** `Interactive_STM_Simulator.__init__: attribute map_z` (final): None: the LDOS-topo heights Map's maps were computed at (a copy). [F4 (a): Map keeps the LDOS-topo heights its maps were computed at]
- **added** `Interactive_STM_Simulator.__init__: attributes map_spec_ldos, map_spec_ldos_gpu, map_marker_coords` (final): None: Map's own marker spectra (NumPy and GPU copy) and the markers they were computed for. [F4 (a): Map's own marker spectra]
- **added** `Interactive_STM_Simulator.__init__: attributes map_spec_current, map_spec_current_E` (final): None: the Map markers' current from the spectra's pass and its samples' GPU energies (for Feenstra on Map's own spectra). [F4 (a): Map's own marker spectra]
- **changed** `Interactive_STM_Simulator._feenstra_iv: new branch which == 'map_spec'` (final): Returns feenstra.iv_ratio(map_spec_current_E, map_spec_current), the Map markers' own I/V; [F4 (a): Map's own marker spectra]
- **changed** `Interactive_STM_Simulator._feenstra_iv (docstring)` (final): Adds 'map_spec' for the Map markers (map_spec_current at map_spec_current_E); [F4 (a)]
- **changed** `Interactive_STM_Simulator._contour_segments` (final): Heights read from Map's kept map_z (was getattr(self, 'current_z_map', None)). [F4 (a): the LDOS-topo heights used for the contour slice]
- **changed** `Interactive_STM_Simulator._contour_segments (docstring)` (final): 'on the Map Topo's heights (current_z_map, ...' becomes 'on the LDOS-topo heights Map's maps were computed at (map_z, Map's own, ...'. [F4 (a)]
- **changed** `Interactive_STM_Simulator._map_shade` (final): z = self.map_z (was self.current_z_map): the shade is made from the heights Map's maps were computed at. [F4 (a): the LDOS-topo heights used for the shading]
- **changed** `Interactive_STM_Simulator._map_shade (docstring)` (final): 'the LDOS topo's heights (current_z_map, the Map Topo's, ...' becomes 'the LDOS-topo heights Map's maps were computed at (map_z, Map's own, ...'. [F4 (a)]
- **changed** `Interactive_STM_Simulator._update_all: comment above Map's typed / reform / needs_ldos / needs_spec tests` (final): 'apart from Line / Path's' becomes 'apart from the other views''; [F4 (a) (stale words)]
- **changed** `Interactive_STM_Simulator._update_all: Map branch, needs_spec` (final): Now needs_ldos or map_marker_coords is None or the markers differ from map_marker_coords (Map's own record). [F4 (a): Map's own marker spectra]
- **changed** `Interactive_STM_Simulator._update_all: Map LDOS compute branch (needs_ldos and Map)` (final): Records self.map_z = np.array(self.current_z_map, copy=True), the heights the maps were just computed at. [F4 (a): Map keeps the LDOS-topo heights; Map's spectra keep their energies]
- **changed** `Interactive_STM_Simulator._update_all: Map LDOS compute branch (comment)` (final): 'with the window's displayed energies (the spectrum's) and the records they were taken with' becomes 'with the LDOS-topo heights (map_z, a copy) and the records they were taken with (the spectrum's energies are kept with Map's marker spectra, below)'. [F4 (a)]
- **changed** `Interactive_STM_Simulator._update_all: the elif needs_ldos branch (Line / Path / Single Point)` (final): Single Point no longer forms eg or writes cached_eg: its else sub-branch only sets ld_up, ld_dn = None, None. [F4 (a): Line / Path keep their data with their energies]
- **changed** `Interactive_STM_Simulator._update_all: marker-spectra compute branch (needs_spec, Single Point / Map)` (final): New branch on mode. In Map the spectra go into Map's own records: map_spec_ldos_gpu, map_spec_ldos, map_marker_coords, map_eg = energies[spec_idx] (their energies, kept with them), map_spec_current and map_spec_current_E. [F4 (a): Map's own marker spectra with their energies]
- **changed** `Interactive_STM_Simulator._update_all: spec_disp (the marker spectra as shown)` (final): In Map, Map's own map_spec_ldos_gpu, divided under Feenstra by _feenstra_iv('map_spec'). [F4 (a): Map's own marker spectra]
- **changed** `Interactive_STM_Simulator._update_all: Map display branch, the contour-slice redraw test` (final): Compares and copies _contour_z against map_z (was current_z_map). [F4 (a): the LDOS-topo heights used for the contour slice]
- **changed** `Interactive_STM_Simulator._draw_spectrum: eg (the spectrum's energies)` (final): Each view's curves are drawn over their own data's energies: map_eg in Map (Map's spectra), cached_spec_eg in Single Point, cached_eg in Line / Path. [F4 (a): every record keeps its own energies]
- **changed** `Interactive_STM_Simulator._draw_spectrum (docstring)` (final): 'drawn over the displayed energies: in Map Map's own (map_eg), in the other views cached_eg' becomes 'drawn over the energies of the data they come from: in Map Map's own marker spectra's (map_eg), in Single Point its marker spectra's (cached_spec_eg), in Line / Path the line's (cached_eg)'. [F4 (a)]
- **changed** `Interactive_STM_Simulator._has_last_run` (final): spec is map_spec_ldos_gpu in Map and cached_spec_ldos_gpu otherwise. [F4 (a): returning to Map (RUN off too) shows Map as last plotted from its own copies]
- **changed** `Interactive_STM_Simulator._has_last_run (docstring)` (final): Single Point 'its' marker spectra; Map 'its own marker spectra (map_spec_ldos_gpu)'. [F4 (a)]
- **changed** `Interactive_STM_Simulator._update_all: the Map LDOS-topo branch (under 'elif self.mode == 'Map'' in 'if needs_topo:')` (final): The Map Topo's cell offsets are now computed once into a local 'offs' (the same self._cell_offsets(n) as before) and passed to the tricontourf call. [F2 (a)]
- **added** `Interactive_STM_Simulator._update_all: the Map LDOS-topo branch, the atoms (if self.show_atoms)` (final): After the clear and the LDOS topo, the atoms are drawn on ax_map again as the full refresh draws them: one scatter per atom type over every image, s=10, tab10 colour, alpha 0.3, zorder 2. [F2 (a)]
- **added** `Interactive_STM_Simulator._update_all: the Map LDOS-topo branch, the unit cell (if self.show_unit_cell)` (final): The unit cell is drawn on ax_map again as the full refresh draws it: cyan, lw 2.0, zorder 4, label 'Unit Cell'. [F2 (a)]
- **added** `Interactive_STM_Simulator._update_all: the Map LDOS-topo branch, the title` (final): ax_map.set_title(f"Map Topo: bias {self._bias_sample()[1]:.3f} V, height {self.ldos_height} Å"). [F2 (a)]
- **added** `Interactive_STM_Simulator._update_all: comment in the Map LDOS-topo branch` (final): A two-line comment: what the full refresh drew on the Map Topo (the atoms, the unit cell, the title) is drawn again on the cleared panel. [F2 (a)]
- **added** `Interactive_STM_Simulator._build_ui: Line / Path branch, local flip_kw` (final): flip_kw = {'tall': 0.0} if mode == 'Line' else {}. [P2's two rows (settled)]
- **changed** `Interactive_STM_Simulator._build_ui: the controls.FlipBox(...) call` (final): Now called as controls.FlipBox((prof_cell, 25), (self._spectrum_and_legend(spec_cell), 10), spacing=10, **flip_kw). [P2's two rows (settled)]
- **changed** `Interactive_STM_Simulator._build_ui: comment above the FlipBox call` (final): Added to 'The spectrum and its legend panel: one sub-sub-section of the 1D sub-section.': in Line the FlipBox never stacks them (tall=0), so the Tip Height profile and the spectrum with its legend stay one row, Line's second row; [P2's two rows (settled)]
- **changed** `Interactive_STM_Simulator._build_ui: Line / Path branch comment` (final): Stale words only: 'side by side or stacked by that room's shape' now reads 'in Line side by side in every room's shape, in Path side by side or stacked by that room's shape'. [P2's two rows (a text it makes false, K9's rule)]
- **changed** `Interactive_STM_Simulator._build_ui: docstring` (final): Stale words only: 'curve panels (Tip Height and Partitioned LDOS, beside or over each other by their room)' now reads '(Tip Height and Partitioned LDOS, side by side in Line, in Path beside or over each other by their room)'. [P2's two rows (a text it makes false, K9's rule)]
- **changed** `Interactive_STM_Simulator._choose_orientation: docstring` (final): 'the shaded energy maps are drawn again' now reads 'the Relief-shaded energy maps are drawn again'. [K9's rule (settled after the last pass)]
- **added** `Interactive_STM_Simulator.__init__: attribute cached_ld_mode` (close-out): New record `self.cached_ld_mode = None` beside cached_ld_*. [item (4) Line / Path sends]
- **added** `Interactive_STM_Simulator.__init__: attributes map_bias, map_ldos_h` (close-out): `self.map_z = None` became `self.map_z, self.map_bias, self.map_ldos_h = None, None, None`. [item (3) the Map Topo]
- **added** `Interactive_STM_Simulator.__init__: attributes map_spec_d_ldos, map_spec_broadening` (close-out): New line `self.map_spec_d_ldos, self.map_spec_broadening = None, None`: the decay model and the broadening that Map's marker spectra were computed with. [item (1) the decomposition's Decay-off rule]
- **changed** `Interactive_STM_Simulator.__init__: comment block on Map's records` (close-out): The map_z sentence now also says the Map Topo is drawn from map_z, with map_bias and map_ldos_h for its title. [items (1), (3); texts]
- **changed** `_map_sample_indices: signature and body` (close-out): The signature is now `_map_sample_indices(self, targets, ends=None)`. [item (2) the energy-marker drag]
- **changed** `_update_all: full refresh, the Map Topo's tricontourf` (close-out): The Map Topo is drawn from `self.map_z`. While map_z is None (before Map's first maps) it is drawn from `self.current_z_map`, as before. [item (3) the Map Topo]
- **changed** `_update_all: full refresh, the Map Topo's title` (close-out): The title now takes `m_bias, m_h = (self.map_bias, self.map_ldos_h)`. [item (3) the Map Topo]
- **removed** `_update_all: needs_topo Map branch, the Map Topo redraw` (close-out): Removed the block that drew the Map Topo right after the LDOS topo was computed. [item (3) the Map Topo]
- **added** `_update_all: needs_ldos Map branch, records map_bias / map_ldos_h` (close-out): After `self.map_z = np.array(self.current_z_map, copy=True)` the line `self.map_bias, self.map_ldos_h = v_b, self.ldos_height` was added. [item (3) the Map Topo]
- **added** `_update_all: needs_ldos Map branch, `if needs_topo:` branch` (close-out): Added after the Map records are set. It holds the removed redraw, with two changes: tricontourf now draws `self.map_z` and the title is `f"Map Topo: bias {self.map_bias:.3f} V, height {self.map_ldos_h} Å"`. [item (3) the Map Topo]
- **added** `_update_all: needs_ldos non-Map branch, inside `if ld_up is not None:`` (close-out): Added `self.cached_ld_mode = self.mode`, written only together with cached_ld_* and cached_eg. [item (4) Line / Path sends]
- **added** `_update_all: needs_spec Map branch, records map_spec_d_ldos / map_spec_broadening` (close-out): Added `self.map_spec_d_ldos, self.map_spec_broadening = decay_model, dict(self.extra_broadening)` where Map's marker spectra are stored, with a comment. [item (1) the decomposition's Decay-off rule]
- **changed** `_on_motion: 'emarker' branch, the clip` (close-out): `np.clip(event.xdata, self.s_emin.val, self.s_emax.val)` became `np.clip(event.xdata, self.map_typed[0], self.map_typed[1])`, which is Map's kept window. [item (2) the energy-marker drag]
- **changed** `_on_rel: 'emarker' branch, the snap` (close-out): `self._map_sample_indices([target_e])` became `self._map_sample_indices([target_e], ends=self.map_typed[:2])`, which snaps among the displayed samples of Map's kept window. [item (2) the energy-marker drag]
- **changed** `_on_rel: 'emarker' branch, the dragged map's call` (close-out): grid_z now stacks `self.map_z` in place of `self.current_z_map`. [item (2) the energy-marker drag]
- **added** `Interactive_STM_Simulator.__init__: attribute map_broadening` (close-out review fix): `self.map_eg, self.map_window, self.map_d_ldos = None, None, None` became `self.map_eg, self.map_window, self.map_d_ldos, self.map_broadening = None, None, None, None`. [After the final pass: the energy-marker drag ('under Map's kept model'); the Decay-off rule's 'Map's records also keep the decay model and the broadening they were computed with']
- **changed** `Interactive_STM_Simulator.__init__: comment block on Map's records` (close-out review fix): Reflowed. 'the decay model (map_d_ldos) and the LDOS-topo heights' became 'the decay model (map_d_ldos), the broadening (map_broadening, a copy of extra_broadening) and the LDOS-topo heights'. [the energy-marker drag; texts]
- **added** `_update_all: needs_ldos Map branch, record map_broadening` (close-out review fix): Added `self.map_broadening = dict(self.extra_broadening)` after `self.map_window, self.map_d_ldos = (e_lo, e_hi), decay_model`. [the energy-marker drag ('under Map's kept model')]
- **changed** `_on_rel: 'emarker' branch, the dragged map's call` (close-out review fix): New flag: `kept_broad = self.map_broadening is not None and self.map_broadening != self.extra_broadening`. [the energy-marker drag ('under Map's kept model')]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py`

- **changed** `StmDecompSession.send_to_decomposition: Single Point / Map branch` (final): Reads the view's own record. In Map: sim.map_spec_ldos with sim.map_eg. [F4 (a): sim_wrapper sends each with its own; in Map, Map's own spectra]
- **changed** `StmDecompSession.send_to_decomposition: Line / Path branch (comment)` (final): New comment above eg = cached_eg: the line LDOS's own energies, kept with it (cached_eg, written only with cached_ld_*). [F4 (a): Line / Path's data with their energies]
- **added** `StmDecompSession.send_to_decomposition: Single Point / Map branch, the check len(spec) < len(sim.marker_coords)` (final review fix): Runs after the 'spec is None' check. When the view's own spectra (Map: map_spec_ldos; [F4 (a): sim_wrapper sends each with its own (fixes a crash the split brought in)]
- **added** `StmDecompSession.send_to_decomposition: comment above the new check` (final review fix): Three lines: one spectrum for every marker, as the simulator's RUN-off redraw asks (_has_last_run); [F4 (a)]
- **added** `StmDecompSession.send_to_decomposition: Single Point / Map branch, Map model refusal` (close-out): After the `spec is None` refusal there is a new branch: `if sim.mode == "Map" and (sim.map_spec_d_ldos != sim._effective_decay_model() or sim.map_spec_broadening != sim.extra_broadening)`. [item (1) the decomposition's Decay-off rule]
- **added** `StmDecompSession.send_to_decomposition: Line / Path branch, view refusal` (close-out): After the `cached_ld_up is None` refusal there is a new branch: `if sim.cached_ld_mode != sim.mode`. [item (4) Line / Path sends]
- **changed** `module docstring, the 'ldos' paragraph` (close-out review fix): Stale words only. 'over the current energy window, and opens/refreshes the decomposition window.' became 'with their own energies, and opens/refreshes the decomposition window; [K9's rule (texts that become false): the Decay-off rule and Line / Path sends add refusals; F4 (a) gives each record its own energies]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/mplexport.py`

- **changed** `_turned (the branch choosing the transform)` (final): The condition `hasattr(c, 'set_offset_transform') and hasattr(c, 'get_offsets')`, which every Collection meets, becomes `parts is None and isinstance(c, PathCollection)`. [F3 (a)]
- **added** `_turned (local import)` (final): `from matplotlib.collections import PathCollection` at the top of the function, like the module's other local matplotlib imports. [F3 (a)]

## `qtgui/mplexport.py`

- **changed** `_turned (the branch choosing the transform)` (final): The same change as in the package copy: set_offset_transform only for a scatter (`parts is None and isinstance(c, PathCollection)`), set_transform for every other artist and for a pre-3.8 contour set's parts. [F3 (a)]
- **added** `_turned (local import)` (final): `from matplotlib.collections import PathCollection` at the top of the function. [F3 (a)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/qtplot.py`

- **changed** `PanelRow._fit_height` (final): Now returns pass_cached(self, ('fit_height', W), lambda: self._fit_height_now(W)). [FINAL_OPEN.md, settled after the last pass: the fill row's height (N1), cached per width and layout pass]
- **changed** `PanelRow._fit_height (docstring)` (final): Adds '(_fit_height_now), worked out once per width and layout pass (pass_cached): the arrangement asks it many times in one pass.' [FINAL_OPEN.md, settled after the last pass: the fill row's height (N1)]
- **added** `PanelRow._fit_height_now` (final): The old _fit_height body moved here unchanged: _parts(W), room = avail minus the free items' minimum widths, and the 60-step bisection on [1, 1e6] over the shaped items' width_for_height. [FINAL_OPEN.md, settled after the last pass: the fill row's height (N1)]

## `LDOS_spread_decomp/mo_ldos_decomp/README.md`

- **changed** `intro paragraph (second paragraph), the first sentence` (final): 'The package reaches the matcher and the simulator through `mo_ldos_projection/_imports.py`, which expects' now reads 'The package reaches the matcher through `mo_ldos_projection/_imports.py` and the simulator through `mo_ldos_projection/_resolve.py` (its resolvers re-exported by `_imports.py`), whi… [FINAL_OPEN.md, settled after the last pass, K9's rule: the mo_ldos_decomp README's intro (the resolvers are _resolve.py's, re-exported by _imports.py)]
- **changed** `intro paragraph (second paragraph), the qtgui sentence` (final): 'which `_imports.py` finds as it finds the simulator (`resolve_qt()`, no folder walking)' now reads 'which `_resolve.py` finds as it finds the simulator (`resolve_qt()`, no folder walking)'. [FINAL_OPEN.md, settled after the last pass, K9's rule: the mo_ldos_decomp README's intro (the resolvers are _resolve.py's, re-exported by _imports.py)]
- **changed** `module table, `_imports.py` row` (final): 'import resolution of the matcher and simulator packages and the simulator's `qtgui` copy (`resolve_qt()`)' now reads 'import resolution of the matcher package; [FINAL_OPEN.md, settled after the last pass, K9's rule: the mo_ldos_decomp README's _imports.py row (the resolvers are _resolve.py's, re-exported by _imports.py)]

## `convergence/README.md`

- **changed** `forces.py paragraph, the save sentence` (final): 'The toolbar's Save button writes the convergence panel as PDF / EPS / SVG / PNG / TIFF.' now reads 'The toolbar's Save button writes the convergence panel as PDF / EPS / SVG, or the window at 720 dpi as PNG / TIFF.', matching forces.py's docstring sentence. [FINAL_OPEN.md, settled after the last pass, K9's rule: convergence/README.md's save sentence (as forces.py's)]

## `LDOS_spread_decomp/stm_from_dft_opt/README.md`

- **changed** `Window paragraph, the curve panels parenthesis (line 33)` (close-out): '(Tip Height and Partitioned LDOS with its legend panel at its right, side by side or stacked by their room)' became '(Tip Height and Partitioned LDOS with its legend panel at its right, side by side in Line, side by side or stacked by their room in Path)'. [FINAL_OPEN.md, 'After the final pass (wf_3409e76f-c47)', K9's rule: the stm README's 'side by side or stacked by their room' (Line now always side by side)]

## `LDOS_spread_decomp/README.md`

- **changed** `Folder-layout sentence (lines 11-13)` (close-out): '`mo_ldos_projection/_imports.py` finds `MOPmatcher_mem/` and `stm_from_dft_opt/` two levels above itself (next to the `mo_ldos_decomp/` folder).' became '`mo_ldos_projection/_imports.py` finds `MOPmatcher_mem/` and `mo_ldos_projection/_resolve.py` (its resolvers re-exported by `_imports.py`) finds … [FINAL_OPEN.md, 'After the final pass (wf_3409e76f-c47)', K9's rule: the wrapper README lines 11-12 (`_resolve.py` finds `stm_from_dft_opt/`)]

