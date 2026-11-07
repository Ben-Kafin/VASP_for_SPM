# Last pass: the per-file change list (rule 5)

From the implementing agents' and fixers' reports (base 2083aa3): the owner's N1-N4 and FINAL_OPEN.md's settled list.
Fixer entries are marked (review fix). Additional changes: none reported. The matcher's qtgui copy is the package copy.

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py`

- **added** `Interactive_STM_Simulator.__init__: attribute cached_spec_eg (+ inline comment)`: New record cached_spec_eg = None beside cached_spec_window: the marker spectra's energies, kept with them (sim_wrapper reads them). [FINAL_OPEN.md settled: O2, Map's caches kept apart (the marker spectra keep their energies with them)]
- **changed** `Interactive_STM_Simulator._update_all: marker-spectra compute branch (needs_spec, Single Point / Map)`: Also records self.cached_spec_eg = self.energies[spec_idx], the energies of the samples the marker spectra were just computed at, next to cached_spec_window. [FINAL_OPEN.md settled: O2, Map's caches kept apart (the marker spectra keep their energies with them)]
- **removed** `Interactive_STM_Simulator._update_all: branch 'if self.mode == 'Map' and self.map_eg is not None: self.cached_eg = self.map_eg' (+ its comment)`: Map no longer writes the shared cached_eg. [FINAL_OPEN.md settled: O2, Map stops writing cached_eg]
- **changed** `Interactive_STM_Simulator._draw_spectrum`: New local eg = self.map_eg in Map, else self.cached_eg (+ comment). [FINAL_OPEN.md settled: O2, Map draws its spectrum against its own energies]
- **changed** `Interactive_STM_Simulator._draw_spectrum (docstring)`: Adds 'The curves are drawn over the displayed energies: in Map Map's own (map_eg), in the other views cached_eg.' [FINAL_OPEN.md settled: O2, Map draws its spectrum against its own energies]
- **changed** `Interactive_STM_Simulator._build_ui: the closing _update_all call (+ comment)`: Now _update_all(full_refresh=True, last_run=mode == 'Map'). [FINAL_OPEN.md settled: O2, returning to Map (with RUN off too)]
- **changed** `Interactive_STM_Simulator._build_ui (docstring)`: 'then _update_all(full_refresh=True).' now continues ', in Map with last_run (with RUN off too, Map's kept maps, energy markers and spectrum are drawn from what is kept).' [FINAL_OPEN.md settled: O2, returning to Map]
- **changed** `Interactive_STM_Simulator._update_all (docstring)`: last_run's callers: '(a decomp step: _decomp_click)' becomes '(a decomp step: _decomp_click; [FINAL_OPEN.md settled: O2, returning to Map]
- **changed** `Interactive_STM_Simulator._update_all: comment above the RUN-off early return`: 'except for a decomp step (last_run)' becomes 'except for a decomp step or a view change into Map (last_run)'. [FINAL_OPEN.md settled: O2, returning to Map]
- **changed** `Interactive_STM_Simulator._decomp_click`: New first branch: 'if not self.is_running and not self._has_last_run(): return'. [FINAL_OPEN.md settled: K7 (d), no step without its redraw]
- **changed** `Interactive_STM_Simulator._decomp_click (docstring)`: Adds that with RUN off and without the last run's data (_has_last_run: dropped by a change made while stopped, or no run in the view yet) a click does nothing: the state is kept, so no step is taken without its redraw. [FINAL_OPEN.md settled: K7 (d)]
- **changed** `Interactive_STM_Simulator._legend_rezoom`: In Map, when the legend's fixed width changed: after the kept self._map_block.rearrange(), it also calls self._rearrange(), so the window is arranged again. [FINAL_OPEN.md settled: K2, the Map legend's width]
- **changed** `Interactive_STM_Simulator._legend_rezoom (docstring)`: Now says: the map block placed again (MapBlock.rearrange: its cells measured anew, the new width with them; [FINAL_OPEN.md settled: K2]
- **changed** `Interactive_STM_Simulator._rearrange (docstring)`: Stale words in its list of callers: adds 'and in Map after a text zoom changed the legend panel's fixed width (_legend_rezoom: a resize-driven change)'. [FINAL_OPEN.md settled: K2]
- **changed** `Interactive_STM_Simulator._make_controls: comment on the Contour / Shading box`: The quoted radio name 'Shading by' becomes 'shading by' (stale word only). [FINAL_OPEN.md settled: the 'shading by' radio]
- **changed** `Interactive_STM_Simulator._build_ui: the controls.Arrangement(...) call`: Passes layouts=['row_top'] when mode == 'Line', so Line allows only its two rows. [N1 (a), the simulator side: Line passes only row_top to controls.Arrangement]
- **added** `Interactive_STM_Simulator._build_ui: comment above the Arrangement call`: Three comment lines: Line allows only its two rows (layouts: row_top). [N1 (a), the simulator side]
- **changed** `Interactive_STM_Simulator._make_controls: the comment above self._cluster_label`: 'at the atom levels' now reads 'at level 3 (the orbitals)'. [FINAL_OPEN.md, settled, K9's rule: the "atom levels" comments (now level 3, the orbitals)]
- **changed** `Interactive_STM_Simulator._fill_heat_row: the comment above the heatmap's short title`: 'at the atom levels' now reads 'at level 3 (the orbitals)'. [FINAL_OPEN.md, settled, K9's rule: the "atom levels" comments (now level 3, the orbitals)]
- **changed** `Interactive_STM_Simulator (class docstring): Single Point's panel list`: 'Single Point: Global Topo | Topo h | Partitioned LDOS' now continues with ', its legend panel at its right'. [FINAL_OPEN.md, settled, K9's rule: the class docstring's panel lists (the legend panel beside the spectrum)]
- **changed** `Interactive_STM_Simulator (class docstring): Line / Path's panel list`: '... over the tip-height profile and the spectrum' now continues with ', its legend panel at its right'. [FINAL_OPEN.md, settled, K9's rule: the class docstring's panel lists (the legend panel beside the spectrum)]
- **changed** `Interactive_STM_Simulator (class docstring): Map's panel list`: 'Map: Global Topo | Topo h | LDOS h | Map Topo | Partitioned LDOS over the rows of energy maps' now reads 'Map: Global Topo | Topo h | LDOS h | Map Topo over the rows of energy maps, the Partitioned LDOS one cell of the map block, its legend panel at its right'. [FINAL_OPEN.md, settled, K9's rule: the class docstring's panel lists (the spectrum as a cell of the map block, the legend panel beside the spectrum)]
- **changed** `Interactive_STM_Simulator (class docstring): Map's boxes`: "... LDOS γ) and Map's Contour / Shading box." now reads "... [FINAL_OPEN.md, settled, K9's rule: the class docstring's Map boxes (the cmap norm and Contour / Shading boxes)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/cmap_shading.py`

- **changed** `SHADE_BY (module constant)`: 'Shading by' becomes 'shading by' (lower case, like 'cmap norm' and 'cmap set'). [FINAL_OPEN.md settled: the 'shading by' radio (cmap_shading.SHADE_BY)]

## `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py`

- **changed** `StmDecompSession.send_to_decomposition: Single Point / Map branch`: eg is now np.asarray(sim.cached_spec_eg, dtype=np.float64), the marker spectra's own energies (+ comment), read after the existing cached_spec_ldos None check. [FINAL_OPEN.md settled: O2 (sim_wrapper's Single Point / Map branch reads the marker spectra's energies)]
- **changed** `StmDecompSession.send_to_decomposition: the cached_eg None check and eg = cached_eg`: Moved from before the branch into the Line / Path branch (same message, before the cached_ld_up check). [FINAL_OPEN.md settled: O2 (the Single Point / Map branch reads that record, not cached_eg)]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/ldos_decomp.py`


## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/controls.py`

- **changed** `Arrangement.__init__ (signature)`: New keyword parameter layouts=None, after fill_slot. [N1 (a): Arrangement takes a list of allowed layouts (default all)]
- **added** `Arrangement._layouts (instance attribute)`: None, or tuple(layouts): the candidates allowed. [N1 (a): Arrangement takes a list of allowed layouts]
- **changed** `Arrangement._candidates_at: want (local lambda)`: Now also requires self._layouts is None or name in self._layouts, so two-group candidates that are not allowed are never sized. [N1 (a): Arrangement takes a list of allowed layouts]
- **changed** `Arrangement._candidates_at: put (inner function)`: New branch at the top: returns without adding the candidate when want(name) is false. [N1 (a): Arrangement takes a list of allowed layouts]
- **changed** `Arrangement._candidates_at (docstring)`: Adds 'the allowed candidates only (layouts)'. [N1 (a): Arrangement takes a list of allowed layouts]
- **changed** `Arrangement (class docstring)`: New sentence describing `layouts`: the allowed candidates by name, every one when None (the default), and the simulator's Line view allows only `row_top`. [N1 (a): Arrangement takes a list of allowed layouts]

## `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/qtplot.py`

- **changed** `PanelRow._natural_height`: New branch: when any visible shaped item fills (PanelRow._fills, i.e. [N1 (a): a row of fill panels reports, as its shape, the height at which its panels fill the row's width]
- **changed** `PanelRow._natural_height (docstring)`: Adds the fill-row case: the height at which its panels fill the row's width (_fit_height). [N1 (a): a row of fill panels reports its fill height as its shape]
- **changed** `PanelRow (class docstring)`: In the natural-height sentence, adds '(a row of fill panels: the height at which its shaped items' widths, with the free items' minimum widths, fill the row's width)'. [N1 (a): a row of fill panels reports its fill height as its shape]
- **added** `_pixel_fonts(fig) (contextlib.contextmanager, placed after _render_scene)`: Reads every font first, then while the block runs gives each one in pixels at its on-screen size: export_root() and every widget in it (QFontInfo of w.font(), which Qt keeps at the widget's screen dpi), plus, in each QGraphicsView's scene, the QGraphicsTextItem / QGraphicsSimpleTextItem fonts (QFont… [N2 (c)]
- **changed** `export_vector, SVG branch`: gen.setSize is now (round(W*k), round(H*k)) with k = WINDOW_IMAGE_DPI / DPI (new local k), and gen.setResolution(int(WINDOW_IMAGE_DPI)) replaces int(DPI). [N2 (c)]
- **changed** `export_vector docstring`: 'SVG (QSvgGenerator)' becomes 'SVG (QSvgGenerator at WINDOW_IMAGE_DPI, as the window images, with every font drawn as its pixel size on screen, _pixel_fonts, so the text keeps its size beside the panels)'. [N2 (c)]
- **changed** `TriContourSet.set_norm`: Also puts the new norm into the export record: self._args = self._args[:5] + (norm,) + self._args[6:] (index 5 is the norm), before _recolor, with a short comment. [N3 (a)]
- **changed** `TriContourSet.set_cmap`: Also puts the new base map into the export record: self._args = self._args[:4] + (self._base,) + self._args[5:] (index 4 is the base map), before _recolor, with a short comment. [N3 (a)]
- **changed** `TriContourSet class docstring`: Stale words only: 'The arguments are kept (_args) for the publication export.' becomes 'The arguments, with the colour map and norm given since (set_cmap, set_norm), are kept (_args) for the publication export.' [N3 (a)]
- **changed** `to_cmap docstring`: '(its colours read through matplotlib when installed)' becomes '(its colours read through matplotlib)'. [Settled texts: to_cmap's 'when installed']
- **changed** `to_cmap, the ValueError for an unknown name (branch after _builtin_cmap fails)`: The message 'unknown colour map {c!r} (install matplotlib to use every matplotlib name)' becomes 'unknown colour map {c!r}'. [Settled texts: to_cmap's install hint]
- **changed** `_ViewBox.wheelEvent docstring`: '(Pan)' becomes '(Pan or Zoom)'. _apply_toolbar_mode turns the mouse on for both modes, so the wheel zooms in either one. [Settled texts: _ViewBox.wheelEvent's '(Pan)' to '(Pan or Zoom)']

## `qtgui/qtplot.py`

- **changed** `to_cmap (docstring)`: 'its colours read through matplotlib when installed' becomes 'its colours read through matplotlib' (matplotlib is required in this copy, line 115). [FINAL_OPEN settled, K9's rule: to_cmap's 'when installed']
- **changed** `to_cmap (ValueError text for an unknown name)`: The install hint is gone. The message is now f"unknown colour map {c!r}". [FINAL_OPEN settled, K9's rule: to_cmap's install hint]
- **changed** `_ViewBox.wheelEvent (docstring)`: '(Pan)' becomes '(Pan or Zoom)'. [FINAL_OPEN settled, K9's rule: _ViewBox.wheelEvent]
- **added** `_pixel_fonts(fig) (contextlib.contextmanager)`: New module-level function, the same text as the package copy's. [N2 (c)]
- **changed** `export_vector (SVG branch)`: New local k = WINDOW_IMAGE_DPI / DPI. gen.setSize is now (W*k, H*k), the viewBox stays (0, 0, W, H), and gen.setResolution(int(WINDOW_IMAGE_DPI)) replaces int(DPI). [N2 (c)]
- **changed** `export_vector (docstring)`: 'SVG (QSvgGenerator)' becomes 'SVG (QSvgGenerator at WINDOW_IMAGE_DPI, as the window images, with every font drawn as its pixel size on screen, _pixel_fonts, so the text keeps its size beside the panels)', the same words as the package copy. [N2 (c)]

## `LDOS_spread_decomp/stm_from_dft_opt/README.md`

- **added** `control table, `cmap set` radio row (after the `mode` row)`: New row: the colour-map set of every panel. [FINAL_OPEN settled, K9's rule: the stm README's control table, a row for the 'cmap set' choice]
- **changed** `control table, `cmap norm` row, the colour-map sentence`: Added after '`CET-L6` in `default 2`': '; [FINAL_OPEN settled, K9's rule: the shading maps in the 'cmap norm' row]
- **changed** `control table, `contour slice` row`: 'the sm4_Browser Map window's:' now reads 'the sm4_Browser Map window's, drawn while `Contour` is on (the Contour / Shading box):'. [FINAL_OPEN settled, K9's rule: the contour slice drawn while Contour is on]
- **added** `control table, `Contour`, `Shading`, `shading by` row (after the contour-slice row)`: New row for the Contour / Shading box. Where: bottom strip, Map only, after `cmap norm`, untitled. [FINAL_OPEN settled, K9's rule: a row for the Contour / Shading box]
- **added** `module table, `cmap_shading.py` row (after the `ldos_decomp.py` row)`: New row in the same terms as the `ldos_decomp.py` row: one self-contained module (display machinery, no physics; [N4: a row for cmap_shading.py in the stm README's module table]
- **changed** `module table, `cmap_shading.py` row, the imports clause` (review fix): 'it imports only NumPy and CuPy' now reads 'it imports only the standard library, NumPy and CuPy', as the module's own docstring says (the module has `import math`). [N4: a row for cmap_shading.py in the stm README's module table (review fix)]

## `LDOS_spread_decomp/mo_ldos_decomp/README.md`

- **changed** `module table, `topo_decomp_window.py` row`: 'the figure background is transparent in the exports' now reads 'the figure background is transparent in the PDF, EPS and editable SVG exports'. [FINAL_OPEN settled, K9's rule: the mo_ldos_decomp README's 'transparent in the exports' (the PDF / EPS / editable SVG only)]
- **added** `module table, `_resolve.py` row (after the `_imports.py` row)`: New row: the import resolution of `stm_from_dft` without the matcher (machinery, no physics). [N4: a row for mo_ldos_projection/_resolve.py in the mo_ldos_decomp README]

## `LDOS_spread_decomp/mo_ldos_decomp/run_stm_decomp.py`

- **changed** `module-level `erange` comment`: '# initial energy slider range' now reads '# initial energy box range' (as run_DFT_STM.py). [FINAL_OPEN settled, K9's rule: run_stm_decomp.py's two 'slider' comments (as O16)]
- **changed** ``sim_kwargs` `ldos_bias_sign` comment`: "# 'pos' or 'neg' — which slider sets the bias" now reads "# 'pos' or 'neg' — which box sets the bias". [FINAL_OPEN settled, K9's rule: run_stm_decomp.py's two 'slider' comments (as O16)]

## `convergence/forces.py`

- **changed** `module docstring, the save sentence`: 'writes the 2D panel as PDF / EPS / SVG / PNG / TIFF through matplotlib.' now reads 'writes the 2D panel as PDF / EPS / SVG through matplotlib, or the window at 720 dpi as PNG / TIFF.' [FINAL_OPEN settled, K9's rule: forces.py's save sentence (stale words only)]
- **changed** `forces.<locals>.view_line`: The window line now also gives the roll ('elevation {:.5g}°, azimuth {:.5g}°, roll {:.5g}°, x ...' with ax.roll after ax.azim). [FINAL_OPEN settled: forces.py's 3D view line also gives the roll (part of 'its rotation', K6 (b))]

