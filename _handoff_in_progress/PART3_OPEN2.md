# Part 3: points found during the implementation (for the owner)

Part 3's workflow (wf_237d8a97-ab0) is done: every stage implemented, each file group reviewed twice and fixed, one
cross-module review. These four points need the owner. Each has a recommendation (R). Below them: what I settled
under rules already given, and what is only for information.

**N1. P2 (the Line view's height sharing) cannot be done without a qtgui change.** Two things in the package qtgui
keep it from working:
- `controls.Arrangement` picks Line's layout by score among six candidates (`row_top`, `a_left`, `a_top`, `ab_left`,
  `abf_row`, `abf_col`). Only `row_top` is your two rows, and nothing lets a window keep it.
- Inside `row_top`, the top row's natural height (`PanelRow._natural_height`, the Line Topo at 2:3 and the heatmaps)
  is below the height at which its panels fill the row's width (`_fit_height`), so the top row comes out short and
  the second row takes the rest.
- (a) In qtgui (both package copies): `Arrangement` takes a list of allowed layouts, Line passes only `row_top`; and a
  row of fill panels reports the height where its panels fill its width as its shape. Then the top row grows with
  the window keeping its ratios, up to what the width allows, and the 1D plots take the rest.
- (b) Leave Line's layout to the later views-layout handoff, which states the same rule for Line (and adds the topo
  sub-section in every view).

R: (a), since you asked for it now; the later handoff can build on the same two pieces.

**N2. The Qt-drawn SVG at 720 dpi makes its text about 7.5 times too large.** Qt's SVG writer (qtsvg 6.8,
`qfontToSvg`) writes a point-size font as `pointSizeF * resolution / 72` and the file's size as
`size * 25.4 / resolution` mm. This copy's text uses point sizes (`qfont` sets `setPointSizeF`; the root's app font
too). So `gen.setResolution(720)` keeps the panels but writes every text 720 / 96 = 7.5 times larger relative to
them, against your rule that fonts and panels keep their sizes relative to each other. At the current 100 dpi the
SVG already has the window images' physical size (W / DPI inches), with text in proportion.
- (a) Keep 100 dpi for the Qt-drawn SVG (and its PDF branch): same physical size as the 720 dpi images, everything
  in proportion.
- (b) 720 dpi as written: the same physical size, the text 7.5 times too large.
- (c) 720 dpi plus pixel-size fonts while the file is drawn (a workaround nothing names).

R: (a). Left at 100 dpi in all three copies until answered.

**N3. After a colour-set change, the vector saves show the old maps.** `TriContourSet.set_cmap` (and `set_norm`)
recolour the screen, but the record mplexport rebuilds the filled contours from (`TriContourSet._args`) keeps the old
map. After switching to "default 2", a PDF / EPS / SVG shows the global topo, the LDOS topo, the Line Topo and every
energy map in default 1's maps. The heatmaps, the stripe, the topo ramp and the colour bars follow correctly.
- (a) `set_cmap` and `set_norm` also update that record (qtplot, both package copies).
- (b) mplexport reads the current map and norm instead.

R: (a).

**N4. README rows for the new modules.** K9 adds a row for `ldos_decomp.py` to the stm README's module table. Two
more modules are new: `cmap_shading.py` (stm README) and `mo_ldos_projection/_resolve.py` (the mo_ldos_decomp README,
which now reads as if `_imports.py` defined the resolvers). R: a row for each, in the same terms as K9's.

## Settled under rules already given (done in the follow-up)

- P3 in Path: the square-heatmap cap applies in Line only, as implemented; P3's own words are "so the cap is exactly
  today's", and Path's topography width has never been capped.
- The vector-save hook is `Axes.export_extent` (a callable giving the content height and the y limits), as built in
  all three qtgui copies and the simulator. My brief named it two ways; this is the one every stage used.
- K4 (a): the package copy's savefig no longer passed `facecolor` / `transparent` on to mplexport (the root does);
  the two lines are restored so a transparent save stays transparent.
- With RUN off, a Line / Path marker re-sort left the level-0 curves, the spectrum title and the legend in the old
  order (a click then picked another marker). Under 2a ("artist i always shows marker i") and K7 (d) (redraw from the
  last run's data), the re-sort redraws them in place from the last run's data.
- K9's rule (texts that become false: change only the stale words) for the texts found false: the Save button's
  tooltip, the FigureManager and Figure docstrings, "matplotlib, when installed" in qtplot's docstring, the class
  docstring's full list of the left column (no "cmap set"), the stm README's `cmap_map` row, and PanelCell.bind's
  comment after P5 (i). The root copy had taken sm4_browser's wording for three of them; both copies get the same
  stale-words edit.
- The legend title in the vector saves is written bold, as on screen.
- The P8 "comment in vacupot_plotter.py": it is true as it stands; nothing to change.

## For information

- **Left unused by the port** (nothing calls them; removal is not named, so they stay): `_have_matplotlib()`
  (qtplot), `available()` (mplexport), `Figure.at_design_size` and `_size_override` with its reads, and in
  simulator.py the `colorsys` import and `widgets` in `from .qtgui import controls, widgets`. Say if you want them
  removed.
- **The checks:** `checks/render_windows.py` (`how='screenshot'`) and `checks/check_misc_windows.py` (`dpi=72` PNGs)
  are unchanged, as ruled; under the new saves they now write 720 dpi window images (no longer mplexport files), so
  they run much heavier.
- **Incomplete but true texts, left:** the Arrangement docstring's trigger list (no text-zoom trigger), the run-directly
  comment in plot_dos_cursors_levels.py (no doscar_parser), controls.py's docstrings (no settings paragraph).
