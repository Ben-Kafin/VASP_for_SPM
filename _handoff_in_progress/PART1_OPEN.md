# Part 1: open items for the owner

Part 1 is implemented (sections 3 to 7) and went through two independent reviews per file group
against the plan, then a fixer. These are the items the agents left undone because the plan does not
decide them, plus two comment-only additions that need your approval. Each has a recommendation (**R:**).
Answer per item, or "accept all R except ...". The cross-module review is still running; anything it
finds will be added here.

Still to do on my side (no decision needed): your `__init__.py` docstring ruling, and the byte copy of
`stm_from_dft/qtgui/` over `matcher/qtgui/` (until then `import matcher` fails: the matcher copy still
has the root's absolute `from qtgui import ...`).

## Layout and drawing (behaviour)

**P1. Who converts qtgui's size arguments.** qtgui's own arguments (`fig.cell(min_px=...)`,
`inner_w=`, `PanelRow(spacing=)`, `MapBlock`/`Arrangement`/`FlipBox` `spacing=`, `picker=`) take
pixels at the old 100 DPI and are converted once inside qtgui (`c / (1000 / 72) * em_px()`). The
windows pass the old literals to them (`inner_w=58`, `picker=5`) and convert only their own raw Qt
calls (`setSpacing(round(4 / (1000 / 72) * plt.em_px()))`). A first pass converted at both ends
(c·(em/13.89)², twice); that is fixed.
**R:** accept this convention.

**P2. Line view: the heatmaps stretch in x only (5.4).** `fig.align_row(*line_decomp_axes)` includes
the Line Topo (box aspect 1.5), so every box in the row is held to the Line Topo's height. In a slot
taller than the row's shaped height the heatmaps fill x but not y. In Path they fill x and y.
Options: (a) keep: the distance axis stays level across Line Topo, stripe and heatmaps; (b) in Line,
align only the stripe, heatmaps and colour bars: they fill the slot's height, the Line Topo keeps its
2:3 box, and its 0..L axis is no longer level with theirs.
**R:** your call ((b) is the literal reading of "stretching in x or y").

**P3. The topography cap in Line changed (5.3 says unchanged).** `_topo_cap` takes one heatmap's box
from the row's first panel, which in Line is the Line Topo. Its aspect is now 1.5 instead of 1, so the
topographies' width cap in Line drops from 2h to about 1.33h. (The heatmaps themselves no longer carry
an aspect, so it cannot come from them.)
**R:** count a heatmap as square (ratio 1.0) in that box, so the cap is exactly today's.

**P4. A new heatmap mesh when the coordinates change (6.2).** The plan says "a new mesh only when the
grid shape changes (a new energy range)". qtgui's QuadMesh has no coordinate setter, so after a line
or path length change `set_array` would keep the old distance axis. The code makes a new mesh (and
colour bar on the same cax) when the energies or the distances change.
**R:** keep (new mesh when the grid's coordinates change).

**P5. Three edge cases of the "arranged only on three triggers" rule (5.2).**
- (i) Some calls start a layout pass without arranging (`Arrangement.best_score`, `RideBand.add/take`,
  `Figure.set_portrait`, `_set_control_zoom`); the cells then re-measure their label room and can move
  without an arrangement (e.g. in `_choose_orientation` when the orientation stays).
  **R:** keep the cells' label room under an arrangement epoch, bumped only by an arrangement.
- (ii) Tick text is measured only when painted, so a narrow cell (colour bar, stripe) keeps the width
  from its first size question and its tick labels can run past the cell's right edge, on a cell's first
  pass and after a text-zoom change, until the next resize.
  **R:** in the narrow branch of `Axes._cell_px`, shrink the box so it stays inside its cell (the box
  moves inside its cell, which 5.2 allows; the cell does not move).
- (iii) `Arrangement.resizeEvent` runs before `Figure._relayout` applies the new text zoom, so a window
  keeps the arrangement chosen at the old zoom until the next resize.
  **R:** also arrange when the text zoom, control zoom or portrait setting changed since the last
  arrangement (all three follow a resize, trigger 3).

**P6. One em_px for the whole process (4.1).** `em_px()` has no window argument, so with two windows on
screens of different logical DPI the last one measured wins. Under Qt 6's high-DPI scaling the logical
DPI is usually the same on every screen.
Options: (a) accept one value per process; (b) one per figure (every call site passes its figure);
(c) one value, and a window lays itself out again when a screen change changed it.
**R:** (a).

**P7. Before RUN the heatmap row has no shaped size,** so the 5.4 slot does not apply until the row
has panels. **R:** keep.

## Text that is no longer true (not named by the plan)

**P8.** Comments and docstrings that still describe the old state, outside what section 3.2/5.5/7 name:
- `qtgui/__init__.py` line 2: "the one shared Qt layer of every interactive window of VASP_for_SPM"
  (your ruling covers only the walk paragraph);
- `Figure.fit_window` / `_schedule_fit` docstrings and a `_fit_controls` comment, `OrientedPanes.fit_delta`
  and the `Arrangement` class docstring: they describe the trim that 5.1 removed;
- `screen_scale()`: nothing calls it now; its docstring says windows open this much larger;
- module docstrings that say "qtgui at the repository root": simulator.py, plot_dos_cursors_levels.py,
  four matcher modules (and a comment in vacupot_plotter.py), decomp_window.py, topo_decomp_window.py,
  global_decomposition.py, run_wfn_plane_slice.py;
- the matcher and decomp READMEs' "every text scales with the window's size" (no 10 pt floor), and the
  decomp README's `_imports.py` row (no `resolve_qt()`).
**R:** update each in the same terms, changing only the stale words (for `__init__.py` line 2:
"this package's own copy of the Qt layer of VASP_for_SPM's interactive windows (PySide6 + pyqtgraph)").

## Sizes the agents left unconverted (they are not lengths on screen)

**P9.** Left as they are: 1-pixel and sub-pixel guards and change thresholds (`max(w, 1.0)`, pen
floors, bisection bounds, `+ 0.5` tolerances, `> 2` refits, the 1-px floors in `_topo_cap`); the
legend sample's default `size=6` (drawn only when symbolSize is set); the toolbar icons' 22-px pixmap
coordinates (Qt scales the pixmap to the converted icon size); the unused `COLOR_FRAME_CSS`;
`pickradius=` on curves (qtplot never reads it; `picker=True` uses qtplot's converted default).
**R:** leave them.

**P10.** File writers: Qt's fallback writers (`savefig` 'screen' scale, `export_vector`'s resolution and
page size, the hairline drawn while exporting, `rcParams['figure.dpi']`) and mplexport's
`Figure(..., dpi=q.DPI)` still use DPI. mplexport's figsize now follows pt2px as 4.4 says.
**R:** leave them (written files, not screen sizes).

## A bug the plan's import change exposes

**P11.** `plot_dos_cursors_levels.py` line 29 is `from doscar_parser import SpinAwareDosParser, SpinMode`
(absolute, pre-existing). Run directly it works; imported as part of the package (3.4's new
`from .qtgui import ...` branch) it raises ModuleNotFoundError. Nothing in the repository imports it.
**R:** put it in the two branches: `from .doscar_parser import ...` in the package branch,
`from stm_from_dft.doscar_parser import ...` when run directly.

## Additions the agents made that the plan does not name (comments only)

**P12.** (i) controls.py, the `Checks` class docstring: numeric `indents` "pixels" now reads "pixels (at the
old fixed 100 DPI, converted to the screen with qtplot.em_px)". (ii) mo_ldos_decomp (sim_wrapper,
decomp_window, topo_decomp_window, global_decomposition, run_stm_decomp, run_wfn_plane_slice): the walk's
comment "# The shared Qt layer (qtgui at the repository root): found by walking up from this file." was
replaced by "# The shared Qt layer: the simulator's qtgui copy (_imports.resolve_qt)."
**R:** keep both.

## For information

- widgets.py's `__main__` demo can no longer run as a plain file (its imports are relative, as 3.2 says);
  it runs as `python -m stm_from_dft.qtgui.widgets` from `stm_from_dft_opt/`.
