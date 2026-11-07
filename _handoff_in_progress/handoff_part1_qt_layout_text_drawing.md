# Handoff, part 1: Qt copies, sizes, text, arrangement and drawing in LDOS_spread_decomp

Repository: `VASP_for_SPM`. This part covers the three packages in `LDOS_spread_decomp/`:

- `stm_from_dft_opt/` (the STM simulator, `stm_from_dft/simulator.py`)
- `MOPmatcher_mem/` (the matcher and its plotters, package `matcher/`)
- `mo_ldos_decomp/` (the decomposition package `mo_ldos_projection/` and its run scripts, which use both of the others)

Every window in them is drawn with `qtgui`, the repository's matplotlib-like layer on PySide6 and pyqtgraph. Today each package finds the one `qtgui/` at the repository root by walking up folders.

Part 2 (the energy windows, the integration rule and the GPU work in the simulator) is a separate plan. Do part 1 first. Part 1 changes no computation.

## What is wrong today

- The simulator rebuilds whole panel clusters on every refresh and on every mouse movement during a drag (`_on_motion` ends with `_update_all`). Dragging lags, and the Line/Path heatmap drag does nothing because `ax_ldos` is set to `None` in `_clear_cluster` and never set again.
- `Figure.fit_window` trims every window to its panels on open, after every resize and after every rebuild. A maximized window does not stay full screen, and in the Map view the trim feedback gave one row of topographies over two rows of maps, a fifth of the screen tall.
- Text shrinks below readable sizes: the text zoom can go to `ZOOM_MIN = 0.5`, `_fit_controls` shrinks controls to `CONTROL_FIT_MIN = 0.75` of that, and some specs are 6 to 7.5 pt.
- Sizes are fixed pixel numbers or points converted at a fixed `DPI = 100`, so boxes and text do not follow the screen together.

## 0. First: remove the Claude-written checks and tests

Do this before any other step, so that no check or test runs on the code while it changes. Run no check or test at any point in this job.

Every file in `checks/` was added by Claude Code commits, and every `test_*.py` in `mo_ldos_projection/tests/` was written by Claude. The owner's own scripts are not touched. Delete:

- `checks/check_simulator.py`
- `checks/sim_synthetic.py`
- `checks/render_sim_sizing.py`
- `checks/check_decomp.py`
- `checks/check_matcher_plots.py`
- `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/tests/` (the whole folder: its `test_*.py` files, and the `conftest.py` and `__init__.py` that exist only for them)

`checks/render_windows.py` and `checks/check_misc_windows.py` cover windows of other packages and stay unchanged. If anything else imports a deleted file, report it and change nothing.

## 1. Before you start: the owner's newer local files

Everything in this zip under `disk/` (in repository layout) is the owner's current version of a file that is new or changed relative to GitHub, and every one of those files goes into the repository at its path. Line numbers in this plan refer to them (and to the current root `qtgui/`). They are approximate: find the code by the names given. Do 1.1 before 1.2, because 1.1 reads the repository's current `simulator.py`, which 1.2 replaces.

### 1.1 Archive the repository's current decay code in `wrong_decays/`

The repository's `stm_from_dft/simulator.py` still uses the old decay model: the WKB average over a linearly tilted barrier (`gpu_chen_tunneling_factor(V, E, phi)`, the '0-clamp' toggle, `clamp_occupied=self.t2_clamp`). The repository has none of Chen's model from the book (Eq. 2.40, `_CHEN_SWEEP_SRC`, `kappa0`) and no Feenstra code. The owner's `simulator.py` in `disk/` replaces the old model with Chen's. The old model is kept, unused, in `stm_from_dft/wrong_decays/`, a folder the repository does not have yet.

The owner's own `wrong_decays/tilted_barrier_decay.py` (in `disk/`) was taken from a later local version of the old model, which differs from the repository's code:

- float64 arrays instead of float32
- the exact V → 0 limit instead of the `1e-6` floor on V
- `kappa = 0.512 * sqrt(phi)` without the `max(0.1, phi)` floor
- the Feenstra current and its toggles

Both versions are kept: the owner's file goes in as it is (1.2), and the repository's own version of the old model is archived next to it as `wrong_decays/tilted_barrier_decay_github.py`:

1. Create `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/wrong_decays/`.
2. Write `wrong_decays/tilted_barrier_decay_github.py` from the repository's current `simulator.py`, in the same format as the owner's `tilted_barrier_decay.py` in `disk/`:
   - **The header block:** the owner's, with every statement checked against the repository's code and corrected where it differs (the float32 arrays, the `1e-6` V floor, the `max(0.1, phi)` floor, no Feenstra current), and one added line saying it is the GitHub version of the model that `tilted_barrier_decay.py` holds in its later local form.
   - **The excerpts:** verbatim from the repository's `simulator.py`, every line commented with a leading `# ` (removing it restores the code; methods keep their class indentation), each under the same `# ---- ...` heading as in the owner's file:
     - module level (`gpu_chen_tunneling_factor`)
     - `Unified_STM_Simulator._converge_tip_height`
     - `_effective_decay_model` and `_decay_cache_suffix`
     - `_calculate_ldos_at_points_gpu`
     - `Interactive_STM_Simulator.__init__` (the toggle's state)
     - `_make_controls` (the toggles)
     - `_on_ui_change` (to the end of the decay block)
   - The repository has no `_feenstra_current`, so that excerpt is left out.
3. Move the repository's `stm_from_dft/sim_decay_full.py` and `stm_from_dft/tier3_decay.py` into `wrong_decays/` with `git mv` (keeps their history), with no change to their content. They are byte-identical to the owner's moved copies.
   - `git mv LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/sim_decay_full.py LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/wrong_decays/sim_decay_full.py`
   - `git mv LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/tier3_decay.py LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/wrong_decays/tier3_decay.py`
4. Give `wrong_decays/` no `__init__.py`. Nothing imports these files. After the move, nothing remains at `stm_from_dft/sim_decay_full.py` or `stm_from_dft/tier3_decay.py`.

### 1.2 Copy the owner's files over the repository's

- `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py`
- `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/__init__.py` (its docstring says `wrong_decays/` holds three unused files; with `tilted_barrier_decay_github.py` it holds four)
- `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/feenstra.py` (new; `simulator.py` imports it)
- `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/wrong_decays/tilted_barrier_decay.py` (new; the owner's file, as it is)
- `LDOS_spread_decomp/stm_from_dft_opt/README.md`. Its `wrong_decays/` row also names `tilted_barrier_decay_github.py`: GitHub's earlier version of the same model (float32 arrays, the `1e-6` V floor, `max(0.1, phi)`, no Feenstra).
- `qtgui/widgets.py` (one added spec line, `'width (V)'`)
- `Claude outputs/` at the repository root (new; three patch files: `chen_float64_changes.patch`, `chen_plain_changes.patch`, `feenstra_changes.patch`)

Not in `disk/`: `run_DFT_STM.py`, whose only local differences are the owner's own paths and run values (the repository copy keeps its example ones, `'dir/run'`), and Python cache files (`.gitignore`).

## 2. Rules for this job

1. Implement exactly what this plan says. If the code does not match the plan, or a step seems to need a change the plan does not name, stop and report it instead of choosing.
2. Change no computation: LDOS, tip-height convergence, decay, Feenstra, caches and their file names stay as they are.
3. Add no tests or checks, and run no window (no Qt offscreen platform). The only check is `python -m py_compile` on every changed `.py` file.
4. Keep the simulator names that `mo_ldos_projection/sim_wrapper.py` calls: `_update_all`, `_calculate_ldos_at_points_gpu`, `_combine_ldos_display`, `add_control_box`, `_toggle_run`, `fig`.
5. The repository carries example paths only.
6. `matplotlib_to_real_gui.md` stays unchanged. Where it disagrees with this plan for these three packages (rule 11's trimming with `fit_window`, sizes in pixels), this plan governs.
7. At the end, list per file every function, constant and branch you changed, added or removed.

## 3. One qtgui copy per package, no folder walking

### 3.1 The copies

Copy the root `qtgui/` (`__init__.py`, `qtplot.py`, `controls.py`, `widgets.py`, `mplexport.py`, with the disk `widgets.py`) to:

- `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/`
- `LDOS_spread_decomp/MOPmatcher_mem/matcher/qtgui/`

Every change in sections 4 and 5 goes into both copies identically, so the two copies stay byte-identical. The root `qtgui/` is not changed. The other packages of the repository keep using it.

### 3.2 Inside each copy: relative imports, no walks

- `__init__.py` (about lines 15 to 24): remove the walk and `from qtgui import ...`; import its modules relatively (`from . import qtplot as plt, controls, widgets`), exporting the same names as now.
- `controls.py`: remove the `sys.path` insert (about line 31); lines 36 to 39 become `from . import qtplot as plt`, `from .qtplot import QtCore, QtGui, QtWidgets, qfont, qcolor`, `from . import widgets`, `from .widgets import SPECS, LABELS, short_form`.
- `widgets.py`: remove the `sys.path` insert (about line 26); lines 28 to 29 and 381 become relative (`from . import qtplot as plt`, `from .qtplot import QtWidgets, QtGui, qfont`, `from . import controls`).
- `mplexport.py` line 30: `from . import qtplot as q`.
- `qtplot.py` about line 7404: `from . import mplexport`.
- Docstrings in the copies that say the layer is at the repository root say it is the package's own copy.

### 3.3 `stm_from_dft/__init__.py`: lazy

Make it lazy with a PEP 562 module `__getattr__`. Keep the docstring, `__all__` and the five exported names. `__getattr__` imports the submodule that defines a requested name on first access (`doscar_parser`, `locpot_manager`, `simulator`). Importing `stm_from_dft.qtgui` must not import `simulator.py` (and so not CuPy).

### 3.4 Simulator package

- `simulator.py`: remove the walk block (about lines 25 to 33) and import from the package's copy with the same names as now: `from .qtgui import qtplot as plt`, `from .qtgui import controls, widgets`, `from .qtgui.qtplot import QtWidgets, QtCore`.
- `plot_dos_cursors_levels.py` (about lines 12 to 25) is also run directly (its `__main__` block at the bottom). Use `from .qtgui import ...` when imported as part of the package. When run directly, add the package's parent folder (`stm_from_dft_opt/`, a fixed path from its own file, no upward search) to `sys.path` and import `stm_from_dft.qtgui`.
- `run_DFT_STM.py`: remove the walk (lines 11 to 16); line 17 becomes `from stm_from_dft.qtgui.qtplot import LinearSegmentedColormap`. Nothing else in the file changes.

### 3.5 MOPmatcher_mem

- The `matcher/` modules that walk to `qtgui` (`vacupot_plotter.py`, `spread_plot.py`, `combined_vacupot_plotter.py`, `state_decomposition_plot.py`, `matcher_vacupot.py`, `spread_collapse.py`): remove the walk and import the same names from `.qtgui`.
- Some of these modules can also be run directly and already set up their own package for that run (`spread_collapse.py`, `state_decomposition_plot.py`, `spread_plot.py` with a stub `matcher` package in their `__main__` blocks; `matcher_vacupot.py` with its `sys.path` insert and `try`/`except` imports). Follow each module's existing pattern: the relative import when imported as part of the package, and `matcher.qtgui` in its direct-run branch.
- `run_vacupot_plotter.py`, `run_MOPMatcher.py`, `run_combined_MOPMatcher.py` (in `MOPmatcher_mem/`): remove the walk; `from matcher.qtgui import qtplot as plt` (they already import `matcher`).

### 3.6 mo_ldos_decomp

- `mo_ldos_projection/_imports.py`: add `resolve_qt()`. It returns the simulator's copy as `(qtplot, controls, widgets, mplexport)`, found the same way `resolve_stm()` finds the simulator: the vendored `.stm.qtgui` first, else `stm_from_dft.qtgui` with the same `stm_from_dft_opt` path insert. It must not import `simulator.py`.
- Remove the walks from `mo_ldos_projection/sim_wrapper.py`, `decomp_window.py`, `topo_decomp_window.py` and `global_decomposition.py`, and from the run scripts `mo_ldos_decomp/run_stm_decomp.py` and `run_wfn_plane_slice.py`. They take `plt`, `controls`, `widgets` and `mplexport` (and the names they import from `qtplot`, such as `ScalarMappable`, `LinearSegmentedColormap`, `Normalize`) through `resolve_qt()`.
  - `sim_wrapper.py` must use the simulator's copy, because it builds its controls into the simulator's figure (`controls.SigmaBox(fig, ...)`, `controls.make(fig, ...)`, `sim.add_control_box(...)`).
  - `run_wfn_plane_slice.py` about line 353 (`from qtgui.mplexport import _mpl_norm`) takes `_mpl_norm` from `resolve_qt()`'s `mplexport`.
  - Modules with a direct-run branch (`global_decomposition.py`, `sim_wrapper.py` with their `if __name__ != "__main__":` imports) follow their existing pattern.
- The matcher plotter class `_imports.py` already takes from `matcher` (`RectAEPAWColorPlotter`, used for its data preparation) keeps coming from the matcher package, with the matcher's copy.
- `mo_ldos_projection/sim_wrapper.py` also drops its calls to names the owner's `simulator.py` (from `disk/`) no longer has. They belonged to the old decay model's '0-clamp' toggle:
  - `capture_state` (about line 273): remove `state["t2_clamp"] = sim.t2_clamp`. The simulator has no `t2_clamp`, so this line raises `AttributeError` on a σ commit.
  - `_instantiate` (about line 207): remove `"t2_clamp"` from the list of restored attributes.
  - The topography send (about lines 448 to 451): remove `clamp_occupied=False` from the `_calculate_ldos_at_points_gpu` call (the simulator no longer accepts it, so the call raises `TypeError`), and the comment sentence about `clamp_occupied`.
  - Nothing else in `sim_wrapper.py` changes for this. Every other simulator name it uses exists in the owner's `simulator.py`.

## 4. Every size from font metrics or the screen's size

Rule: every size on screen comes from font metrics or the screen's size. Any size that does not is replaced. This applies in both `qtgui` copies and in every in-scope file that draws: `simulator.py`, `plot_dos_cursors_levels.py`, the `matcher/` plotters and run scripts, and the `mo_ldos_decomp` windows and run scripts.

### 4.1 Units (in `qtplot.py` of each copy)

- `base_font()`: the application font set to 10 pt with `setPointSizeF(10)`.
- `em_px()`: the base font's size in pixels on the window's screen, from `QFontInfo(...).pixelSize()` ("the pixel size of the matched window system font"). Measure on a 100 pt copy of the base font and divide by 10, so the integer pixel size does not round the result. Measure again when a window moves to another screen (`QWindow.screenChanged`).
- `px_per_pt() = em_px() / 10`.
- `pt2px(pt)` becomes `pt * px_per_pt()`. The constant `DPI` is no longer used for anything on screen.

### 4.2 Fonts in points, never below 10 pt

- `qfont` (about line 382): set the size with `setPointSizeF(max(10.0, font_pt(size) * FONT_SCALE * zoom))` instead of `setPixelSize`. Key the cache on the point size. (QFont docs: `setPixelSize` makes a font device dependent; `setPointSize`/`setPointSizeF` set it in a device-independent way.)
- The application default font in `_app()` (about lines 7593 to 7596): `max(10.0, pointSizeF * FONT_SCALE)`.
- The legend (about line 5711): `labelTextSize` as a point string, `f"{max(10.0, font_pt(fontsize, 10.0) * FONT_SCALE * zoom):.1f}pt"` (pyqtgraph takes CSS sizes such as `'9pt'`).
- Any other text that does not go through `qfont` (search `setFont`, `setPixelSize`, `setPointSize`, `labelTextSize`, `size=` on pyqtgraph `LabelItem`/`TextItem`, HTML font sizes) goes through `qfont`.
- `_font_px` (about line 1605) reads `pixelSize()`, which is -1 for a point-sized font. Measure that font with `QFontMetricsF` / `QFontInfo` instead.
- The text zoom keeps working and can grow text. Nothing may set a font below 10 pt. `_fit_controls` may shrink control text toward 10 pt to fit, never below. When the layout still does not fit, it raises the canvas's minimum size, as it does now in its last step.
- Font size specs in points (`fontsize=10`, `fs=7`, `labelsize=6`, `qfont(7.5)`) keep their values. The floor raises the small ones.

### 4.3 Layout and drawing sizes

Replace each hard-coded size with one of these:

- **Room around a text** (row heights, box heights, label gaps, the info panel's fixed height at about lines 6402 and 6695): that text's own `QFontMetricsF` (`lineSpacing`, `height`, `horizontalAdvance`).
- **Everything else**, as a multiple of `em_px()`:
  - margins (`setContentsMargins(6, ...)`, `leg.layout.setContentsMargins(2, ...)`), spacings (`spacing=4`, `spacing=8`, `horSpacing=4`)
  - panel minimums (`min_px=(40, 40)`), the stripe's `inner_w=58`, colour-bar widths, `CURVE_BOX_MIN_PX = (220, 140)` in `simulator.py`
  - legend and canvas minimums, the toolbar icon size (`22 * screen_scale()`), control and box sizes in `controls.py` and `widgets.py`
  - pick tolerances (`picker=5`; convert it as points or as pixels, whichever `qtplot` reads it as today)
- **Sizes given in points** (matplotlib's `lw`, `linewidth`, `ms`, `s`, `markersize`, tick lengths, frame widths, arrow heads, annotation offsets in points, the stripe's 40-point stroke): keep their point values; they go to pixels through `pt2px()`.

Conversion of pixel constants: a constant of `c` pixels becomes `c / (1000 / 72)` times `em_px()`. `1000 / 72` px is the em of a 10 pt font at the old fixed 100 DPI, so every size keeps its proportion to the text. Where today's code multiplies a size by the text zoom (for example `CURVE_BOX_MIN_PX` in `_curves_readable`, tick lengths in `qtplot.py` about line 1859), keep that factor. Round to whole pixels only where a Qt call needs an integer.

### 4.4 Window sizes

- The design size (`Figure._design_px`, the reference of `window_zoom`; about line 6513) is `figsize * 72 * px_per_pt()` instead of `figsize * DPI`.
- `fill_screen` windows open on the screen's available area, as now. Other windows open at their design size, limited to the screen's available area.
- `mplexport` converts screen pixels back to points with the same `pt2px` (about line 93), so the publication files keep matching the screen. matplotlib's own `dpi` for written files (for example `savefig(..., dpi=300)`) is not a screen size and stays.

### 4.5 Not sizes (they stay)

- font sizes in points (subject to the floor)
- lengths in data units (Å, eV): the Line Topo's `margin = p_len * 0.15`, `half_w`, energy ranges
- ratios and factors: stretch factors, `FONT_SCALE`, `ZOOM_MIN`/`ZOOM_MAX`, `CONTROL_FIT_MIN`, hysteresis, `TOPO_MAX_FACTOR`, box aspects
- counts (contour levels, tile counts) and times

## 5. Window arrangement (both qtgui copies; every in-scope window)

### 5.1 Never trim

- `Figure._relayout` (about line 6641): no `self._schedule_fit()` call.
- `show()` (about line 7695): no `f.fit_window()` call.
- `simulator.py` `_rearrange` (about line 1611): no `self.fig.fit_window()` call.
- Any other call of `fit_window` or `_schedule_fit` in the three packages is removed. The functions may stay defined, but nothing calls them. A maximized window stays full screen.

### 5.2 Panels move only on three triggers

Panels are arranged and moved only when:

1. the number of panels changes
2. a crop changes a panel's shape
3. the user resizes the window, maximizing included

Nothing else moves or resizes a panel.

- `Arrangement` (`controls.py` about line 2160) keeps choosing on its own `resizeEvent` and on `rearrange()`, always at the window's actual size.
- Its `eventFilter` (a child's `LayoutRequest` leads to `_arrange`) re-places only when a group's shape changed (the crop case).
  - Today `_arrange`'s signature includes `self._free_min(W, H)`, which changes when a label's width changes. Such a change no longer leads to a new arrangement.
  - A `LayoutRequest` from text inside a panel (a tick label, a colour-bar label) changes nothing.
- A cell's measured room for its labels (`PanelCell` decor, `Axes._decor_px`, `pass_cached`) is measured at an arrangement and kept until the next one.
  - Inside its cell, an axes may move its plot box to fit its labels (`Axes._place`).
  - The cell itself and every other cell stay where they are.
- Apply the same to `MapBlock` (`controls.py` about line 1851) and any other container that lays out again on its children's layout requests.

### 5.3 Unchanged

- the candidate arrangements, `_score`, the hysteresis and `group_cap` (the topographies at most `TOPO_MAX_FACTOR = 2` times a heatmap)
- the curve panels' handling: their readable minimum (`free_min`, `_curves_readable`), then the room the groups leave
- `free_scored` in Single Point, the spectrum as one cell of the Map block, the unit-cell panels' orientation rule

### 5.4 The simulator's panel shapes

- **Keep their x:y ratio (equal scale):** the placement topo / Global Topo, the Map Topo and the energy maps (`set_aspect('equal')`, as now), and the Line Topo.
- **Line Topo** (Line view, about lines 2472 to 2530):
  - `half_w = p_len / 3.0` (was `p_len / 2.0`) and `set_box_aspect(1.5)` (was `1`; qtgui's box aspect is height over width).
  - The panel shows −L/3 to +L/3 across the line by 0 to L along it.
  - The atom overlay and the data mask already use `half_w`. The 0.15 L margin stays.
- **Line/Path heatmaps** (about line 2573): remove `ax_l.set_box_aspect(1)`.
  - The heatmap row stays a shaped group of the `Arrangement` and is sized and scored exactly as today, each heatmap counted as square. Its place and priority do not change.
  - Once a candidate is chosen, the row's rect is its whole slot instead of its shaped size inside the slot, and the heatmaps fill it, stretching in x or y.
  - They never take room from another group's slot or the curve panels' slot.
- **The stripe:** keeps its width (now from 4.3) and follows its row's height.

### 5.5 Docstrings

Docstrings in `simulator.py` that describe trimming or arranging on every rebuild (the view-build docstring about lines 1356 to 1366, `_rearrange`) describe the new behavior.

## 6. Drawing: build once, update in place (simulator.py)

### 6.1 Build

- Panels are built once per view, as now in the view build.
- The Line/Path heatmap row and the Map block are rebuilt only when their panel count changes: the drill-down level changes the number of partitions, or the per-heatmap colour bars are toggled. Then `_rearrange()`.
- Today the row is rebuilt on every `_update_all` (about lines 2439 to 2626, with `_rearrange()` at about 2605 and 2643), and the Map's energy maps are cleared and drawn again on every call (about lines 2672 to 2687). Both stop.

### 6.2 Data changes update items in place

- **Heatmaps:** `QuadMesh.set_array` and `set_norm`. A new mesh in the same axes only when the grid shape changes (a new energy range).
- **The stripe:** `LineCollection.set_array` and `set_norm`.
- **Curves** (profile, spectra): `Line2D.set_data`.
- **Markers:** `PathCollection.set_offsets`, `Line2D.set_xdata` / `set_ydata`, `_RefLine.set_xdata` / `set_ydata`.
- **Colour bars:** their norm.
- **Energy maps:** `TriContourSet` has no `set_array`, so a map whose data changed has its contour set removed and drawn again in its own axes, with no `ax.clear()` and no rebuild of the block. A map whose data did not change is not touched.
- **The Line Topo's contour:** drawn again when the line changes.
- **Redraw:** one `canvas.draw_idle()` after the updates.

### 6.3 `_update_all`

`_update_all(full_refresh=False)` stays the entry point (`sim_wrapper` calls it).

- It works out what changed and recomputes what that change needs, exactly as it computes today.
- It rebuilds only on a structural change (6.1). Otherwise it updates in place (6.2).

### 6.4 Clicks

- `_on_press` (about line 2818) no longer calls `_update_all` on every click. Only the spectrum's step-back click (the `plot_level` decrease) updates the display.
- `_on_pick` (about line 2775), the drill-down: updates the spectrum, and rebuilds the heatmap row only when the partition count changes.
- Selecting a marker (the `mark_dynamic` pick) updates only what shows the active marker.

### 6.5 Drags

No drag calls `_update_all`, does GPU work or recomputes anything while the button is down. `_on_motion` returns early when the dragged item's position has not changed.

- **Line and Path markers** (`marker_ratios`):
  - Draggable from the placement topo, the Line Topo (Line), the stripe, every heatmap and the Tip Height profile (`ax_prof`). All counterparts move together.
  - The marker spectra update during the drag from a display cache: the displayed spectrum of every line or path point, built once per data change through the same display path as today (`_combine_ldos_display` and the active display options).
  - `_on_motion` (about line 2835) today accepts only `ax_prof`, `ax_ldos` (always `None`), `ax_stripe` and `ax_line_topo`. It accepts every heatmap axes too, and the heatmap marker lines (`picker=5`) can be dragged.
- **Line ends (Line) and path nodes (Path):**
  - During the drag only the line or path and its end or node markers move on the placement topo.
  - On release, the tip heights, the LDOS along the line, the heatmaps, stripe, profile, Line Topo and marker spectra update: the work today's `_update_all` does on every motion, done once.
- **Single Point markers:** draggable on the placement topo. The spectrum is computed on release, by the same computation as today.
- **Map markers** (`marker_coords`):
  - Draggable from the Map Topo and from every energy map. All counterparts move together.
  - The spectrum is computed on release, by the same computation as today.
  - The energy maps' marker scatters (about lines 2685 and 2762) get pickers.
- **Unchanged:** the Map energy markers in the spectrum (`emarker`: they move during the drag, and the map slice is recomputed on release), the crop cursors and the contour level line.

`_on_rel` (about line 2896) does the release work above, then clears `active_obj`.

## 7. README text

Change only the sentences that describe what this part changes.

- **`README.md` (root):**
  - the checks paragraph (about lines 98 to 106) and the `checks` table row (about line 25): only `render_windows.py` and `check_misc_windows.py` remain
  - the dependencies (about line 37): drop pytest, which was only for the removed tests
  - where it says every window uses the root `qtgui`: `stm_from_dft_opt` and `MOPmatcher_mem` carry their own copies, and `mo_ldos_decomp` uses the simulator's copy
- **`LDOS_spread_decomp/stm_from_dft_opt/README.md`:**
  - the `qtgui` location (about lines 20 and 81 to 82)
  - remove the check sentence (about lines 83 to 84)
  - the panel and sizing paragraphs (about lines 22 to 30 and 86 to 89): the window is never trimmed, the heatmaps fill their room, the Line Topo is 2:3, text is never below 10 pt, and line ends, path nodes and Single Point/Map markers recompute on release
- **`LDOS_spread_decomp/MOPmatcher_mem/README.md`:** the `qtgui` location (about lines 23 to 24 and 87 to 88); remove the `check_matcher_plots.py` sentence (about line 55).
- **`LDOS_spread_decomp/mo_ldos_decomp/README.md`:**
  - the `qtgui` location (about lines 9 to 10 and 37)
  - remove the `tests/` row (about line 33) and the `check_decomp.py` sentence (about line 60)
  - drop "plus pytest for the tests" (about line 68)

## 8. Finish

- `python -m py_compile` every changed `.py` file. Run nothing else.
- Commit on a new branch. The owner tests on his own machine.
- Report the per-file list of every function, constant and branch changed, added or removed (rule 7).
