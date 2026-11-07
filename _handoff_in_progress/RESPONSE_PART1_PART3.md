# Response document

This document gives the owner's answers to `PART3_QUESTIONS.md` and to `PART1_OPEN.md`, and his decisions made since. Where it differs from `handoff_part3_spectrum_decomp.md` or an earlier answer, this document stands. Each answer gives its basis. Nothing here waits on the owner.

Sections:
1. Part 1's open items (`PART1_OPEN.md`)
2. What is taken from sm4_browser (the owner's rule)
3. 720 dpi for every saved file that is not a matplotlib vector file
4. matplotlib is required
5. The answers to `PART3_QUESTIONS.md`
6. Two colour-map sets in the simulator (added to part 3)

## 1. Part 1's open items (`PART1_OPEN.md`)

The progress log still lists these as not received ("Owner's answers to `PART1_OPEN.md`; apply them"). They are:

**Accept every R except P2.**

**P2, the Line view:** take (a), the shared height, with the leftover height going to the 1D plots. The Line view has two rows:

- **Top row:** the global topography, the Line Topo (its 2:3 box), the topo stripe and the heatmaps. Across the Line Topo, the stripe and the heatmaps the distance axis lines up: a marker at distance s sits at the same height in each of them.
- **Second row:** the topo path 1D plot and the LDOS 1D plots. The markers are drawn in their locations and colours throughout every panel, as now.

How the height is shared:

- **The top row:** its xy-spatial panels grow with the window, keeping their x:y ratio, up to the largest size the row's width allows. The top row's height is the height those panels then take.
- **The second row:** when the window is taller than that, the 1D plots expand vertically and take all the remaining height. No empty band is left above, between or below the rows.
- If something in the code keeps this from working, stop and report it instead of choosing.

**P13, the plane-slice runner's new matcher dependency:** take (b).

- **New module `mo_ldos_projection/_resolve.py`:** it holds `resolve_stm()` and `resolve_qt()`, plus `resolve_doscar_parser()` from the part 2 pre-flight answers once that is added.
  - At module level it imports nothing beyond `os` and `sys`. Each resolver imports only inside its function, as `resolve_stm()` does now.
  - It uses the same vendored-copy-first lookup and the same `stm_from_dft_opt` path insert as today.
- **`_imports.py`:** keeps its matcher imports, `resolve_run_match` and `resolve_blochstates_check`. It re-exports the three resolvers (`from ._resolve import resolve_stm, resolve_qt, resolve_doscar_parser`), so its other callers don't change.
- **`run_wfn_plane_slice.py`:** imports `resolve_qt` from `mo_ldos_projection._resolve`, so a plane-slice run no longer loads `MOPmatcher_mem`.

**The rest:** every other item as recommended (P1, P3 to P12).

**One later change to P10:** P10 kept `export_vector`'s resolution and page size at `DPI`. For `export_vector` that is replaced by section 3 (720 dpi). The rest of P10 stands, including mplexport's `Figure(..., dpi=q.DPI)`.

## 2. What is taken from sm4_browser (the owner's rule)

The owner (2026-11-07): "Anything from the browser that covers something I spent time clarifying today should not rewrite stuff I clarified. anything I said to get from the browser has parts that are not parts I already worked on that can be borrowed as is".

So:
- **Where sm4_browser's code covers something settled** in parts 1 to 3 (the plans, the owner's rulings and his accepted R items), what was settled stands, and that part of the browser's code is not taken as it is.
- **Everything else** in part 3 section 6's pieces is taken as it is.

The places where section 6's pieces cover something settled are below. Line numbers are sm4_browser's at `eb9bb6279e70`.

**In the two package copies** (the simulator's `qtgui` and the matcher's), part 1 stands:

- **The info panel's height:** the branch's `QFontMetricsF(self.info.font()).lineSpacing() * 4.2` (part 1 §4.3), in `FigureManager.__init__` and in the text zoom. Not the browser's `pt2px(7.5) * FONT_SCALE * 4.2` (`qtplot.py` lines 4758 and 5050).
- **Its font:** `qfont(7.5)` keeps part 1's 10 pt floor (part 1 §4.2).
- **mplexport's conversions from screen pixels go through `pt2px`** (part 1 §4.4, "so the publication files keep matching the screen"):
  - the vector files' figure size (`mplexport.py` line 112);
  - the info block's text size (`mplexport.py` line 507).
- **What stays at `DPI` by part 1's accepted P10** ("written files, not screen sizes"): mplexport's `Figure(..., dpi=q.DPI)`. The browser uses `DPI` here too, so it is taken as it is.

**In every copy, the root `qtgui/` included**, part 3 stands:

- **The Qt-drawn SVG (`'qt-svg'`):** exactly as shown. The browser's `export_vector` draws the window, then adds the full info text below it. For this save, nothing is added below the window. Its label is under Q7 (section 5).
- **The vector saves' info block:** below the canvas, or below the legend panel's bottom when the legend panel reaches past the canvas. The browser places it below the canvas only.

**The root `qtgui/`:** part 1 did not change it, so its sizes stay at 100 px per inch (`DPI`), as sm4_browser's are. Sections 3 and 4 apply to it too.

## 3. 720 dpi for every saved file that is not a matplotlib vector file

In every `qtgui` copy, the root included.

- **The owner (2026-11-07):** "the 720dpi for saved images and files (if not vector files) should remain though". Of `export_vector`'s resolution: "this is part of the 720 list".
- **What a 720 dpi file is (the owner, 2026-11-07):** "I want these to be exactly the stuff on screen in that size, but with extra pixel interpolated or wtv is necessary. the font and panels in the image should all be the same relative to eachotehr as they were on teh screen".
  - The file is the window exactly as it is on screen, at its on-screen size and layout. Nothing is laid out again, resized or re-fonted for the file.
  - Only the pixel density goes up to 720 dpi: the window is drawn at that density, with raster content interpolated where needed.
  - Every font and panel keeps its size relative to the others, as on screen.
  - The browser's `window_image` already works this way: it renders the window into an image `k` times its size through a painter scaled by `k` (`p.scale(k, k)`, then `root.render`, `qtplot.py` lines 5584 to 5593), with no new layout.
- **The PNG and TIFF window images:** at 720 dpi, as the browser writes them ("the window at 720 dpi", `WINDOW_IMAGE_DPI = 720`, `Figure.window_image`, its scale `k = dpi / DPI` at `qtplot.py` line 5584 taken as it is).
- **The Qt-drawn "as shown" SVG** (`export_vector`): at 720 dpi instead of `DPI` (`gen.setResolution`, `qtplot.py` line 5730), at the same physical size as the window images, and by the same rule: the window exactly as on screen, only at the higher density. This replaces part 1's P10 for `export_vector`.

## 4. matplotlib is required

The owner (2026-11-07): "matplotlib is necssesary and the code shouldn;t even run without it. no fallback is necessary".

In every `qtgui` copy (the two package copies and the root):

- **`qtplot.py` imports matplotlib at module level** (`import matplotlib`), so importing `qtgui` stops with Python's own `ModuleNotFoundError` when matplotlib is not installed. No window runs without it.
- **The PDF, EPS and editable SVG saves always go through `mplexport`.** These parts of the browser's code are not taken:
  - the fallback in `Figure.savefig`'s `'vector'` path (`if _mplexport.available(): ... else` the console note and `export_vector`);
  - the save dialog's filter that drops the vector saves when matplotlib is missing (`NavigationToolbar._save`, `if not _have_matplotlib(): ...`, browser `qtplot.py` lines 4685 and 4686). The dialog always offers the full list.
- **`export_vector`** is used only for the Qt-drawn "as shown" SVG.
- **The colour map lookup** (`to_cmap`) is left as it is. Its built-in maps also serve names matplotlib doesn't know (pyqtgraph's CET maps, section 6).

## 5. The answers to `PART3_QUESTIONS.md`

**Q1, the legend panel's scrolling stack: (a).**

- It is a Qt vertical scroll bar at the panel's right edge, shown only when the entries don't fit, plus the mouse wheel over the panel.
- **Basis:**
  - the owner chose a plot panel saved with the figure;
  - he asked for scrolling identical to sm4_browser's scroll areas, whose bar is a Qt scroll bar.
- The vector saves leave the bar out, as every control. The Qt-drawn SVG and the window images show it as shown.

**Q2, a title wider than the panel: (a).**

- **Basis:** the owner's rule that the simulator's code runs no checks, which rules out (b).
- Section 2's "the code session stops and reports it" in `handoff_part3_spectrum_decomp.md` does not apply.

**Q3, legend width vs part 1 §5.2: (a).**

- One fixed width: the largest entry over all four levels, from the loaded data.
- **Basis:** part 1 §5.2, "Nothing else moves or resizes a panel".
- **The scroll bar (the owner, 2026-11-07):** "add the scrollbar on the side of the panel".
  - The bar sits at the panel's side, beside the entries, never over them.
  - The fixed width includes the bar's width, so the bar never covers an entry and never widens the panel on a click. When no bar is shown, that strip stays empty.

**Q4, pixels to points in the ported saves:** as in sections 2 and 3.

- **Through `pt2px`:** mplexport's figure size and info text size (part 1 §4.4).
- **At `DPI`:** mplexport's `Figure(dpi=q.DPI)`, by part 1's P10.
- **At 720 dpi:** the PNG and TIFF window images and the Qt-drawn "as shown" SVG (section 3).
- **Basis:** part 1 §4.4, part 1's P10 and the owner's rules in sections 2 and 3.

**Q5, the root `qtgui/` gets section 6: (a).**

- **Basis:**
  - section 6 says "every window in this repository that uses `qtgui` ... and the rest";
  - the owner's decision was every `qtgui` window in VASP_for_SPM.
- It keeps its `DPI` sizes and gets the rules of sections 2 to 4.

**Q6, settings held in no control: (a).**

- **The owner's answer (2026-11-07):** (a).
- **What goes in:** every setting the window makes with the mouse, each as one of the window's own lines (`set_info_line`), updated whenever it changes. In the simulator that is:
  - the decomp level, the active marker and the active element;
  - the marker positions, the Line ends and the Path nodes;
  - Map's energy markers;
  - the crop rectangle.
- **The browser does the same** for its Line window's E-markers (`set_info_line('E-markers', ...)`, `line.py` line 1821).

**Q7, the label of the Qt-drawn SVG save:**

- **The label:** "SVG, the whole window as Qt draws it, controls included, as shown (*.svg)".
- **Basis:** the owner's rule that this save is exactly as shown.
- **The code:** this save also leaves out the full info text that the browser's `export_vector` adds below the window.

**Q8, the hook's name: (a), keep `sm4_settings_changed`.**

- **Basis:** the owner's rule in section 2. Nothing settled covers the name, so the browser's code is taken as it is.

## 6. Two colour-map sets in the simulator (added to part 3)

Files: `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py`, `LDOS_spread_decomp/stm_from_dft_opt/run_DFT_STM.py` and `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py`.

Part 3's rules 1, 3, 5 and 7 apply (rule 2 was dropped by the owner's ruling 9). The `simulator.py` edits wait with part 3's, until part 2's workflow has finished.

**Why part 3 (the owner):** "this should be part of the guit thing inpart 1 really, but if its too late for that then add it as part 4 unless it works and makes sense as part 3".
- Part 1 is implemented.
- Part 3 is not yet, and it already changes the simulator's window.
- The set choice is a control, so part 3 section 6's settings section lists it in the info panel with no extra work.

### The two sets (the owner's rule)

The owner: "I want the current set of default color maps to be saved as default 1 and then default 2 to use CET-l03 for topo, CET-l06 for the map cmap, and then CET_r1 for the line ldos heatmap." Of the heatmaps: "line and path should share the same color map for the heatmap." Of the Mag display: "CET-D01A for the mag cmap." Of |m|: "CET-L19 for |m|".

| Role | Default 1 (today's maps) | Default 2 |
|---|---|---|
| **Topo:** every panel drawn with `cmap_topo` (global topo, LDOS topo, the Line / Path stripe, the topo colour bar) | black, firebrick, yellow: `LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])`, today passed in by `run_DFT_STM.py` and `sim_wrapper.py` | `'CET-L3'` |
| **Map energy maps** (`cmap_map`) | `MAP_CMAP` (midnight blue to light sky blue) | `'CET-L6'` |
| **Line and Path LDOS heatmaps** (one map for both) | `'nipy_spectral'` | `'CET-R1'` |
| **Mag, signed spin density** (collinear spin-polarized runs: up minus down, `_mag_is_signed`), in the maps and the heatmaps | `'bwr'` | `'CET-D1A'` |
| **Mag, \|m\|** (NCL / SOC runs: unsigned, always ≥ 0), in the maps and the heatmaps | `'hot'` | `'CET-L19'` |

**The names:**
- These are pyqtgraph's names for the maps the owner wrote as CET-l03, CET-l06, CET_r1, CET-D01A and CET-L19: `'CET-L3'`, `'CET-L6'`, `'CET-R1'`, `'CET-D1A'` and `'CET-L19'` (pyqtgraph's bundled files, `pyqtgraph/colors/maps/`).
- They load through `qtplot.to_cmap` as names, with no new loading code. matplotlib does not know these names, so `to_cmap` goes on to its built-in lookup (`_builtin_cmap`), which takes them from pyqtgraph (`pg.colormap.get`).

**Any CET map by name (the owner's answer):** any CET name already works wherever a colour map is accepted. No picker is added.

### Choosing the set

- **In the window (the owner's answer):** a choice in the window's left column between "default 1" and "default 2". Changing it recolours every panel in place, with no rearrangement (part 1 §5.2) and no recomputation.
  - The heatmaps already change their map in place (`mesh.set_cmap` in `_fill_heat_row`), and so do the Map colour bars (`sm.set_cmap`).
  - If a panel's colour map cannot be changed in place, stop and report it (rule 1).
- **In the run script (the owner's answer):** the run script names the set the window opens with, as a new constructor argument of `Interactive_STM_Simulator`, `cmap_set` (`'default 1'` or `'default 2'`).
  - **The set it opens with (the owner):** "opens with set 1 as it does currently". `run_DFT_STM.py` names `'default 1'`, and `cmap_set`'s default is `'default 1'`.
- **`cmap_topo` and `cmap_map` go (the owner's answer).** The constructor's `cmap_topo` and `cmap_map` arguments are replaced by `cmap_set`.
  - Both sets are defined in `simulator.py`.
  - `MAP_CMAP` stays, as default 1's Map entry.
- **`run_DFT_STM.py`:** only its colour-map lines change. Every other line, its paths and settings included, stays as it is.
  - Its `cmap_topo=...` line becomes `cmap_set='default 1',`.
  - The commented `cmap_map` line and its comment become one comment that names the two sets.
  - The `LinearSegmentedColormap` import goes, since nothing else in the file uses it.
- **`sim_wrapper.py` (the owner):** "that is part of set 1 so it must apss that". Today it passes its own copy of default 1's topo map, black, firebrick and yellow, as `cmap_topo` (`__init__`, about lines 86 and 107 to 112). It passes the set instead:
  - its `cmap_topo=None` argument becomes `cmap_set='default 1'`, passed on to the simulator in place of `cmap_topo`;
  - its own black, firebrick and yellow map goes, since default 1 holds it.

  This is the one change to `sim_wrapper.py`, an exception to part 3's rule 4 for this only. Nothing in the repository passes `cmap_topo` to the wrapper.

### Where the maps are used today (`simulator.py` on the branch, function names)

- **Topo (`self.cmap_topo`):**
  - the topo colour bar's ramp (`_draw_topo_cbar`);
  - the global topo and LDOS topo contours (`_update_all`);
  - the Line topo contour and the stripe's `LineCollection` (`_fill_heat_row`).
- **Map (`self.cmap_map`):** `_draw_map_slice` and `_map_cmap`, which also picks `'bwr'` and `'hot'` for Mag.
- **Heatmaps:** `_fill_heat_row`, `'nipy_spectral'` written into it, and `'bwr'` and `'hot'` for Mag.

### Not changed

The contour slice's colours (`CONTOUR_COLOR`, `CONTOUR_LEVEL_COLOR`), the marker colours, the decomp curves' colours, and every other window's colour maps.
