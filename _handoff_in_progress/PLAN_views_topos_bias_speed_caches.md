# Plan: the views-layout decisions, the bias toggles, the topos, the LDOS speed-up, |m| and m on demand, the caches folder, the console lines and the shading

Draft of 2026-11-07, for Benjamin's approval. Nothing in it is implemented. Your answers to the questions are written into the sections. One question is still open, in section 11. Section 12 holds what the check of the briefs against your other documents found still missing.

## 0. Sources

- **Sections 2 to 5:** your simulator-views layout decisions document (`PART3_simulator_views_layout_decisions.md`, uploaded 2026-11-07 06:29Z). None of it was implemented. Its rules are quoted or closely transcribed here, each followed by what the code does now and the change.
- **Sections 6 to 10:** your chat after the close-out (2026-11-07, from about 22:15Z): the topo console labels, the speed-up, |m| and m, the caches, the shading.
- **Your answers** to the plan's questions (2026-11-07) are written into the sections. Where you accepted my recommendation, that recommendation is written in.
- **One statement in the decisions document doesn't match the code.** It says Path's LDOS topo panel is "already sized for what it shows (… Path's turned rectangle with 1/8 margins)". Neither your copy from before part 1 nor the current code has a Path LDOS topo panel (`ax_line_topo` is `None` outside Line). It is new here, built by the Path rules in 3.5.

## 1. Rules for the job (standing)

- Implement exactly what is approved. If anything needs changing outside it, stop and ask. Every delivery lists the approved changes and any additional ones separately, and stops if the additional list is not empty.
- Nothing in `wrong_decays/` is touched.
- Agents read files and run only `python -m py_compile`: no tests, no windows. Every agent is Opus 5.5 or higher.
- The two package `qtgui` copies (`stm_from_dft/qtgui` and `matcher/qtgui`) stay byte-identical. The root `qtgui/` follows them for every `qtgui` change here, as it did for `widgets.py`.
- **Order of work:** sections 2 to 5 (the views) first, then 7 and 8, then 9, 6, 10 and 12. 9's keys depend on 5 and 8, and 10's row joins the Map layout of 3.6.
- Work on the branch `claude/jolly-babbage-ppmbdj`. `main` is fast-forwarded only on your OK.
- README rows, docstrings and comments follow every change. The Physics / Machinery tags stay. Chen is the only citation.
- Work starts only after your answers and an explicit "begin".

## 2. Names (decisions document: "Terminology", "Names in the code and on the plots")

**Your rules:**
- The global topo is the topography at the topo bias and the topo tip height.
- The LDOS topo is the topography at the LDOS bias V_b and the LDOS tip height. Only the LDOS topo uses the LDOS tip height.
- Retired names: "Line Topo", "Map Topo", "Path Topo" and a "Single Point topo". The stripe is the Line / Path tip heights, not a topo.
- The names and titles in the code and on the plots are "global topo" and "LDOS topo".
- The stripe's title is "Topo", and only that.
- The topo colour bar keeps "topo". The other panels keep their names.
- "Map Topo" and "Line Topo" in comments, docstrings and prints become "LDOS topo". So do the prints "Map Local Setpoint LDOS" and "Path Setpoint LDOS".
- Unchanged: the cache names `global_topo_*` and `ldos_topo_*`, and the view names.
- Each topo gets its own variable and panel under the new names, listed for approval.

**Now:**
- Map's LDOS topo panel is titled "Map Topo: bias … V, height … Å" (`simulator.py` `_update_all`).
- Line's LDOS-h panel is titled "Topo" and shows the global topo. The stripe has no title.
- `current_z_map` holds the global topo or Map's LDOS topo.
- `ax_map` is the global topo in Single Point, Line and Path, and the Map Topo in Map, beside `ax_map_global`.

**The identifiers, for approval:**

| Now | New | Holds |
|---|---|---|
| `global_z_map` | `global_z_map` (kept) | the global topo, only |
| `current_z_map` | removed: split into `global_z_map` and `ldos_z_map` | — |
| (none) | `ldos_z_map`, with its key `ldos_z_key` = (V_b, LDOS h, decay model, broadening) | the LDOS topo of the whole unit cell, shared by every view |
| `map_z`, `map_bias`, `map_ldos_h` | kept | Map's copy of the LDOS topo its maps were computed at |
| `ax_map` (Single Point, Line, Path), `ax_map_global` (Map) | `ax_global_topo` (every view) | the global topo panel |
| `ax_map` (Map), `ax_line_topo` (Line) | `ax_ldos_topo` (every view) | the LDOS topo panel |
| `cax_topo` (Map only) | `cax_topo` (every view) | the topo colour bar |
| `ax_stripe` | kept, now titled "Topo" | the stripe |

**Titles:**
- The global topo keeps "Global Topo: bias … V, height … Å".
- The LDOS topo becomes "LDOS Topo: bias … V, height … Å".

**Prints:**
- "[*] Map Local Setpoint LDOS: …" and "[*] Path Setpoint LDOS: …" become one print, "[*] LDOS Topo Setpoint LDOS: …", for both the LDOS topo and the stripes.
- "Map Topo" and "Line Topo" in comments and docstrings become "LDOS topo"; the README follows.
- Outside the simulator, `sim_wrapper` reads `global_z_map`, which keeps its name. Nothing outside reads the other renamed names.

## 3. Layout (decisions document: "The principle", "The cursor readout", "The topo sub-section", "Line", "Path", "Map", "Single Point", "The LDOS h box", "The placement-panel minimum")

### 3.1 The principle

**Your rules:**
- Each view is made of panels grouped into sub-sections. The sub-sections keep the same layout relative to each other whatever the settings; Single Point's tall-window rule is the one exception.
- Inside a sub-section, the panels arrange by each panel type's own rules.
- A row has one height, whatever it contains, in every row of every view. That is the plot box's height, so the axes line up; titles and axis labels sit outside it.

**Now:**
- `controls.Arrangement` scores up to six layouts per view and takes the best.
- Only Line is held to its two rows (`layouts=['row_top']`).
- `Figure.align_row` holds Line's cluster row and Map's top row to one height.

**Change:**
- Each view allows only the layout its rules give:
  - Line and Path: `row_top`;
  - Map: `a_top`, the top row over the energy maps;
  - Single Point: `h`, and `v` only by its tall-window rule (3.7).
- `Figure.align_row` holds every row of every view to one plot-box height.

### 3.2 The cursor readout

**Your rules:**
- While the mouse is in any panel, the window's top right shows the mouse's x and y in that panel's axes coordinates, as matplotlib's toolbar did: "(x, y) = (…, …)".
- One set of coordinates only, for the panel under the mouse.
- When the mouse is on no panel, the readout goes away. No artist value.

**Now** (corrected: an earlier draft said there was none):
- Every `qtgui` window has a toolbar across its top: Home, Pan, Zoom, Save, and at its right end a coordinates label (`qtplot.py` `NavigationToolbar.coords`, `show_coords`). It has been there since before this session.
- On mouse motion the label shows "x=…  y=…" for the panel under the mouse, in that panel's data coordinates (turned back for a rotated panel). Off every panel it is blank.
- It is also blank over a colour bar.

**Your later rule (2026-11-08):**
- The coordinates readout ("x=…  y=…" for the panel under the mouse, blank when the mouse is on no panel) is the only part of that toolbar you asked for.
- Home, Pan and Zoom came with the sm4_Browser layer (commit `76e35cc`, 2026-11-01) and don't belong in the simulator.
- Save stays where it is. Nothing moves.

**Change:**
- In the simulator's windows, the toolbar drops Home, Pan and Zoom. Save and the coordinates readout stay exactly where they are, and the readout's text stays "x=…  y=…".
- Since Pan and Zoom are gone, the decomp clicks' Pan / Zoom exception goes with them (12.3).
- Everything else of your rule is already true. Whether the readout should also show over colour bars is in section 11.

### 3.3 The topo sub-section (every view)

**Your rules:**
- From the left: the topo colour bar, the global topo, the LDOS topo, in every view.
- The topo colour bar:
  - is its own panel, always far left;
  - overlaps no text, axis label, tick mark or tick value; the global topo's y label and ticks sit clear of it;
  - keeps its look: the ramp, no ticks, no numbers, "topo" above it;
  - carries the level line (the contour slice) only in Map.
- The global topo carries the crop adjusters and the auto rotate (already true: `_placement`, `_choose_orientation`).
- The LDOS topo:
  - Map and Single Point: takes the global topo's crop and auto rotate (Map already does: `_apply_crop`, `_unit_cell_axes`).
  - Line and Path: sized for what it shows. Line's is a 2:3 box centred on the line; Path's is described in 3.5. It takes neither the crop nor the auto rotate, and Path's turns 90° by its own rule.

**Now:**
- The colour bar exists only in Map, to the right of the Map Topo.
- Single Point and Path have no LDOS topo panel. Line's panel shows the global topo.

**Change:**
- Every view builds `cax_topo | ax_global_topo | ax_ldos_topo` as one `PanelRow`, with the colour bar's cell left of the global topo's y labels.
- `ax_ldos_topo` draws `ldos_z_map` (section 5), with the atoms, unit cell and markers as each view draws them today.
- Single Point's LDOS topo takes the crop and the rotation as Map's does.

### 3.4 Line

**Your rules:**
- **Top row:** the topo sub-section, then the stripe and the heatmaps with their colour bars, at one height, so the LDOS topo's, stripe's and heatmaps' distance axes line up.
- **Second row:** the Tip Height plot and the LDOS 1D plots, at that row's one height.
- **Markers:** drawn in their locations and colours in every panel.
- **Sizing:** the xy-spatial panels grow with the window, keeping their ratio, up to the largest size the row's width allows. When the window is taller than that, the 1D plots expand vertically and take all the remaining height. No empty band.

**Now:**
- The top row is the global topo, then the cluster: the LDOS-h panel, the stripe and the heatmaps.
- The second row is Tip Height and the spectrum with its legend (`FlipBox` with `tall=0`).

**Change:**
- The top row becomes the topo sub-section plus the cluster: the stripe, the heatmaps and their colour bars. Line's `ax_ldos_topo` moves from the cluster into the topo sub-section, keeping its 2:3 corridor drawing (`_fill_heat_row`), now from `ldos_z_map`.
- The markers go on the new LDOS topo panel too.
- The sizing rule is checked against `Arrangement` (`row_top`, fill panels) and completed where it falls short.

### 3.5 Path

**Your rules:**
- The same two rows as Line, with Path's LDOS topo panel in the topo sub-section.
- **The rectangle:** the smallest rectangle at any angle, turned to follow the path, that holds every sample point. A margin of 1/8 of its width is added left and right, and 1/8 of its height top and bottom.
- **The panel's shape:** always the row's height and never wider than square. When the rectangle is wider than tall, the panel turns 90°, so it is always square or taller than wide.

**Now:**
- There is no Path LDOS topo panel.
- The LDOS h box sits above the stripe.
- The 1D plots stack or sit side by side by the room (`FlipBox` defaults).

**Change:**
- A new `ax_ldos_topo` for Path: the LDOS topo tiled, rotated into the rectangle's frame and cropped to it, with the path, its nodes and the markers, built like Line's corridor.
- The rectangle is the minimum-area rectangle of the sample points, found over the convex hull's edge directions; 1/8 margins.
- It turns 90° when wider than tall.
- The LDOS h box moves above it.
- `layouts=['row_top']`, and the second row is held to one row (`FlipBox` with `tall=0`, as in Line).

### 3.6 Map

**Your rules:**
1. **The topo sub-section.**
2. **The spectrum:** in the same row, the same shape and size as one topo panel. Its plot box equals a topo panel's.
3. **The energy maps:** below the top row, in as many rows and columns as make each map largest in a full 16:9 window, chosen again on every resize.
- **Size limits:**
  - a map ≤ a topo panel ≤ 2 × a map, plus the placement-panel minimum (3.9); together these set a minimum window size;
  - when the window can't fit them all, the placement-panel minimum holds, and the 2× cap gives way.
- **Energy markers:** re-sorted into increasing energy on release, in place. Already true: `_on_rel` re-orders Map's columns.

**Now:**
- The top row is the global topo, the Map Topo and the colour bar.
- The spectrum is a cell of the map block, at a map's size.
- `group_cap` caps the topos at 2 × a map. There is no "map ≤ topo", no placement minimum and no minimum window size.

**Change:**
- The top row becomes the topo sub-section, then the spectrum (plot box held equal to a topo panel's), with its legend panel at its right (part 3), over the map block (`a_top`).
- The map block no longer holds the spectrum (no `extra` cell).
- Its columns are the ones that make each map largest in the window at its current size, chosen again on every resize. That is how "in a full 16:9 window" is read here. Kept if `MapBlock` already does this; added if not.
- `Arrangement` gains:
  - the lower limit "a map ≤ a topo panel";
  - the placement-panel minimum as a floor on the LDOS topo panel (Map's placement panel, 3.9), with the 2× cap giving way when both can't hold;
  - the window's minimum size from these limits.

### 3.7 Single Point

**Your rules:**
- **Layout:** Map's top row, the topo sub-section and then the spectrum at one topo panel's size. This replaces today's choice by the window's shape.
- **Tall window:** in a window taller than that one row, the topo sub-section fills the top row and the spectrum the bottom. It switches when the one row would leave an empty band below it, and back when it no longer would.
- **Its spectra:** take their tip heights from the LDOS topo at the current V_b, as Map does.
- **The LDOS topo is shared:** Single Point and Map share it through the `ldos_topo_*` cache. Switching views at the same settings reuses it from memory or the cache.

**Now:**
- The global topo and the spectrum, laid out `h` or `v` by score (`free_scored`).
- The LDOS h box is hidden.
- The spectra take heights from `current_z_map`: the global topo, or Map's last LDOS topo.

**Change:**
- The topo sub-section and the spectrum (with its legend) in one row (`h`), switching to `v` exactly when `h` would leave an empty band below, and back.
- The LDOS h box is shown above `ax_ldos_topo`.
- The spectra's heights are `ldos_z_map` at the nearest grid point (as Map's markers do today), at the current key (section 5).

### 3.8 The LDOS h box

**Your rules:** it sits above every LDOS topo panel, in Map, Single Point, Line and Path, with the editable LDOS h value.

**Now:** it is above Map's LDOS topo, above Line's global-topo panel and above Path's stripe, and hidden in Single Point.

**Change:** it goes above `ax_ldos_topo` in every view.

### 3.9 The placement-panel minimum (atom coverage)

**Your rules:**
- The panel where the markers and the line or path ends are placed must be large enough to see clearly where they go.
- The plotted atom dots may cover at most 1/3 of that panel, whether the Atoms toggle is on or off.
- Overlaps that come only from the cell's periodicity don't count.
- The covered area is counted as seen on screen, with overlapping dots counted once.
- It applies to the placement panel in Single Point, Line and Path (the global topo), and in Map to the LDOS topo panel.

**Now:** none.

**Change:**
- `_placement_min()` in the simulator: the smallest plot-box size, at the panel's shape, at which the union of the atom dots as drawn covers at most 1/3 of the plot box.
- The dots are counted at their drawn size (scatter `s=10` points², at the screen's DPI) over the images inside the panel's current limits. The union is counted on a pixel mask, so coincident images count once.
- The size is found by bisection on the box width and given to `Arrangement` as that panel's floor.

## 4. The bias toggles (decisions document: "The bias toggles (every view)")

**Your rules:**
- A toggle beside E Min and beside E Max picks which end is the bias. Exactly one is on, and one is always on. `ldos_bias_sign` only sets the starting state.
- **An end set to 0:** the bias moves to the other end automatically, and that end's toggle refuses to turn on while it is 0. When the end becomes nonzero again, the bias stays on the other end until you switch it. Both ends at 0 stays an error.
- **Switching the toggle:** triggers the same recomputes, with the same caches, as changing that end's value.

**Now:** `_bias_sample()` picks the end itself: E Min = 0 gives E Max, E Max = 0 gives E Min, otherwise `ldos_bias_sign` decides. There are no toggles.

**Change:**
- **State:** `self.bias_end` ∈ {`'emin'`, `'emax'`}, starting at `'emin'` for `ldos_bias_sign` 'neg' and `'emax'` for 'pos'.
- **Controls:** one toggle beside each of the E Min and E Max boxes in the bottom strip (`chk_bias_emin`, `chk_bias_emax`, each labelled "bias"), in every view, mutually exclusive as 'exact' / 'linearized' are.
- **A box edited to 0.0 while it is the bias end:** `bias_end` moves to the other end and the toggles follow. A click on the toggle of an end at 0.0 leaves it off.
- **Computing V_b:** `_bias_sample()` takes the end from `bias_end`. Its snapping, V_b and the both-zero `ValueError` stay.
- **Recomputes:** a toggle switch runs `_on_ui_change` as a change of that end's value does.
- **Info panel and docs:** both toggles appear in the info panel's settings. `run_DFT_STM.py`'s `ldos_bias_sign` comment, the README and the `_bias_sample` docstring follow. `sim_wrapper` and `run_stm_decomp.py` keep passing `ldos_bias_sign` as the start state.

## 5. The topos (decisions document: "The topos", and Single Point's spectra)

**Your rules:**
- **The global topo and the LDOS topo:** only computed on one unit cell's grid and tiled for display (already true). The grid's duplicate edge is kept (already true).
- **The LDOS topo panels of Line and Path:** show the LDOS topo, from memory or the `ldos_topo_*` cache when it is already computed at that V_b. Otherwise they compute it and save it under that bias's cache name.
- **The stripes keep their own quick line convergence, with no cache of their own.** Your later rule (9.2): their heights are saved inside the Line / Path LDOS cache, with the spectra computed at them. The rules for the convergence:
  - **Starting heights:** the LDOS topo interpolated at the path's sample points, bilinear and periodic in the cell's fractional coordinates. The duplicate edge row and column are left out of the interpolation only.
  - **The setpoint:** at the location of the LDOS topo's tallest height in the unit cell, even when the line or path doesn't go there.
    - **The value:** the integrated LDOS at V_b at that point's xy and its LDOS topo height, by the same call the LDOS topo makes.
    - **Ties at the tallest height:** the one with the largest integrated LDOS, which is the point the LDOS topo's own setpoint came from.
    - **From the cache:** only the heights are kept, so the value is computed again at those points.
  - **The other points:** move up or down from their starting height by the existing rule. A point within 1% of the setpoint isn't moved again.
- **Single Point's spectra** use the LDOS topo at the current V_b, shared with Map (3.7).

**Now:**
- Only Map converges an LDOS topo.
- Line and Path converge their 72 points from a flat start at z_top + LDOS h, with the setpoint the largest current along the line itself.
- Single Point never converges one.

**Change:**
- **`_ensure_ldos_topo()`:** returns `ldos_z_map` at the current key `ldos_z_key`: from memory, from `caches/ldos_topo_*.npy` (with `reuse_cache`), or converged and saved, as Map's code does now.
  - Callers: Map (before its maps), Single Point (before its spectra), Line and Path (before their stripes, and for their `ax_ldos_topo`).
  - A bias or LDOS h change in Line or Path therefore converges the whole-cell LDOS topo too, unless it is in memory or cached.
- **The stripes (Line and Path):**
  - start from the bilinear periodic interpolation of `ldos_z_map` (the grid without its duplicate last row and column) at the sample points;
  - take their setpoint from `_topo_current` at the tallest grid point of `ldos_z_map`, ties to the largest value;
  - then `_converge_tip_height` as now.
  - On a Line / Path LDOS cache hit (9.2), the stripe's heights come from the cache and the stripe is not converged again.
  - G1 (b) stays: Line and Path keep their own heights under their own keys.
- **What this also fixes:**
  - Single Point's spectra no longer depend on which view ran last.
  - The LDOS topo is never drawn from the global heights under an LDOS title.
  - Line's LDOS-h panel stops showing the global topo.

## 6. Console lines (chat, 2026-11-07)

### 6.1 The LDOS line: unchanged

**Your decision:** the current line that updates itself stays as it is: one in-place line per channel, "x/n spectra calculated", rewritten after each launch.

**The one consequence of section 8:** the line's channel label names the channels now computed. It reads "total: " always (none for an unpolarized run, as now), and "|m|: " (NCL) or "m: " (POL) for the second pass while Mag is on. Today POL's labels are "spin up: " and "spin down: ".

### 6.2 What the topography lines say

**Your complaint:** the console isn't clear about which topo is being converged.

**Now:**
- The header "--- INITIALIZING TIP CONVERGENCE ENGINE ---" is the same for every topo.
- The global recompute prints "with broadening" whatever the cause.
- Line prints "Path Setpoint LDOS".

**Change:**
- The header names what is converged: "--- TIP CONVERGENCE: global topo | LDOS topo | Line stripe | Path stripe, bias … V, height … Å, n points ---".
- "with broadening" is dropped.
- The setpoint prints follow section 2.
- The Iter lines and the footer stay as they are.

## 7. The LDOS speed-up (chat, 2026-11-07)

**Why your July copy was faster** (read from the zip you sent, not run):
- Its Decay-on LDOS uses one weight per energy sample: Chen's factor with the bias set to V = |E|. That is one matrix product per sample.
- The exact model (part 2) integrates, at each requested V, over every sample from 0 V to V. That is far more work per V.
- The zip has no custom kernels, streams or batching. Its topography and its heatmap LDOS are separate calls.

**Your rule:** speed it up without losing any accuracy or information. If that can't be done, use the caches of section 9.

### 7.1 The exact kernel: one thread per (tip point, requested V, atom)

**Now:**
- `exact_sweep` runs one thread per (tip point, atom) and loops over the atom's images and, inside that, over every requested V.
- With heavy loads a launch holds one tip point. `pts_per_launch = max(1, 2.5e7 // (n_pairs × atoms))`: with a 0.005 eV DOSCAR step, a −3 to 0.5 V window and more than about 130 atoms, that is one point. Only one thread in each 64-thread block works.

**Change:**
- The thread index runs over (tip point, requested V) pairs. Each thread does, for its one V, exactly what today's thread does for that V: the same images in the same order, the same stencil, pieces and pairs.
- It writes the same output slot (no atomics). Each output gets the same arithmetic in the same order, so the results should be identical.
- The launch chunking and the Python side (`_sweep`) are unchanged, apart from the grid size.
- **The check:** a short script, kept outside the repository, compares the old and new kernels' outputs on your data and times both. You run it once.

### 7.2 The linearized kernel

- Its single walk outward from 0 V is what makes it fast, so it can't be split by V. The two sides of 0 V are independent and could run as two threads.
- It is already one walk per (tip point, atom, image).
- Left as it is. The side split would come back to you only if 7.1's check shows the speed-up isn't enough.

**Unknown until you run it:** the size of the speed-up on your GPU. I can't run the GPU code here.

## 8. |m| and m only on demand (chat, 2026-11-07)

**Your rules:**
- |m| (NCL) and m (POL) are treated the same.
- Neither is computed until the Mag toggle is pressed.
- Once computed, it isn't computed again.
- It is saved as a cache when computed.

**Now:**
- NCL computes the |m| channel in every LDOS call.
- POL computes spin up and spin down in two passes and adds them for the total.
- The topographies already use the total only (`topo_only`).

**Change:**
- **The total by default:** `_calculate_ldos_at_points_gpu` computes the total channel in one pass. NCL uses `dos_up_*` (already the total), POL uses `dos_total_*` (up + down, already built for the topography), unpolarized its one channel.
- **The magnetization channel** (new argument `want_mag`) is computed only while Mag is on, for what is shown: the Line / Path LDOS, Map's maps, the marker spectra, and the dragged map.
  - NCL: |m| from `dos_dn_*`, as now.
  - POL: m from one pass on `dos_up − dos_dn`. Like the total's one pass on up + down, it differs from today's two passes only by float32 rounding.
- **Kept:** the shown data stays in memory as now. The Line, Path and Map mag channel is also saved in `caches/` with its LDOS (9.2), so turning Mag off and on again, or returning to the same settings, loads it instead of computing it.
- **The markers' spectra** (Single Point's and Map's markers) are not cached (9.2). While Mag is on, their mag channel is computed with them on a marker's release or creation, and when Mag is turned on.
- **The current (Feenstra)** comes from the total's pass, as now.
- **Renames:**
  - `cached_ld_up` / `cached_ld_dn` become `cached_ld_total` / `cached_ld_mag`, and `map_ld_up` / `map_ld_dn` become `map_ld_total` / `map_ld_mag` (with their GPU copies).
  - `_combine_ldos_display(total, mag, show_mag)` returns the total, or the mag channel while Mag is on.
- **`sim_wrapper`** (`mo_ldos_decomp`) reads `cached_ld_dn` and calls `_combine_ldos_display`. Those lines follow the renames, with your approval for this one file outside `stm_from_dft_opt`.

## 9. Caches (chat, 2026-11-07)

### 9.1 The caches folder

**Your rule:** every cache the simulator makes in the VASP folder is written to and looked for in a folder named `caches` there. Existing caches are not moved; you may move them yourself.

**Change:**
- One helper, `_cache_path(name)` = `<VASP folder>/caches/<name>`, with the folder made when first needed. The names stay as they are.
- It covers:
  - `global_topo_*.npy` and `ldos_topo_*.npy` (`simulator.py`);
  - the new LDOS and |m| / m caches (9.2);
  - `LOCPOT.npy` (`locpot_manager.py`).
- `reuse_cache` keeps its meaning.
- The cache names stay as they are, including how a height is spelled. A height given as 2 and later as 2.0 still makes two names, as now.

### 9.2 The Line, Path and Map LDOS caches, per decay model

**Your rules:**
- The Line and Path LDOS are cached to disk, so reruns have them to pull from and never need to calculate them again, and they open at once when the decay method is changed.
- Map's spectral heat maps at the energy slices are cached the same way.
- The individual point markers' spectra are not cached. They are calculated on release of a move, or on creation.
- If the spectra are saved in a cache, the tip heights are saved with them.
- The |m| / m channel is saved as a cache when it is calculated.

**Now:** the shown LDOS is kept in memory only, for the current state. A decay-model change, or a return to earlier settings, calculates it again.

**Change:**
- **What a cache file holds:**
  - Line / Path: the LDOS of the run (the total, and the mag channel when it is calculated), its energies, its current, and the stripe's tip heights it was calculated at.
  - Map: the energy maps (the total, and the mag channel when it is calculated), their current, their energies, and the LDOS topo heights they were calculated at (`map_z`).
- **The key:**
  - Line / Path: the view and its geometry (the line ends, or the path nodes and Extend, and the point count); V_b and LDOS h; the window's snapped ends; the decay model, the broadening, the unit cells, the precision and the grid size.
  - Map: V_b and LDOS h; the map energies' samples; the window's snapped ends; the decay model, the broadening, the unit cells, the precision and the grid size.
- **Where:**
  - on disk only, in `caches/`, as `line_ldos_<hash>.npz`, `path_ldos_<hash>.npz` and `map_ldos_<hash>.npz`, with a `_mag` file beside each when the mag channel is calculated;
  - `<hash>` is the SHA-1 of the key's text; the key is stored in the file and checked on load;
  - with `reuse_cache` on, a matching file is loaded and nothing is calculated (no stripe convergence, no LDOS); otherwise it is calculated and saved;
  - each decay model has its own files, so a switch loads that model's file when it exists;
  - in memory, only the shown data, as now.
- **Not cached:** the markers' spectra in Single Point and Map, calculated on a marker's release or creation as now. The Line and Path markers' spectra are rows of the cached Line / Path LDOS.
- **The dragged map** (one map energy moved by a drag in Map) is calculated as now. Since it changes the map energies' samples, the maps are then saved under the file of the new key.
- **Size:** points × energies × atoms × orbitals × 4 bytes. For example, 72 × 350 × 300 × 9 × 4 B ≈ 270 MB per Line run; a Map file is grid points × maps × atoms × orbitals × 4 bytes.

## 10. The shading (chat, 2026-11-07)

**Your requests:**
- The shading is far too dark at its darkest point, in both Height and Relief. The methods need altering.
- A way to adjust the shading's darkness, in the shading sub-section.
- A way to drag the light around:
  - **A top-down panel** the size of a map panel: a centre point, and a light that moves around it in x and y.
  - **A side-view panel:** the camera faces the centre point of the plane, seen from the side. There the light moves only up or down. It keeps its focus on the centre point, so up or down changes the light's angle.
- **A shading sub-section in the Map window:** the last row before the controls, in line with the RUN / STOP button and the toggles box (Decay, Atoms, Norm, …).

**Now:**
- **The darkness:**
  - `_draw_shaded_slice` calls `cmap_shading.shaded_rgba` with its default floor of 0, the addendum's reading of Kovesi's `minval` default.
  - `lstar_stretch` scales each point's L* by a factor that runs from the floor, at the lowest shade value over the maps, to 1 at the highest.
  - With floor 0 the lowest value is L* 0, black. It is the same in Height (the lowest height) and in Relief (the darkest slope).
- **The light:** fixed at Kovesi's defaults, azimuth 45° and elevation 45° (`shade_field` is called without az / el). Only Relief uses it; Height shades by the heights alone.
- **The Contour / Shading box:** in the bottom strip, after 'cmap norm'.
- **The window:** the RUN / STOP button and the toggles are one box at the bottom of the left column. The panel area is to its right, and the bottom strip runs under both.

**Change:**
- **Darkness, both modes:** a number box 'darkness' (0 to 1) in the shading sub-section, adjustable in the window. The floor passed to `shaded_rgba` is 1 − darkness: darkness 1 is today's (the darkest point black), 0 is no shading. It starts at 0.5, so the darkest point starts at half its colour's L*.
- **The light (Relief):**
  - `BoxState` gains `az` (degrees; 0 up, increasing clockwise, Kovesi's convention) and `el` (degrees, 0 to 90), both starting at 45°.
  - `shade_field` receives them, and the shade cache's key includes them.
- **The top view panel** (`ax_light_top`, a map panel's size):
  - Just the control: the plane from above, its centre point, a guide circle of radius R and the light as a dot. No preview behind it.
  - Dragging the dot sets the azimuth only: the dot's angle around the centre. The side view alone sets the elevation.
  - The dot is drawn at R cos(el) from the centre, so it shows the light as seen from above.
- **The side view panel** (`ax_light_side`, the same size):
  - Just the control, with no preview. The plane is seen edge-on, as a horizontal line through the centre point. The view is at right angles to the light's azimuth, so the light lies in the drawn plane.
  - The light is a dot on the arc of radius R around the centre point, at the elevation angle above the line, with a line from it to the centre point (its focus).
  - Dragging it up or down sets the elevation, from 0° to 90°. The dot stays on the arc.
- **Redraw:** the energy maps are shaded again live, while the light is dragged.
- **Height mode:** the two light panels are shown greyed and take no drags, since Height uses no light.
- **The shading sub-section, Map only:**
  - One row at the bottom of Map's panel area, under the energy maps and above the bottom strip.
  - **Its bottom edge lines up with the bottom edge of the left column's RUN / STOP and toggles box** (your rule). The two need not be the same height: only their bottoms line up.
  - It is part of Map's arrangement, so its panels can be a map panel's size, but its slot is pinned to the bottom of the panel area. Any height the window leaves over goes above it, between the energy maps and it, never below it.
  - From the left: the Contour / Shading box (moved there from the strip), the darkness box, the top view panel, the side view panel.
  - It is Map's last row, under the energy maps however many rows they wrap to. Its panels are a map panel's size, and it joins 3.6's size limits and the window's minimum size as Map's fourth sub-section.
- **Where the code goes:** the light's state, the two panels' drawing and their drag rules go in `cmap_shading.py` with the box, so sm4_Browser can take them as it is meant to take the module. The simulator places them.
- **The info panel:** lists the darkness, the azimuth and the elevation with the other settings.

## 11. Open question

1. **3.2, the readout over colour bars:** today the toolbar's coordinates go blank over a colour bar, but your rule says "any panel". Show them over colour bars too? **R: yes**, since a colour bar is a panel of the window.

## 12. Gaps from your other documents

**How these were found:**
- A text-only check of every brief and workflow prompt against your other seven documents covered 616 requirements in all.
- Each flag was then checked against your later rulings, including your mid-turn messages, and against the code.
- Ruled out:
  - the Rejali citation and the README's `wrong_decays/` row: your own ruling of 05:08Z;
  - the matcher's `qtgui` copy, the commits and the `qtgui/__init__.py` docstring: done by hand outside the briefs;
  - Map's legend panel and "one click, one step" with RUN off: done in the follow-up;
  - the shading on the GPU, the 720-dpi SVG and the README's DOSCAR tolerance: in the code.

**What remains:**

12.1 **The DOS plotter's window lines** (response document, part 3 Q6, for every `qtgui` window).
- **Your rule:** "every setting the window makes with the mouse, each as one of the window's own lines (`set_info_line`), updated whenever it changes."
- **Now:** `plot_dos_cursors_levels.py` has no window lines. Its plot level, active element and active atom reach the info panel nowhere.
- **Change:** one `set_info_line` each for the plot level, the active element and the active atom, updated when they change.

12.2 **A right click on the spectrum panel** (part 3 handoff section 2, "The clicks").
- **Your rule:** "Right click anywhere in the spectrum panel or the legend panel (on a curve, on an entry, or on empty space): goes back one level."
- **Now:** only a right click inside the spectrum's plot box goes back. One on its tick labels, axis labels or title does nothing.
- **Change:** a right click anywhere in the spectrum's panel cell goes back one level.

12.3 **The decomp clicks while Pan or Zoom is on.**
- **Now:** the code ignores the spectrum's and the legend panel's decomp clicks while the toolbar's Pan or Zoom is on. Your rules have no such exception. It came from the part 3 brief, not from you.
- **Change:** removed, with Pan and Zoom themselves (3.2). The decomp clicks always act as your rules say.

12.4 **The legend panel's wheel** (response document, part 3 Q1).
- **Your rule:** "a Qt vertical scroll bar at the panel's right edge, shown only when the entries don't fit, plus the mouse wheel over the panel."
- **Now:** the wheel scrolls only over the entry stack, not over the panel's title or its Back row.
- **Change:** the wheel scrolls anywhere over the legend panel.

12.5 **One window line per mouse-made setting** (response document, part 3 Q6).
- **Your rule:** "each as one of the window's own lines".
- **Now:** the simulator groups several settings into one line. The decomp level, the active marker and the active element share the `decomp` line, and every marker position shares the `markers` line. Every value is shown; the form differs.
- **Change:** one line per setting: the decomp level, the active marker, the active element, and each marker's position (with the Line ends and the Path nodes as now).

12.6 **The DOSCAR guard's tolerance** (part 2 answers, A1).
- **Your rule:** "half a unit of the last decimal the printed energy column carries, read from the file's own text, plus 1e-6 eV."
- **Now:** the tolerance is computed per printed energy, from that entry's own decimals. It is the same for VASP's fixed 3-decimal column, and would differ only if entries in one column had different numbers of decimals.
- **Change:** one tolerance for the column, from the decimals the column carries.

## 13. Files

- `stm_from_dft/simulator.py` (sections 2 to 10).
- `stm_from_dft/cmap_shading.py` (section 10).
- `stm_from_dft/qtgui/qtplot.py` and `controls.py`, with `matcher/qtgui` and the root `qtgui/` following (3.1, 3.2, 3.6, 3.9, and 4 if the toggles need a control).
- `stm_from_dft/locpot_manager.py` (9.1).
- `stm_from_dft_opt/README.md` and `run_DFT_STM.py` (comments and docs).
- `mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py` (8: the renamed channels only).
- `stm_from_dft/plot_dos_cursors_levels.py` (12.1), `stm_from_dft/doscar_parser.py` (12.6), and the simulator's decomp clicks, legend panel and window lines (12.2 to 12.5).
