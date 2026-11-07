# Plan: the views-layout decisions, the bias toggles, the topos, the LDOS speed-up, |m| and m on demand, the caches folder, the console lines and the shading

Draft of 2026-11-07, for Benjamin's approval. Nothing in it is implemented. The open questions are in section 11. Each one names the sections it decides. Section 12 is filled from the text-only check of the briefs against your other documents, which is still running.

## 0. Sources

- **Sections 2 to 5:** your simulator-views layout decisions document (`PART3_simulator_views_layout_decisions.md`, uploaded 2026-11-07 06:29Z). None of it was implemented. Its rules are quoted or closely transcribed here, each followed by what the code does now and the change.
- **Sections 6 to 10:** your chat after the close-out (2026-11-07, from about 22:15Z): the per-spectrum console line, the topo console labels, the speed-up, |m| and m, the caches, the shading.
- **Where they differ:** the decisions document wins over this plan unless you answer a question in section 11 otherwise.
- **One statement in the decisions document doesn't match the code.** It says Path's LDOS topo panel is "already sized for what it shows (… Path's turned rectangle with 1/8 margins)". Neither your copy from before part 1 nor the current code has a Path LDOS topo panel (`ax_line_topo` is `None` outside Line). It is new here, built by the Path rules in 3.5.

## 1. Rules for the job (standing)

- Implement exactly what is approved. If anything needs changing outside it, stop and ask. Every delivery lists the approved changes and any additional ones separately, and stops if the additional list is not empty.
- Nothing in `wrong_decays/` is touched.
- Agents read files and run only `python -m py_compile`: no tests, no windows. Every agent is Opus 5.5 or higher.
- The two package `qtgui` copies (`stm_from_dft/qtgui` and `matcher/qtgui`) stay byte-identical. Whether the root `qtgui/` follows is Q20.
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
- The LDOS topo becomes "LDOS Topo: bias … V, height … Å" (Q13).

**Prints:**
- "[*] Map Local Setpoint LDOS: …" and "[*] Path Setpoint LDOS: …" become "[*] LDOS Topo Setpoint LDOS: …" (Q13).
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

**Now:** none.

**Change:**
- A right-aligned label at the window's top right in `qtgui` (`qtplot.py`, both package copies).
- It is set on mouse motion from the axes under the mouse, in that axes' data coordinates, and cleared when the mouse is over no axes.
- It is on in every simulator view. Q14: also the other `qtgui` windows?

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
- Its columns are chosen again on every resize to make each map largest. Kept if `MapBlock` already does this; added if not.
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
- The dots are drawn at their screen size over the images inside the panel's current limits, and the union is counted on a pixel mask, so coincident images count once.
- The size is found by bisection on the box width and given to `Arrangement` as that panel's floor.

## 4. The bias toggles (decisions document: "The bias toggles (every view)")

**Your rules:**
- A toggle beside E Min and beside E Max picks which end is the bias. Exactly one is on, and one is always on. `ldos_bias_sign` only sets the starting state.
- **An end set to 0:** the bias moves to the other end automatically, and that end's toggle refuses to turn on while it is 0. When the end becomes nonzero again, the bias stays on the other end until you switch it. Both ends at 0 stays an error.
- **Switching the toggle:** triggers the same recomputes, with the same caches, as changing that end's value.

**Now:** `_bias_sample()` picks the end itself: E Min = 0 gives E Max, E Max = 0 gives E Min, otherwise `ldos_bias_sign` decides. There are no toggles.

**Change:**
- **State:** `self.bias_end` ∈ {`'emin'`, `'emax'`}, starting at `'emin'` for `ldos_bias_sign` 'neg' and `'emax'` for 'pos'.
- **Controls:** one toggle beside each of the E Min and E Max boxes in the bottom strip (`chk_bias_emin`, `chk_bias_emax`; label Q12), in every view, mutually exclusive as 'exact' / 'linearized' are.
- **A box edited to 0.0 while it is the bias end:** `bias_end` moves to the other end and the toggles follow. A click on the toggle of an end at 0.0 leaves it off.
- **Computing V_b:** `_bias_sample()` takes the end from `bias_end`. Its snapping, V_b and the both-zero `ValueError` stay.
- **Recomputes:** a toggle switch runs `_on_ui_change` as a change of that end's value does.
- **Info panel and docs:** both toggles appear in the info panel's settings. `run_DFT_STM.py`'s `ldos_bias_sign` comment, the README and the `_bias_sample` docstring follow. `sim_wrapper` and `run_stm_decomp.py` keep passing `ldos_bias_sign` as the start state.

## 5. The topos (decisions document: "The topos", and Single Point's spectra)

**Your rules:**
- **The global topo and the LDOS topo:** only computed on one unit cell's grid and tiled for display (already true). The grid's duplicate edge is kept (already true).
- **The LDOS topo panels of Line and Path:** show the LDOS topo, from memory or the `ldos_topo_*` cache when it is already computed at that V_b. Otherwise they compute it and save it under that bias's cache name.
- **The stripes keep their own quick line convergence, with no cache:**
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
  - G1 (b) stays: Line and Path keep their own heights under their own keys.
- **What this also fixes:**
  - Single Point's spectra no longer depend on which view ran last.
  - The LDOS topo is never drawn from the global heights under an LDOS title.
  - Line's LDOS-h panel stops showing the global topo.

## 6. Console lines (chat, 2026-11-07)

### 6.1 One line per finished spectrum, with its time

**Your request:** "one line for each completed spectra also printing the total run time for that spectra as the tip height iteration print per iteration".

**Now:** one in-place line per channel, "x/n spectra calculated", rewritten after each launch.

**Change** (format and timing per Q1 to Q5):
```
--- LDOS: Map energy maps, 2500 spectra ---
   Spectrum 0001/2500 | Time = 0.012s
   …
--- LDOS COMPLETE: Total Time = 30.512s ---
```
- Each time runs from a GPU synchronize before its launch to the one after it (`time()`, as the Iter line uses).
- A launch that finishes several spectra gives each the launch's time divided by their number (Q1).
- Decay off has one product for all spectra (Q4).
- With |m| / m on demand (section 8), a run prints one set of lines for the total. While Mag is on, a second set follows for |m| or m, with the channel in its header.

### 6.2 What the topography lines say

**Your complaint:** the console isn't clear about which topo is being converged.

**Now:**
- The header "--- INITIALIZING TIP CONVERGENCE ENGINE ---" is the same for every topo.
- The global recompute prints "with broadening" whatever the cause.
- Line prints "Path Setpoint LDOS".

**Change** (wording in Q15):
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
- It writes the same output slot (no atomics). Each output gets the same arithmetic in the same order, so the results should be identical (Q8 for the check).
- The launch chunking and the Python side (`_sweep`) are unchanged, apart from the grid size.

### 7.2 The linearized kernel (Q7)

- Its single walk outward from 0 V is what makes it fast, so it can't be split by V. The two sides of 0 V are independent and could run as two threads.
- It is already one walk per (tip point, atom, image).
- Proposed: left as it is unless you want the side split.

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
  - POL: m from one pass on `dos_up − dos_dn` (Q9).
- **Kept:** it is kept in memory under its LDOS's key and saved in `caches/` (section 9). Turning Mag off and on again, or returning to the same settings, reuses it.
- **The current (Feenstra)** comes from the total's pass, as now.
- **Renames:**
  - `cached_ld_up` / `cached_ld_dn` become `cached_ld_total` / `cached_ld_mag`, and `map_ld_up` / `map_ld_dn` become `map_ld_total` / `map_ld_mag` (with their GPU copies).
  - `_combine_ldos_display(total, mag, show_mag)` returns the total, or the mag channel while Mag is on.
- **`sim_wrapper`** (`mo_ldos_decomp`) reads `cached_ld_dn` and calls `_combine_ldos_display`. It follows the renames (Q10).

## 9. Caches (chat, 2026-11-07)

### 9.1 The caches folder

**Your rule:** every cache the simulator makes in the VASP folder is written to and looked for in a folder named `caches` there. Existing caches are not moved; you may move them yourself.

**Change:**
- One helper, `_cache_path(name)` = `<VASP folder>/caches/<name>`, with the folder made when first needed. The names stay as they are.
- It covers:
  - `global_topo_*.npy` and `ldos_topo_*.npy` (`simulator.py`);
  - the new LDOS and |m| / m caches (9.2);
  - `LOCPOT.npy` (`locpot_manager.py`, Q11).
- `reuse_cache` keeps its meaning.

### 9.2 Line and Path LDOS caches, per decay model

**Your rule:** if the speed-up can't make it fast without losing accuracy, the Line and Path LDOS are cached so they open at once when the decay method is changed. Their |m| / m caches likewise.

**Change** (Q5, Q6):
- **What is kept:** the LDOS of a Line or Path run (its total, and its mag channel when computed) together with its energies and current.
- **The key:**
  - the view and its geometry: the line ends, or the path nodes and Extend, and the point count;
  - V_b and LDOS h;
  - the window's snapped ends;
  - the decay model, the broadening, the unit cells, the precision and the grid size.
- **Where:**
  - in memory, one entry per decay model, so a switch and a switch back cost nothing;
  - on disk in `caches/` as `line_ldos_<hash>.npz` / `path_ldos_<hash>.npz` (and `_mag`). `<hash>` is the SHA-1 of the key's text; the key is stored in the file and checked on load.
- **The stripe's heights:** a cached LDOS is only usable with the heights it was computed at, but your topo rules say the stripes have no cache. Q6 asks whether the heights are stored inside the LDOS cache.
- **Size:** points × energies × atoms × orbitals × 4 bytes. For example, 72 × 350 × 300 × 9 × 4 B ≈ 270 MB per Line run.
- **Map's energy maps:** Q5.

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
- **Darkness, both modes:** a number box 'darkness' (0 to 1) in the shading sub-section. The floor passed to `shaded_rgba` is 1 − darkness: darkness 1 is today's (the darkest point black), 0 is no shading. The start value is Q21.
- **The light (Relief):**
  - `BoxState` gains `az` (degrees; 0 up, increasing clockwise, Kovesi's convention) and `el` (degrees, 0 to 90), both starting at 45°.
  - `shade_field` receives them, and the shade cache's key includes them.
- **The top view panel** (`ax_light_top`, a map panel's size):
  - It shows the plane from above, its centre point, a guide circle of radius R and the light as a dot.
  - Dragging the dot sets the azimuth: the dot's angle around the centre.
  - The dot is drawn at R cos(el) from the centre, so it shows the light as seen from above (Q22).
- **The side view panel** (`ax_light_side`, the same size):
  - The plane is seen edge-on, as a horizontal line through the centre point. The view is at right angles to the light's azimuth, so the light lies in the drawn plane.
  - The light is a dot on the arc of radius R around the centre point, at the elevation angle above the line, with a line from it to the centre point (its focus).
  - Dragging it up or down sets the elevation, from 0° to 90°. The dot stays on the arc.
- **Redraw:** the energy maps are shaded again while the light is dragged (Q23).
- **Height mode:** the two light panels are shown greyed and take no drags, since Height uses no light (Q24).
- **The shading sub-section, Map only:**
  - One row at the bottom of Map's panel area, under the energy maps and above the bottom strip, level with the left column's RUN / toggles box.
  - From the left: the Contour / Shading box (moved there from the strip), the darkness box, the top view panel, the side view panel (Q25).
  - Its panels are a map panel's size. It joins 3.6's size limits and the window's minimum size as Map's fourth sub-section (Q28).
- **Where the code goes:** the light's state, the two panels' drawing and their drag rules go in `cmap_shading.py` with the box, so sm4_Browser can take them as it is meant to take the module. The simulator places them (Q26).
- **The info panel:** lists the darkness, the azimuth and the elevation with the other settings.

## 11. Open questions (each with my recommendation)

1. **6.1, the per-spectrum time.** (a) Keep the launches, and each spectrum gets its launch's time divided by its spectra; when a launch holds one spectrum, the time is its own. (b) One spectrum per launch, always its own time, but light runs slow down. (c) One line per launch. **R: (a).** With 7.1, heavy runs already have one spectrum per launch.
2. **6.1, header and footer:** one pair per LDOS call, as above? **R: yes.**
3. **6.1, the caller label** ("Map energy maps", "Line points", "marker spectra", "dragged map") in the header? **R: yes.**
4. **6.1, Decay off:** n lines, each with the product's time divided by n, or one line? **R: n lines, as Decay on.**
5. **9.2:** cache the Line and Path LDOS in memory, on disk, or both? Map's energy maps too? **R: both; Map's too, same key scheme.**
6. **9.2:** store the stripe's heights inside the Line / Path LDOS cache, so a cached run opens without converging? Or converge the stripe and load only the LDOS? **R: inside the LDOS cache.** They are then never kept on their own, only with the LDOS they belong to.
7. **7.2:** split the linearized kernel by side of 0 V? **R: no, not unless 7.1 alone isn't enough.**
8. **7.1, the check:** I write a short comparison of the old and new exact kernels on your data, kept outside the repository. You run it once to confirm identical outputs and see the speed-up. **R: yes.**
9. **8, POL:** the total in one pass on up + down, and m in one pass on up − down. These differ from today's two-pass sums only by float32 rounding. Accept? **R: yes.**
10. **8:** may `sim_wrapper.py` (outside `stm_from_dft_opt`) change, for the renamed channels? **R: yes,** only those lines.
11. **9.1:** move `LOCPOT.npy` into `caches/` too? **R: yes**, since you said all caches. The matcher's `.h5` results are outputs, not the simulator's caches, and stay where they are.
12. **4:** the bias toggles' label: "bias" beside each box? **R: "bias".**
13. **2:** the LDOS topo's title "LDOS Topo: bias … V, height … Å", and the setpoint print "[*] LDOS Topo Setpoint LDOS: …" for both the LDOS topo and the stripes? **R: yes.**
14. **3.2:** the cursor readout only in the simulator's windows, or in every `qtgui` window? **R: the simulator's,** as your rule says.
15. **6.2:** the convergence header's wording as written in 6.2? **R: yes.**
16. **Cache names, an observation:** a height passed as 2 is written "2A", but "2.0A" after a Rerun, so the same height can get two cache names. Fix it, which renames such files, or leave it? **R: leave it unless you want it.** It isn't part of your rules.
17. **3.6, the map grid "in a full 16:9 window":** read as "the columns that make each map largest in the window as it is, chosen again on every resize"? **R: yes.**
18. **3.9:** the dot size counted is the drawn size (`s=10`, points²) at the screen's DPI? **R: yes.**
19. **Order of work:** sections 2 to 5 (the views) first, then 7 and 8, then 9, 6 and 10? **R: yes,** since 9's keys depend on 5 and 8, and 10's row joins the Map layout of 3.6.
20. **The root `qtgui/`:** follows the package copies for 3.2 and 3.6? **R: yes,** as it did for `widgets.py`.
21. **10, the darkness's start value.** **R: 0.5,** so the darkest point is at half its colour's L*. Yours to set.
22. **10, the top view:** does its drag set only the azimuth, with the side view alone setting the elevation as you described? Or does the dot's distance from the centre set the elevation too? **R: azimuth only.**
23. **10:** shade the maps again live while the light is dragged, or on release? **R: live.** The shade is on the cell's grid, a few thousand points.
24. **10, Height mode:** the light panels greyed, with no drags? **R: yes.**
25. **10, the sub-section's order:** the Contour / Shading box, darkness, top view, side view, from the left? **R: yes.**
26. **10:** the light panels' code in `cmap_shading.py`, so sm4_Browser gets them with the module? **R: yes.**
27. **10, the top view's background:** only the centre, the guide circle and the light, or the shaded LDOS topo behind them as a preview? **R: only the centre, the circle and the light,** unless you want the preview.
28. **10 with 3.6:** the shading row is Map's last row, under the energy maps however many rows they wrap to? **R: yes.**

## 12. Gaps from your other documents

Filled from the text-only check of the briefs against your other seven documents (running). Anything a brief dropped or changed, and that no later ruling of yours removed, goes here as a numbered item for approval.

## 13. Files

- `stm_from_dft/simulator.py` (sections 2 to 10).
- `stm_from_dft/cmap_shading.py` (section 10).
- `stm_from_dft/qtgui/qtplot.py` and `controls.py`, with `matcher/qtgui` identical (3.1, 3.2, 3.6, 3.9, 4 if the toggles need a control). The root `qtgui/` per Q20.
- `stm_from_dft/locpot_manager.py` (Q11).
- `stm_from_dft_opt/README.md` and `run_DFT_STM.py` (comments and docs).
- `mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py` (Q10).
