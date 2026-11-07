# Simulator views: layout and topography decisions (for a handoff after parts 1 and 2)

Collected 2026-11-06 from Benjamin's answers. These are not part of the running part 1 and part 2 handoffs.

## Terminology (Benjamin's, for the documentation and every reference)

- **The global topo:** the topography at the topo bias and the topo tip height.
- **The LDOS topo:** the topography at the LDOS bias V_b and the LDOS tip height (the LDOS h value). Only the LDOS topo uses the LDOS tip height.
- **Retired names:** "Line Topo", "Map Topo", "Path Topo" and a "Single Point topo". The panels that showed them are LDOS topo panels.
- **The stripe:** the Line / Path tip heights along the line or path. It is not a topo of its own.

## Names in the code and on the plots

- **Benjamin's rule:** the names and titles used in the code and on the plots are "global topo" and "LDOS topo".
- **The stripe's title:** "Topo", and only that. The stripe sits in the Line / Path heatmap cluster. Today it has no title.
- **The other panels:** each keeps its own name (Tip Height, the heatmaps' partition labels, the LDOS plots).
- **The topo colour bar:** keeps its title "topo".
- **What this changes, in the copy of simulator.py Claude has (from before parts 1 and 2):**
  - **Titles:**
    - "Map Topo: bias … V, height … Å" becomes the LDOS topo's title.
    - Line's LDOS topo panel, titled "Topo" today, takes the LDOS topo's title.
    - The global topo keeps "Global Topo: bias … V, height … Å".
  - **Code identifiers:**
    - `current_z_map` holds the global topo until Map computes an LDOS topo, then the LDOS topo.
    - `ax_map` is the global topo panel in Single Point, Line and Path, and the LDOS topo panel in Map (beside `ax_map_global`).
    - `ax_line_topo` is Line's LDOS topo panel.
    - Since every view will now show both topos, each topo gets its own variable and panel under the new names.
  - **Comments, docstrings and console prints:**
    - "Map Topo" and "Line Topo" in these become "LDOS topo".
    - So do the prints "Map Local Setpoint LDOS" and "Path Setpoint LDOS".
- **Unchanged:** the cache names `global_topo_*` and `ldos_topo_*`, which already match. The view names stay: Single Point, Line, Path, Map.
- **For the handoff:** the exact new identifiers will be listed there for approval, against the code as it stands after parts 1 and 2.

## The principle

- Each view (Single Point, Line, Path, Map) is made of individual panels grouped into sub-sections.
- The sub-sections keep the same layout relative to each other, whatever the settings. Single Point's tall-window rule (below) is Benjamin's one stated exception.
- Inside a sub-section, the panels arrange by each panel type's own rules.
- **A row has one height, no matter what it contains (Benjamin's rule).** Every panel and every sub-section that shares a row takes that height, in every row of every view. Claude's reading, recorded as such: the height is the plot box's, so the axes line up; titles and axis labels sit outside it.

## The cursor readout (every window and panel in this plan)

- **Benjamin's rule:** while the mouse is in any panel, the window's top right shows the mouse's x and y in that panel's axes coordinates, as matplotlib's toolbar did automatically.
- **One set of coordinates only:** the readout is for the panel under the mouse, and there is only ever one.
- **Off every panel:** when the mouse leaves a panel and isn't on another, the readout goes away.
- **What matplotlib did, checked in matplotlib 3.11.2's source:**
  - `NavigationToolbar2._mouse_event_to_message` builds the text from the axes under the mouse, with `Axes.format_coord`, which gives "(x, y) = (…, …)".
  - It returns an empty string when the mouse is over no axes.
  - For an image-like artist under the mouse, it adds that artist's value on a second line. Benjamin's rule asks for the x and y only.
  - In the Qt backend, `NavigationToolbar2QT` puts that text in a right-aligned label at the toolbar's right end, and `FigureManagerQT` adds the toolbar with `addToolBar`, which Qt places in the top toolbar area.

## The topo sub-section (every view)

- **Its panels, from the left:** the topo colour bar, the global topo, the LDOS topo. Every view has all three: Single Point, Line, Path and Map.
- **The topo colour bar:**
  - its own panel in every topo sub-section, always all the way to the left;
  - it doesn't overlap any other text, axis label, tick mark or tick value. The global topo's y-axis label and tick values, on that panel's left, sit clear of it;
  - its own look is unchanged: the topo colour map's ramp, no ticks and no numbers, "topo" above it;
  - **the level line (the contour slice):** only in Map, where it sets the LDOS topo's iso-line on the energy maps. Single Point, Line and Path have no level line on their colour bar.
  - Line and Path gain it. Today only Map has one. Their heatmap colour bars are separate and stay with the heatmaps.
- **The global topo, in every view:** carries the crop adjusters (Crop / Uncrop and the cursor boxes) and the auto rotate. In the copy Claude has this already holds: every view builds its global topo as the crop panel (`_placement`), and its orientation is chosen when the view is built and again on Crop / Uncrop (`_choose_orientation`).
- **The LDOS topo:**
  - **Map and Single Point:** takes the global topo's crop and auto rotate: the same crop limits, and it turns with the global topo. Map already does this in the copy (`_apply_crop` gives the Map Topo the placement topo's limits; `_unit_cell_axes` turns it with the placement topo). Single Point gets the same.
  - **Line and Path:** the LDOS topo panel is already sized for what it shows (Line's 2:3 box centred on the line; Path's turned rectangle with 1/8 margins). It takes neither the global topo's crop nor its auto rotate. Path's panel still turns 90° by its own rule when its rectangle is wider than tall.

## Line

- **Top row:** the topo sub-section (the topo colour bar, the global topo, the LDOS topo panel), then the stripe and the heatmaps with their colour bars. The whole row has one height, so the LDOS topo panel's, the stripe's and the heatmaps' distance axes line up.
- **Second row:** the Tip Height 1D plot and the LDOS 1D plots, at the row's one height.
- **Markers:** drawn in their locations and colours in every panel.
- **Sizing:** the xy-spatial panels grow with the window, keeping their ratio, up to the largest size the row's width allows. When the window is taller than that, the 1D plots expand vertically and take all the remaining height. No empty band.

## Path

- **Rows:** the same two rows as Line, with Path's LDOS topo panel in the topo sub-section.
- **Path's LDOS topo panel, the rectangle:** the smallest rectangle at any angle, turned to follow the path, that holds every sample point of the path. A margin of 1/8 of the rectangle's width is added on the left and right, and 1/8 of its height on the top and bottom.
- **Path's LDOS topo panel, its shape:** always the row's height, and never wider than square. When the rectangle is wider than tall, the panel turns 90°, so it is always square or taller than wide.

## Map

Three sub-sections:

1. **The topo sub-section:** from the left, the topo colour bar, the global topo and the LDOS topo.
2. **The spectrum (marker 1D plot):** in the same row, the same shape and size as one topo panel. Its plot box equals a topo panel's plot box. That is Claude's reading, and Benjamin did not object.
3. **The energy maps:** below the top row, in as many rows and columns as make each map largest in a full 16:9 window. The columns are chosen again on every resize.

**Size limits:**
- a map ≤ a topo panel ≤ 2 × a map
- the placement-panel minimum (below)
- together these set a minimum window size
- **When the window can't fit all of these:** the placement-panel minimum holds, and the 2× cap gives way, so the maps get smaller than half a topo panel. This is Claude's reading of Benjamin's statement that the panel he places markers on must stay large enough to see clearly. It is recorded as such.

**Energy markers:** after a drag, the maps are re-sorted into increasing energy on release, in place, with no rearrangement. This is already in the part 2 pre-flight answers.

## Single Point

- **Layout:** Map's top row: the topo sub-section (from the left, the topo colour bar, the global topo and the LDOS topo), then the spectrum in the same row at one topo panel's size. It replaces today's choice between topo-left / spectrum-right and topo-top / spectrum-below by the window's shape.
- **In a window taller than that one row (Benjamin's rule):** the topo sub-section fills the top row, and the 1D sub-section (the spectrum) fills the bottom.
  - **Claude's reading of "if this ever becomes a problem", recorded as such:** the switch happens when the one row would leave an empty band below it, and it switches back when the window no longer would.
- **Its spectra:** they take their tip heights from the LDOS topo at the current V_b, as Map does. Today they take them from `current_z_map`, which can be the global topo or an LDOS topo at an earlier bias.
- **The LDOS topo is shared:** Single Point and Map use it through the `ldos_topo_*` cache. Switching between the views at the same settings reuses it from memory or from the cache.

## The LDOS h box

- It sits above every LDOS topo panel, because the LDOS topo is converged from that tip height: in Map, in Single Point, in Line and in Path.
- Each LDOS topo panel has the editable LDOS h value above it.

## The placement-panel minimum (atom coverage)

- **What Benjamin needs:** the panel where the markers are drawn on the topo and the line and path ends are shown must be large enough for him to clearly see where he is placing them relative to the topo.
- **The rule that sets it:** the plotted atoms (the atom dots, not the spectrum markers placed by the user) may cover at most 1/3 of the panel, so at least 2/3 of the topo stays visible.
- **It applies whether the Atoms toggle is on or off.**
- **Overlaps that come only from the unit cell's periodicity don't count.**
- **Claude's reading of his statement, recorded as such:**
  - **The counting:** the covered area is counted as it is seen on screen, with overlapping dots counted once.
  - **Where it applies:** to the panels the markers, line ends and path nodes are placed on. That is the placement panel in Single Point, Line and Path, and in Map the LDOS topo panel, which holds the markers in the current code.

## The bias toggles (every view)

- A toggle beside E Min and beside E Max picks which end is the bias. Exactly one is on, and one is always on. `ldos_bias_sign` only sets the starting state.
- **An end set to 0:** the bias moves to the other end automatically, and that end's toggle refuses to turn on while it is 0. When the end becomes nonzero again, the bias stays on the other end until Benjamin switches it. Both ends at 0 stays an error.
- **Switching the toggle:** triggers the same recomputes, with the same caches, as changing that end's value.

## The topos

- **The global topo and the LDOS topo:** only ever computed on a single unit cell's grid and tiled for display. The code already works this way.
- **The grid's duplicate edge:** kept. `np.linspace(0, 1, grid_res)` stays as it is, so the grid and the cache names don't change.
- **The LDOS topo panels of Line and Path:** show the LDOS topo. If it is already computed at that V_b, they take it from memory or the `ldos_topo_*` cache. Otherwise they compute it and save it under that bias's cache name.
- **The stripes (Line / Path tip heights):** keep their own quick line convergence, with no cache. The rules for that convergence:
  - **Starting heights:** the LDOS topo interpolated at the path's sample points: bilinear, periodic in the cell's fractional coordinates. The grid's duplicate edge row and column are left out of the interpolation only, since they repeat the edge at 0.
  - **Setpoint (Benjamin's rule):** taken at the location of the LDOS topo's tallest height in the unit cell, even when the line or path doesn't go there. The stripes then sit on the same LDOS topo tip height as the LDOS topo itself.
    - **The value:** the integrated LDOS at V_b at that point's xy and its LDOS topo height. The LDOS topo and the stripe make the same LDOS call (window sorted([0, V_b]), the same bias and decay setting), so at the same point and height it is the same calculation.
    - **When several grid points share the tallest height:** among them, the one with the largest integrated LDOS. Every grid point within 1% of the setpoint at the start height never moves, so all of them keep the tallest height. The one with the largest value is the point the LDOS topo's own setpoint came from, so the stripe's setpoint equals the LDOS topo's.
    - **Why it matches, from the code:** the LDOS topo starts flat at z_highest_atom + LDOS h, and its setpoint is the largest integrated LDOS on that flat grid. That point is exactly on the setpoint, so it never moves. The LDOS topo's tallest height is therefore z_highest_atom + LDOS h, at the xy where the integrated LDOS at that height was largest. That xy need not be directly over the tallest atom.
    - **From the cache:** the cache keeps only the heights, so the values are computed again at the points at the tallest height. It doesn't need a stored number.
  - **The other points:** every line or path point moves up or down from its starting height with the existing rule. A point within 1% of the setpoint isn't moved again.

## Still open

Nothing, as of 2026-11-06.
