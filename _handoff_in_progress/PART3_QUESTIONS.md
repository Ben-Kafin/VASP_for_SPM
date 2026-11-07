# Part 3: the open choices and the impossible items (for the owner)

Found by reading the code (not by the pre-flight agents, which are still running; they may add more).
Each has options and a recommendation (R).

## What the plan asks that cannot be done as written

**Q1. The legend panel's scrolling stack (section 5).** The plan: the legend panel is a qtplot axes, and its stack
of entries scrolls exactly like sm4_browser's `_MarkerBox`, which is a `QScrollArea`. In this repository's qtgui,
a figure is one pyqtgraph scene (`FigureCanvas(pg.GraphicsView)`, "one pyqtgraph scene in pixel coordinates,
the Qt controls as child widgets"); a `PanelCell` is an empty, mouse-transparent placeholder widget whose
rectangle the layout sets, and the axes is drawn in the scene at that rectangle. An axes of the figure cannot
be put inside a `QScrollArea`.
- (a) The behaviour of `_MarkerBox`, built differently: a Qt vertical scroll bar at the panel's right edge,
  shown only when the entries don't fit, and the mouse wheel over the panel; both move which entries the axes
  shows (its y range). The title and the Back button stay put. The panel stays part of the figure, so the
  saves can rebuild it.
- (b) A second canvas (its own scene) inside a real `QScrollArea`. The legend panel is then no longer part of
  the figure: the vector saves, the cell layout and the click handling all need extra code to treat it as one.

R: (a).

**Q2. "A title wider than the panel: the code session stops and reports it" (section 2).** Rule 3 forbids
running a window, so I cannot measure a title at the window's font. Whether "Orbitals" (title font) is
narrower than the widest entry depends on the font, which I cannot measure without running something.
- (a) You check it on your machine when you test.
- (b) The window prints one console line at build time when a title is wider than the panel (a check
  added to the code, which rule 3 forbids unless you allow it).

R: (a).

## Where part 3 collides with part 1 (both approved)

**Q3. The legend panel's width vs part 1's "Panels move only on three triggers" (5.2: panel count, crop,
resize; "Nothing else moves or resizes a panel").** Part 3: the panel is "as wide as its largest entry". If
that is the largest entry of the current level, every level step changes its width, which moves the spectrum
beside it on a click.
- (a) One width: the largest entry over all four levels, computed from the loaded data (the atom types,
  orbitals, "Molecule"; the markers are colour only, so their count does not change it). Nothing moves on a
  click.
- (b) The current level's largest entry: the panel and the spectrum change width on each step (breaks 5.2).

R: (a).

**Q4. Pixels to points in the ported saves vs part 1's 4.4.** sm4_browser's `mplexport` and `window_image`
convert with its fixed `q.DPI` (100 logical px per inch: `figsize=(W / q.DPI, ...)`, the info block's
`size = info.pixelSize() * 72.0 / q.DPI`, the window image at `WINDOW_IMAGE_DPI / DPI`). Part 1 4.4 made this
repository's `mplexport` convert with `pt2px` ("so the publication files keep matching the screen").
- (a) Port sm4_browser's code with every pixel-to-point and pixel-to-inch conversion done with part 1's size
  API (`pt2px` / `px_per_pt`); matplotlib's own `dpi` for written files stays.
- (b) Port it as is, with `q.DPI` (part 1's 4.4 conversion lost in the saves: text sizes in the files no
  longer match the screen when px_per_pt is not DPI/72).

R: (a).

## Choices the plan leaves open

**Q5. Does the root `qtgui/` get section 6?** It serves `PROCAR_tools/Band_structure_plot.py`,
`charge_density/lv_slice_w_atoms_2system.py` and `convergence/forces.py`. Part 1 left it unchanged; part 3
says "every window in this repository that uses qtgui ... and the rest".
- (a) Yes: the root copy gets the same info panel and saves (with its own DPI-based sizes, since it has no
  part 1 size API).
- (b) No: only the two package copies.

R: (a), the literal reading of "every window".

**Q6. Ruling 8, the window's settings held in no control.** sm4_browser's `settings_text` lists only its
controls (`Control._all`: visible, inside the window, with a setting). Its windows add their own lines with
`set_info_line` (the Line window's E-marker readout). State set with the mouse is in no control, for example
in the simulator: the decomp level and its selection (active marker, active element), the marker positions,
the Line ends and Path nodes, Map's energy markers, and the crop rectangle where no box shows it. The
pre-flight is listing, per window, which settings already sit in a control.
- (a) Every such setting as the window's own lines (`set_info_line`), updated whenever it changes.
- (b) Only the decomp state.
- (c) None: controls only, as sm4_browser.

R: (a).

**Q7. The 'qt-svg' label.** sm4_browser: "SVG, the whole window as Qt draws it, controls included, full info
panel (*.svg)". The plan: this save is exactly as shown and its label says so, not "full info panel".
R: "SVG, the whole window as Qt draws it, controls included, as shown (*.svg)".

**Q8. The hook's name.** sm4_browser's `Control._changed` calls the window's `sm4_settings_changed`.
- (a) Keep the name, so this copy reads the same as sm4_browser's.
- (b) A name without "sm4", e.g. `settings_changed`.

R: (a).

## What waits on these

- Q1, Q2, Q3: the legend panel.
- Q4 to Q8: section 6 (qtgui).
- Not blocked: ldos_decomp.py, the click rules, section 2a (Line / Path marker order and colours).
