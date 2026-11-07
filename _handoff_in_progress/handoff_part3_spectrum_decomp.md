# Handoff, part 3: the spectrum decomposition (decomp) in the simulator, and the info panel and saves of every `qtgui` window

Repository: `VASP_for_SPM`. This part changes:

- the simulator package `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/`: `simulator.py` and a new `ldos_decomp.py` (sections 2 to 4);
- an legend panel beside the spectrum in every view of the simulator (section 5);
- `qtgui/qtplot.py`, `qtgui/controls.py` and `qtgui/mplexport.py`: the info panel and the saves of every window that uses `qtgui`, taken from sm4_browser (section 6).

Line numbers below refer to the owner's `simulator.py` and `qtgui/qtplot.py` as they were before part 1. They are approximate: find the code by the names given.

## What is wrong today

The owner's report: clicking a curve in the spectrum panel spends a while loading, shows the decomposed curves in the next level's colours, then goes back to the undecomposed view.

The cause, found in the code and reproduced with the owner's `qtgui.qtplot` and the simulator's own handlers:

- **One left click runs two handlers.** `FigureCanvas.mousePressEvent` in `qtplot.py` (about lines 6113 to 6119) calls `figure._pick(e)` first, which fires `pick_event`, then fires `button_press_event` with the same event. matplotlib uses the same order.
- **The pick handler steps down.** `_on_pick` (about lines 2776 to 2816) sets the next `plot_level` and calls `_update_all`. In Map, `_get_partitions` then returns the next level's partitions (Au and Molecule at level 1), so the map block is rebuilt with each partition's maps (E Pts maps per partition). That is the long load. The spectrum is redrawn in the next level's colours.
- **The press handler then steps back.** `_on_press` (about lines 2818 to 2825) runs on the same click, after the redraw. It tests the click against the curves now in the panel (`l.contains(event)` over `self.ax_spec.get_lines()`), finds none under the mouse, and sets `plot_level` back one level. Then it calls `_update_all` again, which redraws the undecomposed view and rebuilds the maps again.
- **Overlapping curves.** `Figure._pick` (about lines 6990 to 6998) fires one `pick_event` for every pickable curve under the mouse. `_on_pick` steps down once for each, redrawing between them. With two overlapping marker curves, one click went down two levels before the press handler stepped back one.
- **Double click.** `FigureCanvas.mouseDoubleClickEvent` (about lines 6121 to 6127) fires `pick_event` and `button_press_event` again, with `dblclick=True`. Qt sends a press and a release before the double-click event. `_on_pick` doesn't check `dblclick`, so a double click is handled as two clicks.
- **Neither handler checks the button.** A right click on a curve steps down too.
- **The spectrum's levels don't match the owner's layers.** The layers are markers, Au / Molecule, atom types, and the orbital types of one atom type. The maps and heatmaps follow them: `_get_partitions` gives the element's orbitals, summed over its atoms, at both level 3 and level 4. The spectrum doesn't:
  - its level 3 draws one curve per atom of the element, labelled by VESTA label (`_update_all`, about lines 2383 to 2392);
  - a click there selects that atom (`_on_pick`, about lines 2808 to 2811);
  - its level 4 draws that one atom with its orbitals (about lines 2393 to 2419).

  The older matplotlib copy (`sim_decay_full.py`) and `plot_dos_cursors_levels.py` have the same single-atom spectrum levels.
- **The same handlers under matplotlib** (3.11.2, the same order) step back the same way, so this is in the handlers, not something the Qt port introduced.
- **Not checked:** whether drawing the level 1 maps also raises an error. `FigureCanvas._fire` (about lines 6067 to 6072) catches an error in one handler, prints its traceback, and goes on to the next handler. So an error there would show in the owner's console, and the press handler would still step back.

## 1. Rules for this job

1. Implement exactly what this plan says. If the code does not match the plan, or a step seems to need a change the plan does not name, stop and report it instead of choosing.
2. Before writing any code, report:
   - the functions `ldos_decomp.py` will have (names, arguments, what each returns);
   - how `simulator.py` will call them;
   - how the legend panel is built and placed in each view;
   - which functions of sm4_browser's `qtplot.py`, `controls.py` and `mplexport.py` go into this repository's `qtgui`, and what each replaces here (section 6).

   Wait for the owner's approval.
3. Add no tests or checks, and run no window. The only check is `python -m py_compile` on every changed `.py` file.
4. `mo_ldos_projection/sim_wrapper.py` keeps working, with no changes. It reads `sim.plot_level` and `sim.active_marker_idx` and sets them back on a new simulator (`_instantiate`, about lines 205 to 211; `capture_state`, about lines 276 to 277). Both keep working as attributes of the simulator, read and set.
5. At the end, list per file every function, constant and branch you changed, added or removed, in two lists: approved changes, and additional changes you made. If the second list is not empty, stop and report it before committing.

## 2. What the decomp does (the owner's rules)

**The levels (the owner's rule):** the decomp goes down to atom types and then to the orbital types of one atom type, never to single atoms. Today's level 3 (one curve per atom) and level 4 (one atom with its orbitals) go. The new levels:

| Level | The spectrum shows | The legend panel's entries | A left click on a curve or entry |
|---|---|---|---|
| 0 | one curve per marker | each marker | that marker becomes the active marker; level 1 |
| 1 | the Au and Molecule partitions of the active marker | Au, Molecule | level 2 |
| 2 | one curve per atom type | each atom type | that type becomes the active element; level 3 |
| 3 | one curve per orbital type of the active element | each orbital type | nothing (the last level) |

**Level 3, the orbital types of one atom type:**

- **Which orbitals:** the same ones the maps and heatmaps already show at level 3. `_get_partitions` (about lines 914 to 925) takes each column of `self.orbitals` and sums it over the active element's atoms. It leaves out a column whose DOS, and the DOS's first and second differences, all stay below 1e-5. In a spin-polarized run the `_up` and `_down` columns are separate entries, as in the maps.
- **The curves:**
  - the element's colour;
  - one line style per orbital, the same style for its `_up` and `_down` columns;
  - the lighter colour for `_up` columns.

  That is the style today's level 4 gives its orbital curves (`linestyle_map`, `_orbit_base`, `_lighten_color`).
- **Behind them:** the element's total, faded, as today's level 3 draws it.
- **The hover labels** (`widgets.HoverLabels`) that today's level 4 puts on its orbital curves stay on these curves.
- **The legend panel** lists the orbital types. A left click on one, or on a curve, does nothing. A right click goes back to level 2, as everywhere in the spectrum and the legend panel.

**The maps and heatmaps** already stop at the element's orbitals: today their level 3 and level 4 partitions are the same. With level 4 gone, they show those orbital partitions at level 3, as now.

**The clicks:**

- **Left click on a curve:** goes down one level with that curve's option (levels 0 to 2; at level 3 it does nothing).
  - When several curves are under the mouse, only the top one counts: the highest zorder, and for equal zorders the one drawn last.
  - That is the order both matplotlib and Qt draw in. Qt stacks items with the same Z value by insertion order.
- **Left click on an entry in the legend panel:** goes down one level with that entry (levels 0 to 2; at level 3 it does nothing).
- **Left click on neither:** does nothing.
- **Right click anywhere in the spectrum panel or the legend panel** (on a curve, on an entry, or on empty space): goes back one level (levels 1 to 3). It never goes down.
- **The Back button** in the legend panel (levels 1 to 3; level 0 has none): does the same as a right click.
- **The two ends do nothing (the owner's rule):** at level 0 a right click does nothing, and at level 3 a left click does nothing. Neither redraws.
- **Double click:** has no action of its own. The double-click event is ignored, so a double click acts as one left click.
- **One click, one step:** every click changes the level at most once, and the window is redrawn once for it.

**The legend panel (the owner's rule):**

- **No legend on the spectrum:** the spectrum panel draws no legend at any level, in any view.
- **The panel:** the panel to the right of the spectrum, in every view (Single Point, Line, Path, Map), with the clickable buttons for the different curves. It is the spectrum's legend and its clickable options at once.
- **Its order, top to bottom (the owner's rule):**
  1. the level title, which stands for the whole panel;
  2. the Back button;
  3. the stack of entry buttons, the only scrollable section of the panel.

  The title and the Back button are part of the panel but not of the scrolling part: they stay in place while the stack scrolls.
- **Its entries:** one per option of the current level (the table above). Each entry shows:
  - its curve's sample, in the colour, line style and line width its curve is drawn with;
  - its label.
- **Short labels (the owner's rule):** no entry's label is ever longer than it needs to be: the longest are an element symbol with its orbital (level 3) and the word "Molecule" (level 1).
  - Level 1: "Au" and "Molecule". Level 2: the element symbols.
  - Level 3: the element symbol and the orbital, as the maps already label their level 3 partitions (`f"{active_element} {orb}"` in `_get_partitions`), for example "C pz", or "C pz_up" in a spin-polarized run.
  - **Level 0 (the owner's rule):** no text, just each marker's colour (its curve's sample).
    - **Order (the owner's rule):** marker index order, in every view.
      - **Single Point and Map:** the marker index as it is.
      - **Line and Path:** the marker index is kept in the order the markers appear on the LDOS topo stripe, top to bottom (section 2a), so the entries follow the stripe.
- **Clickable entries are buttons (the owner's rule):** every entry that stands for a clickable curve (levels 0, 1 and 2) has a box around its sample and label, and is clearly a clickable button.
  - **Claude's reading, recorded as such:** level 3's entries (the orbital types) stand for curves a left click does nothing on, so they are plain key rows with no box.
- **Not entries:** the faded totals drawn behind the curves (level 1's marker total, level 3's element total) and Map's energy markers.
- **Its Back button (the owner's rule):** at levels 1, 2 and 3 the panel has a Back button that does the same as a right click (back one level). The marker level (0) has none.
  - **Where (the owner's rule):** just under the level title, above the scrollable stack. It stays there while the entries scroll, so it is always reached without scrolling.
  - **Its look (the owner's rule):** a gray background, not white. It says "Back".
  - **What it is:** a Qt control, so the saves leave it out. sm4_browser's export, which section 6 brings here, leaves every control out with its place left empty, so nothing moves.
- **Its title:** at the top of the panel, above the Back button, fixed while the stack scrolls. It is saved with the figure.
  - **Never wider than the panel (the owner's rule):** the title is never longer than the panel is wide. The panel's width is set by its largest entry, so the titles are short words.
    - level 0: none;
    - level 1: "Parts" (the owner's word, in place of "Partition");
    - level 2: "Atoms" (the owner's word, in place of "Atom Types");
    - level 3: "Orbitals".
  - **If a title would still be wider than the panel** at the window's font, the code session stops and reports it, rather than widening the panel or shrinking the text (rule 1).
- **Live:** it is rebuilt with the curves whenever the level or the data changes, so it always matches what is drawn.

**What stays as it is:**

- **Map's energy markers** (`emarker_i`, the vertical lines in the spectrum): a left click on one at level 0 starts its drag, as today. At other levels a left click on one does nothing, as today. They are not options.
- **A view change** resets the level to 0 and the active marker to 0 (`_on_mode_change`, about lines 1897 to 1901).
- **Clicks outside the spectrum and legend panels** do what they do today. That includes the marker dragging on the Line / Path panels (the `mark_dynamic` branch of `_on_pick`) and the redraw `_on_press` makes for them.

## 2a. Line and Path: the marker index follows the LDOS topo stripe (the owner's rule)

- **The rule:** in Line and Path, the marker index is kept in the order the markers appear on the LDOS topo stripe, top to bottom. Index 0 is the marker highest on the stripe.
- **When:** like Map's energy markers, which part 2 puts back into increasing energy when one is released (part 2 pre-flight answers, "the Map panels always in increasing energy"). After a Line or Path marker is moved and released (`_on_rel`), the markers are re-sorted by their place on the stripe.
  - The stripe's vertical axis is the distance along the line, from 0 at the bottom to the line's length at the top (`ax_stripe.set(ylim=(0, p_len))`, about line 2547). So top to bottom is `marker_ratios` from largest to smallest.
  - The sort is stable, so markers at the same place keep their relative order.
- **What the permutation is applied to, together:**
  - `marker_ratios`;
  - `active_marker_idx`, so the active marker stays the same marker;
  - the markers' colours (below);
  - every per-marker cache.
- **The colour stays with the marker (the owner's rule):** each Line and Path marker keeps its colour wherever it moves, and only its index changes.
  - Today a marker's colour comes from its index, `m_colors[i]` (for example `_update_all`, about lines 2355, 2522 and 2593). In Line and Path each marker's colour is kept with it instead, and permuted with it in the re-sort.
  - Every panel that draws a marker (the LDOS topo panel, the stripe, the heatmaps, the Tip Height plot, the spectrum and the legend panel) uses that marker's kept colour.
- **The panels:** redrawn once for the new order, with no rearrangement or rebuild, as the Map re-sort.
- **Unchanged:** Single Point and Map keep their marker index as it is, and their colours by index as today.
- **Claude's reading, recorded as such:**
  - markers added by the Markers count box are sorted in the same way;
  - an added marker gets the first colour of `m_colors` that no marker has.

## 3. The new file `ldos_decomp.py`

The owner wants the decomp to be its own file, called by every view of the simulator. It holds everything that depends on the level:

- **The state:** the level (0 to 3) and the selections (the active marker and element).
  - The simulator's `plot_level`, `active_marker_idx` and `active_element` stay readable and settable on the simulator (rule 4).
  - `active_atom` goes, since no level selects an atom.
- **The partitions:** what `_get_partitions` (about lines 896 to 938) returns for levels 0 to 3; its level 4 branch goes. The maps take them from here: `_update_all` (about line 2300) and `_redraw_map_slice` (about line 2718). So do the Line / Path heatmaps.
- **The spectrum's curves:** what each level draws in the spectrum panel, today in `_update_all` (about lines 2347 to 2420).
  - That is the curves, their colours, styles and labels, and which curves are pickable.
  - At level 3, the orbital curves with their hover labels (section 2).
  - Today's level 3 (single atoms) and level 4 branches go.
- **The legend entries:** the legend panel's entries at each level, each with its colour, line style, line width and label, and the panel's title (section 2).
- **The clicks:** the left / right / double-click and Back button rules of section 2. They replace two pieces of code:
  - the spectrum branch of `_on_pick` (the `self.ax_spec` branch, about lines 2793 to 2812);
  - the spectrum part of `_on_press` (its `self.ax_spec` part, about lines 2819 to 2823).

**The decision is made from the curves as they were clicked:** for a click in the spectrum or legend panel, the curve or entry is chosen once, from what was drawn when the click happened, before any redraw. No handler tests the click again after the redraw.

**Helpers** used only by the decomp move into the new file with it:
- `_orbit_base`, `_lighten_color`, the orbital line-style map and the faded-colour factor `S`;
- today they are local to `_update_all` (about lines 2324 to 2339).

**Shared data** stays in the simulator and is passed in: `atomtypes`, `atomnums`, `orbitals`, `_type_color_map`, `m_colors` and `dos_up_gpu`.

**No longer used by the decomp:** `vesta_label_map` and `_get_element_by_index_helper`. They stay wherever else the simulator uses them.

**Not touched:** `plot_dos_cursors_levels.py` has its own levels and stays as it is.

## 4. `simulator.py`

- **The spectrum panel in every view** gets its curves and its click handling from `ldos_decomp.py`, and the legend panel gets its entries from it.
- **The spectrum's legends go:**
  - every `self.ax_spec.legend(...)` call in `_update_all` (about lines 2365, 2375, 2382, 2392 and 2417 to 2419);
  - the `self.ax_spec.add_artist(leg1)` that keeps level 4's second legend (about line 2418).
- **`_get_partitions`** goes, and its callers take the partitions from `ldos_decomp.py`.
- **Kept:**
  - the `pick_event` and `button_press_event` connections (about lines 1194 to 1197);
  - every branch of `_on_pick` and `_on_press` outside the spectrum and legend panels.

## 5. The legend panel

- **Where (the owner's rule):** to the right of the spectrum, in the same row and at the row's height, in every view.
- **One sub-section (the owner's rule):** the spectrum and the legend panel always belong to the same sub-section, so they always sit together. In Line and Path, where the 1D sub-section also holds the Tip Height plot, the spectrum and the legend panel form one sub-sub-section of their own inside it.
- **Width (the owner's rule):** as wide as its largest entry (sample, label and button box), which the short labels keep narrow (section 2). The title is never wider than that (section 2, "Its title").
- **Clicks:** an entry is found by where the click lands in the panel. A left click on an entry is that entry's option (section 2). A right click anywhere in the panel, or the Back button, goes back one level (levels 1 to 3; nothing at level 0).
- **How it is built (the owner's choice):** a plot panel.
  - It is a `qtplot` axes with no ticks or frame, its samples drawn as lines, its labels as texts and its button boxes as rectangles.
  - It is saved with the figure, as the legend is today: the vector saves rebuild every axes with its lines and texts.
- **When the entries don't fit (the owner's rule):** the stack of entries scrolls in the window, under the fixed title and Back button, and the panel takes only the space the layout gives it.
  - **Scrolling (the owner's rule):** identical to sm4_browser's scroll areas: the mouse wheel, and a vertical scroll bar that appears when the entries don't fit. sm4_browser's marker box (`controls._MarkerBox`, a `QScrollArea` with no horizontal bar) is the example.
  - **In the saves (section 6):**
    - the PDF, EPS and vector SVG hold the full list, the panel extending down past the end of the window as far as its list needs;
    - the Qt-drawn SVG and the window images are exactly as shown, the panel scrolled where it is, in its own space.
- **Unchanged:** `qtgui/qtplot.py`'s `Axes.legend` and every other legend in the repository.

## 6. The info panel and the saves, in every `qtgui` window (from sm4_browser)

**The owner's decision:** every window in this repository that uses `qtgui` gets sm4_browser's info panel and sm4_browser's saves. That covers the simulator, the decomposition windows, the MOPMatcher plotters, `run_wfn_plane_slice.py` and the rest.

**The source:** the public repository `Ben-Kafin/sm4_browser`, commit `eb9bb6279e70`. Its `qtplot.py` and `mplexport.py` there are the same as the owner's copies on disk. Each package keeps its own copy of shared logic, so this repository's `qtgui` gets its own copy. It does not import from sm4_browser.

**What this repository's `qtgui` has today:**
- **The info panel** is a Qt text box docked at the bottom of the window (`FigureManager.__init__`, about lines 6376 to 6400). It doesn't wrap (`LineWrapMode.NoWrap`). Its text (`Figure.info_text`, about line 6579) is only the details cut from the panel titles and the window's own lines. `controls.py` has no settings text.
- **No save holds the info panel.** `mplexport.py` has no info panel code, and the screenshot leaves the info panel out (`Figure.savefig`'s docstring, about lines 7376 to 7392).
- **The save list** (`NavigationToolbar.SAVE_FORMATS`, about lines 6313 to 6323):
  - PDF, EPS, SVG, PNG and TIFF publication files, rebuilt by `mplexport` at the window's design size and cropped to what is drawn (300 dpi for the images);
  - PNG and SVG as on screen (pyqtgraph);
  - a PNG screenshot;
  - CSV.

**What it gets, as sm4_browser has it:**

- **The info panel in the window:**
  - it wraps at its width (`LineWrapMode.WidgetWidth`);
  - it shows only once it has text;
  - its text (sm4_browser's `Figure.info_text`) is the details cut from the panel titles and the window's own lines, then an empty line, the line "settings", and the setting of every control in the window (sm4_browser's `controls.settings_text`);
  - it is updated whenever a control's setting changes (sm4_browser's `Control._changed`, which calls the window's hook, `sm4_settings_changed` there, then the coalesced info push).
- **The save list** (sm4_browser's `NavigationToolbar.SAVE_FORMATS`) replaces this repository's:
  - PDF, EPS and SVG, vector and editable: the plots as on screen and the full info panel (sm4_browser's `mplexport`: the canvas at its on-screen size, the full info panel below it, wrapped as on screen, through `_info_lines`, `_info_geometry` and `_info_block`, nothing cropped);
  - SVG of the whole window as Qt draws it, controls included (`'qt-svg'`). Here it differs from sm4_browser, whose label says "full info panel": this save is exactly as shown (see below), and its label in the save dialog says that, not "full info panel";
  - PNG and TIFF of the window at 720 dpi (`Figure.window_image`, `WINDOW_IMAGE_DPI = 720`), the info panel as shown;
  - CSV, the curves' data.
- **The pieces** (sm4_browser's names): `Figure.export_info`, `export_info_panel`, `export_root`, `_export_rect`, `window_image` and the `'vector'`, `'qt-svg'` and `'image'` paths of `Figure.savefig` in `qtplot.py`; `_to_matplotlib`'s canvas-plus-info-block figure and the info block functions in `mplexport.py`; `settings_text` and `Control._changed` in `controls.py`.

**The two scrollable panels, the legend panel and the info panel, in each save (the owner's rules):**
- **In the PDF, EPS and vector SVG:** both are full and expanded, and they don't overlap.
  - The legend panel holds its full list and extends down past the end of the window as far as it needs.
  - The full info panel starts below the canvas, moved down below the bottom of the legend panel when the legend panel reaches past the canvas.
- **In the Qt-drawn SVG and the window images (PNG, TIFF):** exactly as shown. Both panels appear as the window shows them, each scrolled where it is, in its own space.

## 7. Finish

- `python -m py_compile` every changed `.py` file. Run nothing else.
- Commit on the same branch as parts 1 and 2. The owner tests on his own machine.
- Report the two lists of rule 5.
