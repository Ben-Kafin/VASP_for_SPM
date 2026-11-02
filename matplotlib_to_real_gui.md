# From a matplotlib-wired GUI to a real GUI

A guide for turning a program whose "GUI" is matplotlib figures with hand-placed axes and
`matplotlib.widgets` into a real Qt application. It is written to be used on any such program.
The worked example throughout is the sm4_Browser package (PySide6 + pyqtgraph), where every rule
below was applied to every window. Its files `qtplot.py`, `controls.py` and `widgets.py` form a
reusable layer, described in section 9; VASP_for_SPM carries that layer as `qtgui/` at its root,
where this guide now lives, and every interactive window of the suite follows it.

## 0. The goal

The conversion has one goal, and every rule below serves it:

- **Efficient computation, memory and cache use.** Nothing is computed twice when it can be kept.
  Nothing is redrawn that did not change. Nothing is held in memory that can be re-read cheaply.
  Every cache is keyed so that it can never serve a wrong value.
- **Every mathematical feature kept.** The converted program computes exactly what the old one
  computed, through the same functions, unless a change to the mathematics is decided on purpose.
- **Every interactive feature kept.** Every click, drag, key, toggle, readout and window of the old
  program exists in the new one, working the same way.
- **As smooth as possible to interact with.** The window answers a drag or a key at once, never
  freezes while files load, and never queues up stale redraws.
- **As large as possible.** The panels fill the window and the window fits the screen: no row
  leaves an empty band, the panel the user works in takes the largest size its shape allows,
  and the window sizes itself to its panels.

A conversion that looks better but computes something different, loses an interaction, or runs
slower has failed.

## 1. The rules

1. **Automatic layout: nothing is placed by hand.** Every panel and every control sits in a Qt
   layout. No figure-fraction rectangle, no pixel position, no "move this a bit to the left". The
   layout decides positions, so nothing can overlap at any window size.
2. **Everything shared is one function.** A feature used in two windows is one function or class
   that both call: a control group, a drawing, a pipeline step, a console phrase, a record format.
   A window never rebuilds logic that exists elsewhere.
3. **Controls are real widgets, not drawings.** Buttons, boxes and toggles are Qt widgets in the
   window's layout. They are never drawn inside the plot scene.
4. **Windows never scroll.** A panel that holds only text (a readout, a help line) scrolls inside
   itself. When the controls do not fit, their text shrinks a little, and then the window's
   minimum size is raised.
5. **One text-size rule for every window.** All text scales with the window's size relative to its
   design size, and no text is ever cut.
6. **Panel titles are as short as possible.** The details go to a scrolling information panel at
   the bottom of the window.
7. **Spatial panels keep their x:y ratio, and each row of panels shares one height.** A map or an
   image never stretches; curve panels and colour bars follow it.
8. **Number boxes instead of sliders.** A slider survives only for scrubbing through a list (files
   in time order). Every count or value is a number box whose digits step with the arrow keys.
9. **A window owns its state; controls only report.** A control tells its window that it changed.
   The window updates its state, recomputes through the shared functions, and redraws through
   the shared drawing functions.
10. **Comments name the rule they follow, and the README stays current.** Every layout block
    carries a comment such as `# automatic layout: the one rule for every window`.
11. **Panels know their shape; the window arranges itself.** Every plotted panel is a first-class
    citizen: the plots fill the window, a spatial panel keeps its x:y ratio, no plotted panel is
    ever shrunk to make room for another, and what the layout minimizes is the blank space and the
    controls' footprint. A spatial panel's cell tells the layout the height its ratio needs for
    the width it is given (Qt's height-for-width), so a row of panels is exactly as tall as its
    spatial panels ask, the curve panels and colour bars of that row fill that height, and no row
    leaves an empty band above or below its panels; given more room the panels grow until their
    widths fill the row, given less they shrink together. The window's shaped groups (a map with
    its boxes, a row of heatmaps, a block of energy maps) and its curve panels are arranged by one
    rule: every candidate arrangement (the groups in one row across the top with the curves below,
    one group at the left and the other over the curves at its right, one across the top and the
    other beside the curves below it, both stacked at the left with the curves at the right, the
    three side by side, the three stacked) is sized at the window's size, each group as large as
    its ratio allows in its region and the curves keeping a readable share, and the arrangement
    with the best score is taken: the smallest group's scale (its size against the size it would
    have alone in the window) less the blank fraction, so that no group is squeezed for another
    and blank space costs. A block of equal maps wraps into the rows that make the maps largest.
    The arrangement is chosen again on every resize, every rebuild and whenever a panel's shape
    changes, a crop included: cropping the map the user works in to a tall strip turns the layout
    tall, to a wide strip wide. The window opens filling the screen's available area when it is a
    working window, and in every case trims the blank strips at its edges that no panel reaches
    (shorter when the panels are limited by the width, narrower when they are limited by the
    height).

## 2. The architecture

Four layers, each with one job:

| Layer | Job | In sm4_Browser |
|---|---|---|
| Drawing and window layer | Figures, axes, artists, events, the window frame, the layout of panels | `qtplot.py` |
| Control set | Every control kind, every shared control group, the layout helpers | `controls.py` (tables in `widgets.py`) |
| Drawing of data | Every way data is drawn into an axes: heatmaps, curves, images, markers | `plots.py` |
| Windows | One class per window: its state, its layout, its handlers | `line.py`, `map.py`, ... |

The mathematics lives below all four, in plain functions on arrays (filters, calibration,
normalization, fitting), so that the windows, the checks and the command line use the same code.

**Why one plot scene per window.** All panels of a window are views in one pyqtgraph scene on one
canvas. One canvas repaints once for all panels, and pyqtgraph aligns linked views by their
geometry on screen, so linked panels line up exactly.

**Why controls are not in the scene.** Qt's documentation describes `QGraphicsProxyWidget`, the way
to put widgets into a graphics scene, as a convenience bridge that is not suited to
high-performance use. The controls are therefore ordinary widgets in a Qt layout installed on
the canvas's viewport, beside the panels' places.

**How a panel gets its place.** A panel is a *cell*: an empty, mouse-transparent widget in the
layout. Its axes reads the cell's rectangle, subtracts the room its tick labels and title need,
and places its view box there. When the layout moves or resizes the cell, the axes follows. The
layout never knows about plots, and the plots never know about positions.

**How a panel tells its shape.** A cell whose axes keeps a ratio (an equal aspect, a box aspect)
answers the layout's height-for-width question from that ratio plus the room of its ticks and
title, and tells the layout again whenever the ratio changes (new limits, new data). In an
aligned row (`fig.align_row`, a colour bar beside a panel) the room counted is the row's: the
boxes of a row lie in one band, below the tallest title and above the deepest x axis, so a
panel's shape is its box's in that band (a control box over a member counts: the band starts
under it). A row of cells (`PanelRow`) sizes its shaped cells from the row's height and gives the
free cells the rest; a column of such rows gives every row the height it asks for; an
`Arrangement` places the shaped groups and the curve panels by the score of rule 11. That is all
a window has to know: it names its shaped groups and its curve panels, and the arrangement, the
rows and the window follow the panels.

## 3. The anatomy of a window

### 3.1 Figure windows (one analysis window each)

From top to bottom:

1. **Toolbar.** Home, Pan, Zoom, Save, and the cursor's coordinates.
2. **The canvas, holding the window's one layout:**
   1. **Top row.** The window's own name or title, its mode controls, and the pipeline box at the
      right.
   2. **Panel grid.** Rows of cells (`PanelRow`s); a window with spatial groups and curve panels
      arranges them with `Arrangement` (rule 11: the groups as large as their shapes allow, the
      curves in the room left, the blank space minimized; `OrientedPanes` is the one-group case).
      A colour bar or ramp is a narrow cell beside its panel. A control that belongs to one panel
      sits directly under it (or over it: a tip-height box over its topography), centred on the
      panel's box. A box that rides along a cursor sits in a band beside the panel (`RideBand`; a
      `CropPanel` is a panel with its boxes above and its cursor bands).
   3. **Control strip.** Groups of controls in box frames, left to right in the old program's
      order. The strip wraps onto more rows in a narrow window.
   4. **Text panels.** Help text and readouts, each scrolling inside itself.
3. **Information panel.** Docked at the bottom, scrolling, selectable. It holds the details cut
   from the panel titles and the window's own readout lines.

### 3.2 The main window

The main window is a plain Qt main window. A splitter holds the main panes. Each pane is a column
of a toolbar, an embedded canvas (with its own layout of cells) and rows of control boxes. Its
controls are the same control set as the figure windows', made without a figure so that the main
window's font rules them.

### 3.3 The order of windows

- The main window opens every analysis window, one top-level window each, for the current file.
- A companion window (a second view of one window's data) closes when its parent closes, through
  the parent's close event.
- Windows that other windows can send data to are kept in a registry. A pop-up list ("add
  existing") lists them by index and hands the chosen index back.
- A window remembers nothing of another window's internals. It calls the other window's public
  methods, such as adding an entry or restoring a record.

## 4. Layout in practice

### 4.1 The skeleton of a window

The worked example, with the qtplot layer:

```python
fig = plt.figure(figsize=DESIGN)                 # DESIGN: the window's design size in inches

# Controls first: every one from the shared control set.
steps = controls.StepButtons(fig, '< before', 'before >', chevrons=True)
box_v = controls.make(fig, 'drift v', value='')
readout = controls.TextPanel(fig, 'readout', mono=True, lines=8, fixed=True)

# automatic layout: the one rule for every window (nothing placed by hand).
cell_a, cell_b = fig.cell(), fig.cell()
ax_a = fig.add_axes(cell=cell_a)
ax_b = fig.add_axes(cell=cell_b, sharex=ax_a, sharey=ax_a)
fig.align_row(ax_a, ax_b)                        # one row, one height
ax_a.set_anchor('S'); ax_b.set_anchor('S')       # each image at the foot of its cell

panels = QtWidgets.QGridLayout()
panels.addWidget(cell_a, 0, 0)
panels.addWidget(cell_b, 0, 1)
panels.addWidget(controls.UnderPanel(ax_a, steps), 1, 0)   # centred under the image, not the cell
panels.setColumnStretch(0, 1); panels.setColumnStretch(1, 1)
panels.setRowStretch(0, 1)

strip = controls.FlowLayout()                    # wraps in a narrow window
strip.addWidget(controls.box(box_v, title='velocity'))

root = QtWidgets.QVBoxLayout()
root.addLayout(panels, 1)
root.addLayout(strip)
root.addWidget(readout.qt)
fig.set_layout(root)                             # installs the layout on the canvas
plt.show(block=False)
```

Without the qtplot layer the same structure works with plain PySide6 and pyqtgraph: one
`pg.GraphicsView` per window, one `QLayout` set on its `viewport()`, empty transparent widgets as
cells, and a `ViewBox` placed at each cell's rectangle minus its axes' room, re-placed whenever the
cell moves or resizes.

### 4.2 The layout tools

| Need | Tool |
|---|---|
| A panel | `cell = fig.cell()`, then `fig.add_axes(cell=cell)` |
| A row of panels sized by their shapes | `row = qtplot.PanelRow()`, `row.addWidget(cell, stretch)` (a free panel's stretch is its share of the width); `qtplot.RowWidget(row)` makes a row one shaped item of a row or pane above it |
| The shaped groups and the curve panels arranged by rule 11 | `controls.Arrangement([group, group], free, free_share=0.22)`: the groups shaped widgets (a `CropPanel` or `PanelWithBox`, a `RowWidget`, a `MapBlock`, a `Titled` group), `free` the widget of the curve panels (a cell, a `FlipBox`); `.layout_name`, `.scales()`, `.blank_fraction()`, `.explain()`, `.rearrange()`, `.fit_delta()` |
| One main panel and the rest | `controls.OrientedPanes(main, rest, main_share=0.6)`: the one-group case; `.orientation`, `.rearrange()`, `.fit_delta()` |
| Curve panels beside or over each other by their room | `controls.FlipBox((cell, stretch), ...)`: a row when wider than tall, a column when taller |
| A block of equal maps that wraps | `controls.MapBlock(fig)`, `block.set_groups([(cells, bar_or_None), ...], shared_bar=None)`: the maps per row chosen for the largest maps that fit (`.cols`) |
| A label over a group | `controls.Titled(group, label)` |
| The panel the user crops | `controls.CropPanel(ax, cell, *boxes)`: the boxes above, `band_x` over and `band_y` beside the cell for the cursor boxes (`band.add(key, box)`, `band.set_pos(key, px)` from `ax.on_placed`); the cursors are `axvline` / `axhline` with a picker, the shade `axvspan` / `axhspan` |
| A control box over one panel | `controls.PanelWithBox(ax, cell, box)` (the box centred over the panel's box, the pair one shaped item; over a narrow `inner_w` cell the pair is a fixed-width item as wide as the wider of the two); `controls.AbovePanel(ax, *items)` alone for the centring |
| The window sized to its panels | `fig.fit_window()` (done at the first show and after every resize; call it after the panels were rebuilt): a window of shaped rows loses the height its rows do not need, or width when the rows are limited by the height; a window with an `Arrangement` or `OrientedPanes` loses the blank strips they report (`fit_delta`); `plt.figure(figsize=DESIGN, fill_screen=True)` opens a working window on the screen's available area |
| A colour bar or ramp | `fig.cell(inner_w=px)`: as wide as the bar plus its ticks and title |
| One height for a row of panels | `fig.align_row(*axes)` |
| A spatial panel's ratio | `ax.set_aspect('equal')`, `ax.set_box_aspect(r)` |
| Where a panel sits inside its cell | `ax.set_anchor('S')` and the other compass points |
| The window's title | `fig.set_title_cell(cell)`, then `fig.suptitle(...)` |
| Controls under one panel | `controls.UnderPanel(ax, *items)` |
| Boxes that ride on cursors | `controls.RideBand(orient)`, then `band.add(...)`, `band.set_pos(...)` |
| A group frame | `controls.box(*items, title=..., orient='h' or 'v')` |
| Plain rows and columns | `controls.row(...)`, `controls.column(...)`; `'stretch'` adds a stretch |
| A strip that wraps | `controls.FlowLayout(push_last=False)` |
| matplotlib-style subplots | `plt.subplots(...)`, `fig.add_subplot(...)`: an automatic grid of cells |
| Extra rows under an automatic grid | `fig.layout_root().addLayout(...)` |
| Export at the design size | `with fig.at_design_size(): ...` |

### 4.3 Keeping the old look

Each window keeps its old relative order and similar positions. Write the layout so that it reads
like the old screenshot from top to bottom and left to right. Only the exact pixel positions
change.

## 5. Wiring the controls

### 5.1 One interface for every control

Every control has the same interface:

| Member | Meaning |
|---|---|
| `qt` | Its one Qt widget, which goes in the layout |
| `value` | Its current value |
| `set(v)` | Sets the value without firing |
| `on_change(callback)` | The window's handler, called after a user change |
| `show(flag)`, `enable(flag)` | Visibility and whether it takes events |
| `set_short(flag)` | Its short label in a window taller than wide |

Kinds: `Check`, `Checks`, `Radio`, `Button`, `Text` (a number or text box), `Count` (a whole number
with limits, its frame optionally coloured), `Slider` (only for scrubbing a list), `TextPanel`
(scrolling read-only text) and `Label`. One table names every control once: its kind, label,
short label and font size. A control of a given name is therefore identical in every window.

### 5.2 Shared control groups

A group used in more than one window is one class: the filter box, the calibration box (with its
gate and lock rows), the step buttons, the stack toggles, the marker groups, the pop-up list of
open windows. A group gives its state as one dictionary, sets it without firing, and calls one
callback with the new state.

### 5.3 The loop of a change

```
user changes a control
  -> the control calls the window's handler with the new value or state
  -> the handler updates the window's state
  -> the window recomputes through the shared pipeline functions
  -> the window redraws through the shared drawing functions
  -> canvas.draw_idle(): this canvas repaints once, at the next idle moment
```

When a new state cannot be applied (a calibration that a file cannot hold), the handler restores
the old state, puts the control back with `set_state` (which does not fire), and says why on the
console.

### 5.4 Number boxes

Select one or more digits and press Up or Down. The selected digits rise or fall by one in the
place of the last selected digit, carrying as arithmetic does, and the same places stay selected.
With nothing selected, the digit left of the cursor steps. In a box holding two numbers, each
steps on its own. A count stops at its limits. Each step applies at once. Enter applies a typed
value; leaving the box applies it only after an edit.

### 5.5 Events on the canvas

Mouse and key events on the plots keep matplotlib's form, so the old handlers keep working:
`fig.canvas.mpl_connect('button_press_event', handler)`, with `event.inaxes`, `event.xdata` and
`event.ydata`, and pick, motion, release, scroll, key and close events. Draggable objects
(markers, cursors, path points) are helper classes that own their items and call back with the
new position.

### 5.6 Records and console lines

- **Records.** Every saved setting group has one pair of functions, one writing its record keys and
  one reading them. An older record without a key opens with that setting's default.
- **Console lines.** A phrase that several windows print is one function (for example the
  calibration box's state). A window adds only its own context around it.

## 6. Text, titles and the information panel

### 6.1 The text-size rule

- Every window has a design size (`DESIGN`, inches at 100 logical pixels per inch).
- The text zoom is `min(W / W0, H / H0)`, limited to 0.5x to 4x. Plot text, ticks, legends,
  controls, the toolbar icons and the information panel all follow it.
- When the controls do not fit the window, all of the window's control text shrinks together, to
  at most 75 percent of that zoom. If even that does not fit, the window's minimum size is raised
  to what the controls need.
- A larger text size is taken only with a few pixels to spare, so a window at the edge does not
  flip between two sizes on every layout pass.
- In a window taller than it is wide, every label with a short form shows it, and a title that is
  a file name shows only its date and time.

### 6.2 Titles

Write every panel title as `"Short name: details"`. Only the part before the first `": "`, `", "`,
double space or `" ["` is shown; the rest goes to the information panel, one line per panel. A
title must never carry a comma or colon inside its short name, so `"Cost S(a,b): ..."` and not
`"Cost S(a, b) ..."`. A file name is shown exactly as named, so it can be copied from the
information panel into a file search.

## 7. Efficiency, memory, cache and smoothness

### 7.1 Drawing

- **Create items once, then update their data.** A redraw sets new data on existing curve and
  image items; it does not clear and rebuild them.
- **Draw images through a lookup table.** A new colour scale or gamma rebuilds only the table, not
  the image.
- **Draw many curves of one colour as one item.**
- **Move instead of re-imaging.** When a correction is an affine map of an image's frame (a
  constant drift velocity is one), draw the image with that map. A velocity change then
  recomputes a 3 x 3 matrix, and no image is resampled or reprocessed. Check once that the map
  equals the per-pixel resampling.
- **Coalesce mouse motion.** A slow redraw must never queue up a backlog of motion events.
- **Redraw only the canvas that changed**, at the next idle moment (`draw_idle`).
- **Keep matplotlib off the screen.** Use it only to write publication files, rebuilt from the
  live figure at its design size.

### 7.2 Computation

- **Keep the mathematics in plain array functions,** one pipeline used by every window, and
  vectorize them.
- **Compute on change, never on paint.** A paint only draws what was computed.
- **Load on release, not on every slider step.** A scrubbing slider moves freely and loads its
  file when it is released.
- **Do slow work in the background.** Read files in worker processes, place results as they
  arrive through a timer, and keep the window usable meanwhile. Fall back to the main thread
  when workers cannot start.

### 7.3 Memory

- **Keep a least-recently-used set of loaded files** with a fixed limit, so stepping back re-reads
  nothing and memory stays bounded.
- **Keep a small display record** of every file after it was read once (a processed image in
  float32, its frame, its pixel clock), so a file released from memory is still drawn. Computation
  stays in float64.
- **Prefer views to copies** of large arrays, and release a window's arrays when it closes.

### 7.4 Caches on disk

- **Split the stores by origin.** Hand-made settings are written at once on every change and
  never invalidated. Values computed from the files alone are rebuilt whenever missing, so
  deleting that store loses nothing.
- **Key every computed entry by what it depends on:** the file's name, size and modification
  time, the format number of the entry, and every setting that changes it. A changed setting
  then selects another entry and never overwrites one.
- **Version every section** so that a new format rebuilds only that section.
- **Write computed stores at the end of a stage,** never while the user steps through files.

### 7.5 Keeping every feature

- **Inventory before converting.** List every axes, control, text readout, mouse and key handler,
  computation, saved file and window of the old program.
- **Prove the mathematics is unchanged.** Run the old and new code on the same inputs and compare
  the numbers, and keep synthetic data with known truth for the checks.
- **Prove the interactions are unchanged.** Script every click, drag and key of the inventory
  against the new windows.

## 8. The conversion recipe

1. **Inventory and screenshots.** Take every window at its design size and list everything in 7.5.
2. **Pull the mathematics out of the callbacks** into plain functions, and check them against the
   old outputs.
3. **Build the shared pieces first.** For each feature in more than one window, write one control
   group, one drawing function or one pipeline step. Search for repeated code before writing a
   window, and replace each repeat with one function.
4. **Convert one window,** the busiest one, as the worked example. Build its controls, cells and
   layout skeleton in the old order, then `fig.set_layout(root)`.
5. **Map each matplotlib piece:**

   | matplotlib | Real GUI |
   |---|---|
   | `fig.add_axes([l, b, w, h])` | `fig.add_axes(cell=fig.cell())` in a layout |
   | `plt.subplots(r, c)` | the same call, filling an automatic grid |
   | colour bar in a hand-placed `cax` | a narrow cell, `fig.cell(inner_w=...)` |
   | `widgets.Button` | `controls.Button` |
   | `widgets.CheckButtons` | `controls.Check` or `controls.Checks` |
   | `widgets.RadioButtons` | `controls.Radio` |
   | `widgets.TextBox` | `controls.Text` |
   | `widgets.Slider` for a value or count | `controls.Text` or `controls.Count` |
   | `widgets.Slider` for scrubbing a list | `controls.Slider`, loading on release |
   | `fig.text(x, y, ...)` readout | `controls.TextPanel`, or an information-panel line |
   | long `ax.set_title(...)` | `"Short: details"` |
   | `fig.suptitle(...)` | a title cell, `fig.set_title_cell(cell)` |
   | `mpl_connect` handlers | unchanged |

6. **Remove every hand-placement leftover.** Search for `rect=`, `add_axes([`, `set_position(`,
   `fig.text(` with fixed coordinates, pixel offsets and size constants.
7. **Wire the handlers** with the loop in 5.3.
8. **Shorten the titles** and send the details to the information panel.
9. **Do the performance pass** of section 7.
10. **Do the shape pass** of rule 11: every spatial panel's cell shaped, every row a `PanelRow`,
    the shaped groups and the curve panels in an `Arrangement` (a block of equal maps a `MapBlock`,
    curve panels a `FlipBox`), the boxes that belong to a panel over or under it, the panel the
    user crops a `CropPanel` with its cursors, the window fitted; render the window at several
    shapes and with a tall and a wide data set, read the arrangement's scales and blank fraction,
    and look for empty bands.
11. **Convert every other window and every standalone drawing** the same way.
12. **Verify** with section 11 before calling it done.
13. **Update the comments and the README.**

## 9. Reusing the sm4_Browser layer

When the old program is written against `matplotlib.pyplot`, the fastest route is to copy
`qtplot.py` (with `controls.py` and the tables of `widgets.py`) into the project and import it as
`plt`. Most pyplot code then runs unchanged on Qt, and only the layout and the controls are
rewritten:

- `qtplot.py` provides figures, axes, artists, the events, the toolbar, the information panel,
  colour maps by name, the text-size rule, the title rule, the cell layout and the publication
  export.
- `controls.py` provides the control set, the shared groups and the layout helpers.
- `widgets.py` holds the control table (`SPECS`, `LABELS`, `SHORT_LABELS`).

Adapt the control table and the shared groups to the new program, and keep the rules.

## 10. Pitfalls met in practice, with their fixes

| Symptom | Cause | Fix |
|---|---|---|
| Controls keep their font when the window resizes | A style sheet on a container stops the parent's font from reaching its children (seen with PySide6 6.11) | Set the font on every control widget explicitly at each zoom |
| A row of controls grabs spare width and spreads its labels | A line edit's default width policy is expanding, even at a fixed width | Give a fixed-width box a fixed width policy |
| A cropped panel snaps back to its full range | After clearing a panel that shares an axis, its auto-range drags the linked panels | On clear, keep the shared axis's range, as matplotlib's own `clear` does |
| Text size flips between two values on every resize | The fit test uses the zoom it just set, or the window grows during the fit | Compare with the zoom from before the fit, add slack, and fit again until the window size holds |
| Panels jitter by a pixel when data change | Tick-label widths change with the values | Keep the room for decorations grow-only at a given zoom |
| A stand-alone window rejects its own settings | A module run as a script is loaded twice, so its classes exist twice | Pass plain data, such as dictionaries, between the two copies |
| A long title runs into the next panel | No separator before the details | Write titles as `"Short: details"` |
| The window freezes while files load | Reading on the main thread | Worker processes with a timer placing results |
| Dragging lags behind the mouse | Every motion event redraws | Coalesce motion and redraw on idle |
| Empty bands above and below a row of square panels | The grid gives the row a share of the height by stretch; the panels keep their ratio inside a cell too tall for their width | The cells report height-for-width and the row is a `PanelRow` (rule 11) |
| A tall unit cell's map is a small panel in a wide window | The map sits in a row across the window; its height is bounded by the row | `Arrangement`: the map stands at the left with the window's full height when it is taller than wide |
| A curve panel shrinks to the height of the small map beside it | Rule 7 applied to a row the map does not dominate | The map a shaped group of the `Arrangement`; the curve panels fill the room left |
| The heatmaps come out as small as a control box while one map fills the window | One panel was maximized and the others left the remainder | Every plotted panel is a group of the `Arrangement`; the score (the smallest group's scale less the blank) keeps them all large |
| Five wide maps in one row are tiny in a wide, short region | A fixed grid of one row per partition | `MapBlock`: the maps wrap into the rows that make them largest |
| A cropped map keeps the old layout | The crop changed the limits but not the shape the layout saw | `set_xlim` / `set_ylim` announce the new shape; the arrangement follows (a tall crop, a tall layout) |

## 11. Verifying a conversion

1. **Render every window** at its design size, at half and double size, and taller than wide.
   Look for overlap, cut text and empty frames.
   Measure every row's empty band (the row's height less the tallest panel box in it, beyond the
   ticks and title) and the window's fit (its natural height against its height): both within a
   few pixels (rule 11). For a window with shaped groups, check with a tall and a wide synthetic
   data set that the arrangement follows the shapes, that its blank fraction is small (the
   simulator's check asks for less than 15 %) and that no group is below about half of its size
   alone (`Arrangement.scales()`), that every box sits above the panel it belongs to, and that a
   crop of the working panel changes the arrangement.
2. **Check no text is cut:** every label's text width fits its widget at each size.
3. **Check overlap:** no two control rectangles intersect, and no control enters a panel's box.
4. **Check the numbers** against the old program on the same files.
5. **Script the interactions** of the inventory and check their effects.
6. **Scan for repeated logic.** List function bodies that are identical in more than one place and
   make each one function. A short scan with Python's `ast` module, comparing `ast.dump` of each
   function body without its docstring, finds them.
7. **Measure smoothness:** time a full repaint and the response to a drag or toggle on a large data
   set. In the worked example, on a large synthetic directory rendered offscreen, a full paint of
   the main window took about 10 ms and a drift change about 40 ms.
8. **Run the whole check suite** after every change, offscreen (`QT_QPA_PLATFORM=offscreen`).

## 12. Sources

- Qt documentation, QGraphicsProxyWidget: https://doc.qt.io/qt-6.5/qgraphicsproxywidget.html
- Qt documentation, style sheets: https://doc.qt.io/qtforpython-5/overviews/stylesheet.html
- matplotlib's `Axes.clear` and `sharex` / `sharey` (matplotlib 3.11 source, `lib/matplotlib/axes/_base.py`)
- pyqtgraph's `ViewBox.linkedViewChanged` (pyqtgraph 0.14 source), which aligns linked views by
  their geometry on screen
- Qt's height-for-width in layouts (`QLayoutItem::hasHeightForWidth`, `QGridLayout`,
  `QBoxLayout`, Qt 6 source), checked by a prototype: a grid row holding a square cell becomes as
  tall as the cell's column is wide, the free cell beside it fills that height, the row below
  takes the rest
