# Follow-up after part 3: the implementation brief

Sources (a later one overrides an earlier one where they differ): the plans (parts 1 to 3 and the addendum
`handoff_part3_addendum_map_shading_contour.md`), `RESPONSE_PART1_PART3.md`, `PART3_BRIEF.md`, `RESPONSE_PARTS2_3.md`
(the owner's answers O1-O18, K1-K11), `PROGRESS.md` rulings 11 and 12 (Claude's readings for the module), and the
"Settled under rules already given" list of `PART3_OPEN2.md`. `PART3_OPEN2.md` N1-N4 wait for the owner: leave what
they touch as it is today (Line's layout; export_vector's resolution; TriContourSet.set_cmap's export record; README
rows for cmap_shading.py and _resolve.py).

Rules for every agent: implement exactly this; where the code does not match, or a step needs a change nothing names,
stop and report a blocker with options instead of choosing. Read files and run `python -m py_compile` only (no tests,
checks, scripts or windows). One agent edits a given file at a time. No handoff ids in code or comments. Docstrings of
defs written or changed in simulator.py, ldos_decomp.py and cmap_shading.py start with "Physics:" or "Machinery:".

## The simulator (simulator.py, ldos_decomp.py)

- **O3:** move `_display_indices` unchanged into `Unified_STM_Simulator`.
- **O7:** the Decay-off weight becomes a def in the physics block, called by both of `_compute_channel`'s Decay-off
  branches with the same expressions (no change to the computation).
- **O1:** the Norm factors are computed once per data change on the GPU with `quadrature.simpson`, kept as NumPy
  (float32, as computed) beside the drag cache; during a Line / Path drag or a `mark_dynamic` pick `_draw_spectrum`
  divides by them (no GPU work in the drag).
- **O2 (the owner's rule):** Map's energy markers stay where they last were (dragged or typed) and Map's maps stay as
  last plotted, through every toggle change and every view change; returning to Map shows them from what is kept,
  with no recompute. The drag stays as part 2 has it (on release only the dragged energy is computed, at every grid
  point, with the same call and tip heights; then the X3 re-sort; only changed panels redrawn). Claude's reading:
  only an edit of E Min, E Max or E Pts rebuilds the Map energies from the typed ends and recomputes; a change that
  alters the computed LDOS (Decay, a broadening σ, LDOS h, a rerun) recomputes at the kept energies; display-only
  toggles (Norm, Feenstra, Atoms, LDOS γ, Mag, Cells) only redraw. Map's energies, data and caches are kept apart from
  Line / Path's. **The decomp:** a view change still resets the level, marker and element to 0; every decomp step and
  the reset show at once in every view from the stored per-orbital LDOS (`preserve_orbitals=True`), nothing
  recomputed. If a panel rebuild for a changed number of parts keeps a step from showing at once, stop and report.
- **K7 (d):** with RUN off, a decomp step (a left click on a curve or entry, a right click, the Back button) redraws
  the spectrum, the legend panel, the maps and the heatmaps from the last run's data, nothing computed. One click,
  one step, one redraw. **And** (settled): with RUN off, a Line / Path marker re-sort redraws the level-0 curves, the
  spectrum title and the legend in place from the last run's data, so artist i shows marker i.
- **K1 (a):** level 3 is one entry per displayed column, named by its base orbital (s, py, pz, ...), the spin sum, or
  the difference under Mag (as `_combine_ldos_display` combines the channels), in the spectrum, the maps and the
  heatmaps alike; no `_up` / `_down` entries; the legend's width labels (`all_labels`) follow.
- **K2 (a):** Map's legend panel: right of the spectrum, through MapBlock's new side item (qtgui stage), at that row's
  height, counted in the block's width (the legend's fixed width); the spectrum keeps a map's size.
- **K8 (a):** the legend stack calls `set_navigate(False)` (made real in the qtgui stage).
- **The addendum's integration:** the colour-map sets come from `cmap_shading.py` (`SETS`, `cmap_spec` / `make_cmap`;
  simulator.py's own set table removed; `MAP_CMAP` kept as default 1's Map entry built from the module's list); the
  Contour / Shading box in Map only, untitled like the 'cmap norm' box, its Height / Relief radio named 'shading by'
  (`cmap_shading.build_box`, with the figure and the controls module); Contour on: the contour slice as today; Contour
  off: no traces on the energy maps and the topo colour bar's level line hidden; at most one on, both may be off;
  start Contour on, Shading off, Height. Shading on: the energy maps are drawn as an RGBA mesh (qtplot's `pcolormesh`
  with an (M, N, 4) C, added in the qtgui stage), one colour per unique grid point (the duplicate edge left out),
  each point's quad centred on it, tiled by the cell offsets; the colours: the shading map ('CET-I1', Mag signed
  'CET-D11', Mag |m| 'CET-I3') through the maps' own norm, then the module's shading (Height: the LDOS topo height;
  Relief: relief.m with its defaults, `periodic=True`, `panel_turn` 90 when the cell is drawn turned, else 0, K11
  (a)), with the L* stretch over the shown area's height range as the contour slice's level uses it; the colour bars
  show the shading map unshaded; Shading off: the set's maps and the filled contours come back. Display only: no LDOS
  recomputed, nothing rearranged; only the energy maps are redrawn. The box's settings reach the info panel through
  the settings section.
- **Texts (K9's rule):** `_draw_spectrum`'s docstring (no legends, no atom levels); the class docstring's list of the
  left column (the cmap set choice) and of Map's boxes (the Contour / Shading box); any other text in these files the
  changes make false, stale words only.

## The package qtgui (stm_from_dft/qtgui; copied over matcher/qtgui afterwards)

- **controls.py, K2 (a):** `MapBlock` takes a side item for its extra cell: drawn right of the extra cell, at that
  row's height, counted in the block's width and placement, the colour bars kept clear of it.
- **qtplot.py:** K10 (a): `QuadMesh` and `Axes.pcolormesh` take an (M, N, 3) RGB or (M, N, 4) RGBA C (0-1 floats or
  0-255 ints, as matplotlib's), painted as given, with no norm or colour map; K8 (a): `Axes.set_navigate(False)` keeps
  an axes out of Pan and Zoom (the wheel, drags and Home leave it alone), as matplotlib's; K6 (b): a window line for
  each panel whose view limits were changed with the toolbar's Pan or Zoom since its last Home or since the window
  last set its limits itself, giving the panel (its title, else its place) and its x and y limits, removed on Home or
  when the window sets the limits again; K4 (a): the 'vector' savefig route passes `facecolor` and `transparent` on to
  mplexport again, as the root does; K9's rule, stale words only: the Save button's tooltip, the FigureManager and
  Figure docstrings, "when installed" in the module docstring's Colour maps paragraph, PanelCell.bind's comment.
- **mplexport.py:** K10 (a): an RGBA `QuadMesh` is rebuilt with matplotlib's `pcolormesh` taking the RGBA array; the
  legend title written bold, as on screen.
- **__init__.py, K9:** the mplexport line: mplexport writes the PDF, EPS and editable SVG; the window image writes the
  PNG and TIFF (an exception to ruling 1, the owner's answer).

## The root qtgui (qtgui/)

- **qtplot.py:** K6 (b) as in the package copy; K9's rule for the same texts as the package copy, so both read the
  same: the root had taken sm4_browser's wording for the tooltip, the FigureManager docstring and the Figure docstring
  sentence; restore those three to their adf1e25 text and change only their stale words, as in the package copy;
  "when installed".
- **mplexport.py:** the legend title bold.
- **__init__.py:** K9's mplexport line, as the package copy's.

## The other files

- **sim_wrapper.py:** O14 as listed in `PART2_OPEN.md` (the class docstring clause, `launch`'s docstring, the
  σ-commit section comment, `_on_sigma`'s reason; `commit_sigma`'s comment and print kept); K3 (a): in
  `send_to_decomposition`'s Line / Path branch, each marker's kept colour (the simulator's per-marker colour list)
  in place of `m_colors[i % len(m_colors)]`, a second exception to rule 4.
- **run_stm_decomp.py, O15:** `reuse_cache=True  # the run directory's topo caches`; `use_decay=True` with the
  tilted-barrier comment as run_DFT_STM.py now has it.
- **run_DFT_STM.py, O16:** "slider" becomes "box" in the two comments.
- **K5 (a):** run_wfn_plane_slice.py and run_vacupot_plotter.py save after `plt.show(block=False)` (only the save
  moves; their settings and `dpi` arguments stay); their save comments, stale words only (K9).
- **READMEs:** stm README: O10 (b) (the simulator.py row split into the physics and the machinery), K9 (the click
  text, the layout text on the curve panels, the module table's `ldos_decomp.py` row, the save lines, the 'cmap norm'
  row's `cmap_map`); mo_ldos_decomp, MOPmatcher_mem and root READMEs: their save lines (K9); PROCAR_tools README: O17
  (b) (which checks run on every band, which are spot-checked); topo_decomp_window.py's export comment (K9).
- **convergence/forces.py, K6 (b):** a window line with its 3D view (rotation and zoom: elevation, azimuth, and the
  axis limits), updated when they change.
