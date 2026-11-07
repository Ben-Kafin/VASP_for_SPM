# Part 3: the owner's questions (after the pre-flight)

The pre-flight raised about 80 points. Most are settled by the plan, the response document
(`RESPONSE_PART1_PART3.md`) or the owner's earlier rulings, or are mechanics; those are recorded in
`PART3_BRIEF.md` and are being implemented. These nine are not: each is something in the plan that is wrong or
cannot work as written, or a behaviour the plan and the rulings leave open. Each has a recommendation (R).
Until answered, the work they touch is left as it is today (stated per item); everything else goes ahead.

**K1. Level 3 in a spin-polarised run: the plan's rule rests on code that crashes (section 2, "Which orbitals").**
The plan: "In a spin-polarized run the `_up` and `_down` columns are separate entries, as in the maps". In a
COLLINEAR_POL run the simulator's `self.orbitals` has 18 names (`s_up, s_down, py_up, ...`; `mapping_pol[9]`) but
the data it indexes (`dos_up_gpu`, the LDOS arrays) has 9 columns, the spin channels already split by the parser
(`site_dos[:, :, 0::2]` and `1::2`) and summed (or subtracted, Mag) in the LDOS. So `_get_partitions`' level 3
labels column 1 (py) as `s_down` and raises IndexError at the tenth name (CuPy bounds-checks a scalar index). Today
the maps' level 3, the heatmaps' level 3 and the spectrum's level 4 all fail in a POL run. UNPOL and NCL are fine.
- (a) One entry per displayed column, named by its base orbital (s, py, pz, ...), the spin sum (or the
  difference under Mag), in the spectrum, the maps and the heatmaps alike; no `_up`/`_down` entries.
- (b) Separate `_up` and `_down` entries from the two channels kept apart (a new uncombined path through part 2's
  caches; Mag needs its own rule).
- (c) Leave it (it keeps crashing in POL).

R: (a). Until answered: level 3 is built exactly as the maps build it today (works in UNPOL and NCL).

**K2. Map: the legend panel needs a change to MapBlock (section 5, "in every view").** In Map the spectrum is
MapBlock's extra cell, one map's size, in the last row's free slot (part 1 5.3). MapBlock lays out that one cell
and has no place for a second item beside it.
- (a) MapBlock takes a side item for its extra cell: the legend panel right of the spectrum at that row's
  height, counted in the block's width; the spectrum keeps a map's size.
- (b) The spectrum and its legend share one map's width (the spectrum narrower than a map).
- (c) The spectrum and legend leave the block (changes Map's layout, against part 1 5.3).

R: (a). Until answered: the legend panel is built in Single Point, Line and Path; Map's waits.

**K3. The decomposition window's marker colours after a Line / Path re-sort (2a vs rule 4).**
`sim_wrapper.send_to_decomposition` colours the Line / Path markers it sends by index
(`sim.m_colors[i % len(sim.m_colors)]`, two lines). With 2a each marker keeps its colour, so after a re-sort the
decomposition window would show a marker in a different colour from the simulator.
- (a) A two-line change in sim_wrapper (its Line / Path branch takes each marker's kept colour), a second
  exception to rule 4.
- (b) No change; the decomposition window shows index colours.

R: (a). Until answered: sim_wrapper keeps index colours.

**K4. Transparent exports (section 6 vs topo_decomp_window).** sm4_browser's vector save forces white
(`mfig.patch.set_facecolor('white')`, every axes white, `savefig(facecolor='white')`). This repository's mplexport
takes the figure's and each axes' own background, and topo_decomp_window sets its figure patch to "none" on
purpose ("the publication export ... carries NO opaque background rectangle beneath the color layers"). Taken as
it is, sm4_browser's code would make that export opaque.
- (a) Keep this repository's background handling inside sm4_browser's structure.
- (b) sm4_browser's forced white.

R: (a). Until answered: the background handling stays as it is here.

**K5. Two scripts save before the window is shown (section 6 replaces the save routes).**
`run_wfn_plane_slice.py` line 586 (`fig.savefig(png, dpi=300)`) and `run_vacupot_plotter.py` line 101
(`fig.savefig(SAVE_PATH, dpi=SAVE_DPI)`) save from a script. Today these go through the design-size rebuild,
which lays the window out first. Under sm4_browser's savefig a PNG becomes the 720 dpi window image of a window
that is not on screen yet (not laid out; the dpi is ignored), and a vector file would read an unplaced layout.
- (a) Both save after `plt.show(block=False)`, sm4_browser's own pattern (its tools/rebuild_panels.py); their
  PNG is then the 720 dpi window image.
- (b) Leave them (their files come out wrong).

R: (a). Until answered: both scripts are left as they are.

**K6. Do mouse-made view changes count as settings (ruling 8, Q6)?** Your Q6 rule is "every setting the window
makes with the mouse". The toolbar's Pan and Zoom change each panel's view limits with the mouse, and
convergence/forces.py's 3D panel is rotated and zoomed with the mouse. sm4_browser lists neither.
- (a) No: they are navigation, not settings.
- (b) Yes: a window line with each panel's limits while panned or zoomed, and forces.py's 3D view.

R: (a). Until answered: not listed.

**K7. Spectrum and legend clicks while RUN is off (section 2, "one click, one step ... redrawn once").** With RUN
off, `_update_all` does not draw, so today a click changes the level with no redraw, and the next click acts on
curves of another level (a level-2 curve clicked while the state says level 1 gives level 2 with the wrong
selection).
- (a) The spectrum, legend and Back clicks do nothing while RUN is off.
- (b) As today.
- (c) Redraw the spectrum and legend from the caches while stopped (the maps would not follow).

R: (a). Until answered: as today.

**K8. Pan / Zoom over the legend stack.** While the toolbar's Pan or Zoom is on, every axes takes the mouse, so
the wheel would zoom the legend stack and a drag would pan it. qtplot's `Axes.set_navigate` exists but does
nothing.
- (a) Make `set_navigate(False)` keep an axes out of Pan and Zoom (as in matplotlib) and call it on the stack.
  Side effect: the topo colour bar, which already calls `set_navigate(False)`, stops panning too.
- (b) The simulator puts the stack's limits back after any pan or zoom (a moment of panned panel).
- (c) Leave it.

R: (a). Until answered: left as it is.

**K9. Texts that become false (not named by the plan).** After part 3:
- the stm README's click text ("a click on a curve ... descends a level (partition, atom types, atoms, orbitals)
  and a click beside the curves ascends"), its layout text on the curve panels, and its module table (no row for
  `ldos_decomp.py`);
- `_draw_spectrum`'s docstring (legends, the atom levels);
- the save texts: `qtgui/__init__.py`'s mplexport line "(PDF, EPS, SVG, PNG, TIFF)" in all three copies (your
  ruling 1 froze that module list), the stm, mo_ldos_decomp, MOPmatcher_mem and root READMEs' save lines,
  `run_wfn_plane_slice.py` and `run_vacupot_plotter.py`'s save comments, topo_decomp_window's export comment.

R: update each in the same terms, changing only the stale words, `__init__.py`'s line included as an exception to
ruling 1. Until answered: unchanged.
