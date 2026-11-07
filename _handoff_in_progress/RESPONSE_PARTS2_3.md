# Response document: parts 2 and 3

The owner's answers to `PART2_OPEN.md` (O1 to O18), `PART3_OPEN.md` (K1 to K9) and the later K10 and K11. Each answer gives its basis. Nothing here waits on the owner. Points marked "Claude's reading" are readings of the owner's words, recorded as such.

## Part 2 (`PART2_OPEN.md`)

**O1, Norm during a Line / Path marker drag: (a).**
- The Norm factors are computed once per data change on the GPU with `quadrature.simpson`, and kept as NumPy beside the drag cache. During a drag, `_draw_spectrum` divides by them.
- **Basis:** this keeps all three rules at once:
  - every Norm integral comes from `quadrature.simpson` on the GPU (B19, 7.3);
  - no GPU work happens during a drag (A8);
  - the factors keep the float32 they were computed in, NumPy copies included (the owner's precision rule).

**O2, Map after an energy-marker drag: the owner's rule, neither (a) nor (b).**
- **The owner:** "wherever the energy markers were last and the map heatmaps that were last plotted, they should be what is returned to upon togle changes or even map to line to single point to path back to map changes".
  - Map's energy markers stay where they last were, dragged or typed. Map's maps stay as last plotted.
  - Both are kept through every toggle change and every view change.
  - Returning to Map shows them from what is kept, with no recompute.
- **The drag itself** stays as part 2 has it in `_on_rel`: on release, only the dragged energy is computed, at every grid point, with the same call and the same tip heights (the LDOS topo) as the full Map computation. Then the maps re-sort, and only the panels whose map changed are redrawn.
- **Claude's reading, recorded as such:**
  - only an edit of E Min, E Max or E Pts rebuilds the Map energies from the typed ends and recomputes the maps;
  - a change that alters the computed LDOS (Decay, a broadening σ, LDOS h or a rerun) recomputes the maps at the kept energies, without rebuilding them;
  - display-only toggles (Norm, Feenstra, Atoms, LDOS γ, Mag, Cells) only redraw from the kept data.
- **In the code today:**
  - `_update_all` rebuilds the Map energies on every full refresh (branch line 2633);
  - its recompute test includes a view change (`cached_mode != self.mode` in `needs_ldos`);
  - Line and Path write some of the same caches.

  So Map's energies, data and caches are kept apart, and are not rebuilt or recomputed for anything but the changes above.
- **The decomp (the owner):** "decomp should reset but should load instanly no matter the changes since all orbita dos are already calculated as ldos", and "I mean all decomp in any window".
  - A view change still resets the decomp level, the active marker and the active element to 0 (part 3, "What stays as it is").
  - Every decomp step, and the reset, is shown at once in every view of the simulator (Single Point, Line, Path, Map). It comes from the per-orbital LDOS already computed (`preserve_orbitals=True`), with no LDOS recomputed. Today a decomp step already recomputes nothing (`needs_ldos` does not test the level). The time it takes is the rebuild of the Map block or the Line / Path heatmap row when the number of parts changes.
  - **Claude's reading:** "any window" means every view of the simulator, where the click-through decomp lives. If the panel rebuild for a changed number of parts keeps a step from showing at once, stop and report it (rule 1).

**O3, `_display_indices` on the base class: (a).** Move it unchanged into `Unified_STM_Simulator`.
- **Basis:** (b) would duplicate `_snap`'s logic, which the owner's rule forbids.

**O4, every sample at 0 V: (a), accept.**
- **Basis:** the owner's rule: he will never give a bias within one DOS step of 0, so no check for it.

**O5, `_snap`'s exact tie at 0.0: keep.**

**O6, X3's drag cache in Map: accept.**

**O7, the Decay-off weight: (a).** A def in the physics block, called by both branches with the same expressions.
- **Basis:** ruling 5 says the physics block holds the weight.

**O8, defs with no docstring: (a), leave them.**

**O9, tags outside `simulator.py`, `feenstra.py` and `quadrature.py`: (a), leave.**

**O10, the stm README's `simulator.py` row: (b).** Split it into the physics and the machinery.
- **Basis:** ruling 5: the README separates the model from how it is computed.

**O11, the Decay row's Chen references: (a), keep both.**
- **Basis:** ruling 3 asks for Chen, and these are Chen.

**O12, the bottom-strip sentence: (a), keep the tie clause.**

**O13, scipy in the stm README's Dependencies line: keep it.**
- The claim that nothing in `stm_from_dft_opt` imports SciPy is not right. The package's own `qtgui` copy imports it in `tricontourf`, the function that draws every topo and energy map:
  - `from scipy.interpolate import LinearNDInterpolator`, `qtgui/qtplot.py` line 2968 on the branch;
  - the matcher's copy is byte-identical.
- Without SciPy that call fails and prints "the points cannot be triangulated ...; nothing drawn", so the topos and energy maps would draw nothing.
- **What changes:** nothing in the Dependencies line.

**O14, `sim_wrapper.py` texts after X1: as listed.**

**O15, `run_stm_decomp.py` comments: change both.**

**O16, `run_DFT_STM.py` comments: "box" in both.**

**O17, the `procar_dos.py` checks: (b), in the README only.** Say which checks run on every band and which are spot-checked.
- **Basis:** X2 keeps `procar_dos.py` unchanged.

**O18, historical files: leave them.**

**For information, noted:**
- The float32 `grid_z` heights fix falls under part 2's section 9 (every CuPy array float32).
- The commit history is left as it is.
- Part 3's rule 4 is moot after X1.

## Part 3 (`PART3_OPEN.md`)

**K1, level 3 in a spin-polarised run: (a).** One entry per displayed column, named by its base orbital (s, py, pz, ...). Each entry is the spin sum, or the difference under Mag. This applies in the spectrum, the maps and the heatmaps alike, with no `_up` / `_down` entries.
- **Basis:** the owner's layers are the orbital types of one atom type, and spin channels are not orbital types. Each entry is combined as the LDOS display already combines the channels (`_combine_ldos_display`).
- **What it replaces:** the plan's sentence "In a spin-polarized run the `_up` and `_down` columns are separate entries, as in the maps" was Claude's. It was taken from the orbital names without checking the column count, and is withdrawn. In a COLLINEAR_POL run `self.orbitals` has 18 names, but the data has 9 columns per spin.

**K2, the Map legend panel and MapBlock: (a).** MapBlock takes a side item for its extra cell: the legend panel right of the spectrum, at that row's height, counted in the block's width. The spectrum keeps a map's size.
- **Basis:** (a) is the only option that keeps both rules:
  - part 1 §5.3: the spectrum is one map's size;
  - part 3 §5: the legend panel is to the right of the spectrum, in the same row and sub-section.
- The legend panel's fixed width (response document, Q3) is what the block counts.

**K3, the decomposition window's marker colours after a Line / Path re-sort: (a).** A two-line change in `sim_wrapper.send_to_decomposition`: its Line / Path branch takes each marker's kept colour. This is a second exception to rule 4.
- **Basis:**
  - 2a: each Line and Path marker keeps its colour wherever it moves;
  - rule 4 is moot after X1.

**K4, transparent exports: (a), the owner's answer.** Keep this repository's background handling inside sm4_browser's save structure:
- the figure's and each axes' own background go into the file (`mplexport.py` lines 126 and 166 here);
- sm4_browser's forced white is not taken (its `mplexport.py` lines 113, 160 and 537).

So `topo_decomp_window`'s transparent export (its figure background set to none, lines 139 to 143) stays transparent.

**K5, the two scripts that save before the window is shown: (a).** `run_wfn_plane_slice.py` and `run_vacupot_plotter.py` save after `plt.show(block=False)`, so their PNG is the 720 dpi window image.
- **Basis:** the owner's 720 dpi rule: an image is the window exactly as on screen, at 720 dpi.
- Only the save moves. The scripts' settings stay as they are. Their `dpi` arguments stay and are ignored, as sm4_browser's `savefig` ignores them.

**K6, mouse-made view changes as settings: (b), the owner's answer.** Every window's info panel gets:
- a window line with each panel's view limits while that panel is panned or zoomed;
- `convergence/forces.py`'s 3D view (its rotation and zoom) as a window line.

**K7, spectrum and legend clicks while RUN is off: (d), the owner's answer.** This option was Claude's, following the owner's rule that every decomp step shows at once from the stored orbital LDOS.
- With RUN off, a decomp step redraws the spectrum, the legend panel, the maps and the heatmaps from the data of the last run, with nothing computed. A step is a left click on a curve or entry, a right click, or the Back button.
- The screen already shows that last run, so the redraw stays consistent with it.
- "One click, one step, one redraw" holds with RUN off as well.

**K8, Pan / Zoom over the legend stack: (a).** `set_navigate(False)` keeps an axes out of Pan and Zoom, as in matplotlib, and the legend stack calls it.
- **Basis:** qtplot copies matplotlib's behaviour. The topo colour bar already calls `set_navigate(False)`, so the side effect is what that call asks for.

**K9, texts that become false: update each in the same terms, changing only the stale words.**
- **The `qtgui/__init__.py` mplexport line (the owner's answer: (a)):** correct it in all three copies, as an exception to ruling 1. After part 3, `mplexport` writes only the PDF, EPS and editable SVG, and the window image writes the PNG and TIFF.

**K10, the shaded maps need RGBA in `QuadMesh`: (a), the owner's answer.**
- `qtplot`'s `QuadMesh` (`pcolormesh`) accepts an (M, N, 3) RGB or (M, N, 4) RGBA array, as matplotlib's `pcolormesh` does, and `mplexport` passes it on to matplotlib.
- This is done in the two package `qtgui` copies only.
- It keeps the shading in the vector saves, as the shading addendum requires.
- The addendum's sentence that the quad item "already takes one RGBA colour per quad (what `pcolormesh` uses)" holds for `QuadItem` only. That sentence was Claude's.

**K11, the light direction on a turned map: (a), the owner's answer.**
- The light stays where it is on screen: from the upper right as the map is shown, turning with the map.
- **Basis:** a light from the bottom of a shaded relief map produces the relief inversion effect (bumps read as dents). Fixed on screen, the light stays above in every orientation. `relief.m` treats the image as it is shown.
