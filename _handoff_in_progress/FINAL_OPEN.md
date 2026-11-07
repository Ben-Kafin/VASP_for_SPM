# After the follow-up: the owner's questions (F1 to F3) and what is settled

The follow-up workflow (wf_a3502bdb-787) is done: every stage implemented, each file group reviewed twice and fixed,
one cross-module review. Its changes: `FOLLOWUP_CHANGES.md`. Three points need the owner; everything else it raised
is settled below under rules already given, and goes into the last workflow with N1 to N4.

## For the owner

**F1. With RUN off, which geometry does a redraw from the last run use?** K7 (d) redraws the spectrum, legend, maps
and heatmaps from the last run's data when a decomp step is made with RUN off. If something was moved while stopped,
the redraw today pairs the old data with the new geometry: a Line end or Path node moved gives the old line's data
along the new line's distances (heatmaps, stripe, Line Topo); a Single Point / Map marker moved gives the old
position's spectrum under the new position's title. (Display toggles changed while stopped, such as Norm, Feenstra,
Mag, LDOS γ, Atoms, Cells, are display-only under O2, so applying them to the kept data is consistent.)
- (a) As now: the current geometry with the last run's data.
- (b) The last run's geometry with its data (the titles, distances and positions of the last run), the current
  display toggles applied.
- (c) No RUN-off redraw once a marker, Line end or Path node has moved since the last run.

R: (b): the data and their labels then always belong together.

**F2. In Map, the Map Topo loses its title, atoms and unit cell after its LDOS topo is computed (a bug found, older
than this work).** The full refresh draws the Map Topo with its title, atoms and unit cell; Map's LDOS-topo branch
then calls `ax_map.clear()` and redraws only the LDOS topo, the markers and the crop. So after the first entry into
Map, an LDOS h or Topo h rerun, a broadening, Decay or a bias change, the Map Topo shows no title, atoms or unit cell
until a later full refresh that does not recompute the topo.
- (a) The LDOS-topo branch also redraws what the full refresh drew on the Map Topo (title, atoms, unit cell).
- (b) Leave it.

R: (a).

**F3. Turned panels are misplaced in the PDF / EPS / editable SVG (a bug found, older than this work).** mplexport's
`_turned` turns a panel's artists for a turned axes (the unit-cell panels when the cell is drawn turned) by setting
only the offset transform. In matplotlib that moves only a collection's offsets; a collection drawn without offsets
(a pcolormesh QuadMesh, the filled-contour sets, line and polygon collections) keeps its own transform, so in the
file it is neither turned nor in its place. This affects every turned unit-cell panel's filled contours and meshes in
the vector saves (the shaded RGBA maps included), in all three qtgui copies. The screen and the window images are
right.
- (a) In `_turned`, the offset transform only for a scatter, and the artist's transform for every other collection,
  in all three copies.
- (b) Leave it.

R: (a).

## Settled under rules already given (done in the last workflow)

- **O2, Map's caches kept apart (a gap the reviews found):** Map wrote the shared `cached_eg` (the spectrum energies
  Line, Path and Single Point use), so `sim_wrapper`'s Line / Path send could pair Line's curves with Map's energies.
  Map stops writing `cached_eg`: it draws its spectrum against its own energies, and the marker spectra keep their
  energies with them, which `sim_wrapper`'s Single Point / Map branch reads (rule 4 is moot, the owner's note).
- **O2, returning to Map:** with RUN off too, a view change into Map draws the kept maps, energy markers and spectrum
  ("they should be what is returned to ... map to line to single point to path back to map").
- **K7 (d), no step without its redraw:** with RUN off and the last run's data dropped while stopped (Decay or a
  broadening toggled, a marker added), a decomp click does nothing; the old curves cannot then be read at another
  level (the reviewers found a click sequence that set the element to a marker index).
- **K2, the Map legend's width:** after a text-zoom change, the window is arranged again (a resize-driven change, part 1
  5.2 trigger 3), so the block counts the legend's new width at once.
- **The 'shading by' radio:** named 'shading by' (lower case, like 'cmap norm' and 'cmap set'), as ruling 11 recorded.
- **forces.py's 3D view line:** also gives the roll, which matplotlib's default arcball rotation changes (part of "its
  rotation", K6 (b)).
- **K9's rule (texts that become false, stale words only):** the "atom levels" comments (now level 3, the orbitals);
  the class docstring's Map boxes and panel lists (the cmap norm and Contour / Shading boxes, the spectrum as a cell
  of the map block, the legend panel beside the spectrum); the shaded-map docstrings (`_fill_map` and the like);
  `_lighten_color`'s docstring (no `_up` curves); `to_cmap`'s "when installed" and its install hint (all three
  copies); `_ViewBox.wheelEvent`'s "(Pan)" to "(Pan or Zoom)" (all three copies); run_stm_decomp.py's two "slider"
  comments (as O16); forces.py's save sentence; the mo_ldos_decomp README's "transparent in the exports" (the PDF / EPS
  / editable SVG only); the stm README's control table (the contour slice drawn while Contour is on; rows for the
  'cmap set' choice and the Contour / Shading box, like N4's rows; the shading maps in the 'cmap norm' row).
- **Kept as they are:** only Map's caches are kept apart (a view change into Single Point, Line or Path still computes
  that view's LDOS, as the brief has it); an E Min / E Max / E Pts edit is counted by comparing the boxes with the
  energies Map last formed; the 'contour level' line stays while Contour is off (the level is kept); `colorsys` and
  `_lighten_color` stay though nothing calls them now.

## After the last pass (wf_2b54edf1-9c3): two more for the owner

N1 to N4 and the settled list above are implemented, reviewed and fixed (`LAST_CHANGES.md`). The reviews found these.

**F4. Map's kept state still shares records with the other views (a crash on the new RUN-off return to Map).** Map
keeps its own maps and energies now, but three things it shows are still shared: the marker spectra (Single Point
computes them again for its own window), the heights its contour slice and shading are drawn from (`current_z_map`,
which a Topo h rerun, a broadening or a Decay change in any view replaces with the global topo), and the model (a
Decay or broadening change in another view leaves Map's maps from the old model). So after, for example, a run in
Map, an E Max change in Single Point with RUN on, then STOP and back to Map, the spectrum (new window) is drawn over
Map's energies (old window): pyqtgraph raises on the unequal lengths and the Map view is left without maps. A RUN-off
decomp click in Map hits the same pairing. The same kind of gap exists on the Line side (older): Single Point writes
the energy record Line / Path's data are sent with, so a Line send after a Single Point window change pairs Line's
curves with Single Point's energies.
- (a) Every record keeps its own energies, and Map keeps its own copies: Map's marker spectra with their energies
  and the LDOS-topo heights its maps were computed at (used for its contour slice and shading); Line / Path's data
  with their energies (sim_wrapper sends each with its own). Returning to Map then shows Map exactly as last plotted;
  with RUN on, a model change made elsewhere recomputes Map at the kept energies, as O2 says.
- (b) Minimal: a RUN-off return to Map draws only when the shared records still match Map's (the spectra's window,
  the LDOS-topo records, the model); otherwise nothing, as before. And each spectrum is drawn over its own energies,
  so nothing raises.
- (c) Leave it.

R: (a), the full reading of "Map's energies, data and caches are kept apart".

**F5. N1: a Line window too narrow for the two rows places nothing.** With Line allowed only `row_top`, a window too
narrow for the placement topo and the heatmap row to share a row at any height (a narrow or portrait window, or a large
text zoom) has no layout: at the first show nothing is placed, and on a resize the old placement stays and runs past
the window. Before, a stacked layout was taken.
- (a) Leave it: a minimum width for Line.
- (b) When the allowed layouts give none at the window's size, choose from every layout (as before) until the two rows
  fit again.

R: (b).

## Settled under rules already given (next pass, with F1 to F5)

- **P2's two rows:** Line's Tip Height and spectrum stay side by side (P2: "Second row: the topo path 1D plot and the
  LDOS 1D plots"); today the FlipBox stacks them in a tall slot, giving three rows. Path unchanged.
- **The fill row's height (N1):** cached per width and layout pass, so the arrangement does not repeat the bisection;
  the results are the same. (N1's rule is general, so Path's heat row follows it too.)
- **K9's rule:** the mo_ldos_decomp README's intro and `_imports.py` row (the resolvers are `_resolve.py`'s, re-exported
  by `_imports.py`); convergence/README.md's save sentence (as forces.py's); `_choose_orientation`'s "the shaded energy
  maps" becomes "the Relief-shaded energy maps".
