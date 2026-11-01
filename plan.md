# plan.md: the Qt conversion of every interactive window

The single source of truth for this update. A new session (or one whose context was compacted)
reads this file and `git log` on the branch and continues at the first unchecked step.

Branch: `claude/festive-mayer-q2ig83`. Push with `git push -u origin claude/festive-mayer-q2ig83`.
No pull request unless the owner asks.

Status marks: `[ ]` to do, `[~]` in progress, `[x]` done.

---------------------------------------------------------------------------------------------------

## 1. The request and the guide

The owner asked that every package of VASP_for_SPM switch from hard-coded matplotlib GUIs
(figures with hand-placed `add_axes([...])` widgets: `Slider`, `Button`, `CheckButtons`,
`RadioButtons`, `TextBox`, plus `mplcursors` hover labels and `mpl_connect` handlers) to the Qt
style of the guide "From a matplotlib-wired GUI to a real GUI" (the sm4_Browser conversion). The
guide's rules, in short:

1. automatic layout, nothing placed by hand; 2. everything shared is one function; 3. controls
are real Qt widgets; 4. windows never scroll; 5. one text-size rule; 6. short panel titles, the
details in the information panel; 7. spatial panels keep their x:y ratio, a row shares one
height; 8. number boxes instead of sliders (a slider survives only for scrubbing a list);
9. a window owns its state, controls only report; 10. comments name the rule, the README stays
current.  Section 7: efficient drawing (create once, update data; LUT images; coalesced motion;
draw on idle), computation on change, never on paint.  Section 9: copy `qtplot.py`,
`controls.py`, `widgets.py` into the project and import it as `plt`.

## 2. Working rules from the owner (hard constraints, every step)

- Implement exactly this plan; a deviation is stated in the delivery as an "additional change".
- Every delivery lists every function, gate, filter, constant or branch modified, removed or added.
- Never remove, weaken or bypass a gate, filter, threshold or validation to make inputs pass.
- Verify factual and technical claims against sources (library source or documentation) before
  stating them; otherwise say "I don't know".
- The README of every package touched and the root README are kept up to date.
- Layout code comments name the rule, e.g. `# automatic layout: the one rule for every window`.
- Commit messages end with the session's attribution lines; no model names in commits or code.

## 3. Decisions (made under stated assumptions; the owner was not available mid-task)

### 3.1 One shared layer at the repository root: `qtgui/`
- `qtgui/qtplot.py`, `qtgui/controls.py`, `qtgui/widgets.py`, `qtgui/mplexport.py` are the
  sm4_Browser layer (section 9 of the guide) with its sm4-specific parts removed (the sm4 control
  table and composite groups, the `.sm4` file-stamp rule, the scan-area navigator, the marker
  pairs) and what this suite's windows need added (section 3.4).
- Rule 2 (everything shared is one function) forbids a copy per package, so the layer lives once,
  at the root, and every window finds it by walking up from its own file to the folder that holds
  `qtgui/` (the suite's own "fixed sibling path" convention, `mo_ldos_projection/_imports.py`).
  The root README's rule "nothing in one package imports another" gains this one exception.
- `widgets.SPECS` is the one control table of the suite: every control of every window is named
  there once (kind, label, short label, font size, a box's width).

### 3.2 Scope: every window with widgets or canvas handlers, plus the one hand-placed colour bar
| file | old GUI | new |
|---|---|---|
| `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/simulator.py` | 11 sliders, 2 buttons, 2 check groups, 1 radio, text boxes, pick/drag handlers, mplcursors, hand-placed `plt.axes` | the busiest window; the worked example |
| `LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/sim_wrapper.py` | σ slider + box, 'Send to decomp' buttons added onto the simulator's figure | the same controls in the simulator window's layout |
| `.../mo_ldos_projection/decomp_window.py` | checks, area box, metal check, Save, mplcursors, click | |
| `.../mo_ldos_projection/topo_decomp_window.py` | checks, Save, gridspec grid | |
| `.../mo_ldos_projection/global_decomposition.py` | σ slider + box, Save | |
| `LDOS_spread_decomp/mo_ldos_decomp/run_wfn_plane_slice.py` | γ slider + box | |
| `LDOS_spread_decomp/MOPmatcher_mem/matcher/spread_collapse.py` | checks, σ slider, click, mplcursors (two plotter classes) | |
| `.../matcher/spread_plot.py` | checks, σ slider, click, mplcursors | |
| `.../matcher/state_decomposition_plot.py` | checks, Norm check, σ slider, click, mplcursors | |
| `.../matcher/vacupot_plotter.py` | mplcursors, click | |
| `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/plot_dos_cursors_levels.py` | pick/click/key, mplcursors | |
| `PROCAR_tools/Band_structure_plot.py` | 'Orbital' check, mplcursors, click | |
| `convergence/forces.py` | window-start slider, drag/key/scroll handlers, a 3D panel | see 3.5 |
| `charge_density/lv_slice_w_atoms_2system.py` | a colour bar at a hand-placed `add_axes` | a colour-bar cell |

Out of scope (stated, not converted): the scripts that only draw static matplotlib figures with no
widget and no handler (`charge_density/*` except the file above, `dIdV_tip_sample/dIdV_sim.py`,
`convergence/plot_forces.py`, `PROCAR_tools/procar_filter_lorbit14.py`, `unfold_bands.py`,
`POSCAR_tools/Make_pyramid_any_number_sides.py`, the matcher's `combined_vacupot_plotter.py`,
`matcher_vacupot.py`, `wf_dpl_vacupot.py`, the `run_*.py` scripts' own figures). They are not
"GUIs" in the guide's sense (no hand-placed widget, no interaction) and the guide's conversion
target is the interactive window. `sim_decay_full.py` and `tier3_decay.py` are fully commented
out (zero live lines) and are left as they are.

### 3.3 Controls
- Every slider becomes a number box (`controls.Text` for a value, `controls.Count` for a whole
  number with limits) whose digits step with the arrow keys; the one slider kept is the window-
  start slider of `forces.py`, which scrubs through the list of ionic steps (rule 8).
- Groups shared by several windows are one class in `controls.py`: `SigmaBox` (the σ box of the
  spread, state-decomposition, global-decomposition windows and the simulator wrapper),
  `GammaBox` (the γ box of the simulator and the plane slice), `StepButtons`, `RunControls`.
- `mplcursors` is replaced by one class, `widgets.HoverLabels`: the hover label of the artist
  under the cursor, enabled by a left click on the panel and cleared and disabled by a right
  click, exactly the behaviour the windows implemented around `mplcursors.cursor`. The
  `mplcursors` dependency goes.
- Save buttons stay `controls.Button('Save data')`; the mode radio of the simulator a
  `controls.Radio`.

### 3.4 Additions to the layer the windows need (none of them in the sm4 layer)
- `Axes.tricontourf(x, y, z, levels, cmap, norm, vmin, vmax, zorder)`: the scattered points
  interpolated linearly (scipy `LinearNDInterpolator`, the piecewise-linear model matplotlib's
  tricontourf fills) onto a raster, banded by the levels and drawn as one image through a lookup
  table; `offsets=[(dx, dy), ...]` draws the same raster at several positions (the simulator's
  tiled cells: one raster, moved, never re-imaged).
- `Axes.vlines`, `Axes.hlines`, `Axes.add_line`, `Axes.set_facecolor`, `Axes.set(**kw)`,
  `Axes.get_lines`, `Axes.xaxis` / `yaxis` proxies with `set_major_formatter`; `Axes.scatter`
  returning a `PathCollection` (per-point colours, `set_offsets`, `set_facecolors`, pick with
  `event.ind`); `LineCollection` with per-segment colours (`array` + `cmap` + `norm`,
  `set_array`, `set_segments`, `set_color`); `Line2D` and `Patch` as legend proxies;
  `legend(handles=..., title=...)` and a second legend kept through `add_artist`;
  `pcolormesh` with `shading='auto' | 'nearest'` (edges from centres); `Annotation` in
  `'axes fraction'`; `LogNorm`, `SymLogNorm`; `ticker.FuncFormatter`; a `cm` namespace
  (`plt.cm.tab10(x)`, `ScalarMappable`); `fig.patch`; `fig.colorbar(m, ax=[axes])`;
  `get_subplotspec()` on the automatic grid's axes.
- `mplexport`: the publication export of the new artists (tricontourf, PathCollection, coloured
  LineCollection, Patch proxies, LogNorm / SymLogNorm, a norm with `to_matplotlib()`).

### 3.5 `convergence/forces.py`
Its 3D panel (matplotlib `Axes3D`: atoms as spheres, force arrows as cones, rotate / pan / zoom)
has no counterpart in the layer, which draws 2D panels in one pyqtgraph scene. The window is
converted (its Qt layout, the convergence panel with its draggable time marker, the window-start
slider in a box, the arrow keys) and the 3D panel is matplotlib's `Axes3D` on a matplotlib Qt
canvas placed as one cell of the window's layout, with every 3D interaction unchanged. This is
the one documented exception to "one scene per window" and "matplotlib off the screen".

### 3.6 Verification
- `checks/render_windows.py` (new, at the root): every converted window opened offscreen on
  synthetic inputs at its design size, half, double and taller than wide; no control text cut,
  no two controls overlapping, no control inside a panel box; screenshots written for review.
- The mo_ldos_projection pytest suite runs offscreen (`QT_QPA_PLATFORM=offscreen`); its
  widget tests follow the new controls.
- The mathematics is untouched: every window calls the same pipeline functions as before.

## 4. Steps

- [x] 4.1 Survey: inventory of every window, its controls, handlers and matplotlib calls.
- [x] 4.2 The layer: `qtgui/` built from the sm4 files, sm4 parts removed, section 3.4 added,
  `HoverLabels`, `SigmaBox`, `GammaBox`, the suite's control table; a demo window renders
  offscreen (commit 76e35cc).
- [x] 4.3 `checks/render_windows.py`: the render-and-measure helper (commit 76e35cc).
- [~] 4.4 The simulator and its wrapper (the worked example), `plot_dos_cursors_levels.py`.
- [~] 4.5 The decomposition windows: `decomp_window.py`, `topo_decomp_window.py`,
  `global_decomposition.py`, `run_wfn_plane_slice.py`; their tests.
- [~] 4.6 The matcher plots: `spread_collapse.py`, `spread_plot.py`,
  `state_decomposition_plot.py`, `vacupot_plotter.py`.
- [~] 4.7 `Band_structure_plot.py`, `forces.py`, `lv_slice_w_atoms_2system.py`.
- [ ] 4.8 Verification pass (3.6) on every window; the repeated-logic scan.
- [ ] 4.9 READMEs (root and every package touched), this plan ticked, commit, push.
