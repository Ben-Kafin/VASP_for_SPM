# Part 3 pre-flight: the read-only mapping (workflow wf_e2476f62-5f5)

Four mappers read the code at HEAD e1995d7..b7132c0 (part 2 in progress) and proposed the design; two critics checked them.
Their questions are resolved in `PART3_BRIEF.md` (decisions) and `PART3_OPEN.md` (the owner's). Where this file and
`PART3_BRIEF.md` differ, the brief governs. Line numbers are approximate; part 2 has since been finished (2ca3267).

## Report: decomp (The decomp (plan sections 2, 3 and 4): the new file stm_from_dft/ldos_decomp.py and how simulator.py calls it)

### Code map

Snapshots read: **d5** = part 1's finished state `git show d5e08a4:` (equal to 3c19a27); **now** = HEAD e1995d7 (part 2 WIP, committed while I worked), identical to the working tree. **b05** = the plan's line numbers (pre-part-1). qtgui is unchanged since d5. The decomp code in simulator.py is the same in d5 and now except part 2's NumPy-to-CuPy edits: `cp.sum`, `quadrature.simpson(..., dx=self.e_step32)`, and `cp.asnumpy(y)` in `_curve`. All line numbers are approximate.

## simulator.py (class `Interactive_STM_Simulator` unless noted)

**State and shared data**
- `__init__` (d5 762-765 / now 1001-1004): `active_marker_idx = 0`, `plot_level = 0`, `active_element, active_atom = None, None`, `_type_color_map = {'Au': 'orange', 'N': 'blue', 'C': 'brown', 'H': 'grey'}`. It also sets `m_colors`, a list of 10 hex colours (d5 755 / now 994).
- `Unified_STM_Simulator.parse_vasp_outputs` sets:
  - `atomtypes, atomnums` (d5 348 / now 517);
  - `orbitals` (d5 360-386 / now 545-571);
  - `vesta_label_map`, {global atom number: "C1"} (d5 388-394 / now 573-579);
  - `dos_up_gpu` (d5 420 / now 605).
- `_apply_extra_broadening` (d5 463 / now ~648) **rebinds** `self.dos_up_gpu` to a new array (`_dos_up_gpu_orig.copy()`) on every broadening change.

**Spin-polarised mismatch (COLLINEAR_POL).** Read in `doscar_parser.py` HEAD ~160:
- the parser splits the 18 interleaved columns into `spin_up_dos = site_dos[:,:,0::2]` and `spin_down_dos = [:,:,1::2]`, 9 each;
- the simulator then sets `num_cols = spin_up_dos.shape[2]` (= 9);
- so `orbitals = mapping_pol[9]` has **18** names (`s_up, s_down, py_up, ...`) for arrays that have **9** columns. `dos_up_gpu`, `f_ldos_raw` (= ld_up ± ld_dn) and `cached_spec_ldos` all have 9 orbital columns (s, py, pz, ...).

So `enumerate(self.orbitals)` is misaligned: column 1 (py) is labelled `s_down`, and so on, and indexing fails from col 9 on. NCL and UNPOL are aligned (9 names, 9 columns).

**`_get_partitions(f_ldos_raw)`** (d5 887-929 / now 1124-1166; b05 896-938) returns `[(label, data (points, energies))]`. f_ldos_raw has shape (points, E, atoms, orbitals); `None` gives `[]`.
- Level 0: `("Total", sum over axes 2, 3)`.
- Level 1: `("Au", Au atoms)`, `("Molecule", every non-Au atom)`; zeros if empty.
- Level 2: one `(t, ...)` per `atomtypes` entry.
- Level 3 and level 4 are identical. Guarded by `active_element is not None`. For each `col_idx, orb in enumerate(self.orbitals)`:
  - `D = cp.sum(dos_up_gpu[e_idx,:,col_idx], 0)`, `dD = cp.gradient(D)`, `d2D = cp.gradient(dD)`;
  - the column is skipped only if max|D|, max|dD| and max|d2D| are all < 1e-5;
  - otherwise `(f"{element} {orb}", sum(f_ldos_raw[:,:,e_idx,col_idx], axis=2))`.
- Callers (no others in the repository):
  - `_update_all` (d5 2322 / now 2564). In Map this feeds the map block; in Line/Path it feeds `_fill_heat_row`, which is how the heatmaps get their partitions.
  - `_redraw_map_slice(i)` (d5 3031 / now 3278). It computes all partitions, then takes slice i. It is called from `_on_rel`, including X3's re-sort (now ~3540).

**`_draw_spectrum(f_ldos_raw, spec_disp, p_dist, nepts, full_refresh)`** (d5 2446-2618 / now 2692-2865). Callers: `_update_all` (d5 2343 / now 2586), `_move_line_markers` (d5 2894 / now 3141), and the mark_dynamic branch of `_on_pick` (d5 3122 / now 3371).
- **Active spectrum** (d5 2458-2463 / now 2704-2710):
  - Line/Path: `active_idx = int(marker_ratios[min(active_marker_idx, n-1)]*(npts-1))`, `active_ldos = f_ldos_raw[active_idx]`. If f_ldos_raw is None it falls back to `zeros_like(cached_spec_ldos[0])`; level 0 would crash in that case, and it is not reached.
  - SP/Map: `active_idx = min(active_marker_idx, n-1)`, `spec_disp[active_idx]`.
- **Cache key** (d5 2469-2474 / now 2716-2721): `(mode, plot_level, n_marks if level==0, active_element if level>=3, active_atom if level==4, bool(feenstra), len(map_e_targets) if Map)`. When `build` is true the panel is cleared and rebuilt (`_spec_lines`, `_spec_emarkers`); otherwise `_curve` does `set_data` in drawing order (`next(drawn)`).
- **Helpers, all local** (d5 2480-2505 / now 2727-2752):
  - `_curve`;
  - `_orbit_base` (strips `_up` / `_down`);
  - `_lighten_color(c, 0.3)` (colorsys HLS);
  - `unique_bases = sorted(set(_orbit_base(o) for o in self.orbitals))`;
  - `styles = ['-','--',':','-.'] + [(0,(3+i,2)) ...]`, `linestyle_map`;
  - `S = 0.25`.
- **Norm** (d5 2507-2511): `active_ldos_norm /= simpson(sum(active))+1e-15`. Level 0 instead normalises each marker total by its own integral.
- **Level 0** (d5 2513-2532): one curve per marker. `color = m_colors[i % 10]`, `lw 2.5`, `picker=True, pickradius=5`, `label 'marker_i'`, default zorder (2). Legend `ax_spec.legend(loc='upper right')` (2532).
- **Level 1** (2533-2543):
  - faded total: black, lw 1.5, alpha 0.1, zorder 1, no picker;
  - Au: orange, lw 2, zorder 2, picker;
  - Molecule: black, lw 2, zorder 2, picker;
  - legend titled "Partition" (2543).
- **Level 2** (2544-2551): per type, `_type_color_map.get(t,'grey')`, lw 2, `picker=True, pickradius=3`, `label t`. Legend titled "Atom Types" (2551).
- **Level 3** (2552-2562):
  - faded element total: element colour, lw 2, alpha 0.15, zorder 1;
  - one pickable curve per atom (`vesta_label_map`, `label.startswith(element)`), zorder 2;
  - legend titled "Atom Types" (2562).
- **Level 4** (2563-2591):
  - active atom total: lw 2.5, alpha 1.0, zorder 5;
  - its orbital curves over **all** `self.orbitals`, no 1e-5 filter (`col_idx = orbitals.index(orb)`, misaligned in POL): `linestyle_map`, lighter colour for `_up`, lw 1.2, zorder 10, label `"C1 – pz"`, no picker;
  - `self.cursor = widgets.HoverLabels(ax_spec, orb_artists, enabled=True, arm_by_click=False)` on build (2578-2582);
  - the other atoms drawn faded with `S` (2584-2587);
  - `leg1 = legend(...)`, `add_artist(leg1)`, then a second legend "Orbitals" (2588-2591).
- **The five legend calls** are at d5 2532, 2543, 2551, 2562, 2589/2591 plus `add_artist` at 2590. Now they are at 2779, 2790, 2798, 2809, 2836/2838 plus `add_artist` at 2837.
- **ylim** (d5 2593-2596 / now 2840-2843): counts only lines with `get_alpha() == 1.0`, and only level 4's atom total has that. With level 4 gone the block never fires.
- **Title** (d5 2598-2606 / now 2845-2853): `"Partitioned LDOS: marker {active_marker_idx} at {p_dist_val} Å"`, plus the Feenstra ylabel.
- **Energy markers** (d5 2610-2617 / now 2857-2864): `axvline(..., lw=2, picker=5, label=f'emarker_{i}')`, drawn after the curves on build and `set_xdata` in place.

**Other uses**
- `_clear_panels` (d5 1328-1350 / now 1570-1592): resets `_spec_key/_spec_lines/_spec_emarkers/_spec_title`, `_line_disp`, and calls `self.cursor.remove()`.
- `_cluster_label`:
  - made in `_make_controls` (d5 1133 / now 1375);
  - in `_kept_widgets` (1285 / 1527);
  - used as the `Titled` label in `_build_ui` (1405, 1435 / 1647, 1677);
  - `show` / `set_text('LDOS')` (1461-1462 / 1703-1704);
  - `_update_all` sets `f"{quantity}: {active_element}" if plot_level >= 3 else quantity` (d5 2376, 2440 / now 2619, 2686).
- Heatmap titles in `_fill_heat_row` (d5 2809 / now 3056): `p_label.split()[-1] if plot_level >= 3`. Map ylabels in `_fill_map` (d5 2955-2958 / now 3202-3205): the same test. With level 4 gone, `>= 3` means `== 3`.
- `m_colors` readers:
  - marker facecolors in `_update_all` (d5 2163/2171/2174 / now 2404/2412/2415);
  - `_draw_spectrum` level 0;
  - `_make_line_markers` (2847 / 3094);
  - `_fill_map` (2952 / 3199);
  - `_place_map_markers` (3016 / 3263);
  - sim_wrapper `send_to_decomposition` (now 236, 260, by index).

**Event handlers**
- **`_on_pick`** (d5 3074-3123 / now 3323-3372; b05 2776-2816). Branches, in order:
  1. `_cursor_` label → crop cursor drag;
  2. a crop cursor already active → return;
  3. `_zlevel` → level-line drag;
  4. `ends` (Line) → end drag;
  5. `path_pts` (Path) → node drag;
  6. `marks` → `mark_map`;
  7. Map energy-map marker scatters → `mark_map`;
  8. **spectrum** (`mouseevent.inaxes == ax_spec`, d5 3094-3113 / now 3343-3362):
     - `emarker_` → at level 0 `active_obj=('emarker', i)`; return;
     - level 0 → `active_marker_idx = int(label)`, level 1;
     - 1 → 2;
     - 2 → `active_element = label`, level 3;
     - 3 → `active_atom` via `vesta_label_map` and `_get_element_by_index_helper`, level 4;
     - then `_update_all()`;
  9. **mark_dynamic**: any Line2D / _RefLine / PathCollection labelled `marker_` that is not caught above (heatmap / stripe / profile lines, Line Topo scatter) → drag, sets `active_marker_idx`, `_draw_spectrum` in place.

  No branch checks the button or `dblclick`. The spectrum branch must stay ahead of branch 9, because level-0 spectrum curves are labelled `marker_i`.
- **`_on_press`** (d5 3125-3134 / now 3374-3383; b05 2818-2825): `inaxes == ax_spec`, `toolbar.mode == ""`, `not dblclick`. If no `l.get_picker()` line contains the event, it steps back one level and calls `_update_all`.
- **`_get_element_by_index_helper`** (d5 3136-3142 / now 3385-3391): used only by the level-3 pick.
- **`_on_motion`** (d5 3144-3245 / now 3393-3494): its emarker branch (d5 3235 / now 3484) moves `map_e_targets[idx]` and the line found by label.
- **`_on_rel`** (d5 3247-3273 / now 3496-~3555): emarker release recomputes the map. Now it also does X3's stable re-sort, which permutes `_spec_emarkers` and relabels them `emarker_{i}`. The end / node / SP-Map marker release calls `_update_all`.
- **`_on_mode_change`** (d5 1914-1918 / now 2156-2160): `mode`, `active_marker_idx = 0`, `plot_level = 0`, `_build_ui()`. It does not reset `active_element`.
- **Connections**: pick / motion / release / press in `_make_controls` (d5 1185-1188 / now 1427-1430; b05 1194-1197).
- **`_update_all`**: returns before `_draw_spectrum` while `not is_running` (d5 2176-2178).

## sim_wrapper.py
- d5: `_instantiate` (186-209) restores `active_marker_idx` and `plot_level` via setattr before `run_interactive` (193-199). `capture_state` (253-292) reads them (263-264). In b05 these are at 198/208 and 265/276-277.
- **now (X1 applied)**: `_instantiate` (144-153) restores nothing and `capture_state` is gone. sim_wrapper no longer reads or sets `plot_level` / `active_marker_idx`. It still reads `m_colors[i]`, `marker_ratios`, `marks.get_offsets()`, `cached_spec_ldos`, `cached_ld_up/dn`, `_combine_ldos_display`, `cached_eg`, `normalize` and `mode`.
- `vesta_label_map`, `_get_element_by_index_helper` and `active_atom` have no reader outside simulator.py. A grep finds them only in `plot_dos_cursors_levels.py` and `wrong_decays/sim_decay_full.py`, each with its own copy.

## qtgui/qtplot.py (package copy; unchanged since d5)
- **`MouseEvent`** (6006): `x, y` in display pixels from the bottom left, `button` (via `_QT_BUTTONS` 6097: left 1, middle 2, right 3), `dblclick`, `inaxes`, `xdata/ydata`, `guiEvent`.
- **`PickEvent`** (6075): `mouseevent` is the **same object** the press handlers get; plus `artist` and `ind`.
- **`FigureCanvas._event`** (6182): `inaxes = Figure._axes_at` (7085), the last visible axes whose **plot box** (`_px_rect`) holds the point. The title and tick labels are not inaxes.
- **`mousePressEvent`** (6207-6213): `_pick(e)` first, then `_fire('button_press_event', e)`.
- **`mouseDoubleClickEvent`** (6215-6221): the same with `dblclick=True`.
- **`mouseReleaseEvent`** (6223).
- **`_fire`** (6161-6166): handlers in connection order; an exception is printed and the next handler still runs.
- **`Figure._pick`** (7094-7102): the hit list `[a for a in ax._children if a.pickable() and a.contains_px(e.x, e.y)]` is built **once, before any handler runs**. Then one `pick_event` fires per hit, in `_children` order.
- **`Artist`** (2055):
  - `zorder_default = 2.0`;
  - `_add_item` calls `setZValue(zorder)` and then `vb.addItem`;
  - `set_zorder` / `get_zorder`;
  - `pickable()` = picker not None and visible;
  - `contains(event)` → `contains_px(event.x, event.y)`;
  - `remove()` does **not** clear `.axes`.
- **`Line2D.contains_px`** (2293): tolerance = the numeric picker, or 5 for `True`, × em_px/(1000/72). `pickradius` is never read (`_plot_one` 5473 drops it). **`_RefLine.contains_px`** (2595): same tolerance.
- **`Axes`**:
  - `_children` holds artists in the order added;
  - `get_lines()` (4326) returns Line2D and _RefLine in that order;
  - `_register` appends;
  - `clear()` (4867) empties `_children` but **keeps** the axes' `HoverLabels` registry entry; only `Axes.remove()` (~4864) drops it.
- **`set_title`** (4953) shows `short_title(full)` (1559: cut at `': '`, `', '`, a double space, `' ['`). The full text goes to `Figure._set_info`, i.e. the info panel; `set_info_line` (6679) adds a window's own lines.
- **`widgets.HoverLabels`** (widgets.py 162):
  - one per axes; a new one replaces the old;
  - `_hit` tests `a.axes is self.ax`, so curves removed by `ax.clear()` still match (stale labels);
  - its figure-wide `button_press_event` handler does nothing when `arm_by_click=False`.

**Choosing the top curve.** At the click you need:
- the visible pickable lines of ax_spec whose `contains_px(e.x, e.y)` holds, in `get_lines()` order;
- each one's `get_zorder()`.

The top curve is the one with the highest zorder; among equal zorders, the last in that order. This works because each artist's item is added to the ViewBox (pyqtgraph `ViewBox.addItem` parents it to `childGroup`, so the items are siblings) just before `_register` appends it. So `_children` order equals Qt's insertion order, and Z = zorder.

Both `_on_pick` (picks fire in that order with the same mouseevent) and `_on_press` see exactly this set, provided nothing redraws between them. sm4_browser's `_pick` (qtplot 5154) sets no `ind`, so section 6 must keep this copy's `_pick`.

### Proposal

Every item names the plan bullet it implements. Items marked (Q-…) depend on a question below.

## A. New file `stm_from_dft/ldos_decomp.py`
It uses relative imports like `feenstra` and `quadrature`: `numpy`, `cupy` (as part 2 leaves the arrays), `colorsys`, `from .qtgui import qtplot as plt` (`to_rgb`) and `from .qtgui import widgets` (`HoverLabels`). Its docstrings follow the tag rule of Q-D17.

1. **Constants.**
   - `TITLES = ('', 'Parts', 'Atoms', 'Orbitals')` — §2 "Its title", level 0 none.
   - `LAST_LEVEL = 3` — §2 levels table.
   - `ORBITAL_TOL = 1e-5` — §2 level 3 "Which orbitals".
2. **Helpers moved out of `_draw_spectrum`, unchanged** — §3 "Helpers".
   - `_orbit_base(orb) -> str`.
   - `_lighten_color(color, amount=0.3) -> (r, g, b)`.
   - `_linestyle_map(orbitals) -> {base: style}`: built from **all** `orbitals` as today, so an orbital's style never depends on the filter.
   - `S`: Q-D6.
3. **Records** — §3 "spectrum's curves", "legend entries".
   - `Curve` namedtuple: `y, color, linestyle, linewidth, alpha, zorder, label, picker, option, entry, clickable, hover`. `entry` is the legend text, or None when the curve is not an entry.
   - `Entry` namedtuple: `label, color, linestyle, linewidth, clickable, option`.
4. **`top_artist(artists) -> artist | None`.** The artists come in drawing order. It returns the one with the highest `get_zorder()`; among equals, the last. A loop with `>=` does this, matching a stable sort by zorder. — §2 clicks, "only the top one counts".
5. **`class Decomp`** holds the state — §3 "The state".
   - `level` (0..3), `marker` (the active marker index), `element` (the active atom type or None).
   - `__init__()` sets 0, 0, None. There is no `active_atom` (§3 "`active_atom` goes").
6. **`Decomp.orbital_columns(atomtypes, atomnums, orbitals, dos_up) -> [(col, orb)]`.** The active element's surviving orbital columns: today's `cp.sum` / `cp.gradient` ×2 / 1e-5 test, unchanged. The column alignment depends on Q-D1; the result is cached per Q-D12. — §2 level 3 "Which orbitals".
7. **`Decomp.partitions(f_ldos, atomtypes, atomnums, orbitals, dos_up) -> [(label, data)]`.** Exactly `_get_partitions` for levels 0..3, with level 4's branch gone. Level 3 uses item 6 and labels `f"{element} {orb}"`. `f_ldos None` → `[]`; level 3 with `element None` → `[]`. — §3 "The partitions".
8. **`Decomp.curves(marker_spectra, marker_colors, atomtypes, atomnums, orbitals, dos_up, type_colors, norm=None) -> [Curve]`**, in drawing order — §3 "spectrum's curves", §2 table and "Level 3".
   - `marker_spectra`: one displayed (E, atoms, orbitals) array per marker, in index order.
   - `marker_colors`: one colour per marker.
   - `norm`: the integral callable when Norm is on, else None. The +1e-15 and the per-level normalisation stay as today.
   - Levels 1-3 take `marker_spectra[min(marker, n-1)]`, today's clamp; no markers → no curves.
   - **Level 0:** per marker i:
     - sum over atoms and orbitals (divided by `norm(y)+1e-15` under Norm);
     - `marker_colors[i]`, '-', lw 2.5, `label 'marker_i'`, `picker True`;
     - option i, entry '' (no text, §2 "Level 0"), clickable.
   - **Level 1:**
     - faded total (black, lw 1.5, alpha 0.1, zorder 1, no picker, not an entry);
     - Au (orange, lw 2, zorder 2, picker, entry 'Au', clickable);
     - Molecule (black, the same, entry 'Molecule').
   - **Level 2:** per `atomtypes` entry t: `type_colors.get(t,'grey')`, lw 2, picker, option t, entry t, clickable.
   - **Level 3** (element E):
     - faded E total: E's colour, lw 2, alpha 0.15, zorder 1, not an entry — §2 "Behind them";
     - per item 6 column: `sum(s[:, e_idx, col], axis=1)`, E's colour (lightened 0.3 for `_up`), `linestyle_map[_orbit_base(orb)]` shared by `_up`/`_down`, lw 1.2 and zorder 10 (Q-D8), `label = entry = f"{E} {orb}"`, no picker, not clickable, `hover True`.
   - An empty atom set gives a zeros curve of the energy length, as today.
9. **`Decomp.legend(curves) -> (title, entries, back)`.**
   - `title = TITLES[level]`.
   - `entries = [Entry(c.entry, c.color, c.linestyle, c.linewidth, c.clickable, c.option) for c in curves if c.entry is not None]`. So the faded totals and the energy markers are not entries.
   - `back = level > 0`.
   - Order: marker index order at level 0, so it follows the stripe through §2a.
   - Clickable at levels 0-2, plain key rows at level 3.
   - — §2 legend panel "Its entries", "Not entries", "Its Back button", "Its title".
10. **Clicks** — §2 clicks, "The two ends", "Double click", "One click, one step".
    - `Decomp.down(option) -> bool`:
      - level 0 → `marker = option`, level 1;
      - level 1 → 2;
      - level 2 → `element = option`, level 3;
      - level 3 → False.
    - `Decomp.up() -> bool`: levels 1-3 → level −1, True; level 0 → False.
    - `Decomp.press(button, dblclick, item) -> bool` (item is a Curve, an Entry, or None):
      - `dblclick` → False;
      - button 3 → `up()`;
      - button 1 with a clickable item → `down(item.option)`;
      - anything else → False.
    - True means the state changed and the caller redraws once. False means nothing changed and nothing redraws.
11. **`Decomp.hover_labels(ax, lines, curves, old) -> HoverLabels | None`.** Called on a build. It removes `old` and makes `widgets.HoverLabels(ax, [lines of hover curves], enabled=True, arm_by_click=False)` when there are any. So the labels sit only on level 3's orbital curves (Q-D9). — §2 "The hover labels … stay on these curves".
12. **`Decomp.key(curves) -> (level, tuple(c.label for c in curves))`.** Replaces the level part of `_spec_key`, which drops `active_atom` (Q-D11). — §3, the level-dependent part of the redraw.
13. **(Q-D16) `Decomp.info() -> str`**, the decomp state for the info panel, if ruling 8 is read that way.

## B. `simulator.py`
1. **Import and state.**
   - Add `from . import ldos_decomp`.
   - In `__init__` (now ~1001-1003), replace the three lines with `self.decomp = ldos_decomp.Decomp()`.
   - Add class properties `plot_level`, `active_marker_idx` and `active_element` that read and write `self.decomp.level / .marker / .element` (Q-D2).
   - `_on_mode_change` (now 2156) stays unchanged and works through them. sim_wrapper keeps working, now and at d5. — rule 4; §2 "A view change"; §3 state.
2. **Partitions.**
   - `_update_all` (now 2564) and `_redraw_map_slice` (now 3278) call `self.decomp.partitions(f_ldos_raw, self.atomtypes, self.atomnums, self.orbitals, self.dos_up_gpu)`.
   - The Line/Path heatmaps get them through `_update_all` → `_fill_heat_row`, as now.
   - `_get_partitions` is deleted.
   - — §3 "The partitions"; §4 "`_get_partitions` goes".
3. **`_draw_spectrum`** keeps its name, signature, callers and its view machinery: the active index for the title, Map's `map_e_targets`, build or in place, title / ylabel, ylim (Q-D10), energy markers. Its middle becomes:
   - `marker_spectra`: Line/Path `[f_ldos_raw[int(r*(npts-1))] for r in marker_ratios]` (`[]` if None); SP/Map `spec_disp`.
   - `marker_colors`: `m_colors[i % len]`, with Line/Path taking §2a's kept colours from that agent.
   - `norm` as part 2 leaves the integral.
   - `curves = self.decomp.curves(...)`.
   - `key = (mode, self.decomp.key(curves), bool(feenstra), len(map_e_targets) if Map)`.
   - On build: `ax_spec.plot(cached_eg, cp.asnumpy(c.y), color, lw, linestyle, alpha, zorder, label, picker)` per curve, appended to `_spec_lines`, then `self.cursor = self.decomp.hover_labels(...)`. Otherwise `set_data` in order.
   - `self._spec_curves = curves`.
   - The legend panel gets `self.decomp.legend(curves)` on every call (§2 "Live").
   - — §3 "spectrum's curves", §4 "The spectrum panel in every view".
4. **Removed from `_draw_spectrum`:**
   - the local helpers `_orbit_base`, `_lighten_color`, `unique_bases/styles/linestyle_map` and `S` (now 2737-2752);
   - `active_ldos_norm` (now 2754-2758; moved into item A8);
   - the level 0/1/2 branches (moved) and today's level 3 (per atom) and level 4 branches (deleted);
   - all five `ax_spec.legend(...)` calls and `add_artist(leg1)` (now 2779, 2790, 2798, 2809, 2836-2838);
   - level 4's HoverLabels code (now 2825-2829).
   - — §4 "The spectrum's legends go"; §3 "Today's level 3 … and level 4 branches go".
5. **`_clear_panels`**: add `self._spec_curves = []` next to `_spec_lines`.
6. **`_on_pick`**: its spectrum branch (now 3343-3362) becomes a guard that returns for `mouseevent.inaxes` in (`ax_spec`, the legend axes). This keeps level-0 `marker_i` curves out of the mark_dynamic branch. Every other branch is unchanged. — §3 clicks replace "the spectrum branch of `_on_pick`"; §4 "Kept".
7. **`_on_press`**: its spectrum part (now 3375-3383) is replaced (Q-D3/Q-D4/Q-D13/Q-D14).
   - If `inaxes is ax_spec` and `toolbar.mode == ""`: `hits = [l for l in ax_spec.get_lines() if l.pickable() and l.contains(event)[0]]`, then `top = ldos_decomp.top_artist(hits)`.
   - If top is an energy marker, `item = None`. On a left, non-double press at level 0 it also sets `active_obj = ('emarker', i)`; that is today's drag, moved from `_on_pick`.
   - Otherwise `item = self._spec_curves[self._spec_lines.index(top)]`, or None when nothing is under the press.
   - If `self.decomp.press(event.button, event.dblclick, item)` is True, call `self._update_all()` once.
   - For the legend axes: `item = <legend panel>.entry_at(event)`, then the same `press`.
   - The Back button's callback is `if self.decomp.up(): self._update_all()`.
   - The decision is made once, from what is drawn, before any redraw.
   - — §2 clicks, "The decision is made from the curves as they were clicked"; §5 "Clicks".
8. **`_get_element_by_index_helper`** and **`vesta_label_map`**: Q-D7.
9. **Unchanged:**
   - the pick / press / motion / release connections;
   - every `_on_pick` / `_on_press` branch outside the spectrum and legend panels (mark_dynamic and its in-place `_draw_spectrum` included);
   - `_on_motion`;
   - `_on_rel`, apart from §2a's agent;
   - `_move_line_markers`;
   - the `plot_level >= 3` tests of the cluster label, the heatmap titles and the map ylabels (Q-D15).

## C. Interface to the legend panel (built by another agent)
- **What it receives**, from `_draw_spectrum` on every call: `(title, entries, back)`.
  - `title`: '' / 'Parts' / 'Atoms' / 'Orbitals'.
  - `entries`: Entry list in display order. Each has `label` ('' at level 0), `color` (a name, hex string or RGB tuple), `linestyle` ('-', '--', ':', '-.' or a `(0, (on, off))` dash tuple, which `qtplot.mkpen` takes), `linewidth` (pt), `clickable` (button box or plain key row) and `option` (opaque).
  - `back`: whether the Back button shows.
- **What it gives back:**
  - `entry_at(event) -> Entry | None` for a press inside the panel;
  - its Back button calls the simulator's back callback.
- **Constraint:** its items must not be pickable and must not be labelled `marker_…`. Otherwise `_on_pick`'s mark_dynamic branch takes them.

### Questions

- **decomp:Q-D1** (§2 Level 3 'Which orbitals' ('In a spin-polarized run the _up and _down columns are separate entries, as in the maps'); §2 'the lighter colour for _up columns'): In a spin-polarised (COLLINEAR_POL) run the displayed data has no separate up and down columns. The parser splits the 18 interleaved DOSCAR columns into spin_up_dos and spin_down_dos, 9 columns each. f_ldos_raw and cached_spec_ldos are ld_up + ld_dn (or ld_up − ld_dn under Mag), 9 orbital columns. But self.orbitals has 18 names ('s_up','s_down','py_up',...). So _get_partitions' `enumerate(self.orbitals)` (now ~1146) labels column 1 (py) as 's_down', column 2 (pz) as 'py_up', and so on. From col_idx 9 it indexes past the arrays' 9 columns: an IndexError in NumPy, and presumably in CuPy (unverified). Today's level 4 has the same misalignment (`orbitals.index(orb)`, now ~2819). The maps' level-3 partitions are therefore wrong or crash in POL runs, and there are no separate _up/_down entries to copy. NCL and unpolarised runs are aligned.
  - (a) Level 3 (spectrum, maps and heatmaps alike) uses one entry per displayed column, named by its base orbital in POL (self.orbitals[0::2] with '_up' stripped: s, py, pz ...). The data is the spin sum (or the spin difference under Mag); no _up/_down entries, and the lighter-_up rule never applies.
  - (b) Separate _up/_down entries from the two channels kept apart (cached_ld_up / cached_ld_dn and the marker spectra's s_up / s_dn, which are combined today). This adds an uncombined marker cache and touches part 2's caches, and Mag on still needs a rule (one signed column per orbital?).
  - (c) Leave the code as it is (misaligned labels, IndexError past column 8).
  - R: Owner's call, since it changes his stated rule. I recommend (a): it is correct for the data the window displays and needs no new data path. (b) is the literal rule, at the cost listed.
- **decomp:Q-D2** (§3 'The state … The simulator's plot_level, active_marker_idx and active_element stay readable and settable on the simulator (rule 4)'): The plan puts the state in ldos_decomp.py but does not say how the three names stay attributes of the simulator. After part 2's X1 (now HEAD), sim_wrapper no longer reads or sets them: capture_state and the restore in _instantiate are gone. They are still read and set inside simulator.py (_on_mode_change, the mark_dynamic pick, the cluster label, the heatmap and map labels, the title).
  - (a) Decomp object holds level, marker and element; Interactive_STM_Simulator gets three properties (get and set) forwarding to self.decomp. __init__ creates self.decomp in place of the three assignments. No validation, like today's plain attributes.
  - (b) The state stays as plain simulator attributes; ldos_decomp is functions that take and return it. (The decomp then does not 'hold' the state as §3 says.)
  - (c) Both hold copies kept in sync (fragile).
  - R: (a).
- **decomp:Q-D3** (§3 'The clicks … replace … the spectrum branch of _on_pick … the spectrum part of _on_press'; 'The decision is made from the curves as they were clicked'): The plan does not say which handler makes the one decision. qtplot fires every pick_event for the click (hit list built once) and then button_press_event, with the same MouseEvent.
  - (a) In _on_press. Its spectrum part tests ax_spec.get_lines() (pickable, contains) at the press, before any redraw, and takes the top with top_artist. _on_pick's spectrum branch only returns, which it must keep doing so level-0 'marker_i' curves don't reach mark_dynamic.
  - (b) _on_pick collects the spectrum hits of this mouseevent without acting; _on_press decides from that list and clears it.
  - (c) _on_pick decides at the first pick of a mouseevent and marks it handled; _on_press handles the no-hit cases (right click, left on nothing) and skips handled events.
  - R: (a): one handler, one decision, the same geometry the picks used, and it is the only handler that sees right clicks on empty space.
- **decomp:Q-D4** (§2 'What stays as it is': Map's energy markers ('a left click on one at level 0 starts its drag, as today. At other levels a left click on one does nothing'); §2 clicks 'only the top one counts'; right click 'never goes down'): Two cases are undefined. First, a left click can hit both an energy marker and a curve. Today at level 0 the curve's pick steps down and the press then finds the redrawn energy marker and does not step back, so the result is level 1 with no drag. At levels 1-2 the curve steps down. Second, today a right click (or the double-click press) on an energy marker at level 0 also starts its drag, because there is no button check.
  - (a) Energy markers take part in the top test. They are drawn after the curves with the same zorder 2 (zorder 10 at level 3), so an energy marker under the click is the top item at levels 0-2. Its rule applies (drag at level 0, nothing at levels 1-2), and the curve beneath does not count. The drag starts only on a left, non-double press. A right click ignores energy markers and goes back.
  - (b) Curves first: a curve under a left click always takes it, as today's net result at level 0, and the energy marker is ignored.
  - (c) Both: the drag starts and the level steps down.
  - R: (a).
- **decomp:Q-D6** (§3 'Helpers … move into the new file … the faded-colour factor S'): S = 0.25 is used only by level 4's other-atom faded curves (now ~2830-2834), which go. The new level 3's faded element total is drawn 'as today's level 3 draws it': the element colour at alpha 0.15, no S. So a moved S would be unused.
  - (a) Move S anyway (an unused constant).
  - (b) Drop S with level 4.
  - (c) Use S for level 3's faded total (this changes today's level-3 look).
  - R: (b).
- **decomp:Q-D7** (§3 'No longer used by the decomp: vesta_label_map and _get_element_by_index_helper. They stay wherever else the simulator uses them.'): There is no 'else'. vesta_label_map is built in Unified_STM_Simulator.parse_vasp_outputs (now ~573-579) and read only by the spectrum's level 3 and 4 and the level-3 pick. _get_element_by_index_helper (now ~3385) is called only by that pick. Nothing else in the repository uses them; plot_dos_cursors_levels.py and wrong_decays/ have their own copies.
  - (a) Keep both, unused.
  - (b) Delete _get_element_by_index_helper (Interactive_STM_Simulator only); keep vesta_label_map (built in the base class, cheap, possibly useful).
  - (c) Delete both.
  - R: (b).
- **decomp:Q-D8** (§2 Level 3 'the style today's level 4 gives its orbital curves (linestyle_map, _orbit_base, _lighten_color)'; §5 entries 'in the colour, line style and line width its curve is drawn with'): The plan names colour, line style and lighter _up, but not today's level-4 line width 1.2 and zorder 10, nor whether the orbital curves are pickable. A left click does nothing at level 3, and today they have no picker. Separately, P9 (part 1, open): pickradius= is never read by qtplot. Should the new curves carry today's pickradius=5 / 3?
  - (a) Orbital curves lw 1.2, zorder 10, no picker (today's level 4). Levels 0-2 keep picker=True and drop the ignored pickradius.
  - (b) The same, but carry pickradius as today (inert).
  - (c) Other values for lw or zorder.
  - R: (a). The legend's level-3 samples are then at lw 1.2.
- **decomp:Q-D9** (§2 'The hover labels (widgets.HoverLabels) that today's level 4 puts on its orbital curves stay on these curves'): Axes.clear() does not drop the axes' HoverLabels (only Axes.remove does), and Artist.remove() keeps .axes, so HoverLabels._hit still matches removed curves. Today, after leaving level 4, hovering where its orbital curves were still shows their labels. The same would happen after leaving the new level 3.
  - (a) On every spectrum build, remove the old HoverLabels and make a new one only when the level has hover curves (level 3).
  - (b) Keep today's behaviour (made only on a level-3 build, removed only on a view change).
  - R: (a): the labels then stay only on 'these curves'.
- **decomp:Q-D10** (§3 / §4 (not named)): _draw_spectrum's y-limit block (now 2840-2843) counts only curves with get_alpha() == 1.0. Only level 4's active atom total has that, so with level 4 gone the block never fires and the spectrum always autoscales, which is today's behaviour at levels 0-3.
  - (a) Leave the block unchanged (dead code).
  - (b) Delete it.
  - (c) Make it count opaque curves (alpha None), a behaviour change.
  - R: (a), since the plan does not name it. Listed as found.
- **decomp:Q-D11** (§3 'Today's level 3 (single atoms) and level 4 branches go'; §2a 'The colour stays with the marker … redrawn once for the new order, with no rearrangement or rebuild'): The redraw cache key (now 2716-2721) has an active_atom slot (level 4), which goes. It also does not cover two cases that make the in-place path wrong. First, at level 3 the set of orbital curves can change with extra broadening, through the 1e-5 filter on the rebound dos_up_gpu. Second, with §2a a Line/Path re-sort recolours the level-0 curves, and set_data does not recolour.
  - (a) The key becomes (mode, level, the curves' labels, feenstra, the energy-marker count). Labels cover the marker count, the element and its surviving orbitals. Colours are applied in place with set_color when they differ, so no clear.
  - (b) The same key with the level-0 colours in it (a re-sort clears and replots the spectrum).
  - (c) Today's key minus active_atom only.
  - R: (a).
- **decomp:Q-D12** (§2 Level 3 'Which orbitals' (the 1e-5 filter); PART2_DECISIONS A8 (no GPU work during a drag)): The filter runs on the GPU (cp.gradient twice, three cp.max with float() syncs per orbital). At level 3 in Line/Path, every drag motion calls _draw_spectrum (_move_line_markers), so evaluating the filter there breaks A8.
  - (a) Decomp.orbital_columns caches its result keyed on the element and the dos_up array object, compared with `is`. _apply_extra_broadening rebinds dos_up_gpu on every change, so the cache follows it.
  - (b) Compute it in _update_all and hand it to _draw_spectrum (Single Point computes no partitions, so it would need its own call).
  - (c) Recompute on every call (breaks A8 at level 3).
  - R: (a).
- **decomp:Q-D13** (§2 'One click, one step: … the window is redrawn once for it'): While RUN is off, _update_all returns before drawing the spectrum, but the curves stay on screen. Today a click then changes plot_level with no redraw, so later clicks act on curves of another level. For example, a level-2 curve clicked while the state says level 1 gives level 2 with the wrong selection.
  - (a) The spectrum, legend and Back clicks do nothing while RUN is off.
  - (b) As today: the state steps and the redraw waits for RUN.
  - (c) Redraw the spectrum and legend from the caches while stopped (the maps would not follow).
  - R: (a).
- **decomp:Q-D14** (§2 'Right click anywhere in the spectrum panel or the legend panel (on a curve, on an entry, or on empty space)'; today's _on_press toolbar test): There are two open points. First, qtplot's inaxes is the plot box only (_px_rect), not the title or tick labels around it, so 'anywhere' must mean the box or the whole cell. Second, today's _on_press ignores clicks while the toolbar's Pan/Zoom is on (toolbar.mode != ''); _on_pick does not. The plan names neither, nor what a middle click does.
  - (a) The plot box (event.inaxes), as today; all decomp clicks are ignored while Pan/Zoom is on; a middle click does nothing.
  - (b) The whole cell including title and ticks (needs a cell hit test qtplot does not give); Pan/Zoom not tested.
  - R: (a).
- **decomp:Q-D15** (§3 'It holds everything that depends on the level' vs the bullets listed under it): Three level tests stay in simulator.py: the cluster label `f"{quantity}: {active_element}" if plot_level >= 3` (now 2619, 2686), the heatmap titles `p_label.split()[-1] if plot_level >= 3` (_fill_heat_row, now 3056), and the map ylabels (_fill_map, now 3202-3205). The plan's bullets do not list them. With level 4 gone, `>= 3` reads as `== 3` and they keep working through the properties.
  - (a) Leave them as they are.
  - (b) Move them into ldos_decomp (for example a short label per partition and the cluster suffix).
  - R: (a), as the plan's bullets do not name them.
- **decomp:Q-D16** (Owner's ruling 8 ('… whatever the settings of the window is along with any information in the titles that was dropped for the sake of space'); PART3_QUESTIONS Q6): The decomp state is a window setting held in no control: the level, the active marker and the active element. The legend panel's titles are short words for space ('Parts' for Partition, 'Atoms' for Atom Types), and level 3's title drops the element; Single Point has no cluster label showing it.
  - (a) Decomp.info() gives one window line, e.g. 'decomp: Orbitals of C, marker 1' ('decomp: markers' at level 0). The simulator sets it with fig.set_info_line('decomp', …) whenever the state changes.
  - (b) The legend panel's title is given as 'Orbitals: C, marker 1' (and so on), and qtgui's short_title shows the short word and sends the rest to the info panel. This works only if the legend panel draws its title through Axes.set_title.
  - (c) Nothing (controls only).
  - R: (a) if Q6 is answered (a) or (b): it covers level 0 and does not depend on how the legend panel draws its title.
- **decomp:Q-D17** (§3 (new file); PART2_DECISIONS ruling 5 ('Each def's docstring starts with "Physics:" or "Machinery:"' in simulator.py, feenstra.py, quadrature.py)): Ruling 5 names simulator.py, feenstra.py and quadrature.py, not a new module. ldos_decomp.py is all display machinery, and the _draw_spectrum it serves is already tagged 'Machinery:'.
  - (a) Tag every ldos_decomp.py docstring 'Machinery:'.
  - (b) No tags.
  - R: (a), for consistency inside the package.
- **decomp:Q-D18** (Rule 1 (a change the plan does not name); P8 / A15 precedent): Several texts become false. The plan names none of them. - stm_from_dft_opt/README.md lines 66-68: 'a click on a curve of the Partitioned LDOS panel descends a level (partition, atom types, atoms, orbitals) and a click beside the curves ascends'. - The README's module table has no row for ldos_decomp.py. - _draw_spectrum's docstring ('the legends and the hover labels made', '_spec_key: … the element and atom of the atom levels'). The README is being edited by part 2 now.
  - (a) After part 2 commits, update these in the same terms (the new levels and click rules, a README row for ldos_decomp.py, the docstring).
  - (b) Leave them.
  - R: (a).
- **decomp:Q-D19** (§2 Level 3 'It leaves out a column whose DOS, and the DOS's first and second differences, all stay below 1e-5'): The code does not take plain differences. It uses cp.gradient twice (central differences, one-sided at the ends), and the test is on the element's atom-summed DOS of the up channel only (dos_up_gpu).
  - (a) Keep the code's test exactly, since the plan says 'the same ones the maps already show'.
  - (b) Use plain first and second differences (this changes which orbitals the maps show).
  - R: (a).
- **decomp:Q-D20** (§1 rule 3 / §7; PART2_DECISIONS (part 2 in progress)): Part 2 is still converting _draw_spectrum, the partitions and _redraw_map_slice to CuPy: cp.sum, quadrature.simpson(dx=e_step32), cp.asnumpy at plot time, a NumPy drag cache _line_disp. The decomp must take whatever arrays part 2 leaves. That is CuPy from _update_all and NumPy from the Line/Path drag path, which part 2's code passes to cp.sum and quadrature.simpson.
  - (a) Write ldos_decomp.py after part 2 commits, transcribing part 2's final array calls exactly; the integral comes in as the `norm` callable.
  - (b) Write it now against d5's NumPy code and merge later.
  - R: (a).

### External facts (verified)

- Qt QGraphicsItem: 'Items with the same Z value are stacked by insertion order'; items with low Z values are painted before items with high Z values; stackBefore() reorders the insertion order. (https://doc.qt.io/qt-6/qgraphicsitem.html)
- Qt Widgets do not deliver the MouseButtonPress that caused creation of a MouseButtonDblClick event (qwidgetwindow.cpp handleMouseEvent: `if ((event->type() != QEvent::MouseButtonPress) || !QMutableSinglePointEvent::isDoubleClick(event))`, comment QTBUG-25831). A double click on the canvas therefore arrives as press, release, double-click, release, and qtplot fires its press events once with dblclick=False and once with dblclick=True. (https://code.qt.io/cgit/qt/qtbase.git/plain/src/widgets/kernel/qwidgetwindow.cpp?h=6.8)
- QWidget::mouseDoubleClickEvent: the default implementation calls mousePressEvent(); the widget also receives mouse press and mouse release events in addition to the double click event. (https://doc.qt.io/qt-6/qwidget.html)
- pyqtgraph ViewBox.addItem parents each item to the view's childGroup (so an axes' items are siblings, stacked by Z then insertion order) and raises an item's Z to the ViewBox's Z + 1 only if it is lower. ViewBox.__init__ sets the ViewBox's own Z to -100, so zorder values of -99 and above reach Qt unchanged. (https://raw.githubusercontent.com/pyqtgraph/pyqtgraph/master/pyqtgraph/graphicsItems/ViewBox/ViewBox.py)
- matplotlib Axes.draw orders its artists with `artists = sorted(artists, key=attrgetter('zorder'))` (and Axes keeps _children in insertion order), so among equal zorders the artist added last is drawn on top. (https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/lib/matplotlib/axes/_base.py)
- Python's built-in sorted() is guaranteed to be stable. (https://docs.python.org/3/library/functions.html)

### Unverified

- CuPy raises IndexError for an out-of-bounds integer index (the claimed level-3 crash past column 8 in spin-polarised runs, now that f_ldos_raw and dos_up_gpu are CuPy). NumPy certainly raises; CuPy not checked.
- cp.sum and quadrature.simpson accept the NumPy drag cache _line_disp. Part 2's current _draw_spectrum passes it to them during Line/Path drags; CuPy's cupy.sum delegating to a.sum() is from memory, and quadrature.simpson's handling of NumPy input is part 2's in-progress code, not checked.
- The owner's installed Qt/PySide6 and pyqtgraph versions behave as the sources read: qtbase 6.8 branch, pyqtgraph master.
- The legend panel is a qtplot axes of the figure, so its presses arrive as button_press_event with inaxes set to its axes (PART3_QUESTIONS Q1 (a), not yet answered). The proposed _on_press legend branch assumes this; with a separate canvas the same Decomp.press would be called from that canvas's handler instead.
- matplotlib fires pick_event before button_press_event for the same click (the plan states it; not re-checked, and not needed: the design relies only on qtplot's order, which I read).
- Single Point's and Map's spec_disp is never None when _draw_spectrum runs (reasoned from _update_all's needs_spec path, not traced for every caller).

## Report: markers (Plan section 2a, Line and Path: the marker index follows the LDOS topo stripe (top to bottom), and each marker keeps its colour.)

### Code map

State read on 2026-11-07. `simulator.py` on disk is byte-identical to HEAD 791a805, which holds part 2's WIP commit e1995d7, and to my scratch snapshot. So the line numbers below are today's, and approximate. The plan's line numbers refer to b051547, and they match: its m_colors uses at about 2355, 2522 and 2593 are today's `_draw_spectrum` level 0 and `_make_line_markers`.

**Marker state (`Interactive_STM_Simulator.__init__`, about 963 to 1033)**
- `m_colors` (about 994): 10 hex colours.
- `marker_ratios = [0.25, 0.75]` (about 996). It is ascending, so index 0 is the LOWER marker on the stripe.
- `marker_coords` (about 997): Single Point and Map.
- `active_obj`, `_drag_moved`, `active_marker_idx = 0`, `plot_level`, `active_element`, `active_atom` (about 999 to 1003).
- Line and Path share one `marker_ratios` list. Single Point and Map share `marker_coords`. The two lists are independent.

**`run_interactive` (about 1050 to 1057)**
- With `marker_positions`, `marker_ratios` is each position projected onto p1→p2 and clipped to 0..1, in the order given. This happens in Path too.
- If the line has zero length, it is `list(np.linspace(0.1, 0.9, n))`, which is ascending.
- Nothing sorts the list.

**Markers count box**
- `_make_controls` (about 1321) creates `s_num_marks = controls.make(fig, 'Markers', value=len(marker_coords))`, wired to `_on_ui_change` (about 1328).
- `widgets.SPECS['Markers']` is a count with vmin 1 and vmax 10 (qtgui/widgets.py about 78).
- `controls.Count._accept` enforces the limits only on typed or stepped entries. `Count.set` neither clamps nor fires (qtgui/controls.py about 1191 to 1240). So more than 10 markers can only come from `marker_positions`.
- `_build_ui` (about 1629) puts `len(marker_ratios)` into the box in Line and Path, without firing.
- **Handler (`_on_ui_change`, about 2318 to 2324):**
  - an increase replaces every marker: `marker_ratios = list(np.linspace(0.1, 0.9, new_count))`, ascending, so existing markers are not kept;
  - a decrease truncates: `marker_ratios[:new_count]`;
  - `active_marker_idx` is not touched, so it can stay ≥ n;
  - then `_update_all(full_refresh=True)`.

**View change (`_on_mode_change`, about 2156 to 2160)**
- `active_marker_idx = 0`, `plot_level = 0`, then `_build_ui`.
- `marker_ratios` is kept across views.

**Where a Line / Path marker is drawn, and its colour today**
- **Placement topo (`ax_map`)**:
  - `self.marks`, one PathCollection for all markers, made per view in `_build_ui` (about 1658; picker 5).
  - `_update_all` sets its offsets and `set_facecolors(self.m_colors[:len(marker_ratios)])`: Line about 2403/2404, Path about 2410 to 2412.
  - `_place_line_markers` sets only its offsets.
- **Counterparts (`_make_line_markers`, about 3079 to 3105)**:
  - one dict per marker in `self._line_markers`: `'topo'` is a scatter on `ax_line_topo` (Line only); `'prof'` is an axvline on `ax_prof` (Tip Height); `'h'` is one axhline per heatmap plus one on `ax_stripe`;
  - colour `m_colors[i % len(m_colors)]` (about 3094), label `f'marker_{i}'`, picker 5;
  - made only when `_line_markers_n != len(marker_ratios)` (`_fill_heat_row`, about 3075), so only while running;
  - reset in `_clear_cluster` (about 1549).
- **`_place_line_markers` (about 3107 to 3130)** places the marks and every counterpart at `marker_ratios` in index order (stripe and heatmap y = `p_dist[int(r*(npts-1))]`; Line Topo y = `r*p_len`). It sets no colours.
- **`_move_line_markers` (about 3132 to 3142)** calls `_place_line_markers()`, then, while running, `_draw_spectrum(self._line_disp, None, self._line_geom[1])` in place, then `draw_idle`.
- **Spectrum level 0 (`_draw_spectrum`, about 2760 to 2769)**:
  - one curve per marker in index order, colour `m_colors[i % len]`, label `marker_{i}`, picker True;
  - key `_spec_key` (about 2716 to 2721) holds the mode, the level and n_marks, but no colours;
  - **the in-place path `_curve` (about 2733 to 2735) only calls `set_data`: the colour kwargs are ignored**;
  - the title shows `marker {active_marker_idx}` (about 2846; set in place when it changes).
  - Single Point / Map level 0: about 2770 to 2777.
- **Axis directions:**
  - the stripe: `ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len))` in `_fill_heat_row` (about 3013, only when p_len changes), and `p_dist = linspace(0, p_len, npts)` (Line about 2398, Path `_build_path_xy`). So top to bottom is the largest ratio first.
  - The Line Topo `set_ylim(0, p_len)` (about 2994).
  - The heatmap row never turns (`_unit_cell_axes`, about 1880), and nothing inverts these axes. qtplot `Axes` limit setting inverts only for descending limits (qtgui/qtplot.py about 5160 to 5176).
- **Path differences:** `ax_line_topo` is None in Path (`_build_heat_row`, about 2891), so Path has no LDOS topo panel today. The marks are placed at `p_xy[int(r*(npts-1))]`.

**Per-marker caches in Line / Path**
- Index-aligned with `marker_ratios`: `self.marks` (offsets and facecolors), `self._line_markers` (artists and labels), the level-0 curves in `self._spec_lines` (reset in `_clear_panels`, about 1588), and `_spec_title` (follows `active_marker_idx`).
- No per-marker data cache exists. Every marker spectrum is read from `_line_disp` (per line point, `_update_all` about 2584) at `int(r*(npts-1))`.
- `cached_spec_ldos`, `cached_spec_ldos_gpu`, `cached_marker_coords` and `cached_spec_current` belong to Single Point / Map only.

**Drag**
- **`_on_pick` (about 3323):**
  - `self.marks` → `active_obj = ('mark_map', event.ind[0])` (about 3338). In Line / Path this does not change `active_marker_idx`.
  - The `ax_spec` branch (about 3343 to 3362) comes BEFORE the mark_dynamic branch. It sets `active_marker_idx` from a level-0 curve's `marker_` label.
  - The mark_dynamic branch (about 3363 to 3372) catches a Line2D, `_RefLine` or PathCollection labelled `marker_*` and sets `active_obj = ('mark_dynamic', i)` and `active_marker_idx = i`. While running it redraws the spectrum in place from `_line_disp`, then `draw_idle`.
  - `Figure._pick` (qtplot.py about 7094) fires one `pick_event` per pickable child under the mouse, in `ax._children` order.
- **`_on_press` (about 3374 to 3383)** handles `ax_spec` only. It makes no redraw for marker drags: part 1 moved that redraw into the mark_dynamic branch, and b051547's `_on_press` called `_update_all` on every press.
- **`_on_motion`:**
  - mark_map in Line (about 3442 to 3447): projection, clip.
  - mark_map in Path (about 3448 to 3451): nearest sample k/(npts-1).
  - Assignment `marker_ratios[idx] = new`, then `_move_line_markers` (about 3454 to 3460).
  - mark_dynamic (about 3470 to 3483): `ax_prof` uses x/p_len; `ax_ldos`, `ax_stripe`, `ax_line_topo` and the heatmaps use y/p_len.
  - The index in `active_obj` is fixed for the whole drag.
- **`_on_rel` (about 3496 to 3555):**
  - part 2's X3 emarker branch is written (about 3497 to 3545): `order = np.argsort(self.map_e_targets, kind='stable')`; if not the identity, it permutes `map_e_idx`, `map_e_targets`, `cached_ld_up` / `_dn`, their `_gpu` copies and `cached_current` / `_E`; it permutes `_spec_emarkers` and relabels them `emarker_{i}`; it redraws moved maps with `_redraw_map_slice`; then one `draw_idle`.
  - The second branch (about 3546 to 3553) handles `'end'`, `'path_node'` and Single Point / Map `'mark_map'` with `_update_all()`.
  - **There is no branch for a Line / Path marker release** (mark_map in Line / Path, mark_dynamic). Then `_drag_moved = False` and `active_obj = None`.

**Single Point / Map colours by index (to stay):** `_update_all` about 2415; `_draw_spectrum` about 2776; `_fill_map` about 3199; `_place_map_markers` about 3263 (recolours only when the count changes).

**qtplot setters available (qtgui/qtplot.py):**
- `Artist.set_label` (about 2083);
- `Line2D.set_color` (about 2261; `_build` reuses the same item, calling `setData`);
- `PathCollection.set_offsets` / `set_facecolors` (about 2441 to 2456; `set_color` is an alias; edgecolors are kept);
- `_RefLine.set_color` (about 2585, `setPen` in place).

**sim_wrapper (HEAD == disk)**
- `send_to_decomposition` (about 206 to 268):
  - Single Point / Map colours `sim.m_colors[i % len]` (about 236);
  - Line / Path iterates `sim.marker_ratios`, takes positions from `sim.marks.get_offsets()[i]`, and colours `sim.m_colors[i % len(sim.m_colors)]` (about 248 to 260), sent as `marker_colors` to `DecompWindow` (decomp_window.py about 310, 615).
- At HEAD it no longer reads or sets `active_marker_idx` or `plot_level`: part 2's X1 removed `capture_state` and the restore in `_instantiate`.
- At d5e08a4, `_instantiate` (about 186 to 209) set both as plain attributes before `run_interactive`, and assigned `sim.marker_ratios = list(post[...])` after it.

### Proposal

Each item names the plan bullet it implements. Everything is in `simulator.py` unless named otherwise. Names in backticks are proposed names, for approval.

**P1. Where the kept colour lives** (2a "The colour stays with the marker")
- A new list `self.line_marker_colors`, parallel to `marker_ratios`: entry i is marker i's colour, a colour string taken from `m_colors`.
- Two helpers:
  - `_free_marker_color()`: the first colour of `m_colors` that is not in `line_marker_colors` (2a, Claude's reading). For the case where none is free, see Q2a-10.
  - `_marker_colors()`: the markers' colours in index order for the current view. Line / Path give `line_marker_colors`. Single Point / Map give `[m_colors[i % len(m_colors)] for i in range(len(marker_coords))]`, today's colours.
- **Lifecycle:**
  - **Initialised** in `__init__`, right after `marker_ratios` (about 996).
  - **New run:** formed again in `run_interactive` wherever `marker_ratios` is formed again (about 1050 to 1057). The order and colours of a freshly formed set are Q2a-1.
  - **View change:** not reset, because `marker_ratios` is kept (Q2a-13). `active_marker_idx` goes to 0, as in plan section 2.
  - **Markers count change:** P6.

**P2. Every panel that draws a marker uses its kept colour** (2a "Every panel that draws a marker ...")
- **LDOS topo panel, Tip Height, heatmaps and stripe:** `_make_line_markers` (about 3094) takes `color = self.line_marker_colors[i]`.
- **Placement topo `self.marks`:** `_update_all`, Line (about 2404) and Path (about 2412), sets `set_facecolors(self.line_marker_colors)`. The plan's list does not name this panel (Q2a-4).
- **Spectrum level-0 curves (Line / Path):** they take colour i from `_marker_colors()`, through `ldos_decomp`'s level-0 curve list (Q2a-6). The in-place path applies the colour as well as the data (`Line2D.set_color`), because today `_curve` ignores the colour in place (about 2733).
- **Legend panel level-0 entries:** the samples take the same list, in index order.
- Single Point / Map: nothing changes (P7).

**P3. The re-sort on release** (2a "When"; "The sort is stable"; "top to bottom is marker_ratios from largest to smallest")
- In `_on_rel`, a new `elif` beside X3's emarker branch and the existing end / node branch: `self.active_obj is not None and self._drag_moved and self.mode in ('Line', 'Path') and self.active_obj[0] in ('mark_map', 'mark_dynamic')`.
- It calls a new `_resort_line_markers()` before `_drag_moved` and `active_obj` are cleared.
- The order is `order = sorted(range(n), key=lambda i: -float(self.marker_ratios[i]))`. Python's sort is stable. `np.argsort(-r, kind='stable')` is equivalent; **not** `np.argsort(r, kind='stable')[::-1]`, which reverses ties.
- If `order` is the identity, nothing more is done: the drag already drew everything (`_move_line_markers`).
- No sort during a drag: the index held in `active_obj` stays valid until release.
- No sort on a line-end or path-node release: the ratios do not change, and `p_dist` grows with r, so the stripe order is unchanged.

**P4. The permutation, applied together** (2a "What the permutation is applied to")
- `marker_ratios = [marker_ratios[j] for j in order]` (it stays a list: sim_wrapper and `_toggle_run` iterate it).
- `line_marker_colors`, permuted the same way.
- `active_marker_idx`: `a = min(active_marker_idx, n - 1)`, then the new index is `order.index(a)`, so the same marker stays active. It is set through the simulator attribute. Plan section 3 keeps that attribute settable; its setter must only store the value (no redraw, no level change), because P5 redraws once.
- **The per-marker caches, exactly:**
  1. `self._line_markers`, the artists of each marker: `'topo'` (Line only), `'prof'`, and every `'h'` line including the stripe's (Q2a-8);
  2. `self.marks`, its offsets and facecolors, written in the new index order;
  3. the level-0 curves in `self._spec_lines` (Q2a-7);
  4. the legend panel's level-0 entries;
  5. `_spec_title`, which follows the new active index.
- **Not permuted, because they are not per marker:** `_line_disp`, `_line_geom`, `_line_markers_n`, `_spec_key` (the count and level are unchanged, so no rebuild), every `cached_*` (per line point, or Single Point / Map only), `_stripe_lc`, `_prof_line`, the heatmap meshes. `active_obj` and `_drag_moved` are cleared after.

**P5. One redraw for the new order, with no rearrangement or rebuild** (2a "The panels")
- `self.marks.set_facecolors(self.line_marker_colors)`.
- The `_line_markers` artists take the new order: recoloured by index under Q2a-8 (b), or permuted and relabelled under (a).
- Then `_move_line_markers()`:
  - `_place_line_markers()`, all positions;
  - while running, the spectrum updated in place from `_line_disp` (the key is unchanged, so there is no `ax_spec.clear()`), with the legend panel's level-0 entries updated in the same call (plan section 2 "Live"; same count and no text, so its width does not change);
  - one `draw_idle`.
- Not called: `_update_all`, `_build_heat_row`, `_make_line_markers`, `_rearrange`, `_build_ui`, and no GPU work.

**P6. The Markers count box** (2a, Claude's reading)
- After the change, the list is sorted with the same stable key.
- Colours go in index order after the sort, each marker taking `_free_marker_color()`. Today's handler places all markers anew, so every marker is new and the colours come out as `m_colors` by index (Q2a-2).
- **Removal:** both lists are truncated to the first k (the top k on the stripe). The remaining markers keep their colours. The removed colours become free. No re-sort is needed (Q2a-3).
- `_update_all(full_refresh=True)` follows, as today: the count changed, so `_make_line_markers` and the spectrum rebuild.

**P7. Single Point and Map are unchanged** (2a "Unchanged")
- They have no sort and no kept colours.
- They keep `m_colors` by index at about 2415, 2776, 3199 and 3263.
- Their `_on_rel` `mark_map` branch stays as it is (`_update_all`).

**P8. sim_wrapper** (rule 4)
- No change to the file.
- `active_marker_idx` and `plot_level` are not read at HEAD (X1). Had the earlier restore stayed, setting them would still work as plain attributes.
- `marker_ratios` and `marks.get_offsets()` stay in step in index order, so the positions sent are right.
- Its Line / Path colours stay `m_colors` by index, so they differ from the kept colours (Q2a-5).

**P9. Constraints on part 3's other pieces, which the drag needs**
- The replacement of `_on_pick`'s `ax_spec` branch must stay before the mark_dynamic branch and take every pick in `ax_spec`. The level-0 curves are labelled `marker_*`, so they would otherwise start a marker drag.
- The legend panel's sample artists carry no `marker_*` label with a picker, for the same reason.

### Questions

- **markers:Q2a-1** (2a 'The rule' (index 0 is the marker highest on the stripe) vs 2a 'When' (re-sorted after a release)): The code starts unsorted: the default marker_ratios=[0.25, 0.75] (__init__ about 996), run_interactive's projection of marker_positions in the order given, and its fallback linspace(0.1, 0.9, n), which is ascending. The plan sorts only on release, plus the count box in Claude's reading. So until the first release, index 0 is the LOWEST marker, against the rule. The plan also does not say how a freshly formed set is coloured.
  - (a) Sort only where the plan says: on release, and the count box (Claude's reading).
  - (b) Also sort wherever a set is formed (__init__, run_interactive), then colour by index after the sort: index 0 at the top gets m_colors[0].
  - (c) Also sort there, but colour in the order given first (marker_positions[i] gets m_colors[i], as in Single Point / Map), and permute the colours with the sort.
  - R: (b): the rule then holds from the first draw. It also matches P6, so after every fresh start the level-0 entries read m_colors in order from top to bottom.
- **markers:Q2a-2** (2a Claude's reading: markers added by the Markers count box are sorted the same way; an added marker gets the first colour of m_colors that no marker has): _on_ui_change (about 2321 to 2322) does not add markers. On any increase it replaces every marker with list(np.linspace(0.1, 0.9, new_count)), so no existing marker keeps its place, and 'an added marker' beside kept ones never happens. The reading assumes existing markers stay.
  - (a) Keep today's placement. Every marker is new: sort top to bottom, then give each marker the first free colour in index order. The colours come out as m_colors by index, and the old markers' colours are not kept because the markers are not kept.
  - (b) Change the placement: existing markers stay where they are with their colours, and only the new ones are added with the first free colours, then sorted. The owner must say where an added marker goes; the plan does not.
  - R: (a): the plan names no change to where the box puts markers (rule 1). (b) only if the owner gives the new placement rule.
- **markers:Q2a-3** (2a permutation / 'Unchanged'; the count box removal (the plan is silent)): A decrease truncates marker_ratios[:k] (about 2324). Once the list is sorted, that always removes the markers lowest on the stripe; today it removes the highest indices of whatever order there is. active_marker_idx is never clamped. After removal it can be ≥ n: the spectrum then shows marker min(i, n-1) (about 2705) while the title prints the raw index (about 2846). P4 needs a valid index, and uses min(i, n-1).
  - (a) Truncate both lists to the first k (the bottom markers go). The remaining markers keep their colours, the removed colours become free, and active_marker_idx is left as today; P4 clamps only inside the re-sort.
  - (b) As (a), and also clamp active_marker_idx to n-1 at the removal, so the title matches the curve shown.
  - (c) Remove a different marker (for example the one dragged last). The plan does not name one.
  - R: (a): it is today's rule applied to the sorted list, with no new behaviour. (b) is a small fix the plan does not name.
- **markers:Q2a-4** (2a 'Every panel that draws a marker (the LDOS topo panel, the stripe, the heatmaps, the Tip Height plot, the spectrum and the legend panel)'): Two panels do not match the list. The placement (global) topo ax_map also draws every Line / Path marker (self.marks, coloured m_colors[:n] at about 2404 and 2412), and the list does not name it. Path has no LDOS topo panel today (ax_line_topo is None in Path, about 2891): there the counterparts are only the stripe, the heatmaps and the Tip Height lines.
  - (a) The placement topo's marks also take the kept colours. Path uses the kept colours on the panels it has.
  - (b) Only the panels listed. The placement topo keeps colour by index, so its colours would no longer match the other panels.
  - R: (a): 'every panel that draws a marker' covers it, and (b) would show the same marker in two colours.
- **markers:Q2a-5** (Rule 4 (sim_wrapper keeps working with no changes) vs 2a 'the colour stays with the marker'): sim_wrapper.send_to_decomposition sends Line / Path marker colours as sim.m_colors[i % len(sim.m_colors)], by index (about 260). After a re-sort, the decomposition window would draw marker i's curve in a colour that differs from the simulator's kept colour. Separately, rule 4's description of sim_wrapper is out of date: at HEAD (part 2's X1) it no longer reads or sets active_marker_idx or plot_level, so that part of rule 4 holds trivially.
  - (a) No sim_wrapper change. Accept that, in Line / Path, the decomposition window shows index colours.
  - (b) The owner waives rule 4 for one line: sim_wrapper takes sim.line_marker_colors[i] (or sim._marker_colors()[i]) in its Line / Path branch.
  - (c) No sim_wrapper change: in Line / Path, m_colors itself is kept permuted (marker i's colour at m_colors[i], free colours after it) and restored to the palette order for Single Point / Map. This changes what m_colors means, and plan section 3 passes it in as shared data.
  - R: (b) if the owner allows the one-line change; otherwise (a). (c) is not recommended.
- **markers:Q2a-6** (Section 3 'Shared data ... is passed in: ..., m_colors, ...'): With m_colors passed in, ldos_decomp would colour the level-0 curves and the legend entries by index (m_colors[i]). In Line / Path, 2a needs the kept colours.
  - (a) The simulator passes the markers' colours in index order (_marker_colors(): the kept colours in Line / Path, m_colors by index in Single Point / Map) in place of m_colors.
  - (b) Pass m_colors and the kept colours both, and ldos_decomp chooses by view.
  - R: (a): one list, already in index order, and ldos_decomp needs no knowledge of views.
- **markers:Q2a-7** (2a 'every per-marker cache' and 'redrawn once ... as the Map re-sort'; section 2 'only the top one counts: ... drawn last'): The level-0 spectrum curves are kept in _spec_lines in drawing order. Today's in-place update only calls set_data (about 2733), so after a re-sort the colours would be stale. Two ways to give the curves the new order.
  - (a) In place, by index: curve i gets marker i's data and colour (Line2D.set_color). The curve list, the drawing order and ldos_decomp's level-0 list stay index-aligned, and the labels never change.
  - (b) X3's pattern: permute _spec_lines, relabel each curve marker_{i}, then set_data in the new list order. The curves then stack in their creation order, not index order.
  - R: (a): ldos_decomp's top-curve rule and its curve-to-option mapping stay index-aligned, and nothing is relabelled.
- **markers:Q2a-8** (2a 'every per-marker cache' and 'as the Map re-sort' (X3 permutes _spec_emarkers and relabels them)): For the counterpart artists in _line_markers (the Line Topo scatter, the Tip Height axvline, the heatmap and stripe axhlines), the plan does not say whether the artists are permuted with the markers or recoloured where they stand.
  - (a) X3's pattern: _line_markers = [_line_markers[j] for j in order], and every artist relabelled marker_{i}. The colours ride with the artists; _place_line_markers then re-sets the same positions.
  - (b) By index: each index's artists are recoloured to that marker's kept colour (PathCollection.set_facecolors, _RefLine.set_color), and _place_line_markers moves them. Labels stay right because artist i is always marker i.
  - R: (b): every panel follows one rule (artist i shows marker i, the same as self.marks, which has to be written by index, and the spectrum under Q2a-7 (a)), labels cannot go stale, and which overlapping marker the mouse picks stays as today. (a) is the literal X3 pattern if the owner prefers it.
- **markers:Q2a-9** (2a 'top to bottom is marker_ratios from largest to smallest'; 'markers at the same place keep their relative order'): The stripe, the heatmaps and the Tip Height lines draw a marker at the sample p_dist[int(r*(npts-1))]. The Line Topo and the placement topo draw it at the exact r. Two markers with different r can share a stripe sample, so 'the same place' is either exactly equal ratios or the same stripe sample.
  - (a) Sort key: marker_ratios exactly, as the plan writes; ties are only exactly equal ratios.
  - (b) Sort key: the stripe sample int(r*(npts-1)); markers on the same sample keep their relative order.
  - R: (a): the plan's own words, and it matches the order on the panels that draw the exact place.
- **markers:Q2a-10** (2a Claude's reading: 'the first colour of m_colors that no marker has'): The box's vmax is 10, the same as len(m_colors), so the box alone never runs out of colours. But run_interactive(marker_positions=...) can give more than 10 markers (Count.set does not clamp), and then no colour is free. Today the spectrum and the counterparts cycle m_colors[i % 10], while self.marks gets m_colors[:n], which holds only 10 colours.
  - (a) When no colour is free: m_colors[i % len(m_colors)], today's colour for that index.
  - (b) Refuse more than 10 markers (a new check, which the plan does not name).
  - R: (a).
- **markers:Q2a-11** (Section 2 'Clicks outside the spectrum and legend panels do what they do today ... the mark_dynamic branch of _on_pick and the redraw _on_press makes for them'): Today _on_press makes no redraw for marker drags; it handles ax_spec only. Part 1 moved that redraw into _on_pick's mark_dynamic branch, which redraws the spectrum in place from _line_disp. b051547's _on_press called _update_all on every press.
  - (a) Keep today's behaviour: the redraw in the mark_dynamic branch, and none in _on_press.
  - (b) Bring back a redraw in _on_press.
  - R: (a): it is 'what they do today'. (b) would add a second redraw per press, against section 2's one click, one redraw.
- **markers:Q2a-12** (2a (the plan does not say what a view change does to the kept colours); section 2 'A view change resets the level to 0 and the active marker to 0'): Line and Path share marker_ratios, and a view change keeps it, including Line ↔ Path and a trip through Single Point or Map. The plan does not say whether the kept colours are reset on a view change.
  - (a) Keep them with marker_ratios: they are the same markers. Only active_marker_idx goes to 0, the top marker.
  - (b) Reset them to m_colors by index at every view change.
  - R: (a): the plan's rule is that a marker keeps its colour wherever it moves.
- **markers:Q2a-13** (PROGRESS ruling 9 and the status line 'simulator.py edits wait until part 2's workflow has finished'): Part 2 is still writing _on_rel (X3 is there now, HEAD e1995d7, marked unverified, in progress). The new Line / Path branch (P3) sits in the same function, next to X3's branch.
  - (a) Add the 2a branch only after part 2's _on_rel is final, beside X3's branch, without touching it.
  - (b) Add it now.
  - R: (a).

### External facts (verified)

- NumPy: a stable sort keeps items with the same key in the same relative order. kind='stable' selects the stable algorithm (timsort or radix sort). stable=True guarantees that the relative order of equal values is kept, and internally selects kind='stable'. (https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
- numpy.argsort: 'stable=True: the returned array will maintain the relative order of a values which compare as equal'; it selects kind='stable' internally. (https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)
- Python: 'The built-in sorted() function is guaranteed to be stable', meaning the relative order of elements that compare equal is not changed. (https://docs.python.org/3/library/functions.html#sorted)
- pyqtgraph ViewBox: 'By default, the positive y-axis points upward on the screen. Use invertY(True) to reverse the y-axis.' qtplot inverts only for descending limits, so ax_stripe's ylim (0, p_len) puts 0 at the bottom. (https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/viewbox.html)

### Unverified

- What pyqtgraph's ScatterPlotItem.setData does when the brush and pen lists are shorter than the points. With more than 10 markers, self.marks.set_facecolors(m_colors[:n]) gives qtplot's _refresh 10 brushes for n points (it zips them against the n-long mask). This exists today and was not checked against pyqtgraph's source.
- Whether part 2's X1 changes to sim_wrapper.py are final (HEAD e1995d7 is marked 'unverified, in progress'). If capture_state and the post-run restore came back, its `sim.marker_ratios = list(post['marker_ratios'])` would leave line_marker_colors at a different length, and P1 would then need a guard that this proposal does not include.
- That a qtplot Axes' _children list is in insertion order, which Figure._pick follows. I read only that _pick iterates ax._children; I did not trace how the list is built.
- sm4_browser was not consulted for this area: the plan names no sm4_browser source for 2a, and I did not check whether it has a comparable marker-order or kept-colour rule.

## Report: legend (The legend panel: plan section 5, the legend-panel rules of section 2, and the legend panel's part of the saves in sections 5 and 6)

### Code map

## State of the files read

- **Repository:** branch `claude/jolly-babbage-ppmbdj`, HEAD `791a805`; the working tree was clean while I read.
- **simulator.py:** HEAD holds part 2's committed WIP (`e1995d7`). Part 2 changed no line in the layout code I map here (`_make_controls` through `_rearrange`, about lines 1291–1945) apart from one line near 1387.
  - `_draw_spectrum` (about 2692) has part 2's `quadrature.simpson` edits; its structure is part 1's.
- **qtgui copies:** `stm_from_dft/qtgui/` and `MOPmatcher_mem/matcher/qtgui/` differ only in `__pycache__`. The root `qtgui/` is the pre-part-1 layer.
- **Line numbers:** approximate. Package copy first; root copy or plan line in brackets.

## qtgui architecture (stm_from_dft/qtgui)

### The figure and its cells
- **One pyqtgraph scene.** `FigureCanvas(pg.GraphicsView)` (qtplot.py ~6111) is "one pyqtgraph scene in pixel coordinates, the Qt controls as child widgets".
  - The window's Qt layout sits on `canvas.viewport()` (`Figure.set_layout` ~6817).
- **Cells are placeholders, not widgets that hold axes.** `Figure.cell` (~6833) makes a `PanelCell` (~3717):
  - an empty widget, transparent to the mouse (`WA_TransparentForMouseEvents`, ~3730), whose rectangle the layout sets;
  - `rect_px` (~3885) maps it onto the viewport;
  - an `Axes(cell=…)` (~4221) adds its `ViewBox` to the scene and places it at the cell's rectangle less its tick axes and title (`_cell_px` ~4416, `_px_rect` ~4503, `_place` ~4654);
  - `min_px` and `inner_w` are pixels at the old 100 DPI.
  - **So no axes can live inside a `QScrollArea`.**
- **Placement callbacks:** `Axes.on_placed(cb)` (~4627) runs after every placement: a resize, a text zoom, an axis re-measured.

### The rows and blocks
- **`PanelRow`** (~3932) and **`RowWidget`** (~4150): rows of shaped, free and fixed items.
- **`Figure.align_row`** (~7459): the boxes of a row share one top and bottom (`_px_rect`'s band).
- **`controls.FlipBox`** (~1758): free panels side by side, or stacked, by aspect ratio.
- **`controls.MapBlock`** (~1869):
  - wrapping equal-sized map cells, plus one `extra` cell (the spectrum) in the last row's free slot (`set_groups` ~1926; placed in `_arrange` ~2124);
  - it calls `c._decor()` on the extra (`_tb_of` ~1968, `_metrics` ~1920), so the extra must be a `PanelCell`.
- **`controls.Arrangement`** (~2180):
  - shaped groups plus a free widget;
  - with one group, the candidates `'h'` (free widget at the right, full height) and `'v'` (free widget below), ~2441–2466;
  - `_free_min` (~2295) takes the larger of `free.minimumSizeHint()` and `free_min()`.
- **`OrientedPanes`** (~350): the simulator does not use it now.

### Qt controls beside or over a cell
- **`PanelWithBox`** (~303): `AbovePanel` (~299) over the cell. `UnderPanel._follow` (~280) sets its margins from `Axes.on_placed`, so the box centres on the plot box. This is the LDOS h box (`simulator._pair` ~1454).
- **`CropPanel`** (~1805): the Topo h and Crop/Uncrop boxes above the cell, plus two `RideBand`s (~492). The cursor boxes are positioned by hand inside their band (`set_pos`, from `simulator._place_cursor_boxes` ~2039 via `on_placed`).
- **`Titled`** (~1716): the LDOS label over the heatmap row and the map block.
- **No Qt control overlaps a plot box anywhere.**

### Info panel dock
- `FigureManager.__init__` (~6492–6504; root ~6400):
  - a `QPlainTextEdit` with `NoWrap`, `qfont(7.5)`, fixed height of 4.2 lines;
  - in a `QDockWidget` at the bottom, hidden while empty;
  - its text comes from `Figure.info_text` (~6683).

### Text measurement, the 10 pt floor, `em_px`
- **Width:** `QFontMetricsF(font).horizontalAdvance(text)` (`Axes.legend` ~5811, 5828, 5843), or a drawn text's `t.item.boundingRect().width()` (`_cell_px` ~4429, `MapBlock._title_w` ~1998).
  - `_TextItem` (~1357) adds a document margin of `1/(1000/72)*em_px()` on each side.
- **Floor:** `qfont` (~420): `pt = max(10.0, font_pt(size)*FONT_SCALE*zoom)`. Controls get it too (`widgets._font`).
- **`em_px`** (~337): `QFontInfo` of a 100 pt `base_font`, divided by 10. Part 1's convention: a pixel constant c becomes `c/(1000/72)*em_px()`, times `font_zoom` for sizes that follow text.

### An axes with no ticks or frame
- `Axes.axis('off')` (~4769): hides both tick axes, so `_decor_px` is 0; removes the border; sets the background to `None`.
  - `set_facecolor('w')` (~4811) can restore white.
  - mplexport writes `m.axis('off')` when `not ax._axison` (mplexport ~214).
- `set_navigate` (~4792) is a no-op.
- `_apply_toolbar_mode` (~5417) turns the mouse on for every axes except colour bars while Pan or Zoom is on.

### Artists available
- **`Line2D`** (~2156): any line style, including dash tuples (`mkpen` ~527).
- **`Text`** (~3250): data, axes or figure coordinates; `cell=`; `bbox`.
  - A data text is a `TextItem` in the view box, unscaled by the view.
- **`LineCollection`** (~3031): string line styles only (~3053).
- **`PolyCollection`** (~3177): fill only, edges ignored.
- **No rectangle patch:** `Patch` (~2537) is only a legend proxy without an axes, and `Axes.patches` is always `[]` (~4308). sm4_browser has `Circle` with `add_patch`, but no Rectangle.

### Events
- **Press** (`mousePressEvent` ~6207; root 6113): `figure._pick(e)` first, then `button_press_event`.
  - `_pick` (~7094; root 6990) fires once for every pickable child under the mouse.
- **Double click** (~6215): fires both again with `dblclick=True`.
- **Wheel** (~6289): fires `scroll_event` (`guiEvent` is the `QWheelEvent`), then `super().wheelEvent`.
- **`_event`** (~6182): `inaxes` comes from `_axes_at` (~7085, box test), and `xdata`/`ydata` from the view range, inversion included.

### Saves today
- **mplexport** `_to_matplotlib` (~121):
  - laid out at the design size (`at_design_size` ~7049);
  - each visible axes via `_axes` (~156) at `_pos` (`_px_rect` as figure fractions), its artists via `_artist` (~282), limits from `viewRange`;
  - figure texts of the plot area only;
  - written with `bbox_inches='tight'` (~466).
- **`export_csv`** (~7572): every visible `Line2D` of every visible axes.

## sm4_browser (eb9bb62)

### `controls._MarkerBox` (~1908)
- A `QScrollArea` with `setWidgetResizable(True)`, `NoFrame`, and the horizontal bar always off. The vertical bar keeps Qt's default policy (as needed).
- Its rows sit in a transparent inner `QVBoxLayout` (spacing 2, trailing stretch).
- `sizeHint` = the content's height, capped by `cap()`. When capped, its width grows by the scroll bar's width. `minimumSizeHint` = the smaller of the content and the cap.
- The window places it in its bottom control strip (map.py ~392; line.py ~2883) with `set_height_cap` at half of box plus panel height (map.py ~405, line.py ~2900).
- The wheel and the bar are Qt's own. **It grows with its rows; the legend panel must not** ("takes only the space the layout gives it").
- The only other scroll area is `TextPanel` (`QPlainTextEdit`, both bars as needed, ~1162).

### Saves
- **`Figure.savefig`** (qtplot ~5496) has the `'vector'`, `'qt-svg'`, `'image'` and `'csv'` routes. Helpers: `export_root` (~5535), `_export_rect` (~5548), `export_info` (~5564), `window_image` (~5576: `root.render` with `DrawChildren` at `WINDOW_IMAGE_DPI/DPI`), `_render_scene` (~5651).
- **`export_vector`** (~5694) draws the whole window and appends the full info text below it. The plan says this repository's qt-svg is "exactly as shown" instead.
- **mplexport** `_to_matplotlib` (~105):
  - the figure is `(W, H+H_info)` px at `q.DPI`; `_pos(ax, mfig)` measures y from the top;
  - figure texts sit in a canvas band `Bbox(0, H_info/(H+H_info), 1, 1)`;
  - `_info_block(mfig, fig, H, H_info)` writes the info lines starting at H;
  - texts are written at `t._size*FONT_SCALE*font_zoom`, with no 10 pt floor;
  - sm4_browser's `qfont` itself has no floor.
- **`settings_text`** (controls ~1264) lists only controls with a `setting()`. A non-stateful Button (Back) and a raw `QScrollBar` add no line.

## simulator.py: how each view builds the spectrum

- **Controls and connections:** `_make_controls` (~1291) makes every control once. `mpl_connect` pick/motion/release/press ~1427–1430 (plan ~1194–1197); no `scroll_event` connection.
- **Kept widgets:** `_kept_widgets` (~1526), `_detach` (~1502), `_empty_layout` (~1510).
- **Clearing:** `_clear_panels` (~1570) removes every axes but no figure text.
- **`_cell_axes`** (~1495): a free panel, `(axes, cell)`.
- **`_build_ui`** (~1594):
  - **Single Point** (~1634–1639): `self.ax_spec, spec_cell = self._cell_axes()`; `groups, free = [topo], spec_cell`. The Arrangement has `free_scored=True`, with candidates `'h'` and `'v'`.
  - **Line / Path** (~1640–1659): `ax_prof` and `ax_spec` cells in `FlipBox((prof_cell, 25), (spec_cell, 10), spacing=10)`, the free widget beside or under the heatmap row.
  - **Map** (~1660–1687): `self.ax_spec, spec_cell = self._cell_axes()`; `self._spec_cell`; `self._map_block.set_groups([], None, extra=spec_cell)`, then `_build_map_block` (~3181) with `extra=self._spec_cell`; `free=None`.
  - The Arrangement is made at ~1695.
- **`_curves_readable`** (~1717): sets each curve axes' cell minimum to `CURVE_BOX_MIN_PX` (220×140 at 100 DPI, × zoom) plus its decor, and returns `(sum of widths + spacing, max of heights)`.
- **`_draw_spectrum`** (~2692; plan `_update_all` ~2347–2420):
  - redraws the curves when `_spec_key` changes, else updates them in place;
  - legends at ~2779, 2790, 2798, 2809, and 2836–2838 (`add_artist(leg1)` ~2837);
  - title "Partitioned LDOS: marker i at … Å" (~2846);
  - Map energy markers are `axvline`s labelled `emarker_i` (~2861).
- **Click and drag handlers:**
  - `_on_pick`: spectrum branch ~3343–3362 (plan ~2793–2812); mark_dynamic ~3363.
  - `_on_press` (~3374; plan ~2819–2823).
  - `_on_motion` energy-marker drag ~3484; `_on_rel` ~3496 (X3 re-sort).
  - `_on_mode_change` (~2156; plan ~1897).
- **Colours and partitions:**
  - `m_colors` is set at ~994.
  - `_get_partitions` (~1124; plan ~896–938) builds level-3 labels as `f"{active_element} {orb}"`.
  - Orbital names (~545–571) can be as long as `dx2-y2_down`.

## Against the plan (not in my area, noted)

- **sim_wrapper (rule 4):** the plan's rule 4 cites `_instantiate`/`capture_state` reading `plot_level` and `active_marker_idx`.
  - At `d5e08a4` they did (~186–264). In HEAD, part 2's X1 removed `capture_state`, and sim_wrapper no longer reads either. The attributes still have to stay.
- **Marker colours:** sim_wrapper still colours Line/Path markers by index, `sim.m_colors[i]` (~236, 260). That will no longer match section 2a's kept colours.

### Proposal

Each item names the plan bullet it implements. Wherever an item depends on an open choice, it states the recommended option (R) of the question named; the other options are in the questions.

## L-1. What the panel is
*Section 5 "How it is built"; section 2 "The panel" and "Its order"; questions L1, L2, L10.*

**A composite widget**, built in simulator.py next to `_placement` and `_pair` (L10 R). It holds a `QGridLayout`:

| Row | Column 0 | Column 1 |
|---|---|---|
| 0 | the title cell (spans both columns) | |
| 1 | the Back button (spans both columns) | |
| 2 | the stack cell | the vertical scroll bar |

**Row 0, the title cell** (L2 R): a `fig.cell(...)` holding one figure text:
- `fig.text(0.5, 0.5, title, cell=title_cell, ha='center', va='center', fontsize=10, fontweight='bold')` (L15 R);
- the cell's height is set to the text's height at each text zoom.

**Row 1, the Back button:** `controls.Button(fig, 'Back')`, made once in `_make_controls` and kept across views (`_kept_widgets`, `_detach`), as the LDOS h box is.

**Row 2, column 0, the stack:** a `PanelCell` with a qtplot Axes:
- `ax.axis('off')`: no ticks and no frame, and no decor, so its box is its cell;
- `ax.set_facecolor('w')` (L6 R).

**Row 2, column 1:** a `QtWidgets.QScrollBar(Qt.Vertical)` (L1 R). Its size policy has `setRetainSizeWhenHidden(True)` (L4 R).

**Per view:** the composite is built with the view in `_build_ui` (part 1's trigger 1). Its two cells go in `_panel_cells`. `_clear_panels` removes the stack axes with the others, and also removes the title text, because figure texts are not removed by `ax.remove()`.

## L-2. How the stack is drawn
*Section 5 "its samples drawn as lines, its labels as texts and its button boxes as rectangles"; section 2 "Its entries" and "Clickable entries are buttons".*

**Units:** data units are on-screen pixels.
- `ax.set_xlim(0, w)` and `ax.set_ylim(s + h, s)`: y runs downward; w and h are the box from `ax._px_rect()`; s is the scroll value.
- Entry k fills `y` from `k·row_h` to `(k+1)·row_h`.

**Each entry draws:**
- **the sample:** `ax.plot([x_s, x_s + L], [y_c, y_c], color=…, ls=…, lw=…)` with no picker. It uses its curve's colour, style and width, including level 3's dash tuples, which only `Line2D` draws.
- **the label:** `ax.text(x_s + L + gap, y_c, label, fontsize=10, ha='left', va='center')`. No text at level 0.
- **the button box** (levels 0–2, L6 R):
  - the outline is a `LineCollection` holding one closed 5-point polyline;
  - the fill is a `PolyCollection` quad;
  - colours are those of `controls.FRAME_CSS` (fill `#f0f0f0`, border `#c8c8c8`);
  - every box spans the stack's width less a small inset, so the buttons form one column of equal width.
- **Level 3 rows:** sample and label only, with no box (the plan's recorded reading).

**New named constants** (pixels at the old 100 DPI, converted with `/(1000/72)*em_px()*font_zoom`, part 1's convention):
- `LEGEND_SAMPLE_PX = 26` (qtplot's legend sample is 26 px);
- `LEGEND_PAD_PX = 4`;
- `LEGEND_GAP_PX = 4` (qtplot legend's `horSpacing`);
- `LEGEND_ROW_GAP_PX = 2`.

**Row height:** the label font's `QFontMetricsF.height()` plus the text item's two document margins, plus 2·`LEGEND_PAD_PX`, plus `LEGEND_ROW_GAP_PX`.

**Kept for clicks:** the entries' boxes as drawn, `self._legend_drawn = [(x0, y0, x1, y1, entry_index), ...]`, together with the entries themselves.

## L-3. Title, Back button, and the stack as the only scrolling part
*Section 2 "Its order", "Its title", "Its Back button"; questions L5, L16.*

- **Title texts:** none at level 0, "Parts", "Atoms", "Orbitals", taken from ldos_decomp's title.
- **Fixed in place:** both live in their own layout rows, so they never move with the stack.
- **Back button visibility:** shown at levels 1–3. At level 0 it is hidden with its place kept, and the title text is empty with its place kept (L5 R).
- **Back button look and behaviour:**
  - gray via the stylesheet `QPushButton { background: #c8c8c8; }` (L16 R);
  - its double-click event is ignored, so a double click is one Back (L16 R);
  - its click calls the same back action as a right click.

## L-4. Scrolling
*Section 5 "When the entries don't fit" and "Scrolling"; question L1.*

**The bar is configured the way `QScrollArea::updateScrollBars` configures its bar:**
- range `0 … content − visible`, where `content = n_entries·row_h` and `visible = h`;
- page step = `visible`;
- single step 20 (QScrollArea's own constructor value; it is Qt's number, not converted);
- shown exactly when the range is not empty (the as-needed rule).
- The value is clamped whenever the range changes.

**Moving the view:**
- `bar.valueChanged` → `ax.set_ylim(v + h, v)`. Only the view range changes; no artist is touched.

**The wheel:**
- a new `mpl_connect('scroll_event', ...)`;
- when `event.inaxes` is the stack axes and `|dy| ≥ |dx|`, the handler calls `QtCore.QCoreApplication.sendEvent(bar, event.guiEvent)`;
- that is exactly what `QAbstractScrollArea::wheelEvent` does, so Qt's own slider code takes the step (wheel scroll lines × single step; Ctrl or Shift: a page).

**Kept current:** `ax.on_placed` re-reads h and resets the range. A new text zoom rebuilds the stack (L-7).

**Height:** the panel never asks the layout for room to show its list. Its minimum height is title + Back + one row (L17 R). This differs from `_MarkerBox`, which grows with its rows; the scrolling itself is the same.

## L-5. Width
*Section 5 "Width"; section 2 "Never wider than the panel"; questions L4, L12.*

**One fixed width** (L4 R), set with `setFixedWidth`:
- the largest entry, `2·border + 2·pad + sample + gap + max label advance + 2·doc margins`, plus the inset;
- plus the bar's `sizeHint().width()`, always reserved.

**Labels counted:** every label any level can show for the loaded data, measured at the window's label font (`qfont(10, zoom=fig.font_zoom)`):
- `'Au'`, `'Molecule'`;
- each atom type;
- `f"{t} {orb}"` for every atom type × every `self.orbitals` column.

The width changes only when the text zoom changes (a resize).

**Title check:** each title's `QFontMetricsF(qfont(10, weight='bold', zoom)).horizontalAdvance(title)` plus its two doc margins must not exceed the panel width. The plan's stop condition is about this; the code session cannot measure it (L12).

## L-6. Placement in each view
*Section 5 "Where" and "One sub-section"; questions L3, L7, L18.*

**The spectrum's sub-section:** `spec_sub = controls.widget_of(controls.row(spec_cell, legend))`.
- The spectrum cell has stretch 1; the legend has a fixed width.
- **Vertical extent** (L3 R): the legend's top and bottom margins follow `ax_spec`'s plot box through `ax_spec.on_placed`, as `UnderPanel._follow` does. The panel then spans the spectrum's plot box, in the same row.

**Single Point:**
- `groups, free = [topo], spec_sub`.
- Both arrangements, `'h'` and `'v'`, and `free_scored` stay as they are.

**Line / Path:**
- `controls.FlipBox((prof_cell, 25), (spec_sub, 10), spacing=10)`.
- The spectrum and its legend form a sub-sub-section inside the 1D sub-section. The stretches are unchanged (L18 R).

**Map** (L7 R): `MapBlock` takes the legend panel as a side item of its extra cell.
- The spectrum keeps one map's size.
- The legend sits right of the spectrum at that row's band.
- The block counts it in its width; the colour bars stay clear of it.

**`_curves_readable`:** adds the legend's width plus the row spacing to the width it returns, in Single Point, Line and Path.

## L-7. Entries, order, and live rebuilds
*Section 2 "Its entries", "Short labels", "Level 0", "Order", "Not entries", "Live".*

**Source:** the entries come from ldos_decomp (L-9), in its order.
- Level 0: samples only, in marker index order. In Line and Path, the kept colours and the stripe order of section 2a.
- Faded totals and Map's energy markers are never entries.

**Rebuild trigger:** after `_draw_spectrum` has drawn or updated the curves, the panel compares ldos_decomp's `(title, entries)` with what it drew. When they differ it rebuilds:
- `ax.clear()` (the axis stays off), entries drawn anew, limits set, bar range reset;
- the scroll goes to the top on a level change, and is kept (clamped) otherwise (L13 R);
- the title text is set and the Back button shown or hidden.

A new `font_zoom`, seen in `on_placed`, also rebuilds the stack and resets the width.

## L-8. Clicks
*Section 5 "Clicks"; section 2 "The clicks", "The two ends do nothing", "Double click", "One click, one step"; question L9.*

In `_on_press` (`button_press_event`). The legend's artists have no picker, so `pick_event` never fires for them. The branch runs only when `event.dblclick` is false and the toolbar mode is `''`.

**What counts as in the panel:**
- `event.inaxes` is the stack axes; or
- the press's viewport pixel lies inside the panel's content rectangle (title, Back row, bar column).

A right press on the Back button reaches the canvas, because `QAbstractButton` ignores presses that are not left presses.

**Left click on the stack:**
- look up the entry whose drawn box (`self._legend_drawn`) contains `(xdata, ydata)`;
- if found, pass that entry to ldos_decomp's left-click (it acts at levels 0–2);
- a level-3 row (no box) or a gap does nothing.

**Right click anywhere in the panel:** ldos_decomp's back (levels 1–3; nothing at level 0). This includes the bar (L9 R: the bar's own context menu is turned off and its right press handled by a small event filter).

**One decision, one redraw:** the entry is chosen once from the list kept as drawn, before any redraw; the update runs once.

## L-9. The interface needed from ldos_decomp.py
*Section 3 "The legend entries"; another agent designs it.*

**Per state:**
- the title: `str`, or `None` at level 0;
- whether Back is available (levels 1–3);
- an ordered list of entries. Each entry carries:
  - `color` (matplotlib spec);
  - `ls` (string or dash tuple);
  - `lw` (points);
  - `label` (`str`, or `None` at level 0);
  - `clickable` (`bool`);
  - an opaque `option`, handed back on a click.

**Actions:** an entry left-click action taking the `option`, and a back action. Each returns whether the level changed, so the window redraws once.

**Static:** the set of every label any level can show, from `atomtypes` and `orbitals` (needed for L-5).

## L-10. The legend panel in the saves
*Section 5 "In the saves"; section 6 "The two scrollable panels … in each save"; questions L11, L14, L20, L21.*

**PDF, EPS and vector SVG** (the ported sm4_browser mplexport):
- The stack axes is rebuilt like every axes: `Line2D` → plot, `Text` → text, `LineCollection`, `PolyCollection` → `_quads`, axis off.
- **Full-list hook** (L11 R): the window sets `ax.export_extent = fn`, returning `(content_h_px, (y_top, y_bottom))`, much as it sets `home_callback`.
  - `_axes` then uses the rectangle `(x0, y0, w, max(h, content_h))` and the full list's y limits.
  - `_to_matplotlib` computes `H_top = max(H, y0 + content_h)` over such axes and makes the figure `(W, H_top + H_info)`.
  - Figure texts stay in the canvas band, which is the figure's top H.
  - `_info_block` starts at `H_top`, so the full info panel sits below the legend panel's bottom whenever that reaches past the canvas.
- **Texts at screen size** (L20 R): each text is written at its on-screen size, qfont's floor included, so labels stay inside their boxes.
- **Left out:** the Back button and the bar, which are controls; their places stay empty (L21). The title, a figure text, is written at its place.

**Qt-drawn SVG and window images (PNG, TIFF):**
- They render the window as shown (`QWidget.render` with its children): the stack scrolled where it is, the bar and Back as on screen.
- Nothing is added for the legend. Section 6's own change is that the qt-svg must not append the full info text.

**CSV:** the legend axes is left out (L14 R).

### Questions

- **legend:L1** (5 'How it is built' + 'Scrolling (identical to sm4_browser's scroll areas)'): In this qtgui a figure is one pyqtgraph scene, and a PanelCell is only a mouse-transparent placeholder whose rectangle the axes is drawn at (FigureCanvas ~6111, PanelCell ~3717). A qtplot axes therefore cannot sit inside a QScrollArea, as sm4_browser's _MarkerBox does. This refines PART3_QUESTIONS Q1.
  - (a) A QScrollBar at the stack's right edge, configured as QScrollArea configures its own bar: range 0..content−visible, page step = visible, single step 20, shown only while the range is not empty. The wheel over the stack is forwarded to that bar with sendEvent, which is what QAbstractScrollArea::wheelEvent does. Both move the axes' y range. The wheel and bar behaviour are Qt's own code; the panel stays part of the figure.
  - (b) A second FigureCanvas inside a real QScrollArea. The panel is then not part of the figure, so the vector saves, the cell layout, the clicks and the info panel all need extra code to treat it as one.
  - (c) A real QScrollArea laid over the stack cell. Its content is an empty spacer as tall as the list, its viewport is transparent to the mouse, and its bar's value drives the axes' range. Its event routing is unverified, and the wheel would still have to be forwarded.
  - R: (a)
- **legend:L2** (2 'Its order' (title, Back button, stack) + 'Its title … saved with the figure' + 5 'a qtplot axes'): The plan calls the panel a qtplot axes, yet requires the title above the Back button, which a Qt control must provide, both fixed while the stack below them scrolls. One axes cannot do this: pyqtgraph's ViewBox clips its children only to its whole box, so scrolled entries would pass under the title and the Back button. The plan does not say where the title lives.
  - (a) The panel is a composite: a title cell holding a figure text (fig.text(..., cell=)), the Back button, the stack axes and the scroll bar. Both mplexports already write figure texts at their place; this needs no qtgui change.
  - (b) The title is the stack axes' own set_title, with a pad as tall as the Back button, and the button is placed by hand in that gap. mplexport would also need to pass the title's pad: today it calls m.set_title(text, fontsize) only, so in the file the title would drop onto the stack.
  - (c) Two axes: a title axes holding one axes text, plus the stack axes.
  - R: (a)
- **legend:L3** (5 'Where: … in the same row and at the row's height'): 'The row's height' is open. The spectrum's cell holds its title above the box and its x label below it. PART3_simulator_views_layout_decisions.md (a later handoff, context only) records the owner's reading of a row's height as the plot box's.
  - (a) The legend panel spans the spectrum's plot box, from its top to its bottom. Its margins follow ax_spec's box through on_placed, as UnderPanel._follow does.
  - (b) The legend panel spans the spectrum's whole cell, from the top of its title band to the bottom of its x label.
  - R: (a): it matches the reading recorded for the later handoff, so that handoff need not change it.
- **legend:L4** (5 'Width: as wide as its largest entry' vs part 1 5.2 'Panels move only on three triggers'): Two things could change the width on a click, which would move the spectrum on a level step. First, a width taken from the current level's entries. Second, the scroll bar's width added only while it shows, as _MarkerBox.sizeHint adds it. The bar appears on level changes, for example at level 3 with many orbitals. This refines PART3_QUESTIONS Q3.
  - (a) One width: the largest entry any level can show for the loaded data. That covers 'Au', 'Molecule', each atom type, and each atom type with each of self.orbitals, whether or not the 1e-5 filter shows it. The scroll bar's width is always reserved, with its place kept while it is hidden (setRetainSizeWhenHidden). Nothing moves on a click; the width changes only with the text zoom, which only a resize changes.
  - (b) The current level's largest entry, with the bar's width only while the bar is shown. The panel and the spectrum change width on level steps and when the bar appears, which breaks 5.2.
  - (c) As (a) but without reserving the bar: the bar is drawn over the right edge of the stack. That departs from the look of sm4_browser's scroll area.
  - R: (a)
- **legend:L5** (2 'Its title … level 0: none' and 'Its Back button … the marker level (0) has none'): The plan does not say whether the title's and the Back button's places stay at level 0. If they collapse, the stack's box moves and resizes inside the panel on the step between levels 0 and 1, which part 1 5.2 forbids.
  - (a) Both places kept at level 0 and left empty.
  - (b) Both places collapsed at level 0, so the stack takes their room.
  - R: (a)
- **legend:L6** (5 'its button boxes as rectangles' + 2 'clearly a clickable button'): qtplot has no rectangle artist: Patch is only a legend proxy and Axes.patches is always []. The box's look is not specified either.
  - (a) Existing artists: the outline is a LineCollection holding one closed polyline, and the fill is a PolyCollection quad. mplexport exports both and export_csv takes neither. Look: fill #f0f0f0 and border #c8c8c8 (controls.FRAME_CSS, 'the same frame in every window'), on a white stack (ax.set_facecolor('w') after axis('off')).
  - (b) A new qtplot Rectangle patch, with Axes.add_patch and an mplexport branch, like sm4_browser's Circle. This is a qtgui addition the plan does not name.
  - (c) As (a), but outline only, with no fill, on the window's gray.
  - R: (a)
- **legend:L7** (5 'Where … in every view' and 'One sub-section' (Map)): In Map the spectrum is MapBlock's extra cell, one map's size in the last row's free slot (controls.MapBlock ~1926/2124; part 1 5.3 keeps it so). MapBlock calls _decor() on the extra, so the extra must be a PanelCell, and it has no place for a second item. A legend beside the spectrum needs a MapBlock change the plan does not name. Separately, PART3_simulator_views_layout_decisions.md (Map, item 2) moves Map's spectrum to the top row in a later handoff, which would make this change moot.
  - (a) MapBlock takes a side item for its extra cell: the legend panel right of the spectrum, at that row's band. It is counted in the block's width and its placement, and the colour bars are kept clear of it. The spectrum keeps a map's size.
  - (b) The extra becomes a composite (spectrum plus legend) of one map's width. The spectrum then becomes narrower than a map by the legend's width, and MapBlock must still accept a non-PanelCell extra.
  - (c) The spectrum and its legend leave the block and become Map's free widget in the Arrangement. That changes Map's layout and goes against part 1 5.3.
  - R: (a)
- **legend:L8** (5 'Scrolling … the mouse wheel' (with the toolbar's Pan / Zoom)): While Pan or Zoom is on, _apply_toolbar_mode (~5417) turns the mouse on for every axes except colour bars. In that mode the wheel would also zoom the stack, a drag would pan it, and Home would restore whatever limits were set last. The plan does not cover this.
  - (a) Make qtplot's Axes.set_navigate (a no-op now) keep an axes out of Pan and Zoom, as matplotlib does, and call it on the stack axes. Side effect: the topo colour bar, which already calls set_navigate(False), would stop panning too.
  - (b) Fix it inside the simulator: a home_callback that puts the scroll limits back, and the limits reset after any pan or zoom. This needs no qtgui change, but leaves a moment where the panel is panned.
  - (c) Leave it as it is.
  - R: (a)
- **legend:L9** (2 'Right click anywhere in … the legend panel … goes back' vs 5 'identical to sm4_browser's scroll areas'): QScrollBar handles only the left (and optionally middle) button, so the canvas never sees a right press on it. Its context menu holds 'Scroll here', 'Top', 'Bottom' and the rest, as in sm4_browser. The owner's two rules conflict only on the bar itself.
  - (a) Keep Qt's own scroll-bar behaviour, as in sm4_browser: a right click on the bar opens its menu.
  - (b) A right click on the bar goes back too: the bar's context menu is turned off, and an event filter on the bar handles the right press.
  - R: (b)
- **legend:L10** (rule 2 'how the legend panel is built and placed' + 'which functions … go into qtgui'): The plan names only sm4_browser functions for qtgui, and only simulator.py and ldos_decomp.py for the simulator package. It does not say where the panel's widget code lives.
  - (a) In simulator.py, next to _placement and _pair. qtgui gets only the generic hooks of L7, L8, L11 and L14.
  - (b) A generic class in qtgui/controls.py, in both package copies (the root qtgui as section 6's Q5 decides).
  - (c) A new module in the simulator package.
  - R: (a)
- **legend:L11** (5 'In the saves: the PDF, EPS and vector SVG hold the full list' + 6 'moved down below the bottom of the legend panel'): sm4_browser's mplexport rebuilds every axes at its on-screen box with its on-screen limits, and writes the info block at the canvas height H. Holding the full list needs a hook sm4_browser does not have. Separately, without matplotlib the 'vector' save falls back to export_vector, which draws the window as shown and so cannot hold the full list.
  - (a) The window sets an attribute on the axes: ax.export_extent = fn returning (content_h_px, (y_top, y_bottom)), much as it sets home_callback. mplexport then: (1) draws the axes with height max(h, content_h) and those limits; (2) makes the figure (W, max(H, y0 + content_h) + H_info); (3) starts _info_block at that top. The no-matplotlib fallback stays as shown, with a console note.
  - (b) The same, through a documented qtplot method, Axes.set_export_extent(fn).
  - (c) As (a), and the Qt fallback also draws the full list.
  - R: (a)
- **legend:L12** (2 'If a title would still be wider than the panel at the window's font, the code session stops and reports it'): Under rule 3 the code session runs no window, so it cannot measure a title at the window's font. The check could be written in code: QFontMetricsF(qfont(10, weight='bold', zoom)).horizontalAdvance(title), plus the text item's two document margins, compared with the panel width. Running it is a check rule 3 does not allow. This restates PART3_QUESTIONS Q2. Under L4(a) the widest entries are level-3 labels such as 'Au dx2-y2_down', so the four titles are very likely narrower, but that is not measured.
  - (a) The owner checks it on his machine.
  - (b) The window prints one console line at build time when a title is wider than the panel. This is a check, so it needs the owner's leave under rule 3.
  - (c) Both.
  - R: (a)
- **legend:L13** (2 'Live: it is rebuilt … whenever the level or the data changes'): The plan does not say where the stack's scroll position goes after a rebuild.
  - (a) Back to the top on a level change; kept (clamped to the new range) when the same level's list is redrawn.
  - (b) Always back to the top.
  - (c) Always kept (clamped).
  - R: (a)
- **legend:L14** (6 'CSV, the curves' data'): export_csv writes every visible Line2D of every visible axes, so the legend's sample lines would become extra two-point columns. They must be Line2D: level 3 uses dash tuples, and LineCollection takes only string line styles (~3053).
  - (a) export_csv skips an axes the window marks, for example ax.csv = False. This is a small qtgui change.
  - (b) Accept the extra columns.
  - R: (a)
- **legend:L15** (2 'Its title' / 'Its entries' (font sizes not given)): The plan gives no size or weight for the labels or the title.
  - (a) As today's spectrum legend: labels at 10 pt, and the title bold at the same 10 pt (qtplot's legend-title rule).
  - (b) The title at the axes-title size, 12 pt.
  - R: (a)
- **legend:L16** (2 'Its Back button … a gray background, not white … says Back' + 'Double click: … acts as one left click'): Three things are open. (1) The exact gray. (2) Whether 'Back' gets a widgets.SPECS entry, which would be a qtgui change. (3) Double clicks: Qt's default double-click handler repeats the press and QAbstractButton does not override it, so a double click would very likely give two clicks and two steps back (inferred, not run).
  - (a) Gray #c8c8c8 (the box frames' border colour). No SPECS entry: controls.Button(fig, 'Back') with the default spec, font size 8. The button's double-click event is ignored, so a double click is one Back.
  - (b) As (a), with a 'Back' entry added to widgets.SPECS.
  - (c) Leave Qt's double click as it is: two steps back.
  - R: (a)
- **legend:L17** (5 'the panel takes only the space the layout gives it'): The plan does not give the panel's smallest height. Whatever it is, it must never raise the 1D row's minimum.
  - (a) Title + Back + one entry row, below the spectrum's readable minimum.
  - (b) No minimum (0).
  - R: (a)
- **legend:L18** (5 'One sub-section … In Line and Path … a sub-sub-section … inside the 1D sub-section'): FlipBox((prof_cell, 25), (spec_cell, 10)) shares the 1D room by stretch. With the legend inside the spectrum's item, the spectrum becomes narrower than today by the legend's width.
  - (a) Keep the 25:10 stretches.
  - (b) Change the stretches so the spectrum keeps today's width.
  - R: (a): the plan names no change to them.
- **legend:L19** (ruling 8 (info panel: '… any information in the titles that was dropped for the sake of space') × 2 'Its title' / 'Level 0: no text'): The legend titles are short words in place of the old ones ('Parts' for 'Partition', 'Atoms' for 'Atom Types'), and level 0 shows no text, so the marker index is not written. Whether the info panel should carry these is open; it ties to PART3_QUESTIONS Q6 and the ruling-8 agent.
  - (a) The window adds one line of its own (set_info_line) with the level's full word and the selection, for example 'decomp: Partition, marker 2', 'Atom Types, marker 2' or 'Orbitals of C, marker 2'. It is updated with the legend.
  - (b) No line: the short words are the owner's own words.
  - R: (a), coordinated with the ruling-8 design.
- **legend:L20** (6 (sm4_browser's mplexport) × 5 'saved with the figure'): The legend's boxes are sized from the on-screen fonts, which include qfont's 10 pt floor. sm4_browser's _text writes t._size*FONT_SCALE*font_zoom with no floor. Below the floor, the labels in the file would be smaller than their boxes. This ties to PART3_QUESTIONS Q4.
  - (a) The port writes each text at its on-screen size (qfont's floored size, converted with part 1's pt2px).
  - (b) Port as is.
  - R: (a)
- **legend:L21** (2 'Its Back button … a Qt control, so the saves leave it out' vs 6 'In the Qt-drawn SVG and the window images: exactly as shown'): The Qt-drawn SVG and the window images draw the window with its controls (sm4_browser's 'controls included'). The Back button and the scroll bar therefore appear in those saves.
  - (a) Left out of the PDF, EPS and vector SVG, with their places empty. Shown as on screen in the Qt-drawn SVG and the window images.
  - (b) Hidden for the Qt-drawn SVG and the window images too.
  - R: (a)

### External facts (verified)

- QAbstractScrollArea's verticalScrollBarPolicy and horizontalScrollBarPolicy both default to Qt::ScrollBarAsNeeded, and the vertical scroll bar sits next to the viewport. (https://doc.qt.io/qt-6/qabstractscrollarea.html)
- QAbstractScrollArea::wheelEvent sends the wheel event to the horizontal bar when |angleDelta().x()| > |angleDelta().y()|, otherwise to the vertical bar (QCoreApplication::sendEvent). An as-needed bar is shown when its minimum < maximum. (https://raw.githubusercontent.com/qt/qtbase/6.8/src/widgets/widgets/qabstractscrollarea.cpp)
- QScrollAreaPrivate::updateScrollBars sets vbar->setRange(0, widget height − viewport height) and vbar->setPageStep(viewport height). The QScrollArea constructor sets vbar->setSingleStep(20) and hbar->setSingleStep(20). widgetResizable makes the area resize its widget to avoid scroll bars where possible. (https://raw.githubusercontent.com/qt/qtbase/6.8/src/widgets/widgets/qscrollarea.cpp)
- QAbstractSlider's wheel handling scrolls by QApplication::wheelScrollLines() × (delta/120) × the effective single step, accumulating fractions. With Ctrl or Shift it scrolls by up to one page step. The event is accepted only if the value changed. (https://raw.githubusercontent.com/qt/qtbase/6.8/src/widgets/widgets/qabstractslider.cpp)
- QSizePolicy::setRetainSizeWhenHidden(true) means the layout is not changed when the widget is hidden (default false). (https://doc.qt.io/qt-6/qsizepolicy.html)
- QAbstractButton::mousePressEvent calls e->ignore() and returns for any button other than Qt::LeftButton. The file defines no mouseDoubleClickEvent override. mouseReleaseEvent calls click when hitButton(position) holds. (https://raw.githubusercontent.com/qt/qtbase/6.8/src/widgets/widgets/qabstractbutton.cpp)
- An ignored mouse event is propagated up the parent widget chain until a widget accepts it or an event filter consumes it (unless WA_NoMousePropagation). (https://doc.qt.io/qt-6/qmouseevent.html)
- QWidget::mouseDoubleClickEvent's default implementation calls mousePressEvent(), and the widget also receives the press and release events. (https://doc.qt.io/qt-6/qwidget.html)
- QWidget::render's default flags are DrawWindowBackground | DrawChildren, and DrawChildren renders the widget's children recursively. (https://doc.qt.io/qt-6/qwidget.html)
- QScrollBar::mousePressEvent returns without handling unless the button is left (or middle, with the style's SH_ScrollBar_MiddleClickAbsolutePosition). QScrollBar overrides contextMenuEvent with a menu of Scroll here, Top, Bottom, Page up, Page down, Scroll up and Scroll down, when the style's SH_ScrollBar_ContextMenu hint is set. (https://raw.githubusercontent.com/qt/qtbase/6.8/src/widgets/widgets/qscrollbar.cpp)
- By default a QEvent's isAccepted() is true (subclasses may clear it). (https://doc.qt.io/qt-6/qevent.html)
- With QGraphicsItem::ItemClipsChildrenToShape, an item clips the painting of all its direct and indirect descendants to its own shape. (https://doc.qt.io/qt-6/qgraphicsitem.html)
- pyqtgraph's ViewBox sets ItemClipsChildrenToShape on itself. Its wheelEvent calls ev.ignore() and returns when mouse pan/zoom is not enabled on either axis. (https://raw.githubusercontent.com/pyqtgraph/pyqtgraph/master/pyqtgraph/graphicsItems/ViewBox/ViewBox.py)
- pyqtgraph's TextItem displays unscaled text: it cancels the parent's scale in updateTransform, so the text keeps its size inside a scaled ViewBox. (https://raw.githubusercontent.com/pyqtgraph/pyqtgraph/master/pyqtgraph/graphicsItems/TextItem.py)

### Unverified

- That a right press on a QScrollBar is consumed and never reaches the canvas. Inferred from QScrollBar::mousePressEvent returning without ignore() and QEvent's default accepted state (both verified); the full delivery path was not traced.
- That a double click on a QPushButton emits clicked() twice. Inferred: QWidget's default mouseDoubleClickEvent calls mousePressEvent, QAbstractButton defines no override, and the release clicks. QPushButton's own source was not checked.
- That a press ignored by the canvas's child widgets (the Back button, container QWidgets) reaches FigureCanvas.mousePressEvent with the position in viewport coordinates. QGraphicsView's viewport routing was not read in Qt's source; the simulator's clicks through PanelWithBox and other containers today suggest it.
- That forwarding the QWheelEvent with sendEvent from inside the scroll_event callback scrolls the bar exactly as QAbstractScrollArea's own forwarding does, and that the event then continuing into super().wheelEvent has no other effect.
- That QAbstractSlider.setRange clamps the current value into the new range. The proposal clamps explicitly anyway.
- The pyqtgraph version on the owner's machine: the ViewBox and TextItem behaviour above was read from pyqtgraph master.
- Whether matplotlib's axis('off') also hides the axes' face patch in the file. It does not matter: the export draws on a white figure either way.
- Whether the four titles ('Parts', 'Atoms', 'Orbitals') at the title font are narrower than the widest entry: this cannot be measured without running Qt (L12).
- How the Arrangement, FlipBox and MapBlock actually size the panel on screen (the legend's fixed width, the following margins): read from the code only, not run (rule 3).

## Report: port (Plan section 6: sm4_browser's info panel and saves in every qtgui window, with the owner's ruling 8 (the info panel lists every widget/toggle setting, the window's own settings, and everything cut from the titles))

### Code map

State read: VASP_for_SPM HEAD 791a805, a clean tree; simulator.py as part 2's WIP commit e1995d7 left it. sm4_browser at eb9bb62 (HEAD). Line numbers are approximate. "copy" means `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/qtgui/` (its .py files are byte-identical to `MOPmatcher_mem/matcher/qtgui/`; only `__pycache__` differs). "root" means `/home/user/VASP_for_SPM/qtgui/`. The plan's numbers match root: SAVE_FORMATS 6313, FigureManager 6375 with the info widget at 6398-6402, info_text 6579, savefig 7376.

## 1. sm4_browser pieces (eb9bb62) and their counterparts here

### qtplot.py

- **Module docstring, ~70-89**: info panel, saves, settings section.
  - Copy ~60-96 and root: "The publication export lays a window out at its design size (Figure.at_design_size)" and the old save list. Text only.
- **`WINDOW_IMAGE_DPI = 720`, ~259**: none in copy or root. New constant.
- **`NavigationToolbar` Save tooltip, ~4627-4629**: copy 6373.
- **`SAVE_FORMATS`, comment ~4663-4670, list 4671-4679**: 7 entries:
  1. "PDF, vector, editable: the plots as on screen and the full info panel (*.pdf)" 'vector'
  2. EPS, same, 'vector'
  3. SVG, same, 'vector'
  4. "SVG, the whole window as Qt draws it, controls included, full info panel (*.svg)" 'qt-svg'
  5. "PNG, the window at 720 dpi (*.png)" 'image'
  6. "TIFF, the window at 720 dpi (*.tif)" 'image'
  7. "CSV, the curves' data (*.csv)" 'csv'

  Copy 6404-6417 / root 6313-6323 has 9 entries: PDF/EPS/SVG/PNG/TIFF 'publication', PNG/SVG 'screen', PNG 'screenshot', CSV.
- **`NavigationToolbar._save`, 4681-4706**: copy 6419-6445.
  - Differences: the copy drops 'publication' when matplotlib is missing (sm4 drops 'vector'), and calls `savefig(..., dpi=300 if how == 'publication')`; sm4 passes no dpi.
  - Copy-only parts to keep: the `mode` property (~6385), the `_rotated` swap in `show_coords` (~6447-6466), and part 1's em_px icon and coords sizes (6355, 6380).
- **`FigureManager.__init__`, 4738-4772**:
  - info widget 4749-4762: `LineWrapMode.WidgetWidth`, height `pt2px(7.5)*FONT_SCALE*4.2`;
  - hook 4770-4772: `self.window.sm4_settings_changed = figure._settings_changed`;
  - docstring 4729-4736.
  - Copy 6469-6523 (info 6490-6503): NoWrap, height `QFontMetricsF(font).lineSpacing()*4.2` (part 1), no hook. Root 6375-6402: NoWrap, sm4's height.
- **`FigureManager.set_info`, 4790-4802**: copy 6525 is identical.
- **`FigureManager.show`, 4804-4809**: calls `self.figure._settings_changed()`.
  - Copy 6539-6553 (part 1): measures `_EM[0] = _measure_em(self.window)` and connects `screenChanged` (`_screen_changed`, 6555). Both have to be kept together.
- **`_settings_text = None`, module global with comment, 4848-4852**: none here.
- **`Figure.__init__` end, 4909-4911**: `self.export_info_panel = getattr(self.manager, 'info', None)`.
  - Copy 6610-6663 has no such line. It does have `fill_screen` and `_size_override` (the latter used only by `at_design_size`).
- **`Figure._set_info`, 4915-4924**: ends with `self._settings_changed()`. Copy 6666-6677 starts the same 50 ms `QTimer` inline, so the behaviour is the same.
- **`Figure._settings_changed`, 4926-4930**: the coalesced push. None here.
- **`Figure.info_text`, 4936-4951**: title details, then window lines, then, in a FigureManager window, `"\nsettings\n" + _settings_text(window)`. Copy 6683-6688 / root 6579: title details and window lines only.
- **`Figure._push_info`, 4953-4968**: copy 6690 is identical.
- **`Figure._apply_zoom` info font, 5047-5052**: copy 6795-6800 keeps part 1's font-metrics height. Nothing to port.
- **`Figure.savefig`, 5496-5533**: 'vector' goes to `_mplexport.export`, or `export_vector` without matplotlib; 'qt-svg' goes to `export_vector`; 'image' (every other extension, `dpi` ignored) goes to `window_image`; 'csv' goes to `export_csv`. An unknown `how` falls through to `window_image`.
  - Copy 7480-7530 has 'publication' (mplexport at 300 dpi, raster too), 'screen' (pyqtgraph exporters), 'screenshot' (`canvas.grab()`, no dock) and 'csv'.
- **`Figure.export_root`, 5535-5546**: none here.
- **`Figure._export_rect`, 5548-5562**: canvas geometry, plus the dock when it is shown docked. None here.
- **`Figure.export_info`, 5564-5574**: `export_info_fn`, else the panel's text. None here.
- **`Figure.window_image`, 5576-5603**: `k = dpi / DPI`, a `QImage` with `root.render(p, QPoint(0,0), QRegion(rect), DrawWindowBackground|DrawChildren)`, dots per metre set. None here.
- **`export_csv`, 5630**: copy 7572 is identical.
- **`_render_scene(fig, painter)`, 5651-5674**: renders `export_root()` over `_export_rect()` with its children (controls included), caches off, `QuadItem.EXPORTING`.
  - Copy 7593 has `_render_scene(fig, painter, W, H)`: `scene.render` only (no controls), with a WINDOW_BG fill.
- **`export_vector(fig, path)`, 5694-5770**: `_export_rect` plus the full info text below it (`_mplexport._info_lines` / `_info_geometry`); PDF through QPicture then QPdfWriter; SVG through QSvgGenerator; EPS through Ghostscript.
  - Copy 7631-7680: the scene only, at `_canvas_size()`, with no info block.
- **Module-end binding `from sm4_Browser import mplexport as _mplexport`, 5944-5950**: the copy imports `from . import mplexport` inside savefig (~7506); root uses `from qtgui import mplexport` (7404).
- **Unchanged (identical in both)**: `Axes.set_title` → `_set_info` (sm4 ~3487-3501, copy 4953-4969), `suptitle` (sm4 5352-5361, copy 7337-7347), `_register_control` (copy 7104), `set_info_line` (copy 6679).
- **Copy-only code, no sm4 counterpart**:
  - `Figure.at_design_size` (copy 7049, root 6945). Its only caller is `mplexport.to_matplotlib`.
  - `rcParams['savefig.dpi']: 300` (7851).

### controls.py

- **Module docstring "Settings:" paragraph, 28-36**: none in copy (header 1-25).
- **`Control`, 398-525**:
  - `stateful = False` (411);
  - `setting_label` (418-421);
  - `show` calls `_changed` (470-473);
  - `_fire` calls `_changed` first (483-486);
  - base `setting()` (488-495);
  - `_changed()` (497-510);
  - `remove` calls `_changed` (517-522).

  Copy `Control` 643-743 lacks all of these. Root's `Control` is identical to the copy's (diffed).
- **`Check.setting` 566, `set` → `_changed` 569-578**: copy `Check` 746 (set 776).
- **`Checks.setting` 663 (one `(row label, on/off)` per row), `set` 666-676**: copy 790 (set 870).
- **`Radio.setting` 737, `set` 741-753**: copy 886 (set 941).
- **`Button.setting` 772 (only when stateful), `set_text` → `_changed` 775-778**: copy 955 (set_text ~972).
- **`Text.setting` 976, `set` 979-990**: copy `Text` 1090 (set 1176).
  - `Count` (1191) and the copy-only `Value` (1251) reach `Text.set` through `super().set`, so they inherit the setting and the `_changed` call.
  - Copy-only `Text._accept` (~1146) calls `_fire`.
- **`Slider._moved` 1098 (the valueChanged slot), `setting` 1117, `set` 1120-1131**:
  - **Collision**: the copy's slot is named `_changed(self, v)` (1358, connected at 1349; root 1347/1356). It would override the new `Control._changed()`.
  - `SigmaBox._changed(self, v)` (copy 1600) is not a `Control`, so it does not collide.
- **`settings_text(root)`, 1264-1294**: every live visible control inside `root` with a setting, sorted by `(y, x)` from `mapTo(root)`, as 'label: value' lines.
- **`plt._settings_text = settings_text`, 1297-1301**: none here.
- **sm4 windows setting `setting_label` / `stateful`**:
  - `map.py` 312-313 ('switch to'), 447 (`γ {V} V`);
  - `line.py` 1286-1288;
  - `feenstra.py` 330;
  - `controls.py` 1405, 1979.
- **sm4 window line**: `line.py` 1821, `set_info_line('E-markers', _int_e_readout())` (1740-1748, the selected marker starred).

### mplexport.py

sm4 is 538 lines, the copy 468.

- **`to_matplotlib`, 96-104**: no `at_design_size`. Copy 108-118 wraps it in `with fig.at_design_size()`.
- **`_to_matplotlib`, 106-136**:
  - W, H from `_canvas_size()`; `H_info` from `_info_geometry`;
  - `Figure(figsize=(W/q.DPI, (H+H_info)/q.DPI), dpi=q.DPI)`, white;
  - every visible figure text through `BboxTransformTo` (the top H);
  - `_info_block`.

  Copy 121-146:
  - design size `figsize=(W0/pt2px(72), H0/pt2px(72))` (part 1), but `dpi=q.DPI`;
  - patch facecolor from `fig.patch`;
  - figure texts filtered: below the lowest panel by more than 2 % is dropped (the control strip).
- **`_pos(ax, mfig)`, 138-146**: fractions of mfig pixels (`get_figwidth()*dpi`). Copy `_pos(ax)` 149-153 uses fractions of the canvas.
- **`_axes`, 149-237**:
  - on-screen text sizes, `k = q.FONT_SCALE * font_zoom` (167); white facecolor;
  - side-by-side secondary axes `_side_right_edge`/`SIDE_GAP_PT` (41, 240-262).

  Copy 156-250:
  - points as set (zoom 1, because it was laid out at design size);
  - `ax.get_facecolor()`;
  - rotation swap `_turned` (269);
  - extra legends and legend title (226-249);
  - legend proxies `_proxy` (77);
  - Log/SymLog norms and the `norm.to_matplotlib()` hook (55-74);
  - TriContourSet, PathCollection, per-segment LineCollection;
  - ContourSet with norm and alpha.
- **`_artist`**: sm4 281-356 includes `q.Circle` (no Circle class in the copy) and the Annotation size ×k. Copy 282-380.
- **`_text`** (377, ×k) and **`_colorbar`** (406, ×k): copy 401 and 430 use points as set.
- **`_info_lines`, 429-472**: the panel text re-wrapped with `QTextLayout`, in the panel's font and text option, at the viewport width less the document margins. None here.
- **`_info_geometry`, 475-490**: none here.
- **`_info_block`, 493-519**: `size = QFontInfo.pixelSize()*72/q.DPI`. None here.
- **`export(fig, path)`, 522-538**: vector only (ValueError otherwise), not cropped, facecolor white, fonttype 42 / 'none'. Copy `export(fig, path, dpi=300, facecolor=None, transparent=False)` 451-468 also writes raster and crops with `bbox_inches='tight'`.

## 2. Part-1 code in the copy that the incoming code meets

- **em_px size API, qtplot 318-360**: `base_font`, `em_px`, `px_per_pt`, `pt2px` = pt·px_per_pt. `DPI = 100` is "no longer used for anything on screen" (266).
- **`qfont` (420-437)**: point sizes with a 10 pt floor, `max(10, pt·1.25·zoom)`. Root and sm4 `qfont` use pixel sizes from `pt2px` at DPI 100, with no floor.
- **Points that collide**:
  - the info-panel height (6496);
  - `FigureManager.show` (6539);
  - `window_image`'s `k = dpi/DPI`;
  - mplexport's figsize/dpi/`_pos` pairing;
  - `_info_block`'s `72/q.DPI`;
  - `export_vector`'s `setResolution(DPI)` and page size. Part 1's open item P10 recommends leaving Qt's fallback writers and mplexport's `dpi=q.DPI` alone.
- **Arrangement (controls `Arrangement` ~2180, the 5.2 triggers)**:
  - Today a publication save re-lays the window at design size and back (`at_design_size` calls `_layout.setGeometry`). The port removes that.
  - The dock's first appearance resizes the window (`set_info`), which is a resize trigger.
- **Heatmap fill (`PanelCell.fill_as`, `fill_slot`)**: the on-screen export reads `ax._px_rect()` as drawn. No collision.

## 3. Windows (ruling 8)

Common to all:

- Every control below is a qtgui `Control` that `settings_text` would list when it is visible.
- A Button lists only when stateful. Labels, TextPanels and box titles (`controls.box(title=...)` makes a `Label`) are never listed.
- `Axes.set_title` / `suptitle` send the cut part (`short_title`: first line, cut at ': ', ', ', a double space or ' [') to the info panel today.
- `short_label` also cuts axis labels after ', '. That cut goes nowhere, in sm4 as well. No window's label was found with one.

### Simulator: `simulator.py`, `.qtgui` copy

**Controls** (`_make_controls` 1291-1430):

- `mode` Radio;
- the broadening Checks (rows = atom types), with `σ <type>` Values shown only while their row is on (`_on_broadening_change` 2245);
- RUN Button (label RUN/STOP, not stateful);
- the `sim toggles` Checks (rows Atoms, Decay if `show_decay_toggle`, Norm, Feenstra, Mag, Cell, Extend; `toggle_defs` 1305-1312);
- `Topo h` Value with a Rerun Button;
- Crop / Uncrop Buttons;
- the cursor Texts `x lo`, `x hi`, `y lo`, `y hi` (label '' in `LABELS`, so `setting_label` becomes the name);
- `LDOS h` Value with Rerun (hidden in Single Point, 1701);
- the strip (1394-1402): Cells, Markers (Count), Path Pts (Path only), E Pts (Map only), E Min, E Max (Value), `LDOS γ` (GammaBox Text), `cmap norm` Check (Map only) (visibility 1622-1625);
- `_cluster_label` (a Label; no setting).

Added by `sim_wrapper._attach_widgets` (156-187): `σ (eV)` SigmaBox Value; 'topo' and 'ldos' Buttons (no setting).

**Not qtgui Controls**: only layout containers (`_hook_layout`, `CropPanel`, `RideBand`, `PanelWithBox`, `Titled`, `FlipBox`, `Arrangement`, `MapBlock`).

**Window state in no control:**

- `plot_level`, `active_marker_idx`, `active_element`, `active_atom` (1001-1003; `_on_pick` 3323, `_on_press` 3374);
- `marker_coords` (Single Point/Map) and `marker_ratios` (Line/Path) (992-995, drags; printed on STOP, `_toggle_run` 2162);
- `p1`/`p2` (Line ends) and `path_nodes`;
- `map_e_targets`/`map_e_idx` (shown as the map titles "E = … eV"; X3 re-sort in `_on_rel` ~3520);
- the applied crop `crop_x`/`crop_y` (1369, 2084-2104). The cursor boxes show pending cursors; they show the crop only right after Crop.
- `cell_rotated` (1889-1934, printed only);
- `z_frac`, the contour level (1804);
- `is_running`;
- the applied `topo_height`/`ldos_height`. The boxes hold pending values until Rerun (2210, 2223).
- V_b, `_bias_sample()` (1203). Shown only in Map's title.
- `_broadening_topo_dirty`;
- launch settings: workdir, `unit_cell_num`, grid_res, `reuse_cache`, `ldos_bias_sign`, cmaps, `npts`, the decay model (`_effective_decay_model` 690).

**Titles today:**

- `suptitle(spin_mode.name)` (1381): not cut.
- "Global Topo: bias V, height Å" (2380) and "Map Topo: bias V, height Å" (2383): cut, the rest goes to the info panel.
- "Partitioned LDOS: marker k at … Å" (2846-2852): cut.
- "Topo" (2890), "Tip Height" (2901), 'topo' (1795), "1e<exp>": not cut.
- Heatmap and map labels at level ≥ 3 (3054-3058, 3203-3210): the window's own code drops the element (`split()[-1]`); the cluster Label shows "LDOS: <element>".

### Other windows

- **`plot_dos_cursors_levels.py`**, copy:
  - No controls.
  - State: `plot_level`, `active_element`, `active_atom`, `show_mag` (the 'm' key) (204-206, 364-394).
  - Title "DOS: spin-polarized" / "DOS: SOC" is cut (356-362).
- **`decomp_window.py`**, copy via `resolve_qt`:
  - Controls: `collapse` Checks (Vertical, Horizontal, Amplitude), `area ≥` Text, `Metal curve` Check, Save data Button (438-456).
  - Not in controls: snapshot σ, energy window, marker count and colours.
  - Title "marker m [comp]: x …, y … Å" is cut (626-629).
- **`topo_decomp_window.py`**:
  - Controls: `topo view` Checks (Amplitude, Norm), Save data (262-276).
  - suptitle "MO topo decomposition: σ…, window…, topo…" is cut (282-287); "<comp> <offset>: not found" is cut.
  - Not in controls: offsets, mask_levels.
  - Sets `fig.patch` to 'none' for a transparent export (139-144).
- **`global_decomposition.py`**:
  - Controls: SigmaBox, Save data (433-452).
  - Title "Global decomposition: legacy (…)" is cut (425-427).
  - Not in controls: energy range, the extended/legacy mode.
- **`run_wfn_plane_slice.py`**:
  - Control: GammaBox `γ` (473-500).
  - Titles are file basenames (448); in portrait they show the stamp and the full name goes to the info panel.
  - suptitle "vertical plane through atoms [..]" is cut at ' [' (578).
  - Not in controls: norm_mode, cmap, invert, tol, n_s×n_z, the plane.
  - Scripted `fig.savefig(png, dpi=300)` before show (586).
- **`spread_collapse.py`**, matcher copy:
  - Controls: `collapse` Checks (+Curve), `Segmentation` Check (box hidden unless Horizontal), SigmaBox (hidden unless curve mode) (617-648, 1195-1212).
  - "Spread: all components, all entities" is cut (1016).
- **`spread_plot.py`**:
  - Controls: `spread view` Checks, SigmaBox (512-524).
  - "<comp> <label>,  E = … eV" is cut (494).
- **`state_decomposition_plot.py`**:
  - Controls: `spread view`, `Norm` Check, SigmaBox (805-821).
  - "molecule <what>: a, b" and ": no contributions kept" are cut (470-479, 664).
  - Mouse state: the hover arming in `_on_click` (884-904).
- **`vacupot_plotter.py`, `combined_vacupot_plotter.py`, `matcher_vacupot.py` `__main__`, `run_MOPMatcher.py`, `run_combined_MOPMatcher.py`, `run_vacupot_plotter.py`**:
  - No controls.
  - Titles from PlotConfig, cut by `short_title`.
  - The settings are PlotConfig fields.
  - `run_vacupot_plotter` saves by script with `fig.savefig(SAVE_PATH, dpi=SAVE_DPI)` before show (29-32, 100-101).
- **Root-qtgui users**:
  - `PROCAR_tools/Band_structure_plot.py`: `Orbital` Check (375-380); "Fatbands: dynamic normalized, …" (501) and the suptitle "Orbital-resolved fatbands: per element, …" (580) are cut.
  - `charge_density/lv_slice_w_atoms_2system.py`: no controls; "Slice/Cropped view/System k: axis …, fraction …" are cut (272, 368, 476).
  - `convergence/forces.py`: `window start` Slider and a readout Label (358-371). `canvas3d`, a matplotlib FigureCanvasQTAgg, is a raw widget in the layout (150-156); its rotation, zoom (`state['lims']`) and step (`state['step']`, keys) are in no control. The 3D title "step k of n | t fs" is matplotlib's, not qtplot's (208-211).
  - The `checks/render_windows.py` (`how='screenshot'`, 274) and `checks/check_misc_windows.py` (`savefig(..., dpi=72)` 255, 323) also use root.
- **Only raw Qt or matplotlib widget in any window**: `forces.py`'s `canvas3d` (grep for `QCheckBox`, `QLineEdit`, … found nothing else).

## 4. Other facts

- `sim_wrapper` at HEAD (part 2 X1) no longer has `capture_state` or the `plot_level`/`active_marker_idx` reads that plan rule 4 cites; `git show d5e08a4` still has them (196, 253-264).
- Texts that would become false:
  - copy/root `qtgui/__init__.py` mplexport line "(PDF, EPS, SVG, PNG, TIFF)";
  - `stm_from_dft_opt/README.md` 48-49;
  - `mo_ldos_decomp/README.md` 41-42;
  - `MOPmatcher_mem/README.md` 16, 32-33;
  - root `README.md` 35-36, 97-99;
  - `run_wfn_plane_slice.py` 327, 586-588;
  - `run_vacupot_plotter.py` 29-30;
  - `topo_decomp_window.py` 140-143.

### Proposal

Under ruling 9 no approval of the design is waited for. Every item below is the brief for the implementing session; items that depend on a question say so.

## Order of work

- **qtgui first.** It touches no file part 2 is editing, so it can start now.
- **Copies.** Edit the copy, then copy it byte for byte over `matcher/qtgui/` (part 1's convention). Root follows only if Q1 approves it.
- **Simulator last.** The ruling-8 lines in the simulator wait until part 2's workflow and part 3's decomp/simulator edits are done.
- **Window files.** No window file needs an edit to get the panel and the saves, and `sim_wrapper.py` keeps its API.

## 6.1 — "it wraps at its width (LineWrapMode.WidgetWidth)"

- `FigureManager.__init__` (copy ~6494, root ~6400): change `NoWrap` to `WidgetWidth`.
- Keep the copy's part-1 height `QFontMetricsF(...).lineSpacing()*4.2` (6496). Do not take sm4's `pt2px(7.5)*FONT_SCALE*4.2`; root already has it.
- Docstring as sm4 (4729-4736).

## 6.2 — "it shows only once it has text"

- `FigureManager.set_info` stays as it is (identical to sm4).
- `FigureManager.show` (copy 6539): add sm4's `self.figure._settings_changed()` after `_shown = True`, keeping part 1's `_EM[0] = _measure_em(...)` and the `screenChanged` hookup.
- Note: in a FigureManager window the text always holds "settings", so the dock appears at the first push after show. The window then grows by the dock's height, as `set_info` does today when a title is cut.

## 6.3 — "its text … then an empty line, 'settings', and the setting of every control"

qtplot:

- Add the module global `_settings_text = None` with sm4's comment (4848-4852), just before `class Figure`.
- Replace `Figure.info_text` (copy 6683) with sm4's 4936-4951.

controls:

- `Control` (copy 643):
  - `stateful = False`;
  - `self.setting_label = self.label if self.label.strip() else str(name)` in `__init__`;
  - base `setting()` returning None;
  - docstring as sm4's (398-407).
- Per-kind `setting()` exactly as sm4:
  - `Check` (on/off);
  - `Checks` (one line per row);
  - `Radio` (the option);
  - `Button` (its text, only if stateful);
  - `Text` (the box text; `Count` and the copy-only `Value` inherit it);
  - `Slider` (its readout).
  - `TextPanel` and `Label` have none.
- `settings_text(root)` (sm4 1264-1294) placed after `make()`, then `plt._settings_text = settings_text` with sm4's comment (1297-1301).
- The module docstring gets sm4's "Settings:" paragraph, worded for this copy's kinds (adding Value).

## 6.4 — "updated whenever a control's setting changes (Control._changed → the window's hook → the coalesced info push)"

- `Control._changed()` as sm4's (497-510). It is called:
  - at the start of `Control._fire`;
  - in `Control.show`;
  - in `Control.remove`;
  - at the end of `set()` in `Check`, `Checks`, `Radio`, `Text` (`Count` and `Value` reach it through `super().set`) and `Slider`;
  - in `Button.set_text`.
- Rename `Slider._changed(self, v)` to `_moved` (sm4's name) and update its `valueChanged.connect` (copy 1349/1358, root 1347/1356). See Q11.
- `FigureManager.__init__` gets the hook `self.window.sm4_settings_changed = figure._settings_changed` (Q12 on the name).
- Add `Figure._settings_changed` (sm4 4926-4930).
- `Figure._set_info` ends with `self._settings_changed()` instead of its inline timer; the behaviour is the same.

## 6.5 — "The save list replaces this repository's"

- `SAVE_FORMATS` becomes sm4's 7 entries and comment, with the qt-svg label per Q2.
- The Save tooltip becomes sm4's.
- `_save` becomes sm4's: drop 'vector' without matplotlib; `savefig(path, how=how)` with no dpi.
- Keep the copy's toolbar `mode` property, the rotation swap in `show_coords`, and the em_px icon sizes.

## 6.6 — "The pieces" in qtplot

- Add `WINDOW_IMAGE_DPI = 720` in the configuration block.
- `Figure.__init__` ends with `self.export_info_panel = getattr(self.manager, 'info', None)` and sm4's comment.
- Replace `Figure.savefig` with sm4's 5496-5533 (vector, qt-svg, image, csv; docstring).
- Add `export_root`, `_export_rect`, `export_info` and `window_image` (sm4 5535-5603), with the conversion of Q3.
- Replace `_render_scene` with sm4's `_render_scene(fig, painter)`. It renders `export_root()` over `_export_rect()` with its children.
- Replace `export_vector` with sm4's, modified per Q2.
- Add the module-end binding `from . import mplexport as _mplexport` (root: `from qtgui import mplexport as _mplexport`), replacing savefig's local import.
- Docstrings: the module docstring's layout and "Also here" paragraphs, and the `Figure` docstring ("A saved file is the window at its size on screen"), as sm4's.
- Dead code goes per Q8.

## 6.7 — mplexport

- `to_matplotlib` without `at_design_size`.
- `_to_matplotlib` as sm4's canvas-plus-info-block figure. The figure size and dpi follow Q3, so that `_pos`'s fractions stay right. Figure texts go through sm4's canvas transform (Q7 on the filter).
- `_pos(ax, mfig)`.
- `_info_lines`, `_info_geometry` and `_info_block`; the block's font size per Q3.
- `export(fig, path)`: vector only, no crop, fonttype 42 / 'none'; facecolor per Q5.
- Module docstring as sm4's.
- Texts at their on-screen size per Q4.
- Keep every copy-only feature (Q6). sm4's side-by-side secondary axes and Circle are not named by the plan and are not ported.

## 6.8 — "the two scrollable panels in each save"

- **PDF / EPS / vector SVG:** the info block starts below the lower of the canvas bottom and the bottom of the legend panel's full list (mechanism in Q14).
- **Qt SVG and PNG/TIFF:** render the window as shown. Both panels appear scrolled where they are, in their own space (Q2, Q13).

## 6.9 — ruling 8, what sm4's design covers and what it does not

**Covered with no window edit:**

- every visible Check, Checks row, Radio, Text, Count, Value, GammaBox and SigmaBox box, and Slider;
- the cut part of every title and suptitle;
- each window's own `set_info_line` lines.

**Not covered:**

- state held in no control (mouse, keys or code);
- the applied values when a box holds a pending one;
- raw Qt or matplotlib widgets (`forces.py`'s `canvas3d`);
- the toolbar's Pan/Zoom toggles;
- RUN-style buttons that are not stateful;
- hidden controls (left out by design);
- axis-label cuts.

How the uncovered items reach the panel: Q16-Q24. The recommendation is sm4's own mechanism, `fig.set_info_line(key, text)`, which `line.py` 1821 uses.

## 6.10 — sim_wrapper

No change. It calls `controls.SigmaBox`, `controls.make`, `controls.box`, `sim.add_control_box` and `plt.fignum_exists`. Its controls are made on `sim.fig`, so they are listed in the settings section through the simulator window's hook.

### Questions

- **port:S6-Q1** (§6 'every window in this repository that uses qtgui … and the rest'; §0 'qtgui/qtplot.py, controls.py, mplexport.py'): Part 1 left the root qtgui/ unchanged. Root is used by PROCAR_tools/Band_structure_plot.py, charge_density/lv_slice_w_atoms_2system.py, convergence/forces.py and checks/*. Root has no em_px API: it is DPI-based with pixel-size qfont, like sm4_browser. It imports absolutely (`from qtgui import …`). The two package copies are byte-identical.
  - (a) All three copies. Edit stm_from_dft/qtgui and copy it over matcher/qtgui. Port root separately with sm4's own DPI conversions and its absolute import `from qtgui import mplexport as _mplexport`.
  - (b) The two package copies only; the root windows keep today's info panel and saves.
  - (c) Root only.
  - R: (a): only this gives the plan's literal 'every window'. Root takes sm4's code nearly verbatim, because its size rules are sm4's.
- **port:S6-Q2** (§6 save list, 'qt-svg' … differs from sm4_browser … exactly as shown; §6 'In the Qt-drawn SVG … exactly as shown'): The plan treats only the label as different. sm4's export_vector (5694-5770) does more: it paints the window as shown, then appends the full info text below it, so the info appears twice (the dock as shown, and in full). export_vector is also the fallback for 'vector' when matplotlib is missing.
  - (a) qt-svg is exactly as shown: export_vector without the appended block. Only the no-matplotlib fallback of 'vector' keeps the full block, through a parameter such as export_vector(fig, path, full_info=False). Label: 'SVG, the whole window as Qt draws it, controls included, as shown (*.svg)'.
  - (b) Drop the appended block everywhere, the fallback included.
  - (c) Keep sm4's export_vector; only the label changes, and then it says 'as shown' over a file that also holds the full info.
  - R: (a)
- **port:S6-Q3** (§6 pieces (window_image, mplexport canvas-plus-info-block); PART3_QUESTIONS Q4; PART1_OPEN P10): sm4 converts pixels with its fixed DPI = 100. This copy's DPI is 'no longer used for anything on screen' (qtplot 266); its logical pixels per inch are 72·px_per_pt(). Two traps: 1. sm4's `_pos(ax, mfig)` divides by `mfig.get_figwidth()*mfig.dpi`. Pairing part 1's figsize W/pt2px(72) with the copy's dpi=q.DPI would make every axes fraction wrong. In sm4 the figure's pixels equal logical pixels exactly. 2. `_info_block`'s `pixelSize()*72/q.DPI` and `window_image`'s `k = dpi/DPI` also assume DPI. PART1_OPEN P10, still open, recommends leaving Qt's fallback writers and mplexport's `dpi=q.DPI` as they are.
  - (a) In the copy: - mpl Figure(figsize=(W/pt2px(72), (H+H_info)/pt2px(72)), dpi=pt2px(72)), so one figure pixel is one logical pixel and `_pos` stays sm4's; - the info block's size from QFontInfo(font).pointSizeF(), or pixelSize()/px_per_pt(); - window_image k = dpi/(72·px_per_pt()), the image's inch size then equal to get_size_inches(); - export_vector's resolution and page size as P10 is ruled.
  - (b) sm4's code as is, with q.DPI throughout, keeping part 1's figsize: the axes fractions in the files would then be wrong.
  - (c) sm4's code as is, including figsize W/q.DPI: the files' physical size and text no longer match part 1's points.
  - R: (a). Root takes sm4's DPI code verbatim.
- **port:S6-Q4** (§6 'PDF, EPS and SVG … the plots as on screen'): The plan names only _to_matplotlib's canvas-plus-info-block figure and the info functions. Once the file is the on-screen canvas (not the design size at zoom 1), every text must be at its on-screen size too. sm4 multiplies by k = FONT_SCALE·font_zoom in _axes, _colorbar, _text, the Annotation and the legend fontsize (mplexport 167, ~347, 377, 406). This copy's on-screen rule differs: qfont sets max(10 pt, pt·1.25·zoom), a 10 pt floor from part 1; root has no floor. The legend's shrink-to-fit is not reproduced, in sm4 either.
  - (a) Port the on-screen size with this copy's rule, max(10, pt·FONT_SCALE·zoom), through one helper. Root uses sm4's k unchanged.
  - (b) sm4's k without the floor: texts below 10 pt in the file where the screen shows 10 pt.
  - (c) Leave the copy's points as set: the files' text 1.25×zoom smaller than the screen's.
  - R: (a)
- **port:S6-Q5** (§6 vector saves (sm4's _to_matplotlib / export)): sm4 forces white: mfig.patch white, every axes white, savefig(facecolor='white'). This copy takes fig.patch and each axes' facecolor and honours transparent. topo_decomp_window sets fig.patch and its axes to 'none' (139-144, 'the publication export … carries NO opaque background'); sm4's code would make that export opaque.
  - (a) Keep the copy's facecolor handling (fig.patch, ax.get_facecolor()) inside sm4's structure; the info block's text sits on that background.
  - (b) sm4's forced white.
  - R: (a)
- **port:S6-Q6** (§6 'the canvas-plus-info-block figure and the info block functions in mplexport.py'): Replacing mplexport wholesale with sm4's would drop copy-only features that windows use: - the rotated-axes swap (_turned; the simulator's set_rotated); - LogNorm, SymLogNorm and the norm.to_matplotlib() hook (run_wfn_plane_slice's _GammaNorm); - legend proxies (_proxy; plt.Patch handles); - extra legends and the legend title; - TriContourSet (topo_decomp); - PathCollection (scatter markers); - per-segment LineCollection colours; - ContourSet norm and alpha. sm4-only features not named by the plan: the side-by-side secondary axes (_side_right_edge, SIDE_GAP_PT) and q.Circle (no Circle class here).
  - (a) Port only the plan-named pieces (_to_matplotlib's structure, _pos(ax, mfig), _info_lines, _info_geometry, _info_block, export, to_matplotlib, the docstring) plus the text sizes of S6-Q4; keep every copy-only feature; do not port the side axes or Circle.
  - (b) sm4's mplexport wholesale.
  - R: (a)
- **port:S6-Q7** (§6 vector saves): The copy's _to_matplotlib drops figure texts more than 2 % below the lowest panel (the old control-strip readouts). sm4 exports every visible figure text. Today no window has a figure text other than its suptitle, and the filter would need rewriting for _pos(ax, mfig).
  - (a) sm4's rule: every visible figure text.
  - (b) Keep the copy's filter, re-expressed in canvas fractions.
  - R: (a)
- **port:S6-Q8** (§6 'replaces'; rule 5 (list removed functions)): After the port, Figure.at_design_size (copy 7049, root 6945) has no caller; its only caller was mplexport.to_matplotlib. The same holds for Figure._size_override (read in _canvas_size), rcParams['savefig.dpi'], and savefig's 'publication', 'screen' and 'screenshot' routes. at_design_size is the owner's pre-part-1 code.
  - (a) Remove at_design_size, _size_override and its read in _canvas_size, and 'savefig.dpi'; list them in rule 5's lists.
  - (b) Keep at_design_size unused and correct its docstring.
  - (c) Keep everything as it is.
  - R: (a): sm4 has none of them, and its savefig sends any other 'how' to window_image.
- **port:S6-Q9** (§6 saves in every window): Two scripted saves run before the window is shown: - run_wfn_plane_slice.py 586, fig.savefig(png, dpi=300); - run_vacupot_plotter.py 101, fig.savefig(SAVE_PATH, dpi=SAVE_DPI). Today they go through at_design_size, which lays the window out before show. Under sm4's savefig a .png becomes window_image (dpi ignored) of a window that is not laid out yet. A vector file reads _px_rect of an unplaced layout, and the info panel may still be empty. sm4's own pattern (tools/rebuild_panels.py 518) is plt.show(block=False), then fig.savefig, then plt.show().
  - (a) Both callers save after plt.show(block=False), sm4's pattern; their PNG/TIFF is the 720 dpi window image; update their comments and the README rows.
  - (b) Leave both callers as they are.
  - (c) Keep a scripted raster route (dpi given → the old matplotlib raster at design size), against 'replaces'.
  - R: (a)
- **port:S6-Q10** (§6 (if root is ported, S6-Q1)): checks/render_windows.py 274 (how='screenshot') and checks/check_misc_windows.py 255 and 323 (png with dpi=72) use root qtgui. Under sm4's savefig both would write 720 dpi window images. PART1_OPEN says the two remaining checks are unchanged, and rule 3 adds no checks.
  - (a) Leave the checks unchanged: they run, writing window images.
  - (b) Update them to the new hows.
  - R: (a)
- **port:S6-Q11** (§6 'Control._changed' (sm4)): This copy's and root's Slider uses the name _changed(self, v) for its valueChanged slot (copy 1349/1358). Adding sm4's Control._changed() would be shadowed by it on every Slider (forces.py's 'window start'). sm4 renamed the slot _moved (1098).
  - (a) Rename Slider._changed to _moved and its connect, as sm4.
  - (b) Give Control._changed another name, which breaks sm4's reading.
  - R: (a)
- **port:S6-Q12** (§6 'calls the window's hook, sm4_settings_changed there'; PART3_QUESTIONS Q8): The plan names the hook only as it is in sm4. It is an attribute set on the QMainWindow that Control._changed reads.
  - (a) Keep the name sm4_settings_changed, so the copy reads like sm4.
  - (b) A neutral name such as settings_changed, in both qtplot and controls.
  - R: (a)
- **port:S6-Q13** (§2 legend panel Back button 'a Qt control, so the saves leave it out'; §5/§6 'Qt-drawn SVG and window images … exactly as shown'): The Qt SVG and the PNG/TIFF render the window with its children (sm4's _render_scene and window_image use DrawChildren), so every control, the Back button included, appears in them. Only the vector saves leave controls out with their places empty. §2's 'the saves leave it out' conflicts with §5/§6.
  - (a) Read §2 as the vector saves (PDF, EPS, vector SVG); the Qt SVG and images show the Back button as on screen.
  - (b) Hide the Back button while the Qt SVG and images are rendered, a change outside sm4.
  - R: (a)
- **port:S6-Q14** (§6 'The full info panel starts below the canvas, moved down below the bottom of the legend panel when the legend panel reaches past the canvas'; §5 vector saves hold the full list): sm4's _to_matplotlib places the info block at the canvas height H. Nothing in qtplot knows an axes' 'full list' extent beyond its on-screen _px_rect(), so the plan's rule needs a mechanism sm4 does not have. The legend panel's own side belongs to the legend-panel area.
  - (a) A generic qtgui hook: an axes may carry its rectangle for the vector saves (e.g. ax._export_px_rect(), default its _px_rect()). mplexport's _pos and the rebuild read it; _to_matplotlib takes H_plot = max(H, the lowest export bottom), makes the figure H_plot + H_info tall, and puts the info block at H_plot.
  - (b) Simulator-specific code in mplexport.
  - R: (a)
- **port:S6-Q15** (§6 (not named); ruling 1 (qtgui/__init__.py docstring: 'the module list stays unchanged'); part 2 A15 practice): After the port these texts are false: - qtgui/__init__.py's mplexport line '(PDF, EPS, SVG, PNG, TIFF)', in all three copies (ruling 1 froze the module list); - stm_from_dft_opt/README.md 48-49; - mo_ldos_decomp/README.md 41-42; - MOPmatcher_mem/README.md 16 and 32-33; - root README.md 35-36 and 97-99; - run_wfn_plane_slice.py 327 and 586-588; - run_vacupot_plotter.py 29-30; - topo_decomp_window.py 140-143. Part 2 is editing some of these READMEs now.
  - (a) Update each in the same terms, changing only the stale words, after part 2's workflow has finished; __init__.py's mplexport line too, as an exception to ruling 1.
  - (b) The same, but leave __init__.py untouched.
  - (c) Leave every text as it is.
  - R: (a), with __init__.py named to the owner as an exception to ruling 1.
- **port:S6-Q16** (Ruling 8 'whatever the settings of the window is'; PART3_QUESTIONS Q6): sm4's settings section lists controls only. Its windows add their own lines by set_info_line (line.py 1821, the E-marker readout). The plan does not say which state held in no control counts.
  - (i) Interactive state only (set by the mouse, keys or code) that no control or uncut title shows, plus applied values that differ from what a box shows.
  - (ii) (i) plus the launch settings that shape what is drawn and are shown nowhere (the simulator's grid_res, unit cells, cache reuse, decay model, ldos_bias_sign, cmaps; the plane slice's norm_mode, tol, n_s×n_z; PlotConfig options).
  - (iii) Controls only, as sm4.
  - R: (i) for every window; (ii) only where the owner names the inputs.
- **port:S6-Q17** (Ruling 8 (mechanism)): There are two ways to bring window state into the panel.
  - (a) sm4's mechanism: each window calls fig.set_info_line(key, text) where its state changes, through one helper per window. _set_info ignores an unchanged text, and the push is coalesced (50 ms).
  - (b) A new qtgui pull hook (fig.window_info_fn read by info_text), with the window calling the push where its state changes.
  - R: (a): no new qtgui API.
- **port:S6-Q18** (Ruling 8, the simulator): Proposed simulator lines (keys and texts), one helper refreshing them. Keys: - 'spectrum': the decomp level by name with marker k and element X, after part 3's levels 0-3; - 'markers': Single Point/Map 'k (x, y) Å', Line/Path 'k at d Å along the line', the active one starred as sm4's E-marker readout; - 'line': Line ends with the length, or the Path nodes with the length; - 'map energies': Map, the snapped energies; - 'crop': the applied crop, or 'none'; - 'orientation': the unit-cell panels as drawn or turned 90°; - 'contour': Map, z_frac and its height; - 'bias': V_b and which end; - 'applied heights': Topo h and LDOS h as applied; - 'state': running or stopped, and 'global topo stale' when _broadening_topo_dirty. The helper runs at the end of _update_all, in _on_rel (after the X3 re-sort), _set_level, _on_crop, _on_uncrop, _set_cell_rotation and _toggle_run. A view clears the keys that do not apply to it (text '').
  - (a) All of these lines, implemented after part 2's workflow and part 3's ldos_decomp and simulator edits (one workflow edits simulator.py at a time).
  - (b) A subset the owner names.
  - (c) Only the decomp state.
  - R: (a)
- **port:S6-Q19** (Ruling 8 'toggles'; sm4 'stateful' buttons): RUN/STOP shows the running state in its label, but a plain Button has no setting. sm4's precedent is map.py 312-313: stateful = True with setting_label 'switch to', for a label that names the action.
  - (a) The 'state' window line of S6-Q18 ('running' / 'stopped').
  - (b) btn_run.stateful = True with a setting_label (e.g. 'RUN button: STOP').
  - (c) Nothing.
  - R: (a)
- **port:S6-Q20** (Ruling 8 'all the settings of the widgets or toggles'): sm4's Checks.setting names each row by its row label only. The simulator's broadening rows then read 'Au: on', with the box title 'broadening' (a Label) not part of the line. The σ boxes read 'σ Au: 0.05'. No setting label repeats within any window of this repository.
  - (a) As sm4 (row labels only).
  - (b) A Checks-level prefix (e.g. 'broadening Au: on'), a change to sm4's Checks.setting.
  - R: (a)
- **port:S6-Q21** (§3 'plot_dos_cursors_levels.py … stays as it is' vs ruling 8): plot_dos_cursors_levels.py gets the new info panel and saves through qtgui with no edit. Its state (plot_level, active_element, active_atom, show_mag by the 'm' key) is in no control, and listing it would need an edit to that file.
  - (a) No edit: title details and an empty settings section only.
  - (b) Add window lines for its level and selection, an exception to §3.
  - R: (a), unless the owner lifts §3 for it.
- **port:S6-Q22** (Ruling 8 'widgets'; §6 saves): convergence/forces.py embeds a matplotlib FigureCanvasQTAgg (canvas3d), a raw widget that is not a Control. Its rotation, zoom (state['lims']) and the key-driven step are in no control; the step shows in its own matplotlib title. In the vector saves its place is empty, as today, since mplexport rebuilds only qtplot axes. In the window images it is rendered as a child widget (unverified, see unverified).
  - (a) No lines: the step is visible in its title.
  - (b) Window lines for the step and the 3D view (elevation, azimuth, limits).
  - R: (a) under S6-Q16 (i), because the step is shown; (b) if the owner counts the 3D view.
- **port:S6-Q23** (Ruling 8, the other windows): Under S6-Q16 (i): - decomp_window, global_decomposition, the matcher plotters, Band_structure_plot and lv_slice have no interactive state outside controls and titles; - decomp_window's snapshot σ and energy window are set by the sender, not interactively; - state_decomposition_plot's hover arming is transient; - topo_decomp_window's σ, window and topo are in its cut suptitle, so they already reach the panel.
  - (a) No window lines in these windows.
  - (b) Lines for their snapshot and launch settings (S6-Q16 (ii)).
  - R: (a)
- **port:S6-Q24** (Ruling 8 'toggles'): The toolbar's Pan and Zoom are checkable QActions, not Controls. Pan and zoom also change each panel's view limits, which no control holds.
  - (a) Not listed: navigation, not settings; sm4 does not list them.
  - (b) A window line when Pan or Zoom is on.
  - R: (a)
- **port:S6-Q25** (§6 'PNG and TIFF of the window at 720 dpi'): Qt writes TIFF through the Qt Image Formats add-on, not Qt GUI's built-in formats. Whether the PySide6 wheel in use ships the TIFF writer is unverified. sm4's window_image raises OSError when Qt cannot write the format; _save prints 'save failed'. A full-screen simulator window at 720 dpi is about 14100×7500 px ARGB32, about 0.42 GB (arithmetic at 96 logical dpi).
  - (a) As sm4: the owner checks TIFF on his machine.
  - (b) Drop the TIFF entry when QImageWriter lacks 'tiff', a change outside sm4.
  - R: (a)
- **port:S6-Q26** (§6 'its text … then an empty line, the line settings'): In a window with no controls (the vacupot plotters, lv_slice, plot_dos_cursors_levels), sm4's info_text still ends with a bare 'settings' line, and the dock always appears after show.
  - (a) As sm4.
  - (b) Omit 'settings' when a window has no settings, a change to sm4's info_text.
  - R: (a)
- **port:S6-Q27** (Rule 4 (sim_wrapper reads sim.plot_level / active_marker_idx; _instantiate ~205-211, capture_state ~276-277)): At HEAD, after part 2's X1, sim_wrapper.py no longer has capture_state and does not read plot_level or active_marker_idx. At d5e08a4 they are at 196 and 253-264. Section 6 does not depend on this; reported for the decomp area.
  - (a) Keep plot_level and active_marker_idx readable and settable on the simulator anyway, as rule 4 says.
  - (b) Treat rule 4's reads as gone.
  - R: (a): costs nothing and keeps rule 4 true.

### External facts (verified)

- QWidget visibility: 'If an ancestor is not visible, the widget won't become visible until all its ancestors are shown'. So a control inside a hidden box frame reports isVisible() False, and sm4's settings_text leaves it out. (https://doc.qt.io/qt-6/qwidget.html)
- QWidget::isAncestorOf returns true if this widget is a parent (or grandparent and so on to any level) of the given child, and both widgets are within the same window. (https://doc.qt.io/qt-6/qwidget.html)
- QWidget::render 'Renders the widget into the painter's QPainter::device()'; transformations and settings applied to the painter are used when rendering. RenderFlag DrawChildren renders the widget's children recursively into the target; DrawWindowBackground renders the widget's background even if autoFillBackground is not set. So sm4's window_image and _render_scene include the window's controls. (https://doc.qt.io/qt-6/qwidget.html)
- QPlainTextEdit lineWrapMode: 'The default mode is WidgetWidth which causes words to be wrapped at the right edge of the text edit. Wrapping occurs at whitespace, keeping whole words intact.' (https://doc.qt.io/qt-6/qplaintextedit.html)
- TIFF read/write is provided by the Qt Image Formats add-on module ('optional support for other image file formats'), not by the formats Qt GUI supports by default. (https://doc.qt.io/qt-6/qtimageformats-index.html)
- matplotlib Figure: figsize is interpreted in inches; dpi is 'Dots per inch' (default rcParams['figure.dpi'] = 100). So a figure's pixel size is figsize × dpi, which is what sm4's _pos(ax, mfig) divides by. (https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html)

### Unverified

- Whether the PySide6 pip wheel in use bundles the Qt Image Formats TIFF writer (qtiff). Without it, a TIFF save raises sm4's OSError.
- sm4's export_vector docstring says Qt paints widgets into a pixmap when they are rendered straight into a printer-type device, hence its QPicture-then-QPdfWriter route. Not checked in Qt's documentation; the render() page fetched mentions only macOS rendering through a QPixmap.
- That `from . import mplexport as _mplexport` at the end of qtplot works while mplexport itself runs `from . import qtplot as q` (Python's handling of a partially initialised submodule). sm4_browser relies on the same pattern with absolute imports; not checked in Python's documentation.
- The placement semantics of QWidget::render(painter, targetOffset, sourceRegion, flags), i.e. that the source region's origin lands at targetOffset, which window_image and _render_scene rely on.
- Whether Qt allocates and saves a QImage of about 14100×7500 px ARGB32 (about 0.42 GB, by arithmetic) for a full-screen simulator window at 720 dpi.
- Whether a save made right after plt.show(block=False) already holds the settings section: the push waits on a 50 ms QTimer, and show(block=False) runs processEvents only once. sm4's checks/save_check.py asserts that it does.
- Whether QWidget.render draws the contents of forces.py's FigureCanvasQTAgg (the 3D panel) into the window images.
- matplotlib's savefig without bbox_inches writes the full figure size. The page fetched gave this only as general knowledge.
- The plan's statement that sm4_browser's qtplot.py and mplexport.py at eb9bb62 are the same as the owner's copies on disk (not reachable here).
- Whether any window draws an axis label that short_label cuts. A grep of the windows found none, but not every label is a literal.

## Check: plan fidelity

- [contradiction] (decomp + legend) The two reports describe the decomp-to-legend interface differently, and the implementing session cannot build both. (1) Field names and level-0 values differ. Decomp's Entry has label, color, linestyle, linewidth, clickable, option; title and label are '' at level 0. Legend L-9 has color, ls, lw, label, clickable, option; title and label are None at level 0. (2) Legend L-9 needs "the set of every label any level can show, from atomtypes and orbitals (needed for L-5)". Decomp proposes no such function. (3) Who finds the entry differs. Decomp C expects a legend-panel object with `entry_at(event) -> Entry | None`, then calls `Decomp.press(button, dblclick, item)`. Legend L-8 instead looks the entry up in simulator code (`self._legend_drawn`, (xdata, ydata)), gates on `not event.dblclick` and the toolbar itself, and calls "ldos_decomp's left-click" with the option. So the button / double-click rule sits in `_on_press` and not in ldos_decomp. (4) Under L10 (a) the panel is simulator code, not an object that could have an `entry_at` method.
  - Evidence: Plan §3: "The legend entries: the legend panel's entries at each level, each with its colour, line style, line width and label, and the panel's title". Plan §3: "The clicks: the left / right / double-click and Back button rules of section 2. They replace two pieces of code" (the click rules belong in ldos_decomp.py). Decomp C, "What it gives back: entry_at(event) -> Entry | None" versus legend L-8, "look up the entry whose drawn box (self._legend_drawn) contains (xdata, ydata); if found, pass that entry to ldos_decomp's left-click".
  - Fix: Fix one interface in both briefs: one Entry with one set of field names, and one level-0 value ('' or None) for the title and the label. Either add a decomp function for the full label set (e.g. `all_labels(atomtypes, orbitals)`, after Q-D1) or drop L-9's static item. The legend lookup only returns the Entry, or None. The button, double-click (and, under Q-D13, RUN) rules live only in `Decomp.press` / `Decomp.up`, as §3 places them.
- [contradiction] (decomp + legend) The two reports read "anywhere in the legend panel" differently. - Decomp B7 and Q-D14 (a): only presses with `event.inaxes` equal to the legend axes (the stack) reach `press`. Pan/Zoom is tested. Nothing is said about the title cell, the Back row or the scroll bar. - Legend L-8: the stack axes OR any viewport pixel inside the composite's content rectangle (the title, the Back row and the bar column). L9 (b) adds an event filter so that a right press on the scroll bar also goes back. Each report raises the point as its own question, with different options. The owner gets two questions about one rule.
  - Evidence: Plan §2: "Right click anywhere in the spectrum panel or the legend panel (on a curve, on an entry, or on empty space): goes back one level (levels 1 to 3)." Plan §5: "A right click anywhere in the panel, or the Back button, goes back one level". Decomp Q-D14 (a): "The plot box (event.inaxes), as today". Legend L-8: "the press's viewport pixel lies inside the panel's content rectangle (title, Back row, bar column)".
  - Fix: Merge Q-D14 and L-8 / L9 into one question to the owner. It covers the spectrum (the plot box or the whole cell) and the legend panel (the stack only, or the whole composite including the title, the Back row and the bar), and the same Pan/Zoom rule for both. Both briefs then implement that one answer.
- [contradiction] (legend + port) The two reports propose different qtgui hooks for the vector saves to hold the legend's full list. - Legend L-10 / L11 (a): an attribute the window sets, `ax.export_extent = fn`, returning `(content_h_px, (y_top, y_bottom))`. Option (b) is `Axes.set_export_extent(fn)`. mplexport computes `H_top = max(H, y0 + content_h)`. - Port S6-Q14 (a): a method `ax._export_px_rect()` that defaults to `_px_rect()`, read by `_pos` and the rebuild, with `H_plot`. Other qtgui edits from the legend report are missing from the port's brief, though both reports edit the same files: - L14 (a), export_csv skips a marked axes (`ax.csv = False`). Port 6.5 / 6.6 calls export_csv "identical" and leaves it unchanged. - L8 (a), `Axes.set_navigate` made real in qtplot. - L7 (a), a side item for MapBlock in controls.py. The no-matplotlib fallback also differs. Port S6-Q2 (a) keeps the full info block there. Legend L11 (a) keeps the legend as shown and adds a console note. The result is a vector file with a full info panel and a cut legend list, which plan §6 rules out.
  - Evidence: Plan §6: "In the PDF, EPS and vector SVG: both are full and expanded, and they don't overlap. The legend panel holds its full list and extends down past the end of the window as far as it needs. The full info panel starts below the canvas, moved down below the bottom of the legend panel when the legend panel reaches past the canvas." Plan §6 also lists "CSV, the curves' data".
  - Fix: Choose one hook name and signature in both briefs, and put one question to the owner instead of L11 and S6-Q14. Add L14's export_csv skip, L8's set_navigate and L7's MapBlock change to the port's qtgui change list, so that one session edits each qtgui file once and copies it over matcher/qtgui. State one rule for the no-matplotlib fallback.
- [contradiction] (decomp + legend + port) Ruling 8's decomp line is proposed three different ways. - Decomp Q-D16 (a): `Decomp.info()` set with `fig.set_info_line('decomp', …)`, text 'decomp: Orbitals of C, marker 1' or 'decomp: markers'. - Legend L19 (a): texts 'decomp: Partition, marker 2', 'Atom Types, marker 2', 'Orbitals of C, marker 2'. - Port S6-Q18 (a): key 'spectrum', "the decomp level by name with marker k and element X". It is written by one helper at the end of `_update_all`, `_on_rel`, `_set_level` and others. Two more problems: - The active marker and its place already reach the info panel. `Axes.set_title` sends the full text "Partitioned LDOS: marker k at … Å" to `Figure._set_info` when short_title cuts it at ': ' (qtplot copy ~4963-4967 and ~1572). The port's code map says so. Decomp Q-D16 and legend L19 ("the marker index is not written") do not, so the proposed lines repeat it. - Decomp Q-D16 (b) needs the legend title to be drawn with `Axes.set_title`, which legend L2 (a) does not do (it uses `fig.text(..., cell=)`).
  - Evidence: Ruling 8: "it should say all the settings of the widgets or toggles and whatever the settings of the window is along with any information in the titles that was dropped for the sake of space". PART3_QUESTIONS Q6 (a): "Every such setting as the window's own lines (set_info_line), updated whenever it changes."
  - Fix: Ask the owner one question, combining Q-D16, L19 and S6-Q18 / Q6. Give one key, one text per level, and one writer: the port's helper calls the decomp's `info()`. Say whether the marker index is repeated, given the cut spectrum title already carries it. Drop Q-D16 (b) if L2 (a) is approved.
- [contradiction] (legend (vs decomp Q-D1)) Legend L-5 measures the fixed width over f"{t} {orb}" for every atom type and every `self.orbitals` name. Decomp Q-D1 shows (confirmed at HEAD) that in a COLLINEAR_POL run `self.orbitals` has 18 names ('s_up', 's_down', …) for 9 data columns: - simulator.py ~541: `num_cols = dos_parser.spin_up_dos.shape[2]`, then `mapping_pol[9]`; - doscar_parser.py ~160-161: `site_dos[:,:,0::2]` and `site_dos[:,:,1::2]`. Under Q-D1 (a) the level-3 labels are the base names. Under (b) they are _up / _down. L-5's label set and decomp's labels therefore come from different sources and can disagree, against the plan's width rule.
  - Evidence: Plan §2: "In a spin-polarized run the `_up` and `_down` columns are separate entries, as in the maps". Plan §5: "Width (the owner's rule): as wide as its largest entry (sample, label and button box)".
  - Fix: Make L-5 depend on Q-D1. The width's label set must come from the same decomp code that makes the level-3 labels (see the all-labels function in the first finding), not from `self.orbitals` directly.
- [wrong_fact] (port) Port 6.6 copies sm4's docstrings and comments "as sm4's". This would put false statements into this repository's qtgui: (1) sm4's Layout paragraph (sm4 qtplot ~52-53) says "a column of panels one width (Figure.align_column)". Neither qtgui copy here has align_column: grep finds only `align_row`, at copy ~7459 and root ~7355. (2) Several sm4 texts say the qt-svg save has "the full info panel below": the "Also here" paragraph, the SAVE_FORMATS comment (sm4 4663-4670), the `savefig` docstring ('qt-svg': "with the full info panel below it") and the `export_vector` docstring. Port 6.5 takes "sm4's 7 entries and comment", and 6.6 takes sm4's savefig "docstring". Under S6-Q2 (a) and plan §6 the qt-svg save is exactly as shown, so only the dialog label would be corrected.
  - Evidence: Plan §6: "SVG of the whole window as Qt draws it, controls included ('qt-svg'). Here it differs from sm4_browser, whose label says 'full info panel': this save is exactly as shown (see below), and its label in the save dialog says that".
  - Fix: In the brief: - Change only the copy's sentences that become false (the design-size export sentence and the save list). - Do not take sm4's align_column clause. - Word every qt-svg text the same way under S6-Q2: the SAVE_FORMATS comment, the savefig docstring, the export_vector docstring and the module docstring.
- [contradiction] (port) Port S6-Q3 (a) changes two values: - mplexport's matplotlib Figure dpi, from `q.DPI` to `pt2px(72)`; - window_image's factor k, to `dpi/(72·px_per_pt())`. This contradicts two open recommendations: - the orchestrator's PART3_QUESTIONS Q4 R, "matplotlib's own dpi for written files stays"; - PART1_OPEN P10 R, "leave them (written files, not screen sizes)" for mplexport's `Figure(..., dpi=q.DPI)` and the Qt fallback writers. The port cites P10 but does not say that its option (a) overrides Q4's R. Its reason does hold: sm4's `_pos(ax, mfig)` divides logical pixels by `get_figwidth()*dpi`, so dpi must equal logical pixels per inch. The owner should see the conflict, not two recommendations that disagree.
  - Evidence: Plan §6: "the canvas at its on-screen size, the full info panel below it" and "PNG and TIFF of the window at 720 dpi (Figure.window_image, WINDOW_IMAGE_DPI = 720)". PART3_QUESTIONS Q4 (a): "matplotlib's own dpi for written files stays."
  - Fix: Put S6-Q3 to the owner as a replacement for Q4's R. State that option (a) changes the dpi Q4 keeps, why `_pos` needs it, and that P10 is still open.
- [beyond_plan] (decomp) Decomp B7 moves the start of the energy-marker drag. Today it starts in `_on_pick`'s ax_spec branch (HEAD ~3345: `if self.plot_level == 0: self.active_obj = ('emarker', …)`), with no toolbar test and no button test. In B7 it starts in `_on_press`, inside `toolbar.mode == ""`, on a left, non-double press. Q-D4 raises the button and double-click part. Nobody raises the new Pan / Zoom gate: with Pan or Zoom on, an energy marker could no longer be dragged. Q-D14 (a) speaks only of "decomp clicks", and the drag is not one.
  - Evidence: Plan §2, "What stays as it is": "Map's energy markers (`emarker_i`, the vertical lines in the spectrum): a left click on one at level 0 starts its drag, as today."
  - Fix: Add the Pan/Zoom gate to Q-D4 as an explicit option: the drag starts with Pan/Zoom on as today (the drag start is tested before the toolbar test), or it does not.
- [beyond_plan] (decomp) Decomp A8 and B3 replace the plan's shared datum `m_colors` with a `marker_colors` list, one colour per marker, built by the simulator. Decomp raises no question about it. Only the markers report raises it, as Q2a-6. The two reports agree on the mechanism (in-place `set_color` under Q-D11 (a) and Q2a-7 (a); index colours in Single Point / Map). The decomp item should still be marked as depending on Q2a-6.
  - Evidence: Plan §3: "Shared data stays in the simulator and is passed in: `atomtypes`, `atomnums`, `orbitals`, `_type_color_map`, `m_colors` and `dos_up_gpu`."
  - Fix: Mark decomp A8's `marker_colors` argument and B3's "marker_colors" line as depending on Q2a-6. If the owner answers (b), the signature takes `m_colors` plus the kept colours.
- [contradiction] (markers + legend) The two reports handle the legend differently after a Line / Path re-sort. - Markers P5: the legend's level-0 entries are "updated in the same call", with no rebuild. - Legend L-7: the panel rebuilds (`ax.clear()`, entries drawn anew, limits and bar reset) whenever ldos_decomp's (title, entries) differ from what it drew. A colour permutation at level 0 differs, so every re-sort clears and redraws the legend axes. The spectrum, by contrast, is recoloured in place (Q-D11 (a), Q2a-7 (a)).
  - Evidence: Plan §2a: "The panels: redrawn once for the new order, with no rearrangement or rebuild, as the Map re-sort." Plan §2: "Live: it is rebuilt with the curves whenever the level or the data changes".
  - Fix: Decide one of two: - (a) The legend recolours its samples in place when only colours change, as the spectrum does, and rebuilds only when the labels or the count change. - (b) A clear-and-redraw of the legend's artists is not a "rebuild" in 2a's sense. Raise it as one question to the owner.
- [contradiction] (decomp + legend) Decomp Q-D13 (a), recommended, says: "The spectrum, legend and Back clicks do nothing while RUN is off". But decomp's own B7 Back callback is `if self.decomp.up(): self._update_all()`, with no RUN test. Legend L-3 ("its click calls the same back action as a right click") and L-8 have no RUN test either. Under the recommendation, a Back click while stopped would still change the level without a redraw, which is the very fault Q-D13 describes.
  - Evidence: Plan §2: "One click, one step: every click changes the level at most once, and the window is redrawn once for it."
  - Fix: If Q-D13 (a) is approved, put the RUN test inside `Decomp.press` / `up`, or in the one simulator entry point that both the click and the Back callback call, and say so in both briefs.
- [contradiction] (port) The port contradicts itself about window edits. Its "Order of work" says "No window file needs an edit to get the panel and the saves". Yet: - its own S6-Q9 (a), recommended, edits run_wfn_plane_slice.py and run_vacupot_plotter.py: their scripted `fig.savefig(..., dpi=...)` before show would become window images of an unplaced window; - S6-Q18 (a) adds simulator window lines; - S6-Q15 updates comments in topo_decomp_window.py and other windows.
  - Evidence: Plan §6: "every window in this repository that uses `qtgui` gets sm4_browser's info panel and sm4_browser's saves."
  - Fix: Reword it: no window file needs an edit except as S6-Q9, S6-Q15 and S6-Q16 to S6-Q24 decide. List those files in the brief.
- [beyond_plan] (port) Port 6.6 adds a module-end binding `from . import mplexport as _mplexport` (root: `from qtgui import mplexport as _mplexport`) and removes savefig's local import. The plan does not name this binding among the pieces. The port's own unverified list says it is not checked that the binding works while mplexport runs `from . import qtplot as q` (a circular import). A local import inside savefig and export_vector would serve sm4's code equally well.
  - Evidence: Plan §6: "The pieces (sm4_browser's names): Figure.export_info, export_info_panel, export_root, _export_rect, window_image and the 'vector', 'qt-svg' and 'image' paths of Figure.savefig in qtplot.py; ..."
  - Fix: Raise it as a question: (a) the module-end binding, as sm4 has it; (b) keep the copy's local import inside the functions that use mplexport. Recommend (b), unless the circular import is shown to work.
- [unverified_claim] (legend) Two Qt delivery claims are stated as facts in the proposal, while the report's own unverified list says the routing was not traced: - L-8: "A right press on the Back button reaches the canvas, because QAbstractButton ignores presses that are not left presses." The QAbstractButton part is sourced. That the ignored press propagates through the layout containers to `FigureCanvas.mousePressEvent`, in viewport coordinates, is listed as unverified. - L9: "QScrollBar handles only the left (and optionally middle) button, so the canvas never sees a right press on it." The unverified list says "That a right press on a QScrollBar is consumed and never reaches the canvas" is inferred.
  - Evidence: Plan §2: "The Back button in the legend panel (levels 1 to 3; level 0 has none): does the same as a right click." Plan §2: "Right click anywhere in ... the legend panel ... goes back one level".
  - Fix: Mark both statements as unverified in L-8 and L9, or cite Qt's QApplication::notify mouse-propagation source. Then L9's event filter (and any filter on the Back button) rests on a sourced fact.
- [beyond_plan] (legend) L-2 makes look and size choices without raising them as questions. Only the fill and border colours are raised (L6): - every button box spans the stack's width less an inset, so the buttons form one column of equal width; - four new named constants with chosen values: LEGEND_SAMPLE_PX = 26, LEGEND_PAD_PX = 4, LEGEND_GAP_PX = 4, LEGEND_ROW_GAP_PX = 2; - the row-height formula.
  - Evidence: Plan §2: "every entry that stands for a clickable curve (levels 0, 1 and 2) has a box around its sample and label, and is clearly a clickable button." Plan §5: "Width (the owner's rule): as wide as its largest entry (sample, label and button box)".
  - Fix: List these under L6 / L15 as choices for approval: equal-width boxes versus boxes fitted to each entry, and the four constants with their values.
- [beyond_plan] (legend) L-6 changes part 1's `_curves_readable` to add the legend's width and the row spacing, in Single Point, Line and Path. This is not raised as a question. It may also be redundant. The legend's own code map says `Arrangement._free_min` "takes the larger of free.minimumSizeHint() and free_min()", and a row holding the spectrum cell and a fixed-width legend already reports both in its minimumSizeHint.
  - Evidence: Plan §5: "the panel takes only the space the layout gives it". Plan §5: "In Line and Path ... the spectrum and the legend panel form one sub-sub-section of their own inside it."
  - Fix: Raise it as a question, or drop it if the sub-section's minimumSizeHint already carries the legend's width.
- [missing_plan_item] (all) No report states the plan's "Unchanged" bullet for legends. It should be in the brief, because: - the legend report adds qtgui edits (L7 MapBlock, L8 set_navigate, L11 an export hook, L14 an export_csv skip); - the port rewrites mplexport (S6-Q6 keeps its legend export); - the decomp removes the spectrum's five legend calls.
  - Evidence: Plan §5: "Unchanged: `qtgui/qtplot.py`'s `Axes.legend` and every other legend in the repository."
  - Fix: Add to the brief: `Axes.legend` and every other legend stay unchanged. Only the five `ax_spec.legend(...)` calls and `add_artist(leg1)` in `_draw_spectrum` go. mplexport keeps its legend export (S6-Q6 (a)).
- [missing_plan_item] (all) The finishing steps are mapped by no report: rule 5's two per-file lists and section 7's py_compile of every changed file plus the commit. Only port S6-Q8 mentions rule 5's lists, for removed functions. No brief lists the files each item changes, which the py_compile step and the rule-5 lists need.
  - Evidence: Rule 5: "At the end, list per file every function, constant and branch you changed, added or removed, in two lists". Plan §7: "`python -m py_compile` every changed `.py` file. Run nothing else. Commit on the same branch as parts 1 and 2."
  - Fix: Add a per-item file list to the brief. It covers ldos_decomp.py and simulator.py; qtplot.py, controls.py and mplexport.py in the package copy, copied over matcher/qtgui; root qtgui under S6-Q1; controls.py MapBlock under L7; and the windows under S6-Q9 / S6-Q15. Every one of these files gets py_compile.
- [other] (markers) P1's lifecycle line "View change: not reset, because marker_ratios is kept (Q2a-13)" cites the wrong question. Q2a-13 is the part-2 timing of the `_on_rel` branch. The view-change question is Q2a-12.
  - Evidence: Plan §2: "A view change resets the level to 0 and the active marker to 0". Plan §2a: "each Line and Path marker keeps its colour wherever it moves".
  - Fix: Change the reference in P1 to Q2a-12.

### Coverage

Mapped means that at least one report gives a proposal item or a question for the bullet.

Section 1 (rules)
- R1 (stop and report): every report raises questions (decomp 19, markers 13, legend 21, port 27).
- R2 (the four report items):
  - ldos_decomp functions: decomp A.
  - simulator calls: decomp B.
  - legend panel per view: legend L-1 to L-6.
  - sm4 functions and what each replaces: port code map §1 and items 6.1 to 6.7.
  - Ruling 9 drops the wait for approval; the port notes this.
- R3 (only py_compile): legend L12, port S6-Q10. No report lists the files to py_compile (finding).
- R4 (sim_wrapper): decomp B1 / Q-D2, markers P8 / Q2a-5, port 6.10 / S6-Q27.
  - All three agree the rule's description is stale at HEAD: X1 removed capture_state and the reads (confirmed with git show).
- R5 (two lists): only port S6-Q8 (finding).

Section 2
- Levels table and "level 3 and 4 go": decomp A1, A5, A7, A8, A10, B4.
- Level 3, which orbitals: A6, Q-D1 (POL misalignment, confirmed at HEAD), Q-D19, Q-D12.
- Level 3, the curves (colour, style, lighter _up): A2, A8, Q-D8.
- Level 3, faded total behind: A8.
- Level 3, hover labels: A11, Q-D9.
- Level 3, legend lists orbitals; left does nothing; right goes back: A9, A10; legend L-2, L-8.
- Maps and heatmaps at level 3: B2.
- Clicks, left on a curve with the top one counting: A4, B7, Q-D3, Q-D4.
- Clicks, left on an entry: A10, L-8.
- Clicks, left on neither: A10.
- Clicks, right anywhere: A10, B7, Q-D14, L-8, L9 (the two reports contradict each other; finding).
- Clicks, Back button: B7, L-1, L-3, L16.
- Clicks, the two ends do nothing and do not redraw: A10.
- Clicks, double click: A10, L-8, L16.
- Clicks, one click one step: A10, B7, Q-D13 (RUN-off gap; finding).
- Legend, no legend on the spectrum: B4.
- Legend, the panel to the right in every view: L-6, L7.
- Legend, order (title, Back, stack): L-1, L-3, L2, L5.
- Legend, entries (sample in its curve's style, label): A9, L-2.
- Legend, short labels: A8, L-2.
- Legend, level 0 (no text; index order; stripe order in Line/Path): A9, L-7, markers P2.
- Legend, clickable entries as boxes; level-3 plain rows: L-2, L6.
- Legend, not entries: A9, L-7.
- Legend, Back button (where, look, a Qt control left out of saves): L-1, L-3, L16, L21, S6-Q13.
- Legend, title (top, fixed, saved; 'Parts' / 'Atoms' / 'Orbitals'): A1, L-1, L-3, L-10, L2, L15.
- Legend, a title wider than the panel stops the session: L12.
- Legend, live: B3, L-7.
- Stays, energy markers: B7, Q-D4 (Pan/Zoom gate not raised; finding).
- Stays, view change resets level and marker: B1 (`_on_mode_change` unchanged), Q2a-12.
- Stays, clicks outside the two panels: B9, P9.
  - Q2a-11 notes the plan's "the redraw _on_press makes for them" no longer exists (confirmed at HEAD ~3374).

Section 2a
- Rule (index 0 at the top): P3, Q2a-1.
- When (re-sort on release): P3, Q2a-13.
- Stripe axis direction: markers code map.
- Stable sort: P3, Q2a-9.
- Permutation applied to the ratios, the active index, the colours and the caches: P4, Q2a-7, Q2a-8.
- Colour stays with the marker: P1, P2.
- Every panel uses the kept colour: P2, Q2a-4, Q2a-6.
- Redrawn once, no rebuild: P5 (conflicts with legend L-7; finding).
- Single Point and Map unchanged: P7.
- Claude's reading (count box sorted; first free colour): P6, Q2a-2, Q2a-3, Q2a-10.

Section 3
- State and settable attributes: A5, B1, Q-D2.
- active_atom goes: A5, B1.
- Partitions for the maps, _redraw_map_slice and heatmaps: A7, B2.
- Spectrum's curves (pickable; level 3 with hover; levels 3 and 4 go): A8, A11, B4.
- Legend entries and title: A9 (the interface contradicts L-9; finding).
- Clicks replace the _on_pick and _on_press spectrum parts: B6, B7, Q-D3.
- Decision made from the curves as clicked: B7, Q-D3.
- Helpers including S: A2, Q-D6.
- Shared data passed in: A6 to A8 (m_colors replaced by marker_colors; finding, Q2a-6).
- vesta_label_map and _get_element_by_index_helper: Q-D7.
- plot_dos_cursors_levels.py not touched: port S6-Q21.

Section 4
- Spectrum and legend fed by ldos_decomp: B3, B7.
- Legends and add_artist go: B4.
- _get_partitions goes: B2.
- Connections and other branches kept: B9.
  - Legend L-4 adds a scroll_event connection, raised under L1.

Section 5
- Where (right, same row, row height): L-6, L3.
- One sub-section; sub-sub-section in Line/Path: L-6, L18; Map: L7.
- Width: L-5, L4 (label set conflicts with Q-D1; finding).
- Clicks: L-8, L9.
- Built as a qtplot axes with lines, texts and boxes: L-1, L-2, L6, L10.
- Saved with the figure: L-10.
- Entries that don't fit scroll under a fixed title and Back; panel takes the given space: L-4, L17, L13.
- Scrolling as sm4's _MarkerBox (wheel, as-needed bar): L-4, L1, L8.
- Saves (vector full list; Qt as shown): L-10, L11, L20, L21; port 6.8, S6-Q14 (hooks contradict; finding).
- Unchanged Axes.legend and other legends: no report (finding).

Section 6
- Every qtgui window: port S6-Q1, S6-Q9, S6-Q21 to S6-Q23.
- Source eb9bb62, own copy, no import from sm4: port 6.6.
  - "The same as the owner's copies on disk" is unverifiable (port's unverified list).
- Today's state: port code map.
- Info panel wraps: 6.1.
- Shows only once it has text: 6.2, S6-Q26.
- Its text with the 'settings' section: 6.3.
- Updated on change (Control._changed, hook, push): 6.4, S6-Q11, S6-Q12.
- Save list replaces this repository's: 6.5.
- PDF / EPS / SVG vector with the full info: 6.7, S6-Q3 to S6-Q7.
- qt-svg as shown, with its label: S6-Q2 (docstrings would still say "full info"; finding).
- PNG / TIFF at 720 dpi: 6.6, S6-Q3, S6-Q25.
- CSV: 6.5 (legend L14 skip missing; finding).
- The pieces: 6.3, 6.4, 6.6, 6.7.
- Two scrollable panels in each save: 6.8, L-10, S6-Q13, S6-Q14.
- Ruling 8: port 6.9, S6-Q16 to S6-Q24; decomp Q-D16; legend L19 (three proposals; finding).

Section 7
- py_compile, commit, the two lists: no report maps them (finding).

## Check: code facts

- [contradiction] (port) The owner's response document was committed during this run (7ae0a46, _handoff_in_progress/RESPONSE_PART1_PART3.md). It says it overrides the plan where they differ. Its §2 (line 57) and Q4 (lines 118-120) put the vector files' figure size through pt2px (sm4 mplexport line 112), and they keep `Figure(..., dpi=q.DPI)` under P10 (lines 40, 59). If the sm4 functions that divide by the matplotlib figure's pixels are taken unchanged, that pair puts every panel and every info line in the wrong place. This is trap 1 of S6-Q3, and the owner's answer is exactly that combination.
  - Evidence: sm4 mplexport.py `_pos(ax, mfig)` (~138-146): `Wf, Hf = mfig.get_figwidth() * mfig.dpi, mfig.get_figheight() * mfig.dpi; return [x0 / Wf, 1.0 - (y0 + h) / Hf, w / Wf, h / Hf]`. `_info_block` (~493-519) uses the same `Wf, Hf` with `y = H + margin + i * line_h + ascent` in logical pixels. In the copy, figsize W/pt2px(72) inches × dpi 100 gives W·100/(72·px_per_pt()) pixels, not W (qtplot `pt2px` ~355, `DPI = 100.0` ~266). The offset disappears only if px_per_pt() == 100/72.
  - Fix: Report this as a question for the owner; do not choose. Recommended: keep pt2px figsize and dpi=q.DPI as he ruled, and compute `_pos` and `_info_block`'s fractions from the logical pixels W and H + H_info instead of `mfig.get_figwidth()*mfig.dpi`. That is a change to sm4's two functions, and rule 5 has to list it.
- [other] (port) Several port items and questions are already answered or overridden by RESPONSE_PART1_PART3.md, which the report could not have seen. (1) 6.5 says `_save` drops 'vector' without matplotlib, and 6.6 takes sm4's savefig fallback. The response §4 (lines 84-93) makes matplotlib a module-level import of qtplot and drops both the filter and the fallback. (2) S6-Q2(a)'s 'only the no-matplotlib fallback keeps the full block' no longer applies: export_vector serves only the 'as shown' SVG. Its label is the response's Q7 wording, and it has no appended info. (3) S6-Q3's 'export_vector's resolution and page size as P10 is ruled' is replaced by 720 dpi (line 79). (4) S6-Q1 is answered (a) (Q5, line 124), and S6-Q12 is answered (a) (Q8, line 147). (5) S6-Q16/Q17 are answered by Q6 (a) (lines 131-139), with a list of window lines; see the separate finding. (6) 6.10 'sim_wrapper: no change' is overridden: §6 (line 195) changes its `cmap_topo` to `cmap_set`, an exception to rule 4. §6 also adds a colour-map set choice to the simulator's left column, a new control that the settings section will list. (7) The legend report's L1, L4 and L12 are answered (a) by the response's Q1-Q3. The legend report's L19 and decomp's Q-D16 are answered by Q6 (a): the decomp level, active marker and active element go in as a window line through set_info_line.
  - Evidence: RESPONSE_PART1_PART3.md lines 17, 40, 57-59, 79, 84-93, 96-147, 160-200 (commit 7ae0a46; HEAD is now b7132c0).
  - Fix: Re-base the port brief, the legend questions and Q-D16 on the response document before the owner reviews them. Drop the answered questions and re-word 6.5, 6.6, S6-Q2, S6-Q3 and 6.10.
- [missing_plan_item] (port) The owner's Q6 rule is 'every setting the window makes with the mouse'. The contour slice level z_frac is set by dragging the level line on the topo colour bar, but it is not in the response's list (decomp state, marker positions, Line ends, Path nodes, Map energy markers, crop rectangle). S6-Q18 also proposes lines the owner's list does not hold and that are not set with the mouse: 'orientation' (cell_rotated, chosen automatically), 'bias', 'applied heights' and 'state' (running or stopped, global topo stale).
  - Evidence: simulator.py `_on_pick` (~3331): `if label == '_zlevel': self.active_obj = ('zlevel', event.artist.axes)`. `_on_motion` (~3393-3399): `frac = self._ramp_fraction(event, self.active_obj[1]); ... self._set_level(frac)`. `__init__` (~1027): `self.z_frac = 0.5`. RESPONSE lines 131-139.
  - Fix: Ask the owner whether z_frac (Map only) is a line under his Q6 rule. Trim S6-Q18 to his list, plus z_frac if he says yes, and mark orientation, bias, applied heights and state as beyond his answer.
- [contradiction] (decomp) Q-D8 recommends (a): drop the inert `pickradius=` from the level 0-2 curves. PART1_OPEN P9's R is to leave `pickradius=` on curves as it is, and the response document accepts every R except P2. Dropping it is a change no plan names.
  - Evidence: PART1_OPEN.md P9 (~88-93): "`pickradius=` on curves (qtplot never reads it; `picker=True` uses qtplot's converted default). **R:** leave them." RESPONSE_PART1_PART3.md line 17: "Accept every R except P2." simulator.py `_draw_spectrum` (~2769, 2777, 2788-2789, 2796): `picker=True, pickradius=5` / `pickradius=3`.
  - Fix: Re-state Q-D8 with P9 accepted. Recommend (b): carry `pickradius` as today, which is inert. Or put the choice to the owner as an exception to P9.
- [wrong_fact] (decomp) Q-D1 says the maps' level-3 partitions in a spin-polarised run are 'wrong or crash', and option (c) reads 'misaligned labels, IndexError past column 8', with CuPy's behaviour unverified. CuPy's source settles it. A scalar int index is bounds-checked and raises IndexError (CuPy wraps only out-of-range entries of integer arrays). `_get_partitions` indexes `dos_up_gpu[e_idx, :, col_idx]` with col_idx a Python int, and the loop reaches col_idx 9 before it returns. So for any element with atoms, every level-3 call in a POL run raises, and no partition list with misaligned labels is ever returned. Today's spectrum level 4 raises the same way, at `active_ldos_norm[:, a_idx - 1, col_idx]`. The error is printed by `FigureCanvas._fire`, and `_update_all` / `_redraw_map_slice` stop part-way.
  - Evidence: simulator.py `_get_partitions` (~1146-1150): `for col_idx, orb in enumerate(self.orbitals): if e_idx: D = cp.sum(self.dos_up_gpu[e_idx, :, col_idx], axis=0)`. `parse_vasp_outputs` (~545-560): `num_cols = dos_parser.spin_up_dos.shape[2]` (9 after doscar_parser's `site_dos[:, :, 0::2]`, ~143), then `mapping_pol[9]` (18 names). CuPy `_view_getitem` (https://raw.githubusercontent.com/cupy/cupy/main/cupy/_core/_routines_indexing.pyx): `if not (0 <= ind < a._shape[axis_a]): ... raise IndexError(msg)` for scalar ints. CuPy docs (https://docs.cupy.dev/en/stable/user_guide/difference.html): integer-array indices wrap.
  - Fix: Re-word Q-D1 and option (c) to: 'in a POL run every level-3 partition call, and today's level-4 spectrum, raises IndexError at the tenth orbital name'. Move the CuPy claim from unverified to verified, citing the CuPy source; the installed CuPy version is still unchecked.
- [unverified_claim] (decomp) The unverified item 'cp.sum and quadrature.simpson accept the NumPy drag cache' can now be partly settled, and it changes Q-D12's premise. cp.sum hands a NumPy array to its own .sum, so that part runs on the CPU. But quadrature.simpson converts its input with `cp.asarray(y, dtype=cp.float32)` and returns a CuPy array. With Norm on, every Line/Path drag motion (`_move_line_markers` → `_draw_spectrum(self._line_disp, ...)`) already uploads to the GPU, against A8, and then divides a NumPy array in place by a CuPy array (`active_ldos_norm /= norm_factor`; `t_y /= (quadrature.simpson(...) + 1e-15)`). What NumPy does with that mixed in-place division is still unverified. Q-D12 names the 1e-5 filter as the only thing that would break A8; under Norm, part 2's current code already does.
  - Evidence: quadrature.py `simpson` (~114-117): `y = cp.asarray(y, dtype=cp.float32)`; docstring: 'Returns a float32 CuPy array'. simulator.py `_draw_spectrum` (~2754-2767). CuPy sumprod.py (https://raw.githubusercontent.com/cupy/cupy/main/cupy/_math/sumprod.py): `return a.sum(axis, dtype, out, keepdims)`.
  - Fix: Flag this to part 2 as an interaction, not to act on (it is part 2's code). Add it to Q-D12 and Q-D20: the `norm` callable has to stay off the GPU in the drag path, or part 2 decides how.
- [wrong_fact] (port) S6-Q8 option (a) removes `_size_override` 'and its read in _canvas_size'. `_size_override` is read in four functions, and removing the attribute without the other three reads raises AttributeError.
  - Evidence: qtplot.py (package copy): `_canvas_size` (~6711) `if self._size_override is not None:`; `_fit_controls` (~6852) `if lay is None or self._fitting or self._size_override is not None: return`; `_schedule_fit` (~6969) `... or self._size_override is not None:`; `fit_window` (~6996) `if self._size_override is not None or getattr(self, '_closed', False):`. The root copy has the same reads.
  - Fix: List all four reads (and the `__init__` line ~6644) in S6-Q8(a) and in rule 5's list.
- [missing_plan_item] (port) Brief 6.7 changes `_pos(ax)` to sm4's `_pos(ax, mfig)` but does not name its other callers in the copy's mplexport. They are the figure-text filter (gone under S6-Q7(a)), `_axes` and `_colorbar`, and each needs its call changed. S6-Q4(a), on-screen text sizes, also changes the copy-only `_axes` (labels, ticks, title, legend and extra-legend font sizes), `_artist` (the Annotation size), `_text` and `_colorbar`. These are copy functions that rule 5 has to list.
  - Evidence: Copy mplexport.py: `_to_matplotlib` ~142 `bottom = min((_pos(ax)[1] for ax in axes), ...)`; `_axes` ~165 `m = mfig.add_axes(_pos(ax))`; `_colorbar` ~438 `m = mfig.add_axes(_pos(cax))`; `_text` ~407 `fontsize=t._size` (points as set).
  - Fix: Add `_axes`, `_colorbar`, `_artist` and `_text` (copy versions) to 6.7 as changed, with what changes in each.
- [wrong_fact] (markers) Unverified item 1 says that with more than 10 markers, `self.marks.set_facecolors(m_colors[:n])` gives `_refresh` 10 brushes for n points. Q2a-10 says the spectrum and the counterparts cycle `m_colors[i % 10]` 'while self.marks gets m_colors[:n], which holds only 10 colours', implying a difference. Both are wrong. PathCollection's face rows come from `_rgba_rows`, which `np.resize`s the 10 colours to n rows. So the marks also cycle `m_colors[i % 10]` with n brushes, and today every panel's colours agree.
  - Evidence: qtplot.py `PathCollection._face_rows` (~2408-2416) → `_rgba_rows(self._fc, n, self._alpha)`; `_rgba_rows` (~2359-2360): `if len(rows) < n and len(rows): rows = np.resize(rows, (n, 4))`; `_refresh` (~2420-2423) builds one brush per row.
  - Fix: Remove unverified item 1. Re-state Q2a-10's background: with more than 10 markers, today every panel, the marks included, cycles m_colors[i % 10]. Option (a) is then exactly today's colours.
- [wrong_fact] (decomp) The report quotes the copy's `Figure._pick` hit list as `[a for a in ax._children if a.pickable() and a.contains_px(e.x, e.y)]`. That is sm4's line (sm4 qtplot ~5158). The copy builds `(artist, _hit_index)` pairs and sets `ev.ind`. The conclusion (hits built once, then fired in `_children` order) is unchanged.
  - Evidence: Copy qtplot.py `_pick` (~7094-7102): `hits = [(a, getattr(a, '_hit_index', None)) for a in ax._children if a.pickable() and a.contains_px(e.x, e.y)]` … `ev.ind = [ind] if ind is not None else []`.
  - Fix: Correct the quotation.
- [wrong_fact] (decomp) The list of what sim_wrapper still reads (rule 4's surface) is incomplete. At HEAD, sim_wrapper also reads `sim.fig`, `sim.add_control_box`, `sim.use_decay`, `sim.show_mag`, `sim.npts`, `sim.marker_coords`, `sim.lv`, `sim.grid_xy`, `sim.global_topo_bias`, `sim.global_z_map`, `sim.topo_height`, `sim._calculate_ldos_at_points_gpu`, `sim.run_interactive` and `getattr(sim, '_broadening_topo_dirty')`. `npts`, `marker_coords` and `show_mag` sit in the Line/Path and Single Point/Map paths that §2a's re-sort feeds.
  - Evidence: sim_wrapper.py `send_to_decomposition` (~208-268): `int(r * (sim.npts - 1))`, `enumerate(sim.marker_coords)`, `sim.show_mag`, `sim.use_decay`. `_attach_widgets` (~156-187): `sim.fig`, `sim.add_control_box`. `send_topo_to_decomposition` (~277+). HEAD b7132c0 moved the m_colors reads to ~235 and ~259.
  - Fix: Complete the list in the code map. None of the decomp's changes touch these names; state that.
- [wrong_fact] (legend) L-4 forwards the wheel 'when event.inaxes is the stack axes and |dy| ≥ |dx|'. Two code facts are missing. qtplot fires scroll_event only when `angleDelta().y()` is non-zero, so a purely horizontal wheel never reaches the handler. The MouseEvent carries no dx, only `step = dy/120` and `button`, so dx has to be read from `event.guiEvent.angleDelta().x()`. The canvas also always calls `super().wheelEvent(ev)` after the callbacks, so the event goes on to QGraphicsView as well (already in the report's unverified list).
  - Evidence: qtplot.py `FigureCanvas.wheelEvent` (~6289-6296): `dy = ev.angleDelta().y(); if dy: step = dy / 120.0; e = self._event('scroll_event', ev, button='up' if dy > 0 else 'down', step=step); self._fire('scroll_event', e)` then `super().wheelEvent(ev)`.
  - Fix: Re-word L-4: the handler sees only events with a vertical component, and it reads dx from guiEvent.
- [missing_plan_item] (legend) Neither the legend report nor decomp's Q-D18 lists the README's layout text, which becomes incomplete once a legend panel sits beside the spectrum in every view. Q-D18 names only README lines 66-68 and the module table.
  - Evidence: LDOS_spread_decomp/stm_from_dft_opt/README.md ~30-33 ('the curve panels (Tip Height and Partitioned LDOS, side by side ...') and ~96-102 ('the curve panels (the Tip Height profile and the Partitioned LDOS) take the room those leave ...; in Map the spectrum is one cell of the map block, a map's size ...; and in Single Point the spectrum stands beside the placement topo as its equal').
  - Fix: Add these lines to the stale-text question (Q-D18 / A15 practice), to be updated after part 2's workflow has finished.

### Coverage

STATE. During this run HEAD moved to b7132c0 (part 2) and 7ae0a46 (RESPONSE_PART1_PART3.md). b7132c0 only re-words docstrings in simulator.py, with equal hunk sizes, so every simulator line number in the reports still holds. sim_wrapper's lines moved up by one (236→235, 260→259). The tree is clean.

PLAN BULLETS AND THE REPORT THAT MAPS EACH.

§2:
- Levels table and the removal of old levels 3/4: decomp A1, A7, A8, B4.
- Level 3, which orbitals (1e-5 filter, POL _up/_down): decomp A6, Q-D1, Q-D19.
- Level 3 curve style: decomp A8, Q-D8.
- Faded total behind the curves: decomp A8.
- Hover labels: decomp A11, Q-D9.
- Level-3 legend rows and the clicks there: decomp A9/A10, legend L-8.
- Maps and heatmaps stop at the element's orbitals: decomp B2.
- Left click on a curve, top one counts: decomp A4, B7, Q-D3, Q-D4.
- Left click on an entry: legend L-8, decomp C.
- Left click on neither, right click anywhere, the two ends, double click, one click one step: decomp A10, B7, Q-D13, Q-D14; legend L-8, L9, L16.
- Back button: legend L-1, L-3, L16, L21.
- No legend on the spectrum: decomp B4.
- Panel to the right in every view: legend L-6.
- Order title / Back / stack: legend L-1, L-3, L5.
- Entries, short labels, level-0 order: decomp A9; legend L-2, L-5, L-7; markers P2.
- Button boxes, level 3 as plain rows: legend L-2, L6.
- Not entries: decomp A9, legend L-7.
- Title words and width, stop-and-report: legend L-3, L-5, L12.
- Live: legend L-7, decomp B3.
- What stays (emarkers, view change, clicks outside): decomp Q-D4, B1, B9; markers Q2a-11.

§2a:
- The rule and when: markers P3, Q2a-1, Q2a-9.
- What the permutation is applied to: markers P4.
- Colours with the marker, every panel: markers P1, P2, Q2a-4/5/6/8.
- One redraw: markers P5.
- Single Point and Map unchanged: markers P7.
- Claude's reading on the count box: markers P6, Q2a-2/3/10.

§3:
- State: decomp A5, B1, Q-D2.
- Partitions: decomp A7, B2.
- Curves: decomp A8, B3.
- Legend entries: decomp A9, legend L-9.
- Clicks replace the _on_pick and _on_press spectrum parts: decomp B6, B7.
- Decision taken as clicked: decomp A4, B7.
- Helpers move: decomp A2, Q-D6.
- Shared data passed in: decomp A6-A8; markers Q2a-6.
- vesta_label_map and the helper: decomp Q-D7.
- plot_dos_cursors_levels stays as it is: port S6-Q21 only.

§4:
- Spectrum and legend fed by ldos_decomp: decomp B3, C.
- Five legend calls and add_artist go: decomp B4.
- _get_partitions goes: decomp B2.
- Kept connections and branches: decomp B9.

§5:
- Where and one sub-section: legend L-6, L3, L7, L18.
- Width: legend L-5, L4.
- Clicks: legend L-8.
- How it is built: legend L-1, L-2, L6.
- Scrolling: legend L-4, L1, L9.
- Saves: legend L-10, L11; port 6.8, S6-Q14.
- 'Axes.legend unchanged': no report says so, but none changes it.

§6:
- Every window: port S6-Q1, 6.9.
- Wrap, show once it has text, settings text, updates: port 6.1-6.4.
- Save list: port 6.5.
- Vector, qt-svg, 720 dpi images, CSV: port 6.6, 6.7, S6-Q2/Q3/Q25; legend L14.
- Pieces: port 6.3-6.7.
- Two scrollable panels in the saves: port 6.8, legend L-10.
- Ruling 8: port 6.9, S6-Q16-Q24; decomp Q-D16; legend L19.

RE-CHECKED AND FOUND CORRECT (code, current line numbers).

Spin-polarised mismatch:
- doscar_parser keeps 9 of 18 columns (0::2), and `mapping_pol[9]` has 18 names. `f_ldos_raw` and `dos_up_gpu` have 9 orbital columns. NCL and unpolarised are aligned.
- The IndexError is now verified for the scalar column index, from CuPy's source.

_get_partitions:
- Body as described, with levels 3 and 4 identical.
- Its only callers are _update_all (2564) and _redraw_map_slice (3278); that holds repo-wide outside wrong_decays/ and plot_dos_cursors_levels.
- vesta_label_map and _get_element_by_index_helper have no other users.

_draw_spectrum:
- Callers 2586, 3141, 3371.
- Cache key 2716-2721.
- The in-place `_curve` only calls set_data.
- Every level's colour, lw, alpha, zorder and picker as reported.
- The five legend calls at 2779, 2790, 2798, 2809, 2836/2838, add_artist at 2837.
- HoverLabels at 2829.
- The ylim block counts `get_alpha() == 1.0` only (Line2D alpha defaults to None).
- Title at 2846; emarkers axvline with picker 5 at 2861.

Event handlers:
- _on_pick branch order; no button or dblclick checks.
- b051547's `_on_press` called `_update_all` on every press; today's handles ax_spec only.
- _on_motion and _on_rel, X3 as described.
- `_on_mode_change` does not reset active_element.
- Connections 1427-1430.
- `_update_all` returns early while not running.

Line/Path markers:
- `marker_ratios = [0.25, 0.75]`, ascending; run_interactive's projection is unsorted.
- `_on_ui_change` replaces the list on an increase and truncates on a decrease; active_marker_idx is never clamped.
- `Count.set` neither clamps nor fires; Markers is limited 1..10.
- `_make_line_markers` (topo / prof / h, only while heatmaps exist), `_place_line_markers` and `_move_line_markers` as described.
- Stripe ylim (0, p_len); nothing inverts it; the heatmap row never turns; ax_line_topo is None in Path.
- m_colors readers at 2404, 2412, 2415, 2768, 2776, 3094, 3199, 3263; sim_wrapper at 235 and 259.

qtplot (package copy):
- MouseEvent and PickEvent (the same mouseevent object); `_QT_BUTTONS`.
- `_event`: inaxes is the plot box via `_axes_at`.
- Press, double-click and release order; `_fire` keeps going after a handler's error.
- `_pick` builds its hits once, in `_children` order. `Axes._register` appends, so the markers report's unverified insertion-order item is confirmed.
- Artist zorder default 2; `_add_item` sets Z, then addItem. Line2D and _RefLine reuse their item on set_data and set_color. pickradius is never read.
- `clear()` keeps the HoverLabels registry; `remove()` drops it. HoverLabels `_hit` matches removed curves, and `arm_by_click=False` makes its click handler return at once.
- set_title and short_title; set_info_line; info_text order.
- `axis('off')` gives decor 0; set_facecolor restores the background.
- No Rectangle; `patches` is []. LineCollection takes string line styles only. mkpen takes dash tuples. Legend sample 26 px, horSpacing 4.
- PanelCell is mouse-transparent; rect_px; on_placed; set_layout on the viewport.
- MapBlock calls `_decor` on the extra cell; Arrangement candidates 'h' and 'v'; `_free_min`; FlipBox; UnderPanel._follow.
- set_navigate is a no-op; `_apply_toolbar_mode` as described; the topo colour bar calls set_navigate(False).
- qfont's 10 pt floor and em_px. Copy and root FigureManager info widget, SAVE_FORMATS, `_save`, savefig, `_render_scene`, export_vector. at_design_size's only caller is mplexport. rcParams savefig.dpi.

Controls:
- The copy and root Control classes are byte-identical, and the Slider `_changed` collision is at 1349/1358.
- SigmaBox and GammaBox wrap Value/Text controls and are not Controls themselves.
- The simulator's control list is complete (grep), and LABELS gives '' for the cursor boxes.

sm4_browser:
- Control, setting(), `_changed`, settings_text, plt._settings_text.
- FigureManager hook and show(); `_settings_text` global; export_info_panel.
- info_text, savefig, export_root, `_export_rect`, export_info, window_image, `_render_scene`, export_vector, the module-end binding.
- mplexport `_to_matplotlib`, `_pos(ax, mfig)`, `_text` (figure_xy, which the copy also has), `_info_lines`, `_info_geometry`, `_info_block`, export.
- `_MarkerBox`; `_pick` sets no ind; qfont has no floor.
- Transitive dependencies are present in the copy and in the root: QuadItem.EXPORTING, mplexport.available, figure_xy, _have_matplotlib, _ghostscript, export_csv (identical), `_push_info` and `set_info` (identical), screen_scale.

sim_wrapper:
- At HEAD it has no capture_state, plot_level or active_marker_idx.
- At d5e08a4, the restore happens before run_interactive (196-199), capture_state is at 253-264, and marker_ratios is set after the run (203-204).

Windows:
- qtgui use per file as reported; no vendored stm/qtgui copy exists.
- forces.py's canvas3d is the only raw widget.
- The scripted savefig calls are at run_wfn_plane_slice 586 and run_vacupot_plotter 101, and the checks' calls are as reported.

