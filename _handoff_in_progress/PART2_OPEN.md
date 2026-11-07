# Part 2: open items for the owner

Part 2 and its added sections (X1-X3) are implemented. Each file group was reviewed three ways (completeness,
scope, formulas by derivation) and then fixed, and one cross-module review ran at the end. No agent made a change
the plan and decisions do not name. Every changed .py file passes `python -m py_compile` (12 files). The two package
qtgui copies are byte-identical, the root `qtgui/` and every `wrong_decays/` are unchanged since d5e08a4, and the
per-file change list is `PART2_CHANGES.md`.

Each item below needs a ruling. The reviewers confirmed every one against the code. Each has a recommendation (R).
Answer per item, or "accept all R except ...".

## Run-time problems the plan's rules collide on

**O1. Norm during a Line / Path marker drag raises TypeError.** A8: the drag cache is NumPy and no GPU work
happens during a drag. B19 / 7.3: every Simpson integral goes through `quadrature.simpson`, which runs on the GPU.
`_draw_spectrum`'s Norm divides the NumPy drag curves in place by `quadrature.simpson`'s CuPy result, which raises
TypeError on every motion event (Line / Path, Norm on, a marker drag or a `mark_dynamic` pick). No reading satisfies
both rules, so it was left as it is.
- (a) The Norm factors are computed once per data change, on the GPU with `quadrature.simpson`, and kept as NumPy
  beside the drag cache; during a drag `_draw_spectrum` divides by them. No GPU work in the drag, and every Norm
  integral still comes from `quadrature.simpson`. The numbers are the same.
- (b) The drag cache is built already normalized.
- (c) GPU work in the drag: `float(quadrature.simpson(...))`, one host sync per curve per motion (breaks A8).
- (d) A NumPy Simpson for the drag only (breaks "everything integrates with quadrature.py").

R: (a).

**O2. Map after an energy-marker drag (A5).** A5 compares the snapped Map indices on a full refresh. `_on_rel` keeps
`map_e_idx` in step with a dragged map (X3 needs that). So the first full refresh after a drag, which includes
display-only toggles (Norm, Feenstra, Atoms, gamma), re-forms the Map energies from the typed ends, as before part 2,
and now also recomputes the whole Map LDOS.
- (a) Keep: the maps always match their titles, markers and Norm, at the cost of one recompute.
- (b) Compare against the indices of the last LDOS compute, which `_on_rel` does not update: no recompute, but the
  dragged map's data stays under the re-formed title, as before part 2.

R: (a).

**O3. `_display_indices` is defined only on `Interactive_STM_Simulator`.** The base class
`Unified_STM_Simulator._calculate_ldos_at_points_gpu` calls it on the emin / emax path (A12(i)), so a bare
`Unified_STM_Simulator` raises AttributeError there. Nothing in the repository makes a bare one (sim_wrapper and
run_DFT_STM use the interactive class), but `__init__.py` exports it.
- (a) Move `_display_indices` unchanged into `Unified_STM_Simulator` (it reads only `energies`, `k0`, `e0`).
- (b) Call `_snap` directly there (duplicates its logic).
- (c) Accept, and say in the docstring that this path needs the interactive class.

R: (a).

## Edge cases the plan does not decide

**O4. `feenstra.iv_ratio` when every sample is exactly at 0 V** (e0 == 0.0 and E Min and E Max both snap to the 0 V
sample, Feenstra on): CuPy raises IndexError. Before part 2, `np.interp` raised ValueError in the same case.
(a) accept (still an error); (b) raise ValueError explicitly (a check no item names); (c) give that column a value.
R: (a).

**O5. `_snap`'s 'zero_side' tie when the value is exactly 0.0 and the two samples around it are exactly symmetric**
(practically unreachable): it gives the lower sample now. R: keep.

**O6. X3's "drag cache" in Map:** Map has none (`_line_disp` is Line / Path only), so nothing is permuted for it;
every other array X3 lists is permuted. R: accept.

## Ruling 5 (physics vs machinery)

**O7. The Decay-off weight.** The physics block holds only a pointer comment. The weight itself is two inline lines
in `_compute_channel`'s two Decay-off branches, under NUMERICS AND GPU.
- (a) A def in the physics block, called by both branches with the same expressions (no change to the computation).
- (b) Keep the pointer and mark the two inline lines `# Physics: ...`, as the kernel's lines are marked.
- (c) Accept as is.

R: (a), since ruling 5 says the block holds the Decay-off weight.

**O8. Defs with no docstring.** Every existing def docstring in simulator.py now starts with "Physics:" or
"Machinery:". About 25 defs have no docstring (e.g. `__init__`s, `_get_partitions`, `_on_pick`, `_on_rel`, nested
helpers), so there is nothing to tag. (a) leave them; (b) one-line "Machinery:" docstrings for the defs part 2
changed; (c) one for every def. R: (a).

**O9. Tags outside simulator.py, feenstra.py and quadrature.py** (`read_doscar_header`, sim_wrapper's `__init__` and
`launch`): (a) leave, as the decisions sheet scopes ruling 5; (b) prefix "Machinery:". R: (a).

**O10. The stm README's simulator.py row** lists physics (`gpu_chen_tunneling_factor`, the tilted barrier) and
machinery (GPU LDOS, the integration rule, float32, convergence, caches) in one parenthetical. (a) leave (the Decay row
holds the model); (b) split it into the physics and the machinery. R: (b).

## README wording

**O11. The Decay row** names Chen's Eq. 2.40 (plan 11) and keeps its existing "(Eqs. 2.34, 2.37-2.39: ...)" and
"Chen's φ̄". (a) keep both (they are Chen, as ruling 3 asks, and match the simulator's docstring); (b) drop the
parenthetical; (c) also drop "Chen's φ̄". R: (a).

**O12. The bottom-strip sentence** keeps the Map-energy tie clause ("each Map energy snaps to the nearest displayed
sample, a tie going to the sample closer to 0"); three clauses no item names were removed. (a) keep it; (b) keep it
and add topo_bias's snap; (c) remove it. R: (a).

**O13. The stm README's Dependencies line still lists scipy.** After part 2 nothing in `stm_from_dft_opt` imports
SciPy (the decomposition package still does, in its own README). R: drop scipy from that line.

## Texts that became false (not named)

**O14. sim_wrapper.py after X1:**
- the class docstring's "then simulator instances come and go per σ commit" → "one simulator instance; a σ commit
  only sets the MO spread smearing";
- `launch`'s "Prepare the working directory and start the simulator" → "Start the simulator on the run directory";
- the section comment "# σ commit: re-smear → new σ-tagged dir → capture → tear down → re-instantiate → restore." →
  "# σ commit: sets the MO spread smearing for the next Send press.";
- `_on_sigma`'s comment: only its stated reason (the teardown) reworded; the deferral stays;
- `commit_sigma`'s "Real-DOSCAR mode" comment and print: kept (still true);
- the module docstring's removed clause "a σ commit makes a new simulator instance ...": confirmed removed.

R: as listed.

**O15. run_stm_decomp.py:** `reuse_cache=True  # per-σ-dir topo caches` → "# the run directory's topo caches";
`use_decay=True  # Chen barrier model ...` → the tilted barrier, as run_DFT_STM.py's comment now says. R: change both.

**O16. run_DFT_STM.py:** two comments still say "slider" (stale since part 1: the sliders are number boxes):
`erange` "initial energy slider range" and the "Path Pts" slider. R: "box" in both.

**O17. PROCAR_tools/README.md's procar_dos.py section** says the ion-row tot column and the phase identities are
checked on every band; the code spot-checks them every (nbands // spot_check_bands)-th band (only the total-block sums
run on every band). procar_dos.py's own docstring has the same slip, and X2 keeps it unchanged.
(a) cut the README section to what X2 names; (b) say which checks run on every band and which are spot-checked.
R: (b), README only.

**O18. Historical files:** `LDOS_spread_decomp/CHANGE_REPORT.md` and `Claude outputs/feenstra_changes.patch` mention
the removed `synthetic_doscar.py`, `procar_dos.py` and σ-tagged dirs. R: leave them (records).

## For information

- **Commit history.** The workflow's `git rm synthetic_doscar.py` and `git mv procar_dos.py` were already staged when I
  committed a progress-log entry, so commit eb6ce36 holds them, and the importer edits came in the next commit
  (e1995d7). No later commit is affected, and HEAD is consistent. Fixing it would need a history rewrite and a force
  push, so I have left it.
- **Part 3's rule 4 is moot.** After X1, nothing in `mo_ldos_decomp/` reads or sets `plot_level` or
  `active_marker_idx`. Both stay attributes of the simulator.
- **One fix made after the workflow, under section 9** (every CuPy array float32): the two `grid_z` stacks in
  `_update_all` and `_on_rel` now build float32 heights (`cp.asarray(self.current_z_map, dtype=cp.float32)`), as
  `run_interactive` already does. Today every height map is float32 already, so nothing changes in the numbers.
