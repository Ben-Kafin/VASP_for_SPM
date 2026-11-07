# Part 2: the decisions in force (one sheet for the implementing agents)

Sources, in order of precedence (a later one overrides an earlier one where they differ):
1. `handoff_part2_energy_integration_gpu.md` (the plan; line numbers refer to the owner's `disk/` simulator.py
   before part 1, which part 1 has since restructured, so find code by the names given)
2. `PART2_QUESTIONS.md` (A1-A16, B1-B19 with recommendations R)
3. `PART2_ANSWERS.md` (the owner's answers: every R accepted except A1, A2, A4, A9, A11, A15 as answered
   there; group B approved as a block; group C noted, no change; three sections added)
4. The owner's later rulings in `PROGRESS.md` (3 to 7)

This sheet restates 2 to 4 in one place. Where it and the plan differ, this sheet governs.

## Group A, in force

- **A1 (guard tolerance):** a printed energy may differ from its rebuilt value by at most half a unit of the
  last decimal it carries, read from the file's own text (the number of digits after the decimal point of
  that printed energy), plus 1e-6 eV. For 3 decimals: 0.0005 + 1e-6 eV. Strict `>` raises ValueError;
  the message names the largest difference and its file line.
- **A2:** no check for V_b at or across 0 V (the owner never gives a bias within one DOS step of 0).
- **A3:** a Map energy snaps to the nearest sample among the displayed samples (from the snapped E Min
  sample to the snapped E Max sample, an end at exactly 0 V excluded as the display excludes it); a tie
  goes to the sample closer to 0. So a Map energy of exactly 0.0 goes to the nearest sample on the
  window's side of 0, and an end Map energy never lands outside the displayed range.
- **A4:** Map energies that snap to the same sample show the same map. The Map Norm integrates over each
  distinct sample once (sorted, duplicates removed).
- **A5:** `needs_ldos` in Map also compares the snapped Map sample indices; `map_e_targets` holds the
  snapped energies (set where the targets are formed and on release); the energy markers are drawn at the
  snapped energies.
- **A6:** the weight in three branches: a > 0 and b > 0 → the first form; a ≤ 0 and b ≤ 0 → S = ∂S/∂V = 0;
  otherwise → the second form. The same in the CUDA kernel.
- **A7:** the display sweep (preserve_orbitals=True) is chunked too, by the (requested V, sample) pairs per
  point, slicing `out` and `cur` per chunk; the chunk at the collapsed branch likewise.
- **A8:** the drag cache is NumPy, built once per data change (no GPU work during a drag).
- **A9:** moot (the synthetic DOSCAR is removed: a σ commit never re-instantiates the simulator).
- **A10:** the current is computed on every display run (`with_current=True` always on those calls).
- **A11:** `quadrature.simpson(y, x=None, *, dx=1.0, axis=-1)` — SciPy 1.18.1's own signature (dx and axis
  keyword-only), its arithmetic transcribed exactly, its division guards kept as `cp.where` with a safe
  division (fill 0 where SciPy fills 0). `quadrature.py` imports nothing from SciPy. iv_ratio's sample
  exactly at 0 V: a float32 two-neighbour linear interpolation made with `cp.searchsorted` / `cp.where`,
  vectorised over points (not `cp.interp`, which computes in float64). `divide`:
  `cp.where(d != 0, a / cp.where(d != 0, d, 1), 0)`.
- **A12:** (i) a '0 V' end on the emin/emax path (sim_wrapper's topography send) uses the display rule:
  start at the first sample on the window's side of 0. (ii) With E Min = 0.0 and a sample exactly at 0 V,
  that sample is not displayed; the same mirrored for E Max = 0.0.
- **A13:** the both-ends check stays where it is today (inside `if needs_topo:`), tests "both typed ends ==
  0.0", and its message drops "(per ldos_bias_sign)".
- **A14:** as written: only float64 equality e0 == 0.0 is "a sample at 0 V".
- **A15:** update the texts the plan does not name that become false, in the same terms (module docstring,
  gpu_chen_tunneling_factor's, _effective_decay_model's, _sweep's docstrings; stm_from_dft/__init__.py;
  the README intro and simulator row; run_DFT_STM.py's comments) — but NOT anything in `wrong_decays/` and
  NOT the README's `wrong_decays/` row (ruling 4). Two wordings: (1) the first-order relation reads "to
  first order in (E − V/2)/φ and V/φ" (Chen's chapter 2, Eq. 2.37: κ ≈ κ0(1 − ε/2φ̄), ε measured from
  the middle of the bias interval; the V correction starts at second order); (2) the new quadrature.py
  README row describes 7.2 correctly: the cubic through the two samples on each side of 0 V (through
  E−1, 0, E1, E2 above and E−2, E−1, 0, E1 below when a sample is at 0 V), its value at 0 V used by the
  unequal pair next to 0 V when the interval count is even, its exact integral over the interval next to
  0 V when the count is odd, Simpson pairs over the rest.
- **A16:** one helper `_bias_sample()` → (index, V_b), used at the Map Topo title, at the bias in
  `_update_all`, and wherever the bias end is needed; titles print V_b with 3 decimals; cache names use
  `str(V_b)` (section 10); the dead `bias_e` line in `_on_rel` is deleted.

## Group B, in force (approved as a block)

- **B1:** the parser reads E max, E min and NEDOS from line 6 and exposes `self.step` (float64, the plan's
  formula). The simulator keeps `self.e_step`, `self.k0`, `self.e0` (float64) and their float32 copies (for
  example `self.e_step32`, `self.e032`); the old local `step` (the launch chunk) and the span-relative
  `k0` are renamed so they do not collide.
- **B2:** the guard compares each printed first-block energy (absolute, as printed) with emin + step·i;
  strict `>` the A1 tolerance; the error names the largest difference and its file line (7 + i).
- **B3:** `_snap` is a module-level function `_snap(energies, value, kind)` returning a sample index, or a
  module constant `ZERO_V` for an end of exactly 0.0 (never None, which already means "not cached").
- **B4:** `s_feen_width`'s third use, the RUN box in `_make_controls`, becomes
  `controls.box(self.btn_run, self.chk)` (as before Feenstra).
- **B5:** the literal `'decay_chen'` is in `_effective_decay_model`: it becomes `'decay_tilt'` there (also
  the in-memory recompute key), with its docstring.
- **B6:** `cached_eg` in Map and Single Point is built on the CPU (`self.energies[idx]`).
- **B7:** kernel: `lo_i` / `hi_i` are dropped; the stencil comes from k0's span index and e0; `t_split` is
  the first requested sample above 0 V; `cur` is None when `want_cur` = 0.
- **B8:** GPU copies `cached_ld_up_gpu`, `cached_ld_dn_gpu`, `cached_spec_ldos_gpu`, set beside the NumPy
  ones, updated in `_on_rel` and reset wherever the NumPy names are; `cached_current`, `cached_current_E`,
  `cached_spec_current`, `cached_spec_current_E` initialised to None where `cached_iv` etc. were.
- **B9:** every scalar limit handed to a plot item or norm goes through `float()`; np.* calls on CuPy arrays
  in the display path become cp.*; `_combine_ldos_display` stays array-agnostic (sim_wrapper calls it with
  NumPy arrays).
- **B10:** GPU energies e0 + (i − k0)·step with the index difference cast to float32 first.
- **B11:** `integral_from_zero` returns the oriented integral at every sample of both sides (0 at a sample at
  0 V), built from explicit intervals (e0 + step and step above, −e0 below) as section 2 says, by
  cumulative Simpson pairs; the cubic and its exact pieces in closed Lagrange form.
- **B12:** `_converge_tip_height(z_map_gpu, grid_xy_gpu, samples, v_b, target_ldos, ...)`, with one private
  helper for the sample list (E−2..E2 plus 0..V_b) and the positive current, used at the five sites.
- **B13:** ValueError at load when NEDOS < 2 (in the parser, before the step's division) or when 0 V lies
  within two samples of a grid end (in the simulator, where k0 is found).
- **B14:** `current_from_ldos` keeps no input checks.
- **B15:** `with_current` in both the orbital and the collapsed branches.
- **B16:** comment trims: the comment at about 744-746 keeps its Feenstra clause; the one at about 1067 keeps
  its first sentence; the comments on removed attributes (about 784-786, 2249-2251, 2286) go.
- **B17:** the three CPU conversions of heights to float64 (contour, the two np.hstack) stay.
- **B18:** `_fold` as a literal in the three cache f-strings, right after the decay suffix.
- **B19:** every `simpson(` / `gpu_simpson(` left after part 1 (including any in part 1's drag cache) is
  found by grep and moved to `quadrature.simpson`.

## Later rulings (PROGRESS.md 3 to 7)

- **Citations (3):** section 11's Decay row cites Chen only, for the frozen-bias topography and the swept
  dI/dV. No Rejali in any new text.
- **wrong_decays/ (4):** nothing in `wrong_decays/` is written; the README's `wrong_decays/` row stays as it is.
- **Physics vs machinery (5, approved):** no new module. simulator.py opens with a block
  `# ==== THE DECAY MODEL (physics) ====` holding the physics: `KAPPA0_PER_SQRT_EV`, the weight and its V
  derivative (`gpu_chen_tunneling_factor`), the Decay-off weight, and the definitions of I(V) and dI/dV,
  with the citations (Chen's book for the frozen-bias topography and the swept dI/dV; Tersoff and Hamann for
  the constant tip DOS). Everything else sits under `# ==== NUMERICS AND GPU (no physics) ====`. Each def's
  docstring starts with "Physics:" or "Machinery:". The kernel's comment block says which of its lines
  transcribe the physics (the S and ∂S/∂V formulas, the two integrands) and which are machinery (threads,
  chunks, stencil, float32). feenstra.py and quadrature.py follow the same tags. The README describes the
  model (physics, citations) apart from how it is computed (grid, integration rule, GPU, float32, chunking,
  caches).
- **Checks (6):** agents read files and run `python -m py_compile` only.
- **Models (7):** every agent runs on Opus 5.5 or higher.

## Added sections (PART2_ANSWERS.md), done after the other sections

- **X1, the synthetic DOSCAR removed:** exactly as `PART2_ANSWERS.md` lists it (read_doscar_header moved to
  doscar_parser.py; `resolve_doscar_parser()` in _imports.py; `git rm` synthetic_doscar.py; sim_wrapper.py,
  decomp_window.py, mo_ldos_projection/__init__.py, run_stm_decomp.py and the READMEs as listed).
- **X2, procar_dos.py moved to PROCAR_tools/:** exactly as listed (`git mv`; its docstring; __init__.py;
  PROCAR_tools/README.md; the root README's PROCAR_tools row).
- **X3, the Map panels always in increasing energy:** exactly as listed (on an energy-marker release, a stable
  sort of the snapped Map energies applied to every per-map array and the markers; moved panels redrawn in
  place with `_redraw_map_slice`; no arrangement, rebuild or extra GPU work; the Map Norm as A4).

## Part 1 items still open with the owner (PART1_OPEN.md)

Do not act on them. Where part 2 must edit the same code, make part 2's change and leave the rest as it is.
