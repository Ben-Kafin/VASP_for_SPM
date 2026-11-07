# Part 2: questions for the owner before implementation

A read-only pre-flight mapped every part-2 bullet onto your `disk/` simulator.py (and doscar_parser.py,
feenstra.py, sim_wrapper.py, synthetic_doscar.py). Below are the places where the code does not match
the plan, or the plan leaves a choice open. Line numbers are your `disk/` simulator.py before part 1.
Each item gives a recommendation (**R:**). Answer per item, or "accept all R except ...".

Sources the pre-flight read: SciPy 1.18.1 `_quadrature.py` (the newest release; SciPy is not installed
here and your version is unknown), CuPy 13.6.0 and 14.2.0 source, NumPy 1.26.4 / 2.2.6 run in a
scratch folder. The maths in group A was checked symbolically (sympy) and by finite differences; I
re-derived A6 and A15's expansion myself.

## A. Behaviour or physics: your ruling needed

**A1. The guard's half-step tolerance (2).** The printed column is rounded to 3 decimals, so it can
be off by up to 0.0005 eV. Half a step is below that whenever step < 1 meV (NEDOS > 40001 over 40 eV),
so the guard would reject a genuine DOSCAR whose grid is exactly the even grid of line 6. Your file
(max diff 0.00049994 vs half step 0.0025) and the synthetic file (5 decimals) pass either way.
Options: (a) as written, half a step; (b) max(half a step, 0.0005 + a little).
**R:** your call; (b) if you ever run fine NEDOS.

**A2. V_b at or across 0 V (3, 4).** Snapping can give V_b = 0.0 exactly (a sample at 0 V and
|typed bias| < step/2: the current and the setpoint are 0, rat = 1e-20/0) or flip its sign
(topo_bias −0.001 snapping to +0.000969). With E Min = 0.0 and E Max snapping below 0 the display is
empty. The nonzero checks (810-811, 2176) test only typed values.
Options: (a) accept as written; (b) raise ValueError when V_b == 0.0 or its sign differs from the
typed value; (c) snap the bias end to the nearest sample on its typed side of 0.
**R:** (b).

**A3. A Map energy of exactly 0.0, and 'zero_side' at the window's ends (3, 5).** The 0 V rule covers
only E Min / E Max. Map energies of exactly 0.0 are common (an end of 0.0 is a linspace end point;
the middle of a symmetric window with odd E Pts; the energy-marker drag clipped at a 0.0 bound,
2888). The nearest sample may lie on the other side of 0, outside the displayed range (window
[−0.02, 0]: 0.0 snaps to +0.000969 while the display ends at −0.004032). Also, at an exact tie
'zero_side' can put an end map one sample outside the snapped E Min / E Max (needs a float64-exact
tie: practically never).
Options: (a) nearest, as written, even outside the window; (b) the nearest sample on the window's side;
(c) a Map energy of 0.0 goes to the display's end sample.
**R:** (b).

**A4. Repeated Map samples in the Map Norm (5, 7.3).** When (E Max − E Min)/(E Pts − 1) < step,
several Map energies snap to the same sample: zero-width intervals in the Map Norm's x. SciPy's
guarded formula then returns a finite but wrong value (2/3 of the true one); an unguarded port gives
inf/NaN. Today's x (the typed linspace) never repeats. Window [−0.02, 0] with E Pts 20 gives 5
distinct samples.
Options: (a) keep repeats, SciPy's guards keep it finite; (b) de-duplicate before the Norm;
(c) raise / refuse more E Pts than samples in the window.
**R:** your call; (b) if the maps themselves may repeat, (c) if not.

**A5. Map recompute trigger (4, 5).** The Map energies are re-formed from the typed ends on every
full refresh (2231, 2434-2435, 2645-2646), and every box change is a full refresh (2077). A typed end
can move while its snapped sample stays (E Min −2.500 → −2.499 both on sample 5056), yet interior
Map samples move (maps 3 and 9 of 20). Comparing only the snapped ends (2170) leaves needs_ldos False
and the maps stale while their titles, Norm x and markers follow the new energies.
Options: (a) also compare the snapped Map indices, store the snapped energies in map_e_targets, draw
the energy markers at snapped energies; (b) keep typed targets and snap at each use; (c) accept.
**R:** (a).

**A6. The weight at V = 0 with the state above the barrier (6).** 'Otherwise (V cannot be 0 here)' is
not quite right: at V = 0, a = b, so a state with E ≥ φ has a = b ≤ 0 and the written S is 0/0. It is
reached at a requested sample exactly at 0 V, or V_b = 0, where φ ≤ E for a stencil E (|E| ≤ 2 step),
i.e. φ below about 0.01 eV: rare.
**R:** three branches: a > 0 and b > 0 → form 1; a ≤ 0 and b ≤ 0 → S = ∂S/∂V = 0; otherwise → form 2.

**A7. Chunking the display sweep (7.3).** The chunk at 611 is in the preserve_orbitals=False branch,
whose only caller is `_feenstra_current` (removed in 8), so after part 2 nothing reaches it. Every
display sweep (preserve_orbitals=True, 630-638) launches all tip points at once, and with the exact
tilt its work becomes about n²/2 weights per (point, atom, image): ~1e10-1e11 for a 700-sample Line
window or a 4096-point Map with ~200 atoms, in one launch (a long launch can hit the Windows GPU
watchdog).
Options: (a) chunk only line 611, as written (dead code); (b) also chunk the display sweep by
(V, sample) pairs per point.
**R:** (b).

**A8. Part 1's drag cache vs part 2's "CuPy until drawn" (part 1 6.5, part 2 9).** Part 1 says no drag
does GPU work; part 2 keeps display arrays (the drag cache named among them) on the GPU until
`set_data`. Each motion would then copy device → host, and any slice or sum at draw time launches a
kernel.
Options: (a) CuPy, fully reduced when built, so a drag only copies contiguous rows; (b) a NumPy drag
cache built once per data change (no GPU work during a drag).
**R:** (b).

**A9. global_topo_bias round trip through sim_wrapper (4).** sim_wrapper reads `global_topo_bias` as
the bias (as the plan says) but also feeds it back as the next instance's typed `topo_bias` on a σ
commit (capture_state, b051547 line 289). Below σ = 0.03 the synthetic grid changes with σ, so the
re-snap drifts (typed 0.2 → V_b 0.202496 at σ 0.2 → 0.203046 at σ 0.0255, where snapping 0.2 gives
0.198796), cache names included.
Options: (a) accept; (b) the simulator keeps the typed value too (e.g. `topo_bias_typed`) and
sim_wrapper passes that back (changes sim_wrapper, which the plan keeps unchanged).
**R:** your call.

**A10. Current on every display run, or only with Feenstra on (8).** The plan reads as always. Only
"always" keeps "Feenstra: display only, no LDOS recompute" (toggling it does not recompute). It costs
the second integrand in every sweep.
**R:** always, as written.

**A11. SciPy version, signature and the CuPy gaps (7.1, 9).** (i) Transcribe SciPy 1.18.1's `simpson`
(it matches every 7.1 bullet); SciPy < 1.11 integrated even N differently. Its signature is
`simpson(y, x=None, *, dx=1.0, axis=-1)` (dx keyword-only); the plan writes `simpson(y, x=None,
dx=1.0, axis=-1)`. (ii) SciPy guards every division (`apply_where`, fill 0); CuPy has no
apply_where: keep the guards as `cp.where` with a safe division. (iii) `cp.interp` has no dtype/out
and computes in float64; `cp.divide(..., where=)` is not accepted by CuPy (only out/dtype/casting).
For iv_ratio's 0 V sample a float32 two-neighbour interpolation (cp.searchsorted / cp.where) is
needed; for `divide`, `cp.where(d != 0, a / cp.where(d != 0, d, 1), 0)`.
**R:** 1.18.1, the plan's positional signature, guards kept, the two float32 forms above. Tell me
your SciPy version if it is older than 1.11.

**A12. A '0 V' end outside the display (3, 5).** (i) sim_wrapper's topography send is the only
remaining caller of the emin/emax path, and it always passes one end of exactly 0.0
(sorted([0.0, global_topo_bias])); the plan gives that path no 0 V rule. Today a sample exactly at
0 V is included. (ii) With E Min = 0.0 and a sample exactly at 0 V (e0 == 0), "no 0 V value is ever
displayed" excludes it; the plan spells out only E Min = 0.0.
**R:** (i) the display rule (start at the first sample on the window's side of 0);
(ii) exclude, and mirror for E Max = 0.0.

**A13. Both ends 0.0 (4).** Today's check (`if bias_e == 0`, 2175-2177) sits inside `if needs_topo:`,
which is never true in Single Point, so E Min = E Max = 0 does not raise there today.
**R:** keep it where it is, test "both typed ends == 0.0", message without "(per ldos_bias_sign)".

**A14. 0 V very close to, but not on, a sample (2, 7.2).** The even-k pair over (0, E1, E2) is
ill-conditioned as the interval touching 0 V shrinks (float32 relative error 1.6e-5 at 1e-3 of a step,
8e-4 at 1e-5). The exact branch fires only on float64 equality, and the float64 rebuild usually misses
an exact 0 by ~1e-15 even when E_F is on the grid. Practically: 0 V within ~5e-6 eV of a sample.
Options: (a) as written; (b) a tolerance treating |e0| < c·step as on-sample.
**R:** (a).

**A15. Texts the plan does not name that become false (11).** simulator.py: the module docstring (Chen's
element, Eq. 2.40), gpu_chen_tunneling_factor's docstring, _effective_decay_model's, _sweep's;
stm_from_dft/__init__.py says wrong_decays' factor "is not the model used by simulator.py" — after
section 6 the new S/d is that same tilted-barrier expression (with signed V and the window of the new
model); the README intro, simulator row and wrong_decays row say "Chen's matrix element";
run_DFT_STM.py's comments. Two wordings in the plan's own README text: (1) "Its first-order expansion
in (E − V/2)/φ is Chen's Eq. 2.40": expanding the plan's S gives 2kd√m·(1 − V²/(96 m²) + …) with
m = φ − E + V/2, so it reduces to the README's Chen form only to first order in (E − V/2)/φ **and**
V/φ (I checked this against the README's own statement of the Chen form, not the book). (2) The new
quadrature.py row: with a sample at 0 V the cubic is through E−1, 0, E1, E2 (not two samples each
side), and the cubic's f(0) is also used for even counts.
**R:** update the unnamed texts in the same terms, and adjust the two wordings; or name which to leave.

**A16. The bias in titles (4).** The Map Topo title (2122) repeats the sign rule inline on the typed
boxes, inside the full-refresh block that runs before bias_e (2164) and even while not running. The
Global Topo title (2119) changes through global_topo_bias. The bias line 2900 in _on_rel is dead
(assigned, never read).
**R:** one helper `_bias_sample()` → (index, V_b) used at both places; titles print V_b with 3
decimals (cache names use str(V_b) as section 10 says); delete the dead line.

## B. Mechanical gaps where the plan's intent is clear: my default (approve as a block)

- **B1 (2)** `step`, `k0`, `e0`: the parser reads E max / E min / NEDOS from line 6 and exposes
  `self.step` (float64, the plan's formula); the simulator keeps `self.e_step`, `self.k0`, `self.e0`
  (float64) and their float32 copies. (The local `step` launch chunk and the span-relative `k0` are
  renamed so they do not collide.)
- **B2 (2)** Guard: absolute comparison of the printed first-block column with emin + step·i, strict
  "> half a step", the error names the largest difference and its file line (7 + i).
- **B3 (3)** `_snap` is a module function `_snap(energies, value, kind)` returning a sample index, or a
  module constant `ZERO_V` for an end of exactly 0.0 (never None, which already means "not cached").
- **B4 (8)** `s_feen_width` has a third use the plan does not name, in the RUN box
  (`controls.box(self.btn_run, controls.column(self.chk, self.s_feen_width))`, 1157). It becomes
  `controls.box(self.btn_run, self.chk)`, the form before Feenstra.
- **B5 (6, 10)** The literal `'decay_chen'` is in `_effective_decay_model`, not `_decay_cache_suffix`:
  rename it there (also the in-memory recompute key, harmlessly) and its docstring.
- **B6 (5, 9)** `cached_eg` in Map / Single Point is built on the CPU (`self.energies[idx]`), not by a
  float64 CuPy round trip (2232, 2239), so every CuPy array is float32.
- **B7 (7.3)** Kernel: `lo_i`/`hi_i` are dropped; the stencil comes from k0's span index and e0;
  `t_split` = the first requested sample above 0 V; `cur` = None when want_cur = 0.
- **B8 (8, 9)** GPU copies: `cached_ld_up_gpu`, `cached_ld_dn_gpu`, `cached_spec_ldos_gpu`, set beside
  the NumPy ones, updated in `_on_rel` and reset wherever the NumPy names are; the four current
  attributes initialised to None where `cached_iv` etc. were.
- **B9 (9)** Every scalar limit handed to a plot item or norm goes through `float()`; np.* calls on
  CuPy arrays become cp.* in the display path; `_combine_ldos_display` stays array-agnostic (sim_wrapper
  calls it with NumPy arrays).
- **B10 (2, 6)** Energies on the GPU: e0 + (i − k0)·step with the index difference cast to float32
  first (an int64 index times np.float32 gives float64 in NumPy 1.x and 2.x).
- **B11 (7.2)** `integral_from_zero` returns the oriented integral at every sample of both sides
  (0 at a 0 V sample), built from explicit intervals (e0 + step, step; −e0 below) as section 2 says,
  by cumulative Simpson pairs; the cubic and its exact pieces in closed Lagrange form.
- **B12 (7.3)** `_converge_tip_height(z_map_gpu, grid_xy_gpu, samples, v_b, target_ldos, ...)`, with
  one private helper for the sample list (E−2..E2 plus 0..V_b) and the positive current, used at the
  five sites.
- **B13 (2, 7.2)** ValueError at load when NEDOS < 2 or 0 V lies within two samples of a grid end.
- **B14 (7.3)** `current_from_ldos` keeps no input checks (its energies argument is gone).
- **B15 (8)** `with_current` implemented in both the orbital and the collapsed branches.
- **B16 (8)** Comment trims: 744-746 keeps its Feenstra clause, 1067 keeps its first sentence; the
  comments on removed attributes (784-786, 2249-2251, 2286) go.
- **B17 (9)** The three CPU conversions of heights to float64 (1582, 2227, 2276) stay (lossless or
  CPU only).
- **B18 (10)** `_fold` as a literal in the three cache f-strings, after the decay suffix.
- **B19 (7.3)** After part 1, every `simpson(` / `gpu_simpson(` (including any in part 1's drag
  cache) is found by grep and moved to `quadrature.simpson`.

## C. For information only (no change proposed, outside part 2's scope)

- **C1** synthetic_doscar.py writes line 6 as `{emax:16.8f}{emin:16.8f}{nedos:5d}...`; for NEDOS ≥ 10000
  E min and NEDOS fuse ("-30.0000000010001") and the parser (today's and the rebuilt one) fails. Over
  40 eV that happens for σ ≤ 0.024; SIGMA_MIN is 0.01. Pre-existing.
- **C2** sim_wrapper's docstrings say the topography send repeats the convergence's last evaluation;
  after 7.3 it does not (different samples and rule). Its own Norm keeps SciPy simpson in float64.
- **C3** The plan's "NEP 50, which CuPy follows" holds for CuPy ≥ 14; CuPy 13 types scalars by value.
  The prescription (Python floats or np.float32) gives float32 on both. The RawKernel compiles with
  fused multiply-add on, so kernel and CuPy energies can differ by 1 ulp; harmless.
