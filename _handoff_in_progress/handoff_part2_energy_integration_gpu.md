# Handoff, part 2: energy windows, the decay weight, the integration rule and the GPU in the simulator

Repository: `VASP_for_SPM`. This part changes the simulator package `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/` (`doscar_parser.py`, `simulator.py`, `feenstra.py` and a new `quadrature.py`), the package's `README.md`, and one spec line in the three `qtgui/widgets.py` files (section 8).

Do it after part 1 (`handoff_part1_qt_layout_text_drawing.md`), on top of part 1's result. Line numbers below refer to the owner's `simulator.py` in `disk/` before part 1 changed it. They are approximate: find the code by the names given.

## What is wrong today

- **DOS energies:** `SpinAwareDosParser` takes the energies from the DOSCAR's printed energy column, which VASP writes with 3 decimals. The real step is not a 3-decimal number (40/7999 = 0.005000625 eV in the owner's file), so the printed energies are off the grid by up to 0.0005 eV, and their steps read 0.005 or 0.006. Integrating on them costs about 1e-3 relative accuracy, which swamps Simpson's order (section 2).
- **The decay weight:** `gpu_chen_tunneling_factor` is Chen's Eq. 2.40, with the barrier tilt averaged (each state's κ taken at E − V/2, the middle of the bias window) and linearized (κ to first order in (E − V/2)/φ). It becomes the exact tilted barrier (section 6).
- **Window ends:** E Min and E Max become sample indices with `np.searchsorted` (the first sample at or above E Min, the last at or below E Max), so an end lands on the nearest sample inside the range, not the nearest sample.
- **Topography:** the current is integrated with `gpu_simpson` over the samples strictly inside the window. The pieces from 0 V to the first sample and from the last sample to the bias are left out. The frozen-bias weight uses the typed bias, not a DOS sample.
- **Swept LDOS:** the `chen_sweep` kernel and `feenstra.current_from_ldos` use the trapezoid rule, with a linearly interpolated 0 V value. So the topography and the LDOS use different rules.
- **`gpu_simpson`:** assumes evenly spaced samples. The Map Norm integrates over the Map energies (`map_e_targets`), which are not evenly spaced and are not the samples the maps are taken at.
- **The Feenstra current:** a second run after the displayed LDOS, over every sample from 0 V to the window: with Decay on a `current=True` kernel launch, with Decay off a second LDOS call. In Map it covers every sample from E Min to E Max, for the broadened I/V.
- **CPU work:** the Feenstra current (Decay off), `iv_ratio`, `divide` and the Norm integrals run on the CPU.
- **Precision:** every GPU array and the kernel are float64, which the owner's RTX 4080 runs at 1/64 of its float32 rate.

## 1. Rules for this job

1. Implement exactly what this plan says. If the code does not match the plan, or a step seems to need a change the plan does not name, stop and report it instead of choosing.
2. In the decay model, only the weight changes (section 6). Everything else stays:
   - the topography, converged on the current at the frozen bias
   - the displayed LDOS, dI/dV with the bias swept: the state entering at the window edge plus the V derivative of every state already in [0, V]
   - the Feenstra current I(V)
   - Decay off, exp(−2 κ0 d)
   - κ0 in the tip-height convergence step
3. Add no tests or checks, and run no window. The only check is `python -m py_compile` on every changed `.py` file.
4. `mo_ldos_projection/sim_wrapper.py` keeps working unchanged:
   - Keep the names it calls: `_update_all`, `_calculate_ldos_at_points_gpu`, `_combine_ldos_display`, `add_control_box`, `_toggle_run`, `fig`. Called without `with_current` (section 8), `_calculate_ldos_at_points_gpu` returns the same three elements as now.
   - The attributes it reads stay NumPy arrays (section 9).
5. At the end, list per file every function, constant and branch you changed, added or removed.

## 2. DOS energies rebuilt from DOSCAR line 6 (`doscar_parser.py`)

VASP's DOS grid is even, but its printed energy column is rounded to 3 decimals. `SpinAwareDosParser._parse` (about lines 70 to 96) therefore builds the energies from line 6 (E max, E min, NEDOS, E_F) instead of reading them from the column:

```
step     = (emax - emin) / (nedos - 1)
energies = emin + step * np.arange(nedos)        # float64, absolute
self.energies = energies - self.ef
```

- **The step formula:** it is (E max − E min)/(NEDOS − 1), with the first point at E min and the last at E max. The owner's file confirms it: E max 10, E min −30, NEDOS 8000. The first printed energy is −30.000 and the last is 10.000. Rounded to 3 decimals, the rebuilt grid equals all 8000 printed energies. The alternative /NEDOS ends at 9.995 and misses by up to a full step.
- **Precision:** float64. The step is carried to float64's full precision. Consecutive steps agree to about 7e-15 eV at these energies, the float64 limit.
- **E_F:** read from line 6 as now.
- **The printed column** is still read as now, but only for the guard below.
- **The guard:** after the rebuild, the parser compares every printed energy of the first block with its rebuilt value. If any differs by more than half a step, it raises `ValueError`, naming the largest difference and its line. Such a file's grid is not the even grid its line 6 describes.
- **The DOS columns** are read as now.
- **The synthetic DOSCAR** of `mo_ldos_projection/synthetic_doscar.py` writes `np.linspace(EMIN, EMAX, NEDOS)` with 5-decimal energies and its own NEDOS in line 6, so the same rebuild gives its grid and passes the guard. Nothing changes there.
- **Who sees it:** everything that uses the parser's `energies` gets the rebuilt grid: the simulator, `plot_dos_cursors_levels.py`, and the wrapper through the simulator.
- **What the GPU gets (section 9):** no energy array. The parser's `energies` stay float64 on the CPU, and `_snap`, the tie rules, the test for a sample exactly at 0 V and every index conversion use them. Once at load, the simulator takes from them:
  - `step` = (E max − E min)/(NEDOS − 1)
  - `k0`, the last sample at or below 0 V
  - `e0` = `energies[k0]`: 0 V lies −e0 above sample k0, and e0 is exactly 0.0 when a sample is at 0 V
  - `step` and `e0` are computed in float64 and rounded to float32 once. Computed in float32 arithmetic instead, −e0 would be off by 1.5e-3 relative in the owner's file (E_F − E min = 27.78 eV leaves float32 few digits after the point); rounded from float64 it is off by 1.6e-9.
  - On the GPU, sample i's energy is e0 + (i − k0)·step, computed in float32. Every energy in GPU arithmetic is formed this way from its sample index: the weight's E and V (section 6), V_b, the Map energies, and the energies of the Feenstra I/V and the Map Norm. The parser's float64 energies never enter GPU arithmetic. Every interval the integrals use is `step`, except, when no sample is at 0 V, the two touching 0 V: −e0 below and e0 + step above (section 7.2). The samples are therefore exactly equally spaced in every integral, which rounding each energy to float32 would not give (12 different float32 steps in −2.85 to 0.85 eV in the owner's file).

## 3. Energies snap to DOS samples

One helper, `_snap(value, kind)`, gives the DOS sample index for an energy. "Nearest" means the smallest |E_i − value| over all DOSCAR samples, on either side of the value. A tie means two samples at exactly equal distance (float64 equality).

| `kind` | used for | tie goes to |
|---|---|---|
| `'emax'` | E Max | the lower sample |
| `'emin'` | E Min | the higher sample |
| `'zero_side'` | each Map energy, and `topo_bias` | the sample closer to 0 |

- **An end of exactly 0.0:** E Min or E Max equal to 0.0 is exactly 0 V, not a sample. The integrals start there (section 7). `_snap` reports it as "0 V".
- **Where:** every conversion of E Min, E Max, a Map energy or `topo_bias` to a sample index goes through `_snap`. These are the `np.searchsorted(self.energies, ...)` calls at about lines 549 to 550, 945, 949, 2232, 2239 and 2901. The one at about line 2258 goes with the Map's span current (section 8).
  - The searches for the samples around 0 V (about lines 574 to 577) are not end conversions. They become the stencil of section 7. Those at about lines 965 to 966 go with `_feenstra_current` (section 8).

## 4. The bias

- **Which end is the bias:**
  - If E Min is 0.0, the bias end is E Max.
  - If E Max is 0.0, the bias end is E Min.
  - Otherwise `ldos_bias_sign` decides, as now.
  - If both are 0.0, the existing `ValueError` stays (about line 2176).
- **The snapped bias:** V_b is the energy of the bias end's snapped sample, by that end's rule. V_b replaces the typed bias `bias_e` (about lines 2164 and 2900) everywhere it is used:
  - the end of the topography window
  - the frozen-bias weight (`gpu_chen_tunneling_factor(E, V_b, ...)` at about lines 605 and 627)
  - the cached-bias comparisons (`cached_bias_energy_line`, `cached_bias_energy_map`)
  - the cache names (section 10)
  - the titles that state the bias (about line 2122)
- **The global topography:** `topo_bias` keeps its nonzero check (about lines 810 to 811). Its V_b is `_snap(topo_bias, 'zero_side')`, used in `run_interactive` (about lines 870 to 887) and `_recompute_global_topo` (about lines 1931 to 1951). `self.global_topo_bias` holds that V_b, because `sim_wrapper` reads it as the topography's bias.
- **Recompute triggers:** the cached-window comparisons in `_update_all` (`cached_emin`, `cached_emax`, about line 2170) compare the snapped ends, so a typed change that lands on the same samples recomputes nothing.

## 5. What is displayed

- **Spectra, heatmaps, Feenstra and Norm:** the displayed LDOS uses only DOSCAR samples, from the snapped E Min sample to the snapped E Max sample, both included. An end at exactly 0 V adds no point: the display starts at the first sample on the window's side of 0. No 0 V value is ever displayed.
  - `_display_indices` (about line 945) returns these indices.
  - The display ranges at about lines 2232 and 2239 use it.
  - The LDOS calls for the display pass them as `samples=`.
- **Map energies:** `map_e_targets` are formed as now (`np.linspace` of the typed E Min and E Max, about line 2231), and each is snapped with `'zero_side'`. `_map_sample_indices` (about line 949) returns those snapped indices, and everything a Map energy feeds uses the snapped sample's energy:
  - the energy maps, their titles (`E = ... eV`, about lines 2686 and 2763)
  - the Map Norm's `x` (about lines 2655 and 2733)
  - the energy-marker release in `_on_rel` (about lines 2896 to 2915, which today repeats `searchsorted`)
- **The energy marker on release:** it moves to its snapped sample's energy, so it marks the energy its map shows.

## 6. The decay weight: the exact tilted barrier

A state at E (eV from E_F) of an atom image at distance d (Å) from the tip, at bias V, with φ the LOCPOT potential at the tip point minus E_F (as now), carries:

```
S(E, V) = (4 k d / (3 V)) * [ a+^(3/2) - b+^(3/2) ],    a = phi - E + V,   b = phi - E,   x+ = max(x, 0)
w(E, V) = exp(-S(E, V))
```

- **The constant:** k is `KAPPA0_PER_SQRT_EV`, √(2 m_e e)/ħ in Å⁻¹ eV^-1/2, as now.
- **What S is:** the WKB exponent 2∫κ dz across a barrier rising linearly from φ − E at the sample to φ − E + V at the tip, with κ = 0 wherever the state lies above the barrier. A state above one end sees only the part of the barrier above it. Above both ends, w = 1.
- **Relation to Chen:** the barrier's mean height, φ − E + V/2, is the height Chen's Eq. 2.40 linearizes about. To first order in (E − V/2)/φ, S reduces to Chen's exponent.

**Evaluation** (float32 on the GPU, section 9; these forms keep full precision when V is small):
- **a > 0 and b > 0:**
  - S = (4kd/3)(a + √(ab) + b)/(√a + √b)
  - ∂S/∂V = (2kd/3)(√a + 2√b)/(√a + √b)²
  - At V = 0 these give 2kd√b and kd/(2√b), with no division by V.
- **Otherwise** (V cannot be 0 here):
  - S = (4kd/(3V))(a₊^(3/2) − b₊^(3/2))
  - ∂S/∂V = (4kd/3)[(3/2)√a₊/V − (a₊^(3/2) − b₊^(3/2))/V²]
  - With a ≤ 0 and b ≤ 0, both are 0.
- **The weight's derivative:** ∂w/∂V = −(∂S/∂V)·w.

**Where it goes:**
- `gpu_chen_tunneling_factor(E, V, phi, d, topo)` computes this w, and ∂w/∂V when `topo` is False. Its name, arguments and returns stay. Every caller already goes through it: the frozen-bias topography weights (about lines 605 and 627).
- The CUDA kernel computes the same S and ∂S/∂V with the same formulas.

**It is not separable:** Chen's linear exponent factored as w(E, V) = w(E, 0)·w(0, V)/w(0, 0), which let the kernel get the integral at every requested V from one walk outward from 0 V. The exact tilt has no such factorization, so the kernel integrates each requested V on its own (7.3).
- **The work:** per (point, atom, image), the work grows from about n to about n²/2 weight evaluations, for n requested samples on one side of 0 V.

**Cache suffix:** `_decay_chen` (in `_decay_cache_suffix`) becomes `_decay_tilt`, so caches of the linearized weight are never reused.

## 7. One integration rule (new module `stm_from_dft/quadrature.py`)

Everything that integrates in the package uses this module. The `chen_sweep` kernel implements the same formulas in CUDA, because it cannot call Python. All of it is CuPy, float32 (section 9).

### 7.1 `simpson(y, x=None, dx=1.0, axis=-1)`

Composite Simpson written out exactly as `scipy.integrate.simpson` computes it, transcribed from the installed SciPy's source:

- for any spacing of `x`, with the pair formula for unequal intervals
- with `dx` and no `x`: every interval equal to `dx`, as SciPy's `simpson(y, dx=...)`. The equally spaced DOS samples are integrated this way, with `dx=step`.
- an odd number of samples: pairs over every interval
- an even number of samples: pairs over the first N − 2 intervals, plus Cartwright's 3-point segment on the last interval
- 2 samples: the trapezoid, written out (no `cp.trapz`)
- 1 sample: 0

It replaces `gpu_simpson` (about lines 56 to 80).

### 7.2 `integral_from_zero(f, k0, step, e0, axis=-1)`: the oriented integral from 0 V to each sample of one side

Notation:
- E₁ < E₂ < ... are the DOS samples above 0 V.
- E₋₁ > E₋₂ > ... are the samples below 0 V.
- f is the integrand at those samples, given on consecutive DOS samples, with `k0` the position of sample k0 among them.
- The function takes `step` and `e0` (section 2), not an energy array. With no sample at 0 V: E₋₁ = e0, E₋₂ = e0 − step, E₁ = e0 + step, E₂ = e0 + 2·step. With a sample at 0 V (e0 == 0.0): that sample is 0 V and E±n = ±n·step.

**The 0 V value:** f(0) is the value at 0 of the cubic through (E₋₂, f₋₂), (E₋₁, f₋₁), (E₁, f₁), (E₂, f₂), written out in Lagrange form. Q₊ is the exact integral of that cubic from 0 to E₁, and Q₋ the exact integral from 0 to E₋₁.

**A sample E_k above 0, k intervals from 0 V:**
- k odd: Q₊ + `simpson` over (E₁, ..., E_k) with `dx=step`. That is k − 1 intervals, an even count, so `simpson` takes no Cartwright step.
- k even: `simpson` over (0, E₁, E₂) with `x` = (0, E₁, E₂), whose intervals are e0 + step and step, plus `simpson` over (E₂, ..., E_k) with `dx=step`. This is the same integral as `simpson` over (0, E₁, ..., E_k) with `x`, f(0) from the cubic.
- Below 0 the interval touching 0 V is −e0.

**Below 0:** the same, mirrored and oriented from 0. For E₋k the result is the integral from 0 down to E₋k, negative for a positive f, as the kernel's downward walk gives today.

**A sample exactly at 0 V** (E − E_F == 0.0):
- That sample is the 0 V point and nothing is inserted.
- For odd k, the [0, E₁] piece is the exact integral of the cubic through E₋₁, 0, E₁, E₂. Below 0 it is the cubic through E₋₂, E₋₁, 0, E₁.

**What lies outside the window:** the stencil samples there are used only for f(0) and the cubic's piece. Nothing outside [0, V] is integrated.

### 7.3 Where the rule is used

- **The topography current:**
  - Sites: `_converge_tip_height` (about line 283) and the setpoints (about lines 882, 1946, 2185, 2210).
  - The integrand is the weighted total LDOS at the samples E₋₂, E₋₁, E₁, E₂ and every sample from 0 to V_b, requested with `samples=`.
  - The current is the integral from 0 to V_b, taken positive as today (the integral from min(0, V_b) to max(0, V_b)).
  - `_converge_tip_height`'s window arguments become that sample list and V_b. It takes no emin/emax any more.
  - The start height, the largest-integrated-LDOS setpoint and the iteration stay as they are.
- **The `chen_sweep` kernel** (about lines 132 to 244): one thread per (point, atom) as now, looping over the images and the requested samples. One pass per requested V gives both dI/dV and the current. For each requested V, at the stencil samples E₋₂, E₋₁, E₁, E₂ and every sample from 0 to V:
  - w(E, V) and ∂w(E, V)/∂V, once per sample (section 6)
  - two integrands: ρ_o(E)·∂w(E, V)/∂V for each orbital o, and (Σ_o ρ_o(E))·w(E, V) for the current
  - both integrated from 0 V to V by 7.2, with the same cubic 0 V value and piece from that V's stencil values and the same Simpson pairs over the rest
  - dI/dV_o(V) = ρ_o(V)·w(V, V) + the first integral, added to `out` as now
  - the current is the second integral, added to a second output `cur`, (n_p, n_t, n_atoms): each thread adds only to its own atom's slot (summed over its images), and the caller sums `cur` over atoms. No atomics.
  - The kernel's `current` argument becomes `want_cur`: 1, the current is computed and written to `cur`; 0, it is skipped and `cur` is not touched (the NCL |m| channel, and calls without `with_current`).
  - It replaces the separable one-walk sum (the `g = w(0, V)/w(0, 0)` factor, the running trapezoid `C` and the linear 0 V value).
  - Nothing per sample is stored; the output holds only the requested samples.
  - The span (`span_lo` to `span_hi`, about lines 574 to 577) covers E₋₂ to E₂ as well as the requested samples. `lo_i` / `hi_i` become the stencil's indices.
  - The per-launch chunk (`step`, about line 611) is sized by the (requested V, sample) pairs per point instead of the requested samples per point.
  - It receives `step`, `e0` and the span index of k0 instead of the span energy array `E`, and computes a sample's energy as e0 + (j − k0)·step (section 2).
  - The comment block above the kernel describes this.
- **`feenstra.current_from_ldos(ldos, k0, step, e0)`:** I(V) at every sample it is given is `integral_from_zero`. It takes the grid of section 2, not an energy array. Its one caller is the Decay-off current of section 8, whose samples include E₋₂ to E₂.
- **The Norm integrals** (about lines 2344, 2354, 2362, 2449, 2655, 2733): `quadrature.simpson` on the GPU over the displayed samples. In Map, `x` is the GPU energies (section 2) of the snapped Map samples. `from scipy.integrate import simpson` (about line 41) is removed.
- **What goes:** `gpu_simpson` and `_integral_from_zero` (about lines 104 to 123) are removed. Nothing else integrates.

## 8. Feenstra: the current from the display's own pass, and no I/V broadening

The I/V broadening (`broadened`, `width (V)`) is removed at the owner's decision. I/V is the plain I(V)/V at each displayed sample, so the current is needed only at the samples the display shows. The call that computes the displayed LDOS returns the current at the same samples, and there is no separate current run.

**`_calculate_ldos_at_points_gpu`:**
- The `current` keyword is removed and a keyword `with_current=False` is added. It is used only with `v_bias=None` and `topo_only=False`.
- With `with_current=True` the call returns a fourth element: the total current I at the requested samples, in the order requested, CuPy (n_pts, n_samples) float32. The channels are combined as `_combine_ldos_total` combines them: channel 1 for NCL and unpolarized, channel 1 plus channel 2 for collinear polarized. The NCL |m| channel computes no current.
- Without it, the call returns the three elements of today, so `sim_wrapper.py` and the topography callers are unchanged.
- **Decay on:** the kernel's one pass (7.3) gives dI/dV and the current together.
- **Decay off:** the weight exp(−2 κ0 d) does not depend on V. The current is the weighted total LDOS (the channel's collapsed DOS with the same weights as the displayed LDOS) at the stencil samples E₋₂, E₋₁, E₁, E₂ and every sample from 0 V out to the farthest requested sample on each side, integrated by `feenstra.current_from_ldos` (7.2) and taken at the requested samples. It is computed inside the same call, from the same distances and φ.

**The calls that pass `with_current=True`** are the display runs, and each stores its current with the GPU energies (section 2) of its samples (CuPy, float32):
- Line / Path (about line 2226) and the Map grid (about line 2235): `self.cached_current`, `self.cached_current_E`. In Map the samples are the snapped Map energies, one column per map.
- The marker spectra (about line 2277): `self.cached_spec_current`, `self.cached_spec_current_E`.
- The energy-marker release in `_on_rel` (about line 2905): its current goes into column `idx` of `self.cached_current`, and its sample's GPU energy into entry `idx` of `self.cached_current_E`, as its LDOS goes into column `idx` of `cached_ld_up` / `cached_ld_dn`.

**`_feenstra_iv(which)`:** `which` is `'grid'` or `'spec'`. It returns `feenstra.iv_ratio(E, I)` of that stored current and its energies. Its `need` argument, the span logic and the `rec` dict go. The call sites `self._feenstra_iv('grid', need)` (about lines 2298 to 2299 and 2717) and `self._feenstra_iv('spec', self._spec_iv_need)` (about line 2308) become `self._feenstra_iv('grid')` and `self._feenstra_iv('spec')`.

**Map:** the current is computed at the Map energies only. Today's Map current over every sample from E Min to E Max (about lines 2256 to 2259) served the broadening and goes. A dragged energy marker's map is computed on release as now, its current in the same call.

**Removed:**
- `_feenstra_current` (about lines 951 to 976).
- `cached_iv`, `cached_spec_iv`, `_iv_pts`, `_iv_need`, `_spec_iv_pts`, `_spec_iv_need`: set up at about lines 787 to 788, set at about lines 2252 to 2261 and 2287 to 2289. The stored currents replace them.
- The broadening in `simulator.py`:
  - `feenstra_broadened` and `feenstra_width` (about lines 748 to 749)
  - the `('broadened', 'feenstra_broadened')` toggle (about line 1071)
  - the toggles' `indents=` argument (about line 1079), since no toggle is indented any more
  - the `width (V)` box `s_feen_width` (about lines 1081 to 1084) and its two lines in `_on_ui_change` (about lines 2019 to 2020)
  - the comments about them (about lines 744 to 746, 1067 to 1068 and 2017 to 2018)
- In `feenstra.py`, `iv_ratio` keeps only its plain branch. Its `broadening` argument and the exponential mean go. A sample exactly at 0 V still gets the linear interpolation of its neighbours' I/V, among the samples it is given.
- The `'width (V)'` spec line that part 1 brought in with the owner's `qtgui/widgets.py`: removed from the root `qtgui/widgets.py` and from both package copies (`stm_from_dft/qtgui/widgets.py`, `MOPmatcher_mem/matcher/qtgui/widgets.py`), so the two copies stay byte-identical. Nothing else uses it.

## 9. Everything on the GPU

- **`feenstra.py`:**
  - On CuPy: `current_from_ldos` (7.2), `iv_ratio` (the plain I/V of section 8, `cp.interp` for a sample exactly at 0 V), `divide`.
  - Its docstring no longer says "Pure numpy, no GPU".
  - It describes the integration as 7.2 (no trapezoid) and the plain I/V. The paragraph on the port of the sm4_Browser's `iv_broadened` goes; the Feenstra, Stroscio and Fein reference stays.
- **The current and I/V:** the stored currents (section 8), `iv_ratio` and `divide` stay on the GPU.
- **The display arrays stay CuPy until drawn:** the LDOS caches, the combined display, Norm, the Feenstra division and part 1's drag cache. They become NumPy with `cp.asnumpy` only at the call that hands them to a plot item (`set_array`, `set_data`, `set_offsets`, a new mesh or contour).
- **Exception, for `sim_wrapper.py`:** it reads `cached_eg`, `cached_ld_up`, `cached_ld_dn`, `cached_spec_ldos` and `global_z_map` and uses NumPy and SciPy on them. These stay NumPy arrays, filled once per data change as now. The GPU copies the display works from have their own names.
- **Precision: float32 on the GPU** (the owner's decision):
  - **Each value keeps the precision it is made in:** what the GPU computes is float32 and stays float32, its NumPy copies included. What the parsers and the CPU side make is float64 and stays float64 on the CPU. The inputs the GPU computes from are rounded to float32 once, when they go to the GPU: the DOS, the LOCPOT, the atom and tip positions, the lattice vectors, constants such as `KAPPA0_PER_SQRT_EV` and E_F, and `step` and `e0` (section 2). Nothing the GPU computes is converted afterwards.
  - **Every CuPy array is float32:** the DOS arrays and their collapsed forms, the LOCPOT, `inv_lv_gpu`, `locpot_dims_gpu`, `periodic_coord_gpu`, `map_mat_gpu`, the tip positions and grids, the z maps, the distances, φ, the weights, the LDOS outputs, the currents, and the Feenstra and Norm arrays. Every `cp.float64` in `simulator.py` becomes `cp.float32` (about lines 364 to 636, 868 to 875, 1941, 2180 to 2206 and 2276; the one at about line 969 goes with `_feenstra_current`).
  - **`energies_gpu`** (about line 364) goes, with the span array `E_span_gpu` (about line 583): the GPU gets the grid of section 2. The third value `_calculate_ldos_at_points_gpu` returns stays a CuPy array of the samples' energies, because `sim_wrapper.py` calls `.get()` on it: `cp.asarray(self.energies[idx_np])`, the parser's float64 values, not converted. Nothing on the GPU computes with it; it only goes back to the CPU.
  - **The kernel:** `float` for every value, pointer and output (`out`, `cur`), the float math functions (`expf`, `sqrtf`), and the `f` suffix on every literal (`2.0f`, `0.5f`, `4.0f / 3.0f`). An unsuffixed literal is a double in C and makes the expression around it double precision. `KAPPA0_PER_SQRT_EV`, `step` and `e0` are passed as `np.float32`.
  - **Scalars in CuPy arithmetic:** Python floats or `np.float32`, never `np.float64` scalars. Under NumPy's promotion rules (NEP 50), which CuPy follows, a float32 array combined with a Python float stays float32, but combined with a NumPy float64 scalar it becomes float64. No element of `self.energies` enters GPU arithmetic (section 2); any other NumPy float64 scalar that does goes through `float()` first. A CuPy or cupyx call that would compute in float64 from float32 input is given float32 output instead (its `dtype=` or `output=` argument), so the value is made in float32.
  - **`cp.dot`** runs in plain float32. TF32 stays off (`CUPY_TF32` is not set).
  - **The CPU side stays float64:** the parser and its guard, `self.energies`, `step` and `e0` before rounding (section 2), `_snap` and every index conversion.
  - **What `sim_wrapper.py` reads:** `cached_ld_up`, `cached_ld_dn`, `cached_spec_ldos` and `global_z_map` are NumPy float32, `cp.asnumpy` of the float32 GPU arrays with no conversion. `cached_eg` is the parser's float64 energies at the displayed samples. `sim_wrapper.py` converts to float64 where it needs it (about lines 356 and 444), and `DecompWindow` (about lines 312 to 315) and `TopoDecompWindow` (about lines 160 to 165) convert their input, so nothing there changes.

## 10. Cache names

The `global_topo_*` and `ldos_topo_*` names (about lines 871, 1934, 2199) carry V_b, the snapped sample energy, as `{V_b}` (Python's `str` of the float). The tag `_fold` is added after the decay suffix. `_precision_cache_suffix` returns `_f32` instead of `_f64`, and the `global_topo_*` / `ldos_topo_*` height maps are saved as float32. Caches from the old rule or the float64 build are therefore never reused.

## 11. README and docstrings

In `LDOS_spread_decomp/stm_from_dft_opt/README.md`:

- **Package table, `doscar_parser.py` row:** the energies are rebuilt from line 6 as an even grid, E min + i (E max − E min)/(NEDOS − 1), minus E_F. The printed column is only checked against it, within half a step.

- **`Decay` row (about line 51):**
  - On: the exact tilted barrier of section 6, the WKB exponent across the barrier rising linearly from φ − E to φ − E + V. Cite R. Rejali et al., Phys. Rev. B 107, 035406 (2023), as `wrong_decays/tilted_barrier_decay.py` does, and Chen for the frozen-bias topography and the swept dI/dV. Its first-order expansion in (E − V/2)/φ is Chen's Eq. 2.40.
  - The topography's window is [0, V_b] from exactly 0 V, with V_b the bias end's snapped DOS sample, and the weight at V_b.
  - All integrals use the rule of 7.2.
  - The cache suffixes are `_decay_tilt` (on) or `_decay_k0` (off), plus `_fold` and `_f32`.
- **Toggles row (about line 50):** `broadened` (under Feenstra) is removed from the list.
- **`Feenstra` row (about line 52):**
  - "trapezoid rule" becomes the rule of 7.2.
  - "with Decay on, Chen's Eq. 2.40 at each V (the window [0, V], every state weighted at that frozen bias, as the topography's current)" becomes: with Decay on, the current of the exact tilted barrier at each V, the integral from 0 to V of ρ(E) w(E, V), as the topography's current at its bias. It comes from the same pass as the displayed dI/dV.
  - I/V is the plain I(V)/V.
  - The current, I/V and the division are computed on the GPU.
- **The `broadened`, `width (V)` row (about line 53):** removed.
- **Bottom-strip row (about line 57):** one sentence on E Min / E Max snapping and the tie rules, including that an end of 0 is exactly 0 V and makes the other end the bias.
- **Package table:**
  - `simulator.py` row: "Simpson integration" becomes "the integration rule of `quadrature.py`", and it states that the GPU computes in float32 from the float64 DOS grid (one step, exactly equal spacing).
  - `feenstra.py` row: "(numpy only)" becomes "(CuPy)", and `iv_ratio`'s entry becomes "the plain I/V; a sample exactly at 0 V given its neighbours' I/V".
  - A new `quadrature.py` row: `simpson` (SciPy's composite Simpson, written out for the GPU) and `integral_from_zero` (from 0 V: the cubic through the two samples on each side of 0 V, its exact integral over the interval next to 0 V when the interval count is odd, Simpson pairs over the rest).
- **Outputs line (about line 79):** the cache names carry the snapped bias, `_fold` and `_f32`.

In `simulator.py`: the docstrings of `_calculate_ldos_at_points_gpu`, `_converge_tip_height`, `_feenstra_iv`, `_display_indices`, `_map_sample_indices` and the comment block above the kernel describe the new behavior.

## 12. Finish

- `python -m py_compile` every changed `.py` file. Run nothing else.
- Commit on the same branch as part 1. The owner tests on his own machine.
- Report the per-file list of every function, constant and branch changed, added or removed (rule 5).
