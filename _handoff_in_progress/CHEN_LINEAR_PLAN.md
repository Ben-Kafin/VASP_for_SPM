# Linearized decay as a sub-toggle of Decay (temporary plan; deleted when done)

Approved by the owner, 2026-11-07 ("approved. Begin."). This file governs the implementation.

**Scope (owner, 2026-11-07): "it only needs to do anything to the stm_from_dft_opt directory, not the full VASP_for_SPM
suite".** Only files under `LDOS_spread_decomp/stm_from_dft_opt/` change (and this temporary plan file). Not changed:
`mo_ldos_decomp/` (run_stm_decomp.py, sim_wrapper.py), the root `qtgui/`, the matcher's `qtgui/` copy.

## Sources

- The owner's file for the linearized decay: the repository's simulator.py at b051547 (the upload is byte-identical):
  `gpu_chen_tunneling_factor` (the linearized weight and dw/dV, lines 88-101), its separable one-walk kernel
  `_CHEN_SWEEP_SRC` (comment 125-136, source 137-245), `_chen_sweep_kernel` (246-254).
- The owner chose (b): that weight and that one-walk design, run under the package's current numerics.
- Both Decay-on models are Chen's decay; "exact" and "linearized" name them throughout; no name that tells them
  apart contains "chen".

## 1. The exact decay: same computation, new names

The weight, the per-V sweep kernel, the frozen-bias topography and the swept dI/dV compute exactly as now. Renames:
`gpu_chen_tunneling_factor` -> `gpu_exact_factor`; `_CHEN_SWEEP_SRC` / `chen_sweep` / `_chen_sweep_kernel` (and its
build cache) -> `_EXACT_SWEEP_SRC` / `exact_sweep` / `_exact_sweep_kernel`; the CUDA device function `tilt_weight` ->
`exact_weight`; the model name `'decay_tilt'` -> `'decay_exact'` (the cache suffix with it: the existing `_decay_tilt`
topography caches are not found and are computed once more under `_decay_exact`).

## 2. The linearized decay, new

- Physics, from the owner's file: w = exp(-2 kappa0 d + kappa0 d (E - V/2) / phi), dw/dV = -(kappa0 d / (2 phi)) w,
  in the physics block as `gpu_linearized_factor`; its single-walk kernel, w(E,V) = w(E,0) w(0,V) / w(0,0), as
  `_LINEARIZED_SWEEP_SRC` / `linearized_sweep` / `_linearized_sweep_kernel`: one pass from 0 V gives dI/dV(V) = sum
  over the atom's images of rho(V) w(V,V) + w(0,V)/w(0,0) integral_0^V rho(E) dw(E,0)/dV dE, and the current
  I(V) = sum w(0,V)/w(0,0) integral_0^V rho(E) w(E,0) dE, at every requested V.
- Numerics, the package's: float32 with energies formed from the index (e0 + (i - k0) step); the integration rule of
  `quadrature.integral_from_zero` (the stencil cubic around 0 V, the unequal pair, the Simpson pair chains; the steps
  taken from the exact kernel's CUDA, which already implements the rule per V); the snapped V_b.
- The same as the exact option: phi the local barrier per tip point; the frozen-bias topography at V_b, integrated
  from 0 V; the swept LDOS; the Feenstra current from the same pass; the console line 'x/n spectra calculated'.
- Model name `'decay_linearized'`, its own caches. No guard on phi (the owner's file has none).

## 3. The control

Two indented rows under Decay in the toggle list: 'exact' and 'linearized'. Clicking one turns it on and the other
off; exactly one is always on (clicking the one already on leaves it on); 'exact' at start. Both are ignored while
Decay is off (a change then changes no model, so nothing is recomputed). run_interactive keyword
`decay_linearized=False`. show_decay_toggle=False hides both rows with Decay. The info panel lists both rows.

## 4. Everywhere the model is used or compared

`_effective_decay_model` gives 'decay_exact' / 'decay_linearized' / 'decay_k0' (the cache suffixes follow);
`_on_ui_change` compares models, not the Decay bool (the same drops on a model change); the topography calls
(`_topo_current`, `_converge_tip_height`, their callers), the LDOS calls and the energy-marker drag (Map's kept model)
pass the model; sim_wrapper's Decay-off rule for the decomposition unchanged.

## 5. '0-clamp'

The 'sim toggles' line and its comment in the simulator package's `qtgui/widgets.py` become the simulator's real
rows, 'exact' and 'linearized' indented under 'Decay'. Under the scope rule the root and matcher copies keep their old
line, so the matcher's qtgui copy differs from the package copy by that one line. The `Claude outputs/` folder
is already removed (774172a).

## 6. Documentation

Every text describing the current simulator: the stm README's toggles / Decay / Feenstra / simulator.py rows; the
physics block's text and the module docstring (physics apart from machinery, Chen the only citation);
`_calculate_ldos_at_points_gpu`'s docstring; stm_from_dft/__init__.py and quadrature.py where they name the old
functions; run_DFT_STM.py's comment and its `decay_linearized` keyword.

## Not touched

Everything in `wrong_decays/`; the README's `wrong_decays/` row; the historical `CHANGE_REPORT.md`.

## Process

One workflow on the branch: three implementation stages in sequence (names, model and control; the linearized kernel;
texts, run scripts, widgets.py), three reviewers (fidelity to the owner's file and this plan, with the exact option's
diff renames only; every formula derived, the walk against quadrature.py line by line; correctness: CUDA argument
lists, every caller, no old name left outside wrong_decays/ and the history), a fixer (anything needing a choice comes
back to the owner unchosen). py_compile every changed file; commit and push the branch; the change list; this folder deleted in a last commit; main fast-forwarded only on the owner's OK.
