# Chen linearized decay as a sub-toggle of Decay (temporary plan; deleted when done)

**Approved by the owner (2026-11-07):** Decay on/off as now. With Decay on, a sub-toggle chooses the weight:
- **exact** (default): exactly today's code with Decay on: the WKB tilted barrier (`gpu_chen_tunneling_factor`), the
  frozen-bias topography at V_b, the per-V `chen_sweep` dI/dV, the `'decay_tilt'` caches. Untouched.
- **linearized**: Chen's linearized weight w = exp(-2 kappa0 d + kappa0 (E - V/2) d / phi) (the README's and the
  physics block's form, Chen Eq. 2.40 as the repository states it), dw/dV = -(kappa0 d / (2 phi)) w. The topography
  and the LDOS bias treatments are the same as the exact option's (frozen bias at V_b; swept dI/dV), but because the
  weight is separable, w = exp(-2 kappa0 d) exp(kappa0 d E / phi) exp(-kappa0 d V / (2 phi)), the swept LDOS uses a
  single cumulative integral from 0 V for all energies (integral_from_zero's rule), giving dI/dV and the current at
  every requested V, in a linearized mode of the GPU kernel. Its own model name and caches.

Scoping (read-only, wf_4951f6d3-893): the result is in the session scratchpad; the code map lists every place the model
touches (physics block, kernel tilt_weight and its call sites, _compute_channel's three branches, _topo_current and
_converge_tip_height, _effective_decay_model and the records, _on_ui_change, toggle_defs with indents=, _on_rel's
hard-coded 'decay_tilt', run_DFT_STM.py, run_stm_decomp.py, sim_wrapper's defaults, the README rows). The former
linearized code: simulator.py at eb6ce36 (weight lines 86-91; its separable one-walk kernel, float64, trapezoid rule).

**Choices put to the owner:** see the conversation; recorded here when answered.

**The owner's source (2026-11-07):** "This has the linearized form of the decay implemented as I have described. you
shouldn't need to make anything up, just mesh this version of the decay as that secondary linearized toggle." The
uploaded simulator.py is byte-identical to the repository's simulator.py at b051547 (the owner's disk file installed
before part 1). Its decay: `gpu_chen_tunneling_factor` (the linearized weight and dw/dV, lines 88-101), its separable
one-walk `chen_sweep` kernel (`_CHEN_SWEEP_SRC`, lines 137-245: dI/dV = sum rho(V) w(V,V) + w(0,V)/w(0,0)
integral_0^V rho dw(E,0)/dV dE, and the current), `_integral_from_zero` (the trapezoid rule with a linear 0 V value),
float64 GPU arrays, the topography by `gpu_simpson` over the typed bias window. Open with the owner: whether its
numerics come with it, or its weight and one-walk design run under the package's float32 and quadrature rules.

## Decided (owner: "b", 2026-11-07) and what follows from the sources

- **(b):** the owner's linearized weight, its derivative and its separable one-walk kernel (b051547's
  `gpu_chen_tunneling_factor` and `_CHEN_SWEEP_SRC`) run under the package's current numerics: float32 on the GPU
  (part 2 section 9, energies formed from indices e0 + (i - k0) step), the integration rule of
  `quadrature.integral_from_zero` (the stencil cubic around 0 V, the unequal pair, Simpson pairs; the current exact
  kernel already carries this rule in CUDA, per V: the walk takes the same steps once, cumulatively, giving every
  requested V), the snapped V_b, the same topography path as the exact option (frozen bias at V_b, integrated from 0 V).
- **From the owner's file:** phi is the local barrier per tip point (phi_local, as the exact option); the model name
  is the file's own 'decay_chen' (its caches end '_decay_chen_fold_f32', so the old '_decay_chen_f64' files never
  match); no guard on phi (the file has none).
- **The control:** an indented row under Decay in the toggle list ('linearized'), off by default = exact (today's
  behaviour); run_interactive keyword `decay_linearized=False`; ignored while Decay is off (flipping it then changes
  no model, so nothing is recomputed); the info panel lists it with the other toggles.
- **The exact option:** untouched (its weight, kernel, topography, caches 'decay_tilt').
- **Everywhere the model is compared or passed:** a third name through `_effective_decay_model`; `_on_ui_change`
  compares models (not the Decay bool); `_on_rel` passes Map's kept model; the topography and LDOS calls pass the model.
- **Texts:** the physics block and module docstring, the README's toggles / Decay / Feenstra / simulator rows, the
  run scripts' comments (Chen only, ruling 3).
