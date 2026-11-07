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
