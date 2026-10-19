# ==============================================================================
# UNUSED: this module is not imported anywhere in stm_from_dft and is not
# exported by __init__.py. It is kept as reference for the tunneling-barrier
# decay work that is still planned (Tier 3 decay logic, extracted reference).
# Every original line below is commented out with a leading '# '; removing
# that prefix from every line restores the file exactly as it was.
# ==============================================================================
# # -*- coding: utf-8 -*-
# """Tier 3 decay logic — extracted reference, NOT wired into the simulator.
# 
# This module preserves the complete Tier 3 ("t3") model from SPEC v3
# (full_model_decay_spec_v3.md §4, with the July 2026 user ruling that both
# window polarities reach to AND INCLUDE the Fermi grid point:
# 0 <= sign(V)*eps <= |V|). It was implemented and GPU-verified inside
# stm_from_dft/simulator.py (33/33 checks, flat-DOS bowl symmetric to 0.00%,
# shape ~1.02/1.07/1.29/1.71/2.42 x G(0)) and then removed from the live
# simulator at the user's request, which reverted to Tier 2 (+ optional
# occupied-side 0-clamp) only.
# 
# The full working tiered build (t1/t2/t2c/t3) this was lifted from is
# preserved verbatim in the session scratchpad as simulator_v3_full.py; the
# code below is the t3-specific logic plus integration notes. It is NOT
# importable/runnable as-is: the "nested" functions were closures inside
# Unified_STM_Simulator._calculate_ldos_at_points_gpu with access to
# (self, estart, num_e, num_pts, dists, calc_energies_gpu, preserve_orbitals,
# decay_mode, frozen_bias) from the enclosing scope.
# 
# Tier 3 in one paragraph:
#   Topography converges on a SIGNED current: every window state is decayed
#   with frozen arguments at the topo's own signed bias (V = |V_b|,
#   E = eps + max(-V_b, 0)), the setpoint is the signed value at the
#   maximum-magnitude location, and the convergence ratio uses magnitudes so
#   a negative field is never floored away. Display is a signed I(V) curve —
#   for each display grid energy V_j, Simpson-integrate the frozen-decayed
#   site-projected LDOS over the full-grid column window from the Fermi level
#   out to V_j, sign it by V_j — numerically differentiated (central
#   differences, one-sided at the display-range ends) to give dI/dV.
#   No per-state instantaneous V, no clamp, no analytic background under t3.
#   Cache token: '_t3'.
# """
# 
# import numpy as np
# import cupy as cp
# 
# # The frozen evaluations call the UNMODIFIED barrier model from simulator.py:
# # from .simulator import gpu_chen_tunneling_factor, gpu_simpson
# 
# 
# # ---------------------------------------------------------------------------
# # Module-level integrator helper (this one IS standalone-usable).
# # ---------------------------------------------------------------------------
# def gpu_simpson_weights(n, dx, dtype=cp.float32):
#     """Quadrature weight vector reproducing gpu_simpson exactly.
# 
#     Sum_i w_i*y_i equals gpu_simpson(y, x) for a uniform grid of n samples
#     with spacing dx, branch for branch (n == 1 -> 0, n == 2 -> trapezoid,
#     odd n -> composite Simpson, even n -> composite over n-1 plus the
#     Cartwright tail). Used where the integrand tensor is too large to
#     materialize per energy (the preserve_orbitals I(V) column integral).
#     """
#     if n == 1:
#         return cp.zeros(1, dtype=dtype)
#     if n == 2:
#         return cp.full(2, dx / 2.0, dtype=dtype)
#     w = np.zeros(n)
#     if n % 2 == 1:
#         ws = np.ones(n)
#         ws[1:-1:2] = 4
#         ws[2:-2:2] = 2
#         w = (dx / 3.0) * ws
#     else:
#         m = n - 1
#         wm = np.ones(m)
#         wm[1:-1:2] = 4
#         wm[2:-2:2] = 2
#         w[:m] = (dx / 3.0) * wm
#         w[-1] += 5.0 * dx / 12.0
#         w[-2] += 8.0 * dx / 12.0
#         w[-3] -= dx / 12.0
#     return cp.asarray(w, dtype=dtype)
# 
# 
# # ---------------------------------------------------------------------------
# # §4.1 — frozen-argument mapping and the (ruled, inclusive) window.
# # These lived as closures inside _calculate_ldos_at_points_gpu.
# # ---------------------------------------------------------------------------
# TIER3_WINDOW_FILTER = r'''
#         # In _calculate_ldos_at_points_gpu, right after the estart/eend slice
#         # (decay_mode == 'frozen' is the t3 TOPOGRAPHY path; frozen_bias is
#         # that topo's signed stabilization bias):
#         if decay_mode == 'frozen':
#             if frozen_bias is None:
#                 raise ValueError("decay_mode='frozen' requires a signed frozen_bias (eV).")
#             # §4.1 window (amended by user ruling): 0 ≤ sign(V_b)·ε ≤ |V_b| —
#             # both polarities reach to AND INCLUDE the Fermi grid point, so the
#             # frozen E-argument samples are identical for ±|V_b|. The slice
#             # bounds already cap at |V_b|; drop any wrong-side state.
#             keep = np.sign(frozen_bias) * self.energies[estart:eend] >= 0.0
#             if not keep.all():
#                 keep_gpu = cp.asarray(keep)
#                 energy_indices = energy_indices[keep_gpu]
#                 calc_energies_gpu = calc_energies_gpu[keep_gpu]
# '''
# 
# TIER3_FROZEN_K = r'''
#         def _frozen_K(phi_local, v_b, eps_gpu):
#             """§4.1 frozen mapping: V = |V_b|; E = ε + max(−V_b, 0) (≥ 0 in-window)."""
#             return gpu_chen_tunneling_factor(
#                 abs(v_b), (eps_gpu + max(-v_b, 0.0))[:, None], phi_local)
# 
#         # Inside _leading_K (the per-energy decay factor for the topo path):
#         #     if decay_mode == 'frozen':
#         #         return _frozen_K(phi_local, frozen_bias, calc_energies_gpu)
# '''
# 
# # ---------------------------------------------------------------------------
# # §4.3 — signed I(V)-column display -> numerical dI/dV.
# # Closure inside _calculate_ldos_at_points_gpu; dispatched from
# # _compute_channel via:
# #     if use_energy_decay and decay_mode == 't3_display':
# #         return _t3_display_channel(phi_local, dos_gpu, dos_collapsed)
# # ---------------------------------------------------------------------------
# TIER3_DISPLAY_CHANNEL = r'''
#         def _t3_display_channel(phi_local, dos_gpu, dos_collapsed):
#             """§4.3: signed I(V_j) from each display grid energy's frozen column
#             window (full DOSCAR grid, Fermi → V_j), then numerical dI/dV —
#             central differences, one-sided at the two display-range ends."""
#             d_e = float(self.energies[1] - self.energies[0])
#             n_per = dists.shape[0]
#             chunk = max(1, int(16_777_216 // max(1, n_per * num_pts)))
#             if preserve_orbitals:
#                 I_of_v = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]),
#                                   dtype=cp.float32)
#             else:
#                 dos_periodic_full = dos_collapsed[self.atom_indices_periodic_gpu, :]
#                 I_of_v = cp.zeros((num_pts, num_e), dtype=cp.float32)
#             for e_idx in range(num_e):
#                 v_j = float(self.energies[estart + e_idx])
#                 # Column window (user ruling): reaches to and includes the
#                 # Fermi grid point on both polarities: 0 ≤ sign(V_j)·ε ≤ |V_j|.
#                 if v_j > 0:
#                     col = np.nonzero((self.energies >= 0.0) & (self.energies <= v_j))[0]
#                 elif v_j < 0:
#                     col = np.nonzero((self.energies <= 0.0) & (self.energies >= v_j))[0]
#                 else:
#                     continue   # V_j = 0: empty column window → I(V_j) = 0
#                 n_col = len(col)
#                 if n_col == 0:
#                     continue
#                 sgn = 1.0 if v_j > 0 else -1.0
#                 if preserve_orbitals:
#                     # Column carried per orbital channel before this branch's
#                     # reduction: gpu_simpson's exact weights applied per
#                     # (atom, energy) with the periodic-image reduction.
#                     w_col = gpu_simpson_weights(n_col, d_e)
#                     acc = cp.zeros((num_pts, self.num_total_atoms, dos_gpu.shape[2]),
#                                    dtype=cp.float32)
#                     for c0 in range(0, n_col, chunk):
#                         idx_c = col[c0:c0 + chunk]
#                         eps_c = cp.asarray(self.energies[idx_c], dtype=cp.float32)
#                         K_c = _frozen_K(phi_local, v_j, eps_c).astype(cp.float32)
#                         sf = cp.exp(-K_c[:, None, :] * dists[None, :, :] * 1e-10)
#                         w_atom = cp.einsum('mn,inp->imp', self.map_mat_gpu, sf)
#                         acc += cp.einsum('i,imp,mio->pmo', w_col[c0:c0 + len(idx_c)],
#                                          w_atom, dos_gpu[:, cp.asarray(idx_c), :])
#                     I_of_v[:, e_idx] = sgn * acc
#                 else:
#                     y = cp.zeros((num_pts, n_col), dtype=cp.float32)
#                     for c0 in range(0, n_col, chunk):
#                         idx_c = col[c0:c0 + chunk]
#                         eps_c = cp.asarray(self.energies[idx_c], dtype=cp.float32)
#                         K_c = _frozen_K(phi_local, v_j, eps_c).astype(cp.float32)
#                         sf = cp.exp(-K_c[:, None, :] * dists[None, :, :] * 1e-10)
#                         y[:, c0:c0 + len(idx_c)] = cp.einsum(
#                             'ai,iap->pi', dos_periodic_full[:, cp.asarray(idx_c)], sf)
#                     I_of_v[:, e_idx] = sgn * gpu_simpson(
#                         y, cp.asarray(self.energies[col], dtype=cp.float32))
#             didv = cp.zeros_like(I_of_v)
#             if num_e >= 2:
#                 didv[:, 0] = (I_of_v[:, 1] - I_of_v[:, 0]) / d_e
#                 didv[:, num_e - 1] = (I_of_v[:, num_e - 1] - I_of_v[:, num_e - 2]) / d_e
#                 if num_e > 2:
#                     didv[:, 1:num_e - 1] = (I_of_v[:, 2:] - I_of_v[:, :num_e - 2]) / (2.0 * d_e)
#             return didv
# '''
# 
# # ---------------------------------------------------------------------------
# # §4.2 — sign-aware topography convergence pieces.
# # ---------------------------------------------------------------------------
# TIER3_TOPO_SETPOINT = r'''
#     def _topo_setpoint(self, ld_1, engs, bias):
#         """§4.2 setpoint: under t3 the signed value at the maximum-magnitude
#         location; under t1/t2/t2c today's positive maximum, unchanged."""
#         integ = gpu_simpson(ld_1, engs)
#         if self._effective_decay_model() == 't3':
#             return (1.0 if bias > 0 else -1.0) * cp.max(cp.abs(integ))
#         return cp.max(integ)
# '''
# 
# TIER3_CONVERGE_DELTAS = r'''
#     # Inside _converge_tip_height (signature gained decay_mode='instantaneous',
#     # frozen_bias=None, forwarded to the internal _calculate call):
#             cur_ldos = gpu_simpson(ld_1, a_grid)
#             if decay_mode == 'frozen' and frozen_bias is not None and frozen_bias < 0:
#                 cur_ldos = -cur_ldos   # §4.2: topo current is signed; negative bias → negative field
#             rat = cp.maximum(cp.abs(cur_ldos), 1e-20) / cp.abs(target_ldos)
#     # (magnitude ratio: the old positive floor cp.maximum(cur_ldos, 1e-20)
#     #  would clamp a negative field wholesale. Error metric, z-bounds, the
#     #  log-ratio Newton step and the 0.512*sqrt(phi) step-gain kappa are
#     #  unchanged.)
# '''
# 
# # ---------------------------------------------------------------------------
# # Model plumbing (how t3 was selected and routed).
# # ---------------------------------------------------------------------------
# TIER3_PLUMBING = r'''
#     # _effective_decay_model returned 't3' when the tier RadioButtons
#     # ('Off' / 'Tier 2' / 'Tier 3') sat on Tier 3 (clamp checkbox ignored).
#     # Cache token: '_t3' via _decay_cache_suffix = "_" + model.
#     #
#     # _decay_call_kwargs(frozen_bias=None) under t3:
#     #   - topography callers (which pass their signed stabilization bias):
#     #       dict(use_energy_decay=True, decay_mode='frozen', frozen_bias=bias)
#     #     at the four topo sites: run_interactive global topo (topo_bias),
#     #     _recompute_global_topo (self.global_topo_bias), Line/Path
#     #     stabilization (bias_e), Map-mode topo (bias_e) — each on BOTH the
#     #     setpoint _calculate call and the _converge_tip_height call, with
#     #     target_setp = self._topo_setpoint(ld_1, engs, bias).
#     #   - display callers (no bias argument):
#     #       dict(use_energy_decay=True, decay_mode='t3_display')
#     #     at the four display sites: Line/Path LDOS, Map per-target slices,
#     #     single-point spectra, emarker release.
#     #
#     # Known display properties (accepted at implementation time): Map mode
#     # feeds 2-grid-point display ranges per target, so each map panel shows
#     # the one-sided difference at E_j (offset d_e/2 vs Line mode at the same
#     # energy), and a target on the last DOSCAR grid point yields 0; I at the
#     # grid energy nearest the Fermi level is a tiny 2-point trapezoid, so
#     # dI/dV has a 1-2 grid-point startup transient around 0 V.
# '''
# 