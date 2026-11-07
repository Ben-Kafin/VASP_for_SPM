# ==============================================================================
# UNUSED: not imported anywhere. The decay model of stm_from_dft/simulator.py
# before Chen's model (gpu_chen_tunneling_factor, Eqs. 2.34-2.40 of C. J. Chen,
# Introduction to Scanning Tunneling Microscopy, 3rd ed.): the excerpts below
# are verbatim from simulator.py, every original line commented with a leading
# '# ' (removing it restores the code; methods keep their class indentation).
#
# How it worked. Every DOSCAR state at E (eV from E_F) of every periodic atom
# image at distance d from the tip was weighted by exp(-K d), K the WKB average
# of 2 kappa over a barrier rising linearly from phi - E_arg to phi - E_arg + V:
#     K = 8 pi sqrt(2 m) / (3 h V) [(phi - E_arg + V)^(3/2) - (phi - E_arg)^(3/2)],
#     K = 4 pi sqrt(2 m (phi - E_arg)) / h at V = 0
# (the trapezoidal-barrier WKB integral, e.g. the TU Wien dissertation at
# iue.tuwien.ac.at/phd/goes, App. A.3, Eq. A.28; R. Rejali et al., Phys. Rev. B
# 107, 035406 (2023), Eq. 5), named after Chen in this code but not the model of
# Chen's book. phi was the LOCPOT potential at the tip point minus E_F. Every
# state was given V = |E|, as if at the edge of its own bias window, in the
# topography and the display alike; E_arg = max(E, 0) with the '0-clamp' toggle
# on (the default), E_arg = E with it off. Decay off weighted by exp(-2 kappa d),
# kappa = 0.512 sqrt(phi) 1/Angstrom, the constant also used by the tip-height
# convergence step. The 'Feenstra' current integrated the displayed LDOS from
# 0 V. Topography caches carried '_decay_on', '_decay_on_noclamp' or
# '_decay_off'; every caller of _calculate_ldos_at_points_gpu and
# _converge_tip_height passed clamp_occupied=self.t2_clamp.
# ==============================================================================
# #
# # ---- module level --------------------------------------------------------------
# def gpu_chen_tunneling_factor(V, E, phi):
#     """Vectorized Julian Chen barrier model character-for-character, in float64.
#
#     No floor on V: every V != 0 goes through the formula as written; at V = 0 exactly (where the
#     formula is 0/0) K is its exact V -> 0 limit, 4*pi*sqrt(2*m*max(0, phi - E))/h, the 2*kappa
#     of a rectangular barrier of height phi - E.
#     """
#     me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
#     V = cp.abs(cp.asarray(V, dtype=cp.float64))
#     E = cp.asarray(E, dtype=cp.float64)
#     phi = cp.asarray(phi, dtype=cp.float64)
#     at_zero = V == 0.0
#     V_formula = cp.where(at_zero, 1.0, V)      # V = 0 entries only keep the formula finite; they are replaced by the limit below
#     V_j, E_j, phi_j = V_formula * q, E * q, phi * q
#     prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
#     term1 = cp.power(cp.maximum(0.0, phi_j - E_j + V_j), 1.5)
#     term2 = cp.power(cp.maximum(0.0, phi_j - E_j), 1.5)
#     K_limit = 4.0 * cp.pi * cp.sqrt(2.0 * me * cp.maximum(0.0, phi_j - E_j)) / h
#     return cp.where(at_zero, K_limit, prefactor * (term1 - term2))
# #
# # ---- Unified_STM_Simulator._converge_tip_height --------------------------------
#     def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos,
#                              target_threshold=0.01, topo_gain=0.5, max_iter=1000, use_decay=True,
#                              clamp_occupied=False):
#         """Exhaustive Point-Wise Convergence Engine."""
#         t_start = time()
#         print("--- INITIALIZING TIP CONVERGENCE ENGINE ---")
#         for i in range(max_iter):
#             t0 = time()
#             ld_1, _, a_grid = self._calculate_ldos_at_points_gpu(
#                 cp.hstack([grid_xy_gpu, z_map_gpu[:, None]]), emin, emax,
#                 use_energy_decay=use_decay, preserve_orbitals=False, topo_only=True,
#                 clamp_occupied=clamp_occupied)
#             # Topography always uses total LDOS — ld_1 is already combined for POL when topo_only=True.
#             cur_ldos = gpu_simpson(ld_1, a_grid)
#             rat = cp.maximum(cur_ldos, 1e-20) / target_ldos
#
#             active_mask = cp.abs(rat - 1.0) > target_threshold
#             active_count = int(cp.sum(active_mask))
#             max_error = float(cp.max(cp.abs(rat - 1.0)))
#             z_min, z_max = float(cp.min(z_map_gpu)), float(cp.max(z_map_gpu))
#             print(f"   Iter {i+1:02d}: Active Pts ={active_count:4d} | Max Error ={max_error*100:6.2f}% | Z_min ={z_min:6.3f} Å | Z_max ={z_max:6.3f} Å | Range ={z_max-z_min:8.4f} Å | Time ={time()-t0:6.3f}s")
#             if active_count == 0:
#                 break
#
#             pts_active = cp.hstack([grid_xy_gpu, z_map_gpu[:, None]])[active_mask]
#             idx_g = (cp.dot(pts_active, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
#             pot_g = self._get_barrier_potential()
#             phi_l = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
#             kappa = 0.512 * cp.sqrt(phi_l)
#             z_map_gpu[active_mask] += (topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
#         print(f"--- CONVERGENCE COMPLETE: Total Time ={time() - t_start:6.3f}s ---")
#         return z_map_gpu
# #
# # ---- Unified_STM_Simulator._effective_decay_model, _decay_cache_suffix ---------
#     def _effective_decay_model(self):
#         """'decay_off' = Decay off; 'decay_on' = Decay on with the occupied-side
#         0-clamp (the default); 'decay_on_noclamp' = Decay on, the original model
#         without the clamp."""
#         if not getattr(self, 'use_decay', True):
#             return 'decay_off'
#         return 'decay_on' if getattr(self, 't2_clamp', True) else 'decay_on_noclamp'
#
#     def _decay_cache_suffix(self):
#         # Converged topographies depend on the decay model used to produce them;
#         # the tag is the model name, so each of the three models has its own
#         # caches (caches from the pre-tag unclamped model, no suffix, never match).
#         return "_" + self._effective_decay_model()
# #
# # ---- Unified_STM_Simulator._calculate_ldos_at_points_gpu -----------------------
#     def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax,
#                                       use_energy_decay=False, preserve_orbitals=False,
#                                       topo_only=False, clamp_occupied=False):
#         estart = np.searchsorted(self.energies, emin)
#         eend = np.searchsorted(self.energies, emax, side='right')
#         energy_indices = cp.arange(estart, eend)
#         calc_energies_gpu = self.energies_gpu[estart:eend]
#         num_pts, num_e = tip_positions.shape[0], len(calc_energies_gpu)
#         tip_pos_gpu = cp.asarray(tip_positions, dtype=cp.float64)
#         frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)
#
#         wrapped_frac_coords = cp.empty_like(frac_coords)
#         wrapped_frac_coords[:, :2] = frac_coords[:, :2] % 1.0
#         wrapped_frac_coords[:, 2] = frac_coords[:, 2]
#
#         lv_gpu = cp.array(self.lv, dtype=cp.float64)
#         wrapped_tip_pos_gpu = cp.dot(wrapped_frac_coords, lv_gpu)
#
#         grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
#         dists = cp.sqrt(cp.sum((self.periodic_coord_gpu[:, None, :] - wrapped_tip_pos_gpu[None, :, :]) ** 2, axis=2))
#
#         def _compute_channel(pot, dos_gpu, dos_collapsed):
#             phi_local = cp_ndimage.map_coordinates(pot, grid_indices, order=1, mode='wrap') - self.ef
#             if not preserve_orbitals:
#                 output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float64)
#                 dos_periodic = dos_collapsed[self.atom_indices_periodic_gpu, :]
#                 dos_active = dos_periodic[:, energy_indices]
#                 if use_energy_decay:
#                     # 0-clamp (on by default; the '0-clamp' toggle): E anchored at
#                     # the Fermi level for occupied states, moving the eps < 0
#                     # averaging interval to [phi, phi + |eps|]. Off = the original
#                     # decay model.
#                     e_arg = cp.maximum(calc_energies_gpu, 0.0) if clamp_occupied else calc_energies_gpu
#                     K_all = gpu_chen_tunneling_factor(cp.abs(calc_energies_gpu)[:, None], e_arg[:, None], phi_local)
#                     for e_idx in range(num_e):
#                         sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
#                         output_ldos[:, e_idx] = cp.dot(sf.T, dos_active[:, e_idx])
#                 else:
#                     kappa = 0.512 * cp.sqrt(phi_local)
#                     sf = cp.exp(-2.0 * kappa[None, :] * dists)
#                     output_ldos = cp.dot(sf.T, dos_active)
#             else:
#                 output_ldos = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float64)
#                 if use_energy_decay:
#                     e_arg = cp.maximum(calc_energies_gpu, 0.0) if clamp_occupied else calc_energies_gpu
#                     K_all = gpu_chen_tunneling_factor(cp.abs(calc_energies_gpu)[:, None], e_arg[:, None], phi_local)
#                     for e_idx in range(num_e):
#                         sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
#                         w_atom = cp.dot(self.map_mat_gpu, sf)
#                         output_ldos[:, e_idx, :, :] = w_atom.T[:, :, None] * dos_gpu[:, energy_indices[e_idx], :][None, :, :]
#                 else:
#                     kappa = 0.512 * cp.sqrt(phi_local)
#                     sf = cp.exp(-2.0 * kappa[None, :] * dists)
#                     w_atom = cp.dot(self.map_mat_gpu, sf)
#                     output_ldos = w_atom.T[:, None, :, None] * dos_gpu[:, energy_indices, :].transpose(1, 0, 2)[None, :, :, :]
#             return output_ldos
#
#         # All modes use total potential for the barrier
#         barrier_pot = self._get_barrier_potential()
#
#         if topo_only:
#             # Topography only needs ρ_total. Skip the second (magnetization) channel that
#             # _converge_tip_height / topo callers ignore — 2× speedup for NCL_SOC and POL.
#             if self.spin_mode == SpinMode.COLLINEAR_POL:
#                 # Use precomputed (dos_up + dos_dn) instead of computing two channels and adding.
#                 return (_compute_channel(barrier_pot, self.dos_total_gpu, self.dos_total_collapsed),
#                         None, calc_energies_gpu)
#             # NCL_SOC: dos_up_gpu IS already ρ_total per orbital.
#             # COLLINEAR_UNPOL: dos_up_gpu IS the single channel.
#             return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
#                     None, calc_energies_gpu)
#
#         if self.spin_mode == SpinMode.COLLINEAR_POL:
#             # Collinear polarized: two channels (up, down), both against total barrier
#             return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
#                     _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed),
#                     calc_energies_gpu)
#         elif self.spin_mode == SpinMode.NCL_SOC:
#             # NCL: channel 1 = total DOS, channel 2 = |m| DOS
#             return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
#                     _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed),
#                     calc_energies_gpu)
#         else:
#             # Unpolarized: single channel
#             return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
#                     None, calc_energies_gpu)
# #
# # ---- Interactive_STM_Simulator.__init__ (the toggle's state) -------------------
#         self.t2_clamp = True     # occupied-side 0-clamp within the decay model (on by default)
# #
# # ---- Interactive_STM_Simulator._feenstra_current -------------------------------
#     def _feenstra_current(self, pts, need):
#         """
#         The current at the tip positions `pts` (n, 3) on the span of DOSCAR samples that covers the
#         samples `need` (indices into self.energies) and brackets 0 V: the total LDOS
#         (topo_only, with the same Decay, 0-clamp and extra broadening as the display) integrated
#         from 0 V (feenstra.current_from_ldos).  Returns {'I': (n, n_span), 's0': the span's first
#         sample index, 'width': None, 'iv': None} (width / iv: the last I/V formed from it).
#         """
#         E = self.energies
#         n = len(E)
#         k_lo = max(int(np.searchsorted(E, 0.0, side='right')) - 1, 0)     # the last sample at or below 0 V
#         k_hi = min(int(np.searchsorted(E, 0.0, side='left')), n - 1)       # the first sample at or above 0 V
#         s0 = min(int(np.min(need)), k_lo)
#         s1 = max(int(np.max(need)) + 1, k_hi + 1)
#         ld, _, _ = self._calculate_ldos_at_points_gpu(
#             cp.asarray(np.asarray(pts, dtype=np.float64), dtype=cp.float64), E[s0], E[s1 - 1],
#             use_energy_decay=self.use_decay, preserve_orbitals=False, topo_only=True,
#             clamp_occupied=self.t2_clamp)
#         return {'I': feenstra.current_from_ldos(E[s0:s1], cp.asnumpy(ld)), 's0': s0, 'width': None, 'iv': None}
# #
# # ---- Interactive_STM_Simulator._make_controls (the toggles) --------------------
#         # One list defines every toggle: label and the attribute it drives. '0-clamp' sits
#         # indented under 'Decay' as its sub-option (the occupied-side 0-clamp of the decay model,
#         # on by default); it stays visible whether or not Decay is on. 'broadened' sits indented
#         # under 'Feenstra' the same way (the I/V broadened by the 'width (V)' box).
#         toggle_defs = [('Atoms', 'show_atoms'), ('Decay', 'use_decay'),
#                        ('0-clamp', 't2_clamp'),
#                        ('Norm', 'normalize'), ('Feenstra', 'feenstra'),
#                        ('broadened', 'feenstra_broadened'), ('Mag', 'show_mag'),
#                        ('Cell', 'show_unit_cell'), ('Extend', 'path_extend')]
#         if not getattr(self, 'show_decay_toggle', True):
#             toggle_defs = [d for d in toggle_defs if d[0] not in ('Decay', '0-clamp')]
#         self._toggle_defs = toggle_defs
#         labels = [label for label, _ in toggle_defs]
#         self.chk = controls.make(fig, 'sim toggles', labels=labels,
#                                  values=[getattr(self, attr) for _, attr in toggle_defs],
#                                  indents=['    ' if label in ('0-clamp', 'broadened') else 0 for label in labels])
# #
# # ---- Interactive_STM_Simulator._on_ui_change (to the end of the decay block) ----
#     def _on_ui_change(self, *args):
#         """A toggle, a count or a value box changed: the state is read from the controls, the caches invalidated as the change asks, and the window redrawn."""
#         prev_decay = self.use_decay
#         prev_clamp = self.t2_clamp
#         prev_path_extend = self.path_extend
#         states = self.chk.get_status()
#         for (label, attr), state in zip(self._toggle_defs, states):
#             setattr(self, attr, state)
#         self.display_cells = int(self.s_cell.val)
#         self.ldos_gamma = float(self.s_ldos_gamma.value)
#         # The Feenstra width (display only, as 'Feenstra' and 'broadened': they fall through to the redraw);
#         # its box shown only while 'broadened' is on.
#         self.feenstra_width = float(self.s_feen_width.val)
#         self.s_feen_width.show(self.feenstra_broadened)
#
#         # 0-clamp toggled while Decay is off: it only takes effect once Decay is
#         # on, so record the state and do nothing else (no recompute, no redraw).
#         clamp_changed = (self.t2_clamp != prev_clamp)
#         if clamp_changed and not self.use_decay:
#             return
#
#         # Decay toggle, 0-clamp toggle (with Decay on), or path-extend toggle:
#         # invalidate caches and recompute.
#         decay_model_changed = (self.use_decay != prev_decay) or (clamp_changed and self.use_decay)
#         if decay_model_changed or (self.path_extend != prev_path_extend):
#             self.cached_p1 = self.cached_p2 = None
#             self.cached_path_nodes = None
#             self.cached_emin = self.cached_emax = None
#             self.cached_d_topo_line = self.cached_d_topo_map = None
#             self.cached_d_ldos = None
#             self.cached_bias_energy_line = self.cached_bias_energy_map = None
#             self.cached_marker_coords = self.cached_spec_ldos = None
#             self.cached_ld_up = self.cached_ld_dn = None
#             # Only a decay-model change alters the global topo (recompute, or
#             # load the model-tagged cache); path-extend does not.
#             if decay_model_changed:
#                 self._broadening_topo_dirty = True
#                 if self.is_running:
#                     self._recompute_global_topo()
#                     self._broadening_topo_dirty = False
#             if self.is_running:
#                 self._update_all(full_refresh=True)
#             return
