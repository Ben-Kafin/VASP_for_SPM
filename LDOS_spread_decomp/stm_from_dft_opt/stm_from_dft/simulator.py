# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization
Auto-detects ISPIN=1, ISPIN=2, and NCL/SOC calculations.

The window (Interactive_STM_Simulator) is drawn with the suite's shared Qt layer (qtgui at the
repository root: PySide6 + pyqtgraph): its panels are cells of one Qt layout and its controls
real Qt widgets (automatic layout: the one rule for every window, nothing placed by hand), every
slider of the old figure a number box, the orbital hover labels widgets.HoverLabels.  The
physics (Unified_STM_Simulator) is unchanged.

Developed by Benjamin Kafin
"""

import os as _os
import sys as _sys

# The shared Qt layer (qtgui at the repository root): found by walking up from this file.
_root = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_root, 'qtgui')) and _os.path.dirname(_root) != _root:
    _root = _os.path.dirname(_root)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
from scipy.integrate import simpson
import colorsys
from time import time

from qtgui import qtplot as plt
from qtgui import controls, widgets
from qtgui.qtplot import QtWidgets, QtCore

from .doscar_parser import SpinAwareDosParser, SpinMode
from .locpot_manager import LocpotManager


# --- CORE UTILITIES ---
def gpu_simpson(y, x):
    """Vectorized composite Simpson's rule (uniform spacing) for GPU parity.

    Even sample counts use composite Simpson over the first n-1 samples plus a
    3-point parabolic (Cartwright) correction for the final interval.
    n == 2 uses the trapezoidal rule; n == 1 returns 0.
    """
    n = y.shape[1]
    if n == 1:
        return cp.zeros(y.shape[0], dtype=y.dtype)
    if n == 2:
        return cp.trapz(y, x=x, axis=1)
    dx = (x[-1] - x[0]) / (n - 1)
    if n % 2 == 1:
        weights = cp.ones(n)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        return (dx / 3.0) * cp.sum(weights * y, axis=1)
    m = n - 1
    weights = cp.ones(m)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    core = (dx / 3.0) * cp.sum(weights * y[:, :m], axis=1)
    tail = (dx / 12.0) * (5.0 * y[:, -1] + 8.0 * y[:, -2] - y[:, -3])
    return core + tail


def gpu_chen_tunneling_factor(V, E, phi):
    """Vectorized Julian Chen barrier model character-for-character."""
    me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
    V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
    V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
    prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
    term1 = cp.power(cp.maximum(0.0, phi_j - E_j + V_j), 1.5)
    term2 = cp.power(cp.maximum(0.0, phi_j - E_j), 1.5)
    return prefactor * (term1 - term2)


class Unified_STM_Simulator:
    def __init__(self, filepath, unit_cell_num=1):
        self.filepath = filepath
        # 3×3 = 9 periodic images per atom. Far images have sf ≈ exp(-2κd) ≈ 0 well before
        # the second neighbor ring, so the topo and LDOS sums are converged with one ring.
        # Override to higher values only for unusually small unit cells or very low κ.
        self.unit_cell_num = unit_cell_num
        chdir(filepath)
        self.dev = cp.cuda.Device(0)
        print("--- INITIALIZING GPU TENSOR SIMULATOR ---")

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos,
                             target_threshold=0.01, topo_gain=0.5, max_iter=1000, use_decay=True,
                             clamp_occupied=False):
        """Exhaustive Point-Wise Convergence Engine."""
        t_start = time()
        print("--- INITIALIZING TIP CONVERGENCE ENGINE ---")
        for i in range(max_iter):
            t0 = time()
            ld_1, _, a_grid = self._calculate_ldos_at_points_gpu(
                cp.hstack([grid_xy_gpu, z_map_gpu[:, None]]), emin, emax,
                use_energy_decay=use_decay, preserve_orbitals=False, topo_only=True,
                clamp_occupied=clamp_occupied)
            # Topography always uses total LDOS — ld_1 is already combined for POL when topo_only=True.
            cur_ldos = gpu_simpson(ld_1, a_grid)
            rat = cp.maximum(cur_ldos, 1e-20) / target_ldos

            active_mask = cp.abs(rat - 1.0) > target_threshold
            active_count = int(cp.sum(active_mask))
            max_error = float(cp.max(cp.abs(rat - 1.0)))
            z_min, z_max = float(cp.min(z_map_gpu)), float(cp.max(z_map_gpu))
            print(f"   Iter {i+1:02d}: Active Pts ={active_count:4d} | Max Error ={max_error*100:6.2f}% | Z_min ={z_min:6.3f} Å | Z_max ={z_max:6.3f} Å | Range ={z_max-z_min:8.4f} Å | Time ={time()-t0:6.3f}s")
            if active_count == 0:
                break

            pts_active = cp.hstack([grid_xy_gpu, z_map_gpu[:, None]])[active_mask]
            idx_g = (cp.dot(pts_active, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
            pot_g = self._get_barrier_potential()
            phi_l = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
            kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_l))
            z_map_gpu[active_mask] += (topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
        print(f"--- CONVERGENCE COMPLETE: Total Time ={time() - t_start:6.3f}s ---")
        return z_map_gpu

    def _get_barrier_potential(self):
        """Return the single 3D potential grid used for the tunneling barrier."""
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # V_total = V_up + V_dn (stored as [V_up, V_dn])
            return self.locpot_gpu[0] + self.locpot_gpu[1]
        # ISPIN=1 and NCL: locpot_gpu is already the total 3D potential
        return self.locpot_gpu

    def _parse_poscar(self, ifile):
        with open(ifile, 'r') as f:
            lines = f.readlines()
            sf = float(lines[1])
            lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3, 3) * sf
            atomtypes = lines[5].split()
            atomnums = [int(i) for i in lines[6].split()]
            start_line = 7 if lines[7].strip().lower()[0] in ['d', 'c'] else 8
            coord = np.array([[float(c) for c in line.split()[:3]]
                              for line in lines[start_line + 1:sum(atomnums) + start_line + 1]])
            if 'direct' in lines[start_line].lower():
                coord = np.dot(coord, lv)
        return lv, coord, atomtypes, atomnums

    def _unwrap_z_vacuum(self, coord):
        """Place every atom on the slab side of the vacuum gap in z.

        z is the surface-normal direction and is NOT periodic for a slab: an atom
        whose fractional z has wrapped through the cell boundary appears at the top
        of the cell (in the vacuum above the slab) when it physically belongs below
        the slab bottom. Such phantom atoms corrupt the vacuum LDOS as the tip is
        raised. This detects the vacuum as the largest gap in the circular
        fractional-z distribution (definitionally the vacuum in a slab cell) and
        shifts any atom above the gap's midpoint down by one c-vector so the slab is
        contiguous with no atom in the top vacuum. System-independent: no hardcoded
        thresholds, and a no-op when no atom has wrapped.
        """
        inv_lv = inv(self.lv)
        frac = np.dot(coord, inv_lv)
        fz = frac[:, 2] % 1.0
        n = len(fz)
        order = np.argsort(fz)
        fz_sorted = fz[order]
        gaps = np.empty(n)
        gaps[:-1] = np.diff(fz_sorted)
        gaps[-1] = (fz_sorted[0] + 1.0) - fz_sorted[-1]   # circular wrap-around gap
        vac_i = int(np.argmax(gaps))
        low_edge = fz_sorted[vac_i]
        high_edge = fz_sorted[(vac_i + 1) % n] + (1.0 if vac_i == n - 1 else 0.0)
        cut = ((low_edge + high_edge) / 2.0) % 1.0
        fz_unwrapped = fz.copy()
        fz_unwrapped[fz_unwrapped > cut] -= 1.0
        frac[:, 2] = fz_unwrapped
        return np.dot(frac, self.lv)

    def parse_vasp_outputs(self, locpot_path):
        poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar_path)
        self.coord = self._unwrap_z_vacuum(self.coord)

        # --- DOSCAR: auto-detects spin mode ---
        dos_parser = SpinAwareDosParser(join(self.filepath, 'DOSCAR'))
        self.energies, self.ef = dos_parser.energies, dos_parser.ef
        self.energies_gpu = cp.asarray(self.energies, dtype=cp.float32)
        self.spin_mode = dos_parser.spin_mode
        self.is_polarized = dos_parser.is_polarized  # backward compat (True only for COLLINEAR_POL)

        print(f"[*] Detected spin mode: {self.spin_mode.name}")

        # --- Orbital names ---
        num_cols = dos_parser.spin_up_dos.shape[2]
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # num_cols is already per-spin (halved by parser)
            mapping_pol = {
                3: ['s_up', 's_down', 'p_up', 'p_down', 'd_up', 'd_down'],
                9: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down',
                    'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down',
                    'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down'],
                16: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down',
                     'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down',
                     'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down',
                     'fy3x2_up', 'fy3x2_down', 'fxyz_up', 'fxyz_down',
                     'fyz2_up', 'fyz2_down', 'fz3_up', 'fz3_down',
                     'fxz2_up', 'fxz2_down', 'fzx2_up', 'fzx2_down',
                     'fx3_up', 'fx3_down'],
            }
            self.orbitals = mapping_pol.get(num_cols, [])
        else:
            # ISPIN=1 and NCL: no spin suffix
            mapping_unpol = {
                3: ['s', 'p', 'd'],
                9: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'],
                16: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2',
                     'fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3'],
            }
            self.orbitals = mapping_unpol.get(num_cols, [])

        # --- Atom labels ---
        current_global = 1
        self.vesta_label_map = {}
        for idx, t in enumerate(self.atomtypes):
            for n_rel in range(1, self.atomnums[idx] + 1):
                self.vesta_label_map[current_global] = f"{t}{n_rel}"
                current_global += 1

        # --- LOCPOT: auto-detects from section count ---
        lpt_mgr = LocpotManager(self.filepath, spin_mode=self.spin_mode)
        locpot_raw = cp.array(lpt_mgr.get_data(), dtype=cp.float32)

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # LocpotManager stores [V_up, V_dn].
            # Tunneling barrier uses TOTAL electrostatic potential = V_up + V_dn.
            # Spin physics enters through the DOS, not the barrier.
            # We keep both channels for the per-spin barrier in _compute_channel.
            v_total = locpot_raw[0] + locpot_raw[1]
            self.locpot_gpu = cp.stack([v_total, v_total])
        else:
            # ISPIN=1 and NCL: single 3D grid (total potential)
            self.locpot_gpu = locpot_raw

        self.inv_lv_gpu = cp.array(inv(self.lv), dtype=cp.float32)
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape[-3:], dtype=cp.float32)
        else:
            self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape, dtype=cp.float32)

        self.num_total_atoms = sum(self.atomnums)

        # --- DOS arrays on GPU ---
        self.dos_up_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='up'), dtype=cp.float32)
        self.dos_up_collapsed = cp.ascontiguousarray(cp.sum(self.dos_up_gpu, axis=2))

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.dos_dn_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='down'), dtype=cp.float32)
            self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))
        elif self.spin_mode == SpinMode.NCL_SOC:
            # NCL: second channel is magnetization magnitude |m|
            self.dos_dn_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='mag'), dtype=cp.float32)
            self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))
        else:
            self.dos_dn_gpu = None
            self.dos_dn_collapsed = None

        # Store pristine copies for extra broadening re-derivation
        self._dos_up_gpu_orig = self.dos_up_gpu.copy()
        self._dos_dn_gpu_orig = self.dos_dn_gpu.copy() if self.dos_dn_gpu is not None else None

        # Precompute total-ρ DOS for COLLINEAR_POL topography (avoids computing two channels and adding).
        # NCL_SOC: dos_up_gpu IS already ρ_total per orbital. COLLINEAR_UNPOL: single channel reuses dos_up_gpu.
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.dos_total_gpu = self.dos_up_gpu + self.dos_dn_gpu
            self.dos_total_collapsed = cp.ascontiguousarray(cp.sum(self.dos_total_gpu, axis=2))
        else:
            self.dos_total_gpu = None
            self.dos_total_collapsed = None

        inv_lv = inv(self.lv)
        frac_coords = np.dot(self.coord, inv_lv)
        z_filter_mask = frac_coords[:, 2] < 0.9
        self.z_highest_atom = np.max(self.coord[z_filter_mask, 2])

        coords, idx_list = [], []
        base_idx = np.arange(len(self.coord))
        for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
            for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                idx_list.append(base_idx)
        self.periodic_coord_gpu = cp.array(np.concatenate(coords), dtype=cp.float32)
        self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))
        self.map_mat_gpu = cp.zeros((self.num_total_atoms, len(self.atom_indices_periodic_gpu)), dtype=cp.float32)
        self.map_mat_gpu[self.atom_indices_periodic_gpu, cp.arange(len(self.atom_indices_periodic_gpu))] = 1.0

    def _apply_extra_broadening(self):
        """Rebuild working DOS arrays from pristine originals with extra Gaussian broadening."""
        self.dos_up_gpu = self._dos_up_gpu_orig.copy()
        if self._dos_dn_gpu_orig is not None:
            self.dos_dn_gpu = self._dos_dn_gpu_orig.copy()

        dE = self.energies[1] - self.energies[0]
        atom_types_exp = np.repeat(self.atomtypes, self.atomnums)

        for atype, sigma_eV in getattr(self, 'extra_broadening', {}).items():
            if sigma_eV <= 0:
                continue
            sigma_pts = sigma_eV / dE
            indices = [i for i, t in enumerate(atom_types_exp) if t == atype]
            if not indices:
                continue
            idx = cp.array(indices)
            self.dos_up_gpu[idx] = cp_ndimage.gaussian_filter1d(self.dos_up_gpu[idx], sigma=sigma_pts, axis=1)
            if self.dos_dn_gpu is not None:
                self.dos_dn_gpu[idx] = cp_ndimage.gaussian_filter1d(self.dos_dn_gpu[idx], sigma=sigma_pts, axis=1)

        self.dos_up_collapsed = cp.ascontiguousarray(cp.sum(self.dos_up_gpu, axis=2))
        if self.dos_dn_gpu is not None:
            self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))

        # Refresh COLLINEAR_POL total-ρ DOS after broadening changes (NCL_SOC and UNPOL reuse dos_up_gpu).
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            self.dos_total_gpu = self.dos_up_gpu + self.dos_dn_gpu
            self.dos_total_collapsed = cp.ascontiguousarray(cp.sum(self.dos_total_gpu, axis=2))

    def _broadening_cache_suffix(self):
        """Generate cache filename suffix encoding active broadening state."""
        active = sorted((t, s) for t, s in getattr(self, 'extra_broadening', {}).items() if s > 0)
        if not active:
            return ""
        return "_broad_" + "_".join(f"{t}{s:.3f}" for t, s in active)

    def _unit_cell_cache_suffix(self):
        # Periodic-image count affects every sf sum. Embed in cache filename so old caches
        # auto-invalidate when unit_cell_num changes.
        return f"_uc{getattr(self, 'unit_cell_num', 1)}"

    def _effective_decay_model(self):
        """'decay_off' = Decay off; 'decay_on' = Decay on with the occupied-side
        0-clamp (the default); 'decay_on_noclamp' = Decay on, the original model
        without the clamp."""
        if not getattr(self, 'use_decay', True):
            return 'decay_off'
        return 'decay_on' if getattr(self, 't2_clamp', True) else 'decay_on_noclamp'

    def _decay_cache_suffix(self):
        # Converged topographies depend on the decay model used to produce them;
        # the tag is the model name, so each of the three models has its own
        # caches (caches from the pre-tag unclamped model, no suffix, never match).
        return "_" + self._effective_decay_model()

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax,
                                      use_energy_decay=False, preserve_orbitals=False,
                                      topo_only=False, clamp_occupied=False):
        estart = np.searchsorted(self.energies, emin)
        eend = np.searchsorted(self.energies, emax, side='right')
        energy_indices = cp.arange(estart, eend)
        calc_energies_gpu = self.energies_gpu[estart:eend]
        num_pts, num_e = tip_positions.shape[0], len(calc_energies_gpu)
        tip_pos_gpu = cp.asarray(tip_positions, dtype=cp.float32)
        frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)

        wrapped_frac_coords = cp.empty_like(frac_coords)
        wrapped_frac_coords[:, :2] = frac_coords[:, :2] % 1.0
        wrapped_frac_coords[:, 2] = frac_coords[:, 2]

        lv_gpu = cp.array(self.lv, dtype=cp.float32)
        wrapped_tip_pos_gpu = cp.dot(wrapped_frac_coords, lv_gpu)

        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        dists = cp.sqrt(cp.sum((self.periodic_coord_gpu[:, None, :] - wrapped_tip_pos_gpu[None, :, :]) ** 2, axis=2))

        def _compute_channel(pot, dos_gpu, dos_collapsed):
            phi_local = cp_ndimage.map_coordinates(pot, grid_indices, order=1, mode='wrap') - self.ef
            if not preserve_orbitals:
                output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float32)
                dos_periodic = dos_collapsed[self.atom_indices_periodic_gpu, :]
                dos_active = dos_periodic[:, energy_indices]
                if use_energy_decay:
                    # 0-clamp (on by default; the '0-clamp' toggle): E anchored at
                    # the Fermi level for occupied states, moving the eps < 0
                    # averaging interval to [phi, phi + |eps|]. Off = the original
                    # decay model.
                    e_arg = cp.maximum(calc_energies_gpu, 0.0) if clamp_occupied else calc_energies_gpu
                    K_all = gpu_chen_tunneling_factor(cp.abs(calc_energies_gpu)[:, None], e_arg[:, None], phi_local)
                    for e_idx in range(num_e):
                        sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
                        output_ldos[:, e_idx] = cp.dot(sf.T, dos_active[:, e_idx])
                else:
                    kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                    sf = cp.exp(-2.0 * kappa[None, :] * dists)
                    output_ldos = cp.dot(sf.T, dos_active)
            else:
                output_ldos = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float32)
                if use_energy_decay:
                    e_arg = cp.maximum(calc_energies_gpu, 0.0) if clamp_occupied else calc_energies_gpu
                    K_all = gpu_chen_tunneling_factor(cp.abs(calc_energies_gpu)[:, None], e_arg[:, None], phi_local)
                    for e_idx in range(num_e):
                        sf = cp.exp(-1.0 * dists * K_all[e_idx][None, :] * 1e-10)
                        w_atom = cp.dot(self.map_mat_gpu, sf)
                        output_ldos[:, e_idx, :, :] = w_atom.T[:, :, None] * dos_gpu[:, energy_indices[e_idx], :][None, :, :]
                else:
                    kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                    sf = cp.exp(-2.0 * kappa[None, :] * dists)
                    w_atom = cp.dot(self.map_mat_gpu, sf)
                    output_ldos = w_atom.T[:, None, :, None] * dos_gpu[:, energy_indices, :].transpose(1, 0, 2)[None, :, :, :]
            return output_ldos

        # All modes use total potential for the barrier
        barrier_pot = self._get_barrier_potential()

        if topo_only:
            # Topography only needs ρ_total. Skip the second (magnetization) channel that
            # _converge_tip_height / topo callers ignore — 2× speedup for NCL_SOC and POL.
            if self.spin_mode == SpinMode.COLLINEAR_POL:
                # Use precomputed (dos_up + dos_dn) instead of computing two channels and adding.
                return (_compute_channel(barrier_pot, self.dos_total_gpu, self.dos_total_collapsed),
                        None, calc_energies_gpu)
            # NCL_SOC: dos_up_gpu IS already ρ_total per orbital.
            # COLLINEAR_UNPOL: dos_up_gpu IS the single channel.
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    None, calc_energies_gpu)

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # Collinear polarized: two channels (up, down), both against total barrier
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed),
                    calc_energies_gpu)
        elif self.spin_mode == SpinMode.NCL_SOC:
            # NCL: channel 1 = total DOS, channel 2 = |m| DOS
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed),
                    calc_energies_gpu)
        else:
            # Unpolarized: single channel
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed),
                    None, calc_energies_gpu)

    # ------------------------------------------------------------------
    # _combine_ldos: central dispatch for how channels are combined at display time
    # ------------------------------------------------------------------
    def _combine_ldos_total(self, ld_1, ld_2):
        """Combine channels into total LDOS (for topography / non-mag display)."""
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            return ld_1 + ld_2
        # NCL and unpol: ld_1 is already the total
        return ld_1

    def _combine_ldos_display(self, ld_1, ld_2, show_mag):
        """Combine channels for display, respecting show_mag toggle."""
        if not show_mag or ld_2 is None:
            return self._combine_ldos_total(ld_1, ld_2)
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            return ld_1 - ld_2   # spin density (signed)
        if self.spin_mode == SpinMode.NCL_SOC:
            return ld_2           # |m| (unsigned, always ≥ 0)
        return ld_1

    def _mag_is_signed(self):
        """Whether the magnetization display data is signed (needs bwr) or not."""
        return self.spin_mode == SpinMode.COLLINEAR_POL

    def _has_mag_channel(self):
        """Whether this spin mode supports the Mag checkbox."""
        return self.spin_mode in (SpinMode.COLLINEAR_POL, SpinMode.NCL_SOC)


DESIGN = (20, 14)        # the window's design size (inches): its text zoom is its size relative to this (qtplot.window_zoom)


class Interactive_STM_Simulator(Unified_STM_Simulator):
    """
    The simulator's window: one qtgui figure of DESIGN size whose panels are cells of its Qt
    layout and whose controls are Qt widgets, made once per figure (_make_controls) and arranged
    per view (_build_ui): the window's title (the spin mode), a left column with the view radio,
    the broadening box, the boxes other windows add (add_control_box: the decomposition wrapper's
    σ box and its 'Send to decomp' buttons) and the RUN button with the toggles; the panel area
    of the view (Single Point: Global Topo | Topo h | Partitioned LDOS; Line / Path: the top row
    Global Topo | Topo h | LDOS h | the cluster (Topo box, stripe, one heatmap per partition,
    colour bar) over the tip-height profile and the spectrum; Map: Global Topo | Topo h | LDOS h
    | Map Topo | Partitioned LDOS over the rows of energy maps); the bottom strip of number boxes
    (Cells, Path Pts, Markers, E Pts, E Min, E Max, LDOS γ).  Every slider of the old figure is a
    number box (the arrow keys step its digits); the orbital hover labels are widgets.HoverLabels.
    The physics and the caches are Unified_STM_Simulator's, called as before.
    """

    def __init__(self, filepath, erange, ldos_height, cmap_topo, unit_cell_num=1, cmap_map='nipy_spectral'):
        super().__init__(filepath, unit_cell_num=unit_cell_num)
        self.parse_vasp_outputs("LOCPOT")
        self.p1, self.p2 = None, None
        self.erange, self.ldos_height, self.cmap_topo = list(erange), ldos_height, cmap_topo
        # The Map view's energy maps' colour map (as the sm4_Browser's cmap_map: any matplotlib colour-map
        # name or Colormap), and its 'cmap norm': off, every map on one colour scale, the maximum over all
        # of them, with one colour bar; on, each map its own scale with its own bar beside it.
        self.cmap_map = cmap_map
        self.cmap_norm = False
        self.npts = 72
        self.is_running, self.normalize, self.show_mag = False, False, False
        self.show_atoms, self.show_unit_cell = True, False
        self.use_decay = True
        self.t2_clamp = True     # occupied-side 0-clamp within the decay model (on by default)
        self.reuse_cache = True
        self.show_decay_toggle = True
        self.display_cells = 1
        self.mode = 'Single Point'
        # 'Dcmp Norm' checkbox removed from the UI; the per-panel normalization
        # logic it controlled is retained throughout the display code and
        # hard-set off here. Restore a control for this flag to re-enable it.
        self.show_dcmp_norm = False
        # PowerNorm gamma for all LDOS heatmaps (1.0 = linear; <1 squashes the
        # colormap toward the minimum; >1 compresses the low end). Live-controlled
        # by the bottom-row slider.
        self.ldos_gamma = 1.0

        self.m_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1', '#e377c2',
                         '#17becf', '#bcbd22', '#7f7f7f', '#8c564b', '#d62728']
        self.marker_ratios = [0.25, 0.75]
        self.marker_coords = [[self.lv[0, 0] * 0.2, self.lv[1, 1] * 0.2],
                              [self.lv[0, 0] * 0.8, self.lv[1, 1] * 0.8]]
        self.active_obj = None
        self.active_marker_idx = 0
        self.plot_level = 0
        self.active_element, self.active_atom = None, None
        self._type_color_map = {'Au': 'orange', 'N': 'blue', 'C': 'brown', 'H': 'grey'}
        self.extra_broadening = {t: 0.0 for t in self.atomtypes}
        self._broadening_topo_dirty = False

        self.cached_p1, self.cached_p2 = None, None
        self.cached_emin, self.cached_emax = None, None
        self.cached_d_topo_line, self.cached_d_topo_map, self.cached_d_ldos = None, None, None
        self.cached_bias_energy_line, self.cached_bias_energy_map, self.cached_nepts = None, None, None
        self.cached_ld_up, self.cached_ld_dn, self.cached_eg = None, None, None
        self.cached_marker_coords, self.cached_spec_ldos = None, None

        # --- Path mode state ---
        self.path = None                  # initial seed: all atom-index lists [[154],...] OR all [x,y] coords; no mix
        self.path_extend = False          # toggle: extend ends to first local cell wall
        self.path_nodes = None            # live list of N draggable [x, y] node positions
        self.cached_path_nodes = None     # cache-invalidation key for path topo
        self.cached_path_extend = None
        self.fig = None                   # the window (run_interactive); its controls are made once (_make_controls)
        self.cell_rotated = False         # the unit-cell panels turned 90 degrees (Axes.set_rotated): _choose_orientation
        self._orient_pending = False      # an orientation decision waiting for the arrangement's first size
        self._extra_boxes = []            # control boxes another window adds to the left column (add_control_box)

    def run_interactive(self, grid_res=64, topo_bias=0.2, topo_height=2.5,
                        ldos_bias_sign='neg', use_decay=True,
                        show_decay_toggle=True, reuse_cache=True,
                        line_endpoints=None, marker_positions=None, extra_broadening=None,
                        path=None, path_extend=False):
        if topo_bias is None or topo_bias == 0:
            raise ValueError(f"topo_bias must be a nonzero bias voltage in volts; received {topo_bias!r}.")
        self.ldos_bias_sign = ldos_bias_sign
        self.use_decay = use_decay
        self.reuse_cache = reuse_cache
        self.show_decay_toggle = bool(show_decay_toggle)
        if line_endpoints is not None:
            self.p1, self.p2 = np.array(line_endpoints[0], dtype=float), np.array(line_endpoints[1], dtype=float)
        elif self.p1 is None:
            self.p1, self.p2 = np.array([0.0, 0.0]), self.lv[0, :2] + self.lv[1, :2]

        if marker_positions is not None:
            self.marker_coords = [list(pt) for pt in marker_positions]
            v = self.p2 - self.p1
            v_sq = np.dot(v, v)
            if v_sq > 1e-9:
                self.marker_ratios = [np.clip(np.dot(np.array(pt) - self.p1, v) / v_sq, 0, 1) for pt in marker_positions]
            else:
                self.marker_ratios = list(np.linspace(0.1, 0.9, len(marker_positions)))
        self.global_topo_bias = topo_bias
        self.topo_height = topo_height

        # --- Path mode seeding ---
        self.path_extend = path_extend
        if path is not None:
            self.path = path
            lengths = {len(p) for p in path}
            if lengths == {1}:
                # Atom numbers (1-based): look up each atom's xy from self.coord
                self.path_nodes = [self.coord[p[0] - 1, :2].astype(float).copy() for p in path]
            elif lengths == {2}:
                # Cartesian [x, y] coordinates: use directly
                self.path_nodes = [np.asarray(p, dtype=float).copy() for p in path]
            else:
                raise ValueError(
                    "path must be either all atom-number elements like [[154],[223],...] "
                    "or all coordinate elements like [[x,y],[x,y],...] — not mixed."
                )
        elif self.path_nodes is None:
            # Default 3-node path so Path mode is usable even if no seed supplied
            self.path_nodes = [np.array([0.0, 0.0]),
                               0.5 * (self.lv[0, :2] + self.lv[1, :2]),
                               self.lv[0, :2] + self.lv[1, :2]]

        # --- Tip-height slider range: 0 .. (cell-top Z minus highest atom Z) ---
        self.tip_height_max = float(self.lv[2, 2] - self.z_highest_atom)

        # Initialize extra broadening
        self.extra_broadening = {t: 0.0 for t in self.atomtypes}
        if extra_broadening is not None:
            for t, sigma in extra_broadening.items():
                if t in self.extra_broadening:
                    self.extra_broadening[t] = sigma
        self._apply_extra_broadening()

        print("\n--- Phase 1: Global Topography Pre-Calculation ---")
        grid_xy = (np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))[0].ravel()[:, None] * self.lv[0, :2]) + \
                  (np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))[1].ravel()[:, None] * self.lv[1, :2])
        grid_xy_gpu = cp.array(grid_xy, dtype=cp.float32)
        z_fixed = cp.full(grid_xy_gpu.shape[0], self.z_highest_atom + topo_height, dtype=cp.float32)
        t_emin, t_emax = sorted([0.0, topo_bias])
        cache_name = f"global_topo_{topo_bias}V_{topo_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}.npy"
        if self.reuse_cache and exists(cache_name):
            print(f"[*] Loading Cached Global Topography: {cache_name}")
            self.current_z_map = np.load(cache_name)
            z_map_gpu = cp.array(self.current_z_map, dtype=cp.float32)
        else:
            print(f"[*] Cache not found. Calculating Global Topography: {cache_name}")
            ld_1, _, init_engs = self._calculate_ldos_at_points_gpu(
                cp.hstack([grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax,
                use_energy_decay=self.use_decay, preserve_orbitals=False, topo_only=True,
                clamp_occupied=self.t2_clamp)
            target_setp = cp.max(gpu_simpson(ld_1, init_engs))
            print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
            z_map_gpu = self._converge_tip_height(z_fixed, grid_xy_gpu, t_emin, t_emax, target_setp,
                                                  use_decay=self.use_decay, clamp_occupied=self.t2_clamp)
            self.current_z_map = cp.asnumpy(z_map_gpu)
            np.save(cache_name, self.current_z_map)
        self.grid_xy = grid_xy
        self.global_z_map = self.current_z_map.copy()
        self.grid_xy_gpu = grid_xy_gpu
        self.fig = plt.figure(figsize=DESIGN, fill_screen=True)   # opened on the screen; the panes then size it (rule 11)
        self._make_controls()
        self._build_ui()
        plt.show()

    def _get_partitions(self, f_ldos_raw):
        partitions = []
        if f_ldos_raw is not None:
            if self.plot_level == 0:
                partitions.append(("Total", np.sum(f_ldos_raw, axis=(2, 3))))
            elif self.plot_level == 1:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
                mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
                au_data = np.sum(f_ldos_raw[:, :, au_idx, :], axis=(2, 3)) if au_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                mol_data = np.sum(f_ldos_raw[:, :, mol_idx, :], axis=(2, 3)) if mol_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                partitions.extend([("Au", au_data), ("Molecule", mol_data)])
            elif self.plot_level == 2:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                for t in self.atomtypes:
                    t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                    t_data = np.sum(f_ldos_raw[:, :, t_idx, :], axis=(2, 3)) if t_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                    partitions.append((t, t_data))
            elif self.plot_level == 3:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                if getattr(self, 'active_element', None) is not None:
                    e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
                    for col_idx, orb in enumerate(self.orbitals):
                        if e_idx:
                            D = cp.sum(self.dos_up_gpu[e_idx, :, col_idx], axis=0)
                            dD = cp.gradient(D)
                            d2D = cp.gradient(dD)
                            if not (float(cp.max(cp.abs(D))) < 1e-5 and float(cp.max(cp.abs(dD))) < 1e-5 and float(cp.max(cp.abs(d2D))) < 1e-5):
                                orb_data = np.sum(f_ldos_raw[:, :, e_idx, col_idx], axis=2)
                                partitions.append((f"{self.active_element} {orb}", orb_data))
            elif self.plot_level == 4:
                atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                if getattr(self, 'active_element', None) is not None:
                    e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
                    for col_idx, orb in enumerate(self.orbitals):
                        if e_idx:
                            D = cp.sum(self.dos_up_gpu[e_idx, :, col_idx], axis=0)
                            dD = cp.gradient(D)
                            d2D = cp.gradient(dD)
                            if not (float(cp.max(cp.abs(D))) < 1e-5 and float(cp.max(cp.abs(dD))) < 1e-5 and float(cp.max(cp.abs(d2D))) < 1e-5):
                                orb_data = np.sum(f_ldos_raw[:, :, e_idx, col_idx], axis=2)
                                partitions.append((f"{self.active_element} {orb}", orb_data))
        return partitions

    def _build_path_xy(self):
        """Build resampled path sample points from the current draggable nodes.

        Reuses only the path logic: indices->XY (done at seed time), optional
        directional end-extension to the first local unit-cell wall, and
        arc-length resampling to self.npts uniform points.

        Returns (p_xy, p_dist, p_len, verts).
        """
        verts = [np.asarray(n, dtype=float) for n in self.path_nodes]

        if self.path_extend and len(verts) >= 2:
            inv2 = inv(self.lv)[:2, :2]   # 2x2 XY fractional transform

            def _wall_crossing(p0, d):
                # smallest t>0 where ray p0 + t*d crosses a local cell wall
                f0 = p0 @ inv2
                fd = d @ inv2
                ts = []
                for k in range(2):
                    if abs(fd[k]) > 1e-12:
                        target = np.floor(f0[k]) + (1.0 if fd[k] > 0 else 0.0)
                        if np.isclose(f0[k], target) and fd[k] < 0:
                            target -= 1.0
                        t = (target - f0[k]) / fd[k]
                        if t > 1e-9:
                            ts.append(t)
                if not ts:
                    return None
                return p0 + min(ts) * d

            start_ext = _wall_crossing(verts[0], verts[0] - verts[1])
            end_ext = _wall_crossing(verts[-1], verts[-1] - verts[-2])
            if start_ext is not None:
                verts = [start_ext] + verts
            if end_ext is not None:
                verts = verts + [end_ext]

        verts = np.array(verts)
        seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        p_len = float(cum[-1]) if cum[-1] > 1e-12 else 1e-12
        samp = np.linspace(0, p_len, self.npts)
        px = np.interp(samp, cum, verts[:, 0])
        py = np.interp(samp, cum, verts[:, 1])
        p_xy = np.column_stack([px, py])
        p_dist = np.linspace(0, p_len, self.npts)
        return p_xy, p_dist, p_len, verts

    # ------------------------------------------------------------------
    # The window: controls made once, the panels of a view arranged per mode
    # ------------------------------------------------------------------
    def _make_controls(self):
        """
        Every control of the window, made once per figure (a window owns its state; the controls
        only report): the view radio, RUN, the toggles, the broadening box, the tip-height boxes
        with their Rerun buttons, the bottom strip's boxes, the cluster's label; the pick / drag
        handlers connected once.  The window's layout is installed here; _build_ui fills the
        panel area per view and shows the controls of the view.
        """
        fig = self.fig
        # automatic layout: the one rule for every window (nothing placed by hand).
        self.ui_radio = controls.make(fig, 'mode', value=self.mode)
        self.ui_radio.on_change(self._on_mode_change)
        self.btn_run = controls.make(fig, 'RUN')
        self.btn_run.set_text('STOP' if self.is_running else 'RUN')
        self.btn_run.on_change(self._toggle_run)
        # One list defines every toggle: label and the attribute it drives. '0-clamp' sits
        # indented under 'Decay' as its sub-option (the occupied-side 0-clamp of the decay model,
        # on by default); it stays visible whether or not Decay is on.
        toggle_defs = [('Atoms', 'show_atoms'), ('Decay', 'use_decay'),
                       ('0-clamp', 't2_clamp'),
                       ('Norm', 'normalize'), ('Mag', 'show_mag'),
                       ('Cell', 'show_unit_cell'), ('Extend', 'path_extend')]
        if not getattr(self, 'show_decay_toggle', True):
            toggle_defs = [d for d in toggle_defs if d[0] not in ('Decay', '0-clamp')]
        self._toggle_defs = toggle_defs
        labels = [label for label, _ in toggle_defs]
        self.chk = controls.make(fig, 'sim toggles', labels=labels,
                                 values=[getattr(self, attr) for _, attr in toggle_defs],
                                 indents=['    ' if label == '0-clamp' else 0 for label in labels])
        self.chk.on_change(self._on_ui_change)
        # The bottom strip's boxes (the counts and values that were sliders).
        self.s_cell = controls.make(fig, 'Cells', value=self.display_cells)
        self.s_path_pts = controls.make(fig, 'Path Pts', value=len(self.path_nodes))
        self.s_num_marks = controls.make(fig, 'Markers', value=len(self.marker_coords))
        self.s_nepts = controls.make(fig, 'E Pts', value=5 if self.cached_nepts is None else self.cached_nepts)
        self.s_emin = controls.make(fig, 'E Min', value=self.erange[0])
        self.s_emax = controls.make(fig, 'E Max', value=self.erange[1])
        # LDOS heatmap colour-map gamma (PowerNorm): 1.0 = linear; <1 squashes the colour map
        # toward the minimum; >1 compresses the low end. Only re-colours (no recompute).
        self.s_ldos_gamma = controls.GammaBox(fig, self.ldos_gamma, name='LDOS γ', fmt='.2f')
        for c in (self.s_cell, self.s_path_pts, self.s_num_marks, self.s_nepts, self.s_emin, self.s_emax, self.s_ldos_gamma):
            c.on_change(self._on_ui_change)
        # 'cmap norm' (Map): each energy map its own colour scale and bar, or one scale and bar for all.
        self.chk_cmap_norm = controls.make(fig, 'cmap norm', value=self.cmap_norm)
        self.chk_cmap_norm.on_change(self._on_cmap_norm)
        # Extra broadening: one check per atom type, its σ box shown only while the check is on.
        broad_chk_init = [self.extra_broadening.get(t, 0.0) > 0 for t in self.atomtypes]
        self.chk_broad = controls.make(fig, 'broadening', labels=list(self.atomtypes), values=broad_chk_init)
        self.chk_broad.on_change(self._on_broadening_change)
        self.s_broad = {}
        for i, t in enumerate(self.atomtypes):
            box = controls.make(fig, 'σ broad', value=self.extra_broadening.get(t, 0.0), label=f'σ {t}')
            box.on_change(self._on_broadening_change)
            box.show(broad_chk_init[i])
            self.s_broad[t] = box
        # The tip heights: a Value box (0 .. the cell top above the highest atom) with its Rerun button, one column each.
        self.s_topo_h = controls.make(fig, 'Topo h', value=self.topo_height, vmax=self.tip_height_max)
        self.btn_topo_rerun = controls.make(fig, 'Rerun')
        self.btn_topo_rerun.on_change(self._on_topo_height_rerun)
        self.s_ldos_h = controls.make(fig, 'LDOS h', value=self.ldos_height, vmax=self.tip_height_max)
        self.btn_ldos_rerun = controls.make(fig, 'Rerun')
        self.btn_ldos_rerun.on_change(self._on_ldos_height_rerun)
        self._box_topo_h = controls.box(self.s_topo_h, self.btn_topo_rerun, orient='v', title='Topo h')
        self._box_ldos_h = controls.box(self.s_ldos_h, self.btn_ldos_rerun, orient='v', title='LDOS h')
        # The crop of the placement topo (the topo with the atom overlays where the markers, line ends
        # and path nodes are placed), as the sm4_Browser Line window crops its heatmap: two draggable
        # cursors per axis, the area outside each pair shaded, a box riding with each cursor in a band
        # above (x) and right of (y) the panel, 'Crop' applying the ranges between the cursors as the
        # panel's limits and 'Uncrop' restoring the full extent.  A crop changes the panel's shape, so
        # the arrangement and the window follow it (rule 11).
        self.btn_crop = controls.make(fig, 'Crop')
        self.btn_crop.on_change(self._on_crop)
        self.btn_uncrop = controls.make(fig, 'Uncrop')
        self.btn_uncrop.on_change(self._on_uncrop)
        self._box_crop = controls.box(self.btn_crop, self.btn_uncrop, orient='v', title='crop')
        self._cursor_boxes = {}
        for axis in ('x', 'y'):
            for k, part in enumerate(('lo', 'hi')):
                box = controls.make(fig, f'{axis} {part}', value='')
                box.on_change(lambda text, axis=axis, k=k: self._on_cursor_box(axis, k, text))
                self._cursor_boxes[(axis, k)] = box
        self.crop_x = self.crop_y = None           # the applied crop, (lo, hi) Å per axis; None = the full extent of the drawn cells
        self._cursors = None                       # {'x': [lo, hi], 'y': [lo, hi]}, set at the first draw
        self._cursor_artists, self._shade_artists = [], []
        self._crop_ax = None                       # the placement topo's axes (the cursors live on it)
        self._crop_panel = None                    # its CropPanel (the bands the boxes ride in)
        # The label over the Line / Path cluster and the Map rows ('LDOS', or 'LDOS: <element>' at the atom levels).
        self._cluster_label = controls.Label(fig, text='LDOS', fs=10)

        # The window's layout: the title cell, the left control column beside the panel area, the
        # bottom strip (automatic layout: the one rule for every window; nothing placed by hand).
        self._title_cell = fig.cell(min_px=(40, 24), stretch=(True, False))
        fig.set_title_cell(self._title_cell)
        fig.suptitle(self.spin_mode.name, fontsize=12)
        self._hook_layout = QtWidgets.QVBoxLayout()        # the boxes other windows add (add_control_box)
        self._hook_layout.setContentsMargins(0, 0, 0, 0)
        self._hook_layout.setSpacing(4)
        self._left_column = controls.column(
            controls.box(self.ui_radio),
            controls.box(self.chk_broad, controls.column(*self.s_broad.values()), title='broadening'),
            self._hook_layout,
            'stretch',
            controls.box(self.btn_run, self.chk))
        left_w = controls.widget_of(self._left_column)
        left_w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        self._panel_area = QtWidgets.QVBoxLayout()         # the view's arrangement (controls.Arrangement; _build_ui fills it)
        self._panel_area.setContentsMargins(0, 0, 0, 0)
        self._panel_area.setSpacing(6)
        main = QtWidgets.QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(8)
        main.addWidget(left_w)
        main.addLayout(self._panel_area, 1)
        strip = controls.FlowLayout()
        self._strip_boxes = {}
        for c, name in ((self.s_cell, 'Cells'), (self.s_path_pts, 'Path Pts'), (self.s_num_marks, 'Markers'),
                        (self.s_nepts, 'E Pts'), (self.s_emin, 'E Min'), (self.s_emax, 'E Max'),
                        (self.s_ldos_gamma, 'LDOS γ'), (self.chk_cmap_norm, 'cmap norm')):
            frame = controls.box(c)
            self._strip_boxes[name] = frame
            strip.addWidget(frame)
        root = QtWidgets.QVBoxLayout()
        root.setSpacing(6)
        root.addWidget(self._title_cell)
        root.addLayout(main, 1)
        root.addLayout(strip)
        fig.set_layout(root)
        for w in self._extra_boxes:
            self._hook_layout.addWidget(w)
        self._panel_cells = []
        self._arrangement = None                           # the view's controls.Arrangement (the shaped groups and the curve panels)
        self._cluster_row = None                           # Line / Path: the heatmap row's PanelRow (rebuilt per refresh)
        self._map_block = None                             # Map: the energy maps' wrapping block (controls.MapBlock)
        self._spec_cell = None                             # Map: the spectrum's cell, one cell of the block
        self._map_cells = []                               # Map: the cells in the block (rebuilt when the counts change)
        self.line_decomp_axes = []
        self.map_axes = []
        self.cax_list = []
        # The pick / drag handlers, connected once per figure.
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)

    def add_control_box(self, widget):
        """
        A control box of another window placed in this window's left column, under the
        broadening box and above RUN (the decomposition wrapper's σ box and its 'Send to decomp'
        buttons); kept across view changes (the left column is not rebuilt).
        """
        self._extra_boxes.append(widget)
        if getattr(self, '_hook_layout', None) is not None:
            self._hook_layout.addWidget(widget)
            widget.setVisible(True)

    def _cell_offsets(self, n):
        """The in-plane offsets of the periodic images drawn, -n .. n cells each way."""
        return [tuple(i * self.lv[0, :2] + j * self.lv[1, :2]) for i in range(-n, n + 1) for j in range(-n, n + 1)]

    def _panel(self, layout, stretch=1, inner_w=None):
        """A panel in a row or column layout: a new cell added with `stretch`, its axes returned (automatic layout)."""
        cell = self.fig.cell(inner_w=inner_w)
        layout.addWidget(cell, 0 if inner_w is not None else stretch)
        self._panel_cells.append(cell)
        return self.fig.add_axes(cell=cell)

    def _pair(self, box, inner_w=None):
        """
        A new panel with the control box `box` above it (controls.PanelWithBox: the box belongs to
        the panel and is sized with it, never between panels; rule 11 of the guide): (axes, pair).
        A narrow cell (inner_w: the Path stripe) makes the pair a fixed-width item of its row.
        """
        cell = self.fig.cell(inner_w=inner_w)
        self._panel_cells.append(cell)
        ax = self.fig.add_axes(cell=cell)
        pair = controls.PanelWithBox(ax, cell, box)
        box.show()                                 # detached (hidden) since the last build
        return ax, pair

    def _panel_with_box(self, layout, box, stretch=1, inner_w=None):
        """A panel with its control box above it (_pair) added to a row with `stretch`, its axes returned."""
        ax, pair = self._pair(box, inner_w=inner_w)
        layout.addWidget(pair, 0 if inner_w is not None else stretch)
        return ax

    def _placement(self):
        """
        The placement topo in its crop panel (controls.CropPanel): the Topo h box and the Crop /
        Uncrop buttons above it, the x-cursor boxes riding in the band under them, the y-cursor
        boxes in the band at its right; (axes, panel).  The boxes follow the panel wherever it is
        placed (Axes.on_placed).
        """
        cell = self.fig.cell()
        self._panel_cells.append(cell)
        ax = self.fig.add_axes(cell=cell)
        panel = controls.CropPanel(ax, cell, self._box_topo_h, self._box_crop)
        self._box_topo_h.show()
        self._box_crop.show()
        ax.set_rotated(self.cell_rotated)
        for (axis, k), box in self._cursor_boxes.items():
            # Each box in the band its cursor's line runs along: as drawn the x boxes over the panel, the y
            # boxes at its right; the other way round when the panel is turned.
            (panel.band_y if (axis == 'x') == self.cell_rotated else panel.band_x).add((axis, k), box)
        ax.on_placed(self._place_cursor_boxes)
        self._crop_ax, self._crop_panel = ax, panel
        return ax, panel

    def _cell_axes(self):
        """A free panel on its own (the curve panels the arrangement gives the remaining room): (axes, cell)."""
        cell = self.fig.cell()
        self._panel_cells.append(cell)
        return self.fig.add_axes(cell=cell), cell

    @staticmethod
    def _detach(*widgets):
        """The reused widgets (the tip-height boxes, the cluster's label) taken out of the pair or layout about to be deleted: hidden and parentless until the next build puts them back and shows them."""
        for w in widgets:
            if w is not None:
                w.hide()
                w.setParent(None)

    @staticmethod
    def _empty_layout(lay, keep=()):
        """Every item of a layout taken out: the widgets in `keep` detached for reuse, the others deleted, sub-layouts emptied too."""
        if lay is None:
            return
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            sub = it.layout()
            if w is not None:
                w.setParent(None)
                if w not in keep:
                    w.deleteLater()
            elif sub is not None:
                Interactive_STM_Simulator._empty_layout(sub, keep)
                sub.deleteLater()

    def _kept_widgets(self):
        return (self._box_topo_h, self._box_ldos_h, self._cluster_label.qt, self._box_crop) + tuple(b.qt for b in self._cursor_boxes.values())

    def _clear_cluster(self):
        """The Line / Path cluster's axes removed and its row emptied (it is rebuilt per refresh); the LDOS h box detached from its pair first."""
        for ax in list(self.line_decomp_axes):
            if ax in self.fig.axes:
                ax.remove()
        self.line_decomp_axes = []
        for name in ('ax_ldos', 'ax_stripe', 'ax_line_topo', 'cax'):
            ax = getattr(self, name, None)
            if ax is not None and ax in self.fig.axes:
                ax.remove()
            setattr(self, name, None)
        self._detach(self._box_ldos_h)
        self._empty_layout(self._cluster_row, self._kept_widgets())

    def _clear_map_rows(self):
        """The Map rows' axes removed and their cells deleted out of the block (they are rebuilt when the counts change)."""
        for ax in list(self.map_axes) + list(getattr(self, 'cax_list', None) or []):
            if ax in self.fig.axes:
                ax.remove()
        self.map_axes = []
        self.cax_list = []
        for c in self._map_cells:
            c.setParent(None)
            c.deleteLater()
        self._map_cells = []
        if self._map_block is not None:
            self._map_block.set_groups([], None, extra=getattr(self, '_spec_cell', None))

    def _clear_panels(self):
        """Every panel of the view removed: its axes, its cells, the panes of the panel area; the reused boxes and label detached first."""
        self._clear_cluster()
        self._clear_map_rows()
        for ax in list(self.fig.axes):
            ax.remove()
        self._detach(*self._kept_widgets())
        self._empty_layout(self._panel_area, self._kept_widgets())
        self._arrangement = None
        self._spec_cell = None
        self._crop_ax = self._crop_panel = None
        self._cursor_artists, self._shade_artists = [], []
        self._panel_cells = []
        self._cluster_row = None
        self._map_block = None
        self._map_cells = []
        if getattr(self, 'cursor', None) is not None:
            self.cursor.remove()
            self.cursor = None

    def _build_ui(self):
        """
        The panels of the current view arranged by the one rule (controls.Arrangement; rule 11 of
        the guide as the owner stated it: the plots fill the window, spatial panels keep their
        ratio, the heat-map data comes first, a curve panel never grows at its expense yet stays
        readable, blank space and the controls' footprint are what is minimized).  The shaped groups of a
        view: the placement topo in its crop panel (Topo h, Crop / Uncrop and the cursor boxes
        above and beside it); in Line / Path the heatmap row (its label over it); in Map the
        Global Topo and Map Topo pair (their boxes above) and the energy maps' wrapping block.  The
        curve panels (Tip Height and Partitioned LDOS, beside or over each other by their room)
        take the room the heatmaps and topographies leave, at least their readable size
        (_curves_readable), the topographies at most twice a heatmap (_topo_cap); in Map the
        spectrum is a cell of the map block, a map's size; in Single Point it is a first-class
        panel beside the placement topo.
        The arrangement is chosen
        again and the window trimmed (Figure.fit_window) on every view change, rebuild, crop and
        resize; the unit-cell panels' orientation (as drawn, or turned 90 degrees: the way the
        window fits better, _choose_orientation) is decided when the view is built and on Crop /
        Uncrop.  The controls of the view are shown and the others hidden, then
        _update_all(full_refresh=True).  Called at the first show and on every view change; the
        controls themselves are made once (_make_controls).  automatic layout: the one rule for
        every window (nothing placed by hand).
        """
        fig = self.fig
        self._clear_panels()
        mode = self.mode
        self._strip_boxes['Path Pts'].setVisible(mode == 'Path')
        self._strip_boxes['E Pts'].setVisible(mode == 'Map')
        self._strip_boxes['cmap norm'].setVisible(mode == 'Map')
        # The counts the view reads, put in the boxes without firing.
        self.s_num_marks.set(len(self.marker_ratios) if mode in ('Line', 'Path') else len(self.marker_coords))
        if mode == 'Path':
            self.s_path_pts.set(len(self.path_nodes))
        if mode == 'Map':
            self.s_nepts.set(5 if self.cached_nepts is None else self.cached_nepts)
        if mode == 'Single Point':
            # The placement topo; the Partitioned LDOS beside it, a first-class panel (free_scored).
            self.ax_map, topo = self._placement()
            self.ax_spec, spec_cell = self._cell_axes()
            groups, free = [topo], spec_cell
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
        elif mode in ('Line', 'Path'):
            # The placement topo; the heatmap row (rebuilt by _update_all each refresh, the LDOS h box
            # above the Line Topo panel or above the Path stripe) under its label; the Tip Height
            # profile and the spectrum in the room left (at least readable: rule 11), side by side or stacked by that room's shape.
            self.ax_map, topo = self._placement()
            self._cluster_row = plt.PanelRow(spacing=4)
            cluster = controls.Titled(plt.RowWidget(self._cluster_row), self._cluster_label.qt)
            self.ax_prof, prof_cell = self._cell_axes()
            self.ax_spec, spec_cell = self._cell_axes()
            groups, free = [topo, cluster], controls.FlipBox((prof_cell, 25), (spec_cell, 10), spacing=10)
            # Until RUN builds the row, the LDOS h box stands in it alone: reachable, and between no panels.
            self._cluster_row.addWidget(self._box_ldos_h, 0)
            self.line_art, = self.ax_map.plot([], [], 'r--', lw=2.5, zorder=5)
            if mode == 'Line':
                self.ends = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5)
            else:  # Path: draggable nodes instead of two endpoints
                self.path_pts = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5)
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.ax_line_topo = self.ax_stripe = self.ax_ldos = self.cax = None
        elif mode == 'Map':
            # The Global Topo (the crop panel) and the Map Topo (LDOS h above it) side by side as one
            # shaped row; the energy maps as a wrapping block under its label (rebuilt by _update_all
            # when the counts change) with the spectrum as one of its cells (rule 11).  The topo panels keep
            # set_aspect('equal') and take the same crop.
            row = plt.PanelRow(spacing=10)
            self.ax_map_global, topo = self._placement()
            row.addWidget(topo, 10)
            self.ax_map = self._panel_with_box(row, self._box_ldos_h, stretch=10)
            self.ax_map.set_rotated(self.cell_rotated)
            topos = plt.RowWidget(row)
            self._map_block = controls.MapBlock(fig, spacing=4)
            maps = controls.Titled(self._map_block, self._cluster_label.qt)
            # The spectrum is one cell of the map block, a map's size, in the free slot of the block's last
            # row (or a row of its own): never a strip below the maps, never squeezed (rule 11).
            self.ax_spec, spec_cell = self._cell_axes()
            self._spec_cell = spec_cell
            self._map_block.set_groups([], None, extra=spec_cell)
            groups, free = [topos, maps], None
            self.map_axes = []
            self.cax_list = []
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            fig.align_row(self.ax_map_global, self.ax_map)
        # The sizing (rule 11): the heatmaps and the topographies are sized first and fill the window (the
        # topographies at most twice a heatmap, in width and in height); the curve panels, at least readable,
        # take the room those leave (Line / Path: the Tip Height profile and the spectrum in the room beside
        # or under the heatmap row; Map: the spectrum a cell of the map block); in Single Point the spectrum
        # is a first-class panel beside the placement topo (Arrangement free_min / free_scored / group_cap).
        self._arrangement = controls.Arrangement(groups, free, spacing=8, free_min=self._curves_readable,
                                                 free_scored=mode == 'Single Point',
                                                 group_cap=self._topo_cap(mode, groups))
        self._arrangement.on_arranged(self._on_arranged)
        self._panel_area.addWidget(self._arrangement, 1)
        self._box_topo_h.setVisible(True)
        self._box_ldos_h.setVisible(mode != 'Single Point')
        self._cluster_label.show(mode != 'Single Point')
        self._cluster_label.set_text('LDOS')

        self.cached_emin, self.cached_emax = None, None
        self._update_all(full_refresh=True)
        self._rearrange(orient=True)             # the view built: the unit-cell panels' orientation decided

    CURVE_BOX_MIN_PX = (220, 140)      # a curve panel's readable plot box at text zoom 1 (its labels come on top)

    def _curve_axes(self):
        """The curve panels of the view: the spectrum, with the Tip Height profile before it in Line / Path."""
        axs = [getattr(self, 'ax_prof', None)] if self.mode in ('Line', 'Path') else []
        return [a for a in axs + [getattr(self, 'ax_spec', None)] if a is not None]

    def _curves_readable(self):
        """
        The curve panels' readable minimum (rule 11; the Arrangement's free_min): every curve
        panel's cell is given, as its minimum size, the readable plot box (CURVE_BOX_MIN_PX at the
        window's text zoom) plus its own labels (Axes._decor_px), so that a split between two
        panels never squeezes one below it; returns the free widget's minimum (w, h): the panels
        side by side.  The panels take the room the heatmaps and topographies leave beyond it.
        """
        z = float(getattr(self.fig, 'font_zoom', 1.0) or 1.0)
        bw, bh = self.CURVE_BOX_MIN_PX
        ws, hs = [], []
        for ax in self._curve_axes():
            l, t, r, b = ax._decor_px()
            w, h = int(round(bw * z + l + r)), int(round(bh * z + t + b))
            if ax._cell is not None and (ax._cell.minimumWidth(), ax._cell.minimumHeight()) != (w, h):
                ax._cell.setMinimumSize(w, h)
            ws.append(w)
            hs.append(h)
        return (sum(ws) + 6 * (len(ws) - 1), max(hs)) if ws else (0, 0)

    TOPO_MAX_FACTOR = 2.0              # a topography's plot box at most this many times a heatmap's (Map, Line, Path)

    def _topo_axes(self):
        """The topography panels of the view: the placement topo (Global Topo in Map) and the Map Topo."""
        if self.mode == 'Map':
            return [a for a in (getattr(self, 'ax_map_global', None), getattr(self, 'ax_map', None)) if a is not None]
        return [a for a in (getattr(self, 'ax_map', None),) if a is not None]

    def _topo_cap(self, mode, groups):
        """
        The Arrangement's group_cap of the view (None in Single Point: no heatmap): the topography
        group's plot box at most TOPO_MAX_FACTOR times one heatmap's, in width and in height (Map:
        one energy map's box, MapBlock.map_box; Line / Path: one heatmap of the row).  A box's
        height in a region is the region's height less the room around the box; its width follows
        from the panel's own shape (Axes._aspect_want: the unit cell's ratio for a topo and a map).
        The room around the placement topo's box: the crop panel's boxes and band above it and the
        axes' title and x axis (fixed quantities, never the window's current geometry, so the
        arrangement does not chase its own last layout).
        """
        if mode not in ('Map', 'Line', 'Path'):
            return None
        topo_group, heat_group = groups[0], groups[1]

        def around():
            panel = getattr(self, '_crop_panel', None)
            if panel is None:
                return 0.0
            l, t, r, b = panel.cell._decor()
            return panel._box_h() + t + b

        def with_width(h, ax):
            ratio = ax._aspect_want() if ax is not None else None
            return (h / ratio if ratio else None, h)

        def box_a(rect):
            return with_width(max(rect[3] - around(), 1.0), getattr(self, '_crop_ax', None))

        if mode == 'Map':
            def box_b(rect):
                return self._map_block.map_box(rect[2], max(rect[3] - heat_group._label_h(), 1.0))
        else:
            def box_b(rect):
                hm = getattr(self, 'line_decomp_axes', None) or []
                if not hm:
                    return None
                hm_tb = max(a._decor_px()[1] + a._decor_px()[3] for a in hm)
                return with_width(max(rect[3] - heat_group._label_h() - hm_tb, 1.0), hm[0])
        return (topo_group, heat_group, self.TOPO_MAX_FACTOR, box_a, box_b)

    def _rearrange(self, orient=False):
        """The arrangement chosen again for the panels as they now are, and the window trimmed to them (rule 11); with `orient` the unit-cell panels' orientation is decided first (_choose_orientation)."""
        if orient:
            self._choose_orientation()
        if self._arrangement is not None:
            self._arrangement.rearrange()
        self.fig.fit_window()

    # --- the unit-cell panels' orientation (rule 11: turned when the window fits them better that way) ---

    ORIENT_HYSTERESIS = 0.05           # the panels keep their orientation unless the other scores better by this

    def _unit_cell_axes(self):
        """The panels drawn in the unit cell's frame, which turn together: the placement topo, and in Map the Map Topo and every energy map.  The Line / Path heatmap row never turns."""
        axs = [getattr(self, '_crop_ax', None)]
        if self.mode == 'Map':
            axs.append(getattr(self, 'ax_map', None))
            axs.extend(getattr(self, 'map_axes', None) or [])
        figure_axes = self.fig.axes
        return [a for a in axs if a is not None and a in figure_axes]

    def _set_cell_rotation(self, flag):
        """Every unit-cell panel of the view turned 90 degrees (Axes.set_rotated) or back, together; each cursor box moves to the band its cursor's line now runs along."""
        flag = bool(flag)
        for ax in self._unit_cell_axes():
            ax.set_rotated(flag)
        if flag == self.cell_rotated:
            return
        self.cell_rotated = flag
        panel = getattr(self, '_crop_panel', None)
        if panel is not None:
            for key in list(self._cursor_boxes):
                axis = key[0]
                src, dst = (panel.band_x, panel.band_y) if (axis == 'x') == flag else (panel.band_y, panel.band_x)
                w = src.take(key)
                if w is not None:
                    dst.add(key, w)
        self._place_cursor_boxes()

    def _choose_orientation(self):
        """
        The unit-cell panels' orientation: as drawn, or turned 90 degrees, whichever the
        arrangement scores better at the window's size now (controls.Arrangement.best_score: the
        groups' scale less the blank), the current one kept within ORIENT_HYSTERESIS.  Decided
        when a view is built and again on Crop / Uncrop (the shape changes), never on a resize;
        before the arrangement has a size (the first build) the decision waits for its first
        arrangement (_on_arranged).  Returns whether the orientation changed.
        """
        arr = self._arrangement
        if arr is None:
            return False
        cur = self.cell_rotated
        scores = {}
        for flag in (cur, not cur):
            self._set_cell_rotation(flag)
            s = arr.best_score()
            if s is None:
                self._set_cell_rotation(cur)
                self._orient_pending = True
                return False
            scores[flag] = s[0]
        best = cur if scores[cur] >= scores[not cur] - self.ORIENT_HYSTERESIS else (not cur)
        self._set_cell_rotation(best)
        self._orient_pending = False
        print(f"{self.mode}: the unit-cell panels {'turned 90 degrees' if best else 'as drawn'} "
              f"(fit {scores[best]:.3f}; the other way {scores[not best]:.3f}).")
        return best != cur

    def _on_arranged(self, name):
        """After an arrangement: an orientation decision that waited for the arrangement's first size is made now (deferred past the arrangement's own pass; the arrangement then follows it)."""
        if self._orient_pending:
            QtCore.QTimer.singleShot(0, self._orient_now)

    def _orient_now(self):
        if self._orient_pending and self._choose_orientation():
            self._rearrange()

    # --- the crop of the placement topo (as the sm4_Browser Line window's crop cursors) ---

    CURSOR_COLOR = '#303030'
    SHADE_COLOR = '#808080'
    SHADE_ALPHA = 0.45

    def _full_extent(self):
        """((x lo, x hi), (y lo, y hi)) of the drawn cells: the unit cell at every offset of the Cells count."""
        n = int(self.s_cell.val) if getattr(self, 's_cell', None) is not None else 0
        a, b = self.lv[0, :2], self.lv[1, :2]
        corners = np.array([[0.0, 0.0], a, b, a + b])
        pts = np.concatenate([corners + np.asarray(off) for off in self._cell_offsets(n)])
        return (float(pts[:, 0].min()), float(pts[:, 0].max())), (float(pts[:, 1].min()), float(pts[:, 1].max()))

    def _axis_extent(self, axis):
        """The cursor axis's full range as cropped: the applied crop, else the drawn cells' extent."""
        rng = self.crop_x if axis == 'x' else self.crop_y
        if rng is not None:
            return float(rng[0]), float(rng[1])
        ex, ey = self._full_extent()
        return ex if axis == 'x' else ey

    def _reset_cursors(self):
        """Every cursor at its axis's end (the full ranges as cropped)."""
        self._cursors = {'x': list(self._axis_extent('x')), 'y': list(self._axis_extent('y'))}

    def _cursor_value(self, axis, k, v):
        """A value for cursor k (0 the lower, 1 the upper) made valid: inside the axis's extent, the pair keeping 2 % of the extent between them."""
        lo_ext, hi_ext = self._axis_extent(axis)
        v = float(np.clip(float(v), lo_ext, hi_ext))
        other = float(self._cursors[axis][1 - k])
        gap = 0.02 * max(hi_ext - lo_ext, 1e-9)
        v = min(v, other - gap) if k == 0 else max(v, other + gap)
        return float(np.clip(v, lo_ext, hi_ext))

    def _set_cursor(self, axis, k, v):
        """Cursor k of axis 'x' / 'y' moved to v (made valid); the lines, the shades and the boxes follow."""
        if self._cursors is None:
            self._reset_cursors()
        self._cursors[axis][k] = self._cursor_value(axis, k, v)
        self._draw_cursors()

    @staticmethod
    def _cursor_text(v):
        return f"{v:.2f}"

    def _on_cursor_box(self, axis, k, text):
        """A cursor's box: the typed value moves the cursor there (made valid); a text that is not a number is reported and the box put back."""
        try:
            v = float(str(text).strip())
            if not np.isfinite(v):
                raise ValueError("not finite")
        except ValueError:
            print(f"{axis} cursor: '{text}' is not a number; the box is put back.")
            self._place_cursor_boxes()
            return
        self._set_cursor(axis, k, v)

    def _draw_cursors(self):
        """
        The crop cursors on the placement topo: two lines at the x cursors and two at the y
        cursors across the panel (vertical and horizontal as drawn, the other way round when the
        panel is turned; draggable; a press on one is theirs before any marker's),
        the grey shade outside each pair, and the boxes riding with them (_place_cursor_boxes).
        Every cursor is first made valid for the current extent.
        """
        ax = self._crop_ax
        if ax is None or ax not in self.fig.axes:
            return
        if self._cursors is None:
            self._reset_cursors()
        for axis in ('x', 'y'):
            for k in (0, 1):
                self._cursors[axis][k] = self._cursor_value(axis, k, self._cursors[axis][k])
        for a in self._cursor_artists + self._shade_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._cursor_artists, self._shade_artists = [], []
        for axis in ('x', 'y'):
            lo_ext, hi_ext = self._axis_extent(axis)
            lo, hi = self._cursors[axis]
            span = ax.axvspan if axis == 'x' else ax.axhspan
            for a, b in ((lo_ext, lo), (hi, hi_ext)):
                if b > a:
                    self._shade_artists.append(span(a, b, color=self.SHADE_COLOR, alpha=self.SHADE_ALPHA, zorder=1.5))
            line = ax.axvline if axis == 'x' else ax.axhline
            for k, v in enumerate((lo, hi)):
                self._cursor_artists.append(line(v, color=self.CURSOR_COLOR, ls='-', lw=1.8, alpha=0.9,
                                                 label=f'_cursor_{axis}_{k}', picker=6, zorder=14))
        self._place_cursor_boxes()
        self.fig.canvas.draw_idle()

    def _place_cursor_boxes(self):
        """
        The cursor boxes filled and moved onto their cursors (controls.RideBand): each in the band
        its cursor's line runs along (as drawn the x boxes in the band over the panel, the y boxes
        in the band at its right; the other way round when the panel is turned), at the line's
        canvas position (Axes.transData, which knows the turn).
        """
        ax, panel = self._crop_ax, self._crop_panel
        if ax is None or panel is None or self._cursors is None or ax not in self.fig.axes:
            return
        H = self.fig._canvas_size()[1]
        for (axis, k), box in self._cursor_boxes.items():
            v = float(self._cursors[axis][k])
            box.set(self._cursor_text(v))
            px, py = ax.transData.transform([[v, v]])[0]            # the line's canvas position (its own coordinate)
            if (axis == 'x') != self.cell_rotated:                  # the line is vertical on screen: the top band, by x
                panel.band_x.set_pos((axis, k), px)
            else:                                                   # horizontal: the side band, by y from the top
                panel.band_y.set_pos((axis, k), H - py)

    def _crop_ranges(self):
        """(x range, y range) the cursors mean for a crop: a pair left at its axis's ends keeps that axis's crop as it is."""
        out = []
        for axis in ('x', 'y'):
            lo_ext, hi_ext = self._axis_extent(axis)
            lo, hi = self._cursors[axis]
            tol = 1e-9 * max(abs(hi_ext - lo_ext), 1.0)
            full = abs(lo - lo_ext) <= tol and abs(hi - hi_ext) <= tol
            cur = self.crop_x if axis == 'x' else self.crop_y
            out.append(cur if full else (float(lo), float(hi)))
        return out[0], out[1]

    def _apply_crop(self):
        """The placement topo's limits at the applied crop (both axes; the full extent where there is none); in Map the Map Topo takes the same."""
        (ex, ey) = self._full_extent()
        x_rng = self.crop_x if self.crop_x is not None else ex
        y_rng = self.crop_y if self.crop_y is not None else ey
        axes = [self._crop_ax]
        if self.mode == 'Map' and getattr(self, 'ax_map', None) is not None:
            axes.append(self.ax_map)
        for ax in axes:
            if ax is not None and ax in self.fig.axes:
                ax.set_xlim(*x_rng)
                ax.set_ylim(*y_rng)

    def _on_crop(self, event=None):
        """'Crop': what lies between the cursors is kept, on both axes; the cursors then sit at the new edges; the unit-cell panels' orientation is decided again for the new shape, and the arrangement and the window follow."""
        if self._crop_ax is None:
            return
        if self._cursors is None:
            self._reset_cursors()
        self.crop_x, self.crop_y = self._crop_ranges()
        self._apply_crop()
        self._reset_cursors()
        self._draw_cursors()
        self._rearrange(orient=True)

    def _on_uncrop(self, event=None):
        """'Uncrop': the full extent of the drawn cells again, the cursors back at the ends."""
        if self.crop_x is None and self.crop_y is None:
            print("Uncrop: nothing is cropped.")
        self.crop_x = self.crop_y = None
        if self._crop_ax is not None:
            self._apply_crop()
            self._reset_cursors()
            self._draw_cursors()
            self._rearrange(orient=True)

    def _draw_map_slice(self, ax, slice_data, v_max, use_diverging, has_mag):
        """One energy map on `ax`: the slice over the 2 x 2 cells drawn (one raster at four offsets), the colour scale of the view."""
        offs = self._cell_offsets_2x2()
        if use_diverging:
            ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], slice_data, levels=40, cmap='bwr', vmin=-v_max, vmax=v_max, offsets=offs)
        elif self.show_mag and has_mag:
            ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], slice_data, levels=40, cmap='hot', norm=plt.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max), offsets=offs)
        else:
            ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], slice_data, levels=40, cmap=self.cmap_map, norm=plt.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max), offsets=offs)
        ax.set_aspect('equal')

    def _map_cmap(self, use_diverging, has_mag):
        """The energy maps' colour map: 'bwr' for a signed magnetization, 'hot' for |m|, else cmap_map."""
        return 'bwr' if use_diverging else ('hot' if (self.show_mag and has_mag) else self.cmap_map)

    def _map_vmax(self, slice_data, group_vmax):
        """An energy map's colour-scale maximum: its own (the largest |value| of the slice) with 'cmap norm' on, else its group's (the partition's with show_dcmp_norm, the whole set's otherwise); never 0."""
        v = float(np.max(np.abs(slice_data))) if self.cmap_norm else float(group_vmax)
        return v if v > 0 else 1e-15

    def _draw_map_cbar(self, cax, v_max, use_diverging, has_mag, labelsize=None):
        """One colour bar of the energy maps into `cax`: their colour map over 0..v_max (-v_max..v_max for a signed magnetization), the ticks in units of 10^exp with the exponent as the bar's title."""
        cax.clear()
        norm_choice = plt.Normalize(vmin=-v_max, vmax=v_max) if use_diverging else plt.Normalize(vmin=0, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap=self._map_cmap(use_diverging, has_mag), norm=norm_choice)
        sm.set_array([])
        cb = self.fig.colorbar(sm, cax=cax)
        exp = int(np.floor(np.log10(v_max)))
        cb.ax.yaxis.set_major_formatter(plt.ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
        cax.set_title(f"1e{exp}", fontsize=10 if labelsize is None else labelsize)
        if labelsize is not None:
            cax.tick_params(labelsize=labelsize)

    def _on_cmap_norm(self, flag):
        """'cmap norm' (Map): each energy map on its own colour scale with its own colour bar (on), or every map on one scale, the maximum over all of them, with one bar (off); the map block is rebuilt with the bars."""
        self.cmap_norm = bool(flag)
        if self.mode == 'Map' and self.is_running:
            self._update_all(full_refresh=True)

    def _cell_offsets_2x2(self):
        """The offsets of the 2 x 2 cells an energy map shows (the cell and its three neighbours along +a and +b)."""
        return [tuple(nx * self.lv[0, :2] + ny * self.lv[1, :2]) for nx in range(2) for ny in range(2)]

    def _on_mode_change(self, label):
        self.mode = label
        self.active_marker_idx = 0
        self.plot_level = 0
        self._build_ui()

    def _toggle_run(self, event=None):
        self.is_running = not self.is_running
        self.btn_run.set_text('STOP' if self.is_running else 'RUN')
        if self.is_running:
            self._update_all()
        else:
            if self.mode == 'Line' and self.p1 is not None and self.p2 is not None:
                v = self.p2 - self.p1
                m_pos = [self.p1 + r * v for r in self.marker_ratios]
                print("\n--- Line Endpoints & Markers (copy-paste into run_interactive) ---")
                print(f"line_endpoints=([{self.p1[0]:.6f}, {self.p1[1]:.6f}], [{self.p2[0]:.6f}, {self.p2[1]:.6f}]),")
                pos_str = ", ".join(f"[{p[0]:.6f}, {p[1]:.6f}]" for p in m_pos)
                print(f"marker_positions=[{pos_str}]")
            elif self.mode == 'Path' and self.path_nodes is not None:
                p_xy, _, _, _ = self._build_path_xy()
                mark_idx = (np.array(self.marker_ratios) * (self.npts - 1)).astype(int)
                m_pos = p_xy[mark_idx]
                print("\n--- Path Nodes & Markers (current dragged positions) ---")
                node_str = ", ".join(f"[{n[0]:.6f}, {n[1]:.6f}]" for n in self.path_nodes)
                print(f"# path_nodes (free XY, no longer atom indices): [{node_str}]")
                print(f"# path_extend={self.path_extend}")
                pos_str = ", ".join(f"[{p[0]:.6f}, {p[1]:.6f}]" for p in m_pos)
                print(f"marker_positions=[{pos_str}]")
            elif self.mode in ['Single Point', 'Map']:
                print("\n--- Marker Positions (copy-paste into run_interactive) ---")
                pos_str = ", ".join(f"[{p[0]:.6f}, {p[1]:.6f}]" for p in self.marker_coords)
                print(f"marker_positions=[{pos_str}]")

    def _recompute_global_topo(self):
        """Re-run Phase 1 global topography with current broadened DOS (cache-aware)."""
        grid_res = int(np.sqrt(len(self.grid_xy)))
        cache_name = f"global_topo_{self.global_topo_bias}V_{self.topo_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}.npy"
        if self.reuse_cache and exists(cache_name):
            print(f"[*] Loading Cached Global Topography: {cache_name}")
            self.current_z_map = np.load(cache_name)
        else:
            print(f"[*] Calculating Global Topography with broadening: {cache_name}")
            t_emin, t_emax = sorted([0.0, self.global_topo_bias])
            z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.topo_height, dtype=cp.float32)
            ld_1, _, init_engs = self._calculate_ldos_at_points_gpu(
                cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax,
                use_energy_decay=self.use_decay, preserve_orbitals=False, topo_only=True,
                clamp_occupied=self.t2_clamp)
            target_setp = cp.max(gpu_simpson(ld_1, init_engs))
            print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
            z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_emin, t_emax, target_setp,
                                                  use_decay=self.use_decay, clamp_occupied=self.t2_clamp)
            self.current_z_map = cp.asnumpy(z_map_gpu)
            np.save(cache_name, self.current_z_map)
        self.global_z_map = self.current_z_map.copy()

    def _on_topo_height_rerun(self, event=None):
        """Apply the global-topo tip-height box (the 'Topo h' Value box) and recompute the global topo.
        Static-label, single-action button. Does nothing unless the main RUN
        button is active. Height-encoded cache filenames are loaded when
        reuse_cache is True and recomputed and rewritten when reuse_cache is
        False."""
        if not self.is_running:
            return
        self.topo_height = float(self.s_topo_h.val)
        self._recompute_global_topo()
        self._broadening_topo_dirty = False
        self._update_all(full_refresh=True)

    def _on_ldos_height_rerun(self, event=None):
        """Apply the LDOS-topo tip-height box (the 'LDOS h' Value box) and recompute LDOS topo.
        Static-label, single-action button. Does nothing unless the main RUN
        button is active. Resets in-RAM frame caches (forces recompute / reload
        at the new height); disk caches of the same name are loaded when
        reuse_cache is True and recomputed and rewritten when reuse_cache is
        False."""
        if not self.is_running:
            return
        self.ldos_height = float(self.s_ldos_h.val)
        # Invalidate in-RAM caches that depend on LDOS tip height
        self.cached_p1 = self.cached_p2 = None
        self.cached_path_nodes = None
        self.cached_bias_energy_line = self.cached_bias_energy_map = None
        self.cached_d_topo_line = self.cached_d_topo_map = None
        self.cached_emin = self.cached_emax = None
        self.cached_ld_up = self.cached_ld_dn = None
        self.cached_marker_coords = self.cached_spec_ldos = None
        self._update_all(full_refresh=True)

    def _on_broadening_change(self, *args):
        """A broadening check or a σ box changed (the controls report; the window owns the state)."""
        states = self.chk_broad.get_status()
        for i, t in enumerate(self.atomtypes):
            # The σ box of a type is shown only while its check is on.
            self.s_broad[t].show(states[i])
            self.extra_broadening[t] = self.s_broad[t].val if states[i] else 0.0
        self._apply_extra_broadening()
        self.cached_p1 = self.cached_p2 = None
        self.cached_emin = self.cached_emax = None
        self.cached_d_topo_line = self.cached_d_topo_map = None
        self.cached_d_ldos = None
        self.cached_bias_energy_line = self.cached_bias_energy_map = None
        self.cached_marker_coords = self.cached_spec_ldos = None
        self.cached_ld_up = self.cached_ld_dn = None
        self._broadening_topo_dirty = True
        if self.is_running:
            self._recompute_global_topo()
            self._broadening_topo_dirty = False
            self._update_all(full_refresh=True)

    def _on_ui_change(self, *args):
        """A toggle, a count or a value box changed: the state is read from the controls, the caches invalidated as the change asks, and the window redrawn."""
        prev_decay = self.use_decay
        prev_clamp = self.t2_clamp
        prev_path_extend = self.path_extend
        states = self.chk.get_status()
        for (label, attr), state in zip(self._toggle_defs, states):
            setattr(self, attr, state)
        self.display_cells = int(self.s_cell.val)
        self.ldos_gamma = float(self.s_ldos_gamma.value)

        # 0-clamp toggled while Decay is off: it only takes effect once Decay is
        # on, so record the state and do nothing else (no recompute, no redraw).
        clamp_changed = (self.t2_clamp != prev_clamp)
        if clamp_changed and not self.use_decay:
            return

        # Decay toggle, 0-clamp toggle (with Decay on), or path-extend toggle:
        # invalidate caches and recompute.
        decay_model_changed = (self.use_decay != prev_decay) or (clamp_changed and self.use_decay)
        if decay_model_changed or (self.path_extend != prev_path_extend):
            self.cached_p1 = self.cached_p2 = None
            self.cached_path_nodes = None
            self.cached_emin = self.cached_emax = None
            self.cached_d_topo_line = self.cached_d_topo_map = None
            self.cached_d_ldos = None
            self.cached_bias_energy_line = self.cached_bias_energy_map = None
            self.cached_marker_coords = self.cached_spec_ldos = None
            self.cached_ld_up = self.cached_ld_dn = None
            # Only a decay-model change alters the global topo (recompute, or
            # load the model-tagged cache); path-extend does not.
            if decay_model_changed:
                self._broadening_topo_dirty = True
                if self.is_running:
                    self._recompute_global_topo()
                    self._broadening_topo_dirty = False
            if self.is_running:
                self._update_all(full_refresh=True)
            return

        # Path Pts box: adjust number of draggable path nodes.
        if self.mode == 'Path':
            new_nodes = int(self.s_path_pts.val)
            if new_nodes != len(self.path_nodes):
                if new_nodes > len(self.path_nodes):
                    # Append nodes spread along the current path arc length.
                    p_xy, _, _, _ = self._build_path_xy()
                    n_add = new_nodes - len(self.path_nodes)
                    ratios = np.linspace(0.15, 0.85, n_add)
                    for r in ratios:
                        idx = int(r * (self.npts - 1))
                        self.path_nodes.append(np.array(p_xy[idx], dtype=float))
                else:
                    self.path_nodes = self.path_nodes[:new_nodes]
                self.cached_path_nodes = None

        new_count = int(self.s_num_marks.val)
        if self.mode in ('Line', 'Path'):
            if new_count != len(self.marker_ratios):
                if new_count > len(self.marker_ratios):
                    self.marker_ratios = list(np.linspace(0.1, 0.9, new_count))
                else:
                    self.marker_ratios = self.marker_ratios[:new_count]
        else:
            if new_count != len(self.marker_coords):
                if new_count > len(self.marker_coords):
                    for _ in range(new_count - len(self.marker_coords)):
                        self.marker_coords.append([self.lv[0, 0] * 0.5 + (np.random.rand() - 0.5) * 0.1,
                                                   self.lv[1, 1] * 0.5 + (np.random.rand() - 0.5) * 0.1])
                else:
                    self.marker_coords = self.marker_coords[:new_count]
                self.cached_marker_coords = None

        self._update_all(full_refresh=True)

    def _update_all(self, full_refresh=False):
        """
        The window redrawn for its state: the maps (full_refresh), the markers, then, while
        running, the computations the state asks for (needs_topo / needs_ldos / needs_spec, the
        same functions as before) and the panels of the view.  The Line / Path cluster and the Map
        rows are rebuilt as cells of their own PanelRows in the rest pane (automatic layout: the
        one rule for every window; rule 11: the rows size themselves from the panels), after which
        the panes are arranged again and the window trimmed; the periodic images of a map are one
        raster drawn at every cell offset (tricontourf offsets=: moved, never re-imaged).
        """
        if full_refresh:
            self.ax_map.clear()
            n = int(self.s_cell.val)
            if self.mode == 'Map':
                self.ax_map_global.clear()
            t_ax = self.ax_map_global if self.mode == 'Map' else self.ax_map
            z_data = self.global_z_map

            offs = self._cell_offsets(n)
            t_ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], z_data, levels=60, cmap=self.cmap_topo, zorder=1, offsets=offs)
            if self.mode == 'Map':
                self.ax_map.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1, offsets=offs)
            if self.show_atoms:
                # The atoms of one type over every image as one item (one colour, one scatter).
                tr = np.repeat(self.atomtypes, self.atomnums)
                for t_idx, t_name in enumerate(self.atomtypes):
                    m = (tr == t_name)
                    xs = np.concatenate([self.coord[m, 0] + off[0] for off in offs])
                    ys = np.concatenate([self.coord[m, 1] + off[1] for off in offs])
                    t_ax.scatter(xs, ys, s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)
                    if self.mode == 'Map':
                        self.ax_map.scatter(xs, ys, s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)
            if self.show_unit_cell:
                v0, v1, v2, v3 = np.array([0, 0]), self.lv[0, :2], self.lv[0, :2] + self.lv[1, :2], self.lv[1, :2]
                cell_pts = np.array([v0, v1, v2, v3, v0])
                t_ax.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')
                if self.mode == 'Map':
                    self.ax_map.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')

            t_ax.set_aspect('equal')
            t_ax.set_title(f"Global Topo: bias {self.global_topo_bias} V, height {self.topo_height} Å")
            if self.mode == 'Map':
                self.ax_map.set_aspect('equal')
                self.ax_map.set_title(f"Map Topo: bias {self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val} V, height {self.ldos_height} Å")
            if self.mode in ('Line', 'Path'):
                self.ax_map.add_line(self.line_art)
                if self.mode == 'Line':
                    self.ax_map.add_collection(self.ends)
                else:
                    self.ax_map.add_collection(self.path_pts)
            self.ax_map.add_collection(self.marks)
            # The crop (the placement topo's limits) and its cursors, drawn afresh on the cleared panel.
            self._apply_crop()
            self._draw_cursors()

        if self.mode == 'Line':
            v = self.p2 - self.p1
            p_len = norm(v)
            p_dist = np.linspace(0, p_len, self.npts)
            p_xy = np.array([self.p1 + r * v for r in np.linspace(0, 1, self.npts)])
            self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])
            self.ends.set_offsets([self.p1, self.p2])
            self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios]))
            self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        elif self.mode == 'Path':
            p_xy, p_dist, p_len, verts = self._build_path_xy()
            self.line_art.set_data(verts[:, 0], verts[:, 1])
            self.path_pts.set_offsets(np.array(self.path_nodes))
            mark_idx = (np.array(self.marker_ratios) * (self.npts - 1)).astype(int)
            self.marks.set_offsets(p_xy[mark_idx])
            self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        else:
            self.marks.set_offsets(self.marker_coords)
            self.marks.set_facecolors(self.m_colors[:len(self.marker_coords)])

        if not self.is_running:
            self.fig.canvas.draw_idle()
            return

        if getattr(self, '_broadening_topo_dirty', False):
            self._recompute_global_topo()
            self._broadening_topo_dirty = False
            self._update_all(full_refresh=True)
            return

        bias_e = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        nepts = int(self.s_nepts.val) if self.mode == 'Map' else None
        decay_model = self._effective_decay_model()
        needs_topo = ((self.mode == 'Line' and (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy_line != bias_e or self.cached_d_topo_line != decay_model)) or
                      (self.mode == 'Path' and (self.cached_path_nodes is None or not np.array_equal(np.array(self.path_nodes), self.cached_path_nodes) or self.cached_path_extend != self.path_extend or self.cached_bias_energy_line != bias_e or self.cached_d_topo_line != decay_model)) or
                      (self.mode == 'Map' and (self.cached_bias_energy_map != bias_e or self.cached_d_topo_map != decay_model)))
        needs_ldos = (needs_topo or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val or
                      self.cached_d_ldos != decay_model or getattr(self, 'cached_mode', None) != self.mode or
                      (self.mode == 'Map' and self.cached_nepts != nepts))
        needs_spec = (needs_ldos or (self.mode in ['Single Point', 'Map'] and (self.cached_marker_coords is None or not np.array_equal(self.marker_coords, self.cached_marker_coords))))

        if needs_topo:
            if bias_e == 0:
                raise ValueError("Stabilization bias is 0: set a nonzero E Min or E Max (per ldos_bias_sign) before the constant-current window can be built.")
            if self.mode in ('Line', 'Path'):
                l_emin, l_emax = sorted([0.0, bias_e])
                p_xy_gpu = cp.array(p_xy, dtype=cp.float32)
                ld_1, _, l_engs = self._calculate_ldos_at_points_gpu(
                    cp.hstack([p_xy_gpu, cp.full((self.npts, 1), self.z_highest_atom + self.ldos_height, dtype=cp.float32)]),
                    l_emin, l_emax, use_energy_decay=self.use_decay, preserve_orbitals=False, topo_only=True,
                    clamp_occupied=self.t2_clamp)
                target = cp.max(gpu_simpson(ld_1, l_engs))
                print(f"[*] Path Setpoint LDOS: {float(target):.6e}")
                z_line = self._converge_tip_height(cp.full(self.npts, self.z_highest_atom + self.ldos_height, dtype=cp.float32), p_xy_gpu, l_emin, l_emax, target,
                                                   use_decay=self.use_decay, clamp_occupied=self.t2_clamp)
                self.current_z_line = cp.asnumpy(z_line)
                if self.mode == 'Line':
                    self.cached_p1, self.cached_p2 = self.p1.copy(), self.p2.copy()
                else:
                    self.cached_path_nodes = np.array(self.path_nodes).copy()
                    self.cached_path_extend = self.path_extend
                self.cached_bias_energy_line, self.cached_d_topo_line = bias_e, decay_model
            elif self.mode == 'Map':
                t_emin, t_emax = sorted([0.0, bias_e])
                grid_res = int(np.sqrt(len(self.grid_xy)))
                ldos_cache_name = f"ldos_topo_{bias_e}V_{self.ldos_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}.npy"
                if self.reuse_cache and exists(ldos_cache_name):
                    print(f"[*] Loading Cached LDOS Topography: {ldos_cache_name}")
                    self.current_z_map = np.load(ldos_cache_name)
                else:
                    print(f"[*] Cache not found. Calculating LDOS Topography: {ldos_cache_name}")
                    z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.ldos_height, dtype=cp.float32)
                    ld_1, _, init_engs = self._calculate_ldos_at_points_gpu(
                        cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax,
                        use_energy_decay=self.use_decay, preserve_orbitals=False, topo_only=True,
                        clamp_occupied=self.t2_clamp)
                    target_setp = cp.max(gpu_simpson(ld_1, init_engs))
                    print(f"[*] Map Local Setpoint LDOS: {float(target_setp):.6e}")
                    z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_emin, t_emax, target_setp,
                                                          use_decay=self.use_decay, clamp_occupied=self.t2_clamp)
                    self.current_z_map = cp.asnumpy(z_map_gpu)
                    np.save(ldos_cache_name, self.current_z_map)
                self.cached_bias_energy_map, self.cached_d_topo_map = bias_e, decay_model
                self.ax_map.clear()
                n = int(self.s_cell.val)
                self.ax_map.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1, offsets=self._cell_offsets(n))
                self.ax_map.set_aspect('equal')
                self.ax_map.add_collection(self.marks)
                self._apply_crop()                 # the Map Topo takes the placement topo's crop

        if needs_ldos:
            if self.mode in ('Line', 'Path'):
                ld_up, ld_dn, eg = self._calculate_ldos_at_points_gpu(
                    np.hstack([p_xy, self.current_z_line[:, None]]),
                    self.s_emin.val, self.s_emax.val,
                    use_energy_decay=self.use_decay, preserve_orbitals=True,
                    clamp_occupied=self.t2_clamp)
            elif self.mode == 'Map':
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
                eg = cp.array(self.energies[np.searchsorted(self.energies, self.s_emin.val):np.searchsorted(self.energies, self.s_emax.val, side='right')])
                ld_up_list, ld_dn_list = [], []
                grid_z = cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]])
                for t_e in self.map_e_targets:
                    e_idx = np.searchsorted(self.energies, t_e)
                    t_up, t_dn, _ = self._calculate_ldos_at_points_gpu(
                        grid_z, self.energies[e_idx],
                        self.energies[min(e_idx + 1, len(self.energies) - 1)] + 1e-6,
                        use_energy_decay=self.use_decay, preserve_orbitals=True,
                        clamp_occupied=self.t2_clamp)
                    ld_up_list.append(t_up[:, 0:1])
                    if t_dn is not None:
                        ld_dn_list.append(t_dn[:, 0:1])
                ld_up = cp.concatenate(ld_up_list, axis=1)
                ld_dn = cp.concatenate(ld_dn_list, axis=1) if ld_dn_list else None
            else:
                eg = cp.array(self.energies[np.searchsorted(self.energies, self.s_emin.val):np.searchsorted(self.energies, self.s_emax.val, side='right')])
                ld_up, ld_dn = None, None

            if ld_up is not None:
                self.cached_ld_up = cp.asnumpy(ld_up)
                self.cached_ld_dn = cp.asnumpy(ld_dn) if ld_dn is not None else None
            self.cached_eg = cp.asnumpy(eg)
            self.cached_emin, self.cached_emax = self.s_emin.val, self.s_emax.val
            self.cached_d_ldos, self.cached_nepts = decay_model, nepts
            self.cached_mode = self.mode

        if needs_spec and self.mode in ['Single Point', 'Map']:
            m_coords = np.array(self.marker_coords)
            z_marks = []
            inv_lv_np = inv(self.lv)
            for pt in m_coords:
                pt_3d = np.array([pt[0], pt[1], 0.0])
                f_pt = np.dot(pt_3d, inv_lv_np)
                f_pt[:2] = f_pt[:2] % 1.0
                wrapped_pt = np.dot(f_pt, self.lv)
                dist_sq = (self.grid_xy[:, 0] - wrapped_pt[0]) ** 2 + (self.grid_xy[:, 1] - wrapped_pt[1]) ** 2
                z_marks.append(self.current_z_map[np.argmin(dist_sq)])
            z_marks = np.array(z_marks)

            pt_gpu = cp.array(np.hstack([m_coords, z_marks[:, None]]), dtype=cp.float32)
            s_up, s_dn, _ = self._calculate_ldos_at_points_gpu(
                pt_gpu, self.s_emin.val, self.s_emax.val,
                use_energy_decay=self.use_decay, preserve_orbitals=True,
                clamp_occupied=self.t2_clamp)
            s_up_np = cp.asnumpy(s_up)
            s_dn_np = cp.asnumpy(s_dn) if s_dn is not None else None

            spec_ldos = self._combine_ldos_display(s_up_np, s_dn_np, self.show_mag)
            self.cached_spec_ldos = spec_ldos
            self.cached_marker_coords = np.array(self.marker_coords).copy()

        if self.mode in ['Line', 'Path', 'Map'] and self.cached_ld_up is not None:
            f_up = self.cached_ld_up.copy()
            f_dn = self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None
            f_ldos_raw = self._combine_ldos_display(f_up, f_dn, self.show_mag)
            partitions = self._get_partitions(f_ldos_raw)
        else:
            f_ldos_raw = None
            partitions = []

        # --- Determine whether mag data is signed (for colormap selection) ---
        mag_signed = self._mag_is_signed()
        has_mag = self._has_mag_channel()
        use_diverging = self.show_mag and has_mag and mag_signed

        if self.mode in ('Line', 'Path'):
            active_idx = int(self.marker_ratios[min(self.active_marker_idx, len(self.marker_ratios) - 1)] * (self.npts - 1))
            active_ldos = f_ldos_raw[active_idx] if f_ldos_raw is not None else np.zeros_like(self.cached_spec_ldos[0])
        else:
            active_idx = min(self.active_marker_idx, len(self.marker_coords) - 1)
            active_ldos = self.cached_spec_ldos[active_idx]

        self.ax_spec.clear()

        def _orbit_base(orb):
            if orb.endswith('_up'):
                return orb[:-3]
            elif orb.endswith('_down'):
                return orb[:-5]
            return orb

        def _lighten_color(color, amount=0.3):
            c = plt.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*c)
            return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)

        unique_bases = sorted(set(_orbit_base(o) for o in self.orbitals))
        styles = ['-', '--', ':', '-.'] + [(0, (3 + i, 2)) for i in range(max(0, len(unique_bases) - 4))]
        linestyle_map = dict(zip(unique_bases, styles))
        S = 0.25

        active_ldos_norm = active_ldos.copy()
        if self.normalize:
            total_for_norm = np.sum(active_ldos_norm, axis=(1, 2))
            norm_factor = (simpson(total_for_norm, x=self.cached_eg) + 1e-15)
            active_ldos_norm /= norm_factor

        if self.plot_level == 0:
            if self.mode in ('Line', 'Path'):
                for i, r in enumerate(self.marker_ratios):
                    idx = int(r * (self.npts - 1))
                    c_ldos = f_ldos_raw[idx].copy()
                    t_y = np.sum(c_ldos, axis=(1, 2))
                    if self.normalize:
                        t_y /= (simpson(t_y, x=self.cached_eg) + 1e-15)
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_spec.plot(self.cached_eg, t_y, color=color, lw=2.5, picker=True, pickradius=5, label=f'marker_{i}')
            else:
                for i, pt in enumerate(self.marker_coords):
                    c_ldos = self.cached_spec_ldos[i].copy()
                    t_y = np.sum(c_ldos, axis=(1, 2))
                    if self.normalize:
                        t_y /= (simpson(t_y, x=self.cached_eg) + 1e-15)
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_spec.plot(self.cached_eg, t_y, color=color, lw=2.5, picker=True, pickradius=5, label=f'marker_{i}')
            self.ax_spec.legend(loc='upper right', frameon=False)
        elif self.plot_level == 1:
            self.ax_spec.plot(self.cached_eg, np.sum(active_ldos_norm, axis=(1, 2)), color='black', lw=1.5, alpha=0.1, zorder=1)
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
            mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
            au_y = np.sum(active_ldos_norm[:, au_idx, :], axis=(1, 2)) if au_idx else np.zeros_like(self.cached_eg)
            mol_y = np.sum(active_ldos_norm[:, mol_idx, :], axis=(1, 2)) if mol_idx else np.zeros_like(self.cached_eg)
            self.ax_spec.plot(self.cached_eg, au_y, color='orange', lw=2, label='Au', picker=True, pickradius=5, zorder=2)
            self.ax_spec.plot(self.cached_eg, mol_y, color='black', lw=2, label='Molecule', picker=True, pickradius=5, zorder=2)
            self.ax_spec.legend([plt.Line2D([0], [0], color='orange', lw=2), plt.Line2D([0], [0], color='black', lw=2)], ['Au', 'Molecule'], title="Partition", loc='upper right', frameon=False)
        elif self.plot_level == 2:
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            for t in self.atomtypes:
                t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                t_y = np.sum(active_ldos_norm[:, t_idx, :], axis=(1, 2)) if t_idx else np.zeros_like(self.cached_eg)
                self.ax_spec.plot(self.cached_eg, t_y, color=self._type_color_map.get(t, 'grey'), lw=2, label=t, picker=True, pickradius=3)
            self.ax_spec.legend([plt.Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
        elif self.plot_level == 3:
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
            e_y = np.sum(active_ldos_norm[:, e_idx, :], axis=(1, 2)) if e_idx else np.zeros_like(self.cached_eg)
            self.ax_spec.plot(self.cached_eg, e_y, color=self._type_color_map.get(self.active_element, 'grey'), lw=2, alpha=0.15, zorder=1)
            for a_idx, label in self.vesta_label_map.items():
                if label.startswith(self.active_element):
                    y_sum = np.sum(active_ldos_norm[:, a_idx - 1, :], axis=1)
                    self.ax_spec.plot(self.cached_eg, y_sum, color=self._type_color_map.get(self.active_element, 'grey'), lw=2, label=label, picker=True, pickradius=3, zorder=2)
            self.ax_spec.legend([plt.Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
        elif self.plot_level == 4:
            for a_idx, label in self.vesta_label_map.items():
                if label.startswith(self.active_element):
                    y_sum = np.sum(active_ldos_norm[:, a_idx - 1, :], axis=1)
                    element_color = self._type_color_map.get(self.active_element, 'grey')
                    if a_idx == self.active_atom:
                        self.ax_spec.plot(self.cached_eg, y_sum, color=element_color, lw=2.5, alpha=1.0, zorder=5)
                        orb_artists = []
                        for orb in self.orbitals:
                            col_idx = self.orbitals.index(orb)
                            y_orb = active_ldos_norm[:, a_idx - 1, col_idx]
                            ls = linestyle_map[_orbit_base(orb)]
                            p_color = _lighten_color(element_color, 0.3) if orb.endswith('_up') else element_color
                            o_line, = self.ax_spec.plot(self.cached_eg, y_orb, color=p_color, linestyle=ls, lw=1.2, label=f"{label} – {orb}", zorder=10)
                            orb_artists.append(o_line)
                        if getattr(self, 'cursor', None) is not None:
                            self.cursor.remove()
                        # The hover label of the orbital curves (their labels), on from the start as the old cursor was.
                        self.cursor = widgets.HoverLabels(self.ax_spec, orb_artists, enabled=True, arm_by_click=False)
                    else:
                        orig = plt.to_rgb(element_color)
                        lumi = 0.299 * orig[0] + 0.587 * orig[1] + 0.114 * orig[2]
                        faded_color = (S * np.array(orig)) + ((1 - S) * lumi)
                        self.ax_spec.plot(self.cached_eg, y_sum, color=faded_color, lw=1.5, alpha=0.05, zorder=2)
            leg1 = self.ax_spec.legend([plt.Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
            self.ax_spec.add_artist(leg1)
            self.ax_spec.legend([plt.Line2D([0], [0], color='black', linestyle=linestyle_map[b], lw=1.5) for b in unique_bases], unique_bases, title="Orbitals", loc='upper left', frameon=False)

        lines = self.ax_spec.get_lines()
        active_maxes = [np.max(l.get_ydata()) for l in lines if l.get_visible() and l.get_alpha() == 1.0]
        if active_maxes:
            self.ax_spec.set_ylim(0, max(active_maxes) * 1.1)

        p_dist_val = p_dist[active_idx] if self.mode in ('Line', 'Path') else f"[{self.marker_coords[active_idx][0]:.1f}, {self.marker_coords[active_idx][1]:.1f}]"
        self.ax_spec.set(title=f"Partitioned LDOS: marker {self.active_marker_idx} at {p_dist_val} Å", xlabel="Energy (eV)")
        # (The spectrum no longer keeps a square box in Line / Path: a curve panel fills the room the
        # arrangement gives it, so no blank is left around it; rule 11.)

        if self.mode == 'Map':
            if not hasattr(self, 'map_e_targets') or len(self.map_e_targets) != nepts or full_refresh:
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
            for i, e_val in enumerate(self.map_e_targets):
                self.ax_spec.axvline(x=e_val, color='black', linestyle='-', lw=2, picker=5, label=f'emarker_{i}')

        if self.mode in ('Line', 'Path'):
            self._clear_cluster()
            self.ax_prof.clear()

            processed_partitions = []
            global_vmax = 0.0

            if self.normalize and f_ldos_raw is not None:
                total_ldos_line = np.sum(f_ldos_raw, axis=(2, 3))
                norm_denom_line = simpson(total_ldos_line, x=self.cached_eg, axis=1)[:, None] + 1e-15
            else:
                norm_denom_line = 1.0

            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    t_data /= norm_denom_line
                processed_partitions.append((p_label, t_data))
                v_max = np.max(np.abs(t_data))
                if v_max > global_vmax:
                    global_vmax = v_max
            if global_vmax == 0:
                global_vmax = 1e-15

            num_p = max(1, len(partitions))
            # The cluster's row, left to right (a PanelRow: its shaped panels share one height; rule
            # 11): the 'Topo' panel with the LDOS h box above it (Line only), the tip-height stripe
            # (with the LDOS h box above it in Path), one heatmap per partition with a colour bar
            # after each (show_dcmp_norm) or after the last.
            row = self._cluster_row
            if self.mode == 'Line':
                # Line Topo panel (the unwrapped 'Topo' corridor) — Line mode only; the LDOS h box above it.
                self.ax_line_topo = self._panel_with_box(row, self._box_ldos_h, stretch=10)
                self.line_decomp_axes.append(self.ax_line_topo)

                v_line = self.p2 - self.p1
                v_hat = v_line / p_len
                v_perp = np.array([-v_hat[1], v_hat[0]])
                half_w = p_len / 2.0

                n_tile = max(int(self.s_cell.val), 1) + 1
                margin = p_len * 0.15
                all_rx, all_ry, all_z = [], [], []
                for i in range(-n_tile, n_tile + 1):
                    for j in range(-n_tile, n_tile + 1):
                        off = i * self.lv[0, :2] + j * self.lv[1, :2]
                        pts = self.grid_xy + off
                        dx = pts[:, 0] - self.p1[0]
                        dy = pts[:, 1] - self.p1[1]
                        rot_x = dx * v_perp[0] + dy * v_perp[1]
                        rot_y = dx * v_hat[0] + dy * v_hat[1]
                        mask = (rot_x > -half_w - margin) & (rot_x < half_w + margin) & (rot_y > -margin) & (rot_y < p_len + margin)
                        all_rx.append(rot_x[mask])
                        all_ry.append(rot_y[mask])
                        all_z.append(self.global_z_map[mask])
                all_rx, all_ry, all_z = np.concatenate(all_rx), np.concatenate(all_ry), np.concatenate(all_z)
                self.ax_line_topo.tricontourf(all_rx, all_ry, all_z, levels=60, cmap=self.cmap_topo, zorder=1)

                if self.show_atoms:
                    tr = np.repeat(self.atomtypes, self.atomnums)
                    for t_idx, t_name in enumerate(self.atomtypes):
                        m = (tr == t_name)
                        a_rx_all, a_ry_all = [], []
                        for i_tile in range(-n_tile, n_tile + 1):
                            for j_tile in range(-n_tile, n_tile + 1):
                                off = i_tile * self.lv[0, :2] + j_tile * self.lv[1, :2]
                                a_dx = self.coord[m, 0] + off[0] - self.p1[0]
                                a_dy = self.coord[m, 1] + off[1] - self.p1[1]
                                a_rx = a_dx * v_perp[0] + a_dy * v_perp[1]
                                a_ry = a_dx * v_hat[0] + a_dy * v_hat[1]
                                vis = (a_rx > -half_w) & (a_rx < half_w) & (a_ry > 0) & (a_ry < p_len)
                                a_rx_all.append(a_rx[vis])
                                a_ry_all.append(a_ry[vis])
                        if a_rx_all:
                            self.ax_line_topo.scatter(np.concatenate(a_rx_all), np.concatenate(a_ry_all), s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)

                self.ax_line_topo.plot([0, 0], [0, p_len], color='red', ls='--', lw=2.0, zorder=6)
                self.ax_line_topo.scatter([0], [0], c='white', edgecolors='red', s=60, zorder=8)
                self.ax_line_topo.scatter([0], [p_len], c='white', edgecolors='red', s=60, zorder=8)

                for i, r in enumerate(self.marker_ratios):
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_line_topo.scatter([0], [r * p_len], c=color, edgecolors='black', s=80, zorder=10)

                self.ax_line_topo.set_xlim(-half_w, half_w)
                self.ax_line_topo.set_ylim(0, p_len)
                self.ax_line_topo.set_box_aspect(1)
                self.ax_line_topo.set_xlabel("Perp. (Å)", fontsize=10)
                self.ax_line_topo.set_ylabel("Position (Å)", fontsize=10)
                self.ax_line_topo.set_title("Topo", fontsize=10)
            else:
                self.ax_line_topo = None

            # Topo stripe: a narrow cell as wide as its 40-point stroke and its tick axis; in Path the LDOS h box above it.
            if self.mode == 'Path':
                self.ax_stripe = self._panel_with_box(row, self._box_ldos_h, inner_w=58)
            else:
                self.ax_stripe = self._panel(row, inner_w=58)
            self.line_decomp_axes.append(self.ax_stripe)
            lc = plt.LineCollection(
                np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1],
                          np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2),
                cmap=self.cmap_topo,
                norm=plt.Normalize(self.current_z_line.min(), self.current_z_line.max()), linewidth=40)
            lc.set_array(self.current_z_line[:-1])
            self.ax_stripe.add_collection(lc)
            self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len))
            self.ax_stripe.set_xticks([])
            self.ax_prof.plot(p_dist, self.current_z_line, 'k-', lw=1.5)
            self.ax_prof.set(ylabel="Height (Å)", title="Tip Height", xlabel="Dist (Å)")

            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                ax_l = self._panel(row, stretch=10)
                self.line_decomp_axes.append(ax_l)
                cax_l = None
                if self.show_dcmp_norm or p_idx == num_p - 1:
                    cax_l = self._panel(row, inner_w=15)
                    self.line_decomp_axes.append(cax_l)

                v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
                if v_max == 0:
                    v_max = 1e-15

                if use_diverging:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='bwr', shading='auto', vmin=-v_max, vmax=v_max)
                elif self.show_mag and has_mag:
                    # NCL: |m| is non-negative, use sequential colormap
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='hot', shading='auto', norm=plt.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))
                else:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='nipy_spectral', shading='auto', norm=plt.PowerNorm(self.ldos_gamma, vmin=0, vmax=v_max))

                ax_l.set_box_aspect(1)
                ax_l.set_anchor('W')

                # The panel's short title is the partition itself (the cluster's label row says
                # 'LDOS' and, at the atom levels, the element).
                if self.plot_level >= 3:
                    ax_l.set_title(p_label.split()[-1], fontsize=10)
                else:
                    ax_l.set_title(p_label, fontsize=10)
                ax_l.set_yticks([])

                if cax_l is not None:
                    cb = self.fig.colorbar(mesh, cax=cax_l)
                    exp = int(np.floor(np.log10(v_max)))
                    cb.ax.yaxis.set_major_formatter(plt.ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
                    cax_l.set_title(f"1e{exp}", fontsize=10)
                    cax_l.set_anchor('W')

                for i, r in enumerate(self.marker_ratios):
                    idx = int(r * (self.npts - 1))
                    color = self.m_colors[i % len(self.m_colors)]
                    if p_idx == 0:
                        self.ax_prof.axvline(x=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                    ax_l.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                    if p_idx == 0:
                        self.ax_stripe.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')

            # The cluster's row shares one height (the row rule, Figure.align_row): the stripe and the
            # colour bars follow the heatmaps' box aspect at any window size.
            self.fig.align_row(*self.line_decomp_axes)
            self._cluster_label.set_text(f"LDOS: {self.active_element}" if self.plot_level >= 3 else "LDOS")
            self._rearrange()                      # the row is new: the arrangement and the window follow (rule 11)

        elif self.mode == 'Map':
            num_p = max(1, len(partitions))

            if len(self.map_axes) != nepts * num_p or full_refresh:
                # The map block rebuilt: nepts maps per partition, wrapped into the rows that make
                # them largest (controls.MapBlock; rule 11), a colour bar cell per partition
                # (show_dcmp_norm) or one for the whole block (automatic layout).
                self._clear_map_rows()
                groups, cell_bars = [], {}
                for r in range(num_p):
                    cells = []
                    for _ in range(nepts):
                        ax_m, cell = self._cell_axes()
                        ax_m.set_rotated(self.cell_rotated)
                        self.map_axes.append(ax_m)
                        cells.append(cell)
                        if self.cmap_norm:
                            # 'cmap norm': every map its own colour bar beside it (cax_list by the map's index).
                            bar = self.fig.cell(inner_w=12)
                            self._panel_cells.append(bar)
                            self.cax_list.append(self.fig.add_axes(cell=bar))
                            cell_bars[cell] = bar
                    bar = None
                    if self.show_dcmp_norm and not self.cmap_norm:
                        bar = self.fig.cell(inner_w=15)
                        self._panel_cells.append(bar)
                        self.cax_list.append(self.fig.add_axes(cell=bar))
                    groups.append((cells, bar))
                    self._map_cells += cells + ([bar] if bar is not None else []) + [cell_bars[c] for c in cells if c in cell_bars]
                shared = None
                if not self.show_dcmp_norm and not self.cmap_norm:
                    shared = self.fig.cell(inner_w=15)
                    self._panel_cells.append(shared)
                    self.cax_list.append(self.fig.add_axes(cell=shared))
                    self._map_cells.append(shared)
                self._map_block.set_groups(groups, shared_bar=shared, extra=self._spec_cell, cell_bars=cell_bars)
                self._rearrange()                  # the block is new: the arrangement and the window follow (rule 11)

            if not hasattr(self, 'map_e_targets') or len(self.map_e_targets) != nepts or full_refresh:
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
            m_coords_np = np.array(self.marker_coords)

            processed_partitions = []
            global_vmax = 0.0
            if self.normalize and f_ldos_raw is not None:
                total_ldos_map = np.sum(f_ldos_raw, axis=(2, 3))
                s_idx = np.argsort(self.map_e_targets)
                norm_denom_map = simpson(total_ldos_map[:, s_idx], x=self.map_e_targets[s_idx], axis=1)[:, None] + 1e-15
            else:
                norm_denom_map = 1.0

            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    t_data /= norm_denom_map
                t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
                processed_partitions.append((p_label, t_data))
                v_max = np.max(np.abs(t_data))
                if v_max > global_vmax:
                    global_vmax = v_max
            if global_vmax == 0:
                global_vmax = 1e-15

            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                p_vmax = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
                for i, target_e in enumerate(self.map_e_targets):
                    k = p_idx * nepts + i
                    ax = self.map_axes[k]
                    ax.clear()
                    slice_data = t_data[:, i]
                    v_max = self._map_vmax(slice_data, p_vmax)
                    self._draw_map_slice(ax, slice_data, v_max, use_diverging, has_mag)
                    if self.cmap_norm:
                        self._draw_map_cbar(self.cax_list[k], v_max, use_diverging, has_mag, labelsize=6)
                    ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
                    title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
                    if self.plot_level >= 3:
                        ylabel_str = p_label.split()[-1] if i == 0 else ""
                    else:
                        ylabel_str = p_label if i == 0 else ""
                    ax.set_title(title_str, fontsize=10)
                    if ylabel_str:
                        ax.set_ylabel(ylabel_str, fontsize=10)
                    ax.set_aspect('equal')
                    ax.set_xticks([])
                    ax.set_yticks([])

                if self.show_dcmp_norm and not self.cmap_norm:
                    self._draw_map_cbar(self.cax_list[p_idx], p_vmax if p_vmax > 0 else 1e-15, use_diverging, has_mag)

            if not self.show_dcmp_norm and not self.cmap_norm and len(processed_partitions) > 0:
                self._draw_map_cbar(self.cax_list[0], global_vmax, use_diverging, has_mag)

            self._cluster_label.set_text(f"LDOS: {self.active_element}" if self.plot_level >= 3 else "LDOS")

        self.fig.canvas.draw_idle()

    def _redraw_map_slice(self, i):
        target_e = self.map_e_targets[i]
        f_up = self.cached_ld_up.copy()
        f_dn = self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None
        f_ldos_raw = self._combine_ldos_display(f_up, f_dn, self.show_mag)
        partitions = self._get_partitions(f_ldos_raw)

        mag_signed = self._mag_is_signed()
        has_mag = self._has_mag_channel()
        use_diverging = self.show_mag and has_mag and mag_signed

        m_coords_np = np.array(self.marker_coords)
        nepts = len(self.map_e_targets)

        processed_partitions = []
        global_vmax = 0.0
        if self.normalize and f_ldos_raw is not None:
            total_ldos_map = np.sum(f_ldos_raw, axis=(2, 3))
            s_idx = np.argsort(self.map_e_targets)
            norm_denom_map = simpson(total_ldos_map[:, s_idx], x=self.map_e_targets[s_idx], axis=1)[:, None] + 1e-15
        else:
            norm_denom_map = 1.0

        for p_label, p_data in partitions:
            t_data = p_data.copy()
            if self.normalize:
                t_data /= norm_denom_map
            t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
            processed_partitions.append((p_label, t_data))
            v_max = np.max(np.abs(t_data))
            if v_max > global_vmax:
                global_vmax = v_max
        if global_vmax == 0:
            global_vmax = 1e-15

        for p_idx, (p_label, t_data) in enumerate(processed_partitions):
            k = p_idx * nepts + i
            ax = self.map_axes[k]
            ax.clear()
            p_vmax = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
            slice_data = t_data[:, i]
            v_max = self._map_vmax(slice_data, p_vmax)
            self._draw_map_slice(ax, slice_data, v_max, use_diverging, has_mag)
            if self.cmap_norm:
                self._draw_map_cbar(self.cax_list[k], v_max, use_diverging, has_mag, labelsize=6)
            ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
            title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
            if self.plot_level >= 3:
                ylabel_str = p_label.split()[-1] if i == 0 else ""
            else:
                ylabel_str = p_label if i == 0 else ""
            ax.set_title(title_str, fontsize=10)
            if ylabel_str:
                ax.set_ylabel(ylabel_str, fontsize=10)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

    def _on_pick(self, event):
        label = event.artist.get_label() if hasattr(event.artist, 'get_label') else ''
        if isinstance(label, str) and label.startswith('_cursor_'):    # a crop cursor: its press is its own (before any marker's)
            _, _, axis, k = label.split('_')
            self.active_obj = ('cursor', axis, int(k))
            return
        if self.active_obj is not None and self.active_obj[0] == 'cursor':
            return
        if event.artist == getattr(self, 'ends', None) and self.mode == 'Line':
            self.active_obj = ('end', event.ind[0])
        elif event.artist == getattr(self, 'path_pts', None) and self.mode == 'Path':
            self.active_obj = ('path_node', event.ind[0])
        elif event.artist == getattr(self, 'marks', None):
            self.active_obj = ('mark_map', event.ind[0])
        elif getattr(event, 'mouseevent', None) and event.mouseevent.inaxes == self.ax_spec:
            label = event.artist.get_label()
            if label.startswith('emarker_'):
                if self.plot_level == 0:
                    self.active_obj = ('emarker', int(label.split('_')[1]))
                return
            if self.plot_level == 0:
                if label.startswith('marker_'):
                    self.active_marker_idx = int(label.split('_')[1])
                self.plot_level = 1
            elif self.plot_level == 1:
                self.plot_level = 2
            elif self.plot_level == 2:
                self.active_element = label
                self.plot_level = 3
            elif self.plot_level == 3:
                self.active_atom = next(k for k, v in self.vesta_label_map.items() if v == label)
                self.active_element = self._get_element_by_index_helper(self.active_atom)
                self.plot_level = 4
            self._update_all()
        elif isinstance(event.artist, plt.Line2D) and event.artist.get_label().startswith('marker_'):
            self.active_obj = ('mark_dynamic', int(event.artist.get_label().split('_')[1]))
            self.active_marker_idx = int(event.artist.get_label().split('_')[1])
            self._update_all()

    def _on_press(self, event):
        if event.inaxes == self.ax_spec:
            if self.fig.canvas.manager.toolbar.mode == "" and not event.dblclick:
                hit = any(l.contains(event)[0] for l in self.ax_spec.get_lines() if l.get_picker())
                if not hit:
                    self.plot_level = max(0, self.plot_level - 1)
        if self.fig.canvas.manager.toolbar.mode == "":
            self._update_all()

    def _get_element_by_index_helper(self, a):
        curr = 0
        for idx, count in enumerate(self.atomnums):
            if a <= curr + count:
                return self.atomtypes[idx]
            curr += count
        return 'grey'

    def _on_motion(self, event):
        if self.active_obj is None or event.xdata is None:
            return
        if self.active_obj[0] == 'cursor':
            # A crop cursor follows the mouse on the placement topo along its own axis (made valid as it goes).
            if event.inaxes is self._crop_ax and event.ydata is not None:
                _, axis, k = self.active_obj
                self._set_cursor(axis, k, event.xdata if axis == 'x' else event.ydata)
            return
        t_obj, idx = self.active_obj
        if self.mode == 'Line':
            p_len = norm(self.p2 - self.p1)
        elif self.mode == 'Path':
            _, _, p_len, _ = self._build_path_xy()
        else:
            p_len = 1.0

        if t_obj == 'end' and self.mode == 'Line':
            if idx == 0:
                self.p1 = np.array([event.xdata, event.ydata])
            else:
                self.p2 = np.array([event.xdata, event.ydata])
        elif t_obj == 'path_node' and self.mode == 'Path':
            if event.inaxes == self.ax_map:
                self.path_nodes[idx] = np.array([event.xdata, event.ydata])
        elif t_obj == 'mark_map' and event.inaxes == self.ax_map:
            if self.mode == 'Line':
                v = self.p2 - self.p1
                v_sq = np.dot(v, v)
                if v_sq > 1e-9:
                    self.marker_ratios[idx] = np.clip(np.dot(np.array([event.xdata, event.ydata]) - self.p1, v) / v_sq, 0, 1)
            elif self.mode == 'Path':
                # Project click onto the polyline: nearest resampled path point.
                p_xy, _, _, _ = self._build_path_xy()
                d2 = (p_xy[:, 0] - event.xdata) ** 2 + (p_xy[:, 1] - event.ydata) ** 2
                self.marker_ratios[idx] = int(np.argmin(d2)) / (self.npts - 1)
            else:
                self.marker_coords[idx] = [event.xdata, event.ydata]
        elif t_obj == 'mark_dynamic' and self.mode in ('Line', 'Path'):
            if hasattr(self, 'ax_prof') and event.inaxes == self.ax_prof:
                if p_len > 1e-9:
                    self.marker_ratios[idx] = np.clip(event.xdata / p_len, 0, 1)
            elif event.inaxes in [ax for ax in [getattr(self, 'ax_ldos', None), getattr(self, 'ax_stripe', None), getattr(self, 'ax_line_topo', None)] if ax is not None]:
                if p_len > 1e-9:
                    self.marker_ratios[idx] = np.clip(event.ydata / p_len, 0, 1)
        elif t_obj == 'emarker' and event.inaxes == self.ax_spec:
            self.map_e_targets[idx] = np.clip(event.xdata, self.s_emin.val, self.s_emax.val)
            for line in self.ax_spec.get_lines():
                if line.get_label() == f'emarker_{idx}':
                    line.set_xdata([self.map_e_targets[idx], self.map_e_targets[idx]])
            self.fig.canvas.draw_idle()
            return
        self._update_all()

    def _on_rel(self, event):
        if self.active_obj is not None and self.active_obj[0] == 'emarker':
            idx = self.active_obj[1]
            target_e = self.map_e_targets[idx]
            bias_e = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
            e_idx = np.searchsorted(self.energies, target_e)
            if e_idx >= len(self.energies):
                e_idx = len(self.energies) - 1
            grid_z = cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]])
            t_up, t_dn, _ = self._calculate_ldos_at_points_gpu(
                grid_z, self.energies[e_idx],
                self.energies[min(e_idx + 1, len(self.energies) - 1)] + 1e-6,
                use_energy_decay=self.use_decay, preserve_orbitals=True,
                clamp_occupied=self.t2_clamp)
            self.cached_ld_up[:, idx:idx + 1] = cp.asnumpy(t_up[:, 0:1])
            if t_dn is not None and self.cached_ld_dn is not None:
                self.cached_ld_dn[:, idx:idx + 1] = cp.asnumpy(t_dn[:, 0:1])
            self._redraw_map_slice(idx)
            self.fig.canvas.draw_idle()
        self.active_obj = None
