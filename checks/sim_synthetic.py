"""
The simulator on synthetic processed data, ready to plot: no VASP inputs are read and no physics is
computed.  SyntheticSimulator is the interactive simulator with two things replaced:

  parse_vasp_outputs           a synthetic structure set directly: a 10 x 10 x 20 Å cell with a
                               square of four Au atoms (z 5 Å) and an N-C-C molecule above them
                               (z 7-7.5 Å); nine orbitals (ISPIN=1 names); a DOS per atom and
                               orbital on -4..4 eV (E_F = 0): Au d-band at -2 eV and flat s, the
                               molecule's p states as peaks either side of E_F
  _calculate_ldos_at_points_gpu  the LDOS at the tip positions: each atom's DOS weighted by
                               exp(-2 kappa d) over the atom's periodic images (kappa 1/Å), with the
                               simulator's own shapes ((points, energies), or with preserve_orbitals
                               (points, energies, atoms, orbitals)); one channel.  The cell's a and
                               b are the `cell` argument (10 x 10 Å by default; a tall or a wide
                               cell shows the unit-cell panels' orientation)
  _converge_tip_height         the tip height: a bump of 0.6 Å over every atom (images included)
                               on the starting height

Everything that draws (the views, the partitions, the maps, the spectra, the arrangement) is the
simulator's own code.  Used by checks/render_sim_sizing.py; imports checks/check_simulator.py for the
offscreen screen, the import paths and the cupy stand-in.
"""
import os
import sys
import tempfile

import numpy as np

from checks import check_simulator as cs          # noqa: F401  (offscreen screen, sys.path, cupy stand-in)
from stm_from_dft import simulator as simmod

ORBITALS = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
ENERGIES = np.linspace(-4.0, 4.0, 801)
KAPPA = 1.0                                        # 1/Å, the decay of the LDOS with distance


def _peak(e0, w, a=1.0):
    return a * np.exp(-0.5 * ((ENERGIES - e0) / w) ** 2)


def synthetic_dos(types):
    """(atoms, energies, orbitals) DOS for the atom types in order."""
    dos = np.zeros((len(types), len(ENERGIES), len(ORBITALS)))
    for k, t in enumerate(types):
        if t == 'Au':
            dos[k, :, 0] = 0.05
            for j in range(4, 9):
                dos[k, :, j] = _peak(-2.0, 0.45, 0.6) + _peak(-3.1, 0.3, 0.3)
        elif t == 'N':
            dos[k, :, 1] = _peak(-1.3, 0.15, 0.8) + _peak(1.6, 0.2, 0.5)
            dos[k, :, 2] = _peak(-1.3, 0.15, 1.2) + _peak(1.6, 0.2, 0.9)
            dos[k, :, 3] = _peak(-0.6, 0.12, 0.4)
        else:                                      # C
            dos[k, :, 1] = _peak(-0.9, 0.15, 0.6) + _peak(1.1, 0.2, 0.7)
            dos[k, :, 2] = _peak(-0.9, 0.15, 1.0) + _peak(1.1, 0.2, 1.1)
            dos[k, :, 3] = _peak(-2.4, 0.2, 0.3)
    return dos


class SyntheticSimulator(simmod.Interactive_STM_Simulator):
    """The interactive simulator on synthetic processed data (module docstring)."""

    def __init__(self, erange=(-3.0, 0.5), ldos_height=1.25, cmap_topo=None, directory=None, cell=(10.0, 10.0)):
        self._cell = (float(cell[0]), float(cell[1]))          # the unit cell's a and b (Å): its shape
        directory = directory or tempfile.mkdtemp(prefix='simsynth_')
        super().__init__(filepath=directory, erange=list(erange), ldos_height=ldos_height,
                         cmap_topo=cmap_topo, unit_cell_num=1)

    def parse_vasp_outputs(self, locpot_path):
        cp = simmod.cp
        a, b = self._cell
        self.lv = np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, 20.0]])
        self.atomtypes, self.atomnums = ['Au', 'N', 'C'], [4, 1, 2]
        self.coord = np.array([[2.5, 2.5, 5.0], [7.5, 2.5, 5.0], [2.5, 7.5, 5.0], [7.5, 7.5, 5.0],
                               [5.0, 5.0, 7.0], [3.8, 5.6, 7.5], [6.2, 4.4, 7.5]]) * np.array([a / 10.0, b / 10.0, 1.0])
        types = list(np.repeat(self.atomtypes, self.atomnums))
        self.energies, self.ef = ENERGIES.copy(), 0.0
        self.energies_gpu = cp.asarray(self.energies, dtype=cp.float32)
        self.spin_mode = simmod.SpinMode.COLLINEAR_UNPOL
        self.is_polarized = False
        self.orbitals = list(ORBITALS)
        self.vesta_label_map = {}
        g = 1
        for t, n in zip(self.atomtypes, self.atomnums):
            for r in range(1, n + 1):
                self.vesta_label_map[g] = f"{t}{r}"
                g += 1
        self.locpot_gpu = cp.zeros((4, 4, 8), dtype=cp.float32)
        self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape, dtype=cp.float32)
        self.inv_lv_gpu = cp.array(np.linalg.inv(self.lv), dtype=cp.float32)
        self.num_total_atoms = sum(self.atomnums)
        self.dos_up_gpu = cp.array(synthetic_dos(types), dtype=cp.float32)
        self.dos_up_collapsed = cp.ascontiguousarray(cp.sum(self.dos_up_gpu, axis=2))
        self.dos_dn_gpu = self.dos_dn_collapsed = None
        self._dos_up_gpu_orig = self.dos_up_gpu.copy()
        self._dos_dn_gpu_orig = None
        self.dos_total_gpu = self.dos_total_collapsed = None
        self.z_highest_atom = float(np.max(self.coord[:, 2]))
        coords, idx = [], []
        for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
            for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                idx.append(np.arange(len(self.coord)))
        self.periodic_coord_gpu = cp.array(np.concatenate(coords), dtype=cp.float32)
        self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx))
        self.map_mat_gpu = cp.zeros((self.num_total_atoms, len(self.atom_indices_periodic_gpu)), dtype=cp.float32)
        self.map_mat_gpu[self.atom_indices_periodic_gpu, cp.arange(len(self.atom_indices_periodic_gpu))] = 1.0

    def _weights(self, tip_positions):
        """(points, atoms): exp(-2 kappa d) summed over each atom's periodic images."""
        p = np.asarray(tip_positions, dtype=np.float64)[:, :3]
        pc = np.asarray(self.periodic_coord_gpu, dtype=np.float64)
        d = np.sqrt(((p[:, None, :] - pc[None, :, :]) ** 2).sum(-1))
        w_img = np.exp(-2.0 * KAPPA * d)
        idx = np.asarray(self.atom_indices_periodic_gpu)
        w = np.zeros((len(p), self.num_total_atoms))
        for k in range(self.num_total_atoms):
            w[:, k] = w_img[:, idx == k].sum(1)
        return w

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax, use_energy_decay=False,
                                      preserve_orbitals=False, topo_only=False, clamp_occupied=False):
        cp = simmod.cp
        s = np.searchsorted(self.energies, emin)
        e = np.searchsorted(self.energies, emax, side='right')
        w = self._weights(tip_positions)
        dos = np.asarray(self.dos_up_gpu, dtype=np.float64)[:, s:e, :]
        if preserve_orbitals:
            out = w[:, None, :, None] * dos.transpose(1, 0, 2)[None, :, :, :]
        else:
            out = w @ dos.sum(2)
        return cp.asarray(out.astype(np.float32)), None, self.energies_gpu[s:e]

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos, **kw):
        cp = simmod.cp
        xy = np.asarray(grid_xy_gpu, dtype=np.float64)[:, :2]
        pc = np.asarray(self.periodic_coord_gpu, dtype=np.float64)
        d2 = ((xy[:, None, :] - pc[None, :, :2]) ** 2).sum(-1)
        bump = (0.6 * np.exp(-d2 / (2 * 1.2 ** 2))).sum(1)
        return cp.asarray((np.asarray(z_map_gpu, dtype=np.float64) + bump).astype(np.float32))


def open_simulator(grid_res=32, cell=(10.0, 10.0)):
    """A SyntheticSimulator window on a unit cell of `cell` (a, b in Å), opened (RUN not pressed); returns it."""
    from qtgui import qtplot as plt
    cmap = plt.LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])
    sim = SyntheticSimulator(cmap_topo=cmap, cell=cell)
    fa, fb = cell[0] / 10.0, cell[1] / 10.0
    sim.run_interactive(grid_res=grid_res, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', use_decay=True,
                        show_decay_toggle=True, reuse_cache=False, line_endpoints=([1.0 * fa, 1.0 * fb], [9.0 * fa, 9.0 * fb]),
                        marker_positions=[[2.5 * fa, 2.5 * fb], [5.0 * fa, 5.0 * fb]],
                        path=[[1.0 * fa, 1.0 * fb], [5.0 * fa, 5.0 * fb], [9.0 * fa, 2.0 * fb]])
    return sim
