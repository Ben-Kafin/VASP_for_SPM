# -*- coding: utf-8 -*-
"""
Decay figure: the barriers, their weights, and the current and dI/dV they produce, for a set of decay models
side by side, for one LDOS at one tip point.

Run this file directly (Spyder's Run button, or python plot_decay.py) for the synthetic placeholders in the
block at the bottom, or call decay_barrier_figure(energies, ldos, models, v_topo) with an LDOS from a cache or
from the simulator and the simulator's phi_local at the tip point.  Plain matplotlib: the figure is shown and
can be saved from the plot window (or with save_path).

Two sets of models, chosen by DECAYS in the block at the bottom:
  'correct'  the simulator's two Decay-on models (simulator_decays): 'exact', the exact tilted barrier
             (gpu_exact_factor), and 'linearized', Chen's energy-dependent matrix element
             (gpu_linearized_factor).  Nothing of the decay is computed here: the weights are the factor
             functions called as the topography calls them, and I(V) and dI/dV(V) come from the simulator's
             exact_sweep / linearized_sweep kernels launched on this one LDOS as a one-atom, one-orbital,
             one-image DOS at one tip point.
  'wrong'    the superseded decays (wrong_decays): 'oldest' (the earliest stm_from_dft package: one bias for
             the whole window), Tier 2 and Tier 2 with the 0-clamp (stm_from_dft/wrong_decays/
             tilted_barrier_decay.py), Tier 3 (wrong_decays/tier3_decay.py, without its Feenstra division)
             and McDowell's VASP_ldos_map (lib.tunneling_factor as ldos_line.py and ldos_map.py use it).  Those files are commented out and bound to a simulator instance, so their
             factor functions are copied verbatim into wrong_decays and their usages are transcribed there.
             What those files call "chen" is not the model of Chen's book: it is the WKB average of 2 kappa
             over a linearly tilted barrier (the 3/2-power form); the names are kept because the copies are
             verbatim.

The one quantity formed here that no model forms is the barrier height itself (barrier_heights, below): the
simulators only ever need the weight, the figure draws the height.

Currents.  A model's own current where it has one (the kernels for the simulator's models, the signed
column for Tier 3); otherwise the integrated current, feenstra.current_from_ldos on the model's displayed
LDOS: I(V) = the integral from 0 V to V of that LDOS, oriented (Tier 2, Tier 2 0-clamp, McDowell).  The same
integrated current of the bare LDOS is the no-tunneling-decay current.

The figure (rows of equal height; one section per model, side by side):
  row 0  the input LDOS, no weight, across the full window, in black, no x tick labels (it shares its x axis
         with the row below);
  row 1  the integrated current of that LDOS, no tunneling decay, across the full window, in black (no bias
         is applied to it); under its data floor the colour key, a bar coloured by the bias map along the x
         axis, labelled 'vbias'.
         Rows 0 and 1 are each autoscaled and drawn to one common y scale (data units per inch), so their
         heights are in the ratio of their y ranges within the two row units they share, the key band added
         below the current's data floor at that scale;
  (each section also carries three equation strips in bold textbook mathtext, larger than any axis label: its
   barrier equation above the barrier row, the equation taking the barrier to the weight, the squared
   matrix element relative to its prefactor, between the barrier row and the weights row, and the equations
   taking the weights to I(V) and dI/dV(V) between the weights row and the I, dI/dV row)
  row 2  the barrier (the vacuum level measured from E_F: phi at the 0 V end, the model's bias-end height at
         the bias end, phi + V for the simulator's exact model, phi + |V| for the superseded ones, the
         linearized one flat at phi + V/2), phi marked: the topo column (x from 0 V to v_topo) and the LDOS
         column (x from -v_max to v_max), the columns' widths in the ratio |v_topo| : 2 v_max;
  row 3  the weights, same x axes;
  row 4  I(V) and dI/dV(V), two panels of equal width.
How each panel is drawn follows the model's own behaviour, declared on the model (barrier_kind, topo_mode,
sweep_mode), so the same rules give every decay a coherent picture:
  - colour means the bias the model actually passed to its factor at that point, one diverging map over
    -v_max .. v_max in every panel: the signed V for the simulator's models; |V| for the superseded ones, so
    a state at E in Tier 2 or McDowell's line is coloured by |E| (both sides of 0 V run grey to red), Tier 3's
    columns by |V_j|, and the oldest's and McDowell's topography by |V_b|; the fixed W for the oldest display;
  - a barrier that is a segment of one curve (sloped) is drawn as that curve coloured by the bias whose end
    each point is; a flat barrier per bias is drawn as the strips between consecutive biases, a wedge;
  - weights that form a family over the bias (every state's weight changes with V) are drawn as the strips
    between consecutive biases' curves, a shape whose edge along E = V is the entering state's weight;
    weights that each state carries on its own (its own bias |E|) are drawn as one curve coloured by that
    bias, |E|;
  - a topo column with one frozen bias is drawn uniformly in that bias's colour (it is the sweep's slice at
    V = v_topo); a topo column in which each state keeps its own bias is drawn like the sweep, restricted to
    0 V .. v_topo;
  - a display whose bias is one frozen value for every state (the 'oldest' model: the window's width W) is
    drawn as that one barrier from (0 V, phi) with slope 1 to (W, phi + W), its barrier panel's x axis kept
    at the sweep's width but centred on the barrier rather than on 0 V, and one weight curve w(E; W) across
    the window, both in W's colour (W lies past the colour map's range, so that colour is the map's saturated
    end);
  - the I and dI/dV curves are coloured along their length by the bias the model passed at the bias on their
    x axis (|V| for the superseded models).
Every panel type shares its y range across the sections (the whole weights row as one axis that includes 0).
Each decay section (its six panels, labels included) is framed as one panel; the two top rows are not.
The weights row and the dI/dV panels carry
three-level ticks and grid (spacings 1, 0.5 and 0.1, in volts on x and in the displayed power-of-ten units on
y), long/thick and labelled, medium, short/thin.
"""

import os
import sys

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

if globals().get('__package__') in (None, ''):
    # Run as a file (Spyder's Run button, python plot_decay.py; Spyder's runfile may not define
    # __package__ at all): simulator.py imports its neighbours relatively, so it is only importable as part of
    # the package; put the package's parent folder on sys.path and import the package absolutely.  Imported as
    # part of the package, the absolute import works as is.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stm_from_dft.simulator import (gpu_exact_factor, gpu_linearized_factor, KAPPA0_PER_SQRT_EV,
                                    _exact_sweep_kernel, _linearized_sweep_kernel)
from stm_from_dft import feenstra


# ==== THE BARRIER (the one quantity no model forms) ====
def barrier_heights(E, V, phi):
    """
    Physics: the barrier a state at E (eV from E_F) sees at bias V, with phi the local barrier at the tip
    point (the LOCPOT potential at the tip point minus E_F, eV): its height above the state at the 0 V
    (sample) end, phi - E, and at the bias (tip) end, phi - E + V, between which it rises linearly
    (gpu_exact_factor's b and a), and its mean height, phi - E + V/2, the flat barrier the linearized weight
    stands on (gpu_linearized_factor; C. J. Chen, Introduction to Scanning Tunneling Microscopy, 3rd ed.
    (2021), Eqs. 2.36-2.37: phi_bar - epsilon, epsilon = E - V/2 the state's energy from the middle of the
    bias window).  E = 0 gives the barrier itself, the vacuum level measured from E_F.  The simulator never
    needs these as numbers, it uses only the weight exp(-S); the figure draws them.  E, V, phi: scalars or
    arrays that broadcast.  Returns (sample end, tip end, mean).
    """
    b = phi - E
    a = phi - E + V
    mean = phi - E + 0.5 * V
    return b, a, mean


# ==== MACHINERY: the grid (no physics) ====
def grid_parameters(energies):
    """
    Machinery: the DOS grid as simulator.py takes it (its lines defining e_step, k0, e0, e_step32 and
    e032): step = (E max - E min)/(N - 1); k0 the last sample at or below 0 V; e0 = energies[k0], exactly
    0.0 when a sample is at 0 V; step and e0 rounded to float32 once for the GPU, which forms sample i's
    energy as e0 + (i - k0) step.  The kernels read the stencil of two samples on each side of 0 V without
    bounds checks, so the grid must have them (ValueError otherwise, as the simulator raises on such a
    DOSCAR).  Returns (k0, step, e0, step32, e032).
    """
    energies = np.asarray(energies, dtype=np.float64).ravel()
    n = len(energies)
    step = float((energies[-1] - energies[0]) / (n - 1))
    k0 = int(np.searchsorted(energies, 0.0, side='right')) - 1
    n_below = k0 if (k0 >= 0 and energies[k0] == 0.0) else k0 + 1      # the samples strictly below 0 V
    n_above = n - 1 - k0                                                 # the samples strictly above 0 V
    if n_below < 2 or n_above < 2:
        raise ValueError(
            f"energies: 0 V (E_F) lies within two samples of an end of the grid ({energies[0]:.6f} to "
            f"{energies[-1]:.6f} eV from E_F; {n_below} samples below 0 V, {n_above} above): the kernels' "
            f"integrals from 0 V need two samples on each side of it.")
    e0 = float(energies[k0])
    return k0, step, e0, np.float32(step), np.float32(e0)


def gpu_energies(n, k0, step32, e032):
    """Machinery: the GPU energy of every sample i of an n-sample grid, e0 + (i - k0) step in float32, as the
    simulator forms it for the frozen-bias weights.  Returns a float32 numpy array of length n."""
    i = np.arange(n, dtype=np.int64)
    return (e032 + (i - k0).astype(np.float32) * step32).astype(np.float32)


# ==== THE SIMULATOR'S MODELS ====
def decay_weights(model, E, V, phi, d):
    """
    Machinery: the weight w(E, V) of THE DECAY MODEL (simulator.py) for one image at distance d (Angstrom)
    at one tip point with barrier phi (eV): gpu_exact_factor ('exact') or gpu_linearized_factor
    ('linearized') with topo=True, called as the topography calls them, E the samples' float32 GPU energies
    (any shape), V one float32 bias.  Returns w at each E as a float64 numpy array of E's shape.
    """
    factor = gpu_exact_factor if model == 'exact' else gpu_linearized_factor
    e_np = np.asarray(E, dtype=np.float32)
    e_gpu = cp.asarray(e_np.reshape(-1))
    phi_row = cp.asarray([[phi]], dtype=cp.float32)        # (1, n_pts): one tip point
    dists = cp.asarray([[d]], dtype=cp.float32)            # (n_images, n_pts): one image
    w = factor(e_gpu, np.float32(V), phi_row, dists, topo=True)
    return cp.asnumpy(w).reshape(e_np.shape).astype(np.float64)


def sweep_current_didv(model, energies, ldos, phi, d):
    """
    Machinery: I(V) and dI/dV(V) of THE DECAY MODEL (simulator.py, the two lines of the block) for this one
    LDOS, from the simulator's own sweep kernel (exact_sweep for 'exact', linearized_sweep for 'linearized'),
    launched as _calculate_ldos_at_points_gpu's _sweep launches it: the LDOS as a one-atom, one-orbital DOS
    (dos (1, n_e, 1)), one image at distance d (dists (1, 1)), one tip point with barrier phi (phi (1,)),
    want_cur = 1 so the current comes from the same pass as dI/dV.  Requested: every grid sample but the one
    exactly at 0 V (which _display_indices never displays), ascending.
    Returns (v, current, didv): the requested samples' energies (float64) and I and dI/dV there (float64).
    """
    energies = np.asarray(energies, dtype=np.float64).ravel()
    ldos = np.asarray(ldos, dtype=np.float64).ravel()
    n = len(energies)
    k0, step, e0, step32, e032 = grid_parameters(energies)
    u_idx = np.arange(n, dtype=np.int64)
    if e0 == 0.0:
        u_idx = u_idx[u_idx != k0]                                       # the 0 V sample is never displayed
    estart = int(u_idx.min())
    st_lo = k0 - 2 if e0 == 0.0 else k0 - 1                             # the stencil's first sample
    span_lo = min(estart, st_lo)
    k0_span = k0 - span_lo                                               # the span index of k0 (kz)
    t_split = int(np.searchsorted(u_idx, k0, side='right'))            # the first requested sample above 0 V
    n_t = len(u_idx)

    dos3 = cp.ascontiguousarray(cp.asarray(ldos, dtype=cp.float32)[None, :, None])   # (n_atoms, n_e_all, N_ORB)
    dists_c = cp.asarray([[d]], dtype=cp.float32)                      # (n_shift * n_atoms, n_pts_all)
    phi_c = cp.asarray([phi], dtype=cp.float32)                        # (n_pts_all,)
    js_gpu = cp.asarray(u_idx - span_lo, dtype=cp.int32)               # the requested samples as span indices
    out = cp.zeros((1, n_t, 1, 1), dtype=cp.float32)                   # (n_p, n_t, n_atoms, N_ORB): dI/dV
    cur = cp.zeros((1, n_t, 1), dtype=cp.float32)                      # (n_p, n_t, n_atoms): the current

    linearized = model == 'linearized'
    kern = (_linearized_sweep_kernel if linearized else _exact_sweep_kernel)(dos3.shape[2])
    n_thr = 1 if linearized else n_t              # the threads per atom: one tip point, or one per requested V
    kern(
        ((n_thr + 63) // 64, 1), (64,),
        (dos3, dists_c, phi_c, js_gpu, out, cur, np.float32(KAPPA0_PER_SQRT_EV), step32, e032,
         np.int32(1), np.int32(1), np.int32(1), np.int32(0), np.int32(1),
         np.int32(dos3.shape[1]), np.int32(span_lo), np.int32(k0_span),
         np.int32(n_t), np.int32(t_split), np.int32(1)))
    cp.cuda.get_current_stream().synchronize()
    didv = cp.asnumpy(out[0, :, 0, 0]).astype(np.float64)
    current = cp.asnumpy(cur[0, :, 0]).astype(np.float64)
    return energies[u_idx], current, didv


class simulator_decays:
    """
    The simulator's two Decay-on models behind the figure's model interface.  Nothing is recomputed here: the
    weights are decay_weights (the factor functions) and I, dI/dV are sweep_current_didv (the kernels).
    model: 'exact' or 'linearized'; phi: the barrier at the tip point (eV); d: the atom-to-tip distance (A).
    Interface (the same for wrong_decays): name, barrier_kind ('sloped' or 'flat'), topo_mode ('frozen'),
    sweep_mode ('family'), has_current, set_grid(energies), bias_end(V), weights(idx, V), current_didv().
    """
    MODELS = ('exact', 'linearized')

    def __init__(self, model, phi, d):
        if model not in self.MODELS:
            raise ValueError(f"model must be one of {self.MODELS}, not {model!r}")
        self.model, self.phi, self.d = model, float(phi), float(d)
        self.name = model
        self.barrier_kind = 'flat' if model == 'linearized' else 'sloped'
        self.topo_mode = 'frozen'            # the topography at one frozen bias (line 73 of simulator.py)
        self.sweep_mode = 'family'           # every state's weight changes with the swept bias (line 74)
        self.has_current = True
        # The equations, as the figure prints them (bold textbook mathtext): the barrier, and the barrier to
        # the weight (gpu_exact_factor's and gpu_linearized_factor's docstrings).
        if model == 'exact':
            self.barrier_eq = r'$\mathbf{\phi - E + V\,z/d,\quad 0 \leq z \leq d}$'
            self.weight_eq = (r'$\mathbf{w = e^{-S},\quad S = \frac{4kd}{3V}\left[(\phi - E + V)^{3/2}'
                              r' - (\phi - E)^{3/2}\right],\quad k = \sqrt{2 m_e e}/\hbar}$')
        else:
            self.barrier_eq = r'$\mathbf{\bar{\phi} - \epsilon = \phi - E + V/2}$'
            self.weight_eq = (r'$\mathbf{w = e^{-2\kappa_0 d + \kappa_0 (E - V/2)\, d/\phi},'
                              r'\quad \kappa_0 = k\sqrt{\phi}}$')
        # The weights to I and dI/dV (the sweep kernels, simulator.py's _calculate_ldos_at_points_gpu): the
        # same two integrals for both models (the linearized kernel's factorization of w is its route to them).
        self.current_eq = (r'$\mathbf{I(V) = \int_0^V \rho(E)\, w(E, V)\, dE}$' + '\n'
                           + r'$\mathbf{\frac{dI}{dV}(V) = \rho(V)\, w(V, V) + \int_0^V \rho(E)\,'
                           r' \frac{\partial w(E, V)}{\partial V}\, dE}$')

    def set_grid(self, energies, v_max):
        """The LDOS grid: the samples' float32 GPU energies, as the simulator forms them (v_max unused here)."""
        self.energies = np.asarray(energies, dtype=np.float64).ravel()
        k0, step, e0, step32, e032 = grid_parameters(self.energies)
        self.e_gpu = gpu_energies(len(self.energies), k0, step32, e032)

    def bias_end(self, V):
        """The barrier's height at its bias end for bias V (phi + V), or, linearized, its flat height."""
        b, a, mean = barrier_heights(0.0, V, self.phi)
        return mean if self.barrier_kind == 'flat' else a

    def applied_bias(self, V, topo=False):
        """The bias colour of the decay at bias V: the signed V, the bias these factors are passed
        (gpu_exact_factor's a = phi - E + V), for the topography and the sweep alike (topo is ignored)."""
        return V

    def weights(self, idx, V):
        """w(E, V) at the samples idx (indices into the grid) for the bias V."""
        return decay_weights(self.model, self.e_gpu[idx], V, self.phi, self.d)

    def current_didv(self, ldos):
        """(v, I, dI/dV) at the model's own samples (every grid sample but 0 V), from the kernels."""
        return sweep_current_didv(self.model, self.energies, ldos, self.phi, self.d)


# ==== THE INTEGRATED CURRENT ====
def integrated_current(ldos, k0, step32, e032):
    """
    The integrated current of a displayed LDOS: I(V) = the integral from 0 V to V of the LDOS at every sample,
    oriented (negative below 0 V for a positive LDOS, 0 at a sample exactly at 0 V), by feenstra.current_from_ldos
    (the package's quadrature.integral_from_zero rule).  ldos: one row on the grid (float64 numpy); k0, step32,
    e032 as grid_parameters gives them.  Returns a float64 numpy array over the grid.
    """
    ldos = np.asarray(ldos, dtype=np.float64).ravel()
    i_all = feenstra.current_from_ldos(cp.asarray(ldos[None, :], dtype=cp.float32), k0, step32, e032)
    return cp.asnumpy(i_all)[0].astype(np.float64)


# ==== THE SUPERSEDED MODELS ====
class wrong_decays:
    """
    The superseded decays behind the figure's model interface: 'oldest', the earliest stm_from_dft package's
    simulator.py (its gpu_chen_tunneling_factor and _calculate_ldos_at_points_gpu); Tier 2 ('t2') and Tier 2
    with the occupied-side 0-clamp ('t2c') from stm_from_dft/wrong_decays/tilted_barrier_decay.py; Tier 3
    ('t3') from wrong_decays/tier3_decay.py (its Feenstra division, which lives only in sim_decay_full.py,
    left out); and McDowell's VASP_ldos_map ('mcdowell': ldos_map/lib.py's tunneling_factor as ldos_line.py
    and ldos_map.py use it).  Those sources are commented out and bound to a simulator instance (or are a
    separate package), so the factor functions are copied here verbatim (gpu_chen_tunneling_factor_v0 from the
    oldest package, gpu_chen_tunneling_factor, gpu_simpson, tunneling_factor) and the usages are transcribed
    from their call sites, one image at distance d, one tip point with barrier phi, one LDOS row.

    Not Chen.  What tilted_barrier_decay.py and tier3_decay.py call gpu_chen_tunneling_factor is the WKB
    average of 2 kappa over a barrier rising linearly from phi - E to phi - E + V (the 3/2-power form), not
    the model of C. J. Chen's book (that is the simulator's 'linearized', Eqs. 2.34-2.40); the name is kept
    only because the copy is verbatim.  McDowell's tunneling_factor is the same 3/2-power form without the
    V = 0 limit.

    The usages (V and E as each code passes them to its factor; w = exp(-K d 1e-10), d in Angstrom):
      oldest    one bias for the whole window, E signed: the topography's window [0, V_b] gives V = |V_b|
                (frozen); the display's window [emin, emax] gives V = emax - emin, its width W, the same for
                every displayed state, never referenced to 0 V (frozen at W)
      t2        every state at its own bias: V = |E|, E_arg = E              (per state, topo and display alike)
      t2c       the same with E_arg = max(E, 0), the '0-clamp'               (per state, topo and display alike)
      t3        frozen at the bias: V = |V|, E = eps + max(-V, 0)            (a family over V; topo frozen at V_b)
      mcdowell  line (the LDOS): V = |E|, E = |E|                            (per state)
                map (the topo):  V = |V_b|, E = |E|                          (frozen at V_b)
    Currents: t3 its own, the signed Simpson column I(V_j) of _t3_display_channel, dI/dV its central
    differences; oldest, t2, t2c and mcdowell the integrated current, feenstra.current_from_ldos on the
    displayed LDOS rho(E) w(E) (the old simulator's _feenstra_current did this for t2 and t2c; the oldest
    package integrates only its topography, with gpu_simpson; McDowell's package has no current of its own,
    ldos_line integrates nothing and ldos_map's window sum is a topography), their dI/dV the displayed LDOS
    itself.
    The 0 V fix for McDowell: his line factor is 0/0 at a sample exactly at 0 V (V = |E| = 0; a DOSCAR grid
    never lands exactly on E_F, so his code never meets it), and the integral from 0 V needs that sample, so
    there K is the V -> 0 limit of his own expression, 4 pi sqrt(2 m (phi - E)) / h with his constants, the
    fix tilted_barrier_decay.py's gpu_chen_tunneling_factor carries for the same 0/0 (its K_limit).
    """
    MODELS = ('oldest', 't2', 't2c', 't3', 'mcdowell')
    TITLES = {'oldest': 'oldest', 't2': 'Tier 2', 't2c': 'Tier 2, 0-clamp', 't3': 'Tier 3', 'mcdowell': 'McDowell'}

    # ---- verbatim: the earliest stm_from_dft package, simulator.py, gpu_chen_tunneling_factor (the body as it
    #      was; the attribute name carries _v0 only because the later version below keeps the original name) ----
    @staticmethod
    def gpu_chen_tunneling_factor_v0(V, E, phi):
        """Vectorized Julian Chen barrier model character-for-character."""
        me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
        V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
        V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
        prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
        term1 = cp.power(cp.maximum(0.0, phi_j - E_j + V_j), 1.5)
        term2 = cp.power(cp.maximum(0.0, phi_j - E_j), 1.5)
        return prefactor * (term1 - term2)

    # ---- verbatim: stm_from_dft/wrong_decays/tilted_barrier_decay.py, gpu_chen_tunneling_factor ----
    @staticmethod
    def gpu_chen_tunneling_factor(V, E, phi):
        """Vectorized Julian Chen barrier model character-for-character, in float64.

        No floor on V: every V != 0 goes through the formula as written; at V = 0 exactly (where the
        formula is 0/0) K is its exact V -> 0 limit, 4*pi*sqrt(2*m*max(0, phi - E))/h, the 2*kappa
        of a rectangular barrier of height phi - E.
        """
        me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
        V = cp.abs(cp.asarray(V, dtype=cp.float64))
        E = cp.asarray(E, dtype=cp.float64)
        phi = cp.asarray(phi, dtype=cp.float64)
        at_zero = V == 0.0
        V_formula = cp.where(at_zero, 1.0, V)      # V = 0 entries only keep the formula finite; they are replaced by the limit below
        V_j, E_j, phi_j = V_formula * q, E * q, phi * q
        prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
        term1 = cp.power(cp.maximum(0.0, phi_j - E_j + V_j), 1.5)
        term2 = cp.power(cp.maximum(0.0, phi_j - E_j), 1.5)
        K_limit = 4.0 * cp.pi * cp.sqrt(2.0 * me * cp.maximum(0.0, phi_j - E_j)) / h
        return cp.where(at_zero, K_limit, prefactor * (term1 - term2))

    # ---- verbatim: stm_from_dft/wrong_decays/sim_decay_full.py, gpu_simpson ----
    @staticmethod
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

    # ---- verbatim: VASP_ldos_map/ldos_map/lib.py, tunneling_factor (it multiplies its arguments in place,
    #      so it is given fresh arrays) ----
    @staticmethod
    def tunneling_factor(V, E, phi):
        V *= 1.60218e-19
        E *= 1.60218e-19
        phi *= 1.60218e-19
        prefactor = 8 / 3 / V * np.pi * np.sqrt(2 * 9.11e-31) / 6.626e-34
        barrier = (phi - E + V) ** (3 / 2) - (phi - E) ** (3 / 2)
        return prefactor * barrier

    def __init__(self, model, phi, d):
        if model not in self.MODELS:
            raise ValueError(f"model must be one of {self.MODELS}, not {model!r}")
        self.model, self.phi, self.d = model, float(phi), float(d)
        self.name = self.TITLES[model]
        self.barrier_kind = 'sloped'                    # every one a linearly tilted barrier
        # The equations, as the figure prints them (bold textbook mathtext): the barrier, and the barrier to
        # the weight, the same 3/2-power K for every one of these models, each with its own V and E.
        K_EQ = (r'$\mathbf{w = e^{-Kd},\quad K = \frac{8\pi\sqrt{2 m_e}}{3hV}\left[(\phi - E + V)^{3/2}'
                r' - (\phi - E)^{3/2}\right]}$')
        # The weights to I and dI/dV, as current_didv forms them: dI/dV is the displayed decayed LDOS and I its
        # integrated current (oldest, t2, t2c, mcdowell); Tier 3's Simpson column I, then central differences.
        LD_EQ = (r'$\mathbf{\frac{dI}{dV}(V) = \rho(V)\, w(V)}$' + '\n'
                 + r'$\mathbf{I(V) = \int_0^V \rho(E)\, w(E)\, dE = -\int_V^0 \rho(E)\, w(E)\, dE}$')
        if model == 'oldest':
            self.topo_mode, self.sweep_mode, self.has_current = 'frozen', 'frozen', True
            self.barrier_eq = (r'$\mathbf{\phi - E + V\,z/d,\quad V_{topo} = |V_b|,'
                               r'\quad V_{\rho} = W = E_{max} - E_{min}}$')
            self.weight_eq = K_EQ + '\n' + r'$\mathbf{V_{topo} = |V_b|,\quad V_{\rho} = W}$'
            self.current_eq = (r'$\mathbf{\frac{dI}{dV}(V) = \rho(V)\, w(V; W)}$' + '\n'
                               + r'$\mathbf{I(V) = \int_0^V \rho(E)\, w(E; W)\, dE'
                               r' = -\int_V^0 \rho(E)\, w(E; W)\, dE}$')
        elif model in ('t2', 't2c'):
            self.topo_mode, self.sweep_mode, self.has_current = 'per_state', 'per_state', True
            if model == 't2':
                self.barrier_eq = r'$\mathbf{\phi - E + |E|\,z/d}$'
                self.weight_eq = K_EQ + '\n' + r'$\mathbf{V = |E|,\quad E_{arg} = E}$'
            else:
                self.barrier_eq = r'$\mathbf{\phi - \max(E, 0) + |E|\,z/d}$'
                self.weight_eq = K_EQ + '\n' + r'$\mathbf{V = |E|,\quad E_{arg} = \max(E, 0)}$'
            self.current_eq = LD_EQ
        elif model == 't3':
            self.topo_mode, self.sweep_mode, self.has_current = 'frozen', 'family', True
            self.barrier_eq = r"$\mathbf{\phi - E' + |V|\,z/d,\quad E' = \epsilon + \max(-V, 0)}$"
            self.weight_eq = (K_EQ + '\n' + r"$\mathbf{V \to |V|,\quad E \to E' = \epsilon + \max(-V, 0)}$" + '\n'
                              + r'$\mathbf{K(|V|,\ \epsilon + |V|) = K(V, \epsilon)\quad (V < 0)}$')
            self.current_eq = (r'$\mathbf{I(V) = \int_0^V \rho(\epsilon)\, w(\epsilon; V)\, d\epsilon}$' + '\n'
                               + r'$\mathbf{\frac{dI}{dV}(V_j) = \frac{I(V_{j+1}) - I(V_{j-1})}{2\,\Delta E}}$')
        else:
            self.topo_mode, self.sweep_mode, self.has_current = 'frozen', 'per_state', True
            self.barrier_eq = (r'$\mathbf{\phi - |E| + |V|\,z/d,\quad V_{topo} = |V_b|,'
                               r'\quad V_{\rho} = |E|}$')
            self.weight_eq = (K_EQ + '\n'
                              + r'$\mathbf{E \to |E|,\quad V_{topo} = |V_b|,\quad V_{\rho} = |E|}$')
            self.current_eq = LD_EQ

    def set_grid(self, energies, v_max):
        """The LDOS grid (these models used the DOSCAR's energies as they are, the oldest one in float32) and
        the display window's reach v_max: the oldest model's display bias is the window's width, 2 v_max."""
        self.energies = np.asarray(energies, dtype=np.float64).ravel()
        self.k0, self.step, self.e0, self.step32, self.e032 = grid_parameters(self.energies)
        self.display_bias = 2.0 * float(v_max)

    def bias_end(self, V):
        """The barrier's height at its bias end: these models give the bias as |V| (and the state as |E| or
        E_arg), so the vacuum level at the bias end is phi + |V|."""
        return self.phi + np.abs(V)

    def applied_bias(self, V, topo=False):
        """The bias colour of the decay at bias (or state energy) V: the bias these models actually pass to
        their factor there, |V| (|E| per state for t2, t2c and McDowell's line; |V_j| for Tier 3's columns;
        |V_b| for McDowell's map and the oldest's topography); the oldest's display is the exception, one fixed
        bias, the window's width W, for every state (topo=True asks for its topography's |V_b| instead)."""
        if self.model == 'oldest' and not topo:
            return np.full_like(np.asarray(V, dtype=np.float64), self.display_bias)
        return np.abs(np.asarray(V, dtype=np.float64))

    # -- the factor as each usage calls it --
    def _K_tier(self, V, E_arg):
        """K of gpu_chen_tunneling_factor (1/m) for biases V and state arguments E_arg (arrays that
        broadcast against each other), one tip point with barrier phi."""
        K = self.gpu_chen_tunneling_factor(cp.asarray(V, dtype=cp.float64), cp.asarray(E_arg, dtype=cp.float64),
                                           cp.asarray([self.phi], dtype=cp.float64))
        return cp.asnumpy(K).astype(np.float64)

    def _K_mcdowell(self, V, E):
        """K of McDowell's tunneling_factor (1/m) for the biases V and states E (fresh float64 copies, since
        the function multiplies its arguments in place), barrier phi.  Where V is exactly 0 (the 0/0 of his
        expression) K is its V -> 0 limit, 4 pi sqrt(2 m (phi - E)) / h, with his constants (the 0 V fix)."""
        V = np.array(V, dtype=np.float64, copy=True)
        E = np.array(E, dtype=np.float64, copy=True)
        V, E = np.broadcast_arrays(V, E)
        K = np.empty(V.shape, dtype=np.float64)
        nz = V != 0.0
        if np.any(nz):
            K[nz] = self.tunneling_factor(V[nz].copy(), E[nz].copy(), np.array(self.phi, dtype=np.float64))
        if np.any(~nz):
            q = 1.60218e-19
            K[~nz] = 4.0 * np.pi * np.sqrt(2.0 * 9.11e-31 * np.maximum(0.0, (self.phi - E[~nz]) * q)) / 6.626e-34
        return K

    def weights_per_state(self, idx):
        """w(E) at the samples idx when every state carries its own bias.
        t2 / t2c: _compute_channel, K_all = gpu_chen_tunneling_factor(|E|, e_arg, phi), sf = exp(-d K 1e-10);
        mcdowell: ldos_line.py line 285, K = tunneling_factor(|E|, |E|, phi), sf = exp(-d K 1e-10)."""
        E = self.energies[idx]
        if self.model in ('t2', 't2c'):
            e_arg = np.maximum(E, 0.0) if self.model == 't2c' else E
            K = self._K_tier(np.abs(E)[:, None], e_arg[:, None]).reshape(-1)
        elif self.model == 'mcdowell':
            K = self._K_mcdowell(np.abs(E), np.abs(E))
        else:
            raise ValueError(f"{self.model!r} has no per-state weight")
        return np.exp(-1.0 * self.d * K * 1.0e-10)

    def weights(self, idx, V):
        """w(E; V) at the samples idx for the bias V, when the bias is frozen or swept as a family.
        oldest: _compute_channel of the earliest package, K = gpu_chen_tunneling_factor_v0(bias_v, E, phi) with
          bias_v = emax - emin of its window (|V_b| for the topography, W for the display: V here is that
          width), E the window's float32 energies, sf = exp(-d K 1e-10);
        t3: _frozen_K, K = gpu_chen_tunneling_factor(|V|, eps + max(-V, 0), phi), float32, sf = exp(-K d 1e-10);
        mcdowell (the map, ldos_map.py line 230): K = tunneling_factor(|V|, |E|, phi), sf = exp(-d K 1e-10)."""
        E = self.energies[idx]
        if self.model == 'oldest':
            bias_v = cp.asarray(abs(float(V)), dtype=cp.float32)
            K = self.gpu_chen_tunneling_factor_v0(bias_v, cp.asarray(E, dtype=cp.float32),
                                                  cp.asarray([self.phi], dtype=cp.float32))
            K = cp.asnumpy(K).reshape(-1).astype(np.float64)
            return np.exp(-1.0 * self.d * K * 1.0e-10)
        if self.model == 't3':
            K = self._K_tier(abs(float(V)), (E + max(-float(V), 0.0))[:, None]).reshape(-1).astype(np.float32)
            return np.exp(-K * self.d * 1.0e-10).astype(np.float64)
        if self.model == 'mcdowell':
            K = self._K_mcdowell(abs(float(V)), np.abs(E))
            return np.exp(-1.0 * self.d * K * 1.0e-10)
        raise ValueError(f"{self.model!r} has no frozen-bias weight")

    def current_didv(self, ldos):
        """(v, I, dI/dV) at the model's own samples.
        oldest / t2 / t2c / mcdowell: the displayed LDOS rho(E) w(E) is dI/dV (the oldest's w(E; W) at the
          display width W), and I is its integrated current, integrated_current (feenstra.current_from_ldos,
          the integral from 0 V), at every sample but 0 V (McDowell's 0 V sample itself carries the 0 V fix,
          see the class docstring);
        t3: _t3_display_channel: I(V_j) = sign(V_j) * gpu_simpson of rho(eps) w(eps; V_j) over the column of
          samples with 0 <= sign(V_j) eps <= |V_j| (the 0 V sample included; I = 0 at V_j = 0), at every grid
          sample, then dI/dV by central differences, one-sided at the two ends."""
        ldos = np.asarray(ldos, dtype=np.float64).ravel()
        n = len(self.energies)
        all_idx = np.arange(n, dtype=np.int64)
        shown = all_idx[all_idx != self.k0] if self.e0 == 0.0 else all_idx
        if self.model in ('oldest', 't2', 't2c', 'mcdowell'):
            w = (self.weights(all_idx, self.display_bias) if self.model == 'oldest'
                 else self.weights_per_state(all_idx))
            ld = ldos * w
            i_all = integrated_current(ld, self.k0, self.step32, self.e032)
            return self.energies[shown], i_all[shown], ld[shown]
        if self.model == 't3':
            d_e = float(self.energies[1] - self.energies[0])
            i_of_v = np.zeros(n, dtype=np.float64)
            for j in range(n):
                v_j = float(self.energies[j])
                if v_j > 0:
                    col = np.nonzero((self.energies >= 0.0) & (self.energies <= v_j))[0]
                elif v_j < 0:
                    col = np.nonzero((self.energies <= 0.0) & (self.energies >= v_j))[0]
                else:
                    continue                                   # V_j = 0: empty column window, I(V_j) = 0
                if len(col) == 0:
                    continue
                sgn = 1.0 if v_j > 0 else -1.0
                y = (ldos[col] * self.weights(col, v_j)).astype(np.float32)[None, :]
                i_of_v[j] = sgn * float(cp.asnumpy(self.gpu_simpson(
                    cp.asarray(y), cp.asarray(self.energies[col], dtype=cp.float32)))[0])
            didv = np.zeros_like(i_of_v)
            if n >= 2:
                didv[0] = (i_of_v[1] - i_of_v[0]) / d_e
                didv[n - 1] = (i_of_v[n - 1] - i_of_v[n - 2]) / d_e
                if n > 2:
                    didv[1:n - 1] = (i_of_v[2:] - i_of_v[:n - 2]) / (2.0 * d_e)
            return self.energies, i_of_v, didv
        raise ValueError(f"no current for {self.model!r}")


# ==== PLACEHOLDERS ====
def synthetic_ldos(energies, centers, amp, sigma, rho0):
    """Placeholder LDOS: a flat continuum rho0 plus one Gaussian of amplitude amp and width sigma (eV) at each
    center (eV from E_F).  With centers at +-E the states on the two sides of E_F have equal amplitude and
    width; a center at the grid's end is cut in half by it."""
    e = np.asarray(energies, dtype=np.float64)
    rho = np.full_like(e, float(rho0))
    for c in centers:
        rho += amp * np.exp(-0.5 * ((e - c) / sigma) ** 2)
    return rho


# ==== THE FIGURE ====
def three_level_ticks(ax, axis, unit):
    """
    Three tick levels on one axis of ax ('x' or 'y') at spacings unit, unit/2 and unit/10: unit long, thick
    and labelled (the major ticks), unit/2 medium (drawn as centred bar markers on the axis line), unit/10
    short and thin (the minor ticks); the ticks centred on the axis ('inout') so the three levels read as one
    family; a grid behind the data at the same three spacings with the same three weights.
    """
    lo, hi = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    ax_obj = ax.xaxis if axis == 'x' else ax.yaxis
    ax_obj.set_major_locator(MultipleLocator(unit))
    ax_obj.set_minor_locator(MultipleLocator(unit / 10.0))
    ax.tick_params(axis=axis, which='major', direction='inout', length=10, width=1.0)
    ax.tick_params(axis=axis, which='minor', direction='inout', length=4, width=0.4)
    ax.grid(True, axis=axis, which='major', color='0.72', lw=0.8, zorder=0)
    ax.grid(True, axis=axis, which='minor', color='0.88', lw=0.25, zorder=0)
    # The unit/2 level: the half-unit positions that are not whole units.
    k_lo, k_hi = int(np.ceil(min(lo, hi) / (unit / 2.0))), int(np.floor(max(lo, hi) / (unit / 2.0)))
    mids = np.array([k * unit / 2.0 for k in range(k_lo, k_hi + 1) if k % 2 != 0], dtype=np.float64)
    if axis == 'x':
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for m in mids:
            ax.axvline(m, color='0.80', lw=0.5, zorder=0)
        ax.plot(mids, np.zeros_like(mids), marker='|', ms=7, mew=0.7, ls='none', color='k', transform=tr,
                clip_on=False, zorder=5)
    else:
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for m in mids:
            ax.axhline(m, color='0.80', lw=0.5, zorder=0)
        ax.plot(np.zeros_like(mids), mids, marker='_', ms=7, mew=0.7, ls='none', color='k', transform=tr,
                clip_on=False, zorder=5)
    ax.set_axisbelow(True)
    if axis == 'x':
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)


def decade_unit(span):
    """The power of ten that makes a span read as a number between 1 and 10: the 'unit' of its three-level
    ticks (1, 0.5, 0.1 in the displayed units)."""
    span = float(abs(span))
    return 10.0 ** np.floor(np.log10(span)) if span > 0 else 1.0


def decay_barrier_figure(energies, ldos, models, v_topo, v_max=2.0, save_path=None):
    """
    The figure described at the top of this file, one section per model.
    energies: the LDOS's energy grid (eV from E_F), evenly spaced, two samples at least on each side of 0 V;
    ldos: the LDOS on it (one curve: the bare LDOS the decay acts on); models: the decay models to show
    (simulator_decays or wrong_decays instances, each carrying its phi and d; phi is read from the first);
    v_topo: the frozen topo bias (V); v_max: the sweep's reach on each side of 0 V (the sweep's biases are
    the grid samples within it).  save_path: a file to save the figure to (None: not saved).  Returns the
    matplotlib figure (also shown).
    """
    energies = np.asarray(energies, dtype=np.float64).ravel()
    ldos = np.asarray(ldos, dtype=np.float64).ravel()
    n = len(energies)
    k0, step, e0, step32, e032 = grid_parameters(energies)
    phi = models[0].phi
    for m in models:
        m.set_grid(energies, v_max)
    half = 0.5 * step

    def window(lo, hi):
        """The indices of the grid samples from lo to hi (eV), both included."""
        lo, hi = min(lo, hi), max(lo, hi)
        return np.nonzero((energies >= lo - half) & (energies <= hi + half))[0]

    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=-v_max, vmax=v_max)

    def color(energy):
        return cmap(norm(energy))

    def colored_curve(ax, x, y, values, lw=1.2):
        """The curve (x, y) drawn as segments, each coloured by its entry of `values` (one per segment), the
        values being the biases the decay applied."""
        pts = np.column_stack((x, y))
        segments = np.stack((pts[:-1], pts[1:]), axis=1)
        ax.add_collection(LineCollection(segments, colors=[color(v) for v in values], linewidths=lw))
        ax.update_datalim(pts)
        ax.autoscale_view()

    def outer_end(x):
        """For segments between consecutive x, the end farther from 0 V: the bias whose barrier the piece is."""
        return np.where(np.abs(x[1:]) >= np.abs(x[:-1]), x[1:], x[:-1])

    # The sweep's biases: every grid sample within +-v_max but the one at 0 V, each side ordered outward from
    # 0 V (the flat-barrier wedge and the weight shape are built from consecutive pairs on each side).
    in_range = window(-v_max, v_max)
    in_range = in_range[np.abs(energies[in_range]) > half]
    v_pos = np.sort(energies[in_range][energies[in_range] > 0.0])
    v_neg = np.sort(energies[in_range][energies[in_range] < 0.0])[::-1]
    sides = (('pos', v_pos), ('neg', v_neg))
    sweep_idx = np.sort(window(-v_max, v_max))              # the samples of the sweep's x range, 0 V included
    topo_idx = window(0.0, v_topo)

    # The compute, every model: the weights and I, dI/dV.
    results = {}
    for mi, m in enumerate(models):
        res = {}
        # Topo column.
        if m.topo_mode == 'frozen':
            res['w_topo'] = m.weights(topo_idx, v_topo)
        else:
            res['w_topo'] = m.weights_per_state(topo_idx)
        # Sweep column.
        if m.sweep_mode == 'frozen':
            res['w_sweep'] = (energies[sweep_idx], m.weights(sweep_idx, m.display_bias))
        elif m.sweep_mode == 'family':
            shape = {}
            for side, biases in sides:
                curves = []
                for v in biases:
                    sel = window(0.0, v)
                    curves.append((energies[sel], m.weights(sel, v)))
                shape[side] = (biases, curves)
            res['shape'] = shape
        else:
            idx = sweep_idx[np.abs(energies[sweep_idx]) > half]       # per-state weights: every sample but 0 V
            res['w_sweep'] = (energies[idx], m.weights_per_state(idx))
        res['v'], res['current'], res['didv'] = m.current_didv(ldos)
        results[mi] = res

    # Rows 0 and 1 (the bare LDOS and its integrated current): both drawn to one y scale, so their heights split
    # the two row units they share in the ratio of their y ranges, after a fixed band (key_units row units) is
    # set aside below the current's data floor for the colour key.  The current takes matplotlib's 5% margin
    # of its own span; the LDOS takes that same margin in data units, which at the common scale is the same
    # margin in inches above and below its data (5% of its own, much smaller, span would read as touching).
    i0 = integrated_current(ldos, k0, step32, e032)
    lo1, hi1 = float(ldos.min()), float(ldos.max())
    lo2, hi2 = float(i0.min()), float(i0.max())
    margin = 0.05 * (hi2 - lo2)
    lo1, hi1 = lo1 - margin, hi1 + margin
    lo2, hi2 = lo2 - margin, hi2 + margin
    key_units = 0.25                                            # the key band, in row units
    per_unit = ((hi1 - lo1) + (hi2 - lo2)) / (2.0 - key_units)  # data units per row unit, both panels
    h_ldos = (hi1 - lo1) / per_unit
    h_cur0 = (hi2 - lo2) / per_unit + key_units
    cur0_bottom = lo2 - key_units * per_unit                    # the current panel's axis, key band included

    # The grid of panels: the top block (rows 0 and 1 across the window, the LDOS over its current, close
    # together with only the current's title between them, their heights in the ratio above); then the
    # section rows, one row unit each, the topo and LDOS columns in the ratio |v_topo| : 2 v_max, the last
    # row's two panels equal.
    n_sec = len(models)
    EQ = 0.34                                                  # an equation strip, in row units
    fig = plt.figure(figsize=(8.0 * n_sec, 16 + 3.0 * EQ * 3.0))
    outer = GridSpec(7, n_sec, figure=fig, height_ratios=[h_ldos + h_cur0, EQ, 1, EQ, 1, EQ, 1],
                     width_ratios=[1] * n_sec, left=0.12 / n_sec, right=1.0 - 0.03 / n_sec, bottom=0.04,
                     top=0.945, hspace=0.5, wspace=0.14)
    top = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, :], height_ratios=[h_ldos, h_cur0], hspace=0.16)
    ax_ldos = fig.add_subplot(top[0, 0])
    ax_cur0 = fig.add_subplot(top[1, 0], sharex=ax_ldos)
    axes = []
    for col in range(n_sec):
        ratio = [abs(v_topo), 2.0 * v_max]
        g1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, col], width_ratios=ratio, wspace=0.22)
        g2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4, col], width_ratios=ratio, wspace=0.22)
        g3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[6, col], width_ratios=[1, 1], wspace=0.22)
        axes.append(dict(eq_b=fig.add_subplot(outer[1, col]), eq_w=fig.add_subplot(outer[3, col]),
                         eq_i=fig.add_subplot(outer[5, col]),
                         topo_b=fig.add_subplot(g1[0, 0]), ldos_b=fig.add_subplot(g1[0, 1]),
                         topo_w=fig.add_subplot(g2[0, 0]), ldos_w=fig.add_subplot(g2[0, 1]),
                         cur=fig.add_subplot(g3[0, 0]), didv=fig.add_subplot(g3[0, 1])))

    x_label = r'$E - E_F$ (eV)'
    topo_lim = (min(0.0, v_topo), max(0.0, v_topo))
    sweep_lim = (-v_max, v_max)

    def mark_phi(ax):
        """phi on a barrier panel: a dashed line at phi, labelled just above it at the panel's edge away from
        the data: the right edge for a positive topo bias (the barriers rise from phi at 0 V toward the right),
        the left edge for a negative one."""
        ax.axhline(phi, color='0.35', lw=1.0, ls='--', zorder=0.5)
        right = v_topo >= 0.0
        ax.annotate(r'$\phi$', xy=(0.99 if right else 0.01, phi), xycoords=('axes fraction', 'data'),
                    xytext=(0, 2), textcoords='offset points', color='0.35', fontsize=11,
                    ha='right' if right else 'left', va='bottom')

    def draw_barriers(ax, m, lo, hi):
        """The barriers of model m over the biases in [lo, hi] (one side or both), each from its bias to 0 V:
        sloped, the one curve phi -> bias_end(x) coloured by the bias whose end each piece is; flat, the
        strips between consecutive biases' flat heights in the inner bias's colour."""
        if m.barrier_kind == 'sloped':
            xs = energies[window(lo, hi)]
            colored_curve(ax, xs, m.bias_end(xs), m.applied_bias(outer_end(xs)), lw=1.8)
        else:
            for side, biases in sides:
                biases = biases[(biases >= min(lo, hi) - half) & (biases <= max(lo, hi) + half)]
                for k in range(len(biases) - 1):
                    h_k, h_next = m.bias_end(biases[k]), m.bias_end(biases[k + 1])
                    ax.fill_between([0.0, biases[k]], [h_k, h_k], [h_next, h_next],
                                    color=color(m.applied_bias(biases[k])), lw=0, edgecolor='none')

    # Row 0: the input LDOS, no weight, in black; no x tick labels (row 1 below shares the x axis).
    ax_ldos.plot(energies, ldos, color='k', lw=1.2)
    ax_ldos.set_xlim(energies[0], energies[-1])
    ax_ldos.set_ylim(lo1, hi1)
    ax_ldos.set_title(r'$\rho$, no tunneling decay')
    ax_ldos.set_ylabel(r'$\rho$ (arb.)')
    ax_ldos.tick_params(axis='x', labelbottom=False)
    # Row 1: the integrated current of the bare LDOS, in black (no bias is applied to it, it is only integrated
    # over V; the bias colour means a decay's bias); the colour key under its data floor: the top of the label
    # 'vbias' at the floor, the label below it, the bar below the label; y tick labels below the floor
    # suppressed so the band reads as the key's strip.
    ax_cur0.plot(energies, i0, color='k', lw=1.2)
    ax_cur0.set_xlim(energies[0], energies[-1])
    ax_cur0.set_ylim(cur0_bottom, hi2)
    ax_cur0.set_yticks([t for t in ax_cur0.get_yticks() if lo2 - 1e-12 <= t <= hi2 + 1e-12])
    ax_cur0.set_ylim(cur0_bottom, hi2)
    ax_cur0.set_title(r'$I(V) = \int_0^V \rho(E)\,dE = -\int_V^0 \rho(E)\,dE$, no tunneling decay')
    ax_cur0.set_xlabel('V (V)')
    ax_cur0.set_ylabel('$I$ (arb.)')
    ax_cur0.text(0.005, lo2, 'vbias', transform=mtransforms.blended_transform_factory(ax_cur0.transAxes,
                                                                                       ax_cur0.transData),
                 ha='left', va='top', fontsize=10, color='k')
    key_y = lo2 - 0.72 * key_units * per_unit                 # the bar's height, data units, inside the band
    key_pts = np.column_stack((energies, np.full(n, key_y)))
    key_segments = np.stack((key_pts[:-1], key_pts[1:]), axis=1)
    ax_cur0.add_collection(LineCollection(key_segments, colors=[color(v) for v in 0.5 * (energies[:-1] + energies[1:])],
                                          linewidths=6, antialiaseds=False, capstyle='butt'))

    for mi, m in enumerate(models):
        ax = axes[mi]
        res = results[mi]

        # The equation strips: bold textbook mathtext (STIX), larger than any axis label; the barrier equation
        # above the barrier row, the barrier-to-weight equation between the barrier row and the weights row,
        # the weights-to-I, dI/dV equations between the weights row and the I, dI/dV row.
        for key, eq in (('eq_b', m.barrier_eq), ('eq_w', m.weight_eq), ('eq_i', m.current_eq)):
            ax[key].axis('off')
            ax[key].text(0.5, 0.5, eq, transform=ax[key].transAxes, ha='center', va='center', fontsize=15,
                         fontweight='bold', math_fontfamily='stix', linespacing=1.4)

        # Row 1: the barriers, phi marked.
        c_topo = color(m.applied_bias(v_topo, topo=True))    # the bias the topo decay passed to its factor
        if m.topo_mode == 'frozen':
            # One barrier at the frozen bias, in that bias's colour.
            h = m.bias_end(v_topo)
            ys = [phi, h] if m.barrier_kind == 'sloped' else [h, h]
            ax['topo_b'].plot([0.0, v_topo], ys, color=c_topo, lw=1.8)
        else:
            # Each state keeps its own bias: the sweep's drawing restricted to the topo window.
            draw_barriers(ax['topo_b'], m, 0.0, v_topo)
        if m.sweep_mode == 'frozen':
            # The display's one frozen bias W: one barrier from (0 V, phi) to (W, bias_end(W)); this panel's x
            # axis keeps the sweep's width but is centred on the barrier (W/2), not on 0 V, so the whole barrier
            # is in view.
            W = m.display_bias
            ax['ldos_b'].plot([0.0, W], [phi, m.bias_end(W)], color=color(W), lw=1.8)
            ldos_b_lim = (0.5 * W - v_max, 0.5 * W + v_max)
        else:
            draw_barriers(ax['ldos_b'], m, -v_max, v_max)
            ldos_b_lim = sweep_lim
        for key in ('topo_b', 'ldos_b'):
            mark_phi(ax[key])
            ax[key].set_xlabel(x_label)
        ax['topo_b'].set_xlim(*topo_lim)
        ax['ldos_b'].set_xlim(*ldos_b_lim)
        ax['topo_b'].set_title(f'topo barrier, $V_b$ = {v_topo:+.2f} V')
        ax['ldos_b'].set_title(r'$\rho$ sweep barriers')
        ax['topo_b'].set_ylabel('barrier height (eV)')

        # Row 2: the weights.
        e_t = energies[topo_idx]
        if m.topo_mode == 'frozen':
            ax['topo_w'].plot(e_t, res['w_topo'], color=c_topo, lw=1.8)
        else:
            colored_curve(ax['topo_w'], e_t, res['w_topo'], m.applied_bias(0.5 * (e_t[:-1] + e_t[1:]), topo=True),
                          lw=1.8)
        if m.sweep_mode == 'frozen':
            e_s, w_s = res['w_sweep']
            ax['ldos_w'].plot(e_s, w_s, color=color(m.display_bias), lw=1.8)
        elif m.sweep_mode == 'family':
            for side in ('pos', 'neg'):
                biases, curves = res['shape'][side]
                for k in range(len(biases) - 1):
                    e_k, w_k = curves[k]
                    e_next, w_next = curves[k + 1]
                    n_common = len(e_k)                       # the common range: the shorter curve's samples
                    ax['ldos_w'].fill_between(e_k, w_k, w_next[:n_common], color=color(m.applied_bias(biases[k])),
                                              lw=0, edgecolor='none')
        else:
            e_s, w_s = res['w_sweep']
            colored_curve(ax['ldos_w'], e_s, w_s, m.applied_bias(0.5 * (e_s[:-1] + e_s[1:])), lw=1.4)
        ax['topo_w'].set_xlim(*topo_lim)
        ax['ldos_w'].set_xlim(*sweep_lim)
        ax['topo_w'].set_title('weights at $V_b$')
        ax['ldos_w'].set_title('weights, bias swept')
        ax['topo_w'].set_ylabel('weight $w(E, V)$')
        for key in ('topo_w', 'ldos_w'):
            ax[key].set_xlabel(x_label)

        # Row 3: I(V) and dI/dV(V).
        v = res['v']
        v_mid = m.applied_bias(0.5 * (v[:-1] + v[1:]))         # the colours: the bias the decay applied there
        if m.has_current and res['current'] is not None:
            colored_curve(ax['cur'], v, res['current'], v_mid)
        else:
            ax['cur'].text(0.5, 0.5, 'no current in this decay\n' + r'($\rho$ undefined at 0 V)',
                           transform=ax['cur'].transAxes, ha='center', va='center', color='0.5', fontsize=10)
        colored_curve(ax['didv'], v, res['didv'], v_mid)
        ax['cur'].set_xlim(*sweep_lim)
        ax['didv'].set_xlim(*sweep_lim)
        ax['cur'].set_title('$I(V)$')
        ax['didv'].set_title('$dI/dV(V)$')
        ax['cur'].set_ylabel('$I$ (arb.)')
        ax['didv'].set_ylabel('$dI/dV$ (arb.)')
        for key in ('cur', 'didv'):
            ax[key].set_xlabel('V (V)')

    # Shared y ranges: each panel type across every section; the whole weights row as one axis including 0.
    def share(keys, include_zero=False):
        lo = min(axes[mi][k].get_ylim()[0] for mi in range(n_sec) for k in keys)
        hi = max(axes[mi][k].get_ylim()[1] for mi in range(n_sec) for k in keys)
        if include_zero:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        for mi in range(n_sec):
            for k in keys:
                axes[mi][k].set_ylim(lo, hi)
        return lo, hi

    share(('topo_b',))
    share(('ldos_b',))
    w_lo, w_hi = share(('topo_w', 'ldos_w'), include_zero=True)
    cur_axes = [axes[mi]['cur'] for mi in range(n_sec) if models[mi].has_current]
    if cur_axes:
        lo = min(a_.get_ylim()[0] for a_ in cur_axes)
        hi = max(a_.get_ylim()[1] for a_ in cur_axes)
        for mi in range(n_sec):
            axes[mi]['cur'].set_ylim(lo, hi)
    d_lo, d_hi = share(('didv',))

    # Three-level ticks and grid: the weights row and the dI/dV panels.
    for mi in range(n_sec):
        for key in ('topo_w', 'ldos_w'):
            three_level_ticks(axes[mi][key], 'x', 1.0)
            three_level_ticks(axes[mi][key], 'y', decade_unit(w_hi - w_lo))
        three_level_ticks(axes[mi]['didv'], 'x', 1.0)
        three_level_ticks(axes[mi]['didv'], 'y', decade_unit(axes[mi]['didv'].get_ylim()[1] - d_lo))


    # A frame around each decay section: the union of its six panels' tight boxes (titles, ticks and labels
    # included), padded, drawn in figure coordinates.  The frames are boundaries: the spacing between sections
    # is widened, and the left margin moved, until every section's full extent clears its neighbours and the
    # figure's edge by the padding, so nothing of one section lies in another's frame.
    pad = 0.004
    # The gap between the two top panels holds only the current's title; it is widened just enough for that
    # title (the integral, taller than a line of text) to clear the rho panel's frame and ticks by the padding.
    top_hspace = 0.16
    for _ in range(12):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        title_top = ax_cur0.title.get_window_extent(renderer).transformed(inv).y1
        rho_bottom = ax_ldos.get_tightbbox(renderer).transformed(inv).y0
        if title_top + pad <= rho_bottom:
            break
        top_hspace += 0.02
        top = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, :], height_ratios=[h_ldos, h_cur0],
                                      hspace=top_hspace)
        ax_ldos.set_subplotspec(top[0, 0])
        ax_cur0.set_subplotspec(top[1, 0])
    wspace, left = 0.14, 0.12 / n_sec
    renderer = None
    for _ in range(12):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        boxes = [mtransforms.Bbox.union([axes[mi][k].get_tightbbox(renderer) for k in axes[mi]]).transformed(inv)
                 for mi in range(n_sec)]
        gaps = [boxes[mi + 1].x0 - boxes[mi].x1 for mi in range(n_sec - 1)]
        need_gap = min(gaps) < 3.0 * pad if gaps else False
        need_left = boxes[0].x0 < 2.0 * pad
        if not need_gap and not need_left:
            break
        if need_gap:
            wspace += 0.04
        if need_left:
            left += 2.0 * pad - boxes[0].x0 + pad
        outer.update(left=left, wspace=wspace)
    # The section's title, the model's name, sits above its frame as a panel title sits above its panel: centred,
    # the axes title pad above the frame's top edge.  The title's top is at the section's extent, the frame is
    # shortened to begin the title pad below the title, and the barrier equation is moved down its strip to
    # hang the same pad below the frame's top edge.
    title_pad = plt.rcParams['axes.titlepad'] / 72.0 / fig.get_size_inches()[1]     # points -> figure fraction
    inv = fig.transFigure.inverted()
    for mi, (bb, m) in enumerate(zip(boxes, models)):
        title = fig.text(0.5 * (bb.x0 + bb.x1), bb.y1 + pad, m.name, ha='center', va='top', fontsize=14,
                         fontweight='bold', color='0.2')
        frame_top = title.get_window_extent(renderer).transformed(inv).y0 - title_pad
        fig.add_artist(Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, frame_top - (bb.y0 - pad),
                                 transform=fig.transFigure, fill=False, lw=1.0, edgecolor='0.3'))
        strip = axes[mi]['eq_b'].get_position()
        eq = axes[mi]['eq_b'].texts[0]
        eq.set_va('top')
        eq.set_position((0.5, (frame_top - title_pad - strip.y0) / strip.height))

    fig.suptitle(f'Decay barriers: $\\phi$ = {phi:g} eV, d = {models[0].d:g} Å', y=0.985)
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


if __name__ == '__main__':
    # Synthetic placeholders: the barrier phi and the LDOS the decay acts on.  Replace with the simulator's
    # phi_local at a tip point and an LDOS from a cache or from the simulator (on that LDOS's energy grid).
    DECAYS = 'correct'   # 'correct': the simulator's exact and linearized models; 'wrong': Tier 2, Tier 2 0-clamp, Tier 3, McDowell
    PHI = 5.2            # the barrier at the tip point (eV)
    D = 6.0              # the atom-to-tip distance (Angstrom)
    V_TOPO = 2.0         # the frozen topo bias (V)
    V_MAX = 2.0          # the sweep's reach on each side of 0 V (V)
    STEP = 0.01          # the grid step (V); the grid runs -V_MAX .. V_MAX with a sample exactly at 0 V, the sweep's biases
    RHO0 = 1.0           # the flat continuum
    AMP = 1.0            # every state's amplitude
    N_STATES = 10        # states per side of the one at 0 V, evenly spaced out to +-V_MAX (the end pair cut by the grid)
    SIGMA_OVER_SPACING = 0.25   # every state's width as a fraction of the spacing; 0.25 puts the valleys at 0.27 of the peak
    SAVE_PATH = None     # e.g. r'C:\path\decay_barriers.png'; None: shown only

    n_half = int(round(V_MAX / STEP))
    energies = STEP * np.arange(-n_half, n_half + 1, dtype=np.float64)
    spacing = V_MAX / N_STATES
    centers = spacing * np.arange(-N_STATES, N_STATES + 1, dtype=np.float64)
    ldos = synthetic_ldos(energies, centers, AMP, SIGMA_OVER_SPACING * spacing, RHO0)
    if DECAYS == 'correct':
        models = [simulator_decays(name, PHI, D) for name in simulator_decays.MODELS]
    elif DECAYS == 'wrong':
        models = [wrong_decays(name, PHI, D) for name in wrong_decays.MODELS]
    else:
        raise ValueError(f"DECAYS must be 'correct' or 'wrong', not {DECAYS!r}")
    decay_barrier_figure(energies, ldos, models, V_TOPO, v_max=V_MAX, save_path=SAVE_PATH)
