# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization
Auto-detects ISPIN=1, ISPIN=2, and NCL/SOC calculations.

The window (Interactive_STM_Simulator) is drawn with the suite's shared Qt layer (qtgui, this
package's own copy: PySide6 + pyqtgraph): its panels are cells of one Qt layout and its controls
real Qt widgets (automatic layout: the one rule for every window, nothing placed by hand), every
slider of the old figure a number box, the orbital hover labels widgets.HoverLabels.

Tunneling model (Unified_STM_Simulator): the signal at a tip point is each atom's site- and
orbital-projected DOS weighted by the squared tunneling matrix element |M|^2 of every periodic
image of that atom, the tip DOS taken as constant (the s-wave tip of Tersoff and Hamann,
Phys. Rev. B 31, 805 (1985)).  With Decay on, |M|^2 is the exact tilted barrier, exp(-S) with S
the WKB exponent across a barrier rising linearly from phi - E at the sample to phi - E + V at
the tip (gpu_exact_factor); to first order in (E - V/2)/phi and V/phi it is Chen's
energy-dependent matrix element (C. J. Chen, Introduction to Scanning Tunneling Microscopy, 3rd
ed., Oxford 2021, Sec. 2.2.3-2.2.4, Eqs. 2.34-2.41; Eq. 2.37: kappa ~ kappa0 (1 - eps / (2 phi)),
eps the state's energy measured from the middle of the bias interval); with Decay off it is
exp(-2 kappa0 d), the same weight without its energy and bias dependence.  kappa0 =
sqrt(2 m phi) / hbar (Chen Eq. 2.38), phi the LOCPOT potential at the tip point minus E_F.  As in
a measurement, the topography is converged on the current at one frozen bias (the current of
Chen Eq. 2.40, with this weight), and the displayed LDOS is dI/dV with the bias swept (its
derivative at each bias).  The model is the block THE DECAY MODEL (physics) at the top of this
file; everything after it is NUMERICS AND GPU (no physics).

Developed by Benjamin Kafin
"""

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
import colorsys
from time import time

from .qtgui import qtplot as plt
from .qtgui import controls, widgets
from .qtgui.qtplot import QtWidgets, QtCore, QtGui

from .doscar_parser import SpinAwareDosParser, SpinMode
from .locpot_manager import LocpotManager
from . import cmap_shading
from . import contour
from . import feenstra
from . import quadrature
from . import ldos_decomp


# ==== THE DECAY MODEL (physics) ====
# What the simulator computes.  How it is computed (the DOS grid, the integration rule, the GPU, float32, the
# chunks, the caches) is everything after this block, under NUMERICS AND GPU (no physics).
# The tip DOS is constant (the s-wave tip of J. Tersoff and D. R. Hamann, Phys. Rev. B 31, 805 (1985)): at a tip
# point, the DOS rho_o(E) of each atom and orbital o (E in eV from E_F, read as the sample bias in volts) is
# weighted, for every periodic image of the atom at distance d (Angstrom) from the tip, at bias V, by the decay
# weight w(E, V), with phi the LOCPOT potential at the tip point minus E_F (eV) and
# kappa0 = KAPPA0_PER_SQRT_EV sqrt(phi):
#   Decay on:  w(E, V) = exp(-S(E, V)), the exact tilted barrier, with its V derivative dw/dV
#              (gpu_exact_factor, below);
#   Decay off: w = exp(-2 kappa0 d), the same weight without its energy and bias dependence (dw/dV = 0)
#              (gpu_decay_off_factor, below).
# As in a measurement (C. J. Chen, Introduction to Scanning Tunneling Microscopy, 3rd ed., Oxford University Press
# 2021), the topography is taken at one frozen bias and the displayed LDOS is dI/dV with the bias swept:
#   I(V)       = sum over the images of  integral_0^V (sum_o rho_o(E)) w(E, V) dE
#   dI/dV_o(V) = sum over the images of  rho_o(V) w(V, V)  +  integral_0^V rho_o(E) dw(E, V)/dV dE
# (dI/dV: the state entering at the window edge plus the change of every state already in [0, V]), both integrals
# from exactly 0 V, oriented (negative below 0 V for a positive DOS; 0 at V = 0).  The topography: at each tip
# point, the height at which I(V_b), taken positive, equals the setpoint, V_b the bias (every state weighted at
# that frozen bias, w(E, V_b)).  The displayed LDOS at V is dI/dV_o(V); 'Feenstra' divides it by I(V)/V
# (feenstra.py).
_ME, _H, _Q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
# sqrt(2 m_e e) / hbar in 1/Angstrom per sqrt(eV): kappa0 = KAPPA0_PER_SQRT_EV * sqrt(phi in eV) (Chen Eq. 2.38).
KAPPA0_PER_SQRT_EV = float(np.sqrt(2.0 * _ME * _Q) / (_H / (2.0 * np.pi)) * 1e-10)


def gpu_exact_factor(E, V, phi, d, topo):
    """
    Physics: the decay weight of a state at E (eV from E_F) of an atom image at distance d (Angstrom)
    from the tip, at bias V, with phi the local barrier (the LOCPOT potential at the tip point minus E_F,
    eV): the exact tilted barrier,
        S(E, V) = (4 k d / (3 V)) [a+^(3/2) - b+^(3/2)],   a = phi - E + V,   b = phi - E,   x+ = max(x, 0),
        w(E, V) = exp(-S(E, V)),   k = KAPPA0_PER_SQRT_EV,
    the WKB exponent 2 int kappa dz across a barrier rising linearly from phi - E at the sample to
    phi - E + V at the tip, with kappa = 0 wherever the state lies above the barrier: a state above one
    end sees only the part of the barrier above it; above both ends, w = 1.  The barrier's mean height,
    phi - E + V/2, is the height Chen's Eq. 2.40 linearizes about: to first order in (E - V/2)/phi and
    V/phi, S reduces to Chen's exponent (C. J. Chen, Introduction to Scanning Tunneling Microscopy, 3rd
    ed. (2021), Eqs. 2.34, 2.37-2.40).
    Evaluated in the forms that keep full precision when V is small (three branches):
      a > 0 and b > 0:  S = (4kd/3)(a + sqrt(ab) + b)/(sqrt a + sqrt b),
                        dS/dV = (2kd/3)(sqrt a + 2 sqrt b)/(sqrt a + sqrt b)^2  (no division by V);
      a <= 0 and b <= 0:  S = dS/dV = 0;
      otherwise (V != 0 here):  S as above,  dS/dV = (4kd/3)[(3/2) sqrt(a+)/V - (a+^(3/2) - b+^(3/2))/V^2];
    and dw/dV = -(dS/dV) w.
    topo (the bias frozen, the topography): returns w; otherwise (the bias swept, dI/dV): (w, dw/dV).
    E and V are scalars, the GPU energies of DOS samples (np.float32, e0 + (i - k0) step); phi and d are
    float32 CuPy arrays that broadcast (phi (1, n_pts), d (n_images, n_pts)); the result is float32.  The
    exact_sweep kernel computes the same S and dS/dV with the same formulas (exact_weight).
    """
    a = phi - E + V
    b = phi - E
    kd = KAPPA0_PER_SQRT_EV * d
    ap, bp = cp.maximum(a, 0.0), cp.maximum(b, 0.0)
    sa, sb = cp.sqrt(ap), cp.sqrt(bp)
    ss = sa + sb
    both = (a > 0.0) & (b > 0.0)                     # under the barrier at both ends
    none = (a <= 0.0) & (b <= 0.0)                   # above it at both ends: S = 0, w = 1
    diff = ap * sa - bp * sb                         # a+^(3/2) - b+^(3/2)
    S = cp.where(both, (4.0 / 3.0) * kd * (a + cp.sqrt(a * b) + b) / ss,
                 cp.where(none, 0.0, (4.0 / 3.0) * kd * diff / V))
    w = cp.exp(-S)
    if topo:
        return w
    dS = cp.where(both, (2.0 / 3.0) * kd * (sa + 2.0 * sb) / (ss * ss),
                  cp.where(none, 0.0, (4.0 / 3.0) * kd * (1.5 * sa / V - diff / (V * V))))
    return w, -dS * w


def gpu_linearized_factor(E, V, phi, d, topo):
    """
    Physics: the linearized decay weight of a state at E (eV from E_F) of an atom image at distance d
    (Angstrom) from the tip, at bias V, with phi the local barrier (the LOCPOT potential at the tip point
    minus E_F, eV):
        w(E, V) = exp(-2 kappa0 d + kappa0 (E - V/2) d / phi),   kappa0 = KAPPA0_PER_SQRT_EV sqrt(phi),
        dw/dV = -(kappa0 d / (2 phi)) w,
    kappa expanded to first order in eps / phi, eps = E - V/2 the state's energy from the middle of the
    bias window (C. J. Chen, Introduction to Scanning Tunneling Microscopy, 3rd ed. (2021), Eqs. 2.34,
    2.37-2.40).
    topo (the bias frozen, the topography): returns w; otherwise (the bias swept, dI/dV): (w, dw/dV).
    E and V are scalars, the GPU energies of DOS samples (np.float32, e0 + (i - k0) step); phi and d are
    float32 CuPy arrays that broadcast (phi (1, n_pts), d (n_images, n_pts)); the result is float32.
    """
    kappa0 = KAPPA0_PER_SQRT_EV * cp.sqrt(phi)
    w = cp.exp(-2.0 * kappa0 * d + kappa0 * (E - np.float32(0.5) * V) * d / phi)
    if topo:
        return w
    return w, -0.5 * kappa0 * d / phi * w


def gpu_decay_off_factor(phi, d):
    """
    Physics: the decay weight with Decay off of an atom image at distance d (Angstrom) from the tip, with phi
    the local barrier (the LOCPOT potential at the tip point minus E_F, eV):
        w = exp(-2 kappa0 d),   kappa0 = KAPPA0_PER_SQRT_EV sqrt(phi)   (Chen Eq. 2.38),
    the weight of gpu_exact_factor without its energy and bias dependence (dw/dV = 0): every state
    of the image, at every bias, carries the same w.
    phi: (n_pts,) float32 CuPy, one per tip point; d: (n_images, n_pts) float32 CuPy; the result is
    (n_images, n_pts) float32.
    """
    kappa0 = KAPPA0_PER_SQRT_EV * cp.sqrt(phi)
    return cp.exp(-2.0 * kappa0[None, :] * d)


# ==== NUMERICS AND GPU (no physics) ====
# --- CORE UTILITIES ---
# The exact_sweep kernel (Decay on, the bias swept): dI/dV_o(V) and I(V) as THE DECAY MODEL (physics) defines them,
# at chosen DOSCAR samples V, dI/dV per tip point, atom and orbital (out) and I per tip point and atom (cur,
# want_cur = 1), each summed over the atom's images, with w(E, V) = exp(-S(E, V)) the exact tilted barrier and
# dw/dV = -(dS/dV) w (gpu_exact_factor; the kernel's exact_weight computes the same S and dS/dV with the
# same formulas, in its three branches).
# The exact tilt is not separable in E and V (no w(E, 0) w(0, V) / w(0, 0) factor, no single walk from 0 V for
# every V), so each requested V is integrated on its own, in one pass that gives dI/dV and the current: for that
# V, w(E, V) and dw(E, V)/dV are evaluated once at each sample it needs, the stencil E-2, E-1, E1, E2 (and the
# sample at 0 V when there is one) and every sample from 0 V to V; the two integrands, rho_o(E) dw(E, V)/dV for
# each orbital and (sum_o rho_o(E)) w(E, V), are integrated from 0 V to V by the rule of
# quadrature.integral_from_zero, with the same value at 0 V and the same piece next to 0 V from that V's
# stencil values: with no sample at 0 V, f(0) from the cubic through E-2, E-1, E1, E2, and for k (the intervals
# from 0 V to V) odd the cubic's exact integral to E+-1 (Q+, Q-), for k even the unequal Simpson pair over
# (0, E+-1, E+-2) (intervals e0 + step and step above 0 V, -e0 and step below); with a sample at 0 V, the cubic
# through E-1, 0, E1, E2 (below: E-2, E-1, 0, E1) for k odd and the Simpson pair from the 0 V sample for k even;
# then the equal-step Simpson pairs outward to V, summed as they are walked.  A requested V at a sample at 0 V
# gives rho_o(0) w(0, 0) and current 0.  Nothing per sample is stored; the outputs hold only the requested samples.
# Which lines are which: the lines marked 'Physics' in the source transcribe the physics: exact_weight's
# S and dS/dV (and w, dw/dV), the two integrands, and the sums that make dI/dV and I.  Everything else is
# machinery: one thread per (tip point, atom), looping over the atom's images and the requested samples; out
# (n_p, n_t, n_atoms, N_ORB) += dI/dV; cur (n_p, n_t, n_atoms) += the current, each thread in its own atom's slot
# (summed over its images; the caller sums cur over atoms; no atomics); want_cur = 0 skips the current and does not
# touch cur (NULL then); the stencil, f(0), the pieces and the pairs (quadrature.integral_from_zero's rule); a
# sample's energy formed from its span index j as e0 + (j - kz) step in float32 (step, e0 and kz, the span index of
# k0, passed in; no energy array); float for every value, pointer and output, expf / sqrtf and the f suffix on
# every literal; the launch chunked by the caller by the (requested V, sample) pairs per tip point.
_EXACT_SWEEP_SRC = r'''
// Physics: the exact tilted barrier (gpu_exact_factor): a = phi - E + V, b = phi - E, kd = k d;
// w = exp(-S), dw/dV = -(dS/dV) w.  Three branches: under the barrier at both ends (the forms with no division
// by V), above it at both ends (S = dS/dV = 0), above one end only (V != 0 there).
__device__ __forceinline__ void exact_weight(const float E, const float V, const float ph, const float kd,
                                             float* w, float* dw)
{
    const float a = ph - E + V;
    const float b = ph - E;
    float S, dS;
    if (a > 0.0f && b > 0.0f) {
        const float sa = sqrtf(a), sb = sqrtf(b);
        const float ss = sa + sb;
        S = (4.0f / 3.0f) * kd * (a + sqrtf(a * b) + b) / ss;                       // Physics: S
        dS = (2.0f / 3.0f) * kd * (sa + 2.0f * sb) / (ss * ss);                     // Physics: dS/dV
    } else if (a <= 0.0f && b <= 0.0f) {
        S = 0.0f;                                                                    // Physics: S
        dS = 0.0f;                                                                   // Physics: dS/dV
    } else {
        const float ap = fmaxf(a, 0.0f), bp = fmaxf(b, 0.0f);
        const float sa = sqrtf(ap), sb = sqrtf(bp);
        const float diff = ap * sa - bp * sb;                                        // a+^(3/2) - b+^(3/2)
        S = (4.0f / 3.0f) * kd * diff / V;                                           // Physics: S
        dS = (4.0f / 3.0f) * kd * (1.5f * sa / V - diff / (V * V));                  // Physics: dS/dV
    }
    *w = expf(-S);                                                                   // Physics: w
    *dw = -dS * (*w);                                                                // Physics: dw/dV
}

// Machinery: the coefficients of quadrature.integral_from_zero's rule that depend only on e0 and step, with no
// sample at 0 V (th = -e0/step: 0 V lies th steps above sample k0): the cubic's value at 0 V (cf; its third
// term subtracted), its exact integrals from 0 V to E1 (qp) and from 0 V down to E-1 (qm; their second and
// fourth terms subtracted), and the unequal Simpson pairs over (0, E1, E2) (pu, hu) and (0, E-1, E-2) (pd, hd).
struct Rule {
    float cf[4], qp[4], qm[4], pu[3], pd[3], hu, hd;
};

// Machinery: SciPy 1.18.1's Simpson pair for unequal intervals h0, h1 (quadrature._pair) as coefficients:
// pair = h6 (y0 c[0] + y1 c[1] + y2 c[2]), h6 = (h0 + h1)/6, c = (2 - 1/(h0/h1), hsum (hsum/(h0 h1)), 2 - h0/h1),
// every quotient 0 where its divisor is 0.
__device__ __forceinline__ void pair_coef(const float h0, const float h1, float* c, float* h6)
{
    const float hsum = h0 + h1;
    const float hprod = h0 * h1;
    const float h0divh1 = (h1 != 0.0f) ? h0 / h1 : 0.0f;
    c[0] = 2.0f - ((h0divh1 != 0.0f) ? 1.0f / h0divh1 : 0.0f);
    c[1] = hsum * ((hprod != 0.0f) ? hsum / hprod : 0.0f);
    c[2] = 2.0f - h0divh1;
    *h6 = hsum / 6.0f;
}

// Machinery: the integral from 0 V over the first interval (k odd) or the first two (k even) on one side of
// 0 V (up: above, else below, oriented), quadrature.integral_from_zero's rule.  F0 .. F4: the integrand at the
// span indices kz - 2 .. kz + 2 (F0 only with a sample at 0 V, which is F2; with none, E-2, E-1, E1, E2 are
// F1 .. F4).
__device__ __forceinline__ float first_piece(const Rule& R, const bool exact0, const bool up, const bool odd,
                                             const float step, const float F0, const float F1, const float F2,
                                             const float F3, const float F4)
{
    if (exact0) {
        if (up)
            return odd ? (step / 24.0f) * (-F1 + 13.0f * F2 + 13.0f * F3 - F4)     // the cubic through E-1, 0, E1, E2, 0 V to E1
                       : step / 3.0f * (F2 + 4.0f * F3 + F4);                       // the pair from the 0 V sample
        const float dn = -step;
        return odd ? (dn / 24.0f) * (-F3 + 13.0f * F2 + 13.0f * F1 - F0)           // the cubic through E-2, E-1, 0, E1, 0 V to E-1
                   : dn / 3.0f * (F2 + 4.0f * F1 + F0);
    }
    if (odd)
        return up ? (step / 24.0f) * (R.qp[0] * F1 - R.qp[1] * F2 + R.qp[2] * F3 - R.qp[3] * F4)   // Q+
                  : (step / 24.0f) * (R.qm[0] * F1 - R.qm[1] * F2 + R.qm[2] * F3 - R.qm[3] * F4);  // Q- (oriented)
    const float f0 = R.cf[0] * F1 + R.cf[1] * F2 - R.cf[2] * F3 + R.cf[3] * F4;                    // f(0): the cubic at 0 V
    return up ? R.hu * (f0 * R.pu[0] + F3 * R.pu[1] + F4 * R.pu[2])                                 // the pair over (0, E1, E2)
              : -(R.hd * (f0 * R.pd[0] + F2 * R.pd[1] + F1 * R.pd[2]));                            // over (0, E-1, E-2), oriented
}

extern "C" __global__ void exact_sweep(
    const float* __restrict__ dos,      // (n_atoms, n_e_all, N_ORB): the DOS of each atom
    const float* __restrict__ dists,    // (n_shift * n_atoms, n_pts_all): image-tip distances (Angstrom), shift-major
    const float* __restrict__ phi,      // (n_pts_all): the local barrier at each tip point (eV)
    const int* __restrict__ js,         // (n_t): the requested samples as span indices, ascending, each once
    float* __restrict__ out,            // (n_p, n_t, n_atoms, N_ORB), zeroed: dI/dV for points p0 .. p0 + n_p
    float* __restrict__ cur,            // (n_p, n_t, n_atoms), zeroed: the current (want_cur = 1); NULL when want_cur = 0
    const float kappa0_per_sqrt_ev, const float step, const float e0,
    const int n_atoms, const int n_shift, const int n_pts_all, const int p0, const int n_p,
    const int n_e_all, const int span_lo, const int kz,
    const int n_t, const int t_split, const int want_cur)
{
    const int pl = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = blockIdx.y;
    if (pl >= n_p) return;
    const int p = p0 + pl;
    const float ph = phi[p];
    const float* rho = dos + ((long long)a * n_e_all + span_lo) * N_ORB;          // rho[j * N_ORB + o]: span sample j
    float* o_pa = out + ((long long)pl * n_t * n_atoms + a) * N_ORB;             // + t * t_stride: requested sample t
    const long long t_stride = (long long)n_atoms * N_ORB;
    float* c_pa = want_cur ? cur + ((long long)pl * n_t * n_atoms + a) : 0;      // + t * n_atoms: this atom's slot
    const bool exact0 = (e0 == 0.0f);                                             // a sample (span index kz) lies exactly at 0 V
    const int m_lo = exact0 ? 0 : 1;                                              // the stencil: span indices kz - 2 + m, m_lo <= m <= 4

    Rule R;
    {
        const float th = -e0 / step;
        const float thm1 = th - 1.0f, thp1 = th + 1.0f, thm2 = th - 2.0f;
        R.cf[0] = -th * thm1 * thm2 / 6.0f;
        R.cf[1] = thp1 * thm1 * thm2 / 2.0f;
        R.cf[2] = th * thp1 * thm2 / 2.0f;
        R.cf[3] = th * thp1 * thm1 / 6.0f;
        R.qp[0] = thm1 * thm1 * (th * th - 2.0f * th - 1.0f);
        R.qp[1] = thm1 * thm1 * (3.0f * th * th - 2.0f * th - 13.0f);
        R.qp[2] = thm1 * (3.0f * th * th * th - th * th - 13.0f * th - 13.0f);
        R.qp[3] = thm1 * thm1 * thp1 * thp1;
        R.qm[0] = th * th * thm2 * thm2;
        R.qm[1] = th * (3.0f * th * th * th - 8.0f * th * th - 6.0f * th + 24.0f);
        R.qm[2] = th * th * (3.0f * th * th - 4.0f * th - 12.0f);
        R.qm[3] = th * th * (th * th - 2.0f);
        pair_coef(e0 + step, step, R.pu, &R.hu);
        pair_coef(-e0, step, R.pd, &R.hd);
    }

    for (int s = 0; s < n_shift; ++s) {
        const float kd = kappa0_per_sqrt_ev * dists[((long long)s * n_atoms + a) * n_pts_all + p];   // Physics: k d
        for (int t = 0; t < n_t; ++t) {
            const int jv = js[t];                                                 // V's span index
            const float V = e0 + (float)(jv - kz) * step;                         // the bias V (eV)
            // w(E, V) and dw(E, V)/dV at the stencil, once per sample.
            float ws[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, dws[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (int m = m_lo; m < 5; ++m)
                exact_weight(e0 + (float)(m - 2) * step, V, ph, kd, &ws[m], &dws[m]);
            // The side of 0 V, and k, the intervals from 0 V to V: the sample n intervals out (E+-n) is at span
            // index z + dir n.
            const bool up = (t >= t_split);
            const int dir = up ? 1 : -1;
            const int z = (up || exact0) ? kz : kz + 1;
            const int k = up ? jv - z : z - jv;
            float* o_t = o_pa + (long long)t * t_stride;
            if (k == 0) {                                                         // V at the sample at 0 V: both integrals 0
                for (int o = 0; o < N_ORB; ++o)
                    o_t[o] += rho[(long long)jv * N_ORB + o] * ws[2];            // Physics: dI/dV = rho_o(0) w(0, 0)
                continue;
            }
            const bool odd = (k & 1) != 0;
            const int n0 = odd ? 1 : 2;                                           // the pairs start at E+-n0
            const float dx = up ? step : -step;

            // The piece next to 0 V (first_piece) of each integrand, from the stencil.
            float acc[N_ORB];                                                     // dI/dV's integral, per orbital
            float Fc[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                         // the current's integrand at the stencil
            for (int o = 0; o < N_ORB; ++o) {
                float F[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (int m = m_lo; m < 5; ++m) {
                    const float r = rho[(long long)(kz - 2 + m) * N_ORB + o];
                    F[m] = r * dws[m];                                            // Physics: rho_o(E) dw(E, V)/dV
                    if (want_cur) Fc[m] += r;                                     // sum_o rho_o(E)
                }
                acc[o] = first_piece(R, exact0, up, odd, step, F[0], F[1], F[2], F[3], F[4]);
            }
            float acc_c = 0.0f;                                                   // the current's integral
            if (want_cur) {
                for (int m = m_lo; m < 5; ++m) Fc[m] *= ws[m];                    // Physics: (sum_o rho_o(E)) w(E, V)
                acc_c = first_piece(R, exact0, up, odd, step, Fc[0], Fc[1], Fc[2], Fc[3], Fc[4]);
            }

            // The equal-step Simpson pairs (y0 + 4 y1 + y2) outward from E+-n0 to V, walked: the pair's first
            // sample j0 (its weight from the stencil or the previous pair), the weight evaluated at each new sample.
            float sum[N_ORB];
            for (int o = 0; o < N_ORB; ++o) sum[o] = 0.0f;
            float sum_c = 0.0f;
            int j0 = z + dir * n0;
            float w0 = ws[j0 - kz + 2], dw0 = dws[j0 - kz + 2];
            for (int n = n0; n + 2 <= k; n += 2) {
                const int j1 = j0 + dir, j2 = j0 + 2 * dir;
                float w1, dw1, w2, dw2;
                if (n + 1 <= 2) { w1 = ws[j1 - kz + 2]; dw1 = dws[j1 - kz + 2]; }
                else exact_weight(e0 + (float)(j1 - kz) * step, V, ph, kd, &w1, &dw1);
                exact_weight(e0 + (float)(j2 - kz) * step, V, ph, kd, &w2, &dw2);
                const float* r0 = rho + (long long)j0 * N_ORB;
                const float* r1 = rho + (long long)j1 * N_ORB;
                const float* r2 = rho + (long long)j2 * N_ORB;
                float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f;
                for (int o = 0; o < N_ORB; ++o) {
                    sum[o] += r0[o] * dw0 + 4.0f * (r1[o] * dw1) + r2[o] * dw2;  // Physics: rho_o(E) dw(E, V)/dV
                    if (want_cur) { t0 += r0[o]; t1 += r1[o]; t2 += r2[o]; }
                }
                if (want_cur) sum_c += t0 * w0 + 4.0f * (t1 * w1) + t2 * w2;     // Physics: (sum_o rho_o(E)) w(E, V)
                j0 = j2;
                w0 = w2;
                dw0 = dw2;
            }
            const float wV = (k <= 2) ? ws[jv - kz + 2] : w0;                     // w(V, V): V's sample is the last walked

            for (int o = 0; o < N_ORB; ++o)
                o_t[o] += rho[(long long)jv * N_ORB + o] * wV + (acc[o] + dx / 3.0f * sum[o]);   // Physics: dI/dV_o(V)
            if (want_cur)
                c_pa[(long long)t * n_atoms] += acc_c + dx / 3.0f * sum_c;                        // Physics: I(V)
        }
    }
}
'''
_EXACT_SWEEP_KERNELS = {}


def _exact_sweep_kernel(n_orb):
    """Machinery: the exact_sweep kernel compiled for n_orb DOS columns per atom (one build per column count)."""
    kern = _EXACT_SWEEP_KERNELS.get(n_orb)
    if kern is None:
        kern = cp.RawKernel(f"#define N_ORB {int(n_orb)}\n" + _EXACT_SWEEP_SRC, 'exact_sweep')
        _EXACT_SWEEP_KERNELS[n_orb] = kern
    return kern


# --- Energies to DOS samples (CPU, the parser's float64 grid) ---
ZERO_V = '0 V'       # _snap's answer for a window end of exactly 0.0: 0 V itself, not a sample (never None, which means 'not cached')


def _snap(energies, value, kind):
    """
    Machinery: the DOS sample an energy snaps to, as an index into `energies` (ascending: the parser's
    float64 grid, or a consecutive part of it): the nearest sample, the smallest |E_i - value| over every
    sample on either side of the value.  A tie (two samples at exactly equal float64 distance) goes, by `kind`:
      'emax'       E Max                        the lower sample
      'emin'       E Min                        the higher sample
      'zero_side'  a Map energy, or topo_bias   the sample closer to 0
    An end of exactly 0.0 ('emin' or 'emax') is exactly 0 V, not a sample: ZERO_V is returned for it.
    """
    value = float(value)
    if kind in ('emin', 'emax') and value == 0.0:
        return ZERO_V
    n = len(energies)
    i = int(np.searchsorted(energies, value))          # the first sample at or above the value
    if i == 0:
        return 0
    if i == n:
        return n - 1
    d_lo, d_hi = value - energies[i - 1], energies[i] - value
    if d_lo != d_hi:
        return i - 1 if d_lo < d_hi else i
    if kind == 'emax':
        return i - 1
    if kind == 'emin':
        return i
    return i - 1 if abs(energies[i - 1]) <= abs(energies[i]) else i


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

    def _topo_current(self, tip_positions, v_b, decay_model, samples=None):
        """
        Machinery: the topography's current at the frozen bias V_b at the tip positions (n, 3), the
        integral from 0 V to V_b of the weighted total LDOS (every state weighted at that frozen bias),
        taken positive as before (the integral from min(0, V_b) to max(0, V_b)), under the decay model
        decay_model ('decay_exact', 'decay_linearized' or 'decay_k0': _effective_decay_model).
        v_b: V_b, the bias end's DOS sample energy (a Python float, the parser's float64 value); its sample
        index i_b is that sample's own (_snap 'zero_side' of a sample's energy returns that sample).  The
        sample list: the stencil E-2, E-1, E1, E2 (DOS samples k0 - 1 .. k0 + 2, or k0 - 2 .. k0 + 2 with a
        sample at 0 V) and every sample from 0 V to V_b, consecutive; formed here unless given (samples, as
        this returned it for the same v_b).  The weighted total LDOS at those samples is requested with
        samples= (topo_only, the frozen bias at V_b's GPU energy e0 + (i_b - k0) step) and integrated from
        0 V by quadrature.integral_from_zero; its value at V_b is the current.
        Returns (the current, (n,) float32 CuPy; the sample list, an np.int64 array).
        """
        i_b = _snap(self.energies, v_b, 'zero_side')
        if samples is None:
            st_lo = self.k0 - 2 if self.e0 == 0.0 else self.k0 - 1
            samples = np.arange(min(st_lo, i_b), max(self.k0 + 2, i_b) + 1, dtype=np.int64)
        s0 = int(samples[0])
        v_b32 = self.e032 + np.float32(i_b - self.k0) * self.e_step32
        ld_1, _, _ = self._calculate_ldos_at_points_gpu(
            tip_positions, None, None, decay_model=decay_model, preserve_orbitals=False, topo_only=True,
            v_bias=v_b32, samples=samples)
        # Topography always uses total LDOS — ld_1 is already combined for POL when topo_only=True.
        cur = quadrature.integral_from_zero(ld_1, self.k0 - s0, self.e_step32, self.e032)[:, i_b - s0]
        return (cur if v_b > 0 else -cur), samples

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, samples, v_b, target_ldos,
                             target_threshold=0.01, topo_gain=0.5, max_iter=1000, decay_model='decay_exact'):
        """Machinery: Exhaustive Point-Wise Convergence Engine: the tip height at each point is moved
        until the current at the frozen bias v_b (V_b; _topo_current: the weighted total LDOS at the
        sample list `samples`, E-2 .. E2 and every sample from 0 V to V_b, integrated from 0 V to V_b,
        taken positive) is target_ldos, under the decay model decay_model (_effective_decay_model's names)."""
        t_start = time()
        print("--- INITIALIZING TIP CONVERGENCE ENGINE ---")
        for i in range(max_iter):
            t0 = time()
            cur_ldos, _ = self._topo_current(cp.hstack([grid_xy_gpu, z_map_gpu[:, None]]), v_b, decay_model,
                                             samples=samples)
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
            kappa = KAPPA0_PER_SQRT_EV * cp.sqrt(phi_l)
            z_map_gpu[active_mask] += (topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
        print(f"--- CONVERGENCE COMPLETE: Total Time ={time() - t_start:6.3f}s ---")
        return z_map_gpu

    def _get_barrier_potential(self):
        """Machinery: return the single 3D potential grid used for the tunneling barrier."""
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
        """Machinery: place every atom on the slab side of the vacuum gap in z.

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
        # The DOS grid, taken once here from the parser's float64 energies: its step, k0 (the last sample at
        # or below 0 V) and e0 = energies[k0] (0 V lies -e0 above sample k0; e0 is exactly 0.0 when a sample
        # is at 0 V), in float64, and step and e0 rounded to float32 once (e_step32, e032) for the GPU, which
        # forms sample i's energy as e0 + (i - k0) * step.  0 V needs two samples on each side of it.
        self.e_step = float(dos_parser.step)
        k0 = int(np.searchsorted(self.energies, 0.0, side='right')) - 1
        n_below = k0 if (k0 >= 0 and self.energies[k0] == 0.0) else k0 + 1      # the samples strictly below 0 V
        n_above = len(self.energies) - 1 - k0                                    # the samples strictly above 0 V
        if n_below < 2 or n_above < 2:
            raise ValueError(
                f"DOSCAR: 0 V (E_F) lies within two samples of an end of the energy grid "
                f"({float(self.energies[0]):.6f} to {float(self.energies[-1]):.6f} eV from E_F; {n_below} samples "
                f"below 0 V, {n_above} above): the integrals from 0 V need two samples on each side of it.")
        self.k0 = k0
        self.e0 = float(self.energies[k0])
        self.e_step32 = np.float32(self.e_step)
        self.e032 = np.float32(self.e0)
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
        """Machinery: rebuild working DOS arrays from pristine originals with extra Gaussian broadening."""
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
        """Machinery: generate cache filename suffix encoding active broadening state."""
        active = sorted((t, s) for t, s in getattr(self, 'extra_broadening', {}).items() if s > 0)
        if not active:
            return ""
        return "_broad_" + "_".join(f"{t}{s:.3f}" for t, s in active)

    def _unit_cell_cache_suffix(self):
        # Periodic-image count affects every sf sum. Embed in cache filename so old caches
        # auto-invalidate when unit_cell_num changes.
        return f"_uc{getattr(self, 'unit_cell_num', 1)}"

    def _effective_decay_model(self):
        """Machinery: the decay model's name, in the cache names and the in-memory recompute keys.
        'decay_exact' = Decay on, 'exact': the exact tilted barrier (gpu_exact_factor);
        'decay_linearized' = Decay on, 'linearized': the linearized weight (gpu_linearized_factor);
        'decay_k0' = Decay off: exp(-2 kappa0 d), the same weight without its energy and bias dependence
        ('exact' / 'linearized' then make no difference)."""
        if not getattr(self, 'use_decay', True):
            return 'decay_k0'
        return 'decay_linearized' if getattr(self, 'decay_linearized', False) else 'decay_exact'

    def _decay_cache_suffix(self):
        # Converged topographies depend on the decay model used to produce them, so the model name
        # is part of every topography cache name.
        return "_" + self._effective_decay_model()

    def _precision_cache_suffix(self):
        return "_f32"

    def _display_indices(self, emin, emax):
        """
        Machinery: the displayed DOSCAR samples of the window [emin, emax], ascending, as
        _calculate_ldos_at_points_gpu takes them (samples=, and its emin / emax path): from E Min's
        snapped sample (_snap 'emin', a tie to the higher) to E Max's (_snap 'emax', a tie to the lower),
        both included.  An end of exactly 0.0 is 0 V, not a sample, and adds no point: the display then
        starts (E Min = 0.0) at the first sample above 0 V, or ends (E Max = 0.0) at the last sample below
        it, so a sample exactly at 0 V is then not displayed.  No 0 V value is ever displayed.
        """
        lo = _snap(self.energies, emin, 'emin')
        hi = _snap(self.energies, emax, 'emax')
        if lo is ZERO_V:
            lo = self.k0 + 1
        if hi is ZERO_V:
            hi = self.k0 if self.e0 != 0.0 else self.k0 - 1
        return np.arange(lo, hi + 1)

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax,
                                      use_energy_decay=False, preserve_orbitals=False,
                                      topo_only=False, v_bias=None, with_current=False, samples=None, progress='',
                                      decay_model=None):
        """
        Machinery: the LDOS at the tip positions on the displayed DOSCAR samples of the window [emin, emax]
        (_display_indices: each end snapped to a sample by _snap; an end of exactly 0.0 is 0 V and adds no
        sample, the samples starting or ending at the first on the window's side of 0), or, with `samples`,
        on those DOSCAR sample indices, in the order requested (emin and emax then unused): per point and
        energy, or with preserve_orbitals per point, energy, atom and orbital.  The third element returned
        is the samples' energies, the parser's float64 values as a CuPy array (cp.asarray(energies[idx])).
        The weights, I and dI/dV are those of the block THE DECAY MODEL (physics); everything computed here on
        the GPU is float32 (the tip positions, the distances, phi, the weights, the outputs).
        decay_model: 'decay_exact', 'decay_linearized' or 'decay_k0' (_effective_decay_model's names; ValueError
        for another); None: from the Decay bool use_energy_decay ('decay_exact' when True, 'decay_k0' when False).

        Decay off ('decay_k0'): every image of every atom weighted by exp(-2 kappa0 d), which depends on neither E
        nor V.
        Decay on ('decay_exact': the exact tilted barrier, gpu_exact_factor; 'decay_linearized': the linearized
        weight, gpu_linearized_factor):
          v_bias given (topography, from _topo_current): the bias is frozen at v_bias (V_b's GPU energy), each
            state, at its sample's GPU energy e0 + (i - k0) step, weighted by w(E, v_bias); the caller
            integrates the result from 0 V to V_b (quadrature.integral_from_zero), the current at that bias.
          v_bias None (LDOS): the bias is swept, and the value at each requested sample V is dI/dV at V: the
            state entering at the window edge, rho(V) w(V, V), plus the change of every state already in
            [0, V], the integral from 0 V to V of rho(E) dw(E, V)/dV, by quadrature.integral_from_zero's rule
            (the value at 0 V and the piece next to it from the stencil E-2, E-1, E1, E2 around 0 V, Simpson
            pairs outward to V).  That integral always starts at 0 V, so the DOS from the stencil to the
            samples is used whether or not they reach 0 V.  Computed, for 'decay_exact', by the exact_sweep
            kernel (_EXACT_SWEEP_SRC), each requested V on its own, the launches chunked by the (requested V,
            sample) pairs per tip point.
        with_current=True (used only with v_bias None and topo_only False): a fourth element is returned, the
          total current I(V) at the requested samples, in the order requested, (num_pts, num_e) float32 CuPy:
          the integral from 0 V to V of the total DOS weighted at that V, oriented (negative below 0 V for a
          positive DOS, 0 at a sample exactly at 0 V), the channels combined as _combine_ldos_total combines
          them (channel 1 for NCL and unpolarized, channel 1 + channel 2 for collinear polarized; the NCL |m|
          channel computes no current).  Decay on: from the kernel's same pass as dI/dV (want_cur = 1).  Decay
          off: the channel's collapsed DOS with the display's weights at the stencil and every sample from 0 V
          out to the farthest requested sample on each side, integrated by feenstra.current_from_ldos
          (_current_decay_off).  Without with_current the three elements above are returned (sim_wrapper and
          the topography callers).
        With v_bias None and topo_only False (the LDOS, after the tip heights converged) the console shows one
        line per channel, 'x/n spectra calculated' (n the tip points), updated after each launch of tip points
        (Decay on) or once (Decay off), led by `progress` (what is computed) and the channel; the topography
        calls print nothing here.
        """
        # The decay model (decay_model, or the Decay bool's), and from it Decay on or off and the frozen-bias weight.
        if decay_model is None:
            decay_model = 'decay_exact' if use_energy_decay else 'decay_k0'
        if decay_model not in ('decay_exact', 'decay_linearized', 'decay_k0'):
            raise ValueError(f"decay_model must be 'decay_exact', 'decay_linearized' or 'decay_k0', not {decay_model!r}")
        use_energy_decay = decay_model != 'decay_k0'
        linearized = decay_model == 'decay_linearized'
        frozen_factor = gpu_linearized_factor if linearized else gpu_exact_factor
        if samples is None:
            idx_np = self._display_indices(emin, emax)
            estart, eend = int(idx_np[0]), int(idx_np[-1]) + 1
        else:
            idx_np = np.asarray(samples, dtype=np.int64).ravel()
            estart, eend = int(idx_np.min()), int(idx_np.max()) + 1
        energy_indices = cp.asarray(idx_np)
        calc_energies_gpu = cp.asarray(self.energies[idx_np])
        num_pts, num_e = tip_positions.shape[0], len(idx_np)
        show = v_bias is None and not topo_only          # the LDOS itself (after the tip heights converged): its console line

        def _progress(done, label):
            """Machinery: the LDOS's console line, 'x/n spectra calculated' (n the tip points, x those finished),
            rewritten in place (a carriage return) and ended with a newline at n.  The GPU is waited for first,
            since CuPy runs GPU work asynchronously to Python, so x counts finished spectra."""
            if not show:
                return
            cp.cuda.get_current_stream().synchronize()
            lead = f"{progress}, " if progress else ''
            print(f"\r{lead}{label}{done}/{num_pts} spectra calculated", end='\n' if done >= num_pts else '',
                  flush=True)

        tip_pos_gpu = cp.asarray(tip_positions, dtype=cp.float32)
        frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)

        wrapped_frac_coords = cp.empty_like(frac_coords)
        wrapped_frac_coords[:, :2] = frac_coords[:, :2] % 1.0
        wrapped_frac_coords[:, 2] = frac_coords[:, 2]

        lv_gpu = cp.array(self.lv, dtype=cp.float32)
        wrapped_tip_pos_gpu = cp.dot(wrapped_frac_coords, lv_gpu)

        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        dists = cp.sqrt(cp.sum((self.periodic_coord_gpu[:, None, :] - wrapped_tip_pos_gpu[None, :, :]) ** 2, axis=2))

        sweep = use_energy_decay and v_bias is None
        if sweep:
            # The span: the stencil E-2 .. E2 around 0 V (DOS samples k0 - 1 .. k0 + 2, or k0 - 2 .. k0 + 2 with
            # a sample at 0 V) and the requested samples, consecutive, from span_lo to max(eend, k0 + 3) (the
            # kernel reads the DOS from span_lo on; the integral at each V runs from 0 V).
            st_lo = self.k0 - 2 if self.e0 == 0.0 else self.k0 - 1
            span_lo = min(estart, st_lo)
            k0_span = self.k0 - span_lo                                          # the span index of k0
            u_idx, u_inv = np.unique(idx_np, return_inverse=True)                # the requested samples, ascending, once each
            in_order = np.array_equal(u_idx, idx_np)
            js_gpu = cp.asarray(u_idx - span_lo, dtype=cp.int32)
            t_split = int(np.searchsorted(u_idx, self.k0, side='right'))         # the first requested sample above 0 V
            dists_c = cp.ascontiguousarray(dists, dtype=cp.float32)
            n_shift = dists.shape[0] // self.num_total_atoms                    # images are shift-major: shift, then atom
            # The launch chunk, by the (requested V, sample) pairs per tip point: at each V the stencil and every
            # sample from 0 V to V.
            n_pairs = int(np.sum(np.maximum(u_idx, self.k0 + 2) - np.minimum(u_idx, st_lo) + 1))
            pts_per_launch = max(1, int(2.5e7 // max(1, n_pairs * self.num_total_atoms)))   # tip points per launch

        def _sweep(phi_c, dos3, p0, n_p, out, cur):
            """Machinery: the decay model's sweep kernel over tip points p0 .. p0 + n_p: out (n_p, n_unique, atoms,
            columns) += dI/dV, and, unless cur is None (want_cur = 0), cur (n_p, n_unique, atoms) += the current;
            'decay_exact': exact_sweep; 'decay_linearized': its own kernel, launched here."""
            if linearized:
                # 'decay_linearized': the launch of its sweep kernel goes here, into the same out and cur.
                raise NotImplementedError("the swept LDOS of 'decay_linearized' has no sweep kernel here yet")
            _exact_sweep_kernel(dos3.shape[2])(
                ((n_p + 63) // 64, self.num_total_atoms), (64,),
                (dos3, dists_c, phi_c, js_gpu, out, cur, np.float32(KAPPA0_PER_SQRT_EV), self.e_step32, self.e032,
                 np.int32(self.num_total_atoms), np.int32(n_shift), np.int32(num_pts), np.int32(p0), np.int32(n_p),
                 np.int32(dos3.shape[1]), np.int32(span_lo), np.int32(k0_span),
                 np.int32(len(u_idx)), np.int32(t_split), np.int32(0 if cur is None else 1)))

        def _current_decay_off(w_t, dos_rows):
            """Machinery: the Decay-off current (the weight exp(-2 kappa0 d) does not depend on V) at the requested
            samples, in the order requested, (num_pts, num_e) float32: the weighted total LDOS, w_t (num_pts, rows)
            the display's weights on dos_rows (rows, n_e_all), the channel's collapsed DOS, at the stencil E-2 .. E2
            and every sample from 0 V out to the farthest requested sample on each side (consecutive, c_lo to c_hi),
            integrated from 0 V by feenstra.current_from_ldos and taken at the requested samples."""
            st_lo = self.k0 - 2 if self.e0 == 0.0 else self.k0 - 1
            c_lo, c_hi = min(estart, st_lo), max(eend - 1, self.k0 + 2)
            ld_span = cp.dot(w_t, dos_rows[:, c_lo:c_hi + 1])
            i_span = feenstra.current_from_ldos(ld_span, self.k0 - c_lo, self.e_step32, self.e032)
            return i_span[:, cp.asarray(idx_np - c_lo)]

        def _compute_channel(pot, dos_gpu, dos_collapsed, want_cur=False, label=''):
            """Machinery: one channel's LDOS (and, with want_cur, its current) at the tip points under the decay
            model: Decay on with v_bias, the frozen-bias weight (frozen_factor: gpu_exact_factor, or
            gpu_linearized_factor for 'decay_linearized'); Decay on without, the model's sweep (_sweep); Decay
            off, gpu_decay_off_factor.  label leads the channel's console line (_progress)."""
            phi_local = cp_ndimage.map_coordinates(pot, grid_indices, order=1, mode='wrap') - self.ef
            phi_row = phi_local[None, :]
            cur = None                              # the channel's current (want_cur), (num_pts, num_e)
            if not preserve_orbitals:
                output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float32)
                dos_periodic = dos_collapsed[self.atom_indices_periodic_gpu, :]
                dos_active = dos_periodic[:, energy_indices]
                if use_energy_decay and not sweep:
                    for e_idx in range(num_e):
                        e_gpu = self.e032 + np.float32(idx_np[e_idx] - self.k0) * self.e_step32   # the sample's GPU energy
                        sf = frozen_factor(e_gpu, v_bias, phi_row, dists, topo=True)
                        output_ldos[:, e_idx] = cp.dot(sf.T, dos_active[:, e_idx])
                elif sweep:
                    phi_c = cp.ascontiguousarray(phi_local, dtype=cp.float32)
                    dos3 = cp.ascontiguousarray(dos_collapsed[:, :, None], dtype=cp.float32)
                    n_u = len(u_idx)
                    tot = cp.zeros((num_pts, n_u), dtype=cp.float32)
                    tot_c = cp.zeros((num_pts, n_u), dtype=cp.float32) if want_cur else None
                    for p0 in range(0, num_pts, pts_per_launch):
                        n_p = min(pts_per_launch, num_pts - p0)
                        part = cp.zeros((n_p, n_u, self.num_total_atoms, 1), dtype=cp.float32)
                        # want_cur: the kernel's current output too (want_cur = 1), summed over atoms, from the
                        # same pass as dI/dV.
                        part_cur = cp.zeros((n_p, n_u, self.num_total_atoms), dtype=cp.float32) if want_cur else None
                        _sweep(phi_c, dos3, p0, n_p, part, part_cur)
                        tot[p0:p0 + n_p] = part.sum(axis=(2, 3))
                        if want_cur:
                            tot_c[p0:p0 + n_p] = part_cur.sum(axis=2)
                        _progress(p0 + n_p, label)
                    output_ldos = tot if in_order else tot[:, cp.asarray(u_inv)]
                    if want_cur:
                        cur = tot_c if in_order else tot_c[:, cp.asarray(u_inv)]
                else:
                    sf = gpu_decay_off_factor(phi_local, dists)
                    output_ldos = cp.dot(sf.T, dos_active)
                    if want_cur:
                        cur = _current_decay_off(sf.T, dos_periodic)
                    _progress(num_pts, label)
            else:
                output_ldos = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float32)
                if use_energy_decay and not sweep:
                    for e_idx in range(num_e):
                        e_gpu = self.e032 + np.float32(idx_np[e_idx] - self.k0) * self.e_step32   # the sample's GPU energy
                        sf = frozen_factor(e_gpu, v_bias, phi_row, dists, topo=True)
                        w_atom = cp.dot(self.map_mat_gpu, sf)
                        output_ldos[:, e_idx, :, :] = w_atom.T[:, :, None] * dos_gpu[:, energy_indices[e_idx], :][None, :, :]
                elif sweep:
                    phi_c = cp.ascontiguousarray(phi_local, dtype=cp.float32)
                    dos3 = cp.ascontiguousarray(dos_gpu, dtype=cp.float32)
                    # The display sweep, chunked by the (requested V, sample) pairs per tip point, out and cur
                    # sliced per launch; the current (want_cur = 1) from the same pass, per atom, then summed over
                    # atoms; without want_cur cur is None (want_cur = 0).
                    out = output_ldos if in_order else cp.zeros((num_pts, len(u_idx), self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float32)
                    cur_u = cp.zeros((num_pts, len(u_idx), self.num_total_atoms), dtype=cp.float32) if want_cur else None
                    for p0 in range(0, num_pts, pts_per_launch):
                        n_p = min(pts_per_launch, num_pts - p0)
                        _sweep(phi_c, dos3, p0, n_p, out[p0:p0 + n_p], None if cur_u is None else cur_u[p0:p0 + n_p])
                        _progress(p0 + n_p, label)
                    if not in_order:
                        output_ldos = out[:, cp.asarray(u_inv)]
                    if want_cur:
                        cur = cur_u.sum(axis=2) if in_order else cur_u.sum(axis=2)[:, cp.asarray(u_inv)]
                else:
                    sf = gpu_decay_off_factor(phi_local, dists)
                    w_atom = cp.dot(self.map_mat_gpu, sf)
                    output_ldos = w_atom.T[:, None, :, None] * dos_gpu[:, energy_indices, :].transpose(1, 0, 2)[None, :, :, :]
                    if want_cur:
                        cur = _current_decay_off(w_atom.T, dos_collapsed)
                    _progress(num_pts, label)
            return output_ldos, cur

        # All modes use total potential for the barrier
        barrier_pot = self._get_barrier_potential()

        if topo_only:
            # Topography only needs ρ_total. Skip the second (magnetization) channel that
            # _converge_tip_height / topo callers ignore — 2× speedup for NCL_SOC and POL.
            if self.spin_mode == SpinMode.COLLINEAR_POL:
                # Use precomputed (dos_up + dos_dn) instead of computing two channels and adding.
                return (_compute_channel(barrier_pot, self.dos_total_gpu, self.dos_total_collapsed)[0],
                        None, calc_energies_gpu)
            # NCL_SOC: dos_up_gpu IS already ρ_total per orbital.
            # COLLINEAR_UNPOL: dos_up_gpu IS the single channel.
            return (_compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed)[0],
                    None, calc_energies_gpu)

        if self.spin_mode == SpinMode.COLLINEAR_POL:
            # Collinear polarized: two channels (up, down), both against total barrier
            ld_1, cur_1 = _compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed, with_current,
                                           label='spin up: ')
            ld_2, cur_2 = _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed, with_current,
                                           label='spin down: ')
        elif self.spin_mode == SpinMode.NCL_SOC:
            # NCL: channel 1 = total DOS, channel 2 = |m| DOS (the |m| channel computes no current)
            ld_1, cur_1 = _compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed, with_current,
                                           label='total: ')
            ld_2, cur_2 = _compute_channel(barrier_pot, self.dos_dn_gpu, self.dos_dn_collapsed, label='|m|: ')
        else:
            # Unpolarized: single channel
            ld_1, cur_1 = _compute_channel(barrier_pot, self.dos_up_gpu, self.dos_up_collapsed, with_current)
            ld_2, cur_2 = None, None
        if not with_current:
            return ld_1, ld_2, calc_energies_gpu
        # with_current: the total current I at the requested samples, in the order requested, (num_pts, num_e)
        # float32, the channels combined as the total LDOS is (_combine_ldos_total: channel 1 for NCL and
        # unpolarized, channel 1 + channel 2 for collinear polarized).
        return ld_1, ld_2, calc_energies_gpu, self._combine_ldos_total(cur_1, cur_2)

    # ------------------------------------------------------------------
    # _combine_ldos: central dispatch for how channels are combined at display time
    # ------------------------------------------------------------------
    def _combine_ldos_total(self, ld_1, ld_2):
        """Machinery: combine channels into total LDOS (for topography / non-mag display)."""
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            return ld_1 + ld_2
        # NCL and unpol: ld_1 is already the total
        return ld_1

    def _combine_ldos_display(self, ld_1, ld_2, show_mag):
        """Machinery: combine channels for display, respecting show_mag toggle."""
        if not show_mag or ld_2 is None:
            return self._combine_ldos_total(ld_1, ld_2)
        if self.spin_mode == SpinMode.COLLINEAR_POL:
            return ld_1 - ld_2   # spin density (signed)
        if self.spin_mode == SpinMode.NCL_SOC:
            return ld_2           # |m| (unsigned, always ≥ 0)
        return ld_1

    def _mag_is_signed(self):
        """Machinery: whether the magnetization display data is signed (needs a diverging colour map, the 'mag' role's) or not."""
        return self.spin_mode == SpinMode.COLLINEAR_POL

    def _has_mag_channel(self):
        """Machinery: whether this spin mode supports the Mag checkbox."""
        return self.spin_mode in (SpinMode.COLLINEAR_POL, SpinMode.NCL_SOC)


DESIGN = (20, 14)        # the window's design size (inches): its text zoom is its size relative to this (qtplot.window_zoom)
# The colour map of the Map view's energy maps and their colour bars in 'default 1': the sm4_Browser's MAP_CMAP
# (plots.py there, "custom_map", midnight blue to light sky blue), made from that set's Map entry, a colour list
# (cmap_shading.SETS).
MAP_CMAP = cmap_shading.make_cmap(cmap_shading.cmap_spec('default 1', 'map'), plt.to_cmap,
                                  plt.LinearSegmentedColormap.from_list, "custom_map")
# The window's colour maps, made once each from cmap_shading's two colour-map sets (the constructor's cmap_set; the
# 'cmap set' choice in the left column) and kept (_window_cmap), so that a panel's map tells its role by identity
# (_on_cmap_set): by (set, role), and by (None, role) for the energy maps' isoluminant maps while they are shaded
# (cmap_shading.SHADING_CMAPS, the same in both sets).  default 1's Map entry is MAP_CMAP.
_CMAPS = {('default 1', 'map'): MAP_CMAP}


def _window_cmap(set_name, role, shading=False):
    """
    Machinery: the colour map of `role` ('topo', 'map', 'heatmap', 'mag' or '|m|', cmap_shading.ROLES) in the
    colour-map set `set_name` (cmap_shading.SETS), or with shading=True the isoluminant map of a shaded energy map
    ('map', 'mag' or '|m|': cmap_shading.SHADING_CMAPS), as a qtplot Colormap: cmap_shading.cmap_spec's entry made by
    cmap_shading.make_cmap (a name through qtplot's to_cmap, a colour list as a linear ramp through its colours by
    LinearSegmentedColormap.from_list), once, then kept in _CMAPS.
    """
    key = (None if shading else set_name, role)
    cm = _CMAPS.get(key)
    if cm is None:
        cm = cmap_shading.make_cmap(cmap_shading.cmap_spec(set_name, role, shading=shading), plt.to_cmap,
                                    plt.LinearSegmentedColormap.from_list, name=role)
        _CMAPS[key] = cm
    return cm


# The Map view's contour slice (the sm4_Browser Map window's): its traces on the energy maps (one colour, 1 pt,
# never the topo's pixels) and its level line on the topo colour bar (cyan shows on the topo ramp).
CONTOUR_COLOR = 'white'
CONTOUR_LEVEL_COLOR = 'cyan'


class _LegendEvents(QtCore.QObject):
    """
    Machinery: the legend panel's mouse events that the window's canvas never sees, filtered on its two Qt
    widgets (installEventFilter): the Back button's double-click event is dropped (a double click is one
    Back), and a right press on the Back button or on the scroll bar goes back one level, as a right click
    anywhere in the panel (Interactive_STM_Simulator._legend_back).  `sim` is the simulator's window.
    """

    def __init__(self, sim):
        """Machinery: the filter of the window `sim` (kept by the window: Qt does not own an event filter)."""
        super().__init__()
        self._sim = sim

    def eventFilter(self, obj, ev):
        """Machinery: True (the event consumed) for the Back button's double-click event and for a right press on either widget; False (Qt's own handling) for every other event."""
        t = ev.type()
        if t == QtCore.QEvent.Type.MouseButtonDblClick and obj is self._sim._btn_back.qt:
            return True
        if t == QtCore.QEvent.Type.MouseButtonPress and ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._sim._legend_back()
            return True
        return False


class Interactive_STM_Simulator(Unified_STM_Simulator):
    """
    The simulator's window: one qtgui figure of DESIGN size whose panels are cells of its Qt
    layout and whose controls are Qt widgets, made once per figure (_make_controls) and arranged
    per view (_build_ui): the window's title (the spin mode), a left column with the view radio,
    the 'cmap set' choice, the broadening box, the boxes other windows add (add_control_box: the
    decomposition wrapper's σ box and its 'Send to decomp' buttons) and the RUN button with the
    toggles; the panel area of the view (Single Point: Global Topo | Topo h | Partitioned LDOS,
    its legend panel at its right; Line / Path: the top row Global Topo | Topo h | LDOS h | the
    cluster (Topo box, stripe, one heatmap per partition, colour bar) over the tip-height profile
    and the spectrum, its legend panel at its right; Map: Global Topo | Topo h | LDOS h | Map Topo
    over the rows of energy maps, the Partitioned LDOS one cell of the map block, its legend
    panel at its right); the bottom strip of number boxes (Cells, Path Pts, Markers, E Pts,
    E Min, E Max, LDOS γ) and Map's 'cmap norm' and Contour / Shading boxes.  Every slider of
    the old figure is a number box (the arrow keys step its digits); the orbital hover labels
    are widgets.HoverLabels.
    The physics and the caches are Unified_STM_Simulator's, called as before.
    """

    def __init__(self, filepath, erange, ldos_height, cmap_set='default 1', unit_cell_num=1):
        super().__init__(filepath, unit_cell_num=unit_cell_num)
        self.parse_vasp_outputs("LOCPOT")
        self.p1, self.p2 = None, None
        self.erange, self.ldos_height = list(erange), ldos_height
        # The colour-map set the window shows ('default 1' or 'default 2', cmap_shading.SETS; the 'cmap set' choice
        # changes it: _on_cmap_set), and the Map view's 'cmap norm': off, every map on one
        # colour scale, the maximum over all of them, with one colour bar; on, each map its own scale with
        # its own bar beside it.
        self.cmap_set = cmap_set
        self.cmap_norm = False
        # The Map view's Contour / Shading box's state (cmap_shading.BoxState: at most one of Contour and Shading
        # on, both may be off; the shading by Height or Relief), kept through every view change: it starts with
        # Contour on, Shading off and Height (_on_shade_box).  The energy maps' shade while Shading is on, kept
        # with what it was made from (_map_shade).
        self.shade_state = cmap_shading.BoxState()
        self._shade_cache = None
        self.npts = 72
        self.is_running, self.normalize, self.show_mag = False, False, False
        # 'Feenstra': the display divided by I/V, I the total LDOS integrated from 0 V (feenstra.py),
        # display only.
        self.feenstra = False
        self.show_atoms, self.show_unit_cell = True, False
        self.use_decay = True
        # The Decay-on model: 'exact' (False, at start) or 'linearized' (True), the two rows indented under 'Decay'
        # (exactly one on: _on_toggle); with Decay off neither is used (_effective_decay_model).
        self.decay_linearized = False
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
        # Line / Path: the markers in their order on the stripe, top to bottom, and each marker's kept colour
        # (line_marker_colors, permuted with the markers): _form_marker_set.
        self._form_marker_set()
        self.marker_coords = [[self.lv[0, 0] * 0.2, self.lv[1, 1] * 0.2],
                              [self.lv[0, 0] * 0.8, self.lv[1, 1] * 0.8]]
        self.active_obj = None
        self._drag_moved = False          # the dragged item moved since its press (_on_motion): its release work is due (_on_rel)
        # The decomp's state (level 0, marker 0, no element), held by ldos_decomp.Decomp and read and set as
        # plot_level, active_marker_idx and active_element.
        self.decomp = ldos_decomp.Decomp()
        self._type_color_map = {'Au': 'orange', 'N': 'blue', 'C': 'brown', 'H': 'grey'}
        self.extra_broadening = {t: 0.0 for t in self.atomtypes}
        self._broadening_topo_dirty = False

        # Line's and Path's tip heights, each kept with its own key and the bias and decay model it was converged
        # at: Line's (current_z_line) with its ends (cached_p1, cached_p2), cached_bias_energy_line and
        # cached_d_topo_line; Path's (current_z_path) with its nodes and Extend (cached_path_nodes,
        # cached_path_extend), cached_bias_energy_path and cached_d_topo_path.
        self.current_z_line, self.current_z_path = None, None
        self.cached_bias_energy_path, self.cached_d_topo_path = None, None
        self.cached_p1, self.cached_p2 = None, None
        self.cached_emin, self.cached_emax = None, None
        self.cached_d_topo_line, self.cached_d_topo_map, self.cached_d_ldos = None, None, None
        self.cached_bias_energy_line, self.cached_bias_energy_map, self.cached_nepts = None, None, None
        self.cached_ld_up, self.cached_ld_dn, self.cached_eg = None, None, None   # the Line / Path points' LDOS and its energies, kept together (sim_wrapper reads them)
        self.cached_ld_mode = None                                      # the view that computed them, Line or Path, kept with them (sim_wrapper reads it)
        self.cached_ld_up_gpu, self.cached_ld_dn_gpu = None, None      # the GPU copies the display works from
        self.cached_marker_coords, self.cached_spec_ldos = None, None   # Single Point's marker spectra (Map keeps its own, map_spec_*)
        self.cached_spec_ldos_gpu = None                                # the GPU copy the display works from
        self.cached_spec_window = None                                  # the window's snapped ends the marker spectra were computed over
        self.cached_spec_eg = None                                      # the marker spectra's energies, kept with them (sim_wrapper reads them)
        self.cached_current, self.cached_current_E = None, None              # the Line / Path points
        self.cached_spec_current, self.cached_spec_current_E = None, None    # the Single Point markers
        # Map's own energies, data and caches, apart from the other views' and kept through every toggle and view
        # change (_update_all): the snapped Map energies (map_e_idx, their energies map_e_targets) and the typed
        # ends and count they were last formed from (map_typed: E Min, E Max, E Pts); the grid's LDOS at them
        # (map_ld_*, with the GPU copies the display works from) and its current (map_current at map_current_E);
        # the window's snapped ends (map_window), the decay model (map_d_ldos), the broadening (map_broadening, a
        # copy of extra_broadening) and the LDOS-topo heights (map_z, which the Map Topo, the contour slice and the
        # shading are drawn from, with the bias map_bias and the LDOS h map_ldos_h they were computed at, the Map
        # Topo's title) they were computed with; Map's marker spectra (map_spec_ldos, with the GPU copy
        # map_spec_ldos_gpu), their energies (map_eg), the markers they were computed for (map_marker_coords), their
        # current (map_spec_current at map_spec_current_E) and the decay model and the broadening they were
        # computed with (map_spec_d_ldos, map_spec_broadening).
        self.map_e_idx, self.map_e_targets, self.map_typed = None, None, None
        self.map_ld_up, self.map_ld_dn, self.map_ld_up_gpu, self.map_ld_dn_gpu = None, None, None, None
        self.map_current, self.map_current_E = None, None
        self.map_eg, self.map_window, self.map_d_ldos, self.map_broadening = None, None, None, None
        self.map_z, self.map_bias, self.map_ldos_h = None, None, None
        self.map_spec_ldos, self.map_spec_ldos_gpu, self.map_marker_coords = None, None, None
        self.map_spec_current, self.map_spec_current_E = None, None
        self.map_spec_d_ldos, self.map_spec_broadening = None, None

        # --- Path mode state ---
        self.path = None                  # initial seed: all atom-index lists [[154],...] OR all [x,y] coords; no mix
        self.path_extend = False          # toggle: extend ends to first local cell wall
        self.path_nodes = None            # live list of N draggable [x, y] node positions
        self.cached_path_nodes = None     # cache-invalidation key for path topo
        self.cached_path_extend = None
        self.fig = None                   # the window (run_interactive); its controls are made once (_make_controls)
        self.cell_rotated = False         # the unit-cell panels turned 90 degrees (Axes.set_rotated): _choose_orientation
        self.z_frac = 0.5                 # the contour slice: its level as a fraction of the Map Topo's height range
        self._contour_artists = []        # the slice's traces on the energy maps
        self._level_line = None           # the slice's level line on the topo colour bar
        self.cax_topo = None              # Map: the topo colour bar beside the Map Topo
        self._orient_pending = False      # an orientation decision waiting for the arrangement's first size
        self._extra_boxes = []            # control boxes another window adds to the left column (add_control_box)

    # The decomp's state as the simulator's attributes, read and set on the decomp (ldos_decomp.Decomp); a set
    # value is not checked, and the window's decomp line follows it (_decomp_changed).
    @property
    def plot_level(self):
        """Machinery: the decomp's level (0 markers, 1 parts, 2 atom types, 3 orbitals): ldos_decomp.Decomp.level."""
        return self.decomp.level

    @plot_level.setter
    def plot_level(self, value):
        """Machinery: the decomp's level set (ldos_decomp.Decomp.level), then the window's decomp line."""
        self.decomp.level = value
        self._decomp_changed()

    @property
    def active_marker_idx(self):
        """Machinery: the active marker's index (the marker levels 1 to 3 decompose): ldos_decomp.Decomp.marker."""
        return self.decomp.marker

    @active_marker_idx.setter
    def active_marker_idx(self, value):
        """Machinery: the active marker's index set (ldos_decomp.Decomp.marker), then the window's decomp line."""
        self.decomp.marker = value
        self._decomp_changed()

    @property
    def active_element(self):
        """Machinery: the active element (the atom type level 3 shows the orbitals of), or None: ldos_decomp.Decomp.element."""
        return self.decomp.element

    @active_element.setter
    def active_element(self, value):
        """Machinery: the active element set (ldos_decomp.Decomp.element), then the window's decomp line."""
        self.decomp.element = value
        self._decomp_changed()

    def run_interactive(self, grid_res=64, topo_bias=0.2, topo_height=2.5,
                        ldos_bias_sign='neg', use_decay=True, decay_linearized=False,
                        show_decay_toggle=True, reuse_cache=True,
                        line_endpoints=None, marker_positions=None, extra_broadening=None,
                        path=None, path_extend=False):
        """Machinery: the window opened: the run's settings taken (decay_linearized: the Decay-on model,
        'linearized' when True, 'exact' when False), the global topography loaded or converged, the
        controls made and the view built."""
        if topo_bias is None or topo_bias == 0:
            raise ValueError(f"topo_bias must be a nonzero bias voltage in volts; received {topo_bias!r}.")
        self.ldos_bias_sign = ldos_bias_sign
        self.use_decay = use_decay
        self.decay_linearized = bool(decay_linearized)
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
            self._form_marker_set()                # top to bottom on the stripe, then coloured by index
        # The global topography's bias V_b: the energy of topo_bias's snapped DOS sample (a Python float, the
        # parser's float64 value); it replaces topo_bias everywhere (sim_wrapper reads it as the topography's bias).
        self.global_topo_bias = float(self.energies[_snap(self.energies, topo_bias, 'zero_side')])
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
        cache_name = f"global_topo_{self.global_topo_bias}V_{topo_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}_fold{self._precision_cache_suffix()}.npy"
        if self.reuse_cache and exists(cache_name):
            print(f"[*] Loading Cached Global Topography: {cache_name}")
            self.current_z_map = np.load(cache_name)
            z_map_gpu = cp.array(self.current_z_map, dtype=cp.float32)
        else:
            print(f"[*] Cache not found. Calculating Global Topography: {cache_name}")
            init_cur, t_samples = self._topo_current(cp.hstack([grid_xy_gpu, z_fixed[:, None]]),
                                                     self.global_topo_bias, self._effective_decay_model())
            target_setp = cp.max(init_cur)
            print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
            z_map_gpu = self._converge_tip_height(z_fixed, grid_xy_gpu, t_samples, self.global_topo_bias, target_setp,
                                                  decay_model=self._effective_decay_model())
            self.current_z_map = cp.asnumpy(z_map_gpu)
            np.save(cache_name, self.current_z_map)
        self.grid_xy = grid_xy
        self.global_z_map = self.current_z_map.copy()
        self.grid_xy_gpu = grid_xy_gpu
        self.fig = plt.figure(figsize=DESIGN, fill_screen=True)   # opened on the screen's available area; never trimmed to its panels
        self._make_controls()
        self._build_ui()
        plt.show()

    # ------------------------------------------------------------------
    # 'Feenstra': the display divided by I/V (feenstra.py)
    # ------------------------------------------------------------------
    def _map_sample_indices(self, targets, ends=None):
        """
        Machinery: the DOSCAR sample each Map energy is snapped to, in the order given: the nearest of
        the displayed samples (_display_indices of E Min and E Max, so an end at exactly 0 V is excluded
        as the display excludes it), a tie going to the sample closer to 0 (_snap 'zero_side').  A Map
        energy of exactly 0.0 therefore goes to the nearest sample on the window's side of 0, and no Map
        energy lands outside the displayed range.  Map energies on the same sample show the same map.
        Everything a Map energy feeds (its map's LDOS, the map's title, the Map Norm's x, the energy
        marker) uses its sample's energy (map_e_targets = energies[map_e_idx]).  `ends`: the typed E Min
        and E Max of the window whose displayed samples are taken (an energy marker's release: Map's kept
        window, map_typed[:2]); None, the boxes'.
        """
        e_min, e_max = (self.s_emin.val, self.s_emax.val) if ends is None else ends
        disp = self._display_indices(e_min, e_max)
        e_disp = self.energies[disp]
        return np.array([int(disp[0]) + _snap(e_disp, t, 'zero_side')
                         for t in np.asarray(targets, dtype=np.float64).ravel()], dtype=np.int64)

    def _bias_sample(self):
        """
        Machinery: the window's bias end and its snapped DOS sample, (index, V_b).  E Min = 0.0 makes E
        Max the bias end, E Max = 0.0 makes E Min the bias end; otherwise ldos_bias_sign decides, as
        before.  V_b is the energy of the bias end's sample snapped by that end's own rule (_snap 'emin'
        or 'emax'), a Python float (the parser's float64 value); it replaces the typed bias everywhere it
        is used (the topography window's end, the frozen-bias weight, the cached-bias comparisons, the
        cache names, the titles).  With both ends 0.0 the bias end is exactly 0 V: (ZERO_V, 0.0) (the
        topography raises its ValueError for it).
        """
        emin, emax = self.s_emin.val, self.s_emax.val
        if emin == 0.0:
            end, kind = emax, 'emax'
        elif emax == 0.0:
            end, kind = emin, 'emin'
        elif str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative']:
            end, kind = emin, 'emin'
        else:
            end, kind = emax, 'emax'
        i = _snap(self.energies, end, kind)
        if i is ZERO_V:
            return ZERO_V, 0.0
        return i, float(self.energies[i])

    def _feenstra_iv(self, which):
        """
        Machinery: the I/V the display is divided by, feenstra.iv_ratio(E, I) (the plain I(V)/V) of the
        current stored with the displayed LDOS (the display run's own pass, with_current=True) and the GPU
        energies of its samples: which = 'grid' for the Line / Path points (cached_current at
        cached_current_E), 'map' for the Map grid (Map's own map_current at map_current_E, one column per
        map, at the Map energies), 'map_spec' for the Map markers (Map's own map_spec_current at
        map_spec_current_E), 'spec' for the Single Point markers (cached_spec_current at
        cached_spec_current_E).  (n_pts, n_samples) float32 CuPy.
        """
        if which == 'grid':
            return feenstra.iv_ratio(self.cached_current_E, self.cached_current)
        if which == 'map':
            return feenstra.iv_ratio(self.map_current_E, self.map_current)
        if which == 'map_spec':
            return feenstra.iv_ratio(self.map_spec_current_E, self.map_spec_current)
        return feenstra.iv_ratio(self.cached_spec_current_E, self.cached_spec_current)

    def _build_path_xy(self):
        """Machinery: build resampled path sample points from the current draggable nodes.

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
        Machinery: every control of the window, made once per figure (a window owns its state; the controls
        only report): the view radio, RUN, the toggles, the broadening box, the tip-height boxes
        with their Rerun buttons, the bottom strip's boxes, the cluster's label; the pick / drag
        handlers connected once.  The window's layout is installed here; _build_ui fills the
        panel area per view and shows the controls of the view.
        """
        fig = self.fig
        # automatic layout: the one rule for every window (nothing placed by hand).
        self.ui_radio = controls.make(fig, 'mode', value=self.mode)
        self.ui_radio.on_change(self._on_mode_change)
        # The colour-map set: 'default 1' or 'default 2' (cmap_shading.SET_NAMES), every panel recoloured in place on a change.
        self.ui_cmap_set = controls.Radio(fig, 'cmap set', value=self.cmap_set, labels=list(cmap_shading.SET_NAMES), title='cmap set')
        self.ui_cmap_set.on_change(self._on_cmap_set)
        self.btn_run = controls.make(fig, 'RUN')
        # Its label names what a click does ('RUN' while stopped, 'STOP' while running: _toggle_run): the
        # settings section lists it (controls.settings_text).
        self.btn_run.stateful = True
        self.btn_run.setting_label = 'RUN button'
        self.btn_run.set_text('STOP' if self.is_running else 'RUN')
        self.btn_run.on_change(self._toggle_run)
        # One list defines every toggle: label and the attribute it drives.  'exact' and 'linearized' sit indented
        # under 'Decay': the Decay-on model, exactly one of them on ('linearized' drives decay_linearized, 'exact'
        # shows its opposite and drives nothing itself: None); a click on one turns it on and the other off
        # (_on_toggle).  show_decay_toggle=False hides the three.
        toggle_defs = [('Atoms', 'show_atoms'), ('Decay', 'use_decay'),
                       ('exact', None), ('linearized', 'decay_linearized'),
                       ('Norm', 'normalize'), ('Feenstra', 'feenstra'),
                       ('Mag', 'show_mag'),
                       ('Cell', 'show_unit_cell'), ('Extend', 'path_extend')]
        if not getattr(self, 'show_decay_toggle', True):
            toggle_defs = [d for d in toggle_defs if d[0] not in ('Decay', 'exact', 'linearized')]
        self._toggle_defs = toggle_defs
        labels = [label for label, _ in toggle_defs]
        self.chk = controls.make(fig, 'sim toggles', labels=labels,
                                 values=[(not self.decay_linearized) if label == 'exact' else getattr(self, attr)
                                         for label, attr in toggle_defs],
                                 indents=['    ' if label in ('exact', 'linearized') else 0 for label in labels])
        self.chk.on_change(self._on_toggle)
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
        # The Contour / Shading box (Map): the Contour and Shading toggles and the Height / Relief radio ('shading
        # by'), made by cmap_shading.build_box with this figure and the controls module, at the window's box state
        # (shade_state); untitled, as the 'cmap norm' box.  A click applies the box's rules (at most one on, both
        # may be off) and redraws the energy maps only (_on_shade_box).
        self.shade_box = cmap_shading.build_box(fig, controls, on_change=self._on_shade_box, state=self.shade_state,
                                                title=None)
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
        # the arrangement follows it (rule 11; the window keeps its size).
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
        # The label over the Line / Path cluster and the Map rows ('LDOS', or 'LDOS: <element>' at level 3 (the orbitals)).
        self._cluster_label = controls.Label(fig, text='LDOS', fs=10)
        # The legend panel's Back button (Single Point, Line, Path, Map; kept across views, as the tip-height boxes):
        # gray, back one level as a right click (_legend_back: ignored while Pan or Zoom is on); hidden at level 0
        # with its place kept.  Its double-click event is dropped and a right press on it goes back (_LegendEvents).
        self._btn_back = controls.Button(fig, 'Back')
        self._btn_back.qt.setStyleSheet(self.LEGEND_BACK_CSS)
        pol = self._btn_back.qt.sizePolicy()
        pol.setRetainSizeWhenHidden(True)
        self._btn_back.qt.setSizePolicy(pol)
        self._btn_back.on_change(self._legend_back)
        self._legend_events = _LegendEvents(self)
        self._btn_back.qt.installEventFilter(self._legend_events)
        self._legend_rezoom_pending = False        # a redraw of the legend stack at a new text zoom is queued (_legend_placed)

        # The window's layout: the title cell, the left control column beside the panel area, the
        # bottom strip (automatic layout: the one rule for every window; nothing placed by hand).
        self._title_cell = fig.cell(min_px=(40, 24), stretch=(True, False))
        fig.set_title_cell(self._title_cell)
        fig.suptitle(self.spin_mode.name, fontsize=12)
        self._hook_layout = QtWidgets.QVBoxLayout()        # the boxes other windows add (add_control_box)
        self._hook_layout.setContentsMargins(0, 0, 0, 0)
        self._hook_layout.setSpacing(round(4 / (1000 / 72) * plt.em_px()))
        self._left_column = controls.column(
            controls.box(self.ui_radio),
            controls.box(self.ui_cmap_set),
            controls.box(self.chk_broad, controls.column(*self.s_broad.values()), title='broadening'),
            self._hook_layout,
            'stretch',
            controls.box(self.btn_run, self.chk))
        left_w = controls.widget_of(self._left_column)
        left_w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        self._panel_area = QtWidgets.QVBoxLayout()         # the view's arrangement (controls.Arrangement; _build_ui fills it)
        self._panel_area.setContentsMargins(0, 0, 0, 0)
        self._panel_area.setSpacing(round(6 / (1000 / 72) * plt.em_px()))
        main = QtWidgets.QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(round(8 / (1000 / 72) * plt.em_px()))
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
        # The Contour / Shading box after 'cmap norm', in its own frame (the ShadeBox's qt).
        self._strip_boxes['contour / shading'] = self.shade_box.qt
        strip.addWidget(self.shade_box.qt)
        root = QtWidgets.QVBoxLayout()
        root.setSpacing(round(6 / (1000 / 72) * plt.em_px()))
        root.addWidget(self._title_cell)
        root.addLayout(main, 1)
        root.addLayout(strip)
        fig.set_layout(root)
        for w in self._extra_boxes:
            self._hook_layout.addWidget(w)
        self._panel_cells = []
        self._arrangement = None                           # the view's controls.Arrangement (the shaped groups and the curve panels)
        self._cluster_row = None                           # Line / Path: the heatmap row's PanelRow (its panels built again only when their count changes)
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
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)     # the wheel over the legend stack
        # The decomp's line in the new window's info panel (kept current by _decomp_changed).
        self._decomp_changed()

    def add_control_box(self, widget):
        """
        Machinery: a control box of another window placed in this window's left column, under the
        broadening box and above RUN (the decomposition wrapper's σ box and its 'Send to decomp'
        buttons); kept across view changes (the left column is not rebuilt).
        """
        self._extra_boxes.append(widget)
        if getattr(self, '_hook_layout', None) is not None:
            self._hook_layout.addWidget(widget)
            widget.setVisible(True)

    def _cell_offsets(self, n):
        """Machinery: the in-plane offsets of the periodic images drawn, -n .. n cells each way."""
        return [tuple(i * self.lv[0, :2] + j * self.lv[1, :2]) for i in range(-n, n + 1) for j in range(-n, n + 1)]

    def _panel(self, layout, stretch=1, inner_w=None):
        """Machinery: a panel in a row or column layout: a new cell added with `stretch`, its axes returned (automatic layout)."""
        cell = self.fig.cell(inner_w=inner_w)
        layout.addWidget(cell, 0 if inner_w is not None else stretch)
        self._panel_cells.append(cell)
        return self.fig.add_axes(cell=cell)

    def _pair(self, box, inner_w=None):
        """
        Machinery: a new panel with the control box `box` above it (controls.PanelWithBox: the box belongs to
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
        """Machinery: a panel with its control box above it (_pair) added to a row with `stretch`, its axes returned."""
        ax, pair = self._pair(box, inner_w=inner_w)
        layout.addWidget(pair, 0 if inner_w is not None else stretch)
        return ax

    def _placement(self):
        """
        Machinery: the placement topo in its crop panel (controls.CropPanel): the Topo h box and the Crop /
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
        """Machinery: a free panel on its own (the curve panels the arrangement gives the remaining room): (axes, cell)."""
        cell = self.fig.cell()
        self._panel_cells.append(cell)
        return self.fig.add_axes(cell=cell), cell

    # --- the legend panel (right of the spectrum: Single Point, Line, Path, Map) ---

    LEGEND_SAMPLE_PX = 26      # an entry's sample line, as qtplot's legend sample (pixels at the old fixed 100 DPI, converted by em_px and the text zoom where used)
    LEGEND_PAD_PX = 4          # the room inside an entry's box around its sample and label
    LEGEND_GAP_PX = 4          # between an entry's sample and its label (qtplot's legend horSpacing)
    LEGEND_ROW_GAP_PX = 2      # between two entries' boxes, and between a box and the stack's edges
    LEGEND_BOX_FILL = '#f0f0f0'     # a clickable entry's box: controls.FRAME_CSS's fill and border colours
    LEGEND_BOX_EDGE = '#c8c8c8'
    LEGEND_BOX_LW = 0.72            # the box's border in points: FRAME_CSS's 1 px at the old fixed 100 DPI
    LEGEND_BACK_CSS = "QPushButton { background: #c8c8c8; }"      # the Back button's gray

    def _spectrum_and_legend(self, spec_cell):
        """
        Machinery: the spectrum's sub-section in Single Point, Line and Path: its cell and, right of it in one
        row, the legend panel (_legend_build).  Returns the sub-section's widget.
        """
        self._spec_sub = controls.widget_of(controls.row(spec_cell, self._legend_build()))
        return self._spec_sub

    def _legend_build(self):
        """
        Machinery: the legend panel right of the spectrum, a composite of fixed width (_legend_size) spanning
        the spectrum's plot box (_legend_follow): from the top, the title cell (a figure text, 10 pt bold,
        saved with the figure: '' at level 0, 'Parts', 'Atoms', 'Orbitals'), the Back button (hidden at level
        0, its place kept) and the stack, a qtplot axes with no ticks or frame on white whose data units are
        pixels (y downward from the first entry), with a Qt vertical scroll bar at its right edge (shown only
        while the entries do not fit, its strip always reserved; _legend_view).  The stack's entries come
        from the decomp (ldos_decomp.Decomp.legend, _update_legend).  The stack axes is out of the toolbar's
        Pan and Zoom (Axes.set_navigate(False)), left out of the CSV and draws its full list in the vector
        saves (Axes.csv, Axes.export_extent: _legend_extent).  Returns the panel's widget: in Single Point,
        Line and Path the second item of the spectrum's row (_spectrum_and_legend), in Map the map block's
        side item beside the spectrum's cell (controls.MapBlock.set_groups, extra_side).
        """
        fig = self.fig
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(round(2 / (1000 / 72) * plt.em_px()))
        title_cell = fig.cell(min_px=(10, 10), stretch=(True, False))
        stack_cell = fig.cell(min_px=(10, 10))
        self._panel_cells.extend([title_cell, stack_cell])
        lay.addWidget(title_cell)
        lay.addWidget(self._btn_back.qt)           # detached (hidden) since the last build
        bar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        pol = bar.sizePolicy()
        pol.setRetainSizeWhenHidden(True)          # its strip kept while it is hidden
        pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)   # the stack's full height, never a minimum of its own
        bar.setSizePolicy(pol)
        bar.setSingleStep(20)                      # QScrollArea's own single step
        bar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)   # a right press goes back (_LegendEvents)
        bar.installEventFilter(self._legend_events)
        bar.valueChanged.connect(self._legend_scrolled)
        stack = QtWidgets.QHBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(stack_cell, 1)
        stack.addWidget(bar, 0)
        lay.addLayout(stack, 1)
        ax = fig.add_axes(cell=stack_cell)
        ax.axis('off')
        ax.set_facecolor('w')
        ax.set_navigate(False)                     # out of the toolbar's Pan and Zoom (no wheel zoom, drag or Home); the wheel still scrolls it (_on_scroll)
        ax.csv = False                             # export_csv leaves the samples out
        ax.export_extent = self._legend_extent     # the vector saves draw the full list
        title, entries, back = self.decomp.legend([])
        self._legend_title = fig.text(0.5, 0.5, title, cell=title_cell, fontsize=10, fontweight='bold',
                                      ha='center', va='center')
        self._btn_back.show(back)
        self._legend_state = (title, back)
        self._legend_ax, self._legend_bar, self._legend_panel = ax, bar, panel
        self._legend_title_cell, self._legend_stack_cell = title_cell, stack_cell
        self._legend_size()
        self._legend_draw_zoom = float(fig.font_zoom)  # nothing drawn yet: the stack is empty at this zoom
        self._legend_view()
        ax.on_placed(self._legend_placed)
        self.ax_spec.on_placed(self._legend_follow)
        return panel

    def _legend_size(self):
        """
        Machinery: the legend panel's sizes at the window's text zoom (nothing when they are already at it):
        an entry's row height (the label font's height with the text item's two document margins, the
        box's padding and the gap between boxes); the panel's one fixed width, the widest entry any level
        can show for the loaded data (ldos_decomp.all_labels: the box's border and padding, the sample, the
        gap and the widest label) plus the scroll bar's strip; the title cell's height (the title font's
        with its margins); the stack's smallest height, one row (with the title and Back, the panel's
        smallest height).
        """
        z = float(self.fig.font_zoom)
        if z == self._legend_zoom:
            return
        self._legend_zoom = z
        k = 1 / (1000 / 72) * plt.em_px()          # pixels per pixel at the old fixed 100 DPI
        doc = k                                    # a text item's document margin (qtplot._TextItem), each side
        fm = QtGui.QFontMetricsF(plt.qfont(10, zoom=z))
        self._legend_m = 0.5 * self.LEGEND_ROW_GAP_PX * k * z
        self._legend_b = plt.pt2px(self.LEGEND_BOX_LW)
        self._legend_pad = self.LEGEND_PAD_PX * k * z
        self._legend_len = self.LEGEND_SAMPLE_PX * k * z
        self._legend_gap = self.LEGEND_GAP_PX * k * z
        self._legend_row_h = fm.height() + 2 * doc + 2 * self._legend_pad + 2 * self._legend_m
        label_w = max((fm.horizontalAdvance(s) for s in ldos_decomp.all_labels(self.atomtypes, self.orbitals)),
                      default=0.0)
        stack_w = (2 * (self._legend_m + self._legend_b + self._legend_pad) + self._legend_len + self._legend_gap
                   + label_w + 2 * doc)
        bar_w = int(self._legend_bar.sizeHint().width())
        W = int(np.ceil(stack_w)) + bar_w
        self._legend_panel.setFixedWidth(W)
        self._legend_w = float(W - bar_w)          # the stack's width as laid out (its x range)
        title_h = QtGui.QFontMetricsF(plt.qfont(10, weight='bold', zoom=z)).height() + 2 * doc
        self._legend_title_cell.setFixedHeight(int(np.ceil(title_h)))
        self._legend_stack_cell.setMinimumHeight(int(np.ceil(self._legend_row_h)))

    def _legend_draw(self, entries, to_top):
        """
        Machinery: the legend stack drawn anew for `entries` (ldos_decomp.Entry, top to bottom): entry k in
        the row from k to k + 1 row heights; its sample, a line in its curve's colour, line style and line
        width; its label (10 pt) when it has one (level 0: none, the colour only); a box around both for a
        clickable entry (levels 0 to 2: a PolyCollection quad in LEGEND_BOX_FILL under a LineCollection
        outline in LEGEND_BOX_EDGE), none for level 3's plain key rows.  The boxes are kept as drawn
        (_legend_drawn) for the clicks (_legend_entry_at); nothing is pickable or labelled as a marker.
        The scroll goes to the top with `to_top` (a level change), else it is kept, clamped (_legend_view).
        """
        ax, bar = self._legend_ax, self._legend_bar
        self._legend_draw_zoom = float(self.fig.font_zoom)
        m, b, pad, row_h = self._legend_m, self._legend_b, self._legend_pad, self._legend_row_h
        self._legend_content_h = len(entries) * row_h
        if to_top:
            blocked = bar.blockSignals(True)
            try:
                bar.setValue(0)
            finally:
                bar.blockSignals(blocked)
        ax.clear()                                 # the axis stays off; its placement sets the view (_legend_placed)
        x_s = m + b + pad
        x1 = self._legend_w - m
        fills, outlines, drawn, samples = [], [], [], []
        for k, e in enumerate(entries):
            y0, y1, yc = k * row_h + m, (k + 1) * row_h - m, (k + 0.5) * row_h
            if e.clickable:
                quad = [(m, y0), (x1, y0), (x1, y1), (m, y1)]
                fills.append(quad)
                outlines.append(quad + [quad[0]])
                drawn.append((m, y0, x1, y1, e))
            ln, = ax.plot([x_s, x_s + self._legend_len], [yc, yc], color=e.color, lw=e.linewidth,
                          linestyle=e.linestyle, zorder=2.5)
            samples.append(ln)
            if e.label:
                ax.text(x_s + self._legend_len + self._legend_gap, yc, e.label, fontsize=10, ha='left', va='center')
        if fills:
            ax.add_collection(plt.PolyCollection(fills, facecolors=self.LEGEND_BOX_FILL, zorder=1))
            ax.add_collection(plt.LineCollection(outlines, colors=self.LEGEND_BOX_EDGE, linewidths=self.LEGEND_BOX_LW,
                                                 zorder=1.5))
        self._legend_drawn, self._legend_samples, self._legend_entries = drawn, samples, list(entries)
        self._legend_view()

    def _update_legend(self):
        """
        Machinery: the legend panel brought to the spectrum's curves as drawn (_spec_curves; called at the
        end of every _draw_spectrum, built or in place): the title and the Back button from the decomp
        (ldos_decomp.Decomp.legend); the stack drawn anew only when its entries' labels or count, or the
        level, changed (_legend_draw: to the top on a level change), else its samples recoloured in place
        where their colour differs (a Line / Path re-sort).
        """
        if self._legend_ax is None:
            return
        title, entries, back = self.decomp.legend(self._spec_curves)
        if (title, back) != self._legend_state:
            self._legend_title.set_text(title)
            self._btn_back.show(back)
            self._legend_state = (title, back)
        key = (self.plot_level, tuple(e.label for e in entries))
        if key != self._legend_key:
            to_top = self._legend_key is None or self._legend_key[0] != self.plot_level
            self._legend_key = key
            self._legend_draw(entries, to_top)
            return
        for ln, e in zip(self._legend_samples, entries):
            if ln.get_color() != e.color:
                ln.set_color(e.color)
        clickable = [e for e in entries if e.clickable]
        self._legend_drawn = [(x0, y0, x1, y1, e) for (x0, y0, x1, y1, _), e in zip(self._legend_drawn, clickable)]
        self._legend_entries = list(entries)

    def _legend_view(self):
        """
        Machinery: the legend stack's view for its box as placed: x from 0 to the stack's width, y from the
        scroll value down by the box's height (one data unit a pixel); the scroll bar configured as
        QScrollArea configures its own (range 0 to the entries' height less the box's, page step the box's
        height, single step 20), its value clamped into the range, and shown only while the entries do not
        fit.
        """
        ax, bar = self._legend_ax, self._legend_bar
        if ax is None:
            return
        h = max(float(ax._px_rect()[3]), 1.0)
        top = max(int(np.ceil(self._legend_content_h - h)), 0)
        v = min(max(int(bar.value()), 0), top)
        blocked = bar.blockSignals(True)
        try:
            bar.setRange(0, top)
            bar.setPageStep(max(int(h), 1))
            bar.setSingleStep(20)
            bar.setValue(v)
        finally:
            bar.blockSignals(blocked)
        bar.setVisible(top > 0)
        self._legend_h = h
        ax.set_xlim(0.0, self._legend_w)
        ax.set_ylim(v + h, v)

    def _legend_scrolled(self, v):
        """Machinery: the scroll bar moved (a drag, a click, the wheel): the stack's view from the new value down by its box's height; nothing is drawn again."""
        ax = self._legend_ax
        if ax is not None:
            ax.set_ylim(v + self._legend_h, v)

    def _legend_placed(self):
        """Machinery: after every placement of the legend stack (Axes.on_placed: a resize, a text zoom): its view for the new box (_legend_view); at a new text zoom its sizes and entries again, once, after the placement (_legend_rezoom)."""
        if self._legend_ax is None:
            return
        self._legend_view()
        if float(self.fig.font_zoom) != self._legend_draw_zoom and not self._legend_rezoom_pending:
            self._legend_rezoom_pending = True
            QtCore.QTimer.singleShot(0, self._legend_rezoom)

    def _legend_rezoom(self):
        """Machinery: the legend panel at the window's new text zoom: its sizes (_legend_size) and its stack drawn again with the same entries, the scroll kept (clamped); in Map, when its fixed width changed, the map block placed again (MapBlock.rearrange: its cells measured anew, the new width with them; a width change alone does not place the block), then the window arranged again (_rearrange: a resize-driven change), so the block's slot counts the new width at once."""
        self._legend_rezoom_pending = False
        ax = self._legend_ax
        if ax is None or ax not in self.fig.axes or float(self.fig.font_zoom) == self._legend_draw_zoom:
            return
        w0 = self._legend_panel.minimumWidth()
        self._legend_size()
        self._legend_draw(self._legend_entries, False)
        if self.mode == 'Map' and self._map_block is not None and self._legend_panel.minimumWidth() != w0:
            self._map_block.rearrange()
            self._rearrange()

    def _legend_follow(self):
        """Machinery: the legend panel spans the spectrum's plot box: its top and bottom margins follow the box in the spectrum's cell (ax_spec's placement), as controls.UnderPanel._follow does sideways (the panel and the cell share their row's top and height)."""
        panel = self._legend_panel
        cell = getattr(self.ax_spec, '_cell', None)
        if panel is None or cell is None:
            return
        cx, cy, cw, ch = cell.rect_px()
        x0, y0, w, h = self.ax_spec._px_rect()
        top = max(int(round(y0 - cy)), 0)
        bottom = max(int(round(cy + ch - (y0 + h))), 0)
        lay = panel.layout()
        m = lay.contentsMargins()
        if (m.top(), m.bottom()) != (top, bottom):
            lay.setContentsMargins(0, top, 0, bottom)

    def _legend_extent(self):
        """Machinery: the legend stack's full list for the vector saves (its Axes.export_extent): (content_h_px, (y_top, y_bottom)), the height in logical pixels every entry needs and the y limits that show them from the first entry (0) at the top to max(the box's height on screen, that height) at the bottom, one data unit a pixel as on screen."""
        h = float(self._legend_ax._px_rect()[3])
        c = float(self._legend_content_h)
        return (c, (0.0, max(h, c)))

    def _in_legend(self, event):
        """Machinery: whether a canvas press lies in the legend panel: on its stack (event.inaxes), or in its rectangle from the title cell's top to the stack's bottom across the panel's width (the title, the Back button's row, the scroll bar's strip)."""
        if self._legend_ax is None:
            return False
        if event.inaxes is self._legend_ax:
            return True
        if event.x is None or event.y is None:
            return False
        py = self.fig.canvas.viewport().height() - event.y
        tx, ty, tw, th = self._legend_title_cell.rect_px()
        sx, sy, sw, sh = self._legend_stack_cell.rect_px()
        return tx <= event.x <= tx + tw and ty <= py <= sy + sh

    def _legend_entry_at(self, event):
        """Machinery: the entry (ldos_decomp.Entry) whose box, as drawn at the press (_legend_drawn), holds a press on the legend stack, else None (a gap, a level-3 row, the title, the Back button's row)."""
        if event.inaxes is not self._legend_ax or event.xdata is None or event.ydata is None:
            return None
        for x0, y0, x1, y1, e in self._legend_drawn:
            if x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1:
                return e
        return None

    def _legend_back(self):
        """Machinery: the Back button's click, or a right press on the legend panel's Back button or scroll bar (Qt widgets: _LegendEvents): back one level, as a right click in the panel; ignored while the toolbar's Pan or Zoom is on, as the canvas's presses are."""
        if self.fig.canvas.manager.toolbar.mode == "":
            self._decomp_click(plt.MouseButton.RIGHT)

    def _on_scroll(self, event):
        """Machinery: the mouse wheel over the legend stack forwarded to its scroll bar (QCoreApplication.sendEvent, as QAbstractScrollArea forwards its wheel: a mostly vertical wheel), which takes Qt's own step and moves the stack (_legend_scrolled)."""
        ax = self._legend_ax
        if ax is None or event.inaxes is not ax or event.guiEvent is None:
            return
        d = event.guiEvent.angleDelta()
        if abs(d.y()) >= abs(d.x()):
            QtCore.QCoreApplication.sendEvent(self._legend_bar, event.guiEvent)

    @staticmethod
    def _detach(*widgets):
        """Machinery: the reused widgets (the tip-height boxes, the cluster's label) taken out of the pair or layout about to be deleted: hidden and parentless until the next build puts them back and shows them."""
        for w in widgets:
            if w is not None:
                w.hide()
                w.setParent(None)

    @staticmethod
    def _empty_layout(lay, keep=()):
        """Machinery: every item of a layout taken out: the widgets in `keep` detached for reuse, the others deleted, sub-layouts emptied too."""
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
        return (self._box_topo_h, self._box_ldos_h, self._cluster_label.qt, self._box_crop, self._btn_back.qt) + tuple(b.qt for b in self._cursor_boxes.values())

    def _clear_cluster(self):
        """Machinery: the Line / Path cluster's axes removed and its row emptied (it is built again only when its panel count changes: _build_heat_row); the LDOS h box detached from its pair first."""
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
        # The built row's state (_build_heat_row, _fill_heat_row): none until the row is built again.
        self._row_key = None                       # (heatmap count, per-heatmap colour bars) of the built row
        self._heat_axes, self._heat_caxes = [], []  # the heatmaps' axes and their colour bars' (None: no bar)
        self._heat_meshes, self._heat_grids, self._heat_cmaps = [], [], []
        self._heat_cbars, self._heat_titles = [], []
        self._stripe_lc, self._stripe_len, self._prof_line = None, None, None
        self._line_topo_arts, self._line_topo_key, self._line_topo_z = [], None, None
        self._line_markers, self._line_markers_n = [], None

    def _clear_map_rows(self):
        """Machinery: the Map rows' axes removed and their cells deleted out of the block (they are rebuilt when the counts change)."""
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
            self._map_block.set_groups([], None, extra=getattr(self, '_spec_cell', None),
                                       extra_side=getattr(self, '_legend_panel', None))
        # The built block's state (_build_map_block, _fill_map): none until the block is built again.
        self._block_key = None                     # (partitions, E Pts, 'cmap norm', per-partition bars) of the built block
        self._map_state, self._map_cbar_state, self._map_marks = {}, {}, {}
        self._map_titles, self._map_ylabels = {}, {}
        self._map_marks_xy, self._contour_z = None, None

    def _clear_panels(self):
        """Machinery: every panel of the view removed: its axes, its cells, the panes of the panel area; the reused boxes and label detached first."""
        self._clear_cluster()
        self._clear_map_rows()
        for ax in list(self.fig.axes):
            ax.remove()
        # The legend panel (Single Point, Line, Path, Map): its stack axes went with the others; its title, a figure
        # text, goes here; its widgets with the panel area, the Back button detached for the next build.
        if getattr(self, '_legend_title', None) is not None:
            self._legend_title.remove()
        self._legend_ax = self._legend_title = self._legend_bar = self._legend_panel = None
        self._legend_title_cell = self._legend_stack_cell = self._spec_sub = None
        self._legend_drawn, self._legend_samples, self._legend_entries = [], [], []
        self._legend_key = self._legend_state = self._legend_zoom = self._legend_draw_zoom = None
        self._legend_content_h, self._legend_h, self._legend_w = 0.0, 1.0, 1.0
        self._detach(*self._kept_widgets())
        self._empty_layout(self._panel_area, self._kept_widgets())
        self._arrangement = None
        self._spec_cell = None
        self._contour_artists, self._level_line, self.cax_topo = [], None, None
        self._crop_ax = self._crop_panel = None
        self._cursor_artists, self._shade_artists = [], []
        self._panel_cells = []
        self._cluster_row = None
        self._map_block = None
        self._map_cells = []
        # The spectrum's built curves (_draw_spectrum) and the Line / Path drag cache with its Norm factors: none
        # in the new view.
        self._spec_key, self._spec_lines, self._spec_emarkers, self._spec_title = None, [], [], None
        self._spec_curves = []                     # the decomp's records of _spec_lines (ldos_decomp.Curve), in the same order
        self._line_disp = None
        self._line_norm = None
        if getattr(self, 'cursor', None) is not None:
            self.cursor.remove()
            self.cursor = None

    def _build_ui(self):
        """
        Machinery: the panels of the current view arranged by the one rule (controls.Arrangement; rule 11 of
        the guide as the owner stated it: the plots fill the window, spatial panels keep their
        ratio, the heat-map data comes first, a curve panel never grows at its expense yet stays
        readable, blank space and the controls' footprint are what is minimized).  The shaped groups of a
        view: the placement topo in its crop panel (Topo h, Crop / Uncrop and the cursor boxes
        above and beside it); in Line / Path the heatmap row (its label over it), sized and scored
        with each heatmap counted as square, then placed in its whole slot, which its heatmaps fill
        (Arrangement fill_slot, PanelCell.fill_as); in Map the
        Global Topo and Map Topo pair (their boxes above) and the energy maps' wrapping block.  The
        curve panels (Tip Height and Partitioned LDOS, side by side in Line, in Path beside or over each other by their room)
        take the room the heatmaps and topographies leave, at least their readable size
        (_curves_readable), the topographies at most twice a heatmap (_topo_cap); in Map the
        spectrum is a cell of the map block, a map's size; in Single Point it is a first-class
        panel beside the placement topo.
        The window is never trimmed to its panels (no Figure.fit_window): it keeps the size it is
        given, and a maximized window stays full screen.  The panels are built once per view (here)
        and then updated in place (_update_all); the arrangement is chosen again, and the panels
        moved, only when their number changes (a view change; the heatmap row or the Map block
        built again for a new count), when a crop changes a panel's shape, or when the user resizes
        the window (maximizing included); the unit-cell panels' orientation (as drawn, or turned 90 degrees: the way the
        window fits better, _choose_orientation) is decided when the view is built and on Crop /
        Uncrop.  The controls of the view are shown and the others hidden, then
        _update_all(full_refresh=True), in Map with last_run (with RUN off too, Map's kept maps, energy
        markers and spectrum are drawn from what is kept).  Called at the first show and on every view change; the
        controls themselves are made once (_make_controls).  automatic layout: the one rule for
        every window (nothing placed by hand).
        """
        fig = self.fig
        self._clear_panels()
        mode = self.mode
        self._strip_boxes['Path Pts'].setVisible(mode == 'Path')
        self._strip_boxes['E Pts'].setVisible(mode == 'Map')
        self._strip_boxes['cmap norm'].setVisible(mode == 'Map')
        self._strip_boxes['contour / shading'].setVisible(mode == 'Map')
        # The counts the view reads, put in the boxes without firing.
        self.s_num_marks.set(len(self.marker_ratios) if mode in ('Line', 'Path') else len(self.marker_coords))
        if mode == 'Path':
            self.s_path_pts.set(len(self.path_nodes))
        if mode == 'Map':
            self.s_nepts.set(5 if self.cached_nepts is None else self.cached_nepts)
        if mode == 'Single Point':
            # The placement topo; the Partitioned LDOS beside it, a first-class panel (free_scored), with its
            # legend panel at its right (one sub-section).
            self.ax_map, topo = self._placement()
            self.ax_spec, spec_cell = self._cell_axes()
            groups, free = [topo], self._spectrum_and_legend(spec_cell)
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
        elif mode in ('Line', 'Path'):
            # The placement topo; the heatmap row (built by _update_all, again only when its panel
            # count changes; the LDOS h box above the Line Topo panel or above the Path stripe) under
            # its label, filling its whole slot (fill_slot); the Tip Height
            # profile and the spectrum in the room left (at least readable: rule 11), in Line side by side in every
            # room's shape, in Path side by side or stacked by that room's shape.
            self.ax_map, topo = self._placement()
            self._cluster_row = plt.PanelRow(spacing=4)
            cluster = controls.Titled(plt.RowWidget(self._cluster_row), self._cluster_label.qt)
            self.ax_prof, prof_cell = self._cell_axes()
            self.ax_spec, spec_cell = self._cell_axes()
            # The spectrum and its legend panel: one sub-sub-section of the 1D sub-section.  In Line the FlipBox
            # never stacks them (tall=0: no width-to-height ratio is below it), so the Tip Height profile and the
            # spectrum with its legend stay one row, Line's second row; in Path its default ratios.
            flip_kw = {'tall': 0.0} if mode == 'Line' else {}
            groups, free = [topo, cluster], controls.FlipBox((prof_cell, 25), (self._spectrum_and_legend(spec_cell), 10),
                                                             spacing=10, **flip_kw)
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
            # The topo colour bar beside the Map Topo (the browser's: the ramp, no numbers), carrying the
            # contour slice's level line; the Map Topo's iso-line at that level goes on every energy map while
            # the box's Contour is on.
            self.cax_topo = self._panel(row, inner_w=9)
            self._draw_topo_cbar(self.cax_topo)
            self._draw_level_line()
            topos = plt.RowWidget(row)
            self._map_block = controls.MapBlock(fig, spacing=4)
            maps = controls.Titled(self._map_block, self._cluster_label.qt)
            # The spectrum is one cell of the map block, a map's size, in the free slot of the block's last
            # row (or a row of its own): never a strip below the maps, never squeezed (rule 11).  Its legend
            # panel is the block's side item: right of the spectrum at its row's height, its fixed width
            # counted in the block's width (MapBlock extra_side, passed again by every set_groups).
            self.ax_spec, spec_cell = self._cell_axes()
            self._spec_cell = spec_cell
            self._map_block.set_groups([], None, extra=spec_cell, extra_side=self._legend_build())
            groups, free = [topos, maps], None
            self.map_axes = []
            self.cax_list = []
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            fig.align_row(self.ax_map_global, self.ax_map, self.cax_topo)
        # The sizing (rule 11): the heatmaps and the topographies are sized first and fill the window (the
        # topographies at most twice a heatmap, in width and in height); the curve panels, at least readable,
        # take the room those leave (Line / Path: the Tip Height profile and the spectrum in the room beside
        # or under the heatmap row; Map: the spectrum a cell of the map block); in Single Point the spectrum
        # is a first-class panel beside the placement topo (Arrangement free_min / free_scored / group_cap).
        # In Line / Path the heatmap row, sized and scored with each heatmap counted as square, is then
        # placed in its whole slot and its heatmaps fill it (fill_slot; PanelCell.fill_as, _build_heat_row).
        # Line allows only its two rows (layouts: row_top): the placement topo and the heatmap row across the
        # top, as large at their shapes as the width allows (the curve panels keep their readable height),
        # and the curve panels under them in the height that row leaves.
        self._arrangement = controls.Arrangement(groups, free, spacing=8, free_min=self._curves_readable,
                                                 free_scored=mode == 'Single Point',
                                                 group_cap=self._topo_cap(mode, groups),
                                                 fill_slot=cluster if mode in ('Line', 'Path') else None,
                                                 layouts=['row_top'] if mode == 'Line' else None)
        self._arrangement.on_arranged(self._on_arranged)
        self._panel_area.addWidget(self._arrangement, 1)
        self._box_topo_h.setVisible(True)
        self._box_ldos_h.setVisible(mode != 'Single Point')
        self._cluster_label.show(mode != 'Single Point')
        self._cluster_label.set_text('LDOS')
        # The info panel's lines of the view: the map energies (none until the view's maps are drawn) and the
        # contour level (Map).
        self._map_energies_changed()
        self._contour_level_changed()

        self.cached_emin, self.cached_emax = None, None
        # Into Map, with RUN off too: its kept maps, energy markers and spectrum drawn from what is kept (last_run).
        self._update_all(full_refresh=True, last_run=mode == 'Map')
        self._rearrange(orient=True)             # the view built: the unit-cell panels' orientation decided

    CURVE_BOX_MIN_PX = (220, 140)      # a curve panel's readable plot box at text zoom 1 (its labels come on top), in pixels at the old fixed 100 DPI (converted by em_px where used, _curves_readable)

    def _curve_axes(self):
        """Machinery: the curve panels of the view: the spectrum, with the Tip Height profile before it in Line / Path."""
        axs = [getattr(self, 'ax_prof', None)] if self.mode in ('Line', 'Path') else []
        return [a for a in axs + [getattr(self, 'ax_spec', None)] if a is not None]

    def _curves_readable(self):
        """
        Machinery: the curve panels' readable minimum (rule 11; the Arrangement's free_min): every curve
        panel's cell is given, as its minimum size, the readable plot box (CURVE_BOX_MIN_PX at the
        window's text zoom) plus its own labels (Axes._decor_px), so that a split between two
        panels never squeezes one below it; returns the free widget's minimum (w, h): the panels
        side by side, with the legend panel's fixed width (at the window's text zoom: _legend_size)
        and its row's spacing beside the spectrum in Single Point, Line and Path.  The panels take
        the room the heatmaps and topographies leave beyond it.
        """
        z = float(getattr(self.fig, 'font_zoom', 1.0) or 1.0)
        bw, bh = self.CURVE_BOX_MIN_PX
        ws, hs = [], []
        for ax in self._curve_axes():
            l, t, r, b = ax._decor_px()
            w, h = int(round(bw / (1000 / 72) * plt.em_px() * z + l + r)), int(round(bh / (1000 / 72) * plt.em_px() * z + t + b))
            if ax._cell is not None and (ax._cell.minimumWidth(), ax._cell.minimumHeight()) != (w, h):
                ax._cell.setMinimumSize(w, h)
            ws.append(w)
            hs.append(h)
        if not ws:
            return (0, 0)
        width = sum(ws) + 6 / (1000 / 72) * plt.em_px() * (len(ws) - 1)
        if self._spec_sub is not None:
            self._legend_size()
            width += self._legend_panel.minimumWidth() + self._spec_sub.layout().spacing()
        return (width, max(hs))

    TOPO_MAX_FACTOR = 2.0              # a topography's plot box at most this many times a heatmap's (Map, Line, Path)

    def _topo_axes(self):
        """Machinery: the topography panels of the view: the placement topo (Global Topo in Map) and the Map Topo."""
        if self.mode == 'Map':
            return [a for a in (getattr(self, 'ax_map_global', None), getattr(self, 'ax_map', None)) if a is not None]
        return [a for a in (getattr(self, 'ax_map', None),) if a is not None]

    def _topo_cap(self, mode, groups):
        """
        Machinery: the Arrangement's group_cap of the view (None in Single Point: no heatmap): the topography
        group's plot box at most TOPO_MAX_FACTOR times one heatmap's, in width and in height (Map:
        one energy map's box, MapBlock.map_box; Line / Path: one heatmap of the row).  A box's
        height in a region is the region's height less the room around the box; its width follows
        from the panel's own shape (Axes._aspect_want: the unit cell's ratio for a topo and a map;
        in Line a heatmap counted as square, as its row sizes it).
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
                h = max(rect[3] - heat_group._label_h() - hm_tb, 1.0)
                if mode == 'Line':
                    return (h, h)      # one heatmap's box counted as square (ratio 1.0): the row's first panel is the Line Topo (2:3)
                return with_width(h, hm[0])
        return (topo_group, heat_group, self.TOPO_MAX_FACTOR, box_a, box_b)

    # --- the contour slice (the level line on the topo colour bar, the Map Topo's iso-line on the energy maps) ---

    def _draw_topo_cbar(self, ax):
        """Machinery: the topo colour bar beside the Map Topo (the sm4_Browser's draw_topo_cbar): the topo colour map's ramp, low at the bottom, no ticks and no numbers, 'topo' above it; it spans 0 to 1, the fraction of the Map Topo's height range, and carries the contour slice's level line."""
        ax.set_navigate(False)
        ramp = np.linspace(0.0, 1.0, 256)[:, None]
        ax.imshow(ramp, aspect='auto', origin='lower', cmap=self._cmap('topo'), extent=[0, 1, 0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('topo', fontsize=7, pad=2)

    def _draw_level_line(self):
        """Machinery: the contour slice's level line on the topo colour bar: one pickable horizontal line at z_frac (CONTOUR_LEVEL_COLOR, label '_zlevel': _on_pick), drawn once per build, shown while the box's Contour is on and hidden (so not pickable) while it is off (_on_shade_box); _set_level moves it."""
        if self.cax_topo is None:
            return
        self._level_line = self.cax_topo.axhline(self.z_frac, color=CONTOUR_LEVEL_COLOR, lw=1.5, label='_zlevel',
                                                 picker=6, zorder=3.0)
        self._level_line.set_visible(self.shade_state.contour)

    def _set_level(self, frac):
        """Machinery: the contour slice's level set to `frac` of the Map Topo's height range (clipped to 0 .. 1): the level line moved, the traces made anew on every energy map, the window's contour level line (_contour_level_changed)."""
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac == self.z_frac:
            return
        self.z_frac = frac
        if self._level_line is not None:
            self._level_line.set_ydata([frac, frac])
        self._draw_contour_artists()
        self._contour_level_changed()
        self.fig.canvas.draw_idle()

    def _ramp_fraction(self, event, ax):
        """Machinery: the cursor's height on the topo colour bar `ax` as the fraction of its range, from the event's canvas pixel row and the bar's drawn box (Axes.get_window_extent, both from the bottom): the drag follows the cursor's height even when it leaves the narrow bar sideways.  None without a position."""
        if event.y is None:
            return None
        x0, y0, w, h = ax.get_window_extent().bounds
        if not h:
            return None
        return (float(event.y) - y0) / h

    def _contour_segments(self):
        """
        Machinery: the contour slice's traces in the plane, (n, 2, 2): contour.iso_segments
        on the LDOS-topo heights Map's maps were computed at (map_z, Map's own, grid_res x grid_res over the
        unit cell's fractional grid, as the Map Topo is drawn) at the level
        z_frac of their range (the lowest to the highest height, the colour
        range the Map Topo is drawn on), placed on the lattice at the 2 x 2
        images an energy map shows (contour.to_cell, _cell_offsets_2x2); empty
        without heights.
        """
        z = self.map_z
        if z is None:
            return np.zeros((0, 2, 2))
        n = int(round(np.sqrt(len(z))))
        if n < 2 or n * n != len(z):
            return np.zeros((0, 2, 2))
        img = np.asarray(z, dtype=np.float64).reshape(n, n)
        ok = np.isfinite(img)
        if not ok.any():
            return np.zeros((0, 2, 2))
        lo, hi = float(img[ok].min()), float(img[ok].max())
        return contour.to_cell(contour.iso_segments(img, lo + self.z_frac * (hi - lo)), img.shape, self.lv,
                               self._cell_offsets_2x2())

    def _draw_contour_artists(self):
        """Machinery: the contour slice drawn anew (the old traces removed) on every energy map, never on a topo panel: one LineCollection per map in CONTOUR_COLOR, 1 pt, above the map's colours and under the markers (WSxM's flooding, as outlines); nothing outside Map, before the maps exist or while the box's Contour is off (the old traces removed only)."""
        for art in self._contour_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._contour_artists = []
        if self.mode != 'Map' or not getattr(self, 'map_axes', None) or not self.shade_state.contour:
            return
        segs = self._contour_segments()
        if not len(segs):
            return
        figure_axes = self.fig.axes
        for ax in self.map_axes:
            if ax not in figure_axes:
                continue
            lc = plt.LineCollection(segs, colors=CONTOUR_COLOR, linewidths=1.0, zorder=4.5)
            ax.add_collection(lc)
            self._contour_artists.append(lc)

    def _rearrange(self, orient=False):
        """Machinery: the arrangement chosen again for the panels as they now are, at the window's size (rule 11): called when their number changes (the view built; the heatmap row or the Map block built again), when a crop changes a panel's shape (a resize arranges by itself) and in Map after a text zoom changed the legend panel's fixed width (_legend_rezoom: a resize-driven change); the window is never trimmed to them (no Figure.fit_window).  With `orient` the unit-cell panels' orientation is decided first (_choose_orientation)."""
        if orient:
            self._choose_orientation()
        if self._arrangement is not None:
            self._arrangement.rearrange()

    # --- the unit-cell panels' orientation (rule 11: turned when the window fits them better that way) ---

    ORIENT_HYSTERESIS = 0.05           # the panels keep their orientation unless the other scores better by this

    def _unit_cell_axes(self):
        """Machinery: the panels drawn in the unit cell's frame, which turn together: the placement topo, and in Map the Map Topo and every energy map.  The Line / Path heatmap row never turns."""
        axs = [getattr(self, '_crop_ax', None)]
        if self.mode == 'Map':
            axs.append(getattr(self, 'ax_map', None))
            axs.extend(getattr(self, 'map_axes', None) or [])
        figure_axes = self.fig.axes
        return [a for a in axs if a is not None and a in figure_axes]

    def _set_cell_rotation(self, flag):
        """Machinery: every unit-cell panel of the view turned 90 degrees (Axes.set_rotated) or back, together; each cursor box moves to the band its cursor's line now runs along."""
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
        Machinery: the unit-cell panels' orientation: as drawn, or turned 90 degrees, whichever the
        arrangement scores better at the window's size now (controls.Arrangement.best_score: the
        groups' scale less the blank), the current one kept within ORIENT_HYSTERESIS.  Decided
        when a view is built and again on Crop / Uncrop (the shape changes), never on a resize;
        before the arrangement has a size (the first build) the decision waits for its first
        arrangement (_on_arranged).  When the orientation changed, the Relief-shaded energy maps are drawn again for the
        new turn (_reshade_maps: a Relief shade's light keeps its place on screen; nothing else is redrawn).
        Returns whether the orientation changed.
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
        if best != cur:
            self._reshade_maps()
            self.fig.canvas.draw_idle()
        return best != cur

    def _on_arranged(self, name):
        """Machinery: after an arrangement: an orientation decision that waited for the arrangement's first size is made now (deferred past the arrangement's own pass; the arrangement then follows it)."""
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
        """Machinery: ((x lo, x hi), (y lo, y hi)) of the drawn cells: the unit cell at every offset of the Cells count."""
        n = int(self.s_cell.val) if getattr(self, 's_cell', None) is not None else 0
        a, b = self.lv[0, :2], self.lv[1, :2]
        corners = np.array([[0.0, 0.0], a, b, a + b])
        pts = np.concatenate([corners + np.asarray(off) for off in self._cell_offsets(n)])
        return (float(pts[:, 0].min()), float(pts[:, 0].max())), (float(pts[:, 1].min()), float(pts[:, 1].max()))

    def _axis_extent(self, axis):
        """Machinery: the cursor axis's full range as cropped: the applied crop, else the drawn cells' extent."""
        rng = self.crop_x if axis == 'x' else self.crop_y
        if rng is not None:
            return float(rng[0]), float(rng[1])
        ex, ey = self._full_extent()
        return ex if axis == 'x' else ey

    def _reset_cursors(self):
        """Machinery: every cursor at its axis's end (the full ranges as cropped)."""
        self._cursors = {'x': list(self._axis_extent('x')), 'y': list(self._axis_extent('y'))}

    def _cursor_value(self, axis, k, v):
        """Machinery: a value for cursor k (0 the lower, 1 the upper) made valid: inside the axis's extent, the pair keeping 2 % of the extent between them."""
        lo_ext, hi_ext = self._axis_extent(axis)
        v = float(np.clip(float(v), lo_ext, hi_ext))
        other = float(self._cursors[axis][1 - k])
        gap = 0.02 * max(hi_ext - lo_ext, 1e-9)
        v = min(v, other - gap) if k == 0 else max(v, other + gap)
        return float(np.clip(v, lo_ext, hi_ext))

    def _set_cursor(self, axis, k, v):
        """Machinery: cursor k of axis 'x' / 'y' moved to v (made valid); the lines, the shades and the boxes follow."""
        if self._cursors is None:
            self._reset_cursors()
        self._cursors[axis][k] = self._cursor_value(axis, k, v)
        self._draw_cursors()

    @staticmethod
    def _cursor_text(v):
        return f"{v:.2f}"

    def _on_cursor_box(self, axis, k, text):
        """Machinery: a cursor's box: the typed value moves the cursor there (made valid); a text that is not a number is reported and the box put back."""
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
        Machinery: the crop cursors on the placement topo: two lines at the x cursors and two at the y
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
        Machinery: the cursor boxes filled and moved onto their cursors (controls.RideBand): each in the band
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
        """Machinery: (x range, y range) the cursors mean for a crop: a pair left at its axis's ends keeps that axis's crop as it is."""
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
        """Machinery: the placement topo's limits at the applied crop (both axes; the full extent where there is none); in Map the Map Topo takes the same; the window's crop line follows (_crop_changed)."""
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
        self._crop_changed()

    def _on_crop(self, event=None):
        """Machinery: 'Crop': what lies between the cursors is kept, on both axes; the cursors then sit at the new edges; the unit-cell panels' orientation is decided again for the new shape, and the arrangement follows (the window keeps its size)."""
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
        """Machinery: 'Uncrop': the full extent of the drawn cells again, the cursors back at the ends."""
        if self.crop_x is None and self.crop_y is None:
            print("Uncrop: nothing is cropped.")
        self.crop_x = self.crop_y = None
        if self._crop_ax is not None:
            self._apply_crop()
            self._reset_cursors()
            self._draw_cursors()
            self._rearrange(orient=True)

    def _draw_map_slice(self, ax, slice_data, diverging, abs_m, v_min, v_max, gamma):
        """Machinery: one energy map on `ax` as the set's filled contours: the slice over the 2 x 2 cells drawn (one raster at four offsets) in the colour scale it is shown with (the branch: the signed spin density `diverging`, |m| `abs_m`, else the LDOS; v_min, v_max; LDOS γ `gamma`: _set_map_slice's signature); returns its contour set (_show_map_slice removes it when the map is drawn again)."""
        offs = self._cell_offsets_2x2()
        if diverging:
            cs = ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], cp.asnumpy(slice_data), levels=40, cmap=self._cmap('mag'), vmin=v_min, vmax=v_max, offsets=offs)
        elif abs_m:
            cs = ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], cp.asnumpy(slice_data), levels=40, cmap=self._cmap('|m|'), norm=plt.PowerNorm(gamma, vmin=v_min, vmax=v_max), offsets=offs)
        else:
            cs = ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], cp.asnumpy(slice_data), levels=40, cmap=self._cmap('map'), norm=plt.PowerNorm(gamma, vmin=v_min, vmax=v_max), offsets=offs)
        ax.set_aspect('equal')
        return cs

    def _draw_shaded_slice(self, ax, slice_data, diverging, abs_m, v_min, v_max, gamma, shade):
        """
        Machinery: one energy map on `ax` with the box's Shading on, as an RGBA mesh, nothing recomputed: one colour
        per unique grid point of the slice (cmap_shading.unique_points: the cell's duplicate edge left out), on the
        GPU: the branch's isoluminant shading map (_map_role_cmap: CET-I1, CET-D11 for the signed spin density,
        CET-I3 for |m|) through the map's own norm, as its filled contours take it (Normalize(v_min, v_max) for the
        signed spin density, else PowerNorm(LDOS γ `gamma`, v_min, v_max)), then the shade (`shade`, _map_shade) by
        the L* stretch, floor 0 (cmap_shading.shaded_rgba).  The colours, moved to the host, drawn by qtplot's
        pcolormesh with an (M, N, 4) C (shading 'flat'), each point's quad centred on it (cmap_shading.mesh_corners:
        steps a / (N - 1) and b / (N - 1)), one mesh at each offset of the 2 x 2 cells.  Returns the meshes
        (_show_map_slice removes them when the map is drawn again).
        """
        n = int(round(np.sqrt(len(self.grid_xy))))
        cmap = self._map_role_cmap(diverging, abs_m)
        norm = (plt.Normalize(vmin=v_min, vmax=v_max) if diverging
                else plt.PowerNorm(gamma, vmin=v_min, vmax=v_max))
        rgba = cp.asnumpy(cmap_shading.shaded_rgba(cmap_shading.unique_points(slice_data), shade, cmap, norm))
        X, Y = cmap_shading.mesh_corners(self.lv[0, :2] / (n - 1), self.lv[1, :2] / (n - 1), rgba.shape[:2],
                                         offsets=self._cell_offsets_2x2())
        meshes = [ax.pcolormesh(X[k], Y[k], rgba, shading='flat') for k in range(len(X))]
        ax.set_aspect('equal')
        return meshes

    def _map_shade(self):
        """
        Machinery: the energy maps' shade while the box's Shading is on (None while it is off), one for every map,
        since it depends on the heights only: the LDOS-topo heights Map's maps were computed at (map_z, Map's own,
        whose height range the contour slice's level is a fraction of) at the cell's unique grid points (cmap_shading.unique_points:
        row i along b, column j along a), CuPy float32 on the GPU, shaded by cmap_shading.shade_field in the box's
        mode: Height, the heights themselves; Relief, Kovesi's relief.m with its defaults, the steps a / (N - 1) and
        b / (N - 1), periodic=True (the taps wrap around the cell) and panel_turn 90 while the unit-cell panels are
        drawn turned (cell_rotated), else 0, so the light stays at the upper right as the map is shown.  The L*
        stretch then runs over this shade's range over the cell (cmap_shading.lstar_stretch).  Kept (_shade_cache)
        and made again only when the mode, the turn (Relief) or the heights change.
        """
        mode = self.shade_state.shade_mode
        if mode is None:
            return None
        turn = (90.0 if self.cell_rotated else 0.0) if mode == cmap_shading.RELIEF else None
        z = self.map_z
        kept = self._shade_cache
        if kept is not None and kept[0] == (mode, turn) and np.array_equal(kept[1], z):
            return kept[2]
        n = int(round(np.sqrt(len(z))))
        heights = cmap_shading.unique_points(cp.asarray(z, dtype=cp.float32))
        shade = cmap_shading.shade_field(heights, mode, self.lv[0, :2] / (n - 1), self.lv[1, :2] / (n - 1),
                                         panel_turn=0.0 if turn is None else turn, periodic=True)
        self._shade_cache = ((mode, turn), np.array(z, copy=True), shade)
        return shade

    def _map_cmap(self, use_diverging, has_mag):
        """Machinery: the energy maps' colour map for the branch they are drawn in (_map_role_cmap): the signed magnetization (use_diverging), |m| (Mag on and has_mag), else the LDOS."""
        return self._map_role_cmap(use_diverging, bool(self.show_mag and has_mag))

    def _map_role_cmap(self, diverging, abs_m):
        """Machinery: the energy maps' colour map for a branch (the signed spin density `diverging`, |m| `abs_m`, else the LDOS: cmap_shading.map_role, 'mag', '|m|' or 'map'): the window's colour-map set's map of that role (_cmap), or while the box's Shading is on its isoluminant shading map (cmap_shading.SHADING_CMAPS: 'CET-D11', 'CET-I3', 'CET-I1'), which the colour bars then show unshaded."""
        return self._cmap(cmap_shading.map_role(diverging, abs_m), shading=self.shade_state.shading)

    def _map_vmax(self, slice_data, group_vmax):
        """Machinery: an energy map's colour-scale maximum: its own (the largest |value| of the slice) with 'cmap norm' on, else its group's (the partition's with show_dcmp_norm, the whole set's otherwise); never 0."""
        v = float(cp.max(cp.abs(slice_data))) if self.cmap_norm else float(group_vmax)
        return v if v > 0 else 1e-15

    def _map_vmin(self, slice_data, group_vmin):
        """Machinery: an energy map's colour-scale minimum: the smallest value of the slice with 'cmap norm' on, else its group's (the partition's with show_dcmp_norm, the whole set's otherwise)."""
        return float(cp.min(slice_data)) if self.cmap_norm else float(group_vmin)

    def _draw_map_cbar(self, cax, v_min, v_max, use_diverging, has_mag, labelsize=None):
        """Machinery: one colour bar of the energy maps into `cax`: their colour map over v_min..v_max, the ticks in units of 10^exp with the exponent as the bar's title; returns the Colorbar (_set_map_cbar updates it in place)."""
        cax.clear()
        norm_choice = plt.Normalize(vmin=v_min, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap=self._map_cmap(use_diverging, has_mag), norm=norm_choice)
        sm.set_array([])
        cb = self.fig.colorbar(sm, cax=cax)
        exp = int(np.floor(np.log10(v_max)))
        cb.ax.yaxis.set_major_formatter(plt.ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
        cax.set_title(f"1e{exp}", fontsize=10 if labelsize is None else labelsize)
        if labelsize is not None:
            cax.tick_params(labelsize=labelsize)
        return cb

    def _on_cmap_norm(self, flag):
        """Machinery: 'cmap norm' (Map): each energy map on its own colour scale with its own colour bar (on), or every map on one scale, the maximum over all of them, with one bar (off); the map block is rebuilt with the bars."""
        self.cmap_norm = bool(flag)
        if self.mode == 'Map' and self.is_running:
            self._update_all(full_refresh=True)

    def _cmap(self, role, shading=False):
        """Machinery: the colour map of `role` ('topo', 'map', 'heatmap', 'mag' or '|m|') in the window's colour-map set (cmap_set, cmap_shading.SETS), or with shading=True the isoluminant map of a shaded energy map (cmap_shading.SHADING_CMAPS), each made once (_window_cmap)."""
        return _window_cmap(self.cmap_set, role, shading=shading)

    def _on_cmap_set(self, label):
        """
        Machinery: the colour-map set chosen ('cmap set': 'default 1' or 'default 2', cmap_shading.SETS), running or not:
        every panel drawn now takes, in place (set_cmap; nothing recomputed, made anew or arranged again), the
        new set's map of the role it is drawn in: the filled contours of the topo panels (the global topo, the
        Map Topo, the Line Topo) and the topo colour bar's ramp ('topo'), the Line / Path stripe ('topo'), each
        heatmap (its colour bar follows its mesh), each energy map drawn as filled contours (the role of the branch
        it was drawn in, from its _set_map_slice signature) and the energy maps' colour bars; then one draw.  A
        shaded energy map and a colour bar showing a shading map keep them (the shading maps are the same in both
        sets).  What is drawn later takes the set's maps where it is drawn (_cmap).
        """
        old = self.cmap_set
        if label == old:
            return
        self.cmap_set = label

        def same_role(c):
            # The new set's map of the role the kept map `c` (a heatmap's, a colour bar's) has in the old set.
            for role in cmap_shading.ROLES:
                m = _window_cmap(old, role)
                if m is c or m == c:
                    return self._cmap(role)
            return c

        topo = self._cmap('topo')
        figure_axes = self.fig.axes
        for ax in (self.ax_map, getattr(self, 'ax_map_global', None), self.ax_line_topo):
            if ax is not None and ax in figure_axes:
                for cs in ax.collections:
                    if isinstance(cs, plt.TriContourSet):
                        cs.set_cmap(topo)
        if self.cax_topo is not None:
            for im in self.cax_topo.images:
                im.set_cmap(topo)
        if self._stripe_lc is not None:
            self._stripe_lc.set_cmap(topo)
        for p_idx, mesh in enumerate(self._heat_meshes):
            if mesh is not None:
                self._heat_cmaps[p_idx] = same_role(self._heat_cmaps[p_idx])
                mesh.set_cmap(self._heat_cmaps[p_idx])
        for st in self._map_state.values():
            if st['cs'] is not None:
                diverging, abs_m = st['sig'][0], st['sig'][1]
                st['cs'].set_cmap(self._cmap('mag') if diverging else (self._cmap('|m|') if abs_m else self._cmap('map')))
        for st in self._map_cbar_state.values():
            st['cmap'] = same_role(st['cmap'])
            st['cb'].mappable.set_cmap(st['cmap'])
        self.fig.canvas.draw_idle()

    def _cell_offsets_2x2(self):
        """Machinery: the offsets of the 2 x 2 cells an energy map shows (the cell and its three neighbours along +a and +b)."""
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
        """Machinery: re-run Phase 1 global topography with current broadened DOS (cache-aware), at global_topo_bias (V_b, the snapped topo_bias)."""
        grid_res = int(np.sqrt(len(self.grid_xy)))
        cache_name = f"global_topo_{self.global_topo_bias}V_{self.topo_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}_fold{self._precision_cache_suffix()}.npy"
        if self.reuse_cache and exists(cache_name):
            print(f"[*] Loading Cached Global Topography: {cache_name}")
            self.current_z_map = np.load(cache_name)
        else:
            print(f"[*] Calculating Global Topography with broadening: {cache_name}")
            z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.topo_height, dtype=cp.float32)
            init_cur, t_samples = self._topo_current(cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]),
                                                     self.global_topo_bias, self._effective_decay_model())
            target_setp = cp.max(init_cur)
            print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
            z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_samples, self.global_topo_bias, target_setp,
                                                  decay_model=self._effective_decay_model())
            self.current_z_map = cp.asnumpy(z_map_gpu)
            np.save(cache_name, self.current_z_map)
        self.global_z_map = self.current_z_map.copy()

    def _on_topo_height_rerun(self, event=None):
        """Machinery: apply the global-topo tip-height box (the 'Topo h' Value box) and recompute the global topo.
        Static-label, single-action button. Does nothing unless the main RUN
        button is active. Height-encoded cache filenames are loaded when
        reuse_cache is True and recomputed and rewritten when reuse_cache is
        False. The global topo takes the place of the tip heights Map's maps
        are computed at (current_z_map, the LDOS topo), so Map's topo records
        are reset: in Map (now, or on return) the LDOS topo is loaded or
        computed again and the maps computed again at the kept energies."""
        if not self.is_running:
            return
        self.topo_height = float(self.s_topo_h.val)
        self._recompute_global_topo()
        self._broadening_topo_dirty = False
        # current_z_map now holds the global topo: Map's LDOS topo, and its maps at the kept energies, again.
        self.cached_bias_energy_map = self.cached_d_topo_map = None
        self._update_all(full_refresh=True)

    def _on_ldos_height_rerun(self, event=None):
        """Machinery: apply the LDOS-topo tip-height box (the 'LDOS h' Value box) and recompute LDOS topo.
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
        self.cached_bias_energy_path = self.cached_d_topo_path = None
        self.cached_emin = self.cached_emax = None
        self.cached_ld_up = self.cached_ld_dn = None
        self.cached_ld_up_gpu = self.cached_ld_dn_gpu = None
        self.cached_marker_coords = self.cached_spec_ldos = None
        self.cached_spec_ldos_gpu = None
        self._update_all(full_refresh=True)

    def _on_broadening_change(self, *args):
        """Machinery: a broadening check or a σ box changed (the controls report; the window owns the state)."""
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
        self.cached_bias_energy_path = self.cached_d_topo_path = None
        self.cached_marker_coords = self.cached_spec_ldos = None
        self.cached_spec_ldos_gpu = None
        self.cached_ld_up = self.cached_ld_dn = None
        self.cached_ld_up_gpu = self.cached_ld_dn_gpu = None
        self._broadening_topo_dirty = True
        if self.is_running:
            self._recompute_global_topo()
            self._broadening_topo_dirty = False
            self._update_all(full_refresh=True)

    def _on_toggle(self, label, states):
        """
        Machinery: a row of the toggles clicked (controls.Checks: the clicked row already flipped).  'exact' and
        'linearized' (the Decay-on model): the clicked one on and the other off, so a click on the one already on
        leaves it on and exactly one is always on (the rows set by Checks.set, which never fires).  Then the state
        is read from the controls as for every control (_on_ui_change).
        """
        if label in ('exact', 'linearized'):
            labels = [lab for lab, _ in self._toggle_defs]
            self.chk.set(labels.index('exact'), label == 'exact')
            self.chk.set(labels.index('linearized'), label == 'linearized')
        self._on_ui_change(label, states)

    def _on_ui_change(self, *args):
        """Machinery: a toggle, a count or a value box changed: the state is read from the controls, the caches invalidated as the change asks (a decay-model change: _effective_decay_model before and after), and the window redrawn."""
        prev_model = self._effective_decay_model()
        prev_path_extend = self.path_extend
        states = self.chk.get_status()
        for (label, attr), state in zip(self._toggle_defs, states):
            if attr is not None:               # 'exact' drives nothing itself ('linearized' drives decay_linearized)
                setattr(self, attr, state)
        self.display_cells = int(self.s_cell.val)
        self.ldos_gamma = float(self.s_ldos_gamma.value)

        # A decay-model change (Decay, or 'exact' / 'linearized' while Decay is on: with Decay off they change no
        # model) or the path-extend toggle: invalidate caches and recompute.
        decay_model_changed = (self._effective_decay_model() != prev_model)
        if decay_model_changed or (self.path_extend != prev_path_extend):
            # Path's tip heights are dropped by either (the Extend is in Path's key), Line's by a decay-model
            # change only (the Extend leaves Line's points as they are).
            if decay_model_changed:
                self.cached_p1 = self.cached_p2 = None
                self.cached_d_topo_line = None
                self.cached_bias_energy_line = None
            self.cached_path_nodes = None
            self.cached_d_topo_path = None
            self.cached_bias_energy_path = None
            self.cached_emin = self.cached_emax = None
            self.cached_d_ldos = None
            self.cached_ld_up = self.cached_ld_dn = None
            self.cached_ld_up_gpu = self.cached_ld_dn_gpu = None
            # Only a decay-model change alters the global topo (recompute, or
            # load the model-tagged cache); path-extend does not.  Nor does
            # path-extend alter Map's topo and maps or the markers' spectra:
            # only a decay-model change has them computed again (Map's maps at
            # its kept energies).
            if decay_model_changed:
                self.cached_d_topo_map = None
                self.cached_bias_energy_map = None
                self.cached_marker_coords = self.cached_spec_ldos = None
                self.cached_spec_ldos_gpu = None
                self._broadening_topo_dirty = True
                if self.is_running:
                    self._recompute_global_topo()
                    self._broadening_topo_dirty = False
            if self.is_running:
                self._update_all(full_refresh=True)
            self._markers_changed()                # the Path's length (Extend) in the info panel's line and markers lines, running or not
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
                    self._form_marker_set()        # every marker new: top to bottom, then coloured by index
                else:
                    # The first new_count kept (the markers lowest on the stripe go), with their colours.
                    self.marker_ratios = self.marker_ratios[:new_count]
                    self.line_marker_colors = self.line_marker_colors[:new_count]
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

    def _update_all(self, full_refresh=False, last_run=False):
        """
        Machinery: the window redrawn for its state: the maps (full_refresh), the markers, then, while
        running, the computations the state asks for (needs_topo / needs_ldos / needs_spec, the
        same functions as before) and the panels of the view.  With RUN off and `last_run` (a decomp step:
        _decomp_click; a view change into Map: _build_ui) the panels of the view (the spectrum, the legend panel, the heatmaps, the maps) are
        drawn from the data of the last run, nothing computed, when the view has them (_has_last_run);
        otherwise, with RUN off, nothing more is drawn.  Map's energies, data and caches are its own (map_*,
        map_e_idx): kept through every toggle and view change and shown from what is kept; its energies are
        formed again from the typed ends only after an edit of E Min, E Max or E Pts (map_typed), and its
        maps are computed again, at the kept energies, only when the LDOS they show changes.  The Line / Path
        cluster and the Map rows are built again as cells of their own PanelRows in the rest pane (automatic layout: the
        one rule for every window; rule 11: the rows size themselves from the panels) only when
        their panel count changes (_build_heat_row, _build_map_block), after which the panels are
        arranged again (_rearrange; the window is never trimmed); otherwise every panel takes the
        new data in place (_draw_spectrum, _fill_heat_row, _fill_map), with one draw_idle at the
        end; the periodic images of a map are one
        raster drawn at every cell offset (tricontourf offsets=: moved, never re-imaged), or with the
        box's Shading on one RGBA mesh at each cell offset (_draw_shaded_slice).
        """
        if full_refresh:
            self.ax_map.clear()
            n = int(self.s_cell.val)
            if self.mode == 'Map':
                self.ax_map_global.clear()
            t_ax = self.ax_map_global if self.mode == 'Map' else self.ax_map
            z_data = self.global_z_map

            offs = self._cell_offsets(n)
            t_ax.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], z_data, levels=60, cmap=self._cmap('topo'), zorder=1, offsets=offs)
            if self.mode == 'Map':
                # The Map Topo from the LDOS-topo heights Map's maps were computed at (map_z, Map's own); before
                # Map's first maps (none kept), from current_z_map, as before.
                self.ax_map.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1],
                                        self.current_z_map if self.map_z is None else self.map_z,
                                        levels=60, cmap=self._cmap('topo'), zorder=1, offsets=offs)
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
            t_ax.set_title(f"Global Topo: bias {self.global_topo_bias:.3f} V, height {self.topo_height} Å")
            if self.mode == 'Map':
                self.ax_map.set_aspect('equal')
                # Titled with the bias and LDOS h those heights were computed at (map_bias, map_ldos_h, kept beside
                # map_z); before Map's first maps, the window's, as before.
                m_bias, m_h = ((self._bias_sample()[1], self.ldos_height) if self.map_z is None
                               else (self.map_bias, self.map_ldos_h))
                self.ax_map.set_title(f"Map Topo: bias {m_bias:.3f} V, height {m_h} Å")
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
            self._line_geom = (p_xy, p_dist, p_len)        # the line's points as drawn, kept for the drags (_on_motion, _place_line_markers)
            self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])
            self.ends.set_offsets([self.p1, self.p2])
            self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios]))
            self.marks.set_facecolors(self.line_marker_colors)
        elif self.mode == 'Path':
            p_xy, p_dist, p_len, verts = self._build_path_xy()
            self._line_geom = (p_xy, p_dist, p_len)        # the path's points as drawn, kept for the drags (_on_motion, _place_line_markers)
            self.line_art.set_data(verts[:, 0], verts[:, 1])
            self.path_pts.set_offsets(np.array(self.path_nodes))
            mark_idx = (np.array(self.marker_ratios) * (self.npts - 1)).astype(int)
            self.marks.set_offsets(p_xy[mark_idx])
            self.marks.set_facecolors(self.line_marker_colors)
        else:
            self.marks.set_offsets(self.marker_coords)
            self.marks.set_facecolors(self.m_colors[:len(self.marker_coords)])
        self._markers_changed()                    # the markers' and the line's lines in the info panel

        # RUN off: nothing more is drawn, except for a decomp step or a view change into Map (last_run), whose panels are
        # drawn below from the last run's data when the view has them, with nothing computed (compute False).
        if not self.is_running and not (last_run and self._has_last_run()):
            self.fig.canvas.draw_idle()
            return
        compute = self.is_running

        if compute and getattr(self, '_broadening_topo_dirty', False):
            self._recompute_global_topo()
            self._broadening_topo_dirty = False
            self._update_all(full_refresh=True)
            return

        # The bias end's snapped sample and its energy V_b (_bias_sample), which replaces the typed bias.
        i_b, v_b = self._bias_sample()
        nepts = int(self.s_nepts.val) if self.mode == 'Map' else None
        decay_model = self._effective_decay_model()
        needs_topo = compute and (
            (self.mode == 'Line' and (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy_line != v_b or self.cached_d_topo_line != decay_model)) or
            (self.mode == 'Path' and (self.cached_path_nodes is None or not np.array_equal(np.array(self.path_nodes), self.cached_path_nodes) or self.cached_path_extend != self.path_extend or self.cached_bias_energy_path != v_b or self.cached_d_topo_path != decay_model)) or
            (self.mode == 'Map' and (self.cached_bias_energy_map != v_b or self.cached_d_topo_map != decay_model)))
        # The window's ends compared as snapped (_snap: a sample index, or ZERO_V): a typed change that lands on the
        # same samples recomputes nothing.
        e_lo, e_hi = _snap(self.energies, self.s_emin.val, 'emin'), _snap(self.energies, self.s_emax.val, 'emax')
        if self.mode == 'Map':
            # Map's own records (map_*), apart from the other views': its energies are formed again from the typed ends
            # only after an edit of E Min, E Max or E Pts (reform: the typed ends and count differ from map_typed,
            # those they were last formed from; a dragged energy is kept otherwise); its maps are computed again
            # when the LDOS they show changes: the tip heights (needs_topo: the bias, the decay model, LDOS h, a
            # broadening, a Topo h rerun), the window's snapped ends, the decay model, or the snapped Map samples once formed
            # again (compared with those the kept maps were taken at, map_e_idx); never for a view change or a
            # display-only toggle.  Map's own marker spectra (map_spec_*, apart from Single Point's) are computed
            # again with the maps (over a new window among them) or for other markers (map_marker_coords).
            typed = (float(self.s_emin.val), float(self.s_emax.val), nepts)
            reform = compute and (self.map_e_idx is None or typed != self.map_typed)
            needs_ldos = compute and (
                needs_topo or self.map_ld_up_gpu is None or self.map_window != (e_lo, e_hi) or
                self.map_d_ldos != decay_model or
                (reform and not np.array_equal(
                    self._map_sample_indices(np.linspace(self.s_emin.val, self.s_emax.val, nepts)), self.map_e_idx)))
            needs_spec = compute and (
                needs_ldos or self.map_marker_coords is None or
                not np.array_equal(self.marker_coords, self.map_marker_coords))
        else:
            reform = False
            needs_ldos = compute and (needs_topo or self.cached_emin != e_lo or self.cached_emax != e_hi or
                                      self.cached_d_ldos != decay_model or getattr(self, 'cached_mode', None) != self.mode)
            needs_spec = compute and (needs_ldos or (self.mode == 'Single Point' and (self.cached_marker_coords is None or not np.array_equal(self.marker_coords, self.cached_marker_coords))))

        if needs_topo:
            if self.s_emin.val == 0.0 and self.s_emax.val == 0.0:
                raise ValueError("Stabilization bias is 0: set a nonzero E Min or E Max before the constant-current window can be built.")
            if self.mode in ('Line', 'Path'):
                p_xy_gpu = cp.array(p_xy, dtype=cp.float32)
                l_cur, l_samples = self._topo_current(
                    cp.hstack([p_xy_gpu, cp.full((self.npts, 1), self.z_highest_atom + self.ldos_height, dtype=cp.float32)]),
                    v_b, self.use_decay)
                target = cp.max(l_cur)
                print(f"[*] Path Setpoint LDOS: {float(target):.6e}")
                z_line = self._converge_tip_height(cp.full(self.npts, self.z_highest_atom + self.ldos_height, dtype=cp.float32), p_xy_gpu, l_samples, v_b, target,
                                                   use_decay=self.use_decay)
                # Into the view's own record, with its key and the bias and decay model they were converged at.
                if self.mode == 'Line':
                    self.current_z_line = cp.asnumpy(z_line)
                    self.cached_p1, self.cached_p2 = self.p1.copy(), self.p2.copy()
                    self.cached_bias_energy_line, self.cached_d_topo_line = v_b, decay_model
                else:
                    self.current_z_path = cp.asnumpy(z_line)
                    self.cached_path_nodes = np.array(self.path_nodes).copy()
                    self.cached_path_extend = self.path_extend
                    self.cached_bias_energy_path, self.cached_d_topo_path = v_b, decay_model
            elif self.mode == 'Map':
                grid_res = int(np.sqrt(len(self.grid_xy)))
                ldos_cache_name = f"ldos_topo_{v_b}V_{self.ldos_height}A_{grid_res}px{self._broadening_cache_suffix()}{self._unit_cell_cache_suffix()}{self._decay_cache_suffix()}_fold{self._precision_cache_suffix()}.npy"
                if self.reuse_cache and exists(ldos_cache_name):
                    print(f"[*] Loading Cached LDOS Topography: {ldos_cache_name}")
                    self.current_z_map = np.load(ldos_cache_name)
                else:
                    print(f"[*] Cache not found. Calculating LDOS Topography: {ldos_cache_name}")
                    z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.ldos_height, dtype=cp.float32)
                    init_cur, t_samples = self._topo_current(cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]),
                                                             v_b, self.use_decay)
                    target_setp = cp.max(init_cur)
                    print(f"[*] Map Local Setpoint LDOS: {float(target_setp):.6e}")
                    z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_samples, v_b, target_setp,
                                                          use_decay=self.use_decay)
                    self.current_z_map = cp.asnumpy(z_map_gpu)
                    np.save(ldos_cache_name, self.current_z_map)
                self.cached_bias_energy_map, self.cached_d_topo_map = v_b, decay_model
                # The Map Topo is drawn again from it once Map's maps are computed at it (map_z, below).

        if reform:
            # An edit of E Min, E Max or E Pts: the Map energies formed again from the typed ends, np.linspace of
            # them, each snapped to its sample (_map_sample_indices); map_e_targets holds the snapped samples'
            # energies, map_typed the typed ends and count they were formed from.
            self.map_e_idx = self._map_sample_indices(np.linspace(self.s_emin.val, self.s_emax.val, nepts))
            self.map_e_targets = self.energies[self.map_e_idx]
            self.map_typed, self.cached_nepts = typed, nepts

        if needs_ldos and self.mode == 'Map':
            # Every map energy (the kept ones, or those just formed) in one call: the LDOS at each grid point, at
            # each map energy's sample, and the current at those samples only (one column per map), into Map's
            # own caches, with the LDOS-topo heights (map_z, a copy), the bias and LDOS h they were computed at
            # (map_bias, map_ldos_h) and the records they were taken with (the window, the decay model and the
            # broadening: map_window, map_d_ldos, map_broadening; the spectrum's energies are kept with Map's marker
            # spectra, below).
            grid_z = cp.hstack([self.grid_xy_gpu, cp.asarray(self.current_z_map, dtype=cp.float32)[:, None]])
            ld_up, ld_dn, _, cur = self._calculate_ldos_at_points_gpu(
                grid_z, None, None, use_energy_decay=self.use_decay, preserve_orbitals=True, with_current=True,
                samples=self.map_e_idx, progress='Map energy maps')
            self.map_ld_up = cp.asnumpy(ld_up)
            self.map_ld_dn = cp.asnumpy(ld_dn) if ld_dn is not None else None
            self.map_ld_up_gpu, self.map_ld_dn_gpu = ld_up, ld_dn
            self.map_current = cur
            self.map_current_E = cp.asarray((self.map_e_idx - self.k0).astype(np.float32)) * self.e_step32 + self.e032
            self.map_z = np.array(self.current_z_map, copy=True)
            self.map_bias, self.map_ldos_h = v_b, self.ldos_height
            self.map_window, self.map_d_ldos = (e_lo, e_hi), decay_model
            self.map_broadening = dict(self.extra_broadening)
            if needs_topo:
                # The LDOS topo just computed (needs_topo, above): the Map Topo drawn again on the cleared panel
                # from the heights Map's maps were computed at (map_z), with what the full refresh drew on it: the
                # atoms of one type over every image as one scatter, the unit cell and the title (their bias and
                # LDOS h, map_bias and map_ldos_h).
                self.ax_map.clear()
                n = int(self.s_cell.val)
                offs = self._cell_offsets(n)
                self.ax_map.tricontourf(self.grid_xy[:, 0], self.grid_xy[:, 1], self.map_z, levels=60, cmap=self._cmap('topo'), zorder=1, offsets=offs)
                if self.show_atoms:
                    tr = np.repeat(self.atomtypes, self.atomnums)
                    for t_idx, t_name in enumerate(self.atomtypes):
                        m = (tr == t_name)
                        xs = np.concatenate([self.coord[m, 0] + off[0] for off in offs])
                        ys = np.concatenate([self.coord[m, 1] + off[1] for off in offs])
                        self.ax_map.scatter(xs, ys, s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2)
                if self.show_unit_cell:
                    v0, v1, v2, v3 = np.array([0, 0]), self.lv[0, :2], self.lv[0, :2] + self.lv[1, :2], self.lv[1, :2]
                    cell_pts = np.array([v0, v1, v2, v3, v0])
                    self.ax_map.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')
                self.ax_map.set_aspect('equal')
                self.ax_map.set_title(f"Map Topo: bias {self.map_bias:.3f} V, height {self.map_ldos_h} Å")
                self.ax_map.add_collection(self.marks)
                self._apply_crop()                 # the Map Topo takes the placement topo's crop
        elif needs_ldos:
            if self.mode in ('Line', 'Path'):
                l_idx = self._display_indices(self.s_emin.val, self.s_emax.val)
                # At the view's own tip heights: Line's (current_z_line) or Path's (current_z_path).
                z_view = self.current_z_line if self.mode == 'Line' else self.current_z_path
                ld_up, ld_dn, eg, cur = self._calculate_ldos_at_points_gpu(
                    np.hstack([p_xy, z_view[:, None]]),
                    None, None,
                    use_energy_decay=self.use_decay, preserve_orbitals=True, with_current=True,
                    samples=l_idx, progress=f"{self.mode} points")
                # The current from the same pass, stored with its samples' GPU energies e0 + (i - k0) step.
                self.cached_current = cur
                self.cached_current_E = cp.asarray((l_idx - self.k0).astype(np.float32)) * self.e_step32 + self.e032
            else:
                ld_up, ld_dn = None, None

            if ld_up is not None:
                # The line's LDOS and its energies, kept together (Single Point's spectra keep theirs: cached_spec_eg),
                # and the view that computed them (cached_ld_mode: Line's data are never sent with Path's markers).
                self.cached_ld_up = cp.asnumpy(ld_up)
                self.cached_ld_dn = cp.asnumpy(ld_dn) if ld_dn is not None else None
                self.cached_ld_up_gpu, self.cached_ld_dn_gpu = ld_up, ld_dn
                self.cached_eg = cp.asnumpy(eg)
                self.cached_ld_mode = self.mode
            self.cached_emin, self.cached_emax = e_lo, e_hi
            self.cached_d_ldos = decay_model
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
            spec_idx = self._display_indices(self.s_emin.val, self.s_emax.val)
            s_up, s_dn, _, s_cur = self._calculate_ldos_at_points_gpu(
                pt_gpu, None, None,
                use_energy_decay=self.use_decay, preserve_orbitals=True, with_current=True,
                samples=spec_idx, progress='marker spectra')

            spec_ldos = self._combine_ldos_display(s_up, s_dn, self.show_mag)
            if self.mode == 'Map':
                # Map's own marker spectra (map_spec_*), apart from Single Point's, with their energies and the
                # markers they were computed for.
                self.map_spec_ldos_gpu = spec_ldos
                self.map_spec_ldos = cp.asnumpy(spec_ldos)
                self.map_marker_coords = np.array(self.marker_coords).copy()
                self.map_eg = self.energies[spec_idx]          # their energies, kept with them
                # The markers' current from the same pass, stored with its samples' GPU energies e0 + (i - k0) step.
                self.map_spec_current = s_cur
                self.map_spec_current_E = cp.asarray((spec_idx - self.k0).astype(np.float32)) * self.e_step32 + self.e032
                # The decay model and the broadening they were computed with: kept through a Decay or broadening
                # change, they are not sent for the decomposition while these differ from the window's (sim_wrapper).
                self.map_spec_d_ldos, self.map_spec_broadening = decay_model, dict(self.extra_broadening)
            else:
                self.cached_spec_ldos_gpu = spec_ldos
                self.cached_spec_ldos = cp.asnumpy(spec_ldos)
                self.cached_marker_coords = np.array(self.marker_coords).copy()
                self.cached_spec_window = (e_lo, e_hi)
                self.cached_spec_eg = self.energies[spec_idx]      # their energies, kept with them
                # The markers' current from the same pass, stored with its samples' GPU energies e0 + (i - k0) step.
                self.cached_spec_current = s_cur
                self.cached_spec_current_E = cp.asarray((spec_idx - self.k0).astype(np.float32)) * self.e_step32 + self.e032

        # The view's own LDOS as displayed: Line / Path the line's (cached_ld_*), Map its own (map_ld_*).
        if self.mode in ('Line', 'Path'):
            ld_up_kept, ld_dn_kept, iv_which = self.cached_ld_up_gpu, self.cached_ld_dn_gpu, 'grid'
        elif self.mode == 'Map':
            ld_up_kept, ld_dn_kept, iv_which = self.map_ld_up_gpu, self.map_ld_dn_gpu, 'map'
        else:
            ld_up_kept, ld_dn_kept, iv_which = None, None, None
        if ld_up_kept is not None:
            f_up = ld_up_kept.copy()
            f_dn = ld_dn_kept.copy() if ld_dn_kept is not None else None
            f_ldos_raw = self._combine_ldos_display(f_up, f_dn, self.show_mag)
            if self.feenstra:
                # 'Feenstra': every partition of a point divided by that point's total I/V at each energy
                # (Map: at the sample each map's LDOS is taken at); 'Norm', when on, then acts on the ratio.
                f_ldos_raw = feenstra.divide(f_ldos_raw, self._feenstra_iv(iv_which))
            partitions = self.decomp.partitions(f_ldos_raw, self.atomtypes, self.atomnums, self.orbitals, self.dos_up_gpu)
        else:
            f_ldos_raw = None
            partitions = []

        # The marker spectra as shown (Single Point, Map: each view's own, Map's map_spec_*): divided by their I/V
        # under 'Feenstra', else as cached.
        spec_disp = self.map_spec_ldos_gpu if self.mode == 'Map' else self.cached_spec_ldos_gpu
        if self.feenstra and spec_disp is not None and self.mode in ('Single Point', 'Map'):
            spec_disp = feenstra.divide(spec_disp, self._feenstra_iv('map_spec' if self.mode == 'Map' else 'spec'))

        # --- Determine whether mag data is signed (for colormap selection) ---
        mag_signed = self._mag_is_signed()
        has_mag = self._has_mag_channel()
        use_diverging = self.show_mag and has_mag and mag_signed

        if self.mode in ('Line', 'Path'):
            # The drag cache: the displayed LDOS of every line / path point as drawn now (the display
            # path above: _combine_ldos_display and the active display options); a marker drag sets the
            # marker spectra from it and computes nothing while the button is down (_move_line_markers).
            # NumPy: copied to the host once here, so a drag does no GPU work.
            self._line_disp = cp.asnumpy(f_ldos_raw) if f_ldos_raw is not None else None
            # Beside it, the Norm factors: each point's integral over the energies of its total displayed LDOS
            # (what 'Norm' divides a spectrum by), computed here once on the GPU (quadrature.simpson) and kept
            # as NumPy in the float32 they were computed in; under 'Norm' a drag divides the marker spectra by
            # them (_draw_spectrum, line_norm).
            self._line_norm = (cp.asnumpy(quadrature.simpson(cp.sum(f_ldos_raw, axis=(2, 3)), dx=self.e_step32, axis=1))
                               if f_ldos_raw is not None else None)

        self._draw_spectrum(f_ldos_raw, spec_disp, p_dist if self.mode in ('Line', 'Path') else None)

        if self.mode in ('Line', 'Path'):
            processed_partitions = []
            global_vmax = 0.0
            global_vmin = np.inf

            if self.normalize and f_ldos_raw is not None:
                total_ldos_line = cp.sum(f_ldos_raw, axis=(2, 3))
                norm_denom_line = quadrature.simpson(total_ldos_line, dx=self.e_step32, axis=1)[:, None] + 1e-15
            else:
                norm_denom_line = 1.0

            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    t_data /= norm_denom_line
                processed_partitions.append((p_label, t_data))
                v_max = float(cp.max(cp.abs(t_data)))
                if v_max > global_vmax:
                    global_vmax = v_max
                global_vmin = min(global_vmin, float(cp.min(t_data)))
            if global_vmax == 0:
                global_vmax = 1e-15

            # The row is built only when its panel count changes (the partitions of the level, the
            # per-heatmap colour bars); otherwise its panels take the new data in place.
            built = (len(processed_partitions), self.show_dcmp_norm) != self._row_key
            if built:
                self._build_heat_row(len(processed_partitions))
            self._fill_heat_row(processed_partitions, global_vmax, global_vmin, p_dist, p_len, use_diverging, has_mag)
            quantity = "(dI/dV)/(I/V)" if self.feenstra else "LDOS"
            self._cluster_label.set_text(f"{quantity}: {self.active_element}" if self.plot_level >= 3 else quantity)
            if built:
                self._rearrange()                  # the row is new: the arrangement follows (rule 11)

        elif self.mode == 'Map':
            num_p = max(1, len(partitions))
            # The maps at Map's kept energies (map_e_idx, map_e_targets: never formed again here), as many as
            # they are (the E Pts box may hold an edit not yet computed, with RUN off).
            nepts = len(self.map_e_idx)

            # The block is built only when its panel count changes (the partitions of the level, E Pts,
            # the colour bars of 'cmap norm'); otherwise each map takes its new data in place (_fill_map).
            built = (num_p, nepts, self.cmap_norm, self.show_dcmp_norm) != self._block_key
            if built:
                self._build_map_block(num_p, nepts)

            m_coords_np = np.array(self.marker_coords)

            processed_partitions = []
            global_vmax = 0.0
            global_vmin = np.inf
            if self.normalize and f_ldos_raw is not None:
                total_ldos_map = cp.sum(f_ldos_raw, axis=(2, 3))
                # Each distinct snapped Map sample once, ascending, at its GPU energy e0 + (i - k0) step.
                u_e, s_idx = np.unique(self.map_e_idx, return_index=True)
                x_gpu = cp.asarray((u_e - self.k0).astype(np.float32)) * self.e_step32 + self.e032
                norm_denom_map = quadrature.simpson(total_ldos_map[:, s_idx], x=x_gpu, axis=1)[:, None] + 1e-15
            else:
                norm_denom_map = 1.0

            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    t_data /= norm_denom_map
                t_data = cp.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
                processed_partitions.append((p_label, t_data))
                v_max = float(cp.max(cp.abs(t_data)))
                if v_max > global_vmax:
                    global_vmax = v_max
                global_vmin = min(global_vmin, float(cp.min(t_data)))
            if global_vmax == 0:
                global_vmax = 1e-15

            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                p_vmax = float(cp.max(cp.abs(t_data))) if self.show_dcmp_norm else global_vmax
                p_vmin = float(cp.min(t_data)) if self.show_dcmp_norm else global_vmin
                for i, target_e in enumerate(self.map_e_targets):
                    k = p_idx * nepts + i
                    slice_data = t_data[:, i]
                    v_max = self._map_vmax(slice_data, p_vmax)
                    v_min = self._map_vmin(slice_data, p_vmin)
                    self._fill_map(k, p_idx, i, p_label, slice_data, v_min, v_max, target_e, use_diverging, has_mag,
                                   m_coords_np)

                if self.show_dcmp_norm and not self.cmap_norm:
                    self._set_map_cbar(p_idx, p_vmin, p_vmax if p_vmax > 0 else 1e-15, use_diverging, has_mag)

            if not self.show_dcmp_norm and not self.cmap_norm and len(processed_partitions) > 0:
                self._set_map_cbar(0, global_vmin, global_vmax, use_diverging, has_mag)
            if self._map_marks_xy is None or not np.array_equal(self._map_marks_xy, m_coords_np):
                self._place_map_markers()
            # The contour slice on the maps: drawn again when the maps are new or the heights Map's maps were
            # computed at changed (map_z; nothing drawn while the box's Contour is off: _draw_contour_artists).
            if built or self._contour_z is None or not np.array_equal(self._contour_z, self.map_z):
                self._draw_contour_artists()
                self._contour_z = np.array(self.map_z, copy=True)
            self._map_energies_changed()           # the info panel's map energies line: the energies of the maps drawn

            quantity = "(dI/dV)/(I/V)" if self.feenstra else "LDOS"
            self._cluster_label.set_text(f"{quantity}: {self.active_element}" if self.plot_level >= 3 else quantity)
            if built:
                self._rearrange()                  # the block is new: the arrangement follows (rule 11)

        self.fig.canvas.draw_idle()

    def _draw_spectrum(self, f_ldos_raw, spec_disp, p_dist, line_norm=None):
        """
        Machinery: the spectrum panel (ax_spec) for the state.  Its curves are drawn anew (the panel cleared,
        the curves plotted, the hover labels made, as before) when the set of curves changes
        (_spec_key: the view, the level, the curves' labels (the marker count at level 0, the element
        and its shown orbitals at level 3), 'Feenstra', the Map energy markers' count); otherwise the
        same curves take their new data in place (Line2D.set_data, the energy markers
        _RefLine.set_xdata), with the limits and the title.  The curves are drawn over the energies of the
        data they come from: in Map Map's own marker spectra's (map_eg), in Single Point its marker spectra's
        (cached_spec_eg), in Line / Path the line's (cached_eg).  f_ldos_raw: the displayed LDOS of
        the Line / Path points (in a drag the cache, _line_disp); spec_disp: the displayed marker
        spectra (Single Point, Map); p_dist: the Line / Path distances; the Map energy markers at
        Map's kept energies (map_e_targets, never formed again here); line_norm: in a Line / Path drag or a
        mark_dynamic pick, the Norm factors kept beside the drag cache (_line_norm), by which, under
        'Norm', each spectrum is divided here (its point's factor + 1e-15), so the drag does no GPU
        work; None otherwise (under 'Norm' the curves are then divided by their integrals,
        quadrature.simpson, in ldos_decomp.Decomp.curves).  The caller draws the canvas.
        """
        if self.mode in ('Line', 'Path'):
            active_idx = int(self.marker_ratios[min(self.active_marker_idx, len(self.marker_ratios) - 1)] * (self.npts - 1))
            active_ldos = f_ldos_raw[active_idx] if f_ldos_raw is not None else np.zeros_like(self.cached_spec_ldos[0])
            # Every marker's spectrum in index order (level 0), at the marker's point on the line.
            marker_spectra = ([f_ldos_raw[int(r * (self.npts - 1))] for r in self.marker_ratios]
                              if f_ldos_raw is not None else None)
            if line_norm is not None and self.normalize and f_ldos_raw is not None:
                # 'Norm' in a drag: each spectrum divided by its point's kept factor, on the host (new arrays:
                # the drag cache stays as it is).
                active_ldos = active_ldos / (line_norm[active_idx] + 1e-15)
                marker_spectra = [s / (line_norm[int(r * (self.npts - 1))] + 1e-15)
                                  for s, r in zip(marker_spectra, self.marker_ratios)]
        else:
            active_idx = min(self.active_marker_idx, len(self.marker_coords) - 1)
            active_ldos = spec_disp[active_idx]
            marker_spectra = [spec_disp[i] for i in range(len(self.marker_coords))]

        n_marks = len(self.marker_ratios) if self.mode in ('Line', 'Path') else len(self.marker_coords)
        # The spectrum's energies, those of the data its curves come from: Map's own marker spectra's (map_eg) in
        # Map, Single Point's (cached_spec_eg) in Single Point, the line's (cached_eg) in Line / Path.
        eg = (self.map_eg if self.mode == 'Map' else
              self.cached_spec_eg if self.mode == 'Single Point' else self.cached_eg)
        # The level's curves (ldos_decomp: data, style, label, pickable), the markers' colours in index order.
        marker_colors = (self.line_marker_colors if self.mode in ('Line', 'Path')
                         else [self.m_colors[i % len(self.m_colors)] for i in range(n_marks)])
        integral =(lambda y: quadrature.simpson(y, dx=self.e_step32)) if self.normalize and line_norm is None else None
        curves = self.decomp.curves(active_ldos, marker_spectra, marker_colors, self.atomtypes, self.atomnums,
                                    self.orbitals, self.dos_up_gpu, self._type_color_map, eg, integral)
        key = (self.mode, self.plot_level, tuple(c.label for c in curves), bool(self.feenstra),
               len(self.map_e_targets) if self.mode == 'Map' else None)
        build = key != self._spec_key
        if build:
            self.ax_spec.clear()
            self._spec_lines, self._spec_emarkers, self._spec_title = [], [], None
            for c in curves:
                kw = dict(color=c.color, lw=c.linewidth, linestyle=c.linestyle, alpha=c.alpha, zorder=c.zorder,
                          label=c.label, picker=c.picker)
                if c.pickradius is not None:
                    kw['pickradius'] = c.pickradius
                ln, = self.ax_spec.plot(eg, cp.asnumpy(c.y), **kw)
                self._spec_lines.append(ln)
            # The old hover labels removed; new ones on the orbital curves (level 3), on from the start.
            self.cursor = ldos_decomp.hover_labels(self.ax_spec, self._spec_lines, curves, getattr(self, 'cursor', None))
        else:
            # In place: the curves in the order they were drawn take their data, and their colour where it differs.
            for ln, c in zip(self._spec_lines, curves):
                ln.set_data(eg, cp.asnumpy(c.y))
                if ln.get_color() != c.color:
                    ln.set_color(c.color)
        self._spec_curves = curves

        lines = self.ax_spec.get_lines()
        active_maxes = [np.max(l.get_ydata()) for l in lines if l.get_visible() and l.get_alpha() == 1.0]
        if active_maxes:
            self.ax_spec.set_ylim(0, float(max(active_maxes) * 1.1))

        p_dist_val = p_dist[active_idx] if self.mode in ('Line', 'Path') else f"[{self.marker_coords[active_idx][0]:.1f}, {self.marker_coords[active_idx][1]:.1f}]"
        title = f"Partitioned LDOS: marker {self.active_marker_idx} at {p_dist_val} Å"
        if build:
            self.ax_spec.set(title=title, xlabel="Energy (eV)")
            if self.feenstra:
                self.ax_spec.set_ylabel("(dI/dV)/(I/V)")
        elif title != self._spec_title:
            self.ax_spec.set_title(title)
        self._spec_title = title
        # (The spectrum no longer keeps a square box in Line / Path: a curve panel fills the room the
        # arrangement gives it, so no blank is left around it; rule 11.)

        if self.mode == 'Map':
            # The energy markers: drawn with the curves, else moved to their energies (map_e_targets).
            if build:
                for i, e_val in enumerate(self.map_e_targets):
                    self._spec_emarkers.append(self.ax_spec.axvline(x=e_val, color='black', linestyle='-', lw=2, picker=5, label=f'emarker_{i}'))
            else:
                for line, e_val in zip(self._spec_emarkers, self.map_e_targets):
                    line.set_xdata([e_val, e_val])
        self._spec_key = key
        self._update_legend()                      # the legend panel follows the curves drawn

    def _build_heat_row(self, n_heat):
        """
        Machinery: the Line / Path heatmap row built anew, only when its panel count changes (n_heat, the
        partitions of the level, or the per-heatmap colour bars): the old panels removed
        (_clear_cluster), then the cluster's row, left to right (a PanelRow: its shaped panels
        share one height; rule 11): the 'Topo' panel with the LDOS h box above it (Line only),
        the tip-height stripe (with the LDOS h box above it in Path), one heatmap per partition
        with a colour bar after each (show_dcmp_norm) or after the last.  The panels' fixed parts
        (box aspects, anchors, axis labels, ticks) are set here and their data by _fill_heat_row;
        the Tip Height profile is cleared with the row (_fill_heat_row draws its curve and the
        marker lines again).  _update_all then fills the row and calls _rearrange().
        """
        self._clear_cluster()
        self.ax_prof.clear()
        num_p = max(1, n_heat)
        row = self._cluster_row
        if self.mode == 'Line':
            # Line Topo panel (the unwrapped 'Topo' corridor) — Line mode only; the LDOS h box above it.
            self.ax_line_topo = self._panel_with_box(row, self._box_ldos_h, stretch=10)
            self.line_decomp_axes.append(self.ax_line_topo)
            self.ax_line_topo.set_box_aspect(1.5)
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
        self.ax_stripe.set_xticks([])
        self.ax_prof.set(ylabel="Height (Å)", title="Tip Height", xlabel="Dist (Å)")

        for p_idx in range(n_heat):
            ax_l = self._panel(row, stretch=10)
            self.line_decomp_axes.append(ax_l)
            cax_l = None
            if self.show_dcmp_norm or p_idx == num_p - 1:
                cax_l = self._panel(row, inner_w=15)
                self.line_decomp_axes.append(cax_l)
            ax_l._cell.fill_as(1.0)       # sized and scored as a square panel; placed, its box fills its cell (no box aspect)
            ax_l.set_anchor('W')
            ax_l.set_yticks([])
            if cax_l is not None:
                cax_l.set_anchor('W')
            self._heat_axes.append(ax_l)
            self._heat_caxes.append(cax_l)
        n = len(self._heat_axes)
        self._heat_meshes, self._heat_grids, self._heat_cmaps = [None] * n, [None] * n, [None] * n
        self._heat_cbars, self._heat_titles = [None] * n, [None] * n

        # The cluster's row shares one height (the row rule, Figure.align_row): the heatmaps (no box
        # aspect: they fill their cells), the stripe and the colour bars take the row's band, in Line at
        # most the height the Line Topo's box aspect asks for.
        self.fig.align_row(*self.line_decomp_axes)
        self._row_key = (n_heat, self.show_dcmp_norm)

    def _fill_heat_row(self, processed_partitions, global_vmax, global_vmin, p_dist, p_len, use_diverging, has_mag):
        """
        Machinery: the Line / Path panels given the state's data in place (no panel made or removed): the
        Line Topo drawn again when its line, the Cells count, the atoms or the heights changed
        (its contour, atoms, line, ends and limits; the marker scatters stay), the stripe's
        segments, colours and limits (LineCollection.set_segments / set_norm / set_array), the
        Tip Height profile's curve (Line2D.set_data), each heatmap's values and colour scale
        (QuadMesh.set_array / set_cmap / set_norm; its colour bar follows the mesh's norm; a new
        mesh in the same axes only when its grid, the energies or the distances, changed), the
        heatmaps' titles, and the markers' counterparts (made anew when their count changed,
        then placed: _place_line_markers).  processed_partitions, global_vmax and global_vmin
        as _update_all computes them.
        """
        if self.ax_line_topo is not None:
            topo_key = (tuple(np.asarray(self.p1, dtype=float)), tuple(np.asarray(self.p2, dtype=float)),
                        max(int(self.s_cell.val), 1) + 1, bool(self.show_atoms))
            if (topo_key != self._line_topo_key or self._line_topo_z is None
                    or not np.array_equal(self._line_topo_z, self.global_z_map)):
                for art in self._line_topo_arts:
                    art.remove()
                arts = []
                v_line = self.p2 - self.p1
                v_hat = v_line / p_len
                v_perp = np.array([-v_hat[1], v_hat[0]])
                half_w = p_len / 3.0

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
                arts.append(self.ax_line_topo.tricontourf(all_rx, all_ry, all_z, levels=60, cmap=self._cmap('topo'), zorder=1))

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
                            arts.append(self.ax_line_topo.scatter(np.concatenate(a_rx_all), np.concatenate(a_ry_all), s=10, color=plt.cm.tab10(t_idx / 10), alpha=0.3, zorder=2))

                arts += self.ax_line_topo.plot([0, 0], [0, p_len], color='red', ls='--', lw=2.0, zorder=6)
                arts.append(self.ax_line_topo.scatter([0], [0], c='white', edgecolors='red', s=60, zorder=8))
                arts.append(self.ax_line_topo.scatter([0], [p_len], c='white', edgecolors='red', s=60, zorder=8))

                self.ax_line_topo.set_xlim(-half_w, half_w)
                self.ax_line_topo.set_ylim(0, p_len)
                self._line_topo_arts = arts
                self._line_topo_key = topo_key
                self._line_topo_z = np.array(self.global_z_map, copy=True)

        # The stripe and the Tip Height profile from the view's own tip heights: Line's (current_z_line) or
        # Path's (current_z_path).
        z_view = self.current_z_line if self.mode == 'Line' else self.current_z_path
        # The stripe: its segments, colours and limits in place.
        segs = np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1],
                         np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2)
        z_norm = plt.Normalize(float(z_view.min()), float(z_view.max()))
        if self._stripe_lc is None:
            lc = plt.LineCollection(segs, cmap=self._cmap('topo'), norm=z_norm, linewidth=40)
            lc.set_array(z_view[:-1])
            self.ax_stripe.add_collection(lc)
            self._stripe_lc = lc
        else:
            self._stripe_lc.set_segments(segs)
            self._stripe_lc.set_norm(z_norm)
            self._stripe_lc.set_array(z_view[:-1])
        if self._stripe_len != p_len:
            self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len))
            self._stripe_len = p_len
        # The Tip Height profile's curve.
        if self._prof_line is None:
            self._prof_line, = self.ax_prof.plot(p_dist, z_view, 'k-', lw=1.5)
        else:
            self._prof_line.set_data(p_dist, z_view)

        for p_idx, (p_label, t_data) in enumerate(processed_partitions):
            ax_l, cax_l = self._heat_axes[p_idx], self._heat_caxes[p_idx]

            v_max = float(cp.max(cp.abs(t_data))) if self.show_dcmp_norm else global_vmax
            if v_max == 0:
                v_max = 1e-15
            v_min = float(cp.min(t_data)) if self.show_dcmp_norm else global_vmin

            cmap = self._cmap('mag') if use_diverging else (self._cmap('|m|') if (self.show_mag and has_mag) else self._cmap('heatmap'))
            mesh, grid = self._heat_meshes[p_idx], self._heat_grids[p_idx]
            new_mesh = (mesh is None or not np.array_equal(grid[0], self.cached_eg) or not np.array_equal(grid[1], p_dist))
            if new_mesh:
                # A new grid (the energies or the distances): a new mesh in the same axes.
                if mesh is not None:
                    mesh.remove()
                if use_diverging:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, cp.asnumpy(t_data), cmap=self._cmap('mag'), shading='auto', vmin=v_min, vmax=v_max)
                elif self.show_mag and has_mag:
                    # NCL: |m| is non-negative, use sequential colormap
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, cp.asnumpy(t_data), cmap=self._cmap('|m|'), shading='auto', norm=plt.PowerNorm(self.ldos_gamma, vmin=v_min, vmax=v_max))
                else:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, cp.asnumpy(t_data), cmap=self._cmap('heatmap'), shading='auto', norm=plt.PowerNorm(self.ldos_gamma, vmin=v_min, vmax=v_max))
                self._heat_meshes[p_idx] = mesh
                self._heat_grids[p_idx] = (np.array(self.cached_eg, copy=True), np.array(p_dist, copy=True))
            else:
                # The same grid: the values and the colour scale of the same three branches set in place.
                norm_l = plt.Normalize(v_min, v_max) if use_diverging else plt.PowerNorm(self.ldos_gamma, vmin=v_min, vmax=v_max)
                if cmap != self._heat_cmaps[p_idx]:
                    mesh.set_cmap(cmap)
                mesh.set_array(cp.asnumpy(t_data))
                mesh.set_norm(norm_l)
            self._heat_cmaps[p_idx] = cmap

            # The panel's short title is the partition itself (the cluster's label row says
            # 'LDOS' and, at level 3 (the orbitals), the element).
            title = p_label.split()[-1] if self.plot_level >= 3 else p_label
            if title != self._heat_titles[p_idx]:
                ax_l.set_title(title, fontsize=10)
                self._heat_titles[p_idx] = title

            if cax_l is not None:
                exp = int(np.floor(np.log10(v_max)))
                bar = self._heat_cbars[p_idx]
                if new_mesh or bar is None:
                    cb = self.fig.colorbar(mesh, cax=cax_l)
                    cb.ax.yaxis.set_major_formatter(plt.ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
                    cax_l.set_title(f"1e{exp}", fontsize=10)
                    self._heat_cbars[p_idx] = (cb, exp)
                elif exp != bar[1]:
                    # The bar follows the mesh's norm by itself; its tick unit and exponent title follow v_max.
                    bar[0].ax.yaxis.set_major_formatter(plt.ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
                    cax_l.set_title(f"1e{exp}", fontsize=10)
                    self._heat_cbars[p_idx] = (bar[0], exp)

        if self._line_markers_n != len(self.marker_ratios):
            self._make_line_markers(p_dist, p_len)
        self._place_line_markers()

    def _make_line_markers(self, p_dist, p_len):
        """
        Machinery: the Line / Path markers' counterparts made anew for the marker count (the old ones
        removed): on the Line Topo a scatter at the marker's place along the line (pickable,
        label 'marker_<i>': a marker is dragged from the Line Topo), and, once there are
        heatmaps, a dashed line at the marker's distance on the Tip Height profile, on every
        heatmap and on the stripe (picker 5, label 'marker_<i>': dragged from any of them).
        """
        for items in self._line_markers:
            for art in [items['topo'], items['prof']] + items['h']:
                if art is not None:
                    art.remove()
        self._line_markers = []
        for i, r in enumerate(self.marker_ratios):
            idx = int(r * (self.npts - 1))
            color = self.line_marker_colors[i]
            items = {'topo': None, 'prof': None, 'h': []}
            if self.ax_line_topo is not None:
                items['topo'] = self.ax_line_topo.scatter([0], [r * p_len], c=color, edgecolors='black', s=80, zorder=10,
                                                          picker=5, label=f'marker_{i}')
            if self._heat_axes:
                items['prof'] = self.ax_prof.axvline(x=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                for ax_l in self._heat_axes:
                    items['h'].append(ax_l.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}'))
                items['h'].append(self.ax_stripe.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}'))
            self._line_markers.append(items)
        self._line_markers_n = len(self.marker_ratios)

    def _place_line_markers(self, counterparts=True):
        """
        Machinery: the Line / Path markers placed at marker_ratios on the line as last drawn (_line_geom):
        the placement topo's markers, and with `counterparts` the Line Topo's scatters, the
        dashed lines on the heatmaps and the stripe (set_ydata) and on the Tip Height profile
        (set_xdata), all together.  Nothing is computed.
        """
        p_xy, p_dist, p_len = self._line_geom
        if self.mode == 'Line':
            v = self.p2 - self.p1
            self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios]))
        else:
            mark_idx = (np.array(self.marker_ratios) * (self.npts - 1)).astype(int)
            self.marks.set_offsets(p_xy[mark_idx])
        if not counterparts:
            return
        for items, r in zip(self._line_markers, self.marker_ratios):
            y = p_dist[int(r * (self.npts - 1))]
            if items['topo'] is not None:
                items['topo'].set_offsets([[0, r * p_len]])
            for line in items['h']:
                line.set_ydata([y, y])
            if items['prof'] is not None:
                items['prof'].set_xdata([y, y])

    def _move_line_markers(self, last_run=False):
        """
        Machinery: a Line / Path marker dragged: every counterpart moved with it (_place_line_markers, running
        or not) and, while running, the marker spectra set from the display cache (_line_disp and its
        Norm factors _line_norm, built once per data change by _update_all; _draw_spectrum in place); the
        info panel's markers line follows (_markers_changed).  Nothing is computed again while the button
        is down.  With `last_run` (a re-sort: _resort_line_markers) the marker spectra are set from the
        display cache with RUN off too, the last run's data.
        """
        self._place_line_markers()
        if (self.is_running or last_run) and self._line_disp is not None and self._spec_key is not None:
            self._draw_spectrum(self._line_disp, None, self._line_geom[1], line_norm=self._line_norm)
        self._markers_changed()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _line_marker_order(ratios):
        """
        Machinery: the Line / Path markers' order on the stripe, top to bottom: the indices of `ratios` from
        the largest ratio to the smallest (the stripe's distance runs from 0 at the bottom to the line's
        length at the top), a stable sort, so markers at exactly equal ratios keep their order.
        """
        return sorted(range(len(ratios)), key=lambda i: -float(ratios[i]))

    def _form_marker_set(self):
        """
        Machinery: a freshly formed set of Line / Path markers (marker_ratios as just formed: the constructor's,
        run_interactive's, the Markers box's on an increase) put in its order on the stripe, top to bottom
        (_line_marker_order), then coloured by index (line_marker_colors: index 0, the top marker, takes
        m_colors[0]; a marker past the end of m_colors, with no free colour, m_colors[i % len(m_colors)]).
        """
        self.marker_ratios = [self.marker_ratios[j] for j in self._line_marker_order(self.marker_ratios)]
        self.line_marker_colors = [self.m_colors[i % len(self.m_colors)] for i in range(len(self.marker_ratios))]

    def _resort_line_markers(self):
        """
        Machinery: the Line / Path markers put back in their order on the stripe after a marker's release, top
        to bottom (_line_marker_order).  The permutation is applied together to marker_ratios, the kept
        colours (line_marker_colors) and the active marker's index (the same marker stays active); artist i
        shows marker i, so each index's artists take their marker's colour in place (the placement topo's
        markers, and the Line Topo's scatter and the Tip Height, heatmap and stripe lines), then every one is
        moved and the spectrum's curves (the level-0 curves), its title and the legend take their marker's
        data and colour in place (_move_line_markers), running or not (with RUN off from the last run's data,
        the display cache): one redraw, no rearrangement, no rebuild, nothing computed.  Nothing happens when
        the order is unchanged.
        """
        order = self._line_marker_order(self.marker_ratios)
        if order == list(range(len(order))):
            return
        active = order.index(min(self.active_marker_idx, len(order) - 1))
        self.marker_ratios = [self.marker_ratios[j] for j in order]
        self.line_marker_colors = [self.line_marker_colors[j] for j in order]
        self.active_marker_idx = active
        self.marks.set_facecolors(self.line_marker_colors)
        for i, (items, j) in enumerate(zip(self._line_markers, order)):
            if j == i:
                continue
            color = self.line_marker_colors[i]
            if items['topo'] is not None:
                items['topo'].set_facecolors(color)
            if items['prof'] is not None:
                items['prof'].set_color(color)
            for line in items['h']:
                line.set_color(color)
        self._move_line_markers(last_run=True)

    def _build_map_block(self, num_p, nepts):
        """
        Machinery: the Map view's block of energy maps built anew, only when its panel count changes (num_p
        partitions of the level, nepts maps each, the colour bars of 'cmap norm'): nepts maps per
        partition, wrapped into the rows that make them largest (controls.MapBlock; rule 11), a
        colour bar cell per partition (show_dcmp_norm), one per map ('cmap norm') or one for the
        whole block (automatic layout).  The maps' contents are drawn by _fill_map; _update_all
        then calls _rearrange().
        """
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
        self._map_block.set_groups(groups, shared_bar=shared, extra=self._spec_cell, cell_bars=cell_bars,
                                   extra_side=self._legend_panel)
        self._block_key = (num_p, nepts, self.cmap_norm, self.show_dcmp_norm)

    def _fill_map(self, k, p_idx, i, p_label, slice_data, v_min, v_max, target_e, use_diverging, has_mag, m_coords_np):
        """
        Machinery: energy map k (partition p_idx, energy i) given its data in its own axes, with no
        ax.clear(): its contour set (with the box's Shading on, its shaded RGBA meshes) drawn again only
        when its slice, its colour scale or its shading changed (_set_map_slice; an unchanged map is
        not touched), its own colour bar under 'cmap norm'
        (_set_map_cbar), the marker scatter made at its first drawing (pickable: a Map marker is
        dragged from any energy map; _place_map_markers moves it), its title and partition label
        set when their text changes, its ticks removed at its first drawing.
        """
        ax = self.map_axes[k]
        first = k not in self._map_marks
        self._set_map_slice(k, ax, slice_data, v_min, v_max, use_diverging, has_mag)
        if self.cmap_norm:
            self._set_map_cbar(k, v_min, v_max, use_diverging, has_mag, labelsize=6)
        if first:
            self._map_marks[k] = ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30,
                                            edgecolors='white', zorder=5, picker=5)
        title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
        if self.plot_level >= 3:
            ylabel_str = p_label.split()[-1] if i == 0 else ""
        else:
            ylabel_str = p_label if i == 0 else ""
        if self._map_titles.get(k) != title_str:
            ax.set_title(title_str, fontsize=10)
            self._map_titles[k] = title_str
        if ylabel_str and self._map_ylabels.get(k) != ylabel_str:
            ax.set_ylabel(ylabel_str, fontsize=10)
            self._map_ylabels[k] = ylabel_str
        if first:
            ax.set_xticks([])
            ax.set_yticks([])

    def _set_map_slice(self, k, ax, slice_data, v_min, v_max, use_diverging, has_mag):
        """
        Machinery: energy map k drawn in the map's own axes for the view's state: its signature, the colour
        scale (the colour map's branch: the signed spin density, |m| (Mag on and has_mag), else the LDOS; v_min,
        v_max; LDOS γ), with its data (_show_map_slice).
        """
        sig = (bool(use_diverging), bool(self.show_mag and has_mag), float(v_min), float(v_max), float(self.ldos_gamma))
        self._show_map_slice(k, ax, slice_data, sig)

    def _show_map_slice(self, k, ax, slice_data, sig):
        """
        Machinery: energy map k shown with its data and signature `sig` (_set_map_slice's): drawn again when the
        data, the signature or the box's shading (Shading's mode and its shade, _map_shade) differ from what it
        shows, otherwise nothing is touched.  The old drawing removed (its filled contours or its shaded meshes),
        the slice is drawn again as the set's filled contours (_draw_map_slice; TriContourSet has no set_array) or,
        with Shading on, as the shaded RGBA mesh (_draw_shaded_slice); _map_state keeps what it shows.
        """
        shade = self._map_shade()
        st = self._map_state.get(k)
        if (st is not None and st['sig'] == sig and st['shade'] is shade
                and cp.array_equal(st['data'], slice_data)):
            return
        if st is not None:
            if st['cs'] is not None:
                st['cs'].remove()
            for mesh in st['mesh']:
                mesh.remove()
        if shade is None:
            cs, meshes = self._draw_map_slice(ax, slice_data, *sig), []
        else:
            cs, meshes = None, self._draw_shaded_slice(ax, slice_data, *sig, shade)
        self._map_state[k] = {'cs': cs, 'mesh': meshes, 'sig': sig, 'shade': shade, 'data': slice_data.copy()}

    def _reshade_maps(self):
        """
        Machinery: in Map, every energy map drawn now shown again from what it shows (its kept data and signature,
        _map_state) under the box's shading (_show_map_slice: only a map whose shading changed is drawn again, as
        filled contours or as the shaded RGBA mesh), and the energy maps' colour bars given the maps' colour map
        (the set's, or the shading map unshaded while Shading is on: _map_role_cmap); nothing computed, made anew or
        arranged again.  The caller draws the canvas.
        """
        if self.mode != 'Map' or not self._map_state:
            return
        figure_axes = self.fig.axes
        for k, st in list(self._map_state.items()):
            if k < len(self.map_axes) and self.map_axes[k] in figure_axes:
                self._show_map_slice(k, self.map_axes[k], st['data'], st['sig'])
        diverging, abs_m = next(iter(self._map_state.values()))['sig'][:2]
        cmap = self._map_role_cmap(diverging, abs_m)
        for st in self._map_cbar_state.values():
            if st['cmap'] is not cmap:
                st['cmap'] = cmap
                st['cb'].mappable.set_cmap(cmap)

    def _on_shade_box(self, state):
        """
        Machinery: the Contour / Shading box clicked (cmap_shading.build_box has applied its rules to `state`, the
        window's shade_state, and set the toggles to match), running or not, display only (no LDOS recomputed,
        nothing arranged again): the level line on the topo colour bar shown with Contour on and hidden with it
        off; the energy maps drawn again for Shading (_reshade_maps: the shaded RGBA mesh, or the set's maps and
        filled contours again); the contour slice's traces on the energy maps drawn (Contour on) or removed
        (_draw_contour_artists); then one draw.  The info panel's settings section lists the box's controls.
        """
        if self._level_line is not None:
            self._level_line.set_visible(state.contour)
        self._reshade_maps()
        self._draw_contour_artists()
        self.fig.canvas.draw_idle()

    def _set_map_cbar(self, slot, v_min, v_max, use_diverging, has_mag, labelsize=None):
        """
        Machinery: the energy maps' colour bar cax_list[slot] at v_min..v_max: drawn the first time
        (_draw_map_cbar), then updated in place when its scale changes: its norm and colour map
        (ScalarMappable.set_norm / set_cmap; the bar follows), its tick unit and exponent title
        when the exponent changes.
        """
        cax = self.cax_list[slot]
        cmap = self._map_cmap(use_diverging, has_mag)
        sig = (float(v_min), float(v_max))
        st = self._map_cbar_state.get(slot)
        if st is None:
            cb = self._draw_map_cbar(cax, v_min, v_max, use_diverging, has_mag, labelsize=labelsize)
            self._map_cbar_state[slot] = {'cb': cb, 'sig': sig, 'cmap': cmap, 'exp': int(np.floor(np.log10(v_max)))}
            return
        if st['sig'] == sig and st['cmap'] == cmap:
            return
        sm = st['cb'].mappable
        if st['cmap'] != cmap:
            sm.set_cmap(cmap)
        sm.set_norm(plt.Normalize(vmin=v_min, vmax=v_max))
        exp = int(np.floor(np.log10(v_max)))
        if exp != st['exp']:
            st['cb'].ax.yaxis.set_major_formatter(plt.ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10 ** e):.1f}"))
            cax.set_title(f"1e{exp}", fontsize=10 if labelsize is None else labelsize)
        st.update(sig=sig, cmap=cmap, exp=exp)

    def _place_map_markers(self):
        """Machinery: the Map markers (marker_coords) placed on every counterpart together: the Map Topo's scatter and each energy map's (a drag moves them; the spectrum follows on release)."""
        xy = np.array(self.marker_coords)
        recolor = self._map_marks_xy is None or len(self._map_marks_xy) != len(xy)
        cols = self.m_colors[:len(xy)]
        for sc in [self.marks] + list(self._map_marks.values()):
            sc.set_offsets(xy)
            if recolor:
                sc.set_facecolors(cols)
        self._map_marks_xy = np.array(xy, copy=True)

    def _redraw_map_slice(self, i):
        target_e = self.map_e_targets[i]
        f_up = self.map_ld_up_gpu.copy()
        f_dn = self.map_ld_dn_gpu.copy() if self.map_ld_dn_gpu is not None else None
        f_ldos_raw = self._combine_ldos_display(f_up, f_dn, self.show_mag)
        if self.feenstra:
            # 'Feenstra': every map divided by I/V at the sample its LDOS is taken at (as in _update_all).
            f_ldos_raw = feenstra.divide(f_ldos_raw, self._feenstra_iv('map'))
        partitions = self.decomp.partitions(f_ldos_raw, self.atomtypes, self.atomnums, self.orbitals, self.dos_up_gpu)

        mag_signed = self._mag_is_signed()
        has_mag = self._has_mag_channel()
        use_diverging = self.show_mag and has_mag and mag_signed

        m_coords_np = np.array(self.marker_coords)
        nepts = len(self.map_e_targets)

        processed_partitions = []
        global_vmax = 0.0
        global_vmin = np.inf
        if self.normalize and f_ldos_raw is not None:
            total_ldos_map = cp.sum(f_ldos_raw, axis=(2, 3))
            # Each distinct snapped Map sample once, ascending, at its GPU energy e0 + (i - k0) step.
            u_e, s_idx = np.unique(self.map_e_idx, return_index=True)
            x_gpu = cp.asarray((u_e - self.k0).astype(np.float32)) * self.e_step32 + self.e032
            norm_denom_map = quadrature.simpson(total_ldos_map[:, s_idx], x=x_gpu, axis=1)[:, None] + 1e-15
        else:
            norm_denom_map = 1.0

        for p_label, p_data in partitions:
            t_data = p_data.copy()
            if self.normalize:
                t_data /= norm_denom_map
            t_data = cp.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
            processed_partitions.append((p_label, t_data))
            v_max = float(cp.max(cp.abs(t_data)))
            if v_max > global_vmax:
                global_vmax = v_max
            global_vmin = min(global_vmin, float(cp.min(t_data)))
        if global_vmax == 0:
            global_vmax = 1e-15

        for p_idx, (p_label, t_data) in enumerate(processed_partitions):
            k = p_idx * nepts + i
            p_vmax = float(cp.max(cp.abs(t_data))) if self.show_dcmp_norm else global_vmax
            p_vmin = float(cp.min(t_data)) if self.show_dcmp_norm else global_vmin
            slice_data = t_data[:, i]
            v_max = self._map_vmax(slice_data, p_vmax)
            v_min = self._map_vmin(slice_data, p_vmin)
            # The map of energy i in its own axes, in place (no ax.clear(): its marker scatter and the
            # contour slice's trace stay; _fill_map), its own bar under 'cmap norm'.
            self._fill_map(k, p_idx, i, p_label, slice_data, v_min, v_max, target_e, use_diverging, has_mag, m_coords_np)

    def _on_pick(self, event):
        label = event.artist.get_label() if hasattr(event.artist, 'get_label') else ''
        if isinstance(label, str) and label.startswith('_cursor_'):    # a crop cursor: its press is its own (before any marker's)
            _, _, axis, k = label.split('_')
            self.active_obj = ('cursor', axis, int(k))
            return
        if self.active_obj is not None and self.active_obj[0] == 'cursor':
            return
        if label == '_zlevel':                                             # the contour slice's level line
            self.active_obj = ('zlevel', event.artist.axes)
            return
        if event.artist == getattr(self, 'ends', None) and self.mode == 'Line':
            self.active_obj = ('end', event.ind[0])
        elif event.artist == getattr(self, 'path_pts', None) and self.mode == 'Path':
            self.active_obj = ('path_node', event.ind[0])
        elif event.artist == getattr(self, 'marks', None):
            self.active_obj = ('mark_map', event.ind[0])
        elif self.mode == 'Map' and any(event.artist is sc for sc in self._map_marks.values()):
            # A Map marker on an energy map: dragged as on the Map Topo (all counterparts together).
            self.active_obj = ('mark_map', event.ind[0])
        elif getattr(event, 'mouseevent', None) and event.mouseevent.inaxes == self.ax_spec:
            # The spectrum's clicks (the decomp, Map's energy markers) are decided once, at the press (_on_press).
            return
        elif (isinstance(event.artist, (plt.Line2D, plt._RefLine, plt.PathCollection))
              and event.artist.get_label().startswith('marker_')):
            # A Line / Path marker's line on the Tip Height profile, a heatmap or the stripe, or its
            # scatter on the Line Topo: selected (the active marker) and dragged from there.
            self.active_obj = ('mark_dynamic', int(event.artist.get_label().split('_')[1]))
            self.active_marker_idx = int(event.artist.get_label().split('_')[1])
            # Only what shows the active marker is updated: the spectrum, from the display cache.
            if self.is_running and self.mode in ('Line', 'Path') and self._line_disp is not None and self._spec_key is not None:
                self._draw_spectrum(self._line_disp, None, self._line_geom[1], line_norm=self._line_norm)
            self._markers_changed()                # the info panel's markers line stars the active marker
            self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes == self.ax_spec:
            # One decision per press, from the lines drawn at the press (before any redraw): the pickable line on
            # top under it (the highest zorder, ties to the one drawn last).
            hits = [l for l in self.ax_spec.get_lines() if l.pickable() and l.contains(event)[0]]
            top = ldos_decomp.top_artist(hits)
            if top is not None and top.get_label().startswith('emarker_') and event.button == plt.MouseButton.LEFT:
                # A Map energy marker on top under a left press: its drag at level 0 (not on a double click's
                # press; the toolbar's Pan / Zoom does not stop it), nothing at the other levels.
                if self.plot_level == 0 and not event.dblclick:
                    self.active_obj = ('emarker', int(top.get_label().split('_')[1]))
                return
            if self.fig.canvas.manager.toolbar.mode == "":
                item = self._spec_curves[self._spec_lines.index(top)] if top in self._spec_lines else None
                self._decomp_click(event.button, event.dblclick, item)
        elif self._in_legend(event):
            # A press in the legend panel (its stack, title or Back row): one decision, from the boxes drawn at
            # the press (the entry under it, or none); ignored while the toolbar's Pan or Zoom is on.
            if self.fig.canvas.manager.toolbar.mode == "":
                self._decomp_click(event.button, event.dblclick, self._legend_entry_at(event))

    def _decomp_click(self, button, dblclick=False, item=None):
        """
        Machinery: the one entry point of the decomp's clicks: the spectrum's presses (_on_press), the legend
        panel's and its Back button's.  The rules are ldos_decomp's (Decomp.press): a right click goes back one
        level, a left click on a clickable curve or entry goes down one level with its option, a double click's
        press, a middle click, a left click on neither and the two ends do nothing.  `item`: the
        ldos_decomp.Curve or ldos_decomp.Entry under the click, or None; the Back button is a right click
        (button plt.MouseButton.RIGHT, no item).  When the state changed: the window's decomp line, and one
        redraw (_update_all), running or not: with RUN off the spectrum, the legend panel, the heatmaps and the
        maps are drawn from the last run's data, nothing computed (last_run); otherwise nothing is redrawn.
        With RUN off and without the last run's data (_has_last_run: dropped by a change made while stopped,
        or no run in the view yet) a click does nothing: the decomp's state is kept, so no step is taken
        without its redraw.
        """
        if not self.is_running and not self._has_last_run():
            return
        if self.decomp.press(button, dblclick, item):
            self._decomp_changed()
            self._update_all(last_run=True)

    def _has_last_run(self):
        """
        Machinery: whether the view holds the last run's data a RUN-off redraw draws from (_update_all,
        last_run): Line / Path the line's LDOS computed in this view (cached_ld_up_gpu, cached_mode); Single
        Point its marker spectra (cached_spec_ldos_gpu), one for every marker; Map its maps' LDOS
        (map_ld_up_gpu) and its own marker spectra (map_spec_ldos_gpu), one for every marker.  False when they
        are not there (no run in the view yet, data dropped by a change made while stopped, a marker added
        while stopped).
        """
        spec = self.map_spec_ldos_gpu if self.mode == 'Map' else self.cached_spec_ldos_gpu
        spec_ok = spec is not None and len(spec) >= len(self.marker_coords)
        if self.mode in ('Line', 'Path'):
            return self.cached_ld_up_gpu is not None and getattr(self, 'cached_mode', None) == self.mode
        if self.mode == 'Map':
            return self.map_ld_up_gpu is not None and self.map_e_idx is not None and spec_ok
        return spec_ok

    def _decomp_changed(self):
        """
        Machinery: the window's decomp line in the info panel (set_info_line, key 'decomp'): the level's full
        word and the selection (ldos_decomp.Decomp.info), once the window exists.
        """
        if getattr(self, 'fig', None) is not None:
            self.fig.set_info_line('decomp', f"decomp: {self.decomp.info()}")

    def _markers_changed(self):
        """
        Machinery: the window's markers and line lines in the info panel (set_info_line), once the window
        exists, from the state as it is: key 'markers', every marker in index order, the active one starred
        (Single Point / Map 'k (x, y) Å'; Line / Path 'k at d Å', d the distance along the line of the
        sample the marker's spectrum is taken at); key 'line', the Line's ends or the Path's nodes with the
        length (none in Single Point / Map).
        """
        if getattr(self, 'fig', None) is None:
            return
        active = self.active_marker_idx
        if self.mode in ('Line', 'Path'):
            if self.mode == 'Line':
                p_len = float(norm(self.p2 - self.p1))
                points = f"ends ({self.p1[0]:.2f}, {self.p1[1]:.2f}), ({self.p2[0]:.2f}, {self.p2[1]:.2f}) Å"
            else:
                p_len = float(self._build_path_xy()[2])
                points = "nodes " + ", ".join(f"({n[0]:.2f}, {n[1]:.2f})" for n in self.path_nodes) + " Å"
            p_dist = np.linspace(0, p_len, self.npts)
            marks = [f"{k} at {p_dist[int(r * (self.npts - 1))]:.2f} Å" + (" *" if k == active else "")
                     for k, r in enumerate(self.marker_ratios)]
            line = f"line: {points}, length {p_len:.2f} Å"
        else:
            marks = [f"{k} ({pt[0]:.2f}, {pt[1]:.2f}) Å" + (" *" if k == active else "")
                     for k, pt in enumerate(self.marker_coords)]
            line = ''
        self.fig.set_info_line('markers', "markers: " + (", ".join(marks) if marks else "none"))
        self.fig.set_info_line('line', line)

    def _map_energies_changed(self):
        """
        Machinery: the window's map energies line in the info panel (set_info_line, key 'map energies'), once the
        window exists: in Map, while its energy maps are drawn (their block built, _block_key), the snapped energy
        of each map in map order (energies[map_e_idx], in eV as the maps' titles give them); none otherwise (the
        other views, or Map before its maps are drawn).
        """
        if getattr(self, 'fig', None) is None:
            return
        text = ''
        if (self.mode == 'Map' and getattr(self, '_block_key', None) is not None
                and getattr(self, 'map_e_idx', None) is not None):
            text = "map energies: " + ", ".join(f"{e:.3f} eV" for e in self.energies[self.map_e_idx])
        self.fig.set_info_line('map energies', text)

    def _crop_changed(self):
        """
        Machinery: the window's crop line in the info panel (set_info_line, key 'crop'), once the window exists: the
        applied crop's rectangle on the placement topo, x and y each from lo to hi in Å (an axis with no crop of
        its own at the drawn cells' full extent, as _apply_crop draws it), or 'none' when nothing is cropped.
        """
        if getattr(self, 'fig', None) is None:
            return
        if self.crop_x is None and self.crop_y is None:
            text = "crop: none"
        else:
            ex, ey = self._full_extent()
            x_rng = self.crop_x if self.crop_x is not None else ex
            y_rng = self.crop_y if self.crop_y is not None else ey
            text = f"crop: x {x_rng[0]:.2f} to {x_rng[1]:.2f} Å, y {y_rng[0]:.2f} to {y_rng[1]:.2f} Å"
        self.fig.set_info_line('crop', text)

    def _contour_level_changed(self):
        """
        Machinery: the window's contour level line in the info panel (set_info_line, key 'contour level'), once the
        window exists: in Map, the contour slice's level z_frac, the fraction of the Map Topo's height range the
        level line on the topo colour bar marks (_set_level); none in the other views.
        """
        if getattr(self, 'fig', None) is None:
            return
        text = f"contour level: {self.z_frac:.3f} of the Map Topo's height range" if self.mode == 'Map' else ''
        self.fig.set_info_line('contour level', text)

    def _get_element_by_index_helper(self, a):
        curr = 0
        for idx, count in enumerate(self.atomnums):
            if a <= curr + count:
                return self.atomtypes[idx]
            curr += count
        return 'grey'

    def _on_motion(self, event):
        if self.active_obj is not None and self.active_obj[0] == 'zlevel':
            # The contour slice's level line follows the cursor's height on the bar it was picked on
            # while the button is down, wherever the cursor is sideways (_ramp_fraction).
            frac = self._ramp_fraction(event, self.active_obj[1])
            if frac is not None:
                self._set_level(frac)
            return
        if self.active_obj is None or event.xdata is None:
            return
        if self.active_obj[0] == 'cursor':
            # A crop cursor follows the mouse on the placement topo along its own axis (made valid as it goes).
            if event.inaxes is self._crop_ax and event.ydata is not None:
                _, axis, k = self.active_obj
                self._set_cursor(axis, k, event.xdata if axis == 'x' else event.ydata)
            return
        # A drag moves only the dragged item and its counterparts: no _update_all, no GPU work and no
        # recomputation while the button is down (the release does that work once: _on_rel), and a
        # motion that leaves the item where it is returns at once.
        t_obj, idx = self.active_obj
        if self.mode in ('Line', 'Path'):
            p_xy, _, p_len = self._line_geom           # the line / path as last drawn (_update_all)
        else:
            p_len = 1.0

        if t_obj == 'end' and self.mode == 'Line':
            # A line end: only the line and its end markers move on the placement topo.
            new = np.array([event.xdata, event.ydata])
            if np.array_equal(new, self.p1 if idx == 0 else self.p2):
                return
            if idx == 0:
                self.p1 = new
            else:
                self.p2 = new
            self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])
            self.ends.set_offsets([self.p1, self.p2])
        elif t_obj == 'path_node' and self.mode == 'Path':
            # A path node: only the path and its node markers move on the placement topo.
            if event.inaxes != self.ax_map:
                return
            new = np.array([event.xdata, event.ydata])
            if np.array_equal(new, self.path_nodes[idx]):
                return
            self.path_nodes[idx] = new
            _, _, _, verts = self._build_path_xy()
            self.line_art.set_data(verts[:, 0], verts[:, 1])
            self.path_pts.set_offsets(np.array(self.path_nodes))
        elif t_obj == 'mark_map' and (event.inaxes == self.ax_map
                                      or (self.mode == 'Map' and event.inaxes in self.map_axes)):
            if self.mode == 'Line':
                v = self.p2 - self.p1
                v_sq = np.dot(v, v)
                if v_sq <= 1e-9:
                    return
                new = np.clip(np.dot(np.array([event.xdata, event.ydata]) - self.p1, v) / v_sq, 0, 1)
            elif self.mode == 'Path':
                # Project click onto the polyline: nearest resampled path point.
                d2 = (p_xy[:, 0] - event.xdata) ** 2 + (p_xy[:, 1] - event.ydata) ** 2
                new = int(np.argmin(d2)) / (self.npts - 1)
            else:
                new = [event.xdata, event.ydata]
            if self.mode in ('Line', 'Path'):
                if new == self.marker_ratios[idx]:
                    return
                self.marker_ratios[idx] = new
                self._move_line_markers()
                self._drag_moved = True
                return
            if new == list(self.marker_coords[idx]):
                return
            self.marker_coords[idx] = new
            # Single Point / Map: the markers move (Map: on the Map Topo and every energy map); the
            # spectrum is computed on release.
            if self.mode == 'Map':
                self._place_map_markers()
            else:
                self.marks.set_offsets(self.marker_coords)
        elif t_obj == 'mark_dynamic' and self.mode in ('Line', 'Path'):
            new = None
            if hasattr(self, 'ax_prof') and event.inaxes == self.ax_prof:
                if p_len > 1e-9:
                    new = np.clip(event.xdata / p_len, 0, 1)
            elif event.inaxes in [ax for ax in [getattr(self, 'ax_ldos', None), getattr(self, 'ax_stripe', None), getattr(self, 'ax_line_topo', None)] + list(self._heat_axes) if ax is not None]:
                if p_len > 1e-9:
                    new = np.clip(event.ydata / p_len, 0, 1)
            if new is None or new == self.marker_ratios[idx]:
                return
            self.marker_ratios[idx] = new
            self._move_line_markers()
            self._drag_moved = True
            return
        elif t_obj == 'emarker' and event.inaxes == self.ax_spec:
            # Clipped to Map's kept window (the typed ends its energies were last formed from, map_typed[:2]; the
            # boxes may hold an edit not yet computed).
            self.map_e_targets[idx] = np.clip(event.xdata, self.map_typed[0], self.map_typed[1])
            for line in self.ax_spec.get_lines():
                if line.get_label() == f'emarker_{idx}':
                    line.set_xdata([self.map_e_targets[idx], self.map_e_targets[idx]])
            self.fig.canvas.draw_idle()
            return
        else:
            return
        self._markers_changed()                    # the info panel's markers and line lines follow the drag
        self._drag_moved = True
        self.fig.canvas.draw_idle()

    def _on_rel(self, event):
        if self.active_obj is not None and self.active_obj[0] == 'emarker':
            idx = self.active_obj[1]
            target_e = self.map_e_targets[idx]
            # The released energy snapped to its sample among the displayed samples of Map's kept window
            # (_map_sample_indices of the typed ends its energies were last formed from, map_typed[:2], whose
            # snapped ends are map_window); the marker moves to that sample's energy, so it marks the energy its
            # map shows.
            e_idx = int(self._map_sample_indices([target_e], ends=self.map_typed[:2])[0])
            self.map_e_idx[idx] = e_idx
            self.map_e_targets[idx] = self.energies[e_idx]
            for line in self.ax_spec.get_lines():
                if line.get_label() == f'emarker_{idx}':
                    line.set_xdata([self.map_e_targets[idx], self.map_e_targets[idx]])
            # The dragged map by the same call and at the same tip heights as the full Map computation: Map's kept
            # LDOS-topo heights (map_z, not the shared current_z_map), under Map's kept model: its decay model
            # (map_d_ldos; 'decay_tilt' is Decay on: _effective_decay_model) and its broadening (map_broadening).
            # While the window's broadening differs from Map's (changed since, while stopped or in another view), the
            # DOS is rebuilt at Map's for this one call (_apply_extra_broadening) and then at the window's again.
            kept_broad = self.map_broadening is not None and self.map_broadening != self.extra_broadening
            if kept_broad:
                window_broad, self.extra_broadening = self.extra_broadening, dict(self.map_broadening)
                self._apply_extra_broadening()
            try:
                grid_z = cp.hstack([self.grid_xy_gpu, cp.asarray(self.map_z, dtype=cp.float32)[:, None]])
                t_up, t_dn, _, t_cur = self._calculate_ldos_at_points_gpu(
                    grid_z, None, None, use_energy_decay=self.map_d_ldos == 'decay_tilt', preserve_orbitals=True,
                    with_current=True, samples=[e_idx], progress='dragged map')
            finally:
                if kept_broad:
                    self.extra_broadening = window_broad
                    self._apply_extra_broadening()
            # Into Map's own caches (map_*): its LDOS columns, the NumPy and the GPU copies.
            self.map_ld_up[:, idx:idx + 1] = cp.asnumpy(t_up[:, 0:1])
            self.map_ld_up_gpu[:, idx:idx + 1] = t_up[:, 0:1]
            if t_dn is not None and self.map_ld_dn is not None:
                self.map_ld_dn[:, idx:idx + 1] = cp.asnumpy(t_dn[:, 0:1])
                self.map_ld_dn_gpu[:, idx:idx + 1] = t_dn[:, 0:1]
            # Its current, from the same call, into column idx, and its sample's GPU energy e0 + (i - k0) step.
            self.map_current[:, idx] = t_cur[:, 0]
            self.map_current_E[idx] = self.e032 + np.float32(e_idx - self.k0) * self.e_step32
            # The maps back in increasing energy: a stable sort of the snapped Map energies, applied together
            # to the snapped energies and indices, the cached columns and their GPU copies, the current and its
            # energies, and the energy markers and their labels (marker i at map i's energy).  Every map panel whose
            # map changed position is redrawn in place with its new map (_redraw_map_slice: every partition row),
            # and the dragged map; nothing is arranged, resized or rebuilt, and nothing more is computed.
            order = np.argsort(self.map_e_targets, kind='stable')
            redraw = [int(np.flatnonzero(order == idx)[0])]                  # the dragged map's new position
            if not np.array_equal(order, np.arange(len(order))):
                o_gpu = cp.asarray(order)
                self.map_e_idx = self.map_e_idx[order]
                self.map_e_targets = self.map_e_targets[order]
                self.map_ld_up = self.map_ld_up[:, order]
                self.map_ld_up_gpu = self.map_ld_up_gpu[:, o_gpu]
                if self.map_ld_dn is not None:
                    self.map_ld_dn = self.map_ld_dn[:, order]
                if self.map_ld_dn_gpu is not None:
                    self.map_ld_dn_gpu = self.map_ld_dn_gpu[:, o_gpu]
                self.map_current = self.map_current[:, o_gpu]
                self.map_current_E = self.map_current_E[o_gpu]
                self._spec_emarkers = [self._spec_emarkers[j] for j in order]
                for i, line in enumerate(self._spec_emarkers):
                    line.set_label(f'emarker_{i}')
                redraw += [i for i in range(len(order)) if order[i] != i and i != redraw[0]]
            for i in redraw:
                self._redraw_map_slice(i)
            self._map_energies_changed()           # the info panel's map energies line: the snapped energies, in map order
            self.fig.canvas.draw_idle()
        elif (self.active_obj is not None and self._drag_moved
              and (self.active_obj[0] in ('end', 'path_node')
                   or (self.active_obj[0] == 'mark_map' and self.mode in ('Single Point', 'Map')))):
            # The release of a moved line end, path node or Single Point / Map marker: the work today's
            # redraw did on every motion, done once (Line / Path: the tip heights, the LDOS along the
            # line, the heatmaps, stripe, profile, Line Topo and marker spectra; Single Point / Map:
            # the markers' spectrum), by the same computations (_update_all).
            self._update_all()
        elif (self.active_obj is not None and self.mode in ('Line', 'Path')
              and self.active_obj[0] in ('mark_map', 'mark_dynamic')):
            # The release of a Line / Path marker: the markers put back in their order on the stripe, top to
            # bottom, with their colours, the active marker and every counterpart (_resort_line_markers: one
            # redraw when the order changed, nothing otherwise).
            self._resort_line_markers()
        self._drag_moved = False
        self.active_obj = None
