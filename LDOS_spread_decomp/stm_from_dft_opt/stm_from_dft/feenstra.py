# -*- coding: utf-8 -*-
"""
Feenstra's normalized conductance (dI/dV)/(I/V) for the simulator's 'Feenstra' toggle.

Physics: the simulated LDOS stands in for dI/dV, and its integral from 0 V stands in for the
current:

    I(r, V) = integral from 0 to V of LDOS_total(r, E) dE        (current_from_ldos)

with E the DOSCAR energy relative to E_F in eV, read as the sample bias V in volts.  I/V is
then formed as the plain I(V)/V at each sample (iv_ratio), and every displayed array is divided
by it along its energy axis (divide).

Machinery: the integral is quadrature.py's integral_from_zero, from exactly 0 V on the even DOS
grid: the cubic through the two samples on each side of 0 V (through E-1, 0, E1, E2 above and
E-2, E-1, 0, E1 below when a sample is at 0 V), its value at 0 V used by the unequal Simpson
pair next to 0 V when the interval count is even, its exact integral over the interval next to
0 V when the count is odd, Simpson pairs over the rest.  A sample exactly at 0 V, where I/V is
0/0, is given the linear interpolation of its neighbours' I/V (iv_ratio), and where I/V is
exactly 0 the division gives 0 (divide).

CuPy, float32, and no window code: the simulator computes the total LDOS on the GPU and hands
the arrays here, and what is computed here stays on the GPU.  Every array here is a float32
CuPy array (the inputs are taken with cp.asarray and dtype float32, which does not copy the
simulator's float32 GPU arrays), and scalars enter the arithmetic only as Python numbers or
float32 0-d CuPy arrays, so nothing is promoted to float64.

Reference: R. M. Feenstra, J. A. Stroscio and A. P. Fein, "Tunneling spectroscopy of the
Si(111)2x1 surface", Surf. Sci. 181, 295 (1987), doi:10.1016/0039-6028(87)90170-1.

Developed by Benjamin Kafin
"""

import cupy as cp

from . import quadrature


def current_from_ldos(ldos, k0, step, e0):
    """
    Physics: I(V) = integral from 0 to V of the LDOS, per point, at every sample of `ldos`.

    ldos: (n_pts, n_e), the total LDOS of each point on consecutive DOS samples, a float32 CuPy
    array.  k0 (int): the position among them of DOS sample k0, the last sample at or below 0 V.
    step, e0 (np.float32): the DOS grid, step = (E max - E min)/(NEDOS - 1) and e0 the energy of
    sample k0 (E - E_F), exactly 0.0 when a sample is at 0 V; the sample at position j is at
    e0 + (j - k0) step, and no energy array is taken.  V < 0 gives I < 0 for a non-negative LDOS,
    so I/V is positive on both sides; a sample exactly at 0 V gets I = 0.  Returns (n_pts, n_e)
    float32 CuPy at the given samples.

    Machinery: the integral is quadrature.integral_from_zero, the rule of quadrature.py from
    exactly 0 V.  The samples must include E-2 to E2, the two on each side of 0 V (positions
    k0 - 1 to k0 + 2, or k0 - 2 to k0 + 2 when e0 == 0.0), for the value at 0 V and the piece next
    to it; nothing beyond a sample is integrated for its I.  None of this is checked.
    """
    return quadrature.integral_from_zero(ldos, k0, step, e0, axis=-1)


def iv_ratio(energies, current):
    """
    Physics: I/V per point at every sample of `energies`, the plain I(V)/V.

    energies: (n_e,) the GPU energies of the samples, e0 + (i - k0) step, in volts (eV relative
    to E_F), a float32 CuPy array in any order; current: (n_pts, n_e), the current I at those
    samples, a float32 CuPy array.  Returns (n_pts, n_e) float32 CuPy.

    Machinery: a sample exactly at 0 V (0/0) is given the linear interpolation at 0 V of the I/V
    of its neighbours among the given samples, the last one below 0 V and the first one above
    (np.interp's rule, in float32: the two found with cp.searchsorted, every point at once, filled
    in with cp.where); with samples on one side of 0 V only, the I/V of the nearest one.
    """
    e = cp.asarray(energies, dtype=cp.float32)
    I = cp.atleast_2d(cp.asarray(current, dtype=cp.float32))
    order = cp.argsort(e)
    es = e[order]
    Is = I[:, order]
    nz = es != 0.0
    ratio = cp.zeros_like(Is)
    ratio[:, nz] = Is[:, nz] / es[nz][None, :]
    denom = ratio
    if not nz.all():
        # The samples at 0 V are one block of the sorted energies: lo is the last sample below it,
        # hi the first above (-1 / n_e: none on that side).
        zero = cp.zeros(1, dtype=cp.float32)
        lo = int(cp.searchsorted(es, zero, side='left')[0]) - 1
        hi = int(cp.searchsorted(es, zero, side='right')[0])
        if lo < 0:
            r0 = ratio[:, hi]
        elif hi >= es.shape[0]:
            r0 = ratio[:, lo]
        else:
            slope = (ratio[:, hi] - ratio[:, lo]) / (es[hi] - es[lo])
            r0 = slope * (0.0 - es[lo]) + ratio[:, lo]
        denom = cp.where(nz, ratio, r0[:, None])
    out = cp.empty_like(denom)
    out[:, order] = denom
    return out


def divide(arr, iv):
    """
    Physics: the normalized conductance (dI/dV)/(I/V): `arr` (n_pts, n_e, ...) divided along its
    energy axis by `iv` (n_pts, n_e), every partition, atom and orbital of a point divided by the
    same total I/V of that point, so partitions stay additive.  float32 CuPy.

    Machinery: where I/V is exactly 0 (no states anywhere between 0 V and V: 0/0) the result is
    0, so the colour scales stay finite; the divisor is replaced by 1 there (cp.where), so no
    division by 0 is evaluated.
    """
    a = cp.asarray(arr, dtype=cp.float32)
    d = cp.asarray(iv, dtype=cp.float32)
    d = d.reshape(d.shape + (1,) * (a.ndim - d.ndim))
    return cp.where(d != 0, a / cp.where(d != 0, d, 1), 0)
