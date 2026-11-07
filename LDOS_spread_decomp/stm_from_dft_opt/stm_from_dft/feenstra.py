# -*- coding: utf-8 -*-
"""
Feenstra's normalized conductance (dI/dV)/(I/V) for the simulator's 'Feenstra' toggle.

The simulated LDOS stands in for dI/dV, and its integral from 0 V stands in for the current:

    I(r, V) = integral from 0 to V of LDOS_total(r, E) dE        (current_from_ldos)

with E the DOSCAR energy relative to E_F in eV, read as the sample bias V in volts.  I/V is
then formed and, optionally, broadened by the rule of the sm4_Browser's feenstra.iv_broadened
(iv_ratio), and every displayed array is divided by it along its energy axis (divide).

The I/V rule is a port of sm4_Browser/feenstra.py (iv_broadened), with the parts that only
apply to a measured current left out: no Hampel / Savitzky-Golay filters and no '0Vsub' (the
current here is 0 at 0 V by construction).  The rows here are points and the columns energies
(sm4 has energies down the rows and one spectrum per column); the arithmetic is the same.

Pure numpy, no GPU and no window code: the simulator computes the total LDOS on the GPU and
hands the arrays here.

Reference: R. M. Feenstra, J. A. Stroscio and A. P. Fein, "Tunneling spectroscopy of the
Si(111)2x1 surface", Surf. Sci. 181, 295 (1987), doi:10.1016/0039-6028(87)90170-1.

Developed by Benjamin Kafin
"""

import numpy as np


def current_from_ldos(energies, ldos):
    """
    I(V) = integral from 0 to V of the LDOS, per point, at every sample of `energies`.

    energies: (n_e,) ascending, in eV relative to E_F; it must reach 0 (0 on a sample or between
    two samples).  ldos: (n_pts, n_e), the total LDOS of each point at those energies.  The
    trapezoid rule over the samples, with a sample at exactly 0 inserted (its LDOS linearly
    interpolated between the two samples bracketing 0) so that I(0) = 0 exactly; the integral of
    the linear interpolant of the LDOS from 0 to each sample.  V < 0 gives I < 0 for a
    non-negative LDOS, so I/V is positive on both sides.  Returns (n_pts, n_e) float64 at the
    given energies (the inserted sample is not returned).
    """
    e = np.asarray(energies, dtype=np.float64)
    L = np.atleast_2d(np.asarray(ldos, dtype=np.float64))
    if e.ndim != 1 or L.shape[1] != len(e):
        raise ValueError(f"current_from_ldos: energies {e.shape} and ldos {L.shape} do not match.")
    if len(e) == 0 or e[0] > 0.0 or e[-1] < 0.0:
        raise ValueError("current_from_ldos: the energies must reach 0 (E_F) to integrate from it.")
    if len(e) > 1 and np.any(np.diff(e) <= 0):
        raise ValueError("current_from_ldos: the energies must be strictly ascending.")
    exact = np.flatnonzero(e == 0.0)
    if len(exact):
        k0 = int(exact[0])
        e_full, L_full, inserted = e, L, None
    else:
        k0 = int(np.searchsorted(e, 0.0))                        # first sample above 0; e[k0 - 1] < 0 < e[k0]
        t = (0.0 - e[k0 - 1]) / (e[k0] - e[k0 - 1])
        L0 = L[:, k0 - 1] + t * (L[:, k0] - L[:, k0 - 1])
        e_full = np.insert(e, k0, 0.0)
        L_full = np.insert(L, k0, L0, axis=1)
        inserted = k0
    C = np.zeros_like(L_full)
    if len(e_full) > 1:
        C[:, 1:] = np.cumsum(0.5 * (L_full[:, 1:] + L_full[:, :-1]) * np.diff(e_full)[None, :], axis=1)
    I_full = C - C[:, k0:k0 + 1]
    if inserted is not None:
        I_full = np.delete(I_full, inserted, axis=1)
    return I_full


def iv_ratio(energies, current, broadening=0.0):
    """
    I/V per point at every sample of `energies` (the rule of sm4_Browser feenstra.iv_broadened).

    energies: (n_e,) in volts (eV relative to E_F); current: (n_pts, n_e) from current_from_ldos.
    broadening > 0: the exponentially weighted mean of I/V over the samples,
        (sum_k (I_k / V_k) exp(-|V_k - V_i| / broadening)) / (sum_k exp(-|V_k - V_i| / broadening)),
    a sample exactly at 0 V carrying no weight as a source and still receiving a value from its
    neighbours.  broadening <= 0: the plain I/V, a sample exactly at 0 V given the linear
    interpolation of its neighbours' I/V.  Returns (n_pts, n_e) float64.
    """
    e = np.asarray(energies, dtype=np.float64)
    I = np.atleast_2d(np.asarray(current, dtype=np.float64))
    order = np.argsort(e)
    es = e[order]
    Is = I[:, order]
    nz = es != 0.0
    ratio = np.zeros_like(Is)
    ratio[:, nz] = Is[:, nz] / es[nz][None, :]
    if broadening > 0:
        W = np.exp(-np.abs(es[:, None] - es[None, :]) / float(broadening))     # W[i, k]
        W[:, ~nz] = 0.0                                                          # no source at 0 V
        denom = (ratio @ W.T) / (W @ nz.astype(np.float64))[None, :]
    else:
        denom = ratio
        if not nz.all():
            for p in range(Is.shape[0]):
                denom[p, ~nz] = np.interp(es[~nz], es[nz], ratio[p, nz])
    out = np.empty_like(denom)
    out[:, order] = denom
    return out


def divide(arr, iv):
    """
    `arr` (n_pts, n_e, ...) divided along its energy axis by `iv` (n_pts, n_e): every partition,
    atom and orbital of a point divided by the same total I/V of that point, so partitions stay
    additive.  Where I/V is exactly 0 (no states anywhere between 0 V and V: 0/0) the result is
    0, so the colour scales stay finite.  float64.
    """
    a = np.asarray(arr, dtype=np.float64)
    d = np.asarray(iv, dtype=np.float64)
    d = d.reshape(d.shape + (1,) * (a.ndim - d.ndim))
    out = np.zeros(np.broadcast_shapes(a.shape, d.shape), dtype=np.float64)
    np.divide(a, d, out=out, where=(d != 0.0))
    return out
