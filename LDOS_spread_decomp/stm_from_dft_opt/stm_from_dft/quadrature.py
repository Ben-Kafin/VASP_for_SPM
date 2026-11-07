# -*- coding: utf-8 -*-
"""
The package's one integration rule, on the GPU (CuPy, float32).

Machinery, no physics: this module only integrates sampled arrays.  What is integrated and why
(the weighted LDOS, the current, dI/dV) is the decay model's, in simulator.py's block THE DECAY
MODEL (physics).  Everything in the package that integrates uses this module; the simulator's
exact_sweep and linearized_sweep kernels repeat integral_from_zero's formulas in CUDA, because they cannot call Python.

- simpson(y, x=None, *, dx=1.0, axis=-1): composite Simpson's rule, SciPy 1.18.1's
  scipy.integrate.simpson (its own signature) with its arithmetic transcribed.  Nothing is
  imported from SciPy, so the result does not depend on the installed SciPy.
- integral_from_zero(f, k0, step, e0, axis=-1): the oriented integral from 0 V to every DOS
  sample of f, on the even DOS grid (sample i at e0 + (i - k0) step): the cubic through the two
  samples on each side of 0 V (through E-1, 0, E1, E2 above and E-2, E-1, 0, E1 below when a
  sample is at 0 V), its value at 0 V used by the unequal Simpson pair next to 0 V when the
  interval count is even, its exact integral over the interval next to 0 V when the count is
  odd, Simpson pairs over the rest.

Precision: every CuPy array here is float32.  The inputs (the arrays, dx, step, e0) are rounded
to float32 once, when they go to the GPU (cp.asarray with dtype float32, which does not copy a
float32 CuPy array), and scalars enter CuPy arithmetic only as Python floats or float32 0-d CuPy
arrays, so nothing is promoted to float64.
"""

import cupy as cp


def _tupleset(t, i, value):
    """Machinery: the tuple t with entry i replaced by value (SciPy's tupleset)."""
    l = list(t)
    l[i] = value
    return tuple(l)


def _divide_where(cond, a, b):
    """
    Machinery: a / b where cond is True, 0 where it is False (SciPy's
    xpx.apply_where(cond, (a, b), divide, fill_value=0.)), with a safe division: b is replaced by 1
    where cond is False, so no division by 0 is evaluated.  Shapes broadcast as there.
    """
    return cp.where(cond, a / cp.where(cond, b, 1.0), 0.0)


def _pair(y0, y1, y2, h0, h1):
    """
    Machinery: one Simpson pair for unequal intervals, the integral over [x0, x0 + h0 + h1] of the
    parabola through (x0, y0), (x0 + h0, y1), (x0 + h0 + h1, y2): SciPy 1.18.1's _basic_simpson
    formula for given x, its division guards kept (_divide_where).  Arrays broadcast.
    """
    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = _divide_where(h1 != 0, h0, h1)
    return hsum/6.0 * (y0 *
        (2.0 - _divide_where(h0divh1 != 0, cp.ones_like(h0divh1), h0divh1))
        + y1 * (hsum * _divide_where(hprod != 0, hsum, hprod))
        + y2 * (2.0 - h0divh1))


def _basic_simpson(y, start, stop, x, dx, axis):
    """
    Machinery: SciPy 1.18.1's _basic_simpson: the Simpson pairs starting at samples start,
    start + 2, ... (before stop) along axis, summed; with x None every interval is dx, otherwise
    the intervals are the differences of x (the unequal-interval pair, _pair).
    """
    nd = y.ndim
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = _tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = _tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = _tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = dx / 3.0 * cp.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = cp.diff(x, axis=axis)
        sl0 = _tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = _tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0] if h.size else h
        h1 = h[sl1] if h.size else h
        tmp = _pair(y[slice0], y[slice1], (y[slice2] if y.size > 1 else y[0:0, ...]), h0, h1)
        result = cp.sum(tmp, axis=axis)
    return result


def simpson(y, x=None, *, dx=1.0, axis=-1):
    """
    Machinery: the integral of y along `axis` by the composite Simpson's rule, SciPy 1.18.1's
    scipy.integrate.simpson transcribed for CuPy in float32 (its signature: dx and axis
    keyword-only).

    y: the samples; x: their positions along `axis`, 1-D of y's length along `axis` or y's shape,
    in any spacing (None: every interval is dx).  y, x and dx are taken as float32 CuPy arrays
    (cp.asarray with dtype float32, SciPy's type promotion with the result type float32).  The
    equally spaced DOS samples are integrated with dx=step.

    N samples along `axis`:
      - N odd: Simpson pairs over every interval;
      - N even, N >= 4: pairs over the first N - 2 intervals plus Cartwright's 3-point segment on
        the last interval (the parabola through the last three samples, over the last interval);
      - N == 2: the trapezoid, 0.5 (x1 - x0) (y0 + y1), or with dx 0.5 dx (y0 + y1);
      - N == 1: 0.
    SciPy's division guards (xpx.apply_where, fill value 0) are kept as cp.where with a safe
    division (_divide_where), 0 where SciPy gives 0.

    Returns a float32 CuPy array of y's shape without `axis` (0-d for a 1-D y).  Raises ValueError
    as SciPy does for an x that is neither 1-D nor of y's number of dimensions, or whose length
    along `axis` differs from y's.
    """
    y = cp.asarray(y, dtype=cp.float32)
    if x is not None:
        x = cp.asarray(x, dtype=cp.float32)
    dx = cp.asarray(dx, dtype=cp.float32)
    nd = y.ndim
    N = y.shape[axis]
    last_dx = dx
    if x is not None:
        if x.ndim == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            x = cp.reshape(x, tuple(shapex))
        elif x.ndim != y.ndim:
            raise ValueError("If given, shape of x must be 1-D or the same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the same as y.")

    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = _tupleset(slice_all, axis, -1)
            slice2 = _tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val = 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

            slice1 = _tupleset(slice_all, axis, -1)
            slice2 = _tupleset(slice_all, axis, -2)
            slice3 = _tupleset(slice_all, axis, -3)

            h = cp.stack([dx, dx])
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = _tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = _tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = cp.diff(x, axis=axis)
                h = [cp.squeeze(diffs[hm2], axis=axis),
                     cp.squeeze(diffs[hm1], axis=axis)]

            # The correction for the last interval according to Cartwright (Cartwright 2017,
            # Eq. 8, rewritten for the last interval, as SciPy has it).
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = _divide_where(den != 0, num, den)

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = _divide_where(den != 0, num, den)

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = _divide_where(den != 0, num, den)

            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]

        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    return result


def _chain(g, piece1, pair2, dx):
    """
    Machinery: one side of 0 V for integral_from_zero, the integrals from 0 V summed outward by
    cumulative Simpson pairs.

    g: (..., m), m >= 2, the samples outward from 0 V along the last axis (g[..., 0] one interval
    from 0 V, g[..., 1] two intervals, ...).  piece1: (...), the integral from 0 V to g[..., 0];
    pair2: (...), the integral from 0 V to g[..., 1]; dx: the oriented step (step above 0 V,
    -step below), a float32 0-d CuPy array.  Returns (..., m): at g[..., j], j even (an odd count
    of intervals from 0 V), piece1 plus the pairs (dx) from g[..., 0]; j odd (an even count),
    pair2 plus the pairs from g[..., 1].
    """
    s = g[..., :-2] + 4.0 * g[..., 1:-1] + g[..., 2:]      # y0 + 4 y1 + y2 of the pair at each sample
    zero = cp.zeros(g.shape[:-1] + (1,), dtype=cp.float32)
    c_odd = cp.concatenate((zero, cp.cumsum(s[..., 0::2], axis=-1)), axis=-1)
    c_even = cp.concatenate((zero, cp.cumsum(s[..., 1::2], axis=-1)), axis=-1)
    out = cp.empty(g.shape, dtype=cp.float32)
    out[..., 0::2] = piece1[..., None] + dx / 3.0 * c_odd
    out[..., 1::2] = pair2[..., None] + dx / 3.0 * c_even
    return out


def integral_from_zero(f, k0, step, e0, axis=-1):
    """
    Machinery: the oriented integral from 0 V to every sample of f along `axis` (CuPy, float32).

    f: the integrand on consecutive DOS samples along `axis` (any other axes, e.g. tip points
    first), taken as a float32 CuPy array (cp.asarray with dtype float32).  k0 (int): the position
    along `axis` of DOS sample k0, the last sample at or below 0 V.  step: the DOS step,
    (E max - E min)/(NEDOS - 1); e0: the energy of sample k0 (E - E_F), exactly 0.0 when a sample
    is at 0 V.  step and e0 are np.float32 (Python floats are rounded to float32 here, once).  The
    sample at position j along `axis` is at e0 + (j - k0) step: no energy array is taken.

    The samples f must include around k0 (not checked: a position below 0 would wrap around):
      - no sample at 0 V (e0 != 0.0): positions k0 - 1, k0, k0 + 1, k0 + 2 (E-2, E-1, E1, E2),
        so 1 <= k0 <= n - 3 for n samples along `axis`;
      - a sample at 0 V (e0 == 0.0): positions k0 - 2 to k0 + 2 (E-2, E-1, the 0 V sample, E1,
        E2), so 2 <= k0 <= n - 3.

    Returns a float32 CuPy array of f's shape: at each sample E of f, the integral of f from 0 V to
    E, oriented: above 0 V from 0 up to E, below 0 V from 0 down to E (negative for a positive f
    there), 0.0 at a sample exactly at 0 V.  The value at E uses the samples between 0 V and E and
    the stencil (E-2, E-1, E1, E2) for the value at 0 V and the cubic's piece; nothing beyond E is
    integrated.

    The rule.  E1 < E2 < ... are the samples above 0 V, E-1 > E-2 > ... those below, E+-k k
    intervals from 0 V.
      - No sample at 0 V: f(0) is the value at 0 V of the cubic through (E-2, f-2), (E-1, f-1),
        (E1, f1), (E2, f2), and Q+ / Q- are its exact integrals from 0 V to E1 / to E-1, in closed
        Lagrange form with theta = -e0/step.  The interval from 0 V to E1 is e0 + step, from 0 V
        to E-1 it is -e0, every other interval step.
          k odd:  Q+ plus the Simpson pairs (dx = step) over E1, ..., Ek;
          k even: the Simpson pair over (0, E1, E2) for unequal intervals (e0 + step, step; f(0)
                  at 0 V) plus the Simpson pairs over E2, ..., Ek.
        Below 0 V the same, mirrored and oriented: Q-, the pair over (0, E-1, E-2) with intervals
        -e0 and step.
      - A sample exactly at 0 V: that sample is the 0 V point, nothing is inserted, every interval
        is step.  k odd: the exact integral from 0 V to E1 of the cubic through E-1, 0, E1, E2
        (below: from 0 V to E-1, the cubic through E-2, E-1, 0, E1) plus the Simpson pairs from
        E1 (E-1); k even: the Simpson pairs from the 0 V sample.
    The pairs are summed cumulatively outward from 0 V on each side (_chain).
    """
    # A sample exactly at 0 V: e0 rounded to float32 is 0.0 exactly when the float64 e0 is.
    exact = float(e0) == 0.0
    f = cp.asarray(f, dtype=cp.float32)
    step = cp.asarray(step, dtype=cp.float32)
    e0 = cp.asarray(e0, dtype=cp.float32)
    F = cp.moveaxis(f, axis, -1)
    out = cp.empty(F.shape, dtype=cp.float32)
    if exact:
        # Position k0 is 0 V; E+-n is at position k0 +- n, at +-n step.
        fm2, fm1, f0 = F[..., k0 - 2], F[..., k0 - 1], F[..., k0]
        fp1, fp2 = F[..., k0 + 1], F[..., k0 + 2]
        dn = -step
        # From 0 V to E1: the exact integral of the cubic through E-1, 0, E1, E2; from 0 V down to
        # E-1: of the cubic through E-2, E-1, 0, E1 (the same formula mirrored, oriented by dn).
        piece_up = (step / 24.0) * (-fm1 + 13.0 * f0 + 13.0 * fp1 - fp2)
        piece_dn = (dn / 24.0) * (-fp1 + 13.0 * f0 + 13.0 * fm1 - fm2)
        # From 0 V to E2 / E-2: the Simpson pair from the 0 V sample (every interval step).
        pair_up = step / 3.0 * (f0 + 4.0 * fp1 + fp2)
        pair_dn = dn / 3.0 * (f0 + 4.0 * fm1 + fm2)
        out[..., k0] = 0.0
        out[..., k0 + 1:] = _chain(F[..., k0 + 1:], piece_up, pair_up, step)
        out[..., :k0] = cp.flip(_chain(cp.flip(F[..., :k0], axis=-1), piece_dn, pair_dn, dn), axis=-1)
    else:
        # E-2, E-1, E1, E2 at positions k0 - 1, k0, k0 + 1, k0 + 2.  In units of step from E-1 the
        # nodes are at -1, 0, 1, 2 and 0 V at th = -e0/step (0 < th < 1).
        fm2, fm1 = F[..., k0 - 1], F[..., k0]
        fp1, fp2 = F[..., k0 + 1], F[..., k0 + 2]
        th = -e0 / step
        thm1 = th - 1.0
        thp1 = th + 1.0
        thm2 = th - 2.0
        # f(0): the cubic's value at 0 V (Lagrange basis of the nodes -1, 0, 1, 2 at th).
        f0 = (-th * thm1 * thm2 / 6.0 * fm2
              + thp1 * thm1 * thm2 / 2.0 * fm1
              - th * thp1 * thm2 / 2.0 * fp1
              + th * thp1 * thm1 / 6.0 * fp2)
        # Q+: the cubic's exact integral from 0 V to E1 (th to 1 in units of step).
        q_up = (step / 24.0) * (thm1 * thm1 * (th * th - 2.0 * th - 1.0) * fm2
                                - thm1 * thm1 * (3.0 * th * th - 2.0 * th - 13.0) * fm1
                                + thm1 * (3.0 * th * th * th - th * th - 13.0 * th - 13.0) * fp1
                                - thm1 * thm1 * thp1 * thp1 * fp2)
        # Q-: the cubic's exact integral from 0 V down to E-1 (th to 0), oriented.
        q_dn = (step / 24.0) * (th * th * thm2 * thm2 * fm2
                                - th * (3.0 * th * th * th - 8.0 * th * th - 6.0 * th + 24.0) * fm1
                                + th * th * (3.0 * th * th - 4.0 * th - 12.0) * fp1
                                - th * th * (th * th - 2.0) * fp2)
        # From 0 V to E2: the pair over (0, E1, E2), intervals e0 + step and step; from 0 V down
        # to E-2: the pair over (0, E-1, E-2), intervals -e0 and step, oriented.
        pair_up = _pair(f0, fp1, fp2, e0 + step, step)
        pair_dn = -_pair(f0, fm1, fm2, -e0, step)
        out[..., k0 + 1:] = _chain(F[..., k0 + 1:], q_up, pair_up, step)
        out[..., :k0 + 1] = cp.flip(_chain(cp.flip(F[..., :k0 + 1], axis=-1), q_dn, pair_dn, -step),
                                    axis=-1)
    return cp.moveaxis(out, -1, axis)
