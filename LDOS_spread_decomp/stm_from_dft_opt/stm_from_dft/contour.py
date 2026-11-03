"""
The simulator's contour slice (the sm4_Browser Map window's, contour.py there): the iso-line
of the Map Topo at one height as line segments (iso_segments: marching squares on the height
map over the unit cell's fractional grid), placed on the unit cell's lattice at every drawn
image (to_cell).  WSxM's flooding, as outlines: the segments are the perimeters of everything
above the level.  The window (simulator.py) takes the level from the topo colour bar beside the
Map Topo and draws the segments on every energy map.
"""
import numpy as np

# Marching squares.  The corners of a 2 x 2 cell of neighbouring pixels in order around it:
# v0 = (i, j), v1 = (i, j + 1), v2 = (i + 1, j + 1), v3 = (i + 1, j); the cell's case has bit 2**k
# set when corner k is below the level; the edges e0 = v0 v1, e1 = v1 v2, e2 = v2 v3, e3 = v3 v0.
# _CASES: each unambiguous case's segments as pairs of crossed edges (the contour parts the corners
# below from the corners above).  _SADDLE: the two ambiguous cases (5: v0 and v2 below, 10: v1 and
# v3 below) as (the pairs when the cell's centre, the mean of its corners, is below the level: the
# below corners join through the centre; the pairs otherwise), the standard rule.
_CASES = {
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)],
    9: [(0, 2)], 11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(3, 0)],
}
_SADDLE = {
    5: ([(0, 1), (2, 3)], [(3, 0), (1, 2)]),
    10: ([(3, 0), (1, 2)], [(0, 1), (2, 3)]),
}


def iso_segments(img, level):
    """
    The iso-line of a 2D array at `level` as line segments, (n, 2, 2) of
    (row, col) in pixel coordinates (pixel (i, j) at (i, j)), the crossing
    points interpolated linearly along the cell edges (marching squares,
    vectorized over the cells).  Every 2 x 2 cell of neighbouring pixels
    contributes 0, 1 or 2 segments; a cell with a non-finite corner
    contributes none, and so does a cell with no corner below the level.
    A corner at the level counts as above it.  The two ambiguous cases
    (opposite corners below) are split by the mean of the four corners
    (_SADDLE).  Empty, (0, 2, 2), when nothing crosses.
    """
    a = np.asarray(img, dtype=np.float64)
    level = float(level)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2 or not np.isfinite(level):
        return np.zeros((0, 2, 2))
    v = (a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1])                   # v0 .. v3 of every cell
    finite = np.isfinite(v[0]) & np.isfinite(v[1]) & np.isfinite(v[2]) & np.isfinite(v[3])
    with np.errstate(invalid='ignore'):
        below = [c < level for c in v]
        centre_below = 0.25 * (v[0] + v[1] + v[2] + v[3]) < level
    case = below[0] * 1 + below[1] * 2 + below[2] * 4 + below[3] * 8
    ii, jj = np.meshgrid(np.arange(a.shape[0] - 1, dtype=np.float64),
                         np.arange(a.shape[1] - 1, dtype=np.float64), indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        # The crossing along edge k, from corner k towards corner k + 1 (finite on every crossed edge:
        # one corner is below the level, the other is not).
        f = [(level - v[k]) / (v[(k + 1) % 4] - v[k]) for k in range(4)]
    pts = (np.stack([ii, jj + f[0]], axis=-1),                      # e0: along the top row, v0 to v1
           np.stack([ii + f[1], jj + 1.0], axis=-1),                # e1: down the right column, v1 to v2
           np.stack([ii + 1.0, jj + 1.0 - f[2]], axis=-1),          # e2: along the bottom row, v2 to v3
           np.stack([ii + 1.0 - f[3], jj], axis=-1))                # e3: up the left column, v3 to v0
    out = []

    def take(sel, pairs):
        for ea, eb in pairs:
            out.append(np.stack([pts[ea][sel], pts[eb][sel]], axis=1))

    for k, pairs in _CASES.items():
        sel = finite & (case == k)
        if sel.any():
            take(sel, pairs)
    for k, (pairs_below, pairs_above) in _SADDLE.items():
        for pairs, cond in ((pairs_below, centre_below), (pairs_above, ~centre_below)):
            sel = finite & (case == k) & cond
            if sel.any():
                take(sel, pairs)
    if not out:
        return np.zeros((0, 2, 2))
    return np.concatenate(out, axis=0)


def to_cell(segments, shape, lv, offsets=((0.0, 0.0),)):
    """
    Segments of iso_segments ((n, 2, 2) of (row, col) pixel coordinates of
    a height map of `shape` (nv, nu) over the unit cell's fractional grid:
    column j at the fraction j / (nu - 1) of the lattice vector a, row i at
    i / (nv - 1) of b, as the simulator's grid_xy lays its points) placed
    in the plane: (u, v) -> u a + v b with a and b the in-plane parts of
    `lv` (the lattice vectors, rows), repeated at every (x, y) offset in
    `offsets` (the images an energy map shows).  Returns (len(offsets) * n,
    2, 2) of (x, y).
    """
    segs = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    nv, nu = int(shape[0]), int(shape[1])
    lv = np.asarray(lv, dtype=np.float64)
    a, b = lv[0, :2], lv[1, :2]
    u = segs[..., 1] / max(nu - 1, 1)
    v = segs[..., 0] / max(nv - 1, 1)
    xy = u[..., None] * a + v[..., None] * b                                  # (n, 2, 2) of (x, y)
    offs = np.asarray(list(offsets), dtype=np.float64).reshape(-1, 2)
    if not len(segs):
        return np.zeros((0, 2, 2))
    return np.concatenate([xy + o for o in offs], axis=0)
