"""
Machinery: the colour maps of the simulator's panels, the shading of the Map view's LDOS energy maps and
the Contour / Shading box, in one self-contained module (display machinery only, no physics).

What it holds:
- The two colour-map sets by role (SET_NAMES, ROLES, SETS): 'default 1', today's maps, and 'default 2', CET
  maps by their pyqtgraph names.  The isoluminant maps the energy maps use while they are shaded
  (SHADING_CMAPS).  Entries are colour-map names or colour lists; the caller makes its own colour maps from
  them (cmap_spec, map_role, make_cmap).
- The Contour / Shading box: its state and rules (BoxState) and its builder (build_box, giving a ShadeBox),
  which takes the figure and the qtgui controls module from the caller.
- The shading maths, written once for NumPy and CuPy arrays (array_module): the points of a cell's grid
  without its duplicate edge (unique_points); the Height shade (height_shade) and the Relief shade of
  Kovesi's relief.m (relief_shade, light_vector); the colour map through the caller's norm (colormap_rgba,
  cmap_table); the CIELAB L* stretch that applies a shade to the colours (lstar_stretch, shaded_rgba).
- The corners of an RGBA mesh with one quad per point, centred on it and tiled by cell offsets
  (mesh_corners).  The drawing itself is the caller's.

Self-contained: it imports only the standard library and NumPy.  It imports nothing from the simulator and
nothing from qtgui (the box builder is handed the controls module).  CuPy is used only through the arrays
it is given: array_module never imports CuPy, so the module imports and runs where CuPy is not installed.
It is written to be copied into sm4_Browser as it is; each package keeps its own copy.

The CET names ('CET-L3', 'CET-L6', 'CET-R1', 'CET-D1A', 'CET-L19', 'CET-I1', 'CET-D11', 'CET-I3') are the
file names of pyqtgraph's bundled maps (https://github.com/pyqtgraph/pyqtgraph/tree/master/pyqtgraph/colors/maps).
Kovesi describes CET-I1 as "Isoluminant blue to green to orange at lightness 70", CET-I3 as "Isoluminant blue
to pink at lightness 70" and CET-D11 as a "Diverging isoluminant lightblue - lightgrey - orange colour map"
(colorcet.m, https://codeberg.org/peterkovesi/matlabfns/raw/branch/main/Colourmaps/colorcet.m).  The shading
multiplies the colours and never blends them as a weighted sum.  Kovesi's user guide calls transparency
blending "the wrong mechanism to use" and says that with an isoluminant map "there is no interaction between
the colours and the shading" (https://colorcet.com/userguide/index.html).
"""

import math
import sys

import numpy as np


# ---------------------------------------------------------------------- #
#  The colour-map sets                                                     #
# ---------------------------------------------------------------------- #

SET_NAMES = ('default 1', 'default 2')
DEFAULT_SET = 'default 1'

# The roles, by the keys of the simulator's own set table (CMAP_SETS, which these sets replace): 'topo' (every
# topography panel, the Line / Path stripe and the topo colour bar), 'map' (the Map view's energy maps and their
# colour bars), 'heatmap' (the Line and Path LDOS heatmaps, one map for both), 'mag' (the signed spin density of
# a collinear spin-polarized run, in the maps and the heatmaps), '|m|' (|m| of a non-collinear run, never
# negative, in the maps and the heatmaps).
ROLES = ('topo', 'map', 'heatmap', 'mag', '|m|')

# A str is a colour-map name (for the caller's to_cmap).  A tuple is a colour list, to be made into a linear
# ramp through its colours (for the caller's LinearSegmentedColormap.from_list), as default 1's topo map
# (black, firebrick, yellow) and its Map map (midnight blue to light sky blue) are made today.
SETS = {
    'default 1': {
        'topo': ('black', 'firebrick', 'yellow'),
        'map': ('midnightblue', 'lightskyblue'),
        'heatmap': 'nipy_spectral',
        'mag': 'bwr',
        '|m|': 'hot',
    },
    'default 2': {
        'topo': 'CET-L3',
        'map': 'CET-L6',
        'heatmap': 'CET-R1',
        'mag': 'CET-D1A',
        '|m|': 'CET-L19',
    },
}

# The energy maps' colour maps while they are shaded, the same in both sets: isoluminant, so that lightness
# carries only the shading and hue carries the value.
SHADING_CMAPS = {
    'map': 'CET-I1',
    'mag': 'CET-D11',
    '|m|': 'CET-I3',
}


def cmap_spec(set_name, role, shading=False):
    """
    Machinery: the colour-map entry of `role` in the set `set_name`: a colour-map name (str) or a colour list
    (tuple of colour names; make_cmap turns either into the caller's colour map).  With shading=True, the
    isoluminant map of a shaded energy map instead (SHADING_CMAPS; only 'map', 'mag' and '|m|' have one).  ValueError for an unknown set or role, or for a role without a shading map.
    """
    if set_name not in SETS:
        raise ValueError(f"unknown colour-map set {set_name!r} (one of {SET_NAMES})")
    if role not in ROLES:
        raise ValueError(f"unknown colour-map role {role!r} (one of {ROLES})")
    if shading:
        if role not in SHADING_CMAPS:
            raise ValueError(f"the role {role!r} has no shading colour map (only {tuple(SHADING_CMAPS)})")
        return SHADING_CMAPS[role]
    return SETS[set_name][role]


def map_role(diverging, mag):
    """
    Machinery: the role of the energy maps (and of the heatmaps) as they are shown: 'mag' while the signed
    spin density is shown (`diverging`: Mag on, a Mag channel, a signed run), else '|m|' while |m| is shown
    (`mag`: Mag on and a Mag channel), else 'map'.  This is the rule that picks 'bwr', 'hot' or the Map map
    today (the simulator's _map_cmap).
    """
    if diverging:
        return 'mag'
    if mag:
        return '|m|'
    return 'map'


def make_cmap(spec, to_cmap, from_list, name='cmap_shading'):
    """
    Machinery: the caller's colour map of an entry `spec` (cmap_spec).  A name goes through `to_cmap(spec)`.
    A colour list goes through `from_list(name, list(spec))`, a linear ramp through its colours.  A colour list
    must not be given to qtplot's to_cmap, which makes a list into a ListedColormap of its few colours.  With
    qtplot: make_cmap(spec, plt.to_cmap, plt.LinearSegmentedColormap.from_list, name).
    Each call of from_list makes a new colour map, and qtplot's Colormap defines no __eq__, so two calls give
    maps that compare unequal.  A caller that tells a panel's role by its map's identity (the simulator's
    _on_cmap_set: `m is c or m == c`) builds each set's maps once and reuses them.
    """
    if isinstance(spec, str):
        return to_cmap(spec)
    return from_list(name, list(spec))


# ---------------------------------------------------------------------- #
#  The Contour / Shading box                                               #
# ---------------------------------------------------------------------- #

CONTOUR = 'Contour'          # the toggle of the contour slice (its traces on the energy maps, its level line)
SHADING = 'Shading'          # the toggle of the energy maps' shading
SHADE_BY = 'Shading by'      # the Height / Relief radio
HEIGHT = 'Height'
RELIEF = 'Relief'
SHADE_MODES = (HEIGHT, RELIEF)
BOX_TITLE = 'contour / shading'


class BoxState:
    """
    Machinery: the Contour / Shading box's state and its rules.  `contour` and `shading` are the two toggles:
    at most one is on, and both may be off.  Turning Shading on turns Contour off, turning Contour on turns
    Shading off, and turning either off leaves the other as it is.  `shade_by` is 'Height' or 'Relief' and is
    kept while Shading is off.  It starts with Contour on, Shading off and Height.
    """

    def __init__(self, contour=True, shading=False, shade_by=HEIGHT):
        """Machinery: a state; ValueError when both toggles are on or `shade_by` is not one of SHADE_MODES."""
        if contour and shading:
            raise ValueError("Contour and Shading cannot both be on")
        if shade_by not in SHADE_MODES:
            raise ValueError(f"shade_by must be one of {SHADE_MODES}, not {shade_by!r}")
        self.contour = bool(contour)
        self.shading = bool(shading)
        self.shade_by = shade_by

    def set_contour(self, flag):
        """Machinery: Contour on (Shading then off) or off (Shading unchanged); returns the state."""
        self.contour = bool(flag)
        if self.contour:
            self.shading = False
        return self

    def set_shading(self, flag):
        """Machinery: Shading on (Contour then off) or off (Contour unchanged); returns the state."""
        self.shading = bool(flag)
        if self.shading:
            self.contour = False
        return self

    def set_shade_by(self, mode):
        """Machinery: the shading's mode, 'Height' or 'Relief' (ValueError otherwise); returns the state."""
        if mode not in SHADE_MODES:
            raise ValueError(f"shade_by must be one of {SHADE_MODES}, not {mode!r}")
        self.shade_by = mode
        return self

    @property
    def shade_mode(self):
        """Machinery: the mode the energy maps are shaded in now: 'Height' or 'Relief' while Shading is on, else None."""
        return self.shade_by if self.shading else None

    def key(self):
        """Machinery: the state as a tuple (contour, shading, shade_by), for a redraw signature."""
        return (self.contour, self.shading, self.shade_by)

    def copy(self):
        """Machinery: an independent copy of the state."""
        return BoxState(self.contour, self.shading, self.shade_by)

    def __eq__(self, other):
        """Machinery: equal states have equal keys."""
        return isinstance(other, BoxState) and self.key() == other.key()

    def __repr__(self):
        """Machinery: the state as text."""
        return f"BoxState(contour={self.contour}, shading={self.shading}, shade_by={self.shade_by!r})"


class ShadeBox:
    """
    Machinery: the box build_box makes.  `qt` is its framed Qt widget (controls.box), for the caller's layout
    or the strip.  `state` is its BoxState, which build_box's callbacks keep current.  `contour` and
    `shading` are the two toggles (qtgui Check) and `shade_by` is the Height / Relief radio (qtgui Radio).
    The info panel lists their settings as for every other control.
    """

    def __init__(self, qt, state, contour, shading, shade_by):
        """Machinery: the box of the frame `qt`, the state and the three controls (build_box makes it)."""
        self.qt = qt
        self.state = state
        self.contour = contour
        self.shading = shading
        self.shade_by = shade_by

    def sync(self):
        """Machinery: the toggles and the radio set to the state, through the controls' set (which never calls a callback)."""
        self.contour.set(self.state.contour)
        self.shading.set(self.state.shading)
        self.shade_by.set(self.state.shade_by)

    def set_state(self, contour=None, shading=None, shade_by=None):
        """
        Machinery: the box set from code.  Each argument given is applied through the state's rules, in the
        order contour, shading, shade_by, and the controls then show the result.  The change callback is not
        called (as the qtgui controls' set).  Returns the state.
        """
        if contour is not None:
            self.state.set_contour(contour)
        if shading is not None:
            self.state.set_shading(shading)
        if shade_by is not None:
            self.state.set_shade_by(shade_by)
        self.sync()
        return self.state


def _make_control(controls, kind, fig, name, **kw):
    """
    Machinery: one control of the box made as the window's other controls are.  controls.make(fig, name, ...)
    is used when the widgets table names it (controls.SPECS).  Otherwise the kind's class itself is used
    (controls.Check / controls.Radio), which takes the default specification for a name the table lacks.
    """
    specs = getattr(controls, 'SPECS', None)
    if specs is not None and name in specs:
        return controls.make(fig, name, **kw)
    return getattr(controls, kind)(fig, name, **kw)


def build_box(fig, controls, on_change=None, state=None, title=BOX_TITLE, orient='h'):
    """
    Machinery: the Contour / Shading box.  `fig` is the window's qtgui figure and `controls` the qtgui
    controls module (the caller's copy).  The box holds the Contour and Shading toggles (Check) and the
    Height / Relief radio (Radio), framed by controls.box(..., orient=orient, title=title), and starts at
    `state` (default BoxState(): Contour on, Shading off, Height).  A click applies the state's rules and
    sets the other toggle to match, through Check.set, which never fires.  It then calls on_change(state)
    with the state after the rule.  Returns the ShadeBox (its `qt` is the frame).
    """
    st = BoxState() if state is None else state
    chk_contour = _make_control(controls, 'Check', fig, CONTOUR, value=st.contour)
    chk_shading = _make_control(controls, 'Check', fig, SHADING, value=st.shading)
    radio = _make_control(controls, 'Radio', fig, SHADE_BY, value=st.shade_by, labels=list(SHADE_MODES))
    frame = controls.box(chk_contour, chk_shading, radio, orient=orient, title=title)
    sb = ShadeBox(frame, st, chk_contour, chk_shading, radio)

    def _done():
        """Machinery: after a click, the controls set to the state and the caller's callback called with it."""
        sb.sync()
        if on_change is not None:
            on_change(st)

    def _on_contour(flag):
        """Machinery: the Contour toggle clicked."""
        st.set_contour(flag)
        _done()

    def _on_shading(flag):
        """Machinery: the Shading toggle clicked."""
        st.set_shading(flag)
        _done()

    def _on_shade_by(label):
        """Machinery: Height or Relief chosen."""
        st.set_shade_by(label)
        _done()

    chk_contour.on_change(_on_contour)
    chk_shading.on_change(_on_shading)
    radio.on_change(_on_shade_by)
    return sb


# ---------------------------------------------------------------------- #
#  NumPy or CuPy                                                           #
# ---------------------------------------------------------------------- #

def array_module(*arrays):
    """
    Machinery: the array module of the arguments, CuPy or NumPy, for code written once for both (`xp`).
    When the cupy package is already imported (sys.modules), this is cupy.get_array_module(*arrays): "If at
    least one of the arguments is a cupy.ndarray object, the cupy module is returned"
    (https://docs.cupy.dev/en/stable/reference/generated/cupy.get_array_module.html).  Otherwise it is NumPy.
    No CuPy array can exist before the cupy package is imported, because importing a submodule runs its
    parent package first (https://docs.python.org/3/reference/import.html, "Regular packages").  So NumPy
    is then the right answer, and this module never imports CuPy itself.
    """
    cp = sys.modules.get('cupy')
    gam = getattr(cp, 'get_array_module', None) if cp is not None else None
    if gam is None:
        return np
    return gam(*arrays)


def unique_points(values):
    """
    Machinery: the points of a cell's grid without its duplicate edge.  `values` is (N*N,) in the grid's
    order (point i*N + j at the fraction j/(N-1) of the cell vector a and i/(N-1) of b, as the simulator's
    grid_xy), or (N, N) with row i along b and column j along a.  The fractions 0 and 1 are the same points
    of the lattice, so the last row and column repeat the first ones and are left out.  The result is
    (N-1, N-1), row i along b, column j along a, of the same array module (a view).  ValueError for a 1-D
    length that is not a square, an array of another rank, or fewer than 2 points a side.
    """
    xp = array_module(values)
    a = xp.asarray(values)
    if a.ndim == 1:
        n = math.isqrt(int(a.shape[0]))
        if n * n != int(a.shape[0]):
            raise ValueError(f"{a.shape[0]} points are not a square grid")
        a = a.reshape(n, n)
    elif a.ndim != 2:
        raise ValueError(f"values must be (N*N,) or (N, N), not of shape {a.shape}")
    if a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError(f"a grid of shape {a.shape} has no point left without its edge")
    return a[:-1, :-1]


# ---------------------------------------------------------------------- #
#  The shades: Height and Relief                                           #
# ---------------------------------------------------------------------- #

def height_shade(heights):
    """
    Machinery: the Height shade, the heights themselves (rows, cols) as float64 of their array module.  The
    L* stretch (lstar_stretch) then makes the lowest height the darkest and the highest the lightest.
    """
    xp = array_module(heights)
    z = xp.asarray(heights, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"heights must be (rows, cols), not of shape {z.shape}")
    return z.copy()


# Kovesi's derivative5.m, first derivatives ('x', 'y'): the 5-tap prefilter p and derivative d1, as printed
# there (https://codeberg.org/peterkovesi/matlabfns/raw/branch/main/Spatial/derivative5.m):
#   p = [0.037659  0.249153  0.426375  0.249153  0.037659];
#   d1 =[0.109604  0.276691  0.000000 -0.276691 -0.109604];
_D5_P = (0.037659, 0.249153, 0.426375, 0.249153, 0.037659)
_D5_D1 = (0.109604, 0.276691, 0.000000, -0.276691, -0.109604)


def _along(axis, s):
    """Machinery: the index tuple of a 2-D array taking the slice `s` along `axis` (0 or 1) and all of the other axis."""
    return (s, slice(None)) if axis == 0 else (slice(None), s)


def _conv5(z, taps, axis, periodic, xp):
    """
    Machinery: a 1-D convolution of the 2-D array `z` along `axis` with the 5 `taps`, centred, keeping the
    size of `z` (MATLAB's conv2 'same' along one axis): out[k] = sum over m = -2..2 of taps[m + 2] * z[k - m].
    MATLAB defines C(j,k) = sum_p sum_q A(p,q) B(j-p+1,k-q+1), where "p and q run over all values that lead
    to legal subscripts" (https://www.mathworks.com/help/matlab/ref/conv2.html).  Read from that definition,
    a sample outside the array adds nothing, as with periodic=False here.  With periodic=True, the index
    k - m wraps around the axis instead (xp.roll).
    """
    out = xp.zeros_like(z)
    n = int(z.shape[axis])
    for m in range(-2, 3):
        h = float(taps[m + 2])
        if h == 0.0:
            continue
        if periodic:
            out += h * xp.roll(z, m, axis=axis)          # roll(z, m)[k] = z[(k - m) mod n]
        elif m >= 0:
            if n - m > 0:
                out[_along(axis, slice(m, n))] += h * z[_along(axis, slice(0, n - m))]
        else:
            if n + m > 0:
                out[_along(axis, slice(0, n + m))] += h * z[_along(axis, slice(-m, n))]
    return out


def _set_border_replicate(g, b=2):
    """
    Machinery: Kovesi's imsetborder(im, b, 'replicate') on the 2-D array `g`, in place and in his loop order
    (https://codeberg.org/peterkovesi/matlabfns/raw/branch/main/Spatial/imsetborder.m):
        for n = 1:b
           im(n,:,chan) = im(b+1,:,chan);
           im(end-n+1,:,chan) = im(end-b,:,chan);
           im(:,n,chan) = im(:,b+1,chan);
           im(:,end-n+1,chan) = im(:,end-b,chan);
        end
    (MATLAB's 1-based rows n, b+1, end-n+1 and end-b are rows n-1, b, -n and -b-1 here.)  Returns `g`.
    """
    for n in range(1, b + 1):
        g[n - 1, :] = g[b, :]
        g[-n, :] = g[-b - 1, :]
        g[:, n - 1] = g[:, b]
        g[:, -n] = g[:, -b - 1]
    return g


def light_vector(az=45.0, el=45.0, *, panel_turn):
    """
    Machinery: relief.m's light direction (https://www.peterkovesi.com/matlabfns/Geosci/relief.m), as
    (x, y, z) floats in the data frame of a panel drawn turned by `panel_turn`.  relief.m's lines are:
        az = az/180*pi;
        el = el/180*pi;
        I = [cos(el)*sin(az), cos(el)*cos(az), sin(el)];
        I = I./norm(I); % Ensure I is a unit vector
    with "Zero azimuth points upwards and increases clockwise" and the elevation in degrees.  These are taken
    as the screen's (right, up, out of the screen) components.  This is a reading of relief.m: it negates the
    second component's product "to account for the image +ve y increasing downwards", so its second
    component points up the image as shown.  `panel_turn` is the angle in
    degrees, anticlockwise, through which the panel shows its data turned on screen.  It is 0 for a panel
    drawn with data x to the right and y up.  It is 90 for a panel turned by qtplot's Axes.set_rotated (data x
    up, data y to the left).  The screen components are turned back by -panel_turn into data x, y, so the
    light keeps its place on screen.  A caller that keeps the light fixed in the data's x, y passes 0 for
    every panel.  This argument has no default, so the caller makes that choice.
    """
    a = az / 180.0 * math.pi
    e = el / 180.0 * math.pi
    i_r, i_u, i_z = math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)
    nrm = math.sqrt(i_r * i_r + i_u * i_u + i_z * i_z)
    i_r, i_u, i_z = i_r / nrm, i_u / nrm, i_z / nrm
    t = math.radians(float(panel_turn))
    c, s = math.cos(t), math.sin(t)
    return (c * i_r + s * i_u, -s * i_r + c * i_u, i_z)


def relief_shade(heights, step_a, step_b, *, panel_turn, periodic, az=45.0, el=45.0, gradscale=1.0,
                 minval=0.0):
    """
    Machinery: the Relief shade, the Lambertian shading of Kovesi's relief.m
    (https://www.peterkovesi.com/matlabfns/Geosci/relief.m, the same as
    https://codeberg.org/peterkovesi/matlabfns/raw/branch/main/Geosci/relief.m), with its defaults az = 45,
    el = 45, gradscale = 1 and minval = 0.  `heights` is (rows, cols): column j steps along the vector
    `step_a` and row i along `step_b`, both (x, y) in the heights' length unit and need not be at right
    angles.  For the simulator's cell, these are the unique points (unique_points) with step_a = a/(N-1) and
    step_b = b/(N-1).  For an image with x along its columns, they are (dx, 0) and (0, dy) when its row
    index increases along +y (imshow origin='lower'), and (dx, 0) and (0, -dy) for a row-down image
    (origin='upper', the frame relief.m itself takes), whose point (i, j) sits at (j dx, -i dy) in the y-up
    frame used here.  Returns (rows, cols) float64 of the heights' array module.

    relief.m's steps, followed here:
    - Gradients: surfacenormals takes [p,q] = derivative5(im, 'x', 'y'), with 'x' as conv2(p, d1, im,
      'same') and 'y' as conv2(d1, p, im, 'same').  Kovesi's comment is that conv2 "performs a 1D convolution
      down the columns using p then a 1D convolution along the rows using d1".  Here those are the per-step
      gradients along the rows, over the column index j (step_a), and down the columns, over the row index i
      (step_b), each times gradscale (`p = imsetborder(p*gradscale, 2, 'replicate')`).
    - The edges: relief.m has no option for them.  conv2 adds nothing from outside the array, and
      imsetborder's 2-point replicate border follows.  `periodic` has no default, so the caller chooses.
      periodic=False is relief.m's own handling (an image); it needs at least 3 points a side.
      periodic=True is this module's option, not relief.m's, for a grid that repeats with the lattice (the
      simulator's cell): the 5 taps wrap around the cell, and imsetborder is left out because there is no
      border.
    - Gradients in x, y: the per-step gradients are the gradient's components along the step vectors,
      g_a = grad z . step_a and g_b = grad z . step_b.  So grad z is the 2 x 2 solve of those two equations
      (the step vectors must not be parallel).  gradscale multiplies these gradients in height per length of
      x, y.  This is a reading of the addendum's "converted to x and y": relief.m's own gradscale multiplies
      per-pixel gradients, which unit steps (1, 0) and (0, 1) give here.  gradscale = 1 is the slope as
      derivative5 measures it, not the exact slope: its d1 taps give 2(0.276691) + 4(0.109604) = 0.991798
      times the slope of a linear ramp (p sums to 0.999999), and less for features a few grid steps wide
      (the Farid and Simoncelli filter that derivative5.m cites).
    - Normals: relief.m's `mag = sqrt(1 + p.^2 + q.^2); n1 = -p./mag; n2 = -q./mag; n3 = 1./mag;`, each a
      unit vector, here in the data frame (x, y up, z).
    - Shading: relief.m's `shading = I(1)*n1 - I(2)*n2 + I(3)*n3;`, the dot product of the unit normal and
      the light.  The minus is relief.m's "to account for the image +ve y increasing downwards"; in the data
      frame (y up) the dot product is plain.  The light is light_vector(az, el, panel_turn=panel_turn), so a
      turned panel's light keeps its place on screen unless the caller passes 0.
    - Then `shading(shading < minval) = minval;`.  NaN stays NaN.
    """
    xp = array_module(heights)
    z = xp.asarray(heights, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"heights must be (rows, cols), not of shape {z.shape}")
    if not periodic and min(z.shape) < 3:
        raise ValueError(f"relief.m's 2-point border needs at least 3 points a side, not {z.shape}")
    sax, say = float(step_a[0]), float(step_a[1])
    sbx, sby = float(step_b[0]), float(step_b[1])
    det = sax * sby - say * sbx
    if not math.isfinite(det) or det == 0.0:
        raise ValueError("the step vectors are parallel or not finite")
    # derivative5 'x': down the columns (axis 0) with p, then along the rows (axis 1) with d1: along step_a.
    g_a = _conv5(_conv5(z, _D5_P, 0, periodic, xp), _D5_D1, 1, periodic, xp) * float(gradscale)
    # derivative5 'y': down the columns (axis 0) with d1, then along the rows (axis 1) with p: along step_b.
    g_b = _conv5(_conv5(z, _D5_D1, 0, periodic, xp), _D5_P, 1, periodic, xp) * float(gradscale)
    if not periodic:
        g_a = _set_border_replicate(g_a, 2)
        g_b = _set_border_replicate(g_b, 2)
    # grad z from g_a = grad z . step_a and g_b = grad z . step_b.
    zx = (sby * g_a - say * g_b) / det
    zy = (sax * g_b - sbx * g_a) / det
    mag = xp.sqrt(1.0 + zx * zx + zy * zy)
    n1, n2, n3 = -zx / mag, -zy / mag, 1.0 / mag
    lx, ly, lz = light_vector(az, el, panel_turn=panel_turn)
    shading = lx * n1 + ly * n2 + lz * n3
    return xp.where(shading < float(minval), float(minval), shading)


def shade_field(heights, mode, step_a, step_b, *, panel_turn, periodic, az=45.0, el=45.0,
                gradscale=1.0, minval=0.0):
    """
    Machinery: the shade of `heights` (rows, cols) in `mode`.  'Height' gives height_shade(heights) and
    ignores the other arguments.  'Relief' gives relief_shade(heights, step_a, step_b, panel_turn=...,
    periodic=..., az, el, gradscale, minval).  panel_turn and periodic have no default (relief_shade).
    ValueError for another mode.  The shade depends only on the heights, so one shade serves every map of
    the same heights.
    """
    if mode == HEIGHT:
        return height_shade(heights)
    if mode == RELIEF:
        return relief_shade(heights, step_a, step_b, panel_turn=panel_turn, az=az, el=el, gradscale=gradscale,
                            minval=minval, periodic=periodic)
    raise ValueError(f"mode must be one of {SHADE_MODES}, not {mode!r}")


# ---------------------------------------------------------------------- #
#  Colours: the colour map through the caller's norm                       #
# ---------------------------------------------------------------------- #

def cmap_table(cmap):
    """
    Machinery: the (n, 4) float64 RGBA table (NumPy, clipped to 0..1) of a colour map.  A colour-map object
    (callable, with N entries) is sampled at the centres of its N bins, (k + 0.5) / N.  Under qtplot's rule
    (index floor(t * N)) that gives entry k exactly.  An array of (n, 3) or (n, 4) colours is taken as it is,
    with alpha 1 added to RGB rows.
    """
    if callable(cmap) and hasattr(cmap, 'N'):
        n = int(cmap.N)
        tab = np.asarray(cmap((np.arange(n) + 0.5) / n), dtype=np.float64)
    else:
        tab = np.asarray(cmap, dtype=np.float64)
    tab = tab.reshape(-1, tab.shape[-1])
    if tab.shape[1] == 3:
        tab = np.column_stack([tab, np.ones(len(tab))])
    elif tab.shape[1] != 4:
        raise ValueError(f"a colour table needs 3 or 4 columns, not {tab.shape[1]}")
    return np.clip(tab, 0.0, 1.0)


def _norm_xp(values, norm, xp):
    """
    Machinery: qtplot's Normalize or PowerNorm, transcribed for any array module, for CuPy arrays (qtplot's
    norms take NumPy only).  Normalize gives t = (x - vmin) / (vmax - vmin), with a denominator of 1 when
    vmax == vmin, clipped to 0..1 only with clip.  PowerNorm then sets t < 0 to 0 and gives t ** gamma.  NaN
    stays NaN.  A vmin or vmax of None is taken from the finite values, as qtplot's autoscale does, without
    changing the norm.  The norm is recognised by its class name (Normalize, PowerNorm); TypeError for
    any other.
    """
    kind = type(norm).__name__
    if kind not in ('Normalize', 'PowerNorm'):
        raise TypeError(f"a {kind} cannot be applied to CuPy arrays here (only Normalize and PowerNorm)")
    a = xp.asarray(values, dtype=np.float64)
    vmin, vmax = getattr(norm, 'vmin', None), getattr(norm, 'vmax', None)
    if vmin is None or vmax is None:
        ok = xp.isfinite(a)
        has = bool(ok.any())
        if vmin is None:
            vmin = float(xp.min(xp.where(ok, a, np.inf))) if has else 0.0
        if vmax is None:
            vmax = float(xp.max(xp.where(ok, a, -np.inf))) if has else 1.0
    vmin, vmax = float(vmin), float(vmax)
    d = vmax - vmin
    t = (a - vmin) / (d if d != 0 else 1.0)
    if getattr(norm, 'clip', False):
        t = xp.clip(t, 0.0, 1.0)
    if kind == 'PowerNorm':
        t = xp.where(t < 0, 0.0, t)
        t = xp.power(t, float(norm.gamma))
    return t


def _lookup_xp(t, table, xp):
    """
    Machinery: qtplot's Colormap call transcribed for any array module: index clip(floor(t * N), 0, N - 1)
    into the (N, 4) `table`; a non-finite t gives (0, 0, 0, 0).  Returns (..., 4) float64.
    """
    tab = xp.asarray(table, dtype=np.float64)
    n = int(tab.shape[0])
    bad = ~xp.isfinite(t)
    idx = xp.clip(xp.floor(xp.where(bad, 0.0, t) * n), 0, n - 1).astype(np.intp)
    out = tab[idx]
    return xp.where(bad[..., None], 0.0, out)


def colormap_rgba(values, cmap, norm):
    """
    Machinery: the colours of `values` through the caller's `norm` and colour map `cmap`, the colour map
    applied before any shading.  Returns (..., 4) float64 RGBA of the values' array module.
    - NumPy values: the caller's own norm, then its colour map, are called (an array of colours given as
      `cmap` goes through qtplot's lookup rule, _lookup_xp).
    - CuPy values: qtplot's Normalize / PowerNorm rules (_norm_xp) and its colour-map lookup (_lookup_xp) on
      cmap_table(cmap), on the GPU.
    For the energy maps the norm is the one they are drawn with today: PowerNorm(gamma, v_min, v_max), or
    Normalize(v_min, v_max) for the signed spin density.
    """
    xp = array_module(values)
    if xp is np:
        t = np.ma.filled(np.ma.asarray(norm(np.asarray(values)), dtype=np.float64), np.nan)
        if callable(cmap) and hasattr(cmap, 'N'):
            return np.asarray(cmap(t), dtype=np.float64)
        return _lookup_xp(t, cmap_table(cmap), np)
    t = _norm_xp(values, norm, xp)
    return _lookup_xp(t, cmap_table(cmap), xp)


# ---------------------------------------------------------------------- #
#  The L* stretch (CIE L*, sRGB)                                           #
# ---------------------------------------------------------------------- #

# The relative luminance Y of linear sRGB (D65, white Y = 1): the Y row of CSS Color 4's lin_sRGB_to_XYZ,
# "[  87098 /  409605, 175762 / 245763,   12673 /  175545 ]" (https://www.w3.org/TR/css-color-4/#color-conversion-code).
_Y_R, _Y_G, _Y_B = 87098 / 409605, 175762 / 245763, 12673 / 175545
# CIE's L* threshold (6/29)^3 and CSS Color 4's κ = 24389/27 (29^3/3^3) for the inverse.
_LSTAR_EPS = (6 / 29) ** 3
_KAPPA = 24389 / 27


def _srgb_decode(c, xp):
    """
    Machinery: sRGB values (0..1) to linear sRGB, CSS Color 4's lin_sRGB
    (https://www.w3.org/TR/css-color-4/#color-conversion-code): |v| <= 0.04045 gives v / 12.92, else
    sign * ((|v| + 0.055) / 1.055) ** 2.4.
    """
    a = xp.abs(c)
    return xp.where(a <= 0.04045, c / 12.92, xp.sign(c) * xp.power((a + 0.055) / 1.055, 2.4))


def _srgb_encode(c, xp):
    """
    Machinery: linear sRGB to sRGB values, CSS Color 4's gam_sRGB
    (https://www.w3.org/TR/css-color-4/#color-conversion-code): |v| > 0.0031308 gives
    sign * (1.055 * |v| ** (1/2.4) - 0.055), else 12.92 * v.
    """
    a = xp.abs(c)
    return xp.where(a > 0.0031308, xp.sign(c) * (1.055 * xp.power(a, 1 / 2.4) - 0.055), 12.92 * c)


def _y_to_lstar(y, xp):
    """
    Machinery: CIE L* of the relative luminance Y (Yn = 1, the sRGB white), CIE e-ILV 17-23-076
    (https://cie.co.at/eilvterm/17-23-076): L* = 116 f(Y/Yn) - 16, f(t) = t^(1/3) if t > (6/29)^3, else
    (841/108) t + 4/29.
    """
    f = xp.where(y > _LSTAR_EPS, xp.cbrt(y), (841 / 108) * y + 4 / 29)
    return 116.0 * f - 16.0


def _lstar_to_y(lstar, xp):
    """
    Machinery: the relative luminance Y (Yn = 1) of a CIE L*, the inverse of _y_to_lstar: CSS Color 4's
    Lab_to_XYZ Y line, `Lab[0] > κ * ε ? Math.pow((Lab[0]+16)/116,3) : Lab[0]/κ`, with κ = 24389/27 and
    ε = 216/24389 (https://www.w3.org/TR/css-color-4/#color-conversion-code).
    """
    return xp.where(lstar > _KAPPA * _LSTAR_EPS, xp.power((lstar + 16.0) / 116.0, 3), lstar / _KAPPA)


def lstar_stretch(rgba, shade, floor=0.0):
    """
    Machinery: a shade applied to colours by their CIELAB L*.  Each colour's L* is scaled by a factor f, and
    its linear RGB is scaled by one factor, which keeps the colour's chromaticity.  f runs evenly from
    `floor` (0 to 1) at the lowest finite shade value over the given points up to 1 at the highest:
    f = floor + (1 - floor) * (s - s_min) / (s_max - s_min), or 1 everywhere when every shade value is the
    same.  The scaling is exact.  The colour's L* comes from its relative luminance Y (CSS Color 4's sRGB
    decode and Y row, CIE's L*).  The target f * L* goes back to a luminance Y_t (CSS Color 4's inverse).  The
    linear RGB is scaled by Y_t / Y (0 for a black colour) and encoded back to sRGB (CSS Color 4's
    gam_sRGB).  Since f <= 1 the result stays in gamut; it is clipped to 0..1 against rounding.
    `rgba` is (..., 3 or 4), sRGB 0..1, and alpha is kept.  `shade` has the leading shape, from height_shade
    or relief_shade.  A point whose shade is not finite becomes (0, 0, 0, 0) for RGBA input, not drawn;
    for RGB input, which has no alpha, it becomes (0, 0, 0), black.  Returns float64 of
    the arrays' array module (CuPy if either is CuPy).  With floor = 0 the lowest shade value is black.
    """
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError(f"floor must be within 0..1, not {floor}")
    xp = array_module(rgba, shade)
    c = xp.asarray(rgba, dtype=np.float64)
    s = xp.asarray(shade, dtype=np.float64)
    if c.ndim < 1 or c.shape[-1] not in (3, 4) or tuple(c.shape[:-1]) != tuple(s.shape):
        raise ValueError(f"rgba {tuple(c.shape)} and shade {tuple(s.shape)} do not match")
    ok = xp.isfinite(s)
    if not bool(ok.any()):
        return xp.zeros_like(c)
    s_lo = float(xp.min(xp.where(ok, s, np.inf)))
    s_hi = float(xp.max(xp.where(ok, s, -np.inf)))
    fl = float(floor)
    if s_hi > s_lo:
        f = fl + (1.0 - fl) * (s - s_lo) / (s_hi - s_lo)
    else:
        f = xp.ones_like(s)
    lin = _srgb_decode(c[..., :3], xp)
    y = _Y_R * lin[..., 0] + _Y_G * lin[..., 1] + _Y_B * lin[..., 2]
    y_t = _lstar_to_y(f * _y_to_lstar(y, xp), xp)
    pos = y > 0
    k = xp.where(pos, y_t / xp.where(pos, y, 1.0), 0.0)
    rgb = xp.clip(_srgb_encode(lin * k[..., None], xp), 0.0, 1.0)
    out = xp.concatenate([rgb, c[..., 3:]], axis=-1)
    return xp.where(ok[..., None], out, 0.0)


def shaded_rgba(values, shade, cmap, norm, floor=0.0):
    """
    Machinery: the shaded colours of one map: colormap_rgba(values, cmap, norm), the colour map first with
    the caller's norm, then lstar_stretch(..., shade, floor).  `values` and `shade` are (rows, cols) of the
    same points (unique_points of the slice and of the heights).  Returns (rows, cols, 4) float64 of the
    arrays' array module; the caller moves CuPy colours to the host (cupy.asnumpy) to draw them.
    """
    return lstar_stretch(colormap_rgba(values, cmap, norm), shade, floor)


# ---------------------------------------------------------------------- #
#  The mesh: one quad per point                                            #
# ---------------------------------------------------------------------- #

def mesh_corners(step_a, step_b, shape, offsets=((0.0, 0.0),), origin=(0.0, 0.0)):
    """
    Machinery: the corners of an RGBA mesh with one quad per point of a (rows, cols) grid, for a 'flat'
    pcolormesh per offset.  Point (i, j) sits at origin + j * step_a + i * step_b; its quad has the corners
    (j -+ 1/2) * step_a + (i -+ 1/2) * step_b around it, so each quad is centred on its point.  For the
    simulator's cell: step_a = a/(N-1), step_b = b/(N-1), shape (N-1, N-1) (unique_points).  The quads then
    tile the cell exactly, from the fraction -1/(2(N-1)) to 1 - 1/(2(N-1)) along each vector, and the
    lattice offsets (the 2 x 2 cells of an energy map) tile the plane with no gap.
    Returns X, Y as NumPy float64 of shape (len(offsets), rows + 1, cols + 1).  Colour [i, j] of a
    (rows, cols, 4) array fills the quad X[k, i:i+2, j:j+2], Y[k, i:i+2, j:j+2] of offset k.
    """
    rows, cols = int(shape[0]), int(shape[1])
    sa = np.asarray(step_a, dtype=np.float64).reshape(2)
    sb = np.asarray(step_b, dtype=np.float64).reshape(2)
    o = np.asarray(origin, dtype=np.float64).reshape(2)
    jj, ii = np.meshgrid(np.arange(cols + 1) - 0.5, np.arange(rows + 1) - 0.5)
    bx = o[0] + jj * sa[0] + ii * sb[0]
    by = o[1] + jj * sa[1] + ii * sb[1]
    offs = np.asarray(list(offsets), dtype=np.float64).reshape(-1, 2)
    X = np.stack([bx + ox for ox, _ in offs]) if len(offs) else np.zeros((0, rows + 1, cols + 1))
    Y = np.stack([by + oy for _, oy in offs]) if len(offs) else np.zeros((0, rows + 1, cols + 1))
    return X, Y
