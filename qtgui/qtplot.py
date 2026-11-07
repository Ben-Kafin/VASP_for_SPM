"""
qtplot: the drawing and window layer of the suite's windows on PySide6 + pyqtgraph (the
sm4_Browser layer, section 9 of the conversion guide, carried into VASP_for_SPM as the one
shared layer at the repository root, package `qtgui`).

The suite's windows were written against matplotlib's pyplot: figures laid
out in figure fractions, axes, artists returned by plot / imshow / axhline,
mpl-style mouse events (button_press_event, motion_notify_event, ...,
pick_event) and colour maps by name.  This module provides that interface
(the subset the windows use) on Qt, so every window keeps its layout, its
math, and its interaction while all drawing, events, and widgets are
PySide6 / pyqtgraph:

  figure, subplots, show, close, fignum_exists, get_fignums, gcf, pause,
  rcParams, get_cmap                         the pyplot functions
  Figure, FigureCanvas, Axes                 a figure is a Qt window with one
                                             pyqtgraph scene; every axes is a
                                             bare ViewBox placed in its cell of
                                             the window's Qt layout (or, given
                                             a rect, at that figure fraction),
                                             its tick axes outside it (as in
                                             matplotlib), every tick drawn
                                             where the view box maps its
                                             value (_Axis, the tick rule)
  gridspec.GridSpec, GridSpecFromSubplotSpec the subplot grids
  Normalize, PowerNorm, TwoSlopeNorm,        colour normalizations
  LogNorm, SymLogNorm
  Colormap, LinearSegmentedColormap,         colour maps; to_cmap(c) turns any
  ListedColormap, to_cmap                    matplotlib colour-map name, any
                                             matplotlib Colormap object, or a
                                             pyqtgraph ColorMap into one
  LineCollection, PolyCollection,            collections (LineCollection with a colour
  PathCollection, Affine2D, ScalarMappable   per segment: array + cmap + norm), the
                                             scatter collection, transforms
  Line2D, Patch                              also as legend proxies (made without an axes)
  cm, ticker                                 plt.cm.<name>(x), plt.cm.ScalarMappable;
                                             ticker.FuncFormatter / FormatStrFormatter for
                                             ax.yaxis.set_major_formatter (a colour bar's too)
  Axes.tricontourf                           scattered points interpolated linearly (scipy)
                                             onto a raster, banded by the levels, drawn as one
                                             image through a lookup table; offsets= draws the
                                             same raster moved (never re-imaged)
  Axes.vlines / hlines / add_line / set /    the further matplotlib calls the suite's windows
  set_facecolor / get_lines / scatter        make (scatter: a PathCollection with per-point
                                             colours, set_offsets, pick with event.ind)

Speed: images are pyqtgraph ImageItems coloured through a lookup table (a
new gamma or colour limit only rebuilds the table), quad meshes are filled
as one path per colour and cached as a pixmap, many curves of one colour are
one item, and mouse motion is coalesced (a slow redraw never queues up a
backlog of motion events).

Colour maps: wherever the package takes a colour map (the if-main inputs
cmap / cmap_map, line_window_kwargs['cmap'], ...), a matplotlib name
('jet', 'viridis', 'RdBu_r', ...), a matplotlib Colormap object, or a
pyqtgraph ColorMap may be given; to_cmap converts it once (cached) into the
lookup tables pyqtgraph draws with.  matplotlib is used
only to read the colour values of a map given by name or object and, in
mplexport, to write the publication files; it never draws on screen.

Layout (automatic layout: the one rule for every window; nothing is placed by
hand): a window's panels and controls sit in one Qt layout installed on its
canvas (Figure.set_layout).  A panel is a cell of that layout (Figure.cell,
a PanelCell): its axes takes the cell's rectangle less its tick axes and
title (Axes(cell=...)); a row of panels keeps one height (Figure.align_row)
and a spatial panel its x:y ratio (set_aspect / set_box_aspect); a figure
text may sit in a cell (text(..., cell=)).  The controls (controls.py) are
Qt widgets in the same layout.  The matplotlib-style calls (add_subplot,
subplots) put their axes in an automatic grid of cells (Figure.layout_root);
an axes given an explicit rect keeps that figure fraction.  When a window is
too small for its controls' text, the text shrinks to CONTROL_FIT_MIN of its
zoom (Figure._fit_controls, fits_controls) and then the window's smallest
size is raised, so nothing overlaps and no window scrolls.  A saved figure is
the window as it is on screen: nothing is laid out again for the file.

Text size: every window declares its design size as a DESIGN constant
(inches, the figsize it opens with, at DPI logical pixels per inch) and
its text zoom is window_zoom(W, H, W0, H0) = min(W / W0, H / H0) limited to
ZOOM_MIN .. ZOOM_MAX (Figure.font_zoom, set in Figure._relayout on every
resize): plots, ticks, legends, the controls of controls.py, the info panel
and the toolbar follow it.  Shape: a figure window taller than it is wide
is in portrait (Figure.portrait, set_portrait): its controls show their
short labels (widgets.SHORT_LABELS) and a panel title that is a file name
shows only the file's date-and-time stamp (file_stamp; short_title is the
one rule for every title).  The details cut from a title go to the
window's info panel (FigureManager.set_info), a dock at the bottom of the
window.

Also here: the events the windows connect to (FigureCanvas.mpl_connect:
MouseEvent, KeyEvent, PickEvent, CloseEvent), the navigation toolbar
(NavigationToolbar: Home, Pan, Zoom, Save), the saving of a figure, exactly
what the window shows (Figure.savefig: PDF / EPS / SVG through mplexport, the
plots at their on-screen places and sizes and the full info panel below them,
the controls left out with their places empty; an SVG of the whole window as
Qt draws it, controls included, as shown (export_vector); PNG / TIFF, the
window rendered at WINDOW_IMAGE_DPI (Figure.window_image); or the curves as
CSV), the info panel's settings section (the settings of the window's
controls, controls.settings_text), the registry of the Qt
controls in a figure's layout (Figure._register_control, used by controls.py),
and the crash log (CRASH_LOG, log_note, watchdog) for hangs and native
crashes inside Qt.
"""

import os
import re
import sys
import math
import html
import itertools
import contextlib
import traceback
import types

import numpy as np
import matplotlib                       # noqa: F401  required: the vector saves (mplexport); no window runs without it



def _choose_binding():
    """
    The Qt binding to draw with: the one already imported in this Python
    session when there is one (Spyder's IPython console runs its Qt event
    loop with PyQt5 or PyQt6, and IPython refuses a second binding), else
    the one named by PYQTGRAPH_QT_LIB, else PySide6 (then PyQt6, PyQt5,
    PySide2, whichever is installed).
    """
    for lib in ('PySide6', 'PyQt6', 'PyQt5', 'PySide2'):
        if lib in sys.modules:
            return lib
    if os.environ.get('PYQTGRAPH_QT_LIB'):
        return os.environ['PYQTGRAPH_QT_LIB']
    import importlib.util
    for lib in ('PySide6', 'PyQt6', 'PyQt5', 'PySide2'):
        if importlib.util.find_spec(lib) is not None:
            return lib
    return 'PySide6'


os.environ['PYQTGRAPH_QT_LIB'] = _choose_binding()
import pyqtgraph as pg                                          # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets               # noqa: E402

Qt = QtCore.Qt

# A native crash or a hang inside Qt kills the kernel without a Python message; faulthandler
# writes the Python stack of such a crash to this file (and to the console when it can).
CRASH_LOG = os.path.join(os.path.expanduser('~'), 'vasp_for_spm_crash.log')
try:
    import faulthandler
    _crash_file = open(CRASH_LOG, 'w')
    faulthandler.enable(file=_crash_file, all_threads=True)
except Exception:
    _crash_file = None

def log_note(text):
    """A line in the crash log (the step the program was at when Qt failed)."""
    if _crash_file is not None:
        try:
            _crash_file.write(str(text) + "\n")
            _crash_file.flush()
        except Exception:
            pass


class watchdog:
    """
    with qtplot.watchdog('what'): ...  -- `what` is noted in the crash log on entry, and while the
    block runs longer than `after` seconds the Python stack of every thread is written every
    `after` seconds (faulthandler.dump_traceback_later) to the console (sys.__stderr__), or to
    CRASH_LOG when there is no console, so a step that hangs on a file names itself.  Blocks
    nest: leaving an inner block re-arms the enclosing one with its own `what` and `after`.
    """

    def __init__(self, what, after=15.0):
        self.what, self.after = what, float(after)

    _stack = []

    @staticmethod
    def _arm(after):
        try:
            import faulthandler
            out = sys.__stderr__ if sys.__stderr__ is not None else _crash_file
            if out is not None:
                faulthandler.dump_traceback_later(after, repeat=True, file=out)
        except Exception:
            pass

    def __enter__(self):
        log_note(self.what)
        watchdog._stack.append(self)
        self._arm(self.after)
        return self

    def __exit__(self, *exc):
        try:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        if watchdog._stack and watchdog._stack[-1] is self:
            watchdog._stack.pop()
        if watchdog._stack:                         # the enclosing block is watched again
            log_note(watchdog._stack[-1].what)
            self._arm(watchdog._stack[-1].after)
        return False


_LIMIT = 1e15            # coordinates beyond this (or not finite) are never handed to Qt's painter


def _finite_range(a, b):
    """A drawable range from (a, b): None when not finite; widened when (nearly) empty."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(a) and np.isfinite(b)) or max(abs(a), abs(b)) > _LIMIT:
        return None
    if b < a:
        a, b = b, a
    c = 0.5 * (a + b)
    min_span = max(abs(c) * 1e-9, 1e-12)
    if b - a < min_span:
        a, b = c - min_span, c + min_span
    return a, b


def _evpos(ev):
    """A mouse or wheel event's position in widget pixels (Qt 6: position(); Qt 5: pos() / posF())."""
    if hasattr(ev, 'position'):
        return QtCore.QPointF(ev.position())
    if hasattr(ev, 'posF'):
        return QtCore.QPointF(ev.posF())
    return QtCore.QPointF(ev.pos())


def _check_qt_binding():
    """
    PySide6 6.12.0 releases one reference to None on every Qt call that
    returns nothing; before Python 3.12 (None not yet immortal) a painting
    window then aborts with 'Fatal Python error: none_dealloc' after a few
    thousand calls.  Measured here once at import; a warning names the cure.
    """
    if sys.version_info >= (3, 12) or pg.Qt.QT_LIB != 'PySide6':
        return
    try:
        p = QtGui.QPainterPath()
        before = sys.getrefcount(None)
        for _ in range(64):
            p.setFillRule(Qt.FillRule.WindingFill)
        lost = before - sys.getrefcount(None)
    except Exception:
        return
    if lost >= 32:
        import PySide6
        print(f"qtplot WARNING: this PySide6 ({PySide6.__version__}) loses a reference to None on every Qt call "
              f"under Python {sys.version_info.major}.{sys.version_info.minor}; the windows will crash with "
              "'Fatal Python error: none_dealloc' after a while.  Use Python 3.12 or newer, or "
              "pip install \"PySide6<6.12\".")


_check_qt_binding()

# ---------------------------------------------------------------------- #
#  Configuration                                                           #
# ---------------------------------------------------------------------- #

WINDOW_BG = '#f0f0f0'     # the window colour around the white panels
FONT_SCALE = 1.25         # every text (plots, legends, controls) this much larger than matplotlib's points
DPI = 100.0               # logical pixels per inch: figure sizes, widget sizes, and font points as matplotlib's
WINDOW_IMAGE_DPI = 720    # the resolution of the image formats (Figure.window_image): the window rendered at WINDOW_IMAGE_DPI / DPI
ANTIALIAS = True          # antialiased curves (False draws faster, with jagged lines)
ZOOM_MIN, ZOOM_MAX = 0.5, 4.0   # the text zoom of any window: its size relative to its design size, within these
CONTROL_FIT_MIN = 0.75          # the controls' text shrinks to at most this fraction of the zoom to fit its window (every window: Figure._fit_controls)
CONTROL_FIT_SLACK = 8           # px: a larger control text is taken only when the controls fit with this to spare (no flip-flop at the edge)


def fits_controls(need, room, zoom, current):
    """
    The text-fit rule's test (every window: Figure._fit_controls): the controls'
    need (w, h) fits the room (w, h) at `zoom`; a zoom above the one now applied (`current`) must fit
    with CONTROL_FIT_SLACK to spare, so a window at the edge keeps its text size instead of switching
    between two sizes on every layout pass.
    """
    s = CONTROL_FIT_SLACK if current is not None and zoom > current + 1e-9 else 0
    return need[0] <= room[0] - s and need[1] <= room[1] - s
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k', antialias=ANTIALIAS)


def screen_scale():
    """
    How many times larger than 96 dpi the primary screen's logical resolution is: figure windows
    open this much larger, and their text follows (font_zoom).  1 under Qt 6's default high-DPI
    scaling, which passes the scale factor through unrounded and reports 96 dpi; the monitor's own
    dpi, e.g. 1.5 for a 4K screen at 150 %, under Qt 5 without scaling or with Qt's scaling off,
    and then per monitor (QScreen::logicalDotsPerInch; the tick axes, _Axis, are drawn so that no
    such dpi difference between monitors reaches them).
    """
    try:
        scr = QtGui.QGuiApplication.primaryScreen()
        return max(1.0, float(scr.logicalDotsPerInch()) / 96.0) if scr is not None else 1.0
    except Exception:
        return 1.0


def window_zoom(W, H, W0, H0):
    """
    The one text-size rule of every window: the text
    zoom is the window's size relative to its design size (W0, H0), the smaller of the width ratio
    and the height ratio, limited to ZOOM_MIN to ZOOM_MAX.  Every text of a window (plots, ticks,
    legends, controls, info panels, toolbar icons) is multiplied by it.  All four sizes are in the
    same unit (logical pixels here); 1.0 when a size is not a number, zero, or not finite.
    """
    try:
        z = min(float(W) / float(W0), float(H) / float(H0))
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    if not np.isfinite(z):
        return 1.0
    return float(min(max(z, ZOOM_MIN), ZOOM_MAX))


def pt2px(pt):
    """Points to logical pixels at DPI (matplotlib's point: 1/72 inch)."""
    return float(pt) * DPI / 72.0


_EPOCH = [0]            # the layout pass (new_measure_pass): a PanelCell's room is measured once per pass (PanelCell._decor)


def new_measure_pass():
    """A new layout pass: every panel cell measures its room (its axes' tick axes and title, Qt calls) again at its next size question (PanelCell._decor, pass_cached).  The arrangement, the map block, the figure's relayout and an axes' placement start one; so does anything that changes what a pass measures (a panel's shape, a control's font or label, a band's items)."""
    _EPOCH[0] += 1


def pass_cached(obj, key, compute):
    """
    `compute()`'s value for `obj` under `key`, kept for the rest of the current layout pass
    (new_measure_pass drops it): a measurement the arrangement asks hundreds of times in one
    pass (a cell's shape, a crop panel's box and band sizes, a band's size hint) is made once.
    """
    c = obj.__dict__.get('_pass_cache')
    if c is None or c[0] != _EPOCH[0]:
        c = obj.__dict__['_pass_cache'] = (_EPOCH[0], {})
    d = c[1]
    if key not in d:
        d[key] = compute()
    return d[key]


def total_min_hfw(lay, w):
    """
    QLayout.totalMinimumHeightForWidth(w): the smallest height the layout needs at width w (Qt's
    qlayout.cpp: minimumHeightForWidth, which holds the layout's own contents margins, plus the
    parent widget's margins when the layout is that widget's own, plus a menu bar's height).  Some
    PySide6 builds do not expose it: then minimumHeightForWidth with the parent widget's contents
    margins added (the figure's layout has no menu bar).
    """
    f = getattr(lay, 'totalMinimumHeightForWidth', None)
    if f is not None:
        return int(f(int(w)))
    side = top = 0
    parent = lay.parentWidget()
    if parent is not None and parent.layout() is lay:
        m = parent.contentsMargins()
        side, top = m.left() + m.right(), m.top() + m.bottom()
    return int(lay.minimumHeightForWidth(int(w) - side)) + top


# matplotlib's named font sizes in points (relative to the 10 pt default, factors of 1.2).
_FONT_SIZES = {'xx-small': 5.79, 'x-small': 6.94, 'small': 8.33, 'medium': 10.0, 'large': 12.0,
               'x-large': 14.4, 'xx-large': 17.28, 'larger': 12.0, 'smaller': 8.33}


def font_pt(size, default=10.0):
    """A matplotlib font size (points, a named size, or None) as points; `default` for None or an unknown name."""
    if size is None:
        return float(default)
    if isinstance(size, str):
        return _FONT_SIZES.get(size, float(default))
    return float(size)


_font_cache = {}


def qfont(size=10.0, family=None, weight=None, zoom=1.0):
    """A QFont of `size` points at DPI (cached), times FONT_SCALE and a figure's zoom (Figure.font_zoom)."""
    px = max(1, int(round(pt2px(font_pt(size)) * FONT_SCALE * float(zoom))))
    key = (px, family, weight)
    f = _font_cache.get(key)
    if f is None:
        if family in ('monospace', 'mono', 'courier'):
            f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        else:
            f = QtGui.QFont()
            if family and family not in ('sans-serif', 'serif'):
                f.setFamily(family)
        f.setPixelSize(px)
        if weight in ('bold', 'heavy', 'semibold', 'demibold') or (isinstance(weight, (int, float)) and weight >= 600):
            f.setBold(True)
        _font_cache[key] = f
    return QtGui.QFont(f)


# ---------------------------------------------------------------------- #
#  Colours                                                                 #
# ---------------------------------------------------------------------- #

# matplotlib's default colour cycle ('C0' .. 'C9', rcParams['axes.prop_cycle'], the colour a plot
# call without a colour takes), its single-letter colours, and the 'tab:<name>' names of TAB10.
TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']
_BASE = {'b': (0.0, 0.0, 1.0), 'g': (0.0, 0.5, 0.0), 'r': (1.0, 0.0, 0.0), 'c': (0.0, 0.75, 0.75),
         'm': (0.75, 0.0, 0.75), 'y': (0.75, 0.75, 0.0), 'k': (0.0, 0.0, 0.0), 'w': (1.0, 1.0, 1.0)}
_TAB_NAMES = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
_rgba_cache = {}              # (colour string, alpha) -> (r, g, b, a): to_rgba runs for every pen and brush


def to_rgba(c, alpha=None):
    """
    A matplotlib colour specification as an (r, g, b, a) tuple of floats in 0..1 (`alpha` overrides
    the specification's own when given): a string ('none' = transparent, a single letter, 'C0' ..
    'C9', 'tab:blue', '#rgb' / '#rgba' / '#rrggbb' / '#rrggbbaa', a grey level '0.5', or any colour
    name Qt knows), a QColor, or a sequence of 3 or 4 numbers (0..1, or 0..255 when one exceeds 1).
    Strings are cached.  Raises ValueError for an unknown string.
    """
    if isinstance(c, str):
        key = (c, alpha)
        hit = _rgba_cache.get(key)
        if hit is not None:
            return hit
        s = c.strip().lower()
        if s in ('none', ''):
            rgba = (0.0, 0.0, 0.0, 0.0)
        elif s in _BASE:
            rgba = _BASE[s] + (1.0,)
        elif s[:1] == 'c' and s[1:].isdigit():
            rgba = to_rgba(TAB10[int(s[1:]) % len(TAB10)])
        elif s.startswith('tab:') and s[4:] in _TAB_NAMES:
            rgba = to_rgba(TAB10[_TAB_NAMES.index(s[4:])])
        elif s.startswith('#'):
            h = s[1:]
            if len(h) in (3, 4):
                h = ''.join(ch * 2 for ch in h)
            vals = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
            rgba = tuple(vals) + ((1.0,) if len(vals) == 3 else ())
        else:
            try:
                v = float(s)
                if not 0.0 <= v <= 1.0:
                    raise ValueError
                rgba = (v, v, v, 1.0)
            except ValueError:
                q = QtGui.QColor(s.replace(' ', ''))
                if not q.isValid():
                    raise ValueError(f"unknown colour {c!r}")
                rgba = q.getRgbF()
        if alpha is not None:
            rgba = tuple(rgba[:3]) + (float(alpha),)
        rgba = tuple(float(v) for v in rgba)
        _rgba_cache[key] = rgba
        return rgba
    if isinstance(c, QtGui.QColor):
        rgba = c.getRgbF()
    else:
        a = [float(v) for v in np.asarray(c, dtype=np.float64).ravel()]
        if max(a[:3]) > 1.0:
            a = [v / 255.0 for v in a]
        rgba = tuple(a[:3]) + ((a[3],) if len(a) > 3 else (1.0,))
    if alpha is not None:
        rgba = tuple(rgba[:3]) + (float(alpha),)
    return tuple(float(v) for v in rgba)


def to_rgb(c):
    """matplotlib.colors.to_rgb: the (r, g, b) floats of any colour specification."""
    return tuple(to_rgba(c)[:3])


def qcolor(c, alpha=None):
    """A matplotlib colour specification (to_rgba) as a QColor, the components clipped to 0..1."""
    r, g, b, a = to_rgba(c, alpha)
    return QtGui.QColor.fromRgbF(min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0),
                                 min(max(a, 0.0), 1.0))


_LINESTYLES = {'-': Qt.PenStyle.SolidLine, 'solid': Qt.PenStyle.SolidLine,
               '--': Qt.PenStyle.DashLine, 'dashed': Qt.PenStyle.DashLine,
               ':': Qt.PenStyle.DotLine, 'dotted': Qt.PenStyle.DotLine,
               '-.': Qt.PenStyle.DashDotLine, 'dashdot': Qt.PenStyle.DashDotLine}


def mkpen(color, lw=1.5, ls='-', alpha=None):
    """A cosmetic QPen of `lw` points, matplotlib line style `ls`; None for no line."""
    if ls in (None, 'none', 'None', ' ', '') or color is None:
        return None
    if isinstance(color, str) and color.strip().lower() == 'none':
        return None
    pen = QtGui.QPen(qcolor(color, alpha))
    pen.setWidthF(max(pt2px(lw), 0.5))
    pen.setCosmetic(True)
    if isinstance(ls, (tuple, list)) and len(ls) == 2 and isinstance(ls[1], (tuple, list)) and len(ls[1]) >= 2:
        # matplotlib's (offset, (on, off, ...)) dash tuple, lengths in units of the line width (Qt's dash pattern unit).
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([float(v) for v in ls[1]])
        pen.setDashOffset(float(ls[0] or 0.0))
    else:
        pen.setStyle(_LINESTYLES.get(ls, Qt.PenStyle.SolidLine))
    if pen.style() != Qt.PenStyle.SolidLine:
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    return pen


def mkbrush(color, alpha=None):
    """A solid QBrush of `color` (to_rgba); None for no fill (color None or 'none')."""
    if color is None or (isinstance(color, str) and color.strip().lower() == 'none'):
        return None
    return QtGui.QBrush(qcolor(color, alpha))


# ---------------------------------------------------------------------- #
#  Normalizations                                                          #
# ---------------------------------------------------------------------- #

class Normalize:
    """Linear map of [vmin, vmax] onto [0, 1] (matplotlib.colors.Normalize); NaN stays NaN."""

    def __init__(self, vmin=None, vmax=None, clip=False):
        self.vmin = None if vmin is None else float(vmin)
        self.vmax = None if vmax is None else float(vmax)
        self.clip = bool(clip)

    def autoscale_None(self, A):
        a = np.asarray(np.ma.filled(np.ma.asarray(A, dtype=np.float64), np.nan), dtype=np.float64)
        ok = np.isfinite(a)
        if self.vmin is None:
            self.vmin = float(np.min(a[ok])) if ok.any() else 0.0
        if self.vmax is None:
            self.vmax = float(np.max(a[ok])) if ok.any() else 1.0

    def scaled(self):
        return self.vmin is not None and self.vmax is not None

    def _lin(self, value):
        a = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.scaled():
            self.autoscale_None(a)
        d = self.vmax - self.vmin
        out = (a - self.vmin) / (d if d != 0 else 1.0)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out

    def __call__(self, value):
        return self._lin(value)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.vmin + t * (self.vmax - self.vmin)


class PowerNorm(Normalize):
    """((x - vmin) / (vmax - vmin)) ** gamma, values below vmin at 0 (matplotlib.colors.PowerNorm)."""

    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = float(gamma)

    def __call__(self, value):
        t = self._lin(value)
        with np.errstate(invalid='ignore'):
            t = np.where(t < 0, 0.0, t)
            return np.power(t, self.gamma)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.vmin + np.power(np.clip(t, 0.0, None), 1.0 / self.gamma) * (self.vmax - self.vmin)


class TwoSlopeNorm(Normalize):
    """Two linear pieces: vmin -> 0, vcenter -> 0.5, vmax -> 1 (matplotlib.colors.TwoSlopeNorm)."""

    def __init__(self, vcenter, vmin=None, vmax=None):
        super().__init__(vmin, vmax)
        self.vcenter = float(vcenter)

    def __call__(self, value):
        a = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.scaled():
            self.autoscale_None(a)
        lo, c, hi = self.vmin, self.vcenter, self.vmax
        out = np.where(a < c, 0.5 * (a - lo) / ((c - lo) or 1.0), 0.5 + 0.5 * (a - c) / ((hi - c) or 1.0))
        return np.where(np.isfinite(a), out, np.nan)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        lo, c, hi = self.vmin, self.vcenter, self.vmax
        return np.where(t < 0.5, lo + 2 * t * (c - lo), c + 2 * (t - 0.5) * (hi - c))


class LogNorm(Normalize):
    """
    (log x - log vmin) / (log vmax - log vmin), matplotlib.colors.LogNorm (its LogScale
    transform): a value <= 0 is NaN (not drawn), the autoscale reads the positive values only.
    `log` marks the norm so that a colour bar ticks its decades.
    """

    log = True

    def autoscale_None(self, A):
        a = np.asarray(np.ma.filled(np.ma.asarray(A, dtype=np.float64), np.nan), dtype=np.float64)
        ok = np.isfinite(a) & (a > 0)
        if self.vmin is None:
            self.vmin = float(np.min(a[ok])) if ok.any() else 1e-3
        if self.vmax is None:
            self.vmax = float(np.max(a[ok])) if ok.any() else 1.0

    def __call__(self, value):
        a = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.scaled():
            self.autoscale_None(a)
        lo, hi = float(self.vmin), float(self.vmax)
        if not (lo > 0 and hi > lo):
            return np.full(a.shape, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (np.log(a) - math.log(lo)) / (math.log(hi) - math.log(lo))
        out = np.where(a > 0, out, np.nan)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        lo, hi = float(self.vmin), float(self.vmax)
        return lo * np.power(hi / lo, t)


class SymLogNorm(Normalize):
    """
    matplotlib.colors.SymLogNorm (its SymmetricalLogTransform): linear within -linthresh ..
    linthresh, logarithmic in `base` beyond it, the linear half-range as wide as `linscale`
    decades; the transformed value scaled between the transformed vmin and vmax.
    """

    def __init__(self, linthresh, linscale=1.0, vmin=None, vmax=None, clip=False, base=10.0):
        super().__init__(vmin, vmax, clip)
        if linthresh <= 0 or linscale <= 0 or base <= 1:
            raise ValueError("SymLogNorm: linthresh and linscale must be positive and base larger than 1")
        self.linthresh, self.linscale, self.base = float(linthresh), float(linscale), float(base)

    def _trf(self, a):
        adj = self.linscale / (1.0 - 1.0 / self.base)
        lb = math.log(self.base)
        a = np.asarray(a, dtype=np.float64)
        ab = np.abs(a)
        inside = ab <= self.linthresh
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.sign(a) * self.linthresh * (adj - math.log(self.linthresh) / lb + np.log(ab) / lb)
        return np.where(inside, a * adj, out)

    def _inv(self, a):
        adj = self.linscale / (1.0 - 1.0 / self.base)
        a = np.asarray(a, dtype=np.float64)
        inv_thresh = float(self._trf(np.array(self.linthresh)))
        ab = np.abs(a)
        inside = ab <= inv_thresh
        with np.errstate(over='ignore', invalid='ignore'):
            out = np.sign(a) * self.linthresh * np.exp((ab / self.linthresh - adj) * math.log(self.base))
        return np.where(inside, a / adj, out)

    def __call__(self, value):
        a = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.scaled():
            self.autoscale_None(a)
        t0, t1 = float(self._trf(np.array(self.vmin))), float(self._trf(np.array(self.vmax)))
        out = (self._trf(a) - t0) / ((t1 - t0) or 1.0)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return np.where(np.isfinite(a), out, np.nan)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        t0, t1 = float(self._trf(np.array(self.vmin))), float(self._trf(np.array(self.vmax)))
        return self._inv(t0 + t * (t1 - t0))


# ---------------------------------------------------------------------- #
#  Colour maps                                                             #
# ---------------------------------------------------------------------- #

class Colormap:
    """
    A colour map as matplotlib uses one: called with values in [0, 1] it
    gives RGBA floats (bytes=True: uint8), NaN giving the transparent 'bad'
    colour; lut(n) is the uint8 lookup table pyqtgraph draws with.
    """

    def __init__(self, name, table):
        self.name = str(name)
        self._table = np.clip(np.asarray(table, dtype=np.float64).reshape(-1, 4), 0.0, 1.0)
        self.N = len(self._table)
        self._lut_cache = {}

    def __call__(self, X, alpha=None, bytes=False):
        scalar = np.ndim(X) == 0
        x = np.asarray(np.ma.filled(np.ma.asarray(X, dtype=np.float64), np.nan), dtype=np.float64)
        bad = ~np.isfinite(x)
        idx = np.clip(np.floor(np.where(bad, 0.0, x) * self.N), 0, self.N - 1).astype(np.intp)
        out = self._table[idx]
        if alpha is not None:
            out = out.copy()
            out[..., 3] = float(alpha)
        if bad.any():
            out = out.copy()
            out[bad] = 0.0
        if bytes:
            out = (out * 255.0 + 0.5).astype(np.uint8)
        if scalar:
            return tuple(out.reshape(4))
        return out

    def lut(self, n=256):
        """The (n, 4) uint8 lookup table sampled uniformly over [0, 1]."""
        t = self._lut_cache.get(n)
        if t is None:
            t = self((np.arange(n) + 0.5) / n, bytes=True)
            self._lut_cache[n] = t
        return t

    def resampled(self, n):
        return Colormap(self.name, self((np.arange(n) + 0.5) / n))

    def reversed(self):
        return Colormap(self.name + '_r', self._table[::-1])

    def to_pg(self):
        """The same map as a pyqtgraph ColorMap."""
        return pg.ColorMap(np.linspace(0.0, 1.0, self.N), (self._table * 255).astype(np.uint8))

    def __repr__(self):
        return f"<qtplot.Colormap {self.name} ({self.N} colours)>"


class LinearSegmentedColormap(Colormap):
    @staticmethod
    def from_list(name, colors, N=256):
        """Linear interpolation between colours (or (position, colour) pairs), matplotlib's from_list."""
        cols = list(colors)

        def is_pair(c):
            return (isinstance(c, (tuple, list)) and len(c) == 2 and isinstance(c[0], (int, float, np.floating))
                    and not isinstance(c[1], (int, float, np.floating)))
        if cols and all(is_pair(c) for c in cols):
            pos = np.array([float(c[0]) for c in cols], dtype=np.float64)
            rgba = np.array([to_rgba(c[1]) for c in cols])
        else:
            pos = np.linspace(0.0, 1.0, len(cols))
            rgba = np.array([to_rgba(c) for c in cols])
        x = np.linspace(0.0, 1.0, int(N))
        table = np.column_stack([np.interp(x, pos, rgba[:, k]) for k in range(4)])
        return LinearSegmentedColormap(name, table)


class ListedColormap(Colormap):
    """A colour map of the given colours as they are, one table entry each (matplotlib's ListedColormap)."""

    def __init__(self, colors, name='from_list'):
        super().__init__(name, np.array([to_rgba(c) for c in colors]))


def _segment_table(data, N=256):
    """An (N, 4) table from matplotlib's segment data ({'red' / 'green' / 'blue': rows of (x, y0, y1)}; y0 is used), alpha 1."""
    x = np.linspace(0.0, 1.0, N)
    cols = []
    for key in ('red', 'green', 'blue'):
        seg = np.asarray(data[key], dtype=np.float64)
        cols.append(np.interp(x, seg[:, 0], seg[:, 1]))
    return np.column_stack(cols + [np.ones(N)])


_JET = {'red': ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
        'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
        'blue': ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
_cmap_cache = {}


def _builtin_cmap(name):
    """The maps known without matplotlib: jet, gray / grey, and pyqtgraph's own (viridis, magma, ...)."""
    base, rev = (name[:-2], True) if name.endswith('_r') else (name, False)
    if base == 'jet':
        cm = Colormap('jet', _segment_table(_JET))
    elif base in ('gray', 'grey', 'Greys_r'):
        x = np.linspace(0, 1, 256)
        cm = Colormap(base, np.column_stack([x, x, x, np.ones(256)]))
    else:
        pgm = pg.colormap.get(base)
        cm = Colormap(base, pgm.getLookupTable(0.0, 1.0, 256, alpha=True) / 255.0)
    return cm.reversed() if rev else cm


def to_cmap(c):
    """
    Any colour-map specification as a Colormap: a qtplot Colormap as it is;
    a matplotlib colour-map name or Colormap object (its colours read through
    matplotlib); a pyqtgraph ColorMap; a list of colours.  None
    gives matplotlib's default, viridis.  Names are converted once (cached).
    """
    if c is None:
        c = 'viridis'
    if isinstance(c, Colormap):
        return c
    if isinstance(c, pg.ColorMap):
        return Colormap(getattr(c, 'name', 'pyqtgraph') or 'pyqtgraph', c.getLookupTable(0.0, 1.0, 256, alpha=True) / 255.0)
    if isinstance(c, str):
        hit = _cmap_cache.get(c)
        if hit is not None:
            return hit
        cm = None
        try:
            import matplotlib
            try:
                mcm = matplotlib.colormaps[c]
            except AttributeError:                      # matplotlib < 3.5
                from matplotlib import cm as _cm
                mcm = _cm.get_cmap(c)
            cm = Colormap(c, mcm(np.linspace(0.0, 1.0, max(int(getattr(mcm, 'N', 256)), 2))))
        except ImportError:
            pass
        except (KeyError, ValueError):
            pass
        if cm is None:
            try:
                cm = _builtin_cmap(c)
            except Exception as exc:
                raise ValueError(f"unknown colour map {c!r}") from exc
        _cmap_cache[c] = cm
        return cm
    if callable(c) and hasattr(c, 'N'):               # a matplotlib Colormap object
        return Colormap(getattr(c, 'name', 'matplotlib'), c(np.linspace(0.0, 1.0, max(int(c.N), 2))))
    if isinstance(c, (list, tuple)):
        return ListedColormap(c)
    raise ValueError(f"not a colour map: {c!r}")


get_cmap = to_cmap


def norm_lut(cmap, norm, vmin, vmax, n=1024):
    """
    The lookup table that colours data on [vmin, vmax] (pyqtgraph levels)
    through a normalization and a colour map: bin i (data values from
    vmin + i d to vmin + (i + 1) d) takes cmap(norm(its centre)).  A gamma
    or a new norm only rebuilds this table; the image data stay as they are.
    A plain Normalize (or None) is linear, so the table is the colour map's
    own lut(n).  An empty or non-finite range is widened to [vmin, vmin + 1].
    """
    cmap = to_cmap(cmap)
    vmin, vmax = float(vmin), float(vmax)
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        vmax = vmin + 1.0
    if type(norm) is Normalize or norm is None:
        return cmap.lut(n)
    centres = vmin + (np.arange(n) + 0.5) * (vmax - vmin) / n
    return cmap(np.asarray(norm(centres), dtype=np.float64), bytes=True)


class ScalarMappable:
    """A norm and a colour map with an array (the colour bar's source), matplotlib's ScalarMappable."""

    def __init__(self, norm=None, cmap=None):
        self.norm = norm if norm is not None else Normalize()
        self.cmap = to_cmap(cmap)
        self._A = None
        self.colorbar = None

    def set_array(self, A):
        self._A = np.asarray(np.ma.filled(np.ma.asarray(A, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.norm.scaled():
            self.norm.autoscale_None(self._A)

    def get_array(self):
        return self._A

    def set_norm(self, norm):
        self.norm = norm
        self.changed()

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self.changed()

    def get_clim(self):
        return self.norm.vmin, self.norm.vmax

    def changed(self):
        if self.colorbar is not None:
            try:
                self.colorbar.update_normal(self)
            except Exception:
                pass


class _CmNamespace:
    """
    matplotlib's `cm` as the windows use it: plt.cm.<name>(x) colours x in [0, 1] through the
    map of that name (plt.cm.tab10(0.3)), plt.cm.get_cmap(name), plt.cm.ScalarMappable.
    """

    ScalarMappable = ScalarMappable

    @staticmethod
    def get_cmap(name=None, lut=None):
        return to_cmap(name)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return to_cmap(name)


cm = _CmNamespace()


class FuncFormatter:
    """ticker.FuncFormatter: a tick label from func(value, position) (Axes.xaxis / yaxis.set_major_formatter)."""

    def __init__(self, func):
        self.func = func

    def __call__(self, x, pos=None):
        return str(self.func(x, pos))


class FormatStrFormatter:
    """ticker.FormatStrFormatter: a tick label from an old-style format string ('%+.4f')."""

    def __init__(self, fmt):
        self.fmt = str(fmt)

    def __call__(self, x, pos=None):
        return self.fmt % x


class _IgnoredTicker:
    """A matplotlib locator or formatter the layer takes and ignores (a LogNorm colour bar ticks its decades on its own)."""

    def __init__(self, *a, **k):
        pass


ticker = types.SimpleNamespace(FuncFormatter=FuncFormatter, FormatStrFormatter=FormatStrFormatter,
                               LogLocator=_IgnoredTicker, LogFormatterSciNotation=_IgnoredTicker,
                               MaxNLocator=_IgnoredTicker, ScalarFormatter=_IgnoredTicker,
                               NullLocator=_IgnoredTicker, MultipleLocator=_IgnoredTicker)


# ---------------------------------------------------------------------- #
#  Geometry helpers                                                        #
# ---------------------------------------------------------------------- #

class Bbox:
    """A rectangle (x0, y0, width, height), matplotlib's Bbox as the package reads it."""

    def __init__(self, x0, y0, width, height):
        self.x0, self.y0, self.width, self.height = float(x0), float(y0), float(width), float(height)

    @property
    def bounds(self):
        return (self.x0, self.y0, self.width, self.height)

    @property
    def x1(self):
        return self.x0 + self.width

    @property
    def y1(self):
        return self.y0 + self.height

    @property
    def xmin(self):
        return min(self.x0, self.x1)

    @property
    def ymin(self):
        return min(self.y0, self.y1)

    @property
    def xmax(self):
        return max(self.x0, self.x1)

    @property
    def ymax(self):
        return max(self.y0, self.y1)

    def contains(self, x, y):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    @classmethod
    def from_bounds(cls, x0, y0, w, h):
        return cls(x0, y0, w, h)

    def __repr__(self):
        return f"Bbox({self.x0:.4g}, {self.y0:.4g}, {self.width:.4g}, {self.height:.4g})"


class Affine2D:
    """A 2D affine transform built as matplotlib's (rotate_deg, translate, scale), added to transData."""

    def __init__(self, matrix=None):
        self._mtx = np.eye(3) if matrix is None else np.array(matrix, dtype=np.float64)

    def rotate_deg(self, deg):
        a = math.radians(float(deg))
        c, s = math.cos(a), math.sin(a)
        self._mtx = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ self._mtx
        return self

    def rotate(self, rad):
        return self.rotate_deg(math.degrees(rad))

    def translate(self, tx, ty):
        self._mtx = np.array([[1, 0, float(tx)], [0, 1, float(ty)], [0, 0, 1.0]]) @ self._mtx
        return self

    def scale(self, sx, sy=None):
        sy = sx if sy is None else sy
        self._mtx = np.array([[float(sx), 0, 0], [0, float(sy), 0], [0, 0, 1.0]]) @ self._mtx
        return self

    def get_matrix(self):
        return self._mtx.copy()

    def transform(self, pts):
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return p @ self._mtx[:2, :2].T + self._mtx[:2, 2]

    def to_qt(self):
        m = self._mtx
        return QtGui.QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])

    def __add__(self, other):
        return _Composite(self, other)


class _Composite:
    """
    `first + then`: a transform followed by another, as matplotlib composes them
    (Affine2D() ... + ax.transData); AxesImage.set_transform reads the Affine2D out of it.
    """

    def __init__(self, first, then):
        self.first, self.then = first, then

    def __add__(self, other):
        return _Composite(self, other)


class _Transform:
    """transData / transAxes / transFigure: transform(points) gives display pixels (origin bottom left)."""

    def __init__(self, kind, owner):
        self.kind = kind
        self.owner = owner

    def transform(self, pts):
        p = np.asarray(pts, dtype=np.float64)
        shape = p.shape
        p = p.reshape(-1, 2)
        if self.kind == 'figure':
            W, H = self.owner._canvas_size()
            out = np.column_stack([p[:, 0] * W, p[:, 1] * H])
        else:
            ax = self.owner
            x0, y0, w, h = ax._px_rect()
            W, H = ax.figure._canvas_size()
            if self.kind == 'axes':
                fx, fy = p[:, 0], p[:, 1]
            else:
                if getattr(ax, '_rotated', False):
                    p = p[:, ::-1]                       # a rotated axes: data (x, y) sits at view (y, x)
                (xa, xb), (ya, yb) = ax._view_range()
                fx = (p[:, 0] - xa) / ((xb - xa) or 1.0)
                fy = (p[:, 1] - ya) / ((yb - ya) or 1.0)
                if ax.vb.state['xInverted']:
                    fx = 1.0 - fx
                if ax.vb.state['yInverted']:
                    fy = 1.0 - fy
            out = np.column_stack([x0 + fx * w, H - (y0 + h) + fy * h])
        return out.reshape(shape)

    def inverted(self):
        return _Inverse(self)

    def __add__(self, other):
        return _Composite(self, other)


class _Inverse:
    """transData.inverted() / transAxes.inverted(): display pixels (origin bottom left) back to data values or axes fractions."""

    def __init__(self, t):
        self.t = t

    def transform(self, pts):
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ax = self.t.owner
        x0, y0, w, h = ax._px_rect()
        W, H = ax.figure._canvas_size()
        fx = (p[:, 0] - x0) / (w or 1.0)
        fy = (p[:, 1] - (H - (y0 + h))) / (h or 1.0)
        if self.t.kind == 'axes':
            return np.column_stack([fx, fy])
        if ax.vb.state['xInverted']:                 # an inverted axis runs the other way (transData's inverse)
            fx = 1.0 - fx
        if ax.vb.state['yInverted']:
            fy = 1.0 - fy
        (xa, xb), (ya, yb) = ax._view_range()
        out = np.column_stack([xa + fx * (xb - xa), ya + fy * (yb - ya)])
        return out[:, ::-1] if getattr(ax, '_rotated', False) else out      # view (y, x) back to data (x, y)


# ---------------------------------------------------------------------- #
#  Fast graphics items                                                     #
# ---------------------------------------------------------------------- #

class QuadItem(pg.GraphicsObject):
    """
    Filled quadrilaterals, each one colour: quads (n, 4, 2) in data units and
    either colour indices into a lookup table or RGBA colours per quad.
    Every colour is one QPainterPath filled at once (no per-quad Python
    loop when painting), the result recorded in a QPicture and the item
    cached as a pixmap, so redraws of the view (a marker dragged over a
    heatmap) blit instead of refilling.
    """

    def __init__(self, quads=None, colors=None):
        super().__init__()
        self._quads = np.zeros((0, 4, 2))
        self._groups = []          # (QColor, QPainterPath)
        self._picture = None
        self._bounds = QtCore.QRectF()
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        if quads is not None:
            self.setData(quads, colors)

    def setData(self, quads, colors):
        """quads (n, 4, 2); colors (n, 4) uint8 RGBA (one per quad; alpha 0 = not drawn)."""
        q = np.asarray(quads, dtype=np.float64).reshape(-1, 4, 2)
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
        with np.errstate(invalid='ignore'):
            ok = np.all(np.isfinite(q) & (np.abs(q) < _LIMIT), axis=(1, 2)) & (c[:, 3] > 0)
        q, c = q[ok], c[ok]
        self.prepareGeometryChange()
        self._quads = q
        self._colors = c                    # kept for the publication export (mplexport)
        if len(q):
            xs, ys = q[..., 0], q[..., 1]
            self._bounds = QtCore.QRectF(float(xs.min()), float(ys.min()), float(xs.max() - xs.min()),
                                         float(ys.max() - ys.min()))
        else:
            self._bounds = QtCore.QRectF()
        groups = []
        if len(q):
            key = c.view(np.uint32).ravel()
            order = np.argsort(key, kind='stable')
            ks = key[order]
            starts = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
            ends = np.concatenate([starts[1:], [len(ks)]])
            for s, e in zip(starts, ends):
                idx = order[s:e]
                pts = q[idx]                                         # (m, 4, 2)
                closed = np.concatenate([pts, pts[:, :1, :]], axis=1)  # (m, 5, 2)
                x = closed[..., 0].ravel()
                y = closed[..., 1].ravel()
                conn = np.ones(x.shape, dtype=np.int32)
                conn[4::5] = 0
                path = pg.functions.arrayToQPath(x, y, connect=conn, finiteCheck=False)
                path.setFillRule(Qt.FillRule.WindingFill)
                r, g, b, a = (int(v) for v in c[idx[0]])
                groups.append((QtGui.QColor(r, g, b, a), path))
        self._groups = groups
        self._picture = None
        self.update()

    def _render(self):
        pic = QtGui.QPicture()
        p = QtGui.QPainter(pic)
        p.setPen(Qt.PenStyle.NoPen)
        for color, path in self._groups:
            p.fillPath(path, QtGui.QBrush(color))
        p.end()
        self._picture = pic

    EXPORTING = False        # set while a vector file is written: cells drawn with a hairline of their colour

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        if QuadItem.EXPORTING:
            # In a PDF / SVG viewer adjacent fills are antialiased separately and show seams; a
            # hairline of each cell's own colour closes them.
            for color, path in self._groups:
                pen = QtGui.QPen(color)
                pen.setWidthF(0.6)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(QtGui.QBrush(color))
                p.drawPath(path)
            return
        if self._picture is None:
            self._render()
        self._picture.play(p)

    def boundingRect(self):
        return QtCore.QRectF(self._bounds)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """The quads' extent along axis `ax` (0 x, 1 y), for pyqtgraph's auto range; (None, None) without quads."""
        if not len(self._quads):
            return (None, None)
        v = self._quads[..., ax]
        return (float(v.min()), float(v.max()))


class SegmentsItem(pg.GraphicsObject):
    """
    Polylines, each with its own colour and width (a LineCollection with a colour per segment:
    the simulator's tip-height stripe, the matcher's coloured sticks): one QPainterPath per
    (colour, width) group stroked with a cosmetic pen into a QPicture (QuadItem's pattern), so
    painting never loops over segments in Python.
    """

    def __init__(self):
        super().__init__()
        self._segs = []
        self._groups = []          # (QPen, QPainterPath)
        self._picture = None
        self._bounds = QtCore.QRectF()
        self._max_w = 0.0

    def setData(self, segments, colors, widths, style='-'):
        """segments: polylines (m_i, 2); colors (n, 4) uint8 RGBA; widths in points (one, or one per segment)."""
        segs = [np.asarray(sg, dtype=np.float64).reshape(-1, 2) for sg in segments]
        n = len(segs)
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
        if len(c) == 1 and n > 1:
            c = np.repeat(c, n, axis=0)
        w = np.broadcast_to(np.asarray(widths, dtype=np.float64).ravel(), (n,)) if n else np.zeros(0)
        self.prepareGeometryChange()
        self._segs = segs
        pts = np.vstack(segs) if segs else np.zeros((0, 2))
        with np.errstate(invalid='ignore'):
            ok = np.all(np.isfinite(pts) & (np.abs(pts) < _LIMIT), axis=1)
        pts = pts[ok]
        if len(pts):
            self._bounds = QtCore.QRectF(float(pts[:, 0].min()), float(pts[:, 1].min()),
                                         float(pts[:, 0].max() - pts[:, 0].min()), float(pts[:, 1].max() - pts[:, 1].min()))
        else:
            self._bounds = QtCore.QRectF()
        groups = {}
        for sg, col, lw in zip(segs, c, w):
            if col[3] == 0 or len(sg) < 2:
                continue
            groups.setdefault((tuple(int(v) for v in col), float(lw)), []).append(sg)
        self._groups = []
        self._max_w = 0.0
        for (col, lw), parts in groups.items():
            xs, ys, conn = [], [], []
            for sg in parts:
                with np.errstate(invalid='ignore'):
                    sg = np.where(np.isfinite(sg) & (np.abs(sg) < _LIMIT), sg, np.nan)
                xs.append(sg[:, 0])
                ys.append(sg[:, 1])
                k = np.ones(len(sg), dtype=np.int32)
                k[-1] = 0
                conn.append(k)
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            path = pg.functions.arrayToQPath(x, y, connect=np.concatenate(conn), finiteCheck=True)
            pen = QtGui.QPen(QtGui.QColor(*col))
            pen.setWidthF(max(pt2px(lw), 0.5))
            pen.setCosmetic(True)
            pen.setStyle(_LINESTYLES.get(style, Qt.PenStyle.SolidLine))
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            self._groups.append((pen, path))
            self._max_w = max(self._max_w, pen.widthF())
        self._picture = None
        self.update()

    def _render(self):
        pic = QtGui.QPicture()
        p = QtGui.QPainter(pic)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for pen, path in self._groups:
            p.setPen(pen)
            p.drawPath(path)
        p.end()
        self._picture = pic

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, ANTIALIAS)
        if self._picture is None:
            self._render()
        self._picture.play(p)

    def boundingRect(self):
        # The data's extent plus half the widest stroke, which a cosmetic pen draws in pixels.
        r = QtCore.QRectF(self._bounds)
        if self._max_w:
            try:
                pw, ph = self.pixelWidth(), self.pixelHeight()
            except Exception:
                pw = ph = 0.0
            r = r.adjusted(-0.5 * self._max_w * pw, -0.5 * self._max_w * ph, 0.5 * self._max_w * pw, 0.5 * self._max_w * ph)
        return r

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """The segments' extent along axis `ax` (0 x, 1 y), for pyqtgraph's auto range."""
        if self._bounds.isNull():
            return (None, None)
        return (self._bounds.left(), self._bounds.right()) if ax == 0 else (self._bounds.top(), self._bounds.bottom())


class _TextItem(pg.TextItem):
    """pyqtgraph's TextItem with an extra pixel offset (matplotlib's 'offset points')."""

    def __init__(self, *args, offset=(0.0, 0.0), **kw):
        self._offset = QtCore.QPointF(float(offset[0]), -float(offset[1]))
        super().__init__(*args, **kw)
        self.textItem.document().setDocumentMargin(1)

    def updateTextPos(self):
        super().updateTextPos()
        self.textItem.setPos(self.textItem.pos() + self._offset)


class _LegendSample(pg.GraphicsWidget):
    """One legend sample: a short line in the artist's pen and its marker (no click toggling)."""

    def __init__(self, pen=None, symbol=None, size=6, spen=None, sbrush=None, fill=None):
        super().__init__()
        self.pen, self.symbol, self.size, self.spen, self.sbrush, self.fill = pen, symbol, size, spen, sbrush, fill
        self.setFixedWidth(26)
        self.setFixedHeight(14)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 26, 14)

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if self.fill is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.fill)
            p.drawRect(QtCore.QRectF(3, 2, 20, 10))
        if self.pen is not None:
            p.setPen(self.pen)
            p.drawLine(QtCore.QPointF(1, 7), QtCore.QPointF(25, 7))
        if self.symbol is not None:
            p.translate(13, 7)
            pg.graphicsItems.ScatterPlotItem.drawSymbol(p, self.symbol, self.size,
                                                        self.spen if self.spen is not None else QtGui.QPen(Qt.PenStyle.NoPen),
                                                        self.sbrush if self.sbrush is not None else QtGui.QBrush(Qt.BrushStyle.NoBrush))


def _tick_strings(vals, spacing):
    """
    Tick labels with the decimals the spacing needs (0.25 -> 2), the same for every label
    (matplotlib); a negative sign is the minus sign U+2212, and -0 is written as 0.
    """
    spacing = abs(float(spacing)) or 1.0
    d = max(0, int(math.ceil(-math.log10(spacing) - 1e-9)))
    while d < 8 and abs(spacing * 10 ** d - round(spacing * 10 ** d)) > 1e-6 * max(1.0, spacing * 10 ** d):
        d += 1
    out = []
    for v in vals:
        t = f"{v:.{d}f}"
        if float(t) == 0.0:
            t = f"{0.0:.{d}f}"
        out.append(t.replace('-', '\u2212'))
    return out


_GREEK = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε', 'varepsilon': 'ε', 'zeta': 'ζ',
          'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π',
          'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ', 'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ',
          'omega': 'ω', 'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ',
          'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω', 'AA': 'Å', 'times': '×', 'pm': '±', 'cdot': '·', 'infty': '∞',
          'langle': '⟨', 'rangle': '⟩', 'leq': '≤', 'geq': '≥', 'neq': '≠', 'approx': '≈', 'propto': '∝',
          'rightarrow': '→', 'leftarrow': '←', 'to': '→', 'hbar': 'ħ', 'perp': '⊥', 'parallel': '∥', 'sqrt': '√',
          'partial': '∂', 'nabla': '∇', 'sum': 'Σ', 'int': '∫', 'degree': '°', 'circ': '°', 'prime': '′', ',': ' ',
          ';': ' ', ' ': ' ', 'quad': '  ', 'ell': 'ℓ'}
_SUP = str.maketrans('0123456789+-=()n', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ')
_SUB = str.maketrans('0123456789+-=()', '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎')
_MATH_RE = re.compile(r'\$(.*?)\$', re.S)
_CMD_RE = re.compile(r'\\(mathrm|mathit|mathbf|text|rm|bf|it|left|right|displaystyle|,|;| |[A-Za-z]+)')


def _math_group(t, i):
    """The argument starting at t[i]: a {...} group (its inside, the index after it) or one character."""
    if i < len(t) and t[i] == '{':
        depth, j = 0, i
        while j < len(t):
            depth += (t[j] == '{') - (t[j] == '}')
            j += 1
            if depth == 0:
                break
        return t[i + 1:j - 1], j
    if i < len(t) and t[i] == '\\':
        m = _CMD_RE.match(t, i)
        if m:
            return t[i:m.end()], m.end()
    return (t[i], i + 1) if i < len(t) else ('', i)


def _math_part(t, html_out):
    """One mathtext segment (between dollar signs) as text: commands, groups, superscripts and subscripts."""
    out, i = '', 0
    while i < len(t):
        ch = t[i]
        if ch == '\\':
            m = _CMD_RE.match(t, i)
            if m is None:
                out += ch
                i += 1
                continue
            name = m.group(1)
            i = m.end()
            if name in ('mathrm', 'mathit', 'mathbf', 'text', 'rm', 'bf', 'it', 'displaystyle'):
                if i < len(t) and t[i] == '{':
                    g, i = _math_group(t, i)
                    out += _math_part(g, html_out)
                continue
            if name in ('left', 'right'):
                continue
            out += _GREEK.get(name, name)
        elif ch in '^_':
            g, i = _math_group(t, i + 1)
            inner = _math_part(g, html_out)
            if html_out:
                out += f"<{'sup' if ch == '^' else 'sub'}>{inner}</{'sup' if ch == '^' else 'sub'}>"
            else:
                table = _SUP if ch == '^' else _SUB
                conv = inner.translate(table)
                out += conv if all(c in '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎' for c in conv) else (ch + inner)
        elif ch in '{}':
            i += 1
        else:
            out += ch
            i += 1
    return out


def mathtext(text, html_out=False):
    """
    A label with matplotlib mathtext ($...$) as plain text (Unicode Greek letters, Å, superscript
    and subscript digits) or as HTML (<sup>, <sub>) for pyqtgraph's axis labels: the windows keep
    their matplotlib label strings and the layer shows them; the publication export (mplexport)
    gives matplotlib the original strings.  A text without a dollar sign is returned as it is
    (escaped for HTML when html_out).
    """
    t = '' if text is None else str(text)
    if '$' not in t:
        return html.escape(t) if html_out else t
    parts, last = [], 0
    for m in _MATH_RE.finditer(t):
        plain = t[last:m.start()]
        parts.append(html.escape(plain) if html_out else plain)
        inner = m.group(1)
        parts.append(_math_part(html.escape(inner) if html_out else inner, html_out))
        last = m.end()
    tail = t[last:]
    parts.append(html.escape(tail) if html_out else tail)
    return ''.join(parts)


def short_label(text):
    """
    An axis label as simple as the title: the quantity (before the first ', ') and its units
    (the last parentheses), e.g. '(dI/dV)/(I/V), uncalibrated, not normalized (arb. units)'
    -> '(dI/dV)/(I/V) (arb. units)'.
    """
    t = str(text or '').strip()
    depth, cut = 0, None
    for i, ch in enumerate(t):                 # the first ', ' outside parentheses
        depth += (ch == '(') - (ch == ')')
        if depth == 0 and t.startswith(', ', i):
            cut = i
            break
    if cut is None:
        return t
    head = t[:cut]
    tail = ''
    if t.endswith(')'):
        k = t.rfind(' (')
        if k > len(head):
            tail = t[k:]
    return (head + tail).strip()


_FILE_NAME_RE = re.compile(r'\.[A-Za-z][A-Za-z0-9]{0,3}$')     # a title that is a file name: it ends in an extension (a letter, then up to three letters or digits; '0.5' is not one)


def is_file_name(text):
    """Whether a title is a file name: it ends in an extension such as '.vasp' or '.txt'."""
    return bool(_FILE_NAME_RE.search(str(text or '').strip()))


STAMP_RE = re.compile(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})\s*$')   # a date-and-time stamp ending a file name


def file_stamp(name):
    """
    The date-and-time stamp of a file name, 'YYYY_MM_DD_HH_MM_SS_mmm' at the end of the base name
    (STAMP_RE), or None when the name has none (the rule for every file name shown as a title: a
    name without a stamp stays whole in a portrait window).
    """
    base = os.path.splitext(os.path.basename(str(name or '').strip()))[0]
    m = STAMP_RE.search(base)
    if m is None:
        return None
    return '_'.join(m.groups())


def short_title(text, portrait=False):
    """
    The essential part of a title: its first line, cut at the first ': ', ', ', double space,
    or ' ['.  The rest (counts, states, file details) goes to the window's info panel.  The short
    name is shown exactly as written: a component label ('molA'), an element, or a file name
    (anything ending in an extension such as '.vasp') is data, so its case is never changed, and a
    file name can be copied from the info panel into a file search.  With `portrait` (the window
    taller than it is wide, Figure.portrait) a file name is shortened to its date-and-time stamp
    (file_stamp) when it has one; a name without a stamp stays whole.  The one rule for every
    title shown in the suite.
    """
    t = str(text or '').strip()
    t = t.split('\n')[0]
    for sep in (': ', ', ', '  ', ' ['):
        k = t.find(sep)
        if k > 0:
            t = t[:k]
    t = t.strip().rstrip(':,;')
    if is_file_name(t) and portrait:
        stamp = file_stamp(t)
        if stamp:
            return stamp
    return t


class _Axis(pg.AxisItem):
    """
    A tick axis outside an axes' box.  Its size follows its tick labels
    (pyqtgraph sets it); a change of size re-places the owner.  mode
    'func' shows the values of a function of the view coordinate (the
    secondary y axis), mode 'norm' the data values of a colour bar whose
    view runs over the normalized [0, 1].

    The tick rule (generateDrawSpecs, paint): a tick is a value of the
    linked view box's range, labelled from that value, and its mark and
    label sit where the view box maps the value (ViewBox.mapViewToScene,
    the one mapping every artist in the box is drawn with), read at the
    moment the axis is painted and drawn straight into the painter the
    axis is given; the axis line spans the box's edge as the view box
    maps it.  Nothing between a value and its mark depends on the axis
    item's own geometry (which only gives the strip's thickness), on a
    range kept from a signal, or on a cached drawing.  pyqtgraph's
    AxisItem draws from a QPicture, which Qt replays scaled by the paint
    device's logical dpi over the primary screen's (QPicture::play), so
    on a monitor whose dpi differs from the primary's its ticks shrank
    toward the axis's origin while the data stayed in place; no picture
    is kept here.
    """

    def __init__(self, orientation, owner):
        super().__init__(orientation)
        self._owner = None
        self._sci = None
        self._mode = None          # None, ('func', f, finv), ('norm', norm)
        self._formatter = None     # a callable (value, position) -> label (Axes.xaxis / yaxis.set_major_formatter)
        self.setPen(pg.mkPen('k', width=max(pt2px(0.8), 1)))
        self.setTextPen(pg.mkPen('k'))
        self.setStyle(tickLength=int(round(pt2px(3.5))), tickTextOffset=int(round(pt2px(3.5))),
                      autoExpandTextSpace=True, autoReduceTextSpace=True, hideOverlappingLabels=False)
        self.enableAutoSIPrefix(False)
        self._tick_pt = 10.0
        self._label_pt = 10.0
        self._zoom = 1.0
        self.setTickFont(qfont(10))
        self.label.setFont(qfont(10))
        self.setZValue(5)
        self._owner = owner

    def setMaximumWidth(self, w):
        old = self.maximumWidth()
        super().setMaximumWidth(w)
        owner = getattr(self, '_owner', None)
        if abs(old - w) > 0.5 and owner is not None:
            owner._request_place()

    def setMaximumHeight(self, h):
        old = self.maximumHeight()
        super().setMaximumHeight(h)
        owner = getattr(self, '_owner', None)
        if abs(old - h) > 0.5 and owner is not None:
            owner._request_place()

    _STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)

    def _font_px(self):
        f = self.style.get('tickFont')
        return float(f.pixelSize()) if f is not None and f.pixelSize() > 0 else pt2px(10)

    def _mpl_ticks(self, lo, hi, size):
        """matplotlib's MaxNLocator ('auto' bins, steps 1, 2, 2.5, 5, 10): one level of ticks."""
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return []
        horizontal = self.orientation in ('top', 'bottom')
        nbins = int(size / (self._font_px() * (3.0 if horizontal else 2.0)))
        nbins = max(min(nbins, 9), 1)
        raw = span / nbins
        k = 10.0 ** math.floor(math.log10(raw))
        step = next((s * k for s in self._STEPS if s * k >= raw * (1 - 1e-9)), 10 * k)
        first = math.ceil(lo / step - 1e-9) * step
        n = int(math.floor((hi - first) / step + 1e-9)) + 1
        if not 0 <= n <= 200:
            return []
        vals = [first + i * step for i in range(max(n, 0))]
        vals = [0.0 if abs(v) < step * 1e-9 else v for v in vals]
        return [(step, vals)]

    # --- the tick rule: the view box's mapping, drawn afresh at every paint ---

    def _view_edge(self, view):
        """The linked view box's rectangle in this item's coordinates: the edge the axis sits on."""
        return view.mapRectToItem(self, view.rect())

    def _view_pos(self, view, v):
        """Where the view box maps the view coordinate `v` along this axis, in this item's coordinates (x of a horizontal axis, y of a vertical one)."""
        if self.orientation in ('top', 'bottom'):
            return self.mapFromScene(view.mapViewToScene(QtCore.QPointF(float(v), 0.0))).x()
        return self.mapFromScene(view.mapViewToScene(QtCore.QPointF(0.0, float(v)))).y()

    def _strip(self, view):
        """The axis's strip: its own thickness (its geometry across the axis) over the view box's extent along it; the geometry alone without a view."""
        own = self.mapRectFromParent(self.geometry())
        if view is None:
            return own
        edge = self._view_edge(view)
        if self.orientation in ('top', 'bottom'):
            return QtCore.QRectF(edge.left(), own.top(), edge.width(), own.height())
        return QtCore.QRectF(own.left(), edge.top(), own.width(), edge.height())

    def _view_specs(self, p, view):
        """
        pyqtgraph's generateDrawSpecs with the view box's mapping in place of the item's own: the
        axis line along the box's edge, every tick mark at the view box's position of its value
        (_view_pos), the labels laid out as pyqtgraph lays them out (the same offsets, alignment,
        crowding rule and measured text size), the grid lines across the box when the grid is on.
        None when the item is not on a device yet.
        """
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        own = self.mapRectFromParent(self.geometry())      # across the axis: where the marks and labels sit
        edge = self._view_edge(view)                        # along the axis: the box's extent
        far = edge if self.grid is not False else own       # the grid lines start at the box's far edge
        P = QtCore.QPointF
        if self.orientation == 'left':
            span = (P(own.right() - 1.0, edge.top() - 1.0), P(own.right() - 1.0, edge.bottom() + 1.0))
            tickStart, tickStop, tickDir, axis = far.right(), own.right(), -1, 0
        elif self.orientation == 'right':
            span = (P(own.left() + 1.0, edge.top() - 1.0), P(own.left() + 1.0, edge.bottom() + 1.0))
            tickStart, tickStop, tickDir, axis = far.left(), own.left(), 1, 0
        elif self.orientation == 'top':
            span = (P(edge.left() - 1.0, own.bottom() - 1.0), P(edge.right() + 1.0, own.bottom() - 1.0))
            tickStart, tickStop, tickDir, axis = far.bottom(), own.bottom(), -1, 1
        else:
            span = (P(edge.left() - 1.0, own.top() + 1.0), P(edge.right() + 1.0, own.top() + 1.0))
            tickStart, tickStop, tickDir, axis = far.top(), own.top(), 1, 1
        points = [self.mapToDevice(q) for q in span]
        if None in points:
            return None
        length = pg.Point(points[1] - points[0]).length()     # the axis's length in device pixels: how many ticks fit
        if length == 0:
            return None
        rng = view.viewRange()[1 - axis]                    # the view's range along the axis (x of a horizontal one)
        lo, hi = (rng[0], rng[1]) if rng[0] <= rng[1] else (rng[1], rng[0])
        if self._tickLevels is None:
            levels = self.tickValues(lo, hi, length)
            given = None
        else:                                               # set_xticks / set_yticks: the values with their labels
            levels, given = [], []
            for level in self._tickLevels:
                levels.append((None, [val for val, s in level]))
                given.append([s for val, s in level])
        pos_lo, pos_hi = (edge.left(), edge.right()) if axis == 1 else (edge.top(), edge.bottom())
        tol = 0.5                                           # a tick at the very end of the range is drawn
        positions, tickSpecs = [], []
        for i, (spacing, vals) in enumerate(levels):
            positions.append([])
            tickLength = self.style['tickLength'] / ((i * 0.5) + 1.0)
            lineAlpha = self.style['tickAlpha']
            if lineAlpha is None:
                lineAlpha = 255 / (i + 1)
                if self.grid is not False:
                    lineAlpha *= self.grid / 255. * pg.functions.clip_scalar(0.05 * length / (len(vals) + 1), 0., 1.)
            elif isinstance(lineAlpha, float):
                lineAlpha = min(255, max(0, int(round(lineAlpha * 255))))
            elif isinstance(lineAlpha, int):
                if not 0 <= lineAlpha <= 255:
                    raise ValueError("lineAlpha should be [0..255]")
            else:
                raise TypeError("Line Alpha should be of type None, float or int")
            tickPen = self.tickPen()
            if tickPen.brush().style() == Qt.BrushStyle.SolidPattern:
                tickPen = QtGui.QPen(tickPen)
                color = QtGui.QColor(tickPen.color())
                color.setAlpha(int(lineAlpha))
                tickPen.setColor(color)
            for v in vals:
                x = self._view_pos(view, v)
                if not (pos_lo - tol <= x <= pos_hi + tol):
                    positions[i].append(None)
                    continue
                positions[i].append(x)
                p1, p2 = [x, x], [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength * tickDir
                tickSpecs.append((tickPen, pg.Point(p1), pg.Point(p2)))
        drawn = [x for level in positions for x in level if x is not None]
        if drawn and self.style['stopAxisAtTick'][0] is True:
            if axis == 0:
                span[0].setY(max(span[0].y(), min(drawn)))
            else:
                span[0].setX(max(span[0].x(), min(drawn)))
        if drawn and self.style['stopAxisAtTick'][1] is True:
            if axis == 0:
                span[1].setY(min(span[1].y(), max(drawn)))
            else:
                span[1].setX(min(span[1].x(), max(drawn)))
        axisSpec = (self.pen(), span[0], span[1])
        textSpecs = []
        if not self.style['showValues']:
            return axisSpec, tickSpecs, textSpecs
        textOffset = self.style['tickTextOffset'][axis]
        textRects, lastTextSize2 = [], 0
        br = self.boundingRect()
        for i in range(min(len(levels), self.style['maxTextLevel'] + 1)):
            if given is None:
                spacing, vals = levels[i]
                strings = self.tickStrings(vals, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = list(given[i])
            if len(strings) == 0:
                continue
            for j in range(len(strings)):
                if positions[i][j] is None:                 # a tick out of the box: no label
                    strings[j] = None
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    r = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), Qt.AlignmentFlag.AlignCenter, s)
                    r.setHeight(r.height() * 0.8)           # pyqtgraph: the bounding rect is a bit too tall
                    rects.append(r)
                    textRects.append(r)
            if textRects:
                if axis == 0:
                    textSize = sum(r.height() for r in textRects)
                    textSize2 = max(r.width() for r in textRects)
                else:
                    textSize = sum(r.width() for r in textRects)
                    textSize2 = max(r.height() for r in textRects)
            else:
                textSize = textSize2 = 0
            if i > 0:                                       # the first level is always labelled; a crowded further level is not
                textFillRatio = float(textSize) / length
                if any(len(textSpecs) >= nTexts and textFillRatio >= limit
                       for nTexts, limit in self.style['textFillLimits']):
                    break
            lastTextSize2 = textSize2
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None:
                    continue
                x = positions[i][j]
                width, height = rects[j].width(), rects[j].height()
                offset = max(0, self.style['tickLength']) + textOffset
                if self.orientation == 'left':
                    alignFlags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop - offset - width, x - height / 2., width, height)
                elif self.orientation == 'right':
                    alignFlags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop + offset, x - height / 2., width, height)
                elif self.orientation == 'top':
                    alignFlags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
                    rect = QtCore.QRectF(x - width / 2., tickStop - offset - height, width, height)
                else:
                    alignFlags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                    rect = QtCore.QRectF(x - width / 2., tickStop + offset, width, height)
                if br & rect != rect:                       # a label running out of the strip is not drawn
                    continue
                textSpecs.append((rect, alignFlags | Qt.TextFlag.TextDontClip, vstr))
        self._updateMaxTextSize(lastTextSize2)
        return axisSpec, tickSpecs, textSpecs

    def generateDrawSpecs(self, p):
        # The draw specs from the view box's mapping (_view_specs; pyqtgraph's own without a linked view;
        # tickStrings sets self._sci on the way) plus, when the tick labels are mantissas, the power of
        # ten '1e<n>' once: above a vertical axis, at the right end of a horizontal one.
        self._sci = None
        view = self.linkedView()
        specs = self._view_specs(p, view) if view is not None else super().generateDrawSpecs(p)
        self._note_sci()
        if specs is None or self._sci is None or not self.style['showValues']:
            return specs
        axisSpec, tickSpecs, textSpecs = specs
        b = self._strip(view)
        txt = f"1e{self._sci}"
        if self.orientation in ('left', 'right'):
            rect = QtCore.QRectF(b.right() - 12 if self.orientation == 'left' else b.left() + 2, b.top() - 15, 90, 14)
            flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
        else:
            rect = QtCore.QRectF(b.right() - 60, b.bottom() - 14, 75, 14)
            flags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        textSpecs = list(textSpecs) + [(rect, flags | Qt.TextFlag.TextDontClip, txt)]
        return axisSpec, tickSpecs, textSpecs

    def paint(self, p, opt, widget):
        # Drawn afresh at every paint, straight into the painter given: no QPicture (the class docstring).
        specs = self.generateDrawSpecs(p)
        if specs is None:
            return
        p.save()
        try:
            self.drawPicture(p, *specs)
        finally:
            p.restore()

    def linkedViewChanged(self, view, newRange=None):
        super().linkedViewChanged(view, newRange)
        if newRange is None:                       # the view box's size changed (sigResized): the strip follows its edge
            self.prepareGeometryChange()
            self.update()

    def set_tick_pt(self, pt):
        self._tick_pt = font_pt(pt)
        self.setTickFont(qfont(self._tick_pt, zoom=self._zoom))

    def set_label_pt(self, pt):
        self._label_pt = font_pt(pt)
        self.label.setFont(qfont(self._label_pt, zoom=self._zoom))

    def apply_zoom(self, zoom):
        """The axis' fonts at the figure's zoom; its size follows (pyqtgraph re-measures)."""
        if abs(zoom - self._zoom) < 1e-6:
            return
        self._zoom = zoom
        self.setTickFont(qfont(self._tick_pt, zoom=zoom))
        self.label.setFont(qfont(self._label_pt, zoom=zoom))
        self.setStyle(tickLength=int(round(pt2px(3.5) * zoom)) if self.style['tickLength'] else 0,
                      tickTextOffset=int(round(pt2px(3.5) * zoom)))
        if self.label.isVisible():
            self.label.setHtml(self.labelString())
            self.label.setFont(qfont(self._label_pt, zoom=zoom))
        self.textWidth = self.textHeight = 0
        self._adjustSize()
        self.picture = None
        self.update()

    def _note_sci(self):
        """The axes' title moves up over a power-of-ten label at the top of a vertical axis."""
        flag = self._sci is not None and self.orientation in ('left', 'right')
        if flag != getattr(self, '_had_sci', False):
            self.prepareGeometryChange()
            self._had_sci = flag
            if self._owner is not None:
                self._owner._request_place()

    def boundingRect(self):
        # pyqtgraph's bounding rect on the strip (_strip: the box's extent along the axis, so no mark or
        # label of the box's edge is cut): widened along the axis for the labels at its ends (15 px
        # unless hideOverlappingLabels gives another margin), across it for ticks pointing inward, over
        # the box when the grid is on, and over a power-of-ten label above a vertical axis.
        view = self.linkedView()
        br = self._strip(view)
        if view is not None and self.grid is not False:
            br = br | self._view_edge(view)
        hide = self.style['hideOverlappingLabels']
        m = 0 if hide is True else (15 if hide is False else int(hide))
        tl = self.style['tickLength']
        if self.orientation == 'left':
            br = br.adjusted(0, -m, -min(0, tl), m)
        elif self.orientation == 'right':
            br = br.adjusted(min(0, tl), -m, 0, m)
        elif self.orientation == 'top':
            br = br.adjusted(-m, 0, m, -min(0, tl))
        else:
            br = br.adjusted(-m, min(0, tl), m, 0)
        if getattr(self, '_had_sci', False):
            br = br.united(QtCore.QRectF(br.left(), br.top() - 16, br.width() + 80, 16))
        return br

    def tickValues(self, minVal, maxVal, size):
        if self._mode is None:
            return self._mpl_ticks(minVal, maxVal, size)
        if self._mode[0] == 'func':
            f, finv = self._mode[1], self._mode[2]
            try:
                a, b = float(f(minVal)), float(f(maxVal))
            except Exception:
                return super().tickValues(minVal, maxVal, size)
            lo, hi = min(a, b), max(a, b)
            levels = self._mpl_ticks(lo, hi, size)
            return [(sp, [float(finv(v)) for v in vals]) for sp, vals in levels]
        norm = self._mode[1]
        vmin, vmax = norm.vmin, norm.vmax
        if vmin is None or vmax is None or not vmax > vmin:
            return []
        if isinstance(norm, SymLogNorm):
            # A symmetric-log colour bar: 0 and ±linthresh × 10^k up to the limits (matplotlib's SymmetricalLogLocator).
            vals = [0.0]
            k = 0
            while norm.linthresh * 10.0 ** k <= max(abs(vmin), abs(vmax)) * (1 + 1e-9) and k < 40:
                v = norm.linthresh * 10.0 ** k
                if vmin <= v <= vmax:
                    vals.append(v)
                if vmin <= -v <= vmax:
                    vals.append(-v)
                k += 1
            vals = sorted(v for v in vals if vmin <= v <= vmax)
            pos = np.asarray(norm(np.asarray(vals, dtype=np.float64)), dtype=np.float64) if vals else np.zeros(0)
            return [(None, [float(q) for q in pos])]
        if getattr(norm, 'log', False) and vmin > 0:
            # A logarithmic colour bar: a tick at every decade; 2 and 5 within the decades when fewer than three show.
            e0, e1 = math.ceil(math.log10(vmin) - 1e-9), math.floor(math.log10(vmax) + 1e-9)
            vals = [10.0 ** e for e in range(e0, e1 + 1)]
            if len(vals) < 3:
                vals = sorted(m * 10.0 ** e for e in range(e0 - 1, e1 + 1) for m in (1.0, 2.0, 5.0)
                              if vmin * (1 - 1e-9) <= m * 10.0 ** e <= vmax * (1 + 1e-9))
            pos = np.asarray(norm(np.asarray(vals, dtype=np.float64)), dtype=np.float64) if vals else np.zeros(0)
            return [(None, [float(q) for q in pos])]
        levels = self._mpl_ticks(vmin, vmax, size)      # at most one level
        out = []
        for sp, vals in levels:
            v = np.asarray([x for x in vals if vmin - 1e-12 * abs(vmax - vmin) <= x <= vmax + 1e-12 * abs(vmax - vmin)])
            pos = np.asarray(norm(v), dtype=np.float64) if len(v) else np.zeros(0)
            out.append((sp, [float(p) for p in pos]))
        return out

    def _data_values(self, values):
        """The data values of tick positions: the positions themselves, f(position) of a secondary axis, or a colour bar's norm inverted."""
        if self._mode is None:
            return [float(v) for v in values]
        if self._mode[0] == 'func':
            return [float(self._mode[1](v)) for v in values]
        return [float(self._mode[1].inverse(v)) for v in values]

    def tickStrings(self, values, scale, spacing):
        if self._formatter is not None:
            self._sci = None
            return [str(self._formatter(v, i)) for i, v in enumerate(self._data_values(values))]
        if self._mode is None:
            # matplotlib's scientific offset: very small or large ticks as mantissas, the power of ten
            # written once at the end of the axis.
            vals = [float(v) for v in values]
            big = max((abs(v) for v in vals), default=0.0)
            if big and (big < 1e-3 or big >= 1e5):
                e = int(math.floor(math.log10(big)))
                self._sci = e
                k = 10.0 ** e
                return _tick_strings([v / k for v in vals], spacing / k)
            self._sci = None
            return _tick_strings(vals, spacing)
        if self._mode[0] == 'func':
            vals = [float(self._mode[1](v)) for v in values]
        else:
            norm = self._mode[1]
            vals = [float(norm.inverse(v)) for v in values]
            if getattr(norm, 'log', False) or isinstance(norm, SymLogNorm):
                return [f"{v:g}".replace('-', '\u2212') for v in vals]
        big = max((abs(v) for v in vals), default=0.0)
        if big and (big < 1e-3 or big >= 1e5):
            e = int(math.floor(math.log10(big)))
            self._sci = e
            return _tick_strings([v / 10.0 ** e for v in vals], spacing / 10.0 ** e)
        return _tick_strings(vals, spacing)


# Set while a view box handles a drag or the wheel (_ViewBox): a view whose range changes meanwhile was
# changed with the toolbar's Pan or Zoom (Axes._nav_follow), the views linked to it included.
_NAVIGATING = [0]


class _ViewBox(pg.ViewBox):
    """
    A ViewBox without its context menu; navigation only while the toolbar's pan or zoom mode is on;
    the views its drags and its wheel change are marked as changed with the toolbar (_NAVIGATING).
    """

    def __init__(self, axes):
        super().__init__(enableMenu=False, defaultPadding=0.05)
        self._axes = axes
        self.setMouseEnabled(False, False)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        """pyqtgraph's drag (Pan: pan with the left button, zoom with the right; Zoom: the rectangle), under _NAVIGATING."""
        _NAVIGATING[0] += 1
        try:
            super().mouseDragEvent(ev, axis)
        finally:
            _NAVIGATING[0] -= 1

    def wheelEvent(self, ev, axis=None):
        """pyqtgraph's wheel zoom (Pan or Zoom), under _NAVIGATING."""
        _NAVIGATING[0] += 1
        try:
            super().wheelEvent(ev, axis)
        finally:
            _NAVIGATING[0] -= 1


# ---------------------------------------------------------------------- #
#  Artists                                                                 #
# ---------------------------------------------------------------------- #

_child_count = itertools.count()


class Artist:
    """The part of matplotlib's Artist the package uses: label, visibility, zorder, picker, remove."""

    zorder_default = 2.0

    def __init__(self, axes=None, label=None, zorder=None, picker=None, visible=True):
        self.axes = axes
        self.figure = axes.figure if axes is not None else None
        self._label = label if label is not None else f"_child{next(_child_count)}"
        self.zorder = self.zorder_default if zorder is None else float(zorder)
        self._picker = picker
        self._visible = bool(visible)
        self._items = []
        self.colorbar = None

    # items
    def _add_item(self, item, ignore_bounds=False):
        item.setZValue(self.zorder)
        if not self._visible:
            item.setVisible(False)
        self.axes.vb.addItem(item, ignoreBounds=ignore_bounds)
        self.axes._place_item(item)
        self._items.append(item)
        return item

    def get_label(self):
        return self._label

    def set_label(self, s):
        self._label = '' if s is None else str(s)

    def get_visible(self):
        return self._visible

    def set_visible(self, flag):
        self._visible = bool(flag)
        for it in self._items:
            it.setVisible(self._visible)

    def set_zorder(self, z):
        self.zorder = float(z)
        for it in self._items:
            it.setZValue(self.zorder)

    def get_zorder(self):
        return self.zorder

    def set_picker(self, p):
        self._picker = p

    def get_picker(self):
        return self._picker

    def pickable(self):
        return self._picker is not None and self._visible

    def contains_px(self, x, y):
        return False

    def contains(self, event):
        """matplotlib's Artist.contains(mouseevent): (hit, {'ind': [index of the point hit]})."""
        hit = self.contains_px(event.x, event.y)
        ind = getattr(self, '_hit_index', None)
        return bool(hit), {'ind': [ind] if (hit and ind is not None) else []}

    def remove(self):
        ax = self.axes
        for it in self._items:
            try:
                if ax is not None:
                    ax.vb.removeItem(it)
                if it.scene() is not None:
                    it.scene().removeItem(it)
            except Exception:
                pass
        self._items = []
        if ax is not None:
            ax._forget(self)

    def _legend_sample(self):
        return None

    def set(self, **kw):
        for k, v in kw.items():
            f = getattr(self, 'set_' + k, None)
            if f is not None:
                f(v)


# matplotlib marker -> pyqtgraph symbol (an unknown marker draws a circle).
_MARKERS = {'o': 'o', 's': 's', 'D': 'd', 'd': 'd', '^': 't1', 'v': 't', '<': 't3', '>': 't2',
            'x': 'x', 'X': 'x', '+': '+', 'P': '+', '*': 'star', 'p': 'p', 'h': 'h', 'H': 'h', '.': 'o', ',': 's',
            '|': '|', '_': '_'}


def _marker_symbol(m):
    if m in (None, '', 'none', 'None', ' '):
        return None
    return _MARKERS.get(m, 'o')


class Line2D(Artist):
    """
    A curve and/or markers: a pyqtgraph PlotCurveItem for a plain curve, a PlotDataItem with
    markers (the '.' marker at half its size).  get_xdata, get_ydata, set_data, set_color, ...;
    NaN points break the curve and are not drawn.  A picker (a radius in display pixels, or
    True for 5) makes contains_px test the distance to the polyline, or to the points when there
    is no line.
    """

    def __init__(self, axes, x=None, y=None, color='C0', lw=1.5, ls='-', marker=None, ms=6.0, mec=None, mfc=None,
                 mew=1.0, alpha=None, label=None, zorder=None, picker=None, visible=True, fill=None, **kw):
        if not isinstance(axes, Axes):
            # matplotlib's Line2D(xdata, ydata, **style) made without an axes: a legend proxy, nothing drawn.
            axes, x, y = None, axes, x
        lw = kw.pop('linewidth', lw)
        ls = kw.pop('linestyle', ls)
        ms = kw.pop('markersize', ms)
        mfc = kw.pop('markerfacecolor', mfc)
        mec = kw.pop('markeredgecolor', mec)
        mew = kw.pop('markeredgewidth', mew)
        color = kw.pop('c', color)
        super().__init__(axes, label, zorder, picker, visible)
        self._x = np.asarray([] if x is None else x, dtype=np.float64).ravel()
        self._y = np.asarray([] if y is None else y, dtype=np.float64).ravel()
        self._style = dict(color=color, lw=lw, ls=ls, marker=marker, ms=ms, mec=mec, mfc=mfc, mew=mew, alpha=alpha)
        self.item = None
        if axes is not None:
            self._build()

    def _style_args(self):
        s = self._style
        color = s['color']
        pen = mkpen(color, s['lw'], s['ls'], s['alpha'])
        sym = _marker_symbol(s['marker'])
        kw = {'pen': pen}
        if sym is not None:
            ms = s['ms'] if s['marker'] != '.' else s['ms'] * 0.5
            mfc = color if s['mfc'] is None else s['mfc']
            mec = color if s['mec'] is None else s['mec']
            kw.update(symbol=sym, symbolSize=max(pt2px(ms), 1.0),
                      symbolBrush=mkbrush(mfc, s['alpha']) if mkbrush(mfc) is not None else None,
                      symbolPen=mkpen(mec, s['mew'], '-', s['alpha']) if s['mew'] and s['mew'] > 0 else None)
            if kw['symbolBrush'] is None:
                kw['symbolBrush'] = QtGui.QBrush(Qt.BrushStyle.NoBrush)
            if kw['symbolPen'] is None:
                kw['symbolPen'] = QtGui.QPen(Qt.PenStyle.NoPen)
        return kw

    def _build(self):
        if self.axes is None:
            return                                   # a legend proxy: no items
        kw = self._style_args()
        x, y = self._x, self._y
        if len(x) == len(y):
            # A point with a NaN coordinate is not drawn and takes no part in the autoscale (matplotlib).
            with np.errstate(invalid='ignore'):
                bad = ~(np.isfinite(x) & np.isfinite(y)) | (np.abs(x) > _LIMIT) | (np.abs(y) > _LIMIT)
            if bad.any():
                x = np.where(bad, np.nan, x)
                y = np.where(bad, np.nan, y)
        if kw.get('pen') is None and kw.get('symbol') is not None:
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
        if kw.get('symbol') is None:
            # A plain curve: the lighter PlotCurveItem (no scatter child).
            if self.item is None:
                self.item = pg.PlotCurveItem(x, y, pen=kw.get('pen') or QtGui.QPen(Qt.PenStyle.NoPen),
                                             connect='finite', antialias=ANTIALIAS, skipFiniteCheck=False)
                self._add_item(self.item)
            else:
                self.item.setData(x, y, pen=kw.get('pen') or QtGui.QPen(Qt.PenStyle.NoPen), connect='finite')
            return
        if self.item is None:
            self.item = pg.PlotDataItem(x, y, connect='finite', antialias=ANTIALIAS, **kw)
            self._add_item(self.item)
        else:
            self.item.setData(x, y, connect='finite', **kw)

    def get_xdata(self, orig=True):
        return self._x

    def get_ydata(self, orig=True):
        return self._y

    def get_data(self, orig=True):
        return self._x, self._y

    def set_data(self, *args):
        if len(args) == 1:
            x, y = args[0]
        else:
            x, y = args
        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self._build()

    def set_xdata(self, x):
        self.set_data(x, self._y)

    def set_ydata(self, y):
        self.set_data(self._x, y)

    def get_color(self):
        return self._style['color']

    def set_color(self, c):
        self._style['color'] = c
        self._build()

    def set_linewidth(self, lw):
        self._style['lw'] = lw
        self._build()

    def set_linestyle(self, ls):
        self._style['ls'] = ls
        self._build()

    def set_alpha(self, a):
        self._style['alpha'] = a
        self._build()

    def set_markerfacecolor(self, c):
        self._style['mfc'] = c
        self._build()

    def get_alpha(self):
        return self._style['alpha']

    def get_linewidth(self):
        return self._style['lw']

    def get_linestyle(self):
        return self._style['ls']

    def get_marker(self):
        return self._style['marker']

    def contains_px(self, x, y):
        """Display-pixel distance test (matplotlib's pick radius = the picker value); the nearest point's index is kept in _hit_index."""
        tol = float(self._picker) if isinstance(self._picker, (int, float)) and not isinstance(self._picker, bool) else 5.0
        if len(self._x) == 0 or self.axes is None:
            return False
        xy = self.axes.transData.transform(np.column_stack([self._x, self._y]))
        ok = np.all(np.isfinite(xy), axis=1)
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        d[~ok] = np.inf
        self._hit_index = int(np.argmin(d)) if np.isfinite(d).any() else None
        if self._style['ls'] in (None, 'none', 'None', '', ' ') or ok.sum() < 2:
            return bool(len(d)) and float(d.min()) <= tol
        return _polyline_distance(x, y, xy) <= tol

    def _legend_sample(self):
        kw = self._style_args()
        return _LegendSample(kw.get('pen'), kw.get('symbol'), kw.get('symbolSize', 6), kw.get('symbolPen'),
                             kw.get('symbolBrush'))


def _polyline_distance(x, y, xy):
    """Distance from (x, y) to a polyline (n, 2), NaN vertices breaking it."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 0:
        return np.inf
    p, q = xy[:-1], xy[1:]
    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    best = np.inf
    fin = np.all(np.isfinite(xy), axis=1)
    if fin.any():
        best = float(np.min(np.hypot(xy[fin, 0] - x, xy[fin, 1] - y)))
    if ok.any():
        p, d = p[ok], (q - p)[ok]
        L2 = np.sum(d * d, axis=1)
        t = np.clip(((x - p[:, 0]) * d[:, 0] + (y - p[:, 1]) * d[:, 1]) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        best = min(best, float(np.sqrt(np.min((p[:, 0] + t * d[:, 0] - x) ** 2 + (p[:, 1] + t * d[:, 1] - y) ** 2))))
    return best


def _rgba_rows(colors, n, alpha=None, default='C0'):
    """
    Colours as an (n, 4) float array: one colour (a name, a hex string, an (r, g, b[, a]) tuple)
    repeated, a list of colours, or an (n, 3 | 4) array of floats; `alpha` overrides the alpha.
    """
    if colors is None or (isinstance(colors, str) and colors == 'face'):
        colors = default
    if isinstance(colors, str):
        rows = np.tile(to_rgba(colors), (max(n, 1), 1))
    else:
        # Numbers: one (r, g, b[, a]) colour, or an (m, 3 | 4) array of colours; anything else (names,
        # hex strings, tuples mixed in a list) is converted colour by colour.
        try:
            arr = np.asarray(colors, dtype=np.float64)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.ndim == 1 and len(arr) in (3, 4):
            rows = np.tile(to_rgba(tuple(float(v) for v in arr)), (max(n, 1), 1))
        elif arr is not None and arr.ndim == 2 and arr.shape[1] in (3, 4):
            rows = np.ones((arr.shape[0], 4))
            rows[:, :arr.shape[1]] = arr
        else:
            items = list(colors)
            rows = np.array([to_rgba(c) for c in items]) if items else np.zeros((0, 4))
            if len(rows) == 1 and n > 1:
                rows = np.tile(rows, (n, 1))
    if len(rows) < n and len(rows):
        rows = np.resize(rows, (n, 4))
    rows = np.clip(np.asarray(rows, dtype=np.float64).reshape(-1, 4), 0.0, 1.0)
    if alpha is not None:
        rows[:, 3] = float(alpha)
    return rows[:n] if n else rows[:0]


class PathCollection(Artist):
    """
    scatter: markers at `offsets` (n, 2) with their own face colours (`facecolors`: one, a list,
    an (n, 4) array, or `array` through `cmap` and `norm`), edge colours ('face': the face colour;
    'none'; a colour) and sizes (points squared), one pyqtgraph ScatterPlotItem (pxMode: the
    markers keep their size on screen).  set_offsets, get_offsets, set_facecolors, set_sizes,
    set_alpha, set_array; a picker makes contains_px test the distance to the nearest marker
    (within its radius plus the pick radius) and PickEvent.ind holds its index.
    """

    zorder_default = 1.0

    def __init__(self, axes, offsets, sizes=36.0, facecolors=None, edgecolors='face', linewidths=1.0, marker='o',
                 alpha=None, label=None, zorder=None, picker=None, visible=True, cmap=None, norm=None, array=None):
        super().__init__(None, label, zorder, picker, visible)
        self._xy = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        self._sizes = np.atleast_1d(np.asarray(sizes, dtype=np.float64))
        self._fc = facecolors
        self._ec = edgecolors
        self._lw = linewidths
        self._marker = marker
        self._alpha = alpha
        self._array = None if array is None else np.asarray(array, dtype=np.float64).ravel()
        self.cmap = to_cmap(cmap) if (cmap is not None or array is not None) else None
        self.norm = norm if norm is not None else (Normalize() if array is not None else None)
        if self.norm is not None and self._array is not None and not self.norm.scaled():
            self.norm.autoscale_None(self._array)
        self.item = None
        self._hit_index = None
        if axes is not None:
            self._attach(axes)

    def _attach(self, axes):
        self.axes, self.figure = axes, axes.figure
        self._items = []
        self.item = pg.ScatterPlotItem(pxMode=True, antialias=ANTIALIAS)
        self._refresh()
        self._add_item(self.item)

    def _face_rows(self):
        n = len(self._xy)
        if self._array is not None and self.cmap is not None:
            rows = self.cmap(np.asarray(self.norm(self._array), dtype=np.float64))
            rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
            if self._alpha is not None:
                rows[:, 3] = float(self._alpha)
            return np.resize(rows, (n, 4)) if len(rows) and len(rows) != n else rows
        return _rgba_rows(self._fc, n, self._alpha)

    def _refresh(self):
        if self.item is None:
            return
        n = len(self._xy)
        rows = self._face_rows() if n else np.zeros((0, 4))
        sizes = np.resize(self._sizes, (n,)) if n else np.zeros(0)
        px = [max(pt2px(math.sqrt(max(float(sz), 0.0))), 1.0) for sz in sizes]
        brushes = [pg.mkBrush(QtGui.QColor.fromRgbF(*r)) if r[3] > 0 else QtGui.QBrush(Qt.BrushStyle.NoBrush) for r in rows]
        lw = float(np.ravel(self._lw)[0]) if np.ndim(self._lw) else float(self._lw)
        ec = self._ec
        if ec is None or (isinstance(ec, str) and ec in ('none', 'None')) or lw <= 0:
            pens = [QtGui.QPen(Qt.PenStyle.NoPen)] * n
        elif isinstance(ec, str) and ec == 'face':
            pens = [pg.mkPen(QtGui.QColor.fromRgbF(*r), width=pt2px(lw)) for r in rows]
        else:
            erows = _rgba_rows(ec, n, self._alpha)
            pens = [pg.mkPen(QtGui.QColor.fromRgbF(*r), width=pt2px(lw)) for r in erows]
        with np.errstate(invalid='ignore'):
            ok = np.all(np.isfinite(self._xy) & (np.abs(self._xy) < _LIMIT), axis=1) if n else np.zeros(0, dtype=bool)
        xy = self._xy[ok]
        sym = _marker_symbol(self._marker) or 'o'
        self.item.setData(x=xy[:, 0], y=xy[:, 1], size=[p for p, k in zip(px, ok) if k], symbol=sym,
                          brush=[b for b, k in zip(brushes, ok) if k], pen=[p for p, k in zip(pens, ok) if k])
        self._drawn_index = np.flatnonzero(ok)

    def set_offsets(self, offsets):
        self._xy = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        self._refresh()

    def get_offsets(self):
        return self._xy

    def set_facecolors(self, colors):
        self._fc = colors
        self._array = None
        self._refresh()

    set_facecolor = set_facecolors
    set_color = set_facecolors

    def get_facecolors(self):
        return self._face_rows()

    get_facecolor = get_facecolors

    def set_edgecolors(self, colors):
        self._ec = colors
        self._refresh()

    set_edgecolor = set_edgecolors

    def set_sizes(self, sizes):
        self._sizes = np.atleast_1d(np.asarray(sizes, dtype=np.float64))
        self._refresh()

    def get_sizes(self):
        return self._sizes

    def set_alpha(self, a):
        self._alpha = a
        self._refresh()

    def get_alpha(self):
        return self._alpha

    def set_array(self, a):
        self._array = np.asarray(a, dtype=np.float64).ravel()
        if self.cmap is None:
            self.cmap = to_cmap(None)
        if self.norm is None:
            self.norm = Normalize()
        if not self.norm.scaled():
            self.norm.autoscale_None(self._array)
        self._refresh()

    def get_array(self):
        return self._array

    def set_norm(self, norm):
        self.norm = norm
        self._refresh()

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self._refresh()

    def set_clim(self, vmin=None, vmax=None):
        if self.norm is None:
            self.norm = Normalize()
        if vmin is not None:
            self.norm.vmin = float(vmin)
        if vmax is not None:
            self.norm.vmax = float(vmax)
        self._refresh()

    def contains_px(self, x, y):
        if self.axes is None or not len(self._xy):
            return False
        xy = self.axes.transData.transform(self._xy)
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        d[~np.all(np.isfinite(xy), axis=1)] = np.inf
        n = len(self._xy)
        r = 0.5 * np.sqrt(np.maximum(np.resize(self._sizes, (n,)), 0.0)) * DPI / 72.0    # marker radii in pixels (pt2px of the diameter)
        tol = float(self._picker) if isinstance(self._picker, (int, float)) and not isinstance(self._picker, bool) else 0.0
        ok = d <= r + tol
        if not ok.any():
            self._hit_index = None
            return False
        self._hit_index = int(np.argmin(np.where(ok, d, np.inf)))
        return True

    def _legend_sample(self):
        rows = self._face_rows()
        col = QtGui.QColor.fromRgbF(*rows[0]) if len(rows) else qcolor('C0')
        sz = float(np.ravel(self._sizes)[0]) if len(np.ravel(self._sizes)) else 36.0
        return _LegendSample(None, _marker_symbol(self._marker) or 'o', min(pt2px(math.sqrt(max(sz, 0.0))), 10.0),
                             pg.mkPen(col), pg.mkBrush(col))


class Patch(Artist):
    """matplotlib.patches.Patch as the windows use it: a legend proxy (a filled square in its colour) made without an axes."""

    def __init__(self, color=None, facecolor=None, edgecolor=None, label=None, alpha=None, **kw):
        super().__init__(None, label, None, None, True)
        self._color = color if color is not None else (facecolor if facecolor is not None else 'C0')
        self._alpha = alpha

    def get_facecolor(self):
        return to_rgba(self._color, self._alpha)

    def _legend_sample(self):
        return _LegendSample(fill=mkbrush(self._color, self._alpha))


class _RefLine(Artist):
    """axhline / axvline: a pyqtgraph InfiniteLine across the axes."""

    def __init__(self, axes, value, horizontal, color='k', lw=1.5, ls='-', alpha=None, label=None, zorder=None,
                 picker=None, visible=True):
        super().__init__(axes, label, zorder, picker, visible)
        self._value = float(value)
        self._horizontal = bool(horizontal)
        self._style = dict(color=color, lw=lw, ls=ls, alpha=alpha)
        pen = mkpen(color, lw, ls, alpha) or QtGui.QPen(Qt.PenStyle.NoPen)
        bad = not np.isfinite(self._value) or abs(self._value) > _LIMIT
        self.item = pg.InfiniteLine(pos=0.0 if bad else self._value, angle=0 if horizontal else 90, pen=pen,
                                    movable=False)
        self._add_item(self.item, ignore_bounds=True)
        if bad:
            self.item.setVisible(False)

    def get_xdata(self, orig=True):
        return np.array([0.0, 1.0]) if self._horizontal else np.array([self._value] * 2)

    def get_ydata(self, orig=True):
        return np.array([self._value] * 2) if self._horizontal else np.array([0.0, 1.0])

    def set_ydata(self, y):
        if self._horizontal:
            self._value = float(np.ravel(y)[0])
            self.item.setValue(self._value)

    def set_xdata(self, x):
        if not self._horizontal:
            self._value = float(np.ravel(x)[0])
            self.item.setValue(self._value)

    def set_color(self, c):
        self._style['color'] = c
        self.item.setPen(mkpen(c, self._style['lw'], self._style['ls'], self._style['alpha']))

    def get_alpha(self):
        return self._style['alpha']

    def get_color(self):
        return self._style['color']

    def contains_px(self, x, y):
        tol = float(self._picker) if isinstance(self._picker, (int, float)) and not isinstance(self._picker, bool) else 5.0
        p = self.axes.transData.transform([[self._value, self._value]])[0]
        horizontal = self._horizontal != bool(getattr(self.axes, '_rotated', False))    # on screen (a rotated axes turns it)
        return abs((y - p[1]) if horizontal else (x - p[0])) <= tol

    def _legend_sample(self):
        s = self._style
        return _LegendSample(mkpen(s['color'], s['lw'], s['ls'], s['alpha']))


class _Span(Artist):
    """
    axvspan / axhspan: a filled band across the axes (pyqtgraph LinearRegionItem, not movable,
    no edge lines), alpha 0.25 unless given.
    """

    zorder_default = 1.0

    def __init__(self, axes, lo, hi, vertical, color='C0', alpha=None, label=None, zorder=None, visible=True):
        super().__init__(axes, label, zorder, None, visible)
        self._span = (float(lo), float(hi), bool(vertical), color, 0.25 if alpha is None else alpha)
        self._brush = mkbrush(color, 0.25 if alpha is None else alpha)
        self.item = pg.LinearRegionItem(values=(float(lo), float(hi)),
                                        orientation='vertical' if vertical else 'horizontal',
                                        brush=self._brush, pen=QtGui.QPen(Qt.PenStyle.NoPen), movable=False)
        for ln in self.item.lines:
            ln.setPen(QtGui.QPen(Qt.PenStyle.NoPen))
        self._add_item(self.item, ignore_bounds=True)

    def _legend_sample(self):
        return _LegendSample(fill=self._brush)


class AxesImage(Artist):
    """
    imshow: a pyqtgraph ImageItem coloured through a lookup table (norm_lut)
    from the norm and the colour map at levels (vmin, vmax), placed by its
    extent (l, r, b, t; the pixel indices when None) and an optional
    Affine2D (set_transform, applied after the extent: a rotated and
    translated topography).  origin 'upper' puts row 0 at the top edge.  A
    3D (RGB / RGBA) array is drawn as given.  NaN pixels are transparent.
    A page without a size (an empty or degenerate extent) is not drawn.
    """

    zorder_default = 0.0

    def __init__(self, axes, data, cmap=None, norm=None, extent=None, origin='upper', alpha=None, zorder=None,
                 label=None, vmin=None, vmax=None):
        super().__init__(axes, label, zorder, None, True)
        self.cmap = to_cmap(cmap)
        self.origin = origin or 'upper'
        self._alpha = alpha
        self._affine = None
        self.item = pg.ImageItem(axisOrder='row-major')
        self.item.setAutoDownsample(False)
        self._A = None
        self.norm = norm if norm is not None else Normalize(vmin, vmax)
        self._extent = None
        self._set_array(data)
        self.set_extent(extent)
        if alpha is not None:
            self.item.setOpacity(float(alpha))
        self._add_item(self.item)

    def _set_array(self, data):
        A = np.ma.asarray(data)
        if A.dtype.kind in 'fiub' and A.ndim == 2:
            self._A = np.asarray(np.ma.filled(A.astype(np.float64), np.nan), dtype=np.float64)
            if not self.norm.scaled():
                self.norm.autoscale_None(self._A)
            self._rgb = False
        else:
            self._A = np.asarray(np.ma.filled(A, 0))
            self._rgb = True

    def _refresh(self):
        img = self._A
        if self.origin == 'upper':
            img = img[::-1]
        if self._rgb:
            a = np.asarray(img)
            if a.dtype.kind == 'f':
                a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            self.item.setImage(a, autoLevels=False, levels=(0, 255))
            return
        vmin, vmax = self.norm.vmin, self.norm.vmax
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-12 if vmin == vmax else vmin
        lut = norm_lut(self.cmap, self.norm, vmin, vmax)
        self.item.setImage(img, autoLevels=False, levels=(vmin, vmax), lut=lut)

    def _place_transform(self):
        ny, nx = self._A.shape[:2]
        l, r, b, t = self._extent
        ok = (nx > 0 and ny > 0 and all(np.isfinite(v) and abs(v) < _LIMIT for v in (l, r, b, t))
              and abs(r - l) > 0 and abs(t - b) > 0)
        if ok and self._affine is not None:
            m = self._affine.get_matrix()
            ok = bool(np.all(np.isfinite(m)) and abs(np.linalg.det(m[:2, :2])) > 1e-300)
        if not ok:
            # A page without a size (a scan stopped before it began, a zero pixel scale): not drawn.
            self.item.setVisible(False)
            self.item.setTransform(QtGui.QTransform())
            return
        self.item.setVisible(self._visible)
        # Pixel (row, col) of the drawn array (origin flips applied) covers [l + col dx, ...] from (l, b).
        T = QtGui.QTransform()
        T.translate(l, b)
        T.scale((r - l) / nx, (t - b) / ny)
        if self._affine is not None:
            T = T * self._affine.to_qt()
        self.item.setTransform(T)

    def set_extent(self, extent):
        ny, nx = self._A.shape[:2]
        if extent is None:
            extent = (-0.5, nx - 0.5, ny - 0.5, -0.5) if self.origin == 'upper' else (-0.5, nx - 0.5, -0.5, ny - 0.5)
        l, r, b, t = [float(v) for v in extent]
        # origin 'upper': row 0 at the top edge t (the drawn array is flipped in _refresh, so its row 0 sits at b).
        self._extent = (l, r, b, t)
        self._refresh()
        self._place_transform()

    def get_extent(self):
        return list(self._extent)

    def set_transform(self, t):
        """A transform added to transData (matplotlib's Affine2D() ... + ax.transData)."""
        aff = None
        while isinstance(t, _Composite):
            if isinstance(t.first, Affine2D):
                aff = t.first
            t = t.then
        if isinstance(t, Affine2D):
            aff = t
        self._affine = aff
        self._place_transform()

    def set_data(self, data):
        self._set_array(data)
        self._refresh()
        self._place_transform()

    def get_array(self):
        return self._A

    def set_norm(self, norm):
        self.norm = norm
        if not self.norm.scaled() and self._A is not None and not self._rgb:
            self.norm.autoscale_None(self._A)
        self._refresh()
        if self.colorbar is not None:
            self.colorbar.update_normal(self)

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self._refresh()
        if self.colorbar is not None:
            self.colorbar.update_normal(self)

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        if vmin is not None:
            self.norm.vmin = float(vmin)
        if vmax is not None:
            self.norm.vmax = float(vmax)
        self.set_norm(self.norm)

    def get_clim(self):
        return self.norm.vmin, self.norm.vmax


class QuadMesh(Artist):
    """pcolormesh: filled quads (QuadItem) coloured from the norm and the colour map."""

    zorder_default = 1.0

    def __init__(self, axes, X, Y, C, cmap=None, norm=None, vmin=None, vmax=None, zorder=None, label=None,
                 alpha=None, shading='flat'):
        super().__init__(axes, label, zorder, None, True)
        self.cmap = to_cmap(cmap)
        self.norm = norm if norm is not None else Normalize(vmin, vmax)
        self._alpha = alpha
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        C = np.asarray(np.ma.filled(np.ma.asarray(C, dtype=np.float64), np.nan), dtype=np.float64)
        nr, nc = C.shape
        # matplotlib's shading: 'flat' takes X and Y as the cells' edges (one more than C each
        # way); 'nearest' (and 'auto' when X and Y have C's shape) takes them as the cells'
        # centres and puts the edges halfway between neighbours, extrapolated at the borders
        # (matplotlib's _interp_grid); 'gouraud' is drawn as 'nearest' (flat cells, no shading).
        centres = (shading in ('nearest', 'gouraud')
                   or (shading == 'auto' and (X.shape[-1] == nc if X.ndim == 1 else X.shape == C.shape)))
        if centres:
            X = self._edges(X, nc, axis=-1)
            Y = self._edges(Y, nr, axis=0 if Y.ndim == 2 else -1)
            if X.ndim == 2 and X.shape[0] == nr:
                X = self._edges(X, nr, axis=0)
            if Y.ndim == 2 and Y.shape[1] == nc:
                Y = self._edges(Y, nc, axis=-1)
        if X.ndim == 1:
            X = np.broadcast_to(X[None, :], (nr + 1, nc + 1))
        if Y.ndim == 1:
            Y = np.broadcast_to(Y[:, None], (nr + 1, nc + 1))
        self._X, self._Y, self._C = X, Y, C
        if not self.norm.scaled():
            self.norm.autoscale_None(C)
        q = np.stack([np.stack([X[:-1, :-1], Y[:-1, :-1]], -1), np.stack([X[:-1, 1:], Y[:-1, 1:]], -1),
                      np.stack([X[1:, 1:], Y[1:, 1:]], -1), np.stack([X[1:, :-1], Y[1:, :-1]], -1)], axis=2)
        self._quads = q.reshape(-1, 4, 2)
        self.item = QuadItem()
        self._paint()
        self._add_item(self.item)

    @staticmethod
    def _edges(a, n, axis=-1):
        """Cell edges from `n` cell centres along `axis` (midpoints, the borders extrapolated); `a` left as it is when it already holds n + 1 edges."""
        a = np.asarray(a, dtype=np.float64)
        if a.shape[axis] != n:
            return a
        if n == 1:
            lo = np.take(a, [0], axis=axis) - 0.5
            return np.concatenate([lo, lo + 1.0], axis=axis)
        a0, a1 = np.take(a, range(0, n - 1), axis=axis), np.take(a, range(1, n), axis=axis)
        mid = 0.5 * (a0 + a1)
        first = 2.0 * np.take(a, [0], axis=axis) - np.take(mid, [0], axis=axis)
        last = 2.0 * np.take(a, [n - 1], axis=axis) - np.take(mid, [mid.shape[axis] - 1], axis=axis)
        return np.concatenate([first, mid, last], axis=axis)

    def _paint(self):
        colors = self.cmap(np.asarray(self.norm(self._C.ravel()), dtype=np.float64), alpha=self._alpha, bytes=True)
        colors[~np.isfinite(self._C.ravel())] = 0
        self.item.setData(self._quads, colors)

    def set_array(self, C):
        self._C = np.asarray(np.ma.filled(np.ma.asarray(C, dtype=np.float64), np.nan), dtype=np.float64).reshape(self._C.shape)
        self._paint()

    def get_array(self):
        """The cell values as matplotlib gives them: a masked array, the cells given as masked (or NaN) masked."""
        return np.ma.masked_invalid(self._C)

    def set_norm(self, norm):
        self.norm = norm
        self._paint()
        if self.colorbar is not None:
            self.colorbar.update_normal(self)

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self._paint()
        if self.colorbar is not None:
            self.colorbar.update_normal(self)


def _band_levels(z, levels):
    """
    The band edges of a filled contour: `levels` edges as given, or levels + 1 edges evenly over
    the finite data range (matplotlib's locator picks round numbers near them; the bands here
    divide the range exactly).  A flat field gets one band around its value.
    """
    ok = np.isfinite(z)
    lo, hi = (float(z[ok].min()), float(z[ok].max())) if ok.any() else (0.0, 1.0)
    if np.ndim(levels) == 0:
        lev = np.linspace(lo, hi, int(levels) + 1)
    else:
        lev = np.asarray(levels, dtype=np.float64)
    if len(lev) < 2 or not lev[-1] > lev[0]:
        lev = np.array([lo - 0.5, lo + 0.5])
    return lev


def _band_cmap(base, lev, norm):
    """One colour per band: the base map at the norm of the band's midpoint (matplotlib's contourf colouring)."""
    mids = 0.5 * (lev[:-1] + lev[1:])
    cols = np.asarray(base(np.asarray(norm(mids), dtype=np.float64)), dtype=np.float64).reshape(-1, 4)
    cols[~np.isfinite(mids)] = 0.0
    return Colormap(base.name + '_bands', cols)


class ContourSet(Artist):
    """
    contourf on a regular grid: the bands drawn as an image (one colour per band between
    levels, the base map at the norm of each band's midpoint; `norm`, or vmin / vmax, else the
    levels' range).
    """

    zorder_default = 1.0

    def __init__(self, axes, X, Y, Z, levels=10, cmap=None, zorder=None, norm=None, vmin=None, vmax=None, alpha=None):
        super().__init__(axes, None, zorder, None, True)
        Z = np.asarray(np.ma.filled(np.ma.asarray(Z, dtype=np.float64), np.nan), dtype=np.float64)
        ok = np.isfinite(Z)
        lev = _band_levels(Z, levels)
        self.levels = lev
        base = to_cmap(cmap)
        self.norm = norm if norm is not None else Normalize(lev[0] if vmin is None else vmin, lev[-1] if vmax is None else vmax)
        if not self.norm.scaled():
            self.norm.autoscale_None(lev)
        self._args = (np.asarray(X), np.asarray(Y), Z, lev, base, self.norm)
        mids = 0.5 * (lev[:-1] + lev[1:])
        self.cmap = _band_cmap(base, lev, self.norm)
        band = np.clip(np.digitize(Z, lev[1:-1]), 0, len(mids) - 1).astype(np.float64)
        band[~ok] = np.nan
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        x = X[0] if X.ndim == 2 else X
        y = Y[:, 0] if Y.ndim == 2 else Y
        self.item = pg.ImageItem(axisOrder='row-major')
        self.item.setImage(band, autoLevels=False, levels=(0, len(mids)), lut=self.cmap.lut(len(mids)))
        if alpha is not None:
            self.item.setOpacity(float(alpha))
        T = QtGui.QTransform()
        T.translate(float(x[0]), float(y[0]))
        T.scale((float(x[-1]) - float(x[0])) / band.shape[1], (float(y[-1]) - float(y[0])) / band.shape[0])
        self.item.setTransform(T)
        self._add_item(self.item)

    def set_alpha(self, a):
        for it in self._items:
            it.setOpacity(1.0 if a is None else float(a))


class TriContourSet(Artist):
    """
    tricontourf: the scattered points (x, y, z) interpolated linearly (scipy's
    LinearNDInterpolator over their Delaunay triangulation, the piecewise-linear model
    matplotlib's tricontourf fills) onto a raster over their bounding box, `res` cells along its
    longer side (NaN outside the hull: transparent), banded by the levels (_band_levels) and
    drawn as one image through a lookup table (_band_cmap: the base map at the norm of each
    band's midpoint; `norm`, or vmin / vmax, else the levels' range).  `offsets` draws the same
    raster at each (dx, dy): a window that tiles a periodic map computes it once and moves it
    (never re-imaged).  A new colour scale rebuilds the table, not the raster (set_cmap,
    set_norm).  The arguments are kept (_args) for the publication export.
    """

    zorder_default = 1.0

    def __init__(self, axes, x, y, z, levels=10, cmap=None, norm=None, vmin=None, vmax=None, zorder=None, alpha=None,
                 offsets=None, res=256):
        super().__init__(axes, None, zorder, None, True)
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        z = np.asarray(np.ma.filled(np.ma.asarray(z, dtype=np.float64), np.nan), dtype=np.float64).ravel()
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[ok], y[ok], z[ok]
        self._offsets = [(0.0, 0.0)] if offsets is None else [(float(a), float(b)) for a, b in offsets]
        lev = _band_levels(z, levels)
        self.levels = lev
        self._base = to_cmap(cmap)
        self.norm = norm if norm is not None else Normalize(lev[0] if vmin is None else vmin, lev[-1] if vmax is None else vmax)
        if not self.norm.scaled():
            self.norm.autoscale_None(lev)
        self._args = (x, y, z, lev, self._base, self.norm, list(self._offsets))
        self._alpha = alpha
        self._band = None
        self._rect = None
        if len(x) >= 3:
            x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
            w, h = x1 - x0, y1 - y0
            if w > 0 and h > 0:
                nx = int(res) if w >= h else max(2, int(round(res * w / h)))
                ny = int(res) if h > w else max(2, int(round(res * h / w)))
                gx = x0 + (np.arange(nx) + 0.5) * w / nx
                gy = y0 + (np.arange(ny) + 0.5) * h / ny
                GX, GY = np.meshgrid(gx, gy)
                Zr = None
                try:
                    from scipy.interpolate import LinearNDInterpolator
                    Zr = LinearNDInterpolator(np.column_stack([x, y]), z)(GX, GY)
                except Exception as exc:
                    print(f"tricontourf: the points cannot be triangulated ({exc}); nothing drawn.")
                if Zr is not None:
                    n_b = len(lev) - 1
                    band = np.clip(np.digitize(Zr, lev[1:-1]), 0, n_b - 1).astype(np.float64)
                    band[~np.isfinite(Zr)] = np.nan
                    self._band = band
                    self._rect = (x0, y0, w, h, nx, ny)
        self._build_items()

    def _build_items(self):
        if self._band is None:
            return
        x0, y0, w, h, nx, ny = self._rect
        n_b = len(self.levels) - 1
        self.cmap = _band_cmap(self._base, self.levels, self.norm)
        lut = self.cmap.lut(n_b)
        for dx, dy in self._offsets:
            it = pg.ImageItem(axisOrder='row-major')
            it.setImage(self._band, autoLevels=False, levels=(0, n_b), lut=lut)
            if self._alpha is not None:
                it.setOpacity(float(self._alpha))
            T = QtGui.QTransform()
            T.translate(x0 + dx, y0 + dy)
            T.scale(w / nx, h / ny)
            it.setTransform(T)
            self._add_item(it)

    def _recolor(self):
        n_b = len(self.levels) - 1
        self.cmap = _band_cmap(self._base, self.levels, self.norm)
        lut = self.cmap.lut(n_b)
        for it in self._items:
            it.setLookupTable(lut)

    def set_norm(self, norm):
        self.norm = norm
        self._recolor()

    def set_cmap(self, cmap):
        self._base = to_cmap(cmap)
        self._recolor()

    def set_alpha(self, a):
        self._alpha = a
        for it in self._items:
            it.setOpacity(1.0 if a is None else float(a))

    def get_alpha(self):
        return self._alpha

    def get_array(self):
        return self._args[2]

    def get_facecolor(self):
        """The bands' colours, one (r, g, b, a) row per band (matplotlib's ContourSet.get_facecolor)."""
        return np.array(self.cmap._table, dtype=np.float64)

    get_facecolors = get_facecolor


class LineCollection(Artist):
    """
    Line segments / polylines (matplotlib's LineCollection), each with its own colour and width:
    `colors` one colour or one per segment, or `array` through `cmap` and `norm` (set_array
    recolours: the simulator's tip-height stripe), `linewidths` one or one per segment (points),
    one line style; one SegmentsItem, drawn once Axes.add_collection attaches it.  set_segments,
    get_segments, set_array, set_color(s), set_linewidth(s), set_alpha; a picker makes
    contains_px test the distance to the nearest segment and PickEvent.ind holds its index.
    """

    zorder_default = 2.0

    def __init__(self, segments, colors='k', linewidths=1.5, linestyles='-', zorder=None, alpha=None, label=None,
                 cmap=None, norm=None, array=None, color=None, linewidth=None, linestyle=None, picker=None, **kw):
        super().__init__(None, label, zorder, picker, True)
        self._segments = [np.asarray(sg, dtype=np.float64).reshape(-1, 2) for sg in segments]
        self._colors = colors if color is None else color
        lw = linewidths if linewidth is None else linewidth
        self._lw = np.atleast_1d(np.asarray(lw, dtype=np.float64))
        ls = linestyles if linestyle is None else linestyle
        if isinstance(ls, (list, tuple)) and ls and isinstance(ls[0], str):
            ls = ls[0]
        self._ls = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}.get(ls, ls) if isinstance(ls, str) else '-'
        self._alpha = alpha
        self._array = None if array is None else np.asarray(array, dtype=np.float64).ravel()
        self.cmap = to_cmap(cmap) if (cmap is not None or array is not None) else None
        self.norm = norm if norm is not None else (Normalize() if array is not None else None)
        if self.norm is not None and self._array is not None and not self.norm.scaled():
            self.norm.autoscale_None(self._array)
        self.item = None
        self._hit_index = None

    def _seg_colors(self):
        n = len(self._segments)
        if self._array is not None and self.cmap is not None:
            c = self.cmap(np.asarray(self.norm(np.resize(self._array, (n,)) if len(self._array) != n else self._array),
                                     dtype=np.float64), alpha=self._alpha, bytes=True)
            c = np.asarray(c, dtype=np.uint8).reshape(-1, 4)
            c[~np.isfinite(np.resize(self._array, (n,)))] = 0
            return c
        rows = _rgba_rows(self._colors, n, self._alpha, default='k')
        return (np.clip(rows, 0, 1) * 255 + 0.5).astype(np.uint8)

    @property
    def _color(self):
        """The first segment's colour (the legend sample, the export of a one-colour collection)."""
        c = self._seg_colors()
        return tuple(float(v) / 255.0 for v in c[0]) if len(c) else to_rgba(self._colors if isinstance(self._colors, str) else 'k')

    def _paint(self):
        if self.item is not None:
            self.item.setData(self._segments, self._seg_colors(), self._lw, self._ls)

    def _attach(self, axes):
        self.axes = axes
        self.figure = axes.figure
        self._items = []
        self.item = SegmentsItem()
        self._paint()
        self._add_item(self.item)

    def get_segments(self):
        return list(self._segments)

    def set_segments(self, segments):
        self._segments = [np.asarray(sg, dtype=np.float64).reshape(-1, 2) for sg in segments]
        self._paint()

    def set_array(self, a):
        self._array = np.asarray(a, dtype=np.float64).ravel()
        if self.cmap is None:
            self.cmap = to_cmap(None)
        if self.norm is None:
            self.norm = Normalize()
        if not self.norm.scaled():
            self.norm.autoscale_None(self._array)
        self._paint()

    def get_array(self):
        return self._array

    def set_norm(self, norm):
        self.norm = norm
        self._paint()

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self._paint()

    def set_clim(self, vmin=None, vmax=None):
        if self.norm is None:
            self.norm = Normalize()
        if vmin is not None:
            self.norm.vmin = float(vmin)
        if vmax is not None:
            self.norm.vmax = float(vmax)
        self._paint()

    def set_color(self, c):
        self._colors = c
        self._array = None
        self._paint()

    set_colors = set_color
    set_edgecolor = set_color

    def get_color(self):
        return self._color

    def set_linewidth(self, lw):
        self._lw = np.atleast_1d(np.asarray(lw, dtype=np.float64))
        self._paint()

    set_linewidths = set_linewidth

    def get_linewidth(self):
        return self._lw

    def set_alpha(self, a):
        self._alpha = a
        self._paint()

    def get_alpha(self):
        return self._alpha

    def contains_px(self, x, y):
        if self.axes is None or not self._segments:
            return False
        tol = float(self._picker) if isinstance(self._picker, (int, float)) and not isinstance(self._picker, bool) else 5.0
        best, best_i = np.inf, None
        for i, sg in enumerate(self._segments):
            if len(sg) < 1:
                continue
            xy = self.axes.transData.transform(sg)
            d = _polyline_distance(x, y, xy) if len(sg) > 1 else float(np.hypot(xy[0, 0] - x, xy[0, 1] - y))
            if d < best:
                best, best_i = d, i
        self._hit_index = best_i
        return best <= tol

    def _legend_sample(self):
        lw = float(self._lw[0]) if len(self._lw) else 1.5
        return _LegendSample(mkpen(self._color, lw, self._ls))


class PolyCollection(Artist):
    """
    Filled polygons (quads; a triangle gets its last vertex repeated) coloured from `array`
    through the norm and colour map, or by `facecolors` (one colour or one per polygon, 'C0' by
    default); one QuadItem, drawn once Axes.add_collection attaches it; `edgecolors` and
    `antialiased` are ignored.
    """

    zorder_default = 1.0

    def __init__(self, verts, array=None, cmap=None, norm=None, facecolors=None, edgecolors=None,
                 antialiased=False, zorder=None, alpha=None, label=None, **kw):
        super().__init__(None, label, zorder, None, True)
        v = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in verts] if not isinstance(verts, np.ndarray) \
            else np.asarray(verts, dtype=np.float64)
        if isinstance(v, list):
            v = np.array([p if len(p) == 4 else np.vstack([p[:3], p[2:3]]) for p in v]) if v else np.zeros((0, 4, 2))
        self._verts = np.asarray(v, dtype=np.float64).reshape(-1, 4, 2)
        self._array = None if array is None else np.asarray(array, dtype=np.float64).ravel()
        self.cmap = to_cmap(cmap) if (cmap is not None or array is not None) else None
        self.norm = norm if norm is not None else (Normalize() if array is not None else None)
        if self.norm is not None and self._array is not None and not self.norm.scaled():
            self.norm.autoscale_None(self._array)
        self._facecolors = facecolors
        self._alpha = alpha
        self.item = None

    def _colors(self):
        if self._array is not None:
            c = self.cmap(np.asarray(self.norm(self._array), dtype=np.float64), alpha=self._alpha, bytes=True)
            c[~np.isfinite(self._array)] = 0
            return c
        fc = self._facecolors
        if fc is None:
            fc = 'C0'
        if isinstance(fc, str) or (np.ndim(fc) == 1 and len(np.ravel(fc)) in (3, 4) and not isinstance(fc[0], str)):
            c = np.array([to_rgba(fc, self._alpha)] * len(self._verts))
        else:
            c = np.array([to_rgba(f, self._alpha) for f in fc]) if len(fc) and isinstance(fc[0], str) \
                else np.asarray(fc, dtype=np.float64).reshape(-1, 4)
        return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)

    def _attach(self, axes):
        self.axes = axes
        self.figure = axes.figure
        self.item = QuadItem(self._verts, self._colors())
        self._add_item(self.item)

    def set_array(self, a):
        self._array = np.asarray(a, dtype=np.float64).ravel()
        if self.item is not None:
            self.item.setData(self._verts, self._colors())

    def get_array(self):
        return self._array

    def set_norm(self, norm):
        self.norm = norm
        if self.item is not None:
            self.item.setData(self._verts, self._colors())

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        if self.item is not None:
            self.item.setData(self._verts, self._colors())


# matplotlib's ha / va as pyqtgraph TextItem anchors: the fraction of the text's box that sits at its
# position (0 left / top, 1 right / bottom; 'baseline' approximated at 0.8 of the height).
_HA = {'left': 0.0, 'center': 0.5, 'centre': 0.5, 'right': 1.0}
_VA = {'top': 0.0, 'center': 0.5, 'centre': 0.5, 'center_baseline': 0.55, 'baseline': 0.8, 'bottom': 1.0}


class Text(Artist):
    """
    A text in data, axes, or figure coordinates (`coords`; pyqtgraph TextItem): set_text,
    get_text, set_color, set_fontsize, set_visible, set_position.  `fontsize` in points, drawn at
    the figure's zoom (_apply, re-run by Axes._refont); `offset` in logical pixels (matplotlib's
    'offset points', converted by the caller); `bbox` a matplotlib bbox dict (facecolor / fc,
    alpha, edgecolor / ec, lw).  A data text sits in the view box; an axes or figure text sits in
    the scene and is placed again (_place) whenever its axes or the canvas moves.
    """

    zorder_default = 3.0

    def __init__(self, owner, x, y, s, coords='data', fontsize=None, ha='left', va='baseline', color='k',
                 family=None, fontweight=None, weight=None, bbox=None, visible=True, zorder=None, offset=(0, 0),
                 linespacing=None, clip_on=False, wrap=False, rotation=0, fontfamily=None, size=None,
                 horizontalalignment=None, verticalalignment=None, label=None, cell=None, **kw):
        axes = owner if isinstance(owner, Axes) else None
        super().__init__(axes, label, zorder, None, visible)
        self.figure = owner if isinstance(owner, Figure) else owner.figure
        self._owner = owner
        self._coords = coords
        # A figure text in a layout cell (automatic layout: the one rule for every window): x, y are
        # fractions of the cell's rectangle, the text placed again whenever the cell moves.
        self._cell = cell if coords == 'figure' else None
        self._place_pending = False
        self._x, self._y = float(x), float(y)
        self._text = '' if s is None else str(s)
        self._size = font_pt(size if fontsize is None else fontsize)
        self._family = family or fontfamily
        self._weight = fontweight or weight
        self._color = color
        ha = horizontalalignment or ha
        va = verticalalignment or va
        self._ha, self._va = ha, va
        self._bbox = dict(bbox) if bbox else None
        self._rotation = float(rotation or 0)
        self._offset_px = (float(offset[0]), float(offset[1]))
        fill = None
        border = None
        if bbox:
            fc = bbox.get('facecolor', bbox.get('fc', 'white'))
            fill = mkbrush(fc, bbox.get('alpha', 1.0))
            ec = bbox.get('edgecolor', bbox.get('ec'))
            lw = bbox.get('lw', bbox.get('linewidth', 1.0))
            if ec is not None and lw:
                border = mkpen(ec, lw)
        self.item = _TextItem('', anchor=(_HA.get(ha, 0.0), _VA.get(va, 0.8)), fill=fill, border=border,
                              angle=float(rotation or 0), offset=offset)
        self.item.setZValue(self.zorder + 10 if coords != 'data' else self.zorder)
        self._apply()
        if not self._visible:
            self.item.setVisible(False)
        if coords == 'data':
            self.axes.vb.addItem(self.item, ignoreBounds=True)
            self.axes._place_item(self.item)
            if np.isfinite(self._x) and np.isfinite(self._y) and max(abs(self._x), abs(self._y)) < _LIMIT:
                self.item.setPos(self._x, self._y)
            else:
                self.item.setVisible(False)
        else:
            self.figure.canvas.scene().addItem(self.item)
            if self._cell is not None:
                self._cell.bind(self)
            self._place()
        self._items.append(self.item)

    def _apply(self):
        ti = self.item
        ti.setFont(qfont(self._size, self._family, self._weight, zoom=getattr(self.figure, 'font_zoom', 1.0)))
        ti.setColor(qcolor(self._color))
        ti.setText(mathtext(self._text))
        doc = ti.textItem.document()
        opt = doc.defaultTextOption()
        opt.setAlignment({'center': Qt.AlignmentFlag.AlignHCenter, 'right': Qt.AlignmentFlag.AlignRight}.get(
            self._ha, Qt.AlignmentFlag.AlignLeft))
        doc.setDefaultTextOption(opt)
        ti.textItem.setTextWidth(-1)
        if '\n' in self._text and self._ha in ('center', 'right'):
            ti.textItem.setTextWidth(doc.idealWidth() + 2)
        ti.updateTextPos()

    def _place(self):
        self._place_pending = False
        if self._coords == 'data':
            return
        if self._coords == 'axes':
            x0, y0, w, h = self.axes._px_rect()
            self.item.setPos(x0 + self._x * w, y0 + (1.0 - self._y) * h)
        elif self._cell is not None:
            x0, y0, w, h = self._cell.rect_px()
            self.item.setPos(x0 + self._x * w, y0 + (1.0 - self._y) * h)
        else:
            W, H = self.figure._canvas_size()
            self.item.setPos(self._x * W, (1.0 - self._y) * H)

    def _request_place(self):
        if not self._place_pending:
            self._place_pending = True
            QtCore.QTimer.singleShot(0, self._place)

    def figure_xy(self):
        """A figure text's anchor in figure fractions (from its cell when it has one)."""
        if self._cell is None:
            return self._x, self._y
        W, H = self.figure._canvas_size()
        x0, y0, w, h = self._cell.rect_px()
        return (x0 + self._x * w) / W, 1.0 - (y0 + (1.0 - self._y) * h) / H

    def set_text(self, s):
        s = '' if s is None else str(s)
        if s != self._text:
            self._text = s
            self._apply()

    def get_text(self):
        return self._text

    def set_color(self, c):
        self._color = c
        self.item.setColor(qcolor(c))

    def get_color(self):
        return self._color

    def set_fontsize(self, s):
        self._size = font_pt(s)
        self._apply()

    set_size = set_fontsize

    def get_fontsize(self):
        return self._size

    def set_position(self, xy):
        self._x, self._y = float(xy[0]), float(xy[1])
        if self._coords == 'data':
            self.item.setPos(self._x, self._y)
        else:
            self._place()

    def get_position(self):
        return (self._x, self._y)

    def set_fontweight(self, w):
        self._weight = w
        self._apply()

    def set_visible(self, flag):
        self._visible = bool(flag)
        self.item.setVisible(self._visible)

    def remove(self):
        try:
            if self._coords == 'data' and self.axes is not None:
                self.axes.vb.removeItem(self.item)
            elif self.item.scene() is not None:
                self.item.scene().removeItem(self.item)
        except Exception:
            pass
        if self._cell is not None:
            self._cell.unbind(self)
        self._items = []
        if isinstance(self._owner, Axes):
            self._owner._forget(self)
        else:
            self._owner._forget_text(self)


class Annotation(Artist):
    """
    annotate: a text with an offset and/or an arrow from xytext to xy (data coordinates).  With
    textcoords 'offset points' / 'offset pixels' the text sits at xy displaced by xytext; else at
    xytext (data).  With `arrowprops` a line from xytext to xy and an arrow head at xy (filled for
    the '-|>', 'simple', 'fancy' and 'wedge' styles, open otherwise), its colour and `lw` from
    arrowprops.  The arguments are kept (_ann) for the publication export.
    """

    zorder_default = 3.0

    def __init__(self, axes, s, xy, xytext=None, textcoords=None, arrowprops=None, color='k', fontsize=None,
                 fontweight=None, zorder=None, ha='left', va='baseline', **kw):
        super().__init__(axes, None, zorder, None, True)
        self.text = None
        self._ann = dict(s=s, xy=tuple(xy), xytext=None if xytext is None else tuple(xytext), textcoords=textcoords,
                         arrowprops=dict(arrowprops) if arrowprops else None, color=color, fontsize=fontsize,
                         fontweight=fontweight, ha=ha, va=va)
        xy = (float(xy[0]), float(xy[1]))
        if arrowprops is not None and xytext is not None:
            a = (float(xytext[0]), float(xytext[1]))
            col = arrowprops.get('color', color)
            lw = arrowprops.get('lw', arrowprops.get('linewidth', 1.0))
            line = pg.PlotCurveItem([a[0], xy[0]], [a[1], xy[1]], pen=mkpen(col, lw), antialias=True)
            self._add_item(line, ignore_bounds=True)
            ang = self._screen_angle(a, xy)
            style = arrowprops.get('arrowstyle', '->')
            arrow = pg.ArrowItem(angle=ang, tipAngle=40, headLen=pt2px(6 + lw * 2), tailLen=0, tailWidth=0,
                                 pen=mkpen(col, lw), brush=mkbrush(col) if style in ('-|>', 'simple', 'fancy', 'wedge')
                                 else QtGui.QBrush(Qt.BrushStyle.NoBrush))
            arrow.setPos(xy[0], xy[1])
            self._add_item(arrow, ignore_bounds=True)
        if s:
            off = (0.0, 0.0)
            if xytext is not None and textcoords in ('offset points', 'offset pixels'):
                k = pt2px(1) if textcoords == 'offset points' else 1.0
                off = (float(xytext[0]) * k, float(xytext[1]) * k)
                pos = xy
            elif xytext is not None:
                pos = (float(xytext[0]), float(xytext[1]))
            else:
                pos = xy
            self.text = Text(axes, pos[0], pos[1], s, 'data', fontsize=fontsize, color=color, fontweight=fontweight,
                             ha=ha, va=va, offset=off, zorder=self.zorder)
            self._items.append(self.text.item)

    def get_text(self):
        return self.text.get_text() if self.text is not None else ''

    def set_text(self, s):
        if self.text is not None:
            self.text.set_text(s)

    def _screen_angle(self, a, b):
        ax = self.axes
        try:
            pa, pb = ax.transData.transform([a, b])
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        except Exception:
            dx, dy = b[0] - a[0], b[1] - a[1]
        if dx == 0 and dy == 0:
            dx, dy = b[0] - a[0], b[1] - a[1]
        # pyqtgraph's arrow at angle 0 points to the left (-x); screen y runs down.
        return math.degrees(math.atan2(dy, -dx))

    def set_visible(self, flag):
        super().set_visible(flag)


# ---------------------------------------------------------------------- #
#  Colour bar                                                              #
# ---------------------------------------------------------------------- #

class Colorbar:
    """
    A colour bar in its own axes `cax` (its view fixed to [0, 1] x [0, 1],
    no mouse): the colour map evenly along the bar (a 512-row image through
    the map's lookup table), the tick labels on a right-hand side axis at
    their data values through the norm (_Axis mode 'norm', so a gamma
    spaces them as matplotlib does), the label on the right.  update_normal
    follows a new mappable, norm or colour map (the mappables call it
    through their `colorbar` attribute); remove deletes the axes and gives
    the parent axes its rectangle back.
    """

    def __init__(self, figure, mappable, cax, label=None, parent_axes=None, parent_rect=None):
        self.figure = figure
        self.ax = cax
        self.mappable = mappable
        self._parent = parent_axes
        self._parent_rect = parent_rect
        self._label = label or ''
        cax.clear()
        mappable.colorbar = self
        cax._colorbar = self
        cax._is_colorbar = True
        cax._apply_toolbar_mode('')
        cax.xaxis_item.setStyle(showValues=False, tickLength=0)
        cax.xaxis_item.setTicks([[]])
        cax.xaxis_item._dirty = cax.yaxis_item._dirty = True
        cax.yaxis_item.setVisible(False)
        self._axis = cax._make_side_axis('right')
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image.setImage(((np.arange(512) + 0.5) / 512.0)[:, None], autoLevels=False, levels=(0.0, 1.0))
        self.image.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        cax.vb.addItem(self.image)
        cax.vb.disableAutoRange()
        cax.vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        self.update_normal(mappable)
        if label:
            self.set_label(label)
        cax._place()

    def update_normal(self, mappable):
        self.mappable = mappable
        cmap = to_cmap(getattr(mappable, 'cmap', None))
        norm = getattr(mappable, 'norm', None) or Normalize(0, 1)
        self.image.setLookupTable(cmap.lut(512))
        self._axis._mode = ('norm', norm)
        self._axis.picture = None
        self._axis.update()

    def set_label(self, label, **kw):
        self._label = '' if label is None else str(label)
        self._label_kw = dict(kw)
        self.ax._side_label(self._axis, self._label, kw.get('fontsize', kw.get('size')))

    def set_ticks(self, *a, **k):
        pass

    def set_ticklabels(self, *a, **k):
        pass

    def update_ticks(self):
        """matplotlib's update_ticks: the bar's axis drawn again (its locator and formatter attributes are accepted and ignored: the tick rule is the layer's)."""
        self._axis.picture = None
        self._axis.update()

    locator = None
    formatter = None

    def remove(self):
        try:
            if getattr(self.mappable, 'colorbar', None) is self:
                self.mappable.colorbar = None
        except Exception:
            pass
        self.ax.remove()
        if self._parent is not None and self._parent_rect is not None:
            self._parent.set_position(self._parent_rect)


# ---------------------------------------------------------------------- #
#  Axes                                                                    #
# ---------------------------------------------------------------------- #

class _Patch:
    """ax.patch: the axes' background; set_facecolor colours the view box, alpha and visibility are ignored."""

    def __init__(self, ax):
        self._ax = ax

    def set_facecolor(self, c):
        self._ax.vb.setBackgroundColor(qcolor(c))

    def set_alpha(self, a):
        pass

    def set_visible(self, f):
        pass


class _FigPatch:
    """
    fig.patch: the figure's background as matplotlib's Figure.patch.  The screen keeps the window
    colour; the publication export (mplexport) writes the file with this colour and alpha
    (a transparent patch gives a transparent PNG).
    """

    def __init__(self):
        self._rgba = (1.0, 1.0, 1.0, 1.0)

    def set_facecolor(self, c):
        self._rgba = tuple(to_rgba(c))

    def set_alpha(self, a):
        r, g, b, _ = self._rgba
        self._rgba = (r, g, b, 1.0 if a is None else float(a))

    def get_facecolor(self):
        return tuple(self._rgba)

    def get_alpha(self):
        return self._rgba[3]

    def set_visible(self, flag):
        pass


class _AxisProxy:
    """
    ax.xaxis / ax.yaxis as the windows use them: set_major_formatter(fmt) labels the ticks of the
    tick axis through fmt(value, position) (ticker.FuncFormatter, FormatStrFormatter, any
    callable); for a colour-bar axes the y axis is the bar's value axis.  Locators are accepted
    and ignored (the tick rule is the layer's).
    """

    def __init__(self, ax, which):
        self._ax, self._which = ax, which

    def _items(self):
        ax = self._ax
        if self._which == 'y':
            cb = getattr(ax, '_colorbar', None)
            if cb is not None:
                return [cb._axis]
            return [ax.yaxis_item]
        return [ax.xaxis_item]

    def set_major_formatter(self, fmt):
        for it in self._items():
            it._formatter = fmt if callable(fmt) else None
            it.picture = None
            it.update()

    def set_major_locator(self, *a, **k):
        pass

    def set_minor_locator(self, *a, **k):
        pass

    def set_minor_formatter(self, *a, **k):
        pass

    def set_ticks_position(self, *a, **k):
        pass

    def set_label_position(self, *a, **k):
        pass

    def set_visible(self, flag):
        for it in self._items():
            it.setVisible(bool(flag))

    def get_major_ticks(self):
        return []


class _TitleProxy:
    """ax.title: set_fontsize / set_text / get_text on the axes' title."""

    def __init__(self, ax):
        self._ax = ax

    def set_fontsize(self, s):
        self._ax._title_size = font_pt(s)
        if self._ax._title is not None:
            self._ax._title.set_fontsize(s)

    def set_text(self, s):
        self._ax.set_title(s)

    def get_text(self):
        return self._ax._title.get_text() if self._ax._title is not None else ''

    def set_color(self, c):
        if self._ax._title is not None:
            self._ax._title.set_color(c)


class SecondaryAxis:
    """secondary_yaxis: tick labels of f(y) on the right of the axes."""

    def __init__(self, ax, axis_item):
        self._ax = ax
        self._axis = axis_item

    def set_ylabel(self, label, **kw):
        self._ax._side_label(self._axis, label, kw.get('fontsize'))

    def tick_params(self, labelsize=None, **kw):
        if labelsize is not None:
            self._axis.set_tick_pt(labelsize)


# ---------------------------------------------------------------------- #
#  Automatic layout                                                        #
# ---------------------------------------------------------------------- #
#
# automatic layout: the one rule for every window (nothing placed by hand).  A window's controls
# and its panels share one Qt layout on the canvas's viewport (Figure.set_layout): a control is a
# real Qt widget in that layout, a panel is a PanelCell in it, and the axes, colour bars and texts
# bound to a cell (Figure.add_axes(cell=...), Figure.text(..., cell=...)) are drawn on the canvas
# inside the cell's rectangle.  Qt's layout sizes and places the controls and the cells together,
# so a control meant to sit under a panel sits under it, nothing overlaps, and no position is
# picked by hand.  A panel keeps its own rules inside its cell: a spatial panel keeps its x:y ratio
# (an adjustable-'box' aspect shrinks the box about the cell's centre) and a row of panels keeps
# one height (Figure.align_row).

class PanelCell(QtWidgets.QWidget):
    """
    A panel's place in its window's Qt layout (automatic layout: the one rule for every window): an
    empty, mouse-transparent widget whose rectangle the layout sets.  The objects bound to it (an
    Axes made with cell=..., a figure text made with cell=...) are placed again whenever the cell,
    or any widget holding it, moves or changes size.  `stretch` (horizontal, vertical) says whether
    the cell takes the room the layout has to spare in that direction; `min_px` is its smallest size.
    """

    def __init__(self, figure, min_px=(40, 40), stretch=(True, True), inner_w=None):
        super().__init__()
        self.figure = figure
        self._decor_epoch, self._decor_val = -1, None      # the room measured in the current layout pass (_decor)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        pol = QtWidgets.QSizePolicy.Policy
        # inner_w: a narrow panel (a colour bar, a ramp) whose box is inner_w pixels wide at text
        # zoom 1; the cell is as wide as that box plus its tick axes and label (sizeHint).
        self.inner_w = inner_w
        if inner_w is not None:
            self.setSizePolicy(pol.Fixed, pol.Expanding if stretch[1] else pol.Preferred)
            self.setMinimumHeight(int(min_px[1]))
        else:
            self.setSizePolicy(pol.Expanding if stretch[0] else pol.Preferred,
                               pol.Expanding if stretch[1] else pol.Preferred)
            self.setMinimumSize(int(min_px[0]), int(min_px[1]))
        self._bound = []
        self._watched = []
        self._decor_w = 0.0
        self._title_w = 0.0

    def _want_w(self):
        z = getattr(self.figure, 'font_zoom', 1.0)
        return int(round(max(self.inner_w * z + self._decor_w, self._title_w + 2.0)))

    def sizeHint(self):
        if self.inner_w is None:
            return super().sizeHint()
        return QtCore.QSize(self._want_w(), max(self.minimumHeight(), 40))

    def minimumSizeHint(self):
        if self.inner_w is None:
            return super().minimumSizeHint()
        return QtCore.QSize(self._want_w(), self.minimumHeight())

    def _note_decor(self, left_right, title_w=0.0):
        """The tick axes and label, and the title, of the axes in a narrow cell (Axes._cell_px): the cell's width follows."""
        if self.inner_w is not None and (abs(left_right - self._decor_w) > 0.5 or abs(title_w - self._title_w) > 0.5):
            self._decor_w = float(left_right)
            self._title_w = float(title_w)
            self.updateGeometry()

    # --- the panel's shape (rule 11 of the guide: panels know their shape; the window arranges itself) ---

    def shape_ratio(self):
        """The height over width the panel asks for (its axes' locked aspect or box aspect), or None for a free panel; measured once per layout pass (a changed shape starts one: Axes._notify_shape)."""
        return pass_cached(self, 'shape', self._shape_now)

    def _shape_now(self):
        for obj in self._bound:
            want = getattr(obj, '_aspect_want', None)
            if want is not None:
                try:
                    r = want()
                except Exception:
                    r = None
                if r is not None and np.isfinite(r) and r > 0:
                    return float(r)
        return None

    def _decor(self):
        """The room around the panel's box (_measure_decor), measured once per layout pass (new_measure_pass): the arrangement asks a cell its size hundreds of times in one pass."""
        if self._decor_epoch == _EPOCH[0] and self._decor_val is not None:
            return self._decor_val
        self._decor_val = self._measure_decor()
        self._decor_epoch = _EPOCH[0]
        return self._decor_val

    def _measure_decor(self):
        """
        The room around the panel's box, (left, top, right, bottom) px: its axes' tick axes and title
        (Axes._decor_px); in an aligned row (Figure.align_row, a colour bar beside) the row's top and
        bottom, since the row's boxes lie in one band, below the tallest title and above the deepest x
        axis of the row (Axes._px_rect), and the panel's shape is its box's in that band.  A control
        box over a row member (controls.PanelWithBox: the member's cell sits under the box in one
        item) counts in that member's top for the others, since the band starts under the box.
        """
        ax = next((o for o in self._bound if hasattr(o, '_decor_px')), None)
        if ax is None:
            return (0.0, 0.0, 0.0, 0.0)

        def box_above(cell):
            f = getattr(cell.parentWidget(), '_box_h', None)
            return float(f()) if f is not None else 0.0
        try:
            left, top, right, bottom = ax._decor_px()
            row = getattr(ax, '_row', None)
            if row is not None:
                own = box_above(self)
                for m in row:
                    if m is ax or getattr(m, '_cell', None) is None or not m._visible or m.figure is not ax.figure:
                        continue
                    _, mt, _, mb = m._decor_px()
                    # The member's cell starts under its own box; the band's top, in this cell's terms,
                    # is the member's top shifted by the difference of the two boxes above.
                    mt += box_above(m._cell) - own
                    top, bottom = max(top, mt), max(bottom, mb)
            return (left, top, right, bottom)
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)

    def height_for_width(self, w):
        """The cell's height for a width `w`: the box at its ratio plus the room of the tick axes and the title; None for a free panel."""
        r = self.shape_ratio()
        if r is None:
            return None
        left, top, right, bottom = self._decor()
        return max(float(w) - left - right, 1.0) * r + top + bottom

    def width_for_height(self, h):
        """The cell's width for a height `h` (the inverse of height_for_width); None for a free panel."""
        r = self.shape_ratio()
        if r is None:
            return None
        left, top, right, bottom = self._decor()
        return max(float(h) - top - bottom, 1.0) / r + left + right

    def hasHeightForWidth(self):
        """Qt's question: a shaped panel (not a narrow colour-bar cell, which follows its row) answers from its ratio."""
        return self.inner_w is None and self.shape_ratio() is not None

    def heightForWidth(self, w):
        h = self.height_for_width(w) if self.inner_w is None else None
        return -1 if h is None else int(round(h))

    def bind(self, obj):
        """Place `obj` (it has _request_place) in this cell from now on."""
        if obj not in self._bound:
            self._bound.append(obj)
            new_measure_pass()                         # the cell's shape and room may be another's now
        self._watch_ancestors()

    def unbind(self, obj):
        try:
            self._bound.remove(obj)
            new_measure_pass()
        except ValueError:
            pass

    def rect_px(self):
        """The cell's rectangle (x0, y0, w, h) in canvas pixels from the top left."""
        vp = self.figure.canvas.viewport()
        p = self.mapTo(vp, QtCore.QPoint(0, 0)) if vp.isAncestorOf(self) else self.pos()
        return float(p.x()), float(p.y()), float(self.width()), float(self.height())

    def _changed(self):
        # Placed at once (not on a timer): two panels whose axes are linked must never be seen at
        # different geometries, or the link maps one's range onto the other's off by the difference.
        # (Batching the placements to the end of the event-loop turn was tried: it kept removed axes
        # alive into the next turn and, under load, surfaced a crash in the binding's override lookup.)
        for obj in list(self._bound):
            place = getattr(obj, '_place', None)
            if place is not None:
                place()
            else:
                obj._request_place()

    def _watch_ancestors(self):
        """A move of any widget holding the cell moves the cell on the canvas too: watched here."""
        vp = self.figure.canvas.viewport()
        w = self.parentWidget()
        while w is not None and w is not vp:
            if w not in self._watched:
                w.installEventFilter(self)
                self._watched.append(w)
            w = w.parentWidget()

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize):
            self._changed()
        return False

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._changed()

    def moveEvent(self, ev):
        super().moveEvent(ev)
        self._changed()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._watch_ancestors()
        self._changed()


class PanelRow(QtWidgets.QLayout):
    """
    One row of panels (rule 11 of the guide: panels know their shape; the window arranges
    itself).  A shaped item (a PanelCell of an aspect-locked axes, a controls.PanelWithBox, a
    RowWidget: any widget or layout with height_for_width / width_for_height answering a number)
    takes the width its ratio gives at the row's height; a free item (a curve panel, a control)
    takes a share of the remaining width by its stretch; a fixed item (a colour-bar cell, a box
    with a fixed width policy) its own width.  The row's natural height (heightForWidth) is the
    height at which every shaped item, at its nominal share of the width (its stretch among the
    shaped and free items), is as large as that share allows: the layout above gives the row that
    height, and no shaped item leaves an empty band above or below it.  Given more height (a
    taller window), the shaped items grow until their widths fill the row (the free items at their
    minimum widths); given less, they shrink to the height and the free items widen.  A free item
    always takes the row's full height; a shaped item is centred in a row taller than it.
    """

    def __init__(self, spacing=8):
        super().__init__()
        self._items = []            # [QLayoutItem, stretch]
        self._sp = int(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # --- QLayout ---

    def addItem(self, item):
        self._items.append([item, 1])

    def addWidget(self, w, stretch=1):
        w = w.qt if hasattr(w, 'qt') else w
        super().addWidget(w)                       # addChildWidget + addItem
        self._items[-1][1] = int(stretch)
        self.invalidate()

    def insertWidget(self, index, w, stretch=1):
        self.addWidget(w, stretch)
        if 0 <= index < len(self._items) - 1:
            self._items.insert(index, self._items.pop())
            self.invalidate()

    def addLayout(self, lay, stretch=1):
        self.addChildLayout(lay)
        self.addItem(lay)
        self._items[-1][1] = int(stretch)
        self.invalidate()

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i][0] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        if 0 <= i < len(self._items):
            it = self._items.pop(i)[0]
            self.invalidate()
            return it
        return None

    def expandingDirections(self):
        return Qt.Orientation.Horizontal | Qt.Orientation.Vertical

    def hasHeightForWidth(self):
        return any(self._shaped(it) is not None for it, _ in self._visible())

    def heightForWidth(self, w):
        return int(round(self._natural_height(float(w))))

    def minimumHeightForWidth(self, w):
        return self.minimumSize().height()

    def sizeHint(self):
        w = max(self.minimumSize().width(), 1)
        return QtCore.QSize(int(w), int(round(self._natural_height(float(w)))))

    def minimumSize(self):
        items = self._visible()
        w = sum(self._min_w(it) for it, _ in items) + self._sp * max(len(items) - 1, 0)
        h = max([self._min_h(it) for it, _ in items] + [0.0])
        m = self.contentsMargins()
        return QtCore.QSize(int(w + m.left() + m.right()), int(h + m.top() + m.bottom()))

    # --- the kinds of items ---

    def _visible(self):
        return [(it, st) for it, st in self._items if not (it.widget() is not None and it.widget().isHidden())]

    @staticmethod
    def _shaped(it):
        """(height_for_width, width_for_height) of an item shaped now, else None."""
        w = it.widget()
        if w is None:
            lay = it.layout()
            w = lay if lay is not None and hasattr(lay, 'height_for_width') else None
        if w is None or not hasattr(w, 'height_for_width'):
            return None
        try:
            if w.height_for_width(100.0) is None:
                return None
        except Exception:
            return None
        return w.height_for_width, w.width_for_height

    @staticmethod
    def _fixed(it):
        w = it.widget()
        if w is None:
            return False
        return w.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Fixed or w.maximumWidth() < 16777215

    @staticmethod
    def _min_w(it):
        return float(max(it.minimumSize().width(), 1))

    @staticmethod
    def _min_h(it):
        return float(max(it.minimumSize().height(), 1))

    @staticmethod
    def _fixed_w(it):
        return float(max(it.sizeHint().width(), it.minimumSize().width()))

    # --- the sizing rule ---

    def _parts(self, W):
        items = self._visible()
        shaped = [(it, st) for it, st in items if self._shaped(it) is not None]
        fixed = [(it, st) for it, st in items if self._shaped(it) is None and self._fixed(it)]
        free = [(it, st) for it, st in items if self._shaped(it) is None and not self._fixed(it)]
        avail = W - sum(self._fixed_w(it) for it, _ in fixed) - self._sp * max(len(items) - 1, 0)
        return items, fixed, shaped, free, max(avail, 0.0)

    def _natural_height(self, W):
        """The row's natural height at width W: every shaped item at its nominal share of the width, the tightest decides."""
        items, fixed, shaped, free, avail = self._parts(W)
        floor = max([self._min_h(it) for it, _ in items] + [0.0])
        if not shaped:
            return floor
        total = float(sum(st for _, st in shaped) + sum(st for _, st in free)) or 1.0
        h = min(self._shaped(it)[0](max(avail * st / total, 1.0)) for it, st in shaped)
        return max(h, floor)

    def _fit_height(self, W):
        """The largest height at which the shaped items' widths plus the free items' minimum widths fit W."""
        items, fixed, shaped, free, avail = self._parts(W)
        room = avail - sum(self._min_w(it) for it, _ in free)
        lo, hi = 1.0, 1.0e6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            need = sum(self._shaped(it)[1](mid) for it, _ in shaped)
            if need <= room:
                lo = mid
            else:
                hi = mid
        return lo

    def setGeometry(self, rect):
        super().setGeometry(rect)
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        W, H = float(r.width()), float(r.height())
        items, fixed, shaped, free, avail = self._parts(W)
        if not items:
            return
        widths = {}
        if shaped:
            h = min(H, self._fit_height(W))
            for it, _ in shaped:
                widths[id(it)] = min(self._shaped(it)[1](h), avail)
            rest = max(avail - sum(widths.values()), 0.0)
        else:
            h = H
            rest = avail
        tot_free = float(sum(st for _, st in free)) or 1.0
        for it, st in free:
            widths[id(it)] = max(rest * st / tot_free, self._min_w(it))
        for it, _ in fixed:
            widths[id(it)] = self._fixed_w(it)
        total_w = sum(widths.values()) + self._sp * max(len(items) - 1, 0)
        scale = W / total_w if total_w > W and total_w > 0 else 1.0      # nothing fits: everything scaled together
        shaped_ids = {id(it) for it, _ in shaped}
        x = float(r.x())
        for it, _ in items:
            w = widths[id(it)] * scale
            if id(it) in shaped_ids:
                ih = min(h, H)
                it.setGeometry(QtCore.QRect(int(round(x)), int(round(r.y() + 0.5 * (H - ih))), int(round(w)), int(round(ih))))
            else:
                it.setGeometry(QtCore.QRect(int(round(x)), r.y(), int(round(w)), int(round(H))))
            x += w + self._sp

    # --- as a shaped item of a row or pane above (RowWidget, controls.OrientedPanes) ---

    def height_for_width(self, w):
        return self._natural_height(float(w)) if self.hasHeightForWidth() else None

    def width_for_height(self, h):
        items, fixed, shaped, free, avail = self._parts(1.0e9)
        if not shaped:
            return None
        return (sum(self._shaped(it)[1](float(h)) for it, _ in shaped) + sum(self._fixed_w(it) for it, _ in fixed)
                + sum(self._min_w(it) for it, _ in free) + self._sp * max(len(items) - 1, 0))


class RowWidget(QtWidgets.QWidget):
    """A PanelRow as one widget: a shaped item of a row or pane above it (the Map view's two topography panels side by side)."""

    def __init__(self, row):
        super().__init__()
        self.row = row
        self.setLayout(row)

    def height_for_width(self, w):
        return self.row.height_for_width(w)

    def width_for_height(self, h):
        return self.row.width_for_height(h)

    def hasHeightForWidth(self):
        return self.row.hasHeightForWidth()

    def heightForWidth(self, w):
        return self.row.heightForWidth(w)


class _Slot(QtWidgets.QWidget):
    """
    A subplot's slot in the automatic grid: its PanelCell and the colour-bar cells put beside it
    (Figure.colorbar) in one row; shaped as its panel is (the bars follow the row's height), so a
    PanelRow sizes the slot from the panel's ratio.
    """

    def __init__(self, fig):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.cell = PanelCell(fig)
        self.cell._slot = self
        lay.addWidget(self.cell, 1)
        self._cells = [self.cell]
        self._col = 0
        self._row = None

    def _bars_w(self):
        bars = [c for c in self._cells[1:] if not c.isHidden()]
        return sum(float(c.sizeHint().width()) for c in bars) + self.layout().spacing() * len(bars)

    def height_for_width(self, w):
        return self.cell.height_for_width(max(float(w) - self._bars_w(), 1.0))

    def width_for_height(self, h):
        w = self.cell.width_for_height(h)
        return None if w is None else w + self._bars_w()

    def hasHeightForWidth(self):
        return self.cell.hasHeightForWidth()

    def heightForWidth(self, w):
        h = self.height_for_width(w)
        return -1 if h is None else int(round(h))


_SWAP = QtGui.QTransform(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)        # (x, y) -> (y, x): the turn of a rotated axes


class _Rotor(pg.ItemGroup):
    """
    The turn of a rotated axes (Axes.set_rotated): every item drawn into the view is its child
    and its transform maps data (x, y) to view (y, x); with the view's x axis inverted (Axes
    keeps it so while rotated) the picture is turned 90 degrees anticlockwise on screen, x up
    and y to the left: a rotation, never a mirror.  No contents of its own (pyqtgraph's ItemGroup).
    """


class Axes:
    """
    One axes: a pyqtgraph ViewBox placed in its cell of the window's
    layout (`cell`, a PanelCell: automatic layout, the one rule for every
    window; the cell holds the box with its tick axes and title), or at a
    figure-fraction rectangle (`rect`, or a SubplotSpec `spec`; the
    standalone drawings that keep matplotlib's own placement), or inside
    `inset_parent` (matplotlib's axes box, framed), its x and y tick axes outside the box
    (_Axis: every tick drawn where the view box maps its value),
    the title above it.  The plotting methods return artists with
    matplotlib's interface (remove, set_visible, get_label, ...).  sharex /
    sharey link the view to another axes' (a shared-y axes shows no y tick
    values).  The mouse pans and zooms only while the toolbar's Pan or Zoom
    is on; otherwise the windows handle their own mouse events.
    """

    def __init__(self, fig, rect=None, spec=None, sharex=None, sharey=None, inset_parent=None, label='', cell=None):
        self._place_pending = False    # set first: the axis items' zoom below may already ask for a placement
        self.figure = fig
        self._rect = list(rect) if rect is not None else None
        self._spec = spec
        self._cell = cell              # the PanelCell of the window's layout holding this axes (None: rect / spec)
        self._inset_parent = inset_parent
        self._insets = []
        scene = fig.canvas.scene()
        self.vb = _ViewBox(self)
        self.vb.setBorder(pg.mkPen('k', width=max(pt2px(0.8), 1)))
        self.vb.setBackgroundColor('w')
        scene.addItem(self.vb)
        self.xaxis_item = _Axis('bottom', self)
        self.yaxis_item = _Axis('left', self)
        for a in (self.xaxis_item, self.yaxis_item):
            a.linkToView(self.vb)
            scene.addItem(a)
            a.apply_zoom(fig.font_zoom)
        self._side_axes = []           # right / secondary axes
        self._title = None
        self._title_full = ''          # the title as given (set_title); the shown part follows the window's shape
        self._title_size = 12.0
        self.title = _TitleProxy(self)
        self.transData = _Transform('data', self)
        self.transAxes = _Transform('axes', self)
        self.patch = _Patch(self)
        self.xaxis = _AxisProxy(self, 'x')
        self.yaxis = _AxisProxy(self, 'y')
        self._facecolor = (1.0, 1.0, 1.0, 1.0)
        self._children = []            # artists in the order added
        self._axes_texts = []          # Text in axes coordinates (placed with the axes)
        self._legend = None
        self._extra_legends = []       # legends kept beside the current one (add_artist(legend)): (item, args, kw)
        self._visible = True
        self._axison = True
        self._frame = True
        self._cycle = 0
        self._tight = False
        self._xlabel = ''
        self._ylabel = ''
        self._colorbar = None
        self._is_colorbar = False
        self._home = None
        # The view changed with the toolbar's Pan or Zoom since its last Home or since the window last set
        # its limits (_nav_follow, _nav_clear): its view line in the info panel, under this key, meanwhile.
        self._nav_moved = False
        self._nav_pending = False
        self._nav_key = f"view {id(self)}"
        self.vb.sigRangeChanged.connect(self._nav_follow)
        self._aspect = 'auto'
        self._rotated = False          # set_rotated: the axes turned 90 degrees on screen (the data stays as given)
        self._rotor_item = None        # the turned group every item sits in while rotated (_rotor)
        # The saves' hooks, set by the window that owns the axes.  export_extent: None, or a callable
        # returning (content_h_px, (y_top, y_bottom)); the vector saves (mplexport) then draw this axes
        # from its on-screen top at the height max(on-screen h, content_h_px) logical pixels, its y
        # limits (y_top, y_bottom), the canvas part of the file down to the lowest such bottom and the
        # info block below it.  csv: False leaves this axes out of export_csv.
        self.export_extent = None
        self.csv = True
        self._shared_x = sharex
        self._shared_y = sharey
        if sharex is not None:
            self.vb.setXLink(sharex.vb)
        if sharey is not None:
            self.vb.setYLink(sharey.vb)
            # The shared y values are labelled once, on the axes they were shared from.
            self.yaxis_item.setStyle(showValues=False)
            self.yaxis_item._dirty = True
        self._apply_toolbar_mode(fig.canvas.toolbar.mode if fig.canvas.toolbar is not None else '')
        if inset_parent is not None:
            inset_parent._insets.append(self)
        fig._axes.append(self)
        if cell is not None:
            cell.bind(self)
        self._place()

    # --- artists bookkeeping ---

    @property
    def images(self):
        return [a for a in self._children if isinstance(a, AxesImage)]

    @property
    def patches(self):
        """matplotlib's ax.patches: the layer draws no patch artists (spans are items of their own), so always empty."""
        return []

    @property
    def lines(self):
        return [a for a in self._children if isinstance(a, (Line2D, _RefLine))]

    @property
    def collections(self):
        return [a for a in self._children if isinstance(a, (LineCollection, PolyCollection, QuadMesh, PathCollection,
                                                            ContourSet, TriContourSet))]

    @property
    def texts(self):
        """The axes' texts, annotations included (matplotlib's Annotation is a Text)."""
        return [a for a in self._children if isinstance(a, (Text, Annotation))]

    def get_lines(self):
        return self.lines

    def get_children(self):
        return list(self._children)

    def _register(self, artist):
        self._children.append(artist)
        if isinstance(artist, (AxesImage, QuadMesh, ContourSet, TriContourSet)) and not self._tight:
            self._tight = True
            self.vb.setDefaultPadding(0.0)
        if self._cell is not None and self.vb.state['aspectLocked']:
            self._notify_shape()               # new data may change the drawn extent's ratio
        return artist

    def _forget(self, artist):
        try:
            self._children.remove(artist)
        except ValueError:
            pass
        try:
            self._axes_texts.remove(artist)
        except ValueError:
            pass

    def _next_color(self):
        c = TAB10[self._cycle % len(TAB10)]
        self._cycle += 1
        return c

    # --- geometry ---

    def get_position(self, original=False):
        if self._cell is not None:
            # A panel in the window's layout: its box as the cell gives it, in figure fractions.
            W, H = self.figure._canvas_size()
            x0, y0, w, h = self._cell_px()
            return Bbox(x0 / W, 1.0 - (y0 + h) / H, w / W, h / H)
        if self._inset_parent is not None:
            px0, py0, pw, ph = self._inset_parent.get_position().bounds
            x, y, w, h = self._rect
            return Bbox(px0 + x * pw, py0 + y * ph, w * pw, h * ph)
        if self._rect is not None:
            return Bbox(*self._rect)
        if self._spec is not None:
            return self._spec.get_position(self.figure)
        return Bbox(0.125, 0.11, 0.775, 0.77)

    def set_position(self, pos, which='both'):
        b = pos.bounds if hasattr(pos, 'bounds') else tuple(pos)
        if self._cell is not None:     # an explicit rectangle takes the axes out of its layout cell
            self._cell.unbind(self)
            self._cell = None
        if self._inset_parent is not None:
            self._inset_parent = None
        self._rect = [float(v) for v in b]
        self._place()

    def get_subplotspec(self):
        return self._spec

    def _decor_px(self):
        """
        The room the box's surroundings take inside its layout cell, (left, top, right, bottom) in
        pixels: the y tick axis with its label on the left, the title band on top (lifted over a
        power-of-ten label as _place lifts it), the side axes on the right, the x tick axis with
        its label at the bottom.
        """
        xa, ya = self.xaxis_item, self.yaxis_item
        left = float(ya.maximumWidth()) if ya.isVisible() else 0.0
        bottom = float(xa.maximumHeight()) if xa.isVisible() else 0.0
        right = float(sum(a.maximumWidth() for a in self._side_axes if a.isVisible()))
        top = 0.0
        t = self._title
        if t is not None and t.get_text() and t.item.isVisible():
            lift = 15.0 if any(getattr(a, '_had_sci', False) for a in [ya] + self._side_axes) else 0.0
            top = float(t.item.boundingRect().height()) + abs(t._offset_px[1]) + lift
        elif ya.isVisible() or self._side_axes:
            top = 0.5 * float(QtGui.QFontMetricsF(qfont(10.0, zoom=self.figure.font_zoom)).height())   # the top tick value's half
        # The room only grows at one text zoom: the box keeps its size while tick values come and go
        # (a pan, a zoom, a new file), as a figure-fraction box did; a new zoom measures afresh.
        z = self.figure.font_zoom
        cur = (left, top, right, bottom)
        prev = getattr(self, '_decor_max', None)
        if prev is not None and getattr(self, '_decor_zoom', None) == z:
            cur = tuple(max(a, b) for a, b in zip(cur, prev))
        self._decor_max, self._decor_zoom = cur, z
        return cur

    def _cell_px(self):
        """
        The axes' cell in canvas pixels from the top left, (x0, y0, w, h): the box's room in its
        layout cell (the cell less _decor_px), or its figure-fraction rectangle.
        """
        if self._cell is not None:
            cell = self._cell
            cx, cy, cw, ch = cell.rect_px()
            left, top, right, bottom = self._decor_px()
            if cell.inner_w is not None:
                # A narrow panel (a colour bar, a ramp): its box keeps its own width, the box with its
                # tick axes centred in the cell, which is wide enough for them and for the title.
                t = self._title
                tw = float(t.item.boundingRect().width()) if t is not None and t.get_text() else 0.0
                cell._note_decor(left + right, tw)
                bw = cell.inner_w * self.figure.font_zoom
                off = max(0.5 * (cw - (left + bw + right)), 0.0)
                return (cx + off + left, cy + top, max(bw, 1.0), max(ch - top - bottom, 1.0))
            return (cx + left, cy + top, max(cw - left - right, 1.0), max(ch - top - bottom, 1.0))
        W, H = self.figure._canvas_size()
        x, y, w, h = self.get_position().bounds
        return (x * W, (1.0 - y - h) * H, w * W, h * H)

    def set_box_aspect(self, aspect=None):
        """matplotlib's box aspect: the box's height over its width fixed at `aspect` whatever the limits (None frees it); the box shrinks in its cell to it."""
        self._box_aspect = None if aspect is None else float(aspect)
        self._notify_shape()
        self._request_place()

    def get_box_aspect(self):
        return getattr(self, '_box_aspect', None)

    def _aspect_want(self):
        """The box's height over its width that a box aspect (set_box_aspect) or an adjustable-'box' locked aspect asks for, from the limits set; None when the box is free."""
        ba = getattr(self, '_box_aspect', None)
        if ba is not None:
            return ba
        if not (getattr(self, '_adjust_box', False) and self.vb.state['aspectLocked']):
            return None
        tx, ty = getattr(self, '_target_x', None), getattr(self, '_target_y', None)
        if tx is not None and not tx[1] > tx[0]:
            tx = None
        if ty is not None and not ty[1] > ty[0]:
            ty = None
        vx, vy = (ty, tx) if self._rotated else (tx, ty)      # the view's x and y spans (data y and x on a rotated axes)
        if vx is None or vy is None:
            # No limits set on an axis: the drawn data's extent along it (what the autoscaled view
            # shows, in the view's own axes) gives that side of the ratio; a crop of one axis alone
            # changes the shape too.
            try:
                r = self.vb.childrenBoundingRect()
            except Exception:
                return None
            if vx is None:
                if not r.width() > 0:
                    return None
                vx = (float(r.left()), float(r.right()))
            if vy is None:
                if not r.height() > 0:
                    return None
                vy = (float(r.top()), float(r.bottom()))
        return (vy[1] - vy[0]) / (vx[1] - vx[0]) * float(self.vb.state['aspectLocked'])

    def _notify_shape(self):
        """
        The panel's cell told when its shape (its height over its width, _aspect_want) changed,
        so that the rows and panes above size themselves again (rule 11 of the guide: panels know
        their shape; the window arranges itself).
        """
        cell = self._cell
        if cell is None:
            return
        r = self._aspect_want()
        key = None if r is None else round(float(r), 4)
        if key != getattr(self, '_shape_told', 'never'):
            self._shape_told = key
            new_measure_pass()                         # the shape kept for this pass (PanelCell.shape_ratio) is old
            cell.updateGeometry()

    def _own_height(self):
        """The height (px) the box takes on its own: its cell's, shrunk to its locked aspect (adjustable 'box')."""
        x0, y0, w, h = self._cell_px()
        want = self._aspect_want()
        if want is not None and w > 0 and h > 0 and h / w > want:
            return w * want
        return h

    def _px_rect(self):
        """
        The box as drawn, (x0, y0, w, h) in canvas pixels from the top left: the figure-fraction
        rectangle, at the height of its row when it is in one (Figure.align_row: the smallest height
        a locked aspect among the row's visible panels asks for), else shrunk for its own
        adjustable-'box' aspect; a locked aspect then sets the width; the box stays centred on its
        cell; then moved clear of keep_clear_of's other axes.
        """
        x0, y0, w, h = self._cell_px()
        row = getattr(self, '_row', None)
        if row is not None:
            members = [m for m in row if m._visible and m.figure is self.figure]
            if self._cell is not None and members and all(m._cell is not None for m in members):
                # A row of panels in layout cells: every box in the band all of them have room for
                # (below the tallest title, above the deepest x axis), so the boxes share one top and
                # one bottom (the row rule; linked axes then show the same range).
                bands = [m._cell_px() for m in members]
                top = max(b[1] for b in bands)
                bot = min(b[1] + b[3] for b in bands)
                if bot > top:
                    y0, h = top, bot - top
                hh = h
                for m, b in zip(members, bands):
                    want = m._aspect_want()
                    if want is not None and b[2] > 0:
                        hh = min(hh, b[2] * want)
            else:
                hh = min([m._own_height() for m in members] + [h])
        else:
            hh = self._own_height()
        fx, fy = getattr(self, '_anchor_f', (0.5, 0.5))
        if hh < h and w > 0 and h > 0:
            y0, h = y0 + fy * (h - hh), hh
        want = self._aspect_want()
        if want is not None and w > 0 and h > 0:
            # matplotlib's adjustable='box': the box shrinks to the data's aspect, about its anchor
            # (set_anchor; the centre unless set).
            nw = h / want
            if nw < w:
                x0, w = x0 + fx * (w - nw), nw
        r = (x0, y0, w, h)
        other = getattr(self, '_clear_of', None)
        if other is not None and other._visible:
            # This axes' left tick labels start right of the other axes' right-hand labels.
            ox, oy, ow, oh = other._px_rect()
            right = ox + ow + sum(a.maximumWidth() for a in other._side_axes if a.isVisible())
            need = right + self._clear_pad + (self.yaxis_item.maximumWidth() if self.yaxis_item.isVisible() else 0)
            if r[0] < need:
                r = (need, r[1], max(r[2] - (need - r[0]), 10.0), r[3])
        return r

    def _refont(self, zoom):
        """Every text of the axes at the figure's new zoom."""
        for a in [self.xaxis_item, self.yaxis_item] + self._side_axes:
            a.apply_zoom(zoom)
        if self._title is not None:
            self._title._apply()
        for t in self._axes_texts:
            t._apply()
        for a in self._children:
            if isinstance(a, Text):
                a._apply()
            elif isinstance(a, Annotation) and a.text is not None:
                a.text._apply()
        if self._legend is not None and getattr(self, '_legend_args', None) is not None:
            args, kw = self._legend_args
            self.legend(*args, **kw)
            if getattr(self, '_legend_clear', False):
                self.keep_legend_clear()
        if self._extra_legends:
            self._rebuild_extra_legends()

    def _refit_legend(self):
        if self._legend is None or getattr(self, '_legend_args', None) is None:
            return
        args, kw = self._legend_args
        self.legend(*args, **kw)
        if getattr(self, '_legend_clear', False):
            self.keep_legend_clear()

    def keep_clear_of(self, other, pad_pt=6.0):
        """Keep this axes (with its y tick labels and label) right of `other` and its colour-bar labels."""
        self._clear_of = other
        self._clear_pad = pt2px(pad_pt)
        other._dependents = getattr(other, '_dependents', []) + [self]
        self._place()

    def keep_legend_clear(self, pad_px=4.0):
        """
        The y limits raised so that no curve reaches the legend (an upper legend): the visible
        curves' range with 5 % of it below and 2 % above (matplotlib's margins), then the top
        raised so that the legend's height (plus `pad_px`, at most 0.8 of the panel) lies above the
        curves.  Marks the axes so that _place and _refont re-run it when the panel's height or the
        zoom changes.  Nothing happens without a legend or without curves.
        """
        leg = self._legend
        ys = []
        for a in self._children:
            if isinstance(a, Line2D) and a.get_visible():
                y = a.get_ydata()
                y = y[np.isfinite(y)]
                if len(y):
                    ys.append((float(y.min()), float(y.max())))
        if leg is None or not ys:
            return
        lo, hi = min(v[0] for v in ys), max(v[1] for v in ys)
        span = (hi - lo) if hi > lo else (abs(hi) or 1.0)
        lo -= 0.05 * span
        hi += 0.02 * span
        x0, y0, w, h = self._px_rect()
        lh = float(leg.geometry().height()) + 4.0 + pad_px
        f = min(lh / max(h, 1.0), 0.8)
        top = (hi - f * lo) / (1.0 - f)
        self.vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._target_y = (lo, top)
        self.vb.setYRange(lo, top, padding=0)
        self._store_home()
        self._legend_clear = True

    def get_window_extent(self, renderer=None):
        x0, y0, w, h = self._px_rect()
        W, H = self.figure._canvas_size()
        return Bbox(x0, H - (y0 + h), w, h)

    def on_placed(self, callback):
        """Run callback() after every placement of the axes (_place: a resize, a subplots_adjust, a text zoom, an axis re-measured); a window's controls that sit relative to the panel follow it this way."""
        self.__dict__.setdefault('_placed_cbs', []).append(callback)

    def xlabel_row(self):
        """
        The band of the x label at the foot of the bottom axis item, under
        the tick marks and their values: (centre, height) in figure
        fractions from the bottom; None without a shown label.
        """
        xa = self.xaxis_item
        try:
            if getattr(self.figure, '_closed', False) or not (xa.isVisible() and xa.label.isVisible()):
                return None
            W, H = self.figure._canvas_size()
            x0, y0, w, h = self._px_rect()
            lh = float(xa.label.boundingRect().height()) * 0.8      # pyqtgraph's share of the label in the axis height
            yc = y0 + h + float(xa.maximumHeight()) - 0.5 * lh
        except RuntimeError:                                       # the window's items already gone
            return None
        return 1.0 - yc / float(H), lh / float(H)

    def _request_place(self):
        if not self._place_pending:
            self._place_pending = True
            QtCore.QTimer.singleShot(0, self._place)

    def _place(self):
        """
        Place the view box, the tick and side axes, the title (lifted over a power-of-ten label),
        the axes texts, insets and dependent axes at _px_rect; a legend is refitted and
        keep_legend_clear re-run on a size change.
        """
        self._place_pending = False
        new_measure_pass()                             # a placement: the cells' room may have changed
        if getattr(self.figure, '_closed', False):
            return                                     # a placement queued (a row member re-placed) after the window closed
        self._notify_shape()
        x0, y0, w, h = self._px_rect()
        w, h = max(w, 1.0), max(h, 1.0)
        self.vb.setGeometry(QtCore.QRectF(x0, y0, w, h))
        self._placed_rect = (round(x0, 2), round(y0, 2), round(w, 2), round(h, 2))
        row = getattr(self, '_row', None)
        if row is not None and self._cell is not None and not getattr(self.figure, '_row_placing', False):
            # The row's band depends on every member's cell and axes: a member placed with an older band
            # is placed again at once (the boxes of a row share one top and one bottom).
            self.figure._row_placing = True
            try:
                for m in row:
                    if m is self or m._cell is None or not m._visible or m.figure is not self.figure:
                        continue
                    mx, my, mw, mh = m._px_rect()
                    if getattr(m, '_placed_rect', None) != (round(mx, 2), round(my, 2), round(max(mw, 1.0), 2),
                                                            round(max(mh, 1.0), 2)):
                        m._place()
            finally:
                self.figure._row_placing = False
        if self.vb.state['aspectLocked'] and getattr(self, '_target_x', None) is not None \
                and getattr(self, '_target_y', None) is not None and not self.figure.canvas.toolbar.mode:
            self._apply_targets()
        xa, ya = self.xaxis_item, self.yaxis_item
        for a in [xa, ya] + self._side_axes:
            a.picture = None
            a.update()
        aw = ya.maximumWidth() if ya.isVisible() else 0
        ya.setGeometry(QtCore.QRectF(x0 - aw, y0, aw, h))
        ah = xa.maximumHeight() if xa.isVisible() else 0
        xa.setGeometry(QtCore.QRectF(x0, y0 + h, w, ah))
        xr = x0 + w
        for a in self._side_axes:
            sw = a.maximumWidth() if a.isVisible() else 0
            a.setGeometry(QtCore.QRectF(xr, y0, sw, h))
            xr += sw
        if self._title is not None:
            lift = 15.0 if any(getattr(a, '_had_sci', False) for a in [self.yaxis_item] + self._side_axes) else 0.0
            base = getattr(self._title, '_base_offset', None)
            if base is None:
                base = self._title._base_offset = self._title.item._offset.y()
            self._title.item._offset = QtCore.QPointF(self._title.item._offset.x(), base - lift)
            self._title.item.updateTextPos()
            self._title._place()
        for t in self._axes_texts:
            t._place()
        for ins in self._insets:
            ins._place()
        for d in getattr(self, '_dependents', []):
            d._request_place()
        # A legend is refitted to the panel's new width (it never sticks out of the panel).
        if self._legend is not None and getattr(self, '_legend_args', None) is not None \
                and abs(getattr(self, '_legend_w', w) - w) > 2:
            QtCore.QTimer.singleShot(0, self._refit_legend)
        self._legend_w = w
        if getattr(self, '_legend_clear', False) and getattr(self, '_last_h', None) != round(h):
            self._last_h = round(h)
            QtCore.QTimer.singleShot(0, self.keep_legend_clear)
        if getattr(self.figure, '_closed', False):
            return                                     # a placement queued before the window closed: nothing follows it
        for cb in list(getattr(self, '_placed_cbs', [])):
            try:
                cb()
            except Exception:
                traceback.print_exc()

    def _view_range(self):
        if self.vb.state['autoRange'][0] or self.vb.state['autoRange'][1]:
            try:
                self.vb.updateAutoRange()
            except Exception:
                pass
        return self.vb.viewRange()

    # --- visibility, frame ---

    def _all_items(self):
        items = [self.vb, self.xaxis_item, self.yaxis_item] + list(self._side_axes)
        if self._title is not None:
            items.append(self._title.item)
        items.extend(t.item for t in self._axes_texts)
        return items

    def set_visible(self, flag):
        self._visible = bool(flag)
        self.vb.setVisible(self._visible)
        self.xaxis_item.setVisible(self._visible and self._axison)
        self.yaxis_item.setVisible(self._visible and self._axison and not self._is_colorbar)
        for a in self._side_axes:
            a.setVisible(self._visible)
        if self._title is not None:
            self._title.item.setVisible(self._visible and self._title.get_visible())
        for t in self._axes_texts:
            t.item.setVisible(self._visible and t.get_visible())
        for ins in self._insets:
            ins.set_visible(flag)
        self._place()

    def get_visible(self):
        return self._visible

    def set_frame_on(self, b):
        self._frame = bool(b)
        self.vb.setBorder(pg.mkPen('k', width=max(pt2px(0.8), 1)) if self._frame and self._axison else None)

    def axis(self, arg=None, **kw):
        if arg in ('off', False):
            self._axison = False
            self.xaxis_item.setVisible(False)
            self.yaxis_item.setVisible(False)
            self.vb.setBorder(None)
            self.vb.setBackgroundColor(None)
        elif arg in ('on', True):
            self._axison = True
            self.xaxis_item.setVisible(self._visible)
            self.yaxis_item.setVisible(self._visible)
            self.set_frame_on(self._frame)
            self.vb.setBackgroundColor('w')
        elif arg in ('equal', 'scaled', 'image'):
            self.set_aspect('equal')
        self._place()

    def set_axis_off(self):
        self.axis('off')

    def set_axis_on(self):
        self.axis('on')

    def set_navigate(self, b):
        pass

    def get_navigate(self):
        return True

    # --- the view changed with the toolbar's Pan or Zoom: a line in the window's info panel ---

    def _nav_follow(self, *args):
        """
        The view box's range changed (pyqtgraph's sigRangeChanged).  During a drag or the wheel on a view box
        (_NAVIGATING) the change was made with the toolbar's Pan or Zoom: the panel's view is marked so
        (_nav_moved), unless the panel is out of Pan and Zoom (a colour bar, or get_navigate() False).  While it
        is marked, its view line follows every change of its limits (_nav_line, once per event-loop turn).
        """
        if _NAVIGATING[0] and self.get_navigate() and not self._is_colorbar:
            self._nav_moved = True
        if self._nav_moved and not self._nav_pending:
            self._nav_pending = True
            QtCore.QTimer.singleShot(0, self._nav_line)

    def _nav_line(self):
        """
        The panel's view line in the window's info panel (Figure.set_info_line) while its view is the one the
        toolbar's Pan or Zoom made (_nav_moved): the panel by its title, else by its place among the figure's
        panels ('panel k', as export_csv names it), then its x and y limits (get_xlim, get_ylim); none otherwise.
        """
        self._nav_pending = False
        fig = self.figure
        if getattr(fig, '_closed', False):
            return
        text = ''
        if self._nav_moved:
            name = " ".join(str(self._title_full).split())
            if not name:
                name = f"panel {fig._axes.index(self) + 1}" if self in fig._axes else "panel"
            (xa, xb), (ya, yb) = self.get_xlim(), self.get_ylim()
            text = f"view of {name}: x {xa:.5g} to {xb:.5g}, y {ya:.5g} to {yb:.5g}"
        fig.set_info_line(self._nav_key, text)

    def _nav_clear(self):
        """The panel's view is no longer the toolbar's (Home, or the window set its limits again): its view line removed."""
        if self._nav_moved:
            self._nav_moved = False
            self.figure.set_info_line(self._nav_key, '')

    def set_axes_locator(self, locator):
        """matplotlib's axes locator: not needed, a panel's place is its layout cell (automatic layout)."""
        pass

    def set_subplotspec(self, spec):
        self._spec = spec

    def set_zorder(self, z):
        pass

    def set_in_layout(self, flag):
        pass

    def set_facecolor(self, c):
        """The axes' background colour (ax.patch)."""
        self._facecolor = tuple(to_rgba(c))
        self.patch.set_facecolor(c)

    def get_facecolor(self):
        return self._facecolor

    def set(self, **kw):
        """matplotlib's Axes.set: set_<key>(value) for every keyword; xlim / ylim take a pair."""
        for k, v in kw.items():
            f = getattr(self, 'set_' + k)
            if k in ('xlim', 'ylim') and isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2:
                f(*v)
            else:
                f(v)

    def remove(self):
        fig = self.figure
        sc = fig.canvas.scene()
        row = getattr(self, '_row', None)
        if row is not None:                        # the row goes on without this axes
            try:
                row.remove(self)
            except ValueError:
                pass
            self._row = None
            for m in row:
                m._request_place()
        for ins in list(self._insets):
            ins.remove()
        if self._legend is not None:
            self._remove_legend()
        self._drop_extra_legends()
        for it in self._all_items():
            try:
                sc.removeItem(it)
            except Exception:
                pass
        if self._inset_parent is not None:
            try:
                self._inset_parent._insets.remove(self)
            except ValueError:
                pass
        try:
            fig._axes.remove(self)
        except ValueError:
            pass
        if self._cell is not None:
            self._cell.unbind(self)
            fig._auto_dispose(self._cell)
        fig._set_info(self, '')
        self._nav_clear()                          # its view line, when it had one
        fig.__dict__.get('_hover_labels', {}).pop(self, None)     # the hover label registered on this axes (widgets.HoverLabels)

    # --- clearing ---

    def clear(self):
        """
        Every artist, the title (and its info-panel line), the labels, the legend, the side axes
        (secondary and colour-bar axes), the ticks, grid, inversion and colour-bar state, the limits
        (autoscale on), and the aspect reset; the colour cycle restarts.
        """
        if self._legend is not None:
            self._remove_legend()
        self._drop_extra_legends()
        for a in list(self._children):
            try:
                a.remove()
            except Exception:
                pass
        self._children = []
        for t in list(self._axes_texts):
            try:
                t.remove()
            except Exception:
                pass
        self._axes_texts = []
        self.vb.clear()
        self._drop_rotor()
        if self._title is not None:
            try:
                self.figure.canvas.scene().removeItem(self._title.item)
            except Exception:
                pass
            self._title = None
            self._title_full = ''
            self.figure._set_info(self, '')
        sc = self.figure.canvas.scene()
        for a in self._side_axes:
            try:
                sc.removeItem(a)
            except Exception:
                pass
        self._side_axes = []
        self._cycle = 0
        self._tight = False
        self.vb.setDefaultPadding(0.05)
        self._is_colorbar = False
        self._colorbar = None
        for axis in (self.xaxis_item, self.yaxis_item):
            if getattr(axis, '_dirty', True):
                axis.setTicks(None)
                axis._mode = None
                axis.setStyle(showValues=not (axis is self.yaxis_item and self._shared_y is not None),
                              tickLength=int(round(pt2px(3.5))))
                axis.set_tick_pt(10)
                axis.setGrid(False)
                axis._dirty = False
            if axis.labelText:
                axis.setLabel('')
                axis.showLabel(False)
        self.xaxis_item.setVisible(self._visible and self._axison)
        self.yaxis_item.setVisible(self._visible and self._axison)
        self._xlabel = self._ylabel = ''
        self.vb.setAspectLocked(False)
        self._aspect = 'auto'
        self._adjust_box = False
        self._facecolor = (1.0, 1.0, 1.0, 1.0)
        if self.vb.state['xInverted'] != self._rotated:       # a rotated axes keeps its x axis inverted (y to the left)
            self.vb.invertX(self._rotated)
        if self.vb.state['yInverted']:
            self.vb.invertY(False)
        self.vb.enableAutoRange(x=True, y=True)
        # matplotlib's clear (Axes.__clear, Axes.sharex / sharey): an axis shared with another axes
        # keeps that axes' limits instead of autoscaling on this one's data.  pyqtgraph's link follows
        # the other view only when one of the two changes range or size, so it is synced here (which
        # also turns this view's autoscale off on that axis); otherwise this view's autoscale would
        # pull the shared range of every linked panel to its own data at its next limit change.
        for axis, src in ((pg.ViewBox.XAxis, getattr(self, '_shared_x', None)),
                          (pg.ViewBox.YAxis, getattr(self, '_shared_y', None))):
            if src is not None:
                self.vb.enableAutoRange(axis, False)
                self.vb.linkedViewChanged(src.vb, axis)
        self._home = None
        self._nav_clear()                      # the limits are the window's again: no view line
        self._target_x = self._target_y = None
        self._legend_clear = False
        self._place()

    cla = clear

    # --- labels and title ---

    def set_title(self, label, fontsize=None, loc='center', pad=None, fontdict=None, color=None, y=None, **kw):
        """
        The title above the axes: the full text is kept (_title_full) and the shown part is
        short_title(full, figure.portrait); when the two differ the full text goes to the figure's
        info panel (Figure._set_info).  `fontsize` in points (not given: the size last set through
        ax.title.set_fontsize or an earlier call, 12 at first), `loc` 'left' /
        'center' / 'right', `pad` in points (6), `y` the axes fraction of the baseline (1.0).
        """
        if fontdict:
            fontsize = fontdict.get('fontsize', fontsize)
        if self._title is not None:
            self._title.remove()
            self._title = None
        full = '' if label is None else str(label)
        self._title_full = full
        if self._nav_moved:
            self._nav_follow()                 # the view line names the panel by its new title
        label = short_title(full, self.figure.portrait)
        self.figure._set_info(self, full if full.strip() != label else '')
        size = font_pt(fontsize, self._title_size)
        self._title_size = size
        ha = {'left': 'left', 'right': 'right'}.get(loc, 'center')
        x = {'left': 0.0, 'right': 1.0}.get(loc, 0.5)
        pad_px = pt2px(6.0 if pad is None else pad)
        t = Text(self, x, 1.0 if y is None else y, label, 'axes', fontsize=size, ha=ha, va='bottom',
                 color=color or kw.get('c', 'k'), offset=(0, pad_px), fontweight=kw.get('fontweight'))
        try:
            self._axes_texts.remove(t)
        except ValueError:
            pass
        try:
            self._children.remove(t)
        except ValueError:
            pass
        t.item.setVisible(self._visible)
        self._title = t
        if self._cell is not None:
            self._request_place()          # in a layout cell the title takes room from the box (_decor_px)
        return t

    def get_title(self, loc='center'):
        return self._title.get_text() if self._title is not None else ''

    def _retitle(self):
        """The shown title re-derived from the full one for the figure's current shape (a file name: whole, or its stamp)."""
        if self._title is not None and self._title_full:
            self._title.set_text(short_title(self._title_full, self.figure.portrait))

    def _axis_label(self, axis, text, fontsize=None, color=None):
        text = '' if text is None else str(text)
        if not text:
            axis.setLabel('')
            axis.showLabel(False)
            return
        axis.setLabel(mathtext(short_label(text), html_out=True).replace('\n', '<br>'))
        axis.set_label_pt(10 if fontsize is None else fontsize)
        if color is not None:
            axis.label.setDefaultTextColor(qcolor(color))
        axis.showLabel(True)
        axis._adjustSize()

    def set_xlabel(self, label, fontsize=None, labelpad=None, color=None, **kw):
        self._xlabel = '' if label is None else str(label)
        self._axis_label(self._axis_item('x'), label, fontsize or kw.get('size'), color)

    def set_ylabel(self, label, fontsize=None, labelpad=None, color=None, **kw):
        self._ylabel = '' if label is None else str(label)
        self._axis_label(self._axis_item('y'), label, fontsize or kw.get('size'), color)

    def get_xlabel(self):
        return self._xlabel

    def get_ylabel(self):
        return self._ylabel

    def _make_side_axis(self, side='right'):
        a = _Axis(side, self)
        a.apply_zoom(self.figure.font_zoom)
        a.linkToView(self.vb)
        self.figure.canvas.scene().addItem(a)
        self._side_axes.append(a)
        a.setVisible(self._visible)
        self._place()
        return a

    def _side_label(self, axis, text, fontsize=None):
        self._axis_label(axis, text, fontsize)
        self._place()

    # --- ticks ---

    def set_xticks(self, ticks, labels=None, **kw):
        self._set_ticks(self._axis_item('x'), ticks, labels)

    def set_yticks(self, ticks, labels=None, **kw):
        self._set_ticks(self._axis_item('y'), ticks, labels)

    def _set_ticks(self, axis, ticks, labels):
        axis._dirty = True
        ticks = list(np.ravel(ticks)) if ticks is not None else []
        if len(ticks) == 0:
            axis.setTicks([[]])
            axis.setStyle(showValues=False, tickLength=0)
        else:
            labs = labels if labels is not None else [axis.tickStrings([t], 1.0, abs(ticks[-1] - ticks[0]) / max(len(ticks) - 1, 1) or 1.0)[0]
                                                      for t in ticks]
            axis.setTicks([[(float(t), mathtext(str(s))) for t, s in zip(ticks, labs)]])
        axis._adjustSize()
        self._place()

    def _set_tick_labels(self, axis, labels):
        """matplotlib's set_xticklabels / set_yticklabels: the texts of the ticks set before (set_xticks), in order; an empty list hides the values."""
        axis._dirty = True
        labels = list(labels)
        if len(labels) == 0:
            axis.setStyle(showValues=False)
        elif axis._tickLevels:
            ticks = [v for level in axis._tickLevels for v, _ in level]
            axis.setTicks([[(float(t), mathtext(str(s))) for t, s in zip(ticks, labels)]])
        axis._adjustSize()
        self._place()

    def set_xticklabels(self, labels, **kw):
        self._set_tick_labels(self.xaxis_item, labels)

    def set_yticklabels(self, labels, **kw):
        self._set_tick_labels(self.yaxis_item, labels)

    def tick_params(self, axis='both', labelsize=None, labelleft=None, labelbottom=None, labelright=None,
                    left=None, bottom=None, length=None, **kw):
        self.xaxis_item._dirty = self.yaxis_item._dirty = True
        targets = []
        if axis in ('both', 'x'):
            targets.append(self._axis_item('x'))
        if axis in ('both', 'y'):
            targets.append(self._axis_item('y'))
            targets.extend(self._side_axes)
        if labelsize is not None:
            for a in targets:
                a.set_tick_pt(labelsize)
                a._adjustSize()
        if labelleft is not None:
            self.yaxis_item.setStyle(showValues=bool(labelleft))
            self.yaxis_item._adjustSize()
        if labelbottom is not None:
            self.xaxis_item.setStyle(showValues=bool(labelbottom))
            self.xaxis_item._adjustSize()
        if length is not None:
            for a in targets:
                a.setStyle(tickLength=int(round(pt2px(length))))
        if left is False:
            self.yaxis_item.setStyle(tickLength=0)
            self.yaxis_item._adjustSize()
        if bottom is False:
            self.xaxis_item.setStyle(tickLength=0)
            self.xaxis_item._adjustSize()
        self._place()

    def ticklabel_format(self, **kw):
        pass

    def grid(self, visible=True, which='major', axis='both', alpha=None, **kw):
        if visible is None:
            visible = True
        # matplotlib's grid is light grey (#b0b0b0) at the given alpha over white
        val = int(255 * 0.45 * (1.0 if alpha is None else float(alpha))) if visible else False
        self.xaxis_item._dirty = self.yaxis_item._dirty = True
        if axis in ('both', 'x'):
            self.xaxis_item.setGrid(val)
        if axis in ('both', 'y'):
            self.yaxis_item.setGrid(val)

    def minorticks_off(self):
        pass

    # --- limits and aspect ---

    def _range_of(self, args, kw, lo_key, hi_key):
        if len(args) == 1 and np.ndim(args[0]) == 1:
            lo, hi = args[0]
        elif len(args) >= 2:
            lo, hi = args[0], args[1]
        elif len(args) == 1:
            lo, hi = args[0], None
        else:
            lo, hi = None, None
        lo = kw.get(lo_key, kw.get('left' if lo_key == 'xmin' else 'bottom', lo))
        hi = kw.get(hi_key, kw.get('right' if hi_key == 'xmax' else 'top', hi))
        return lo, hi

    def set_xlim(self, *args, **kw):
        lo, hi = self._range_of(args, kw, 'xmin', 'xmax')
        return self._set_lim('x', lo, hi)

    def set_ylim(self, *args, **kw):
        lo, hi = self._range_of(args, kw, 'ymin', 'ymax')
        return self._set_lim('y', lo, hi)

    def _set_lim(self, axis, lo, hi):
        """
        set_xlim / set_ylim of data axis `axis` ('x' / 'y'): the limits go to the view axis that
        shows it (_view_axis: the other one on a rotated axes, whose x axis is inverted for
        ascending data-y limits, so y runs to the left); kept in data terms (_target_x /
        _target_y) for _apply_targets and the shape.
        """
        cur = tuple(sorted(self._get_lim(axis)))
        lo = cur[0] if lo is None else float(lo)
        hi = cur[1] if hi is None else float(hi)
        inv = hi < lo
        a, b = (hi, lo) if inv else (lo, hi)
        if b == a:
            a, b = a - 0.5, b + 0.5
        r = _finite_range(a, b)
        if r is None:
            print(f"qtplot: {axis} limits ({lo}, {hi}) are not drawable; kept as they were.")
            return self._get_lim(axis)
        a, b = r
        view = self._view_axis(axis)
        want = inv != (self._rotated and view == 'x')
        if view == 'x':
            if want != bool(self.vb.state['xInverted']):
                self.vb.invertX(want)
            self.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
        else:
            if want != bool(self.vb.state['yInverted']):
                self.vb.invertY(want)
            self.vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        if axis == 'x':
            self._target_x = (a, b)
        else:
            self._target_y = (a, b)
        self._apply_targets()
        self._notify_shape()                   # new limits are a new shape for a panel that keeps its ratio (a crop)
        return (lo, hi)

    def _get_lim(self, axis):
        """get_xlim / get_ylim of data axis `axis`: the view axis showing it, in the data's own order (a rotated axes' inverted x axis reads as ascending y)."""
        view = self._view_axis(axis)
        a, b = self._view_range()[0 if view == 'x' else 1]
        inverted = bool(self.vb.state['xInverted' if view == 'x' else 'yInverted'])
        if self._rotated and view == 'x':
            inverted = not inverted
        return (b, a) if inverted else (a, b)

    def _apply_targets(self):
        """
        The limits set: with an equal aspect both ranges at once, the one too
        short for the box widened about its centre (matplotlib's 'datalim'),
        so neither requested range is cut.
        """
        tx = getattr(self, '_target_x', None)
        ty = getattr(self, '_target_y', None)
        tx = None if tx is None else _finite_range(*tx)
        ty = None if ty is None else _finite_range(*ty)
        vx, vy = (ty, tx) if self._rotated else (tx, ty)      # the view's ranges: data y along x, x along y on a rotated axes
        if self.vb.state['aspectLocked'] and vx is not None and vy is not None:
            x0, y0, w, h = self._px_rect()
            ratio = float(self.vb.state['aspectLocked'])
            dx, dy = vx[1] - vx[0], vy[1] - vy[0]
            if w > 0 and h > 0 and dx > 0 and dy > 0 and ratio > 0:
                # pyqtgraph's lock: data width / data height = (box width / box height) / ratio, so the
                # y units per pixel are `ratio` times the x units per pixel.  The x units per pixel
                # that hold both requested ranges, then both ranges widened about their centres.
                u = max(dx / w, dy / (h * ratio))
                cx, cy = 0.5 * (vx[0] + vx[1]), 0.5 * (vy[0] + vy[1])
                vx = (cx - 0.5 * u * w, cx + 0.5 * u * w)
                vy = (cy - 0.5 * u * ratio * h, cy + 0.5 * u * ratio * h)
            self.vb.setRange(xRange=vx, yRange=vy, padding=0)
            if getattr(self, '_adjust_box', False):
                self._place_box_only()
        elif vx is not None and vy is not None:
            self.vb.setRange(xRange=vx, yRange=vy, padding=0)
        elif vx is not None:
            self.vb.setXRange(*vx, padding=0)
        elif vy is not None:
            self.vb.setYRange(*vy, padding=0)
        self._store_home()

    def _store_home(self):
        (xa, xb), (ya, yb) = self.vb.viewRange()
        self._home = (xa, xb, ya, yb)
        self._nav_clear()                      # the window set the limits: no view line

    def get_xlim(self):
        return self._get_lim('x')

    def get_ylim(self):
        return self._get_lim('y')

    _ANCHORS = {'C': (0.5, 0.5), 'N': (0.5, 0.0), 'S': (0.5, 1.0), 'E': (1.0, 0.5), 'W': (0.0, 0.5),
                'NE': (1.0, 0.0), 'NW': (0.0, 0.0), 'SE': (1.0, 1.0), 'SW': (0.0, 1.0)}

    def set_anchor(self, anchor, share=False):
        """
        Where a box shrunk to its aspect (or to its row's height) sits in its room: matplotlib's
        anchors 'C' (centre, the default), 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'.  A window
        whose controls sit directly under a panel anchors the panel at 'S' (the drift window).
        """
        self._anchor_f = self._ANCHORS[str(anchor).upper()]
        self._request_place()

    def get_anchor(self):
        f = getattr(self, '_anchor_f', (0.5, 0.5))
        return next(k for k, v in self._ANCHORS.items() if v == f)

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        if anchor is not None:
            self.set_anchor(anchor)
        self._aspect = aspect
        was = getattr(self, '_adjust_box', False)
        self._adjust_box = adjustable in (None, 'box')
        if aspect == 'equal' or (isinstance(aspect, (int, float)) and not isinstance(aspect, bool)):
            self.vb.setAspectLocked(True, self._view_ratio(1.0 if aspect == 'equal' else float(aspect)))
        else:
            self.vb.setAspectLocked(False)
        self._notify_shape()               # the panel's shape follows its aspect (rule 11)

    def _place_box_only(self):
        x0, y0, w, h = self._px_rect()
        if abs(self.vb.geometry().width() - w) > 0.5 or abs(self.vb.geometry().height() - h) > 0.5 \
                or abs(self.vb.geometry().x() - x0) > 0.5 or abs(self.vb.geometry().y() - y0) > 0.5:
            self._request_place()
            for m in (getattr(self, '_row', None) or []):      # the row follows this box's height
                if m is not self:
                    m._request_place()

    def get_aspect(self):
        return self._aspect

    def autoscale(self, enable=True, axis='both', tight=None):
        if enable:
            self.vb.enableAutoRange(x=axis in ('both', 'x'), y=axis in ('both', 'y'))
            self._nav_clear()                  # the window set the limits (to the data's): no view line

    def autoscale_view(self, *a, **k):
        pass

    def relim(self, *a, **k):
        pass

    def invert_yaxis(self):
        self._toggle_invert(self._view_axis('y'))

    def invert_xaxis(self):
        self._toggle_invert(self._view_axis('x'))

    def _toggle_invert(self, view):
        """The direction of view axis `view` ('x' / 'y') reversed."""
        if view == 'x':
            self.vb.invertX(not self.vb.state['xInverted'])
        else:
            self.vb.invertY(not self.vb.state['yInverted'])

    # --- a rotated axes (the simulator's unit-cell panels turned to fit the window) ---

    def _view_axis(self, axis):
        """The view axis ('x' / 'y') that shows data axis `axis`: the other one on a rotated axes."""
        if not self._rotated:
            return axis
        return 'y' if axis == 'x' else 'x'

    def _axis_item(self, axis):
        """The tick axis item (bottom or left) that shows data axis `axis`."""
        return self.xaxis_item if self._view_axis(axis) == 'x' else self.yaxis_item

    def _view_ratio(self, ratio):
        """pyqtgraph's aspect ratio (y units per x unit of the view) for a data aspect `ratio`: its inverse on a rotated axes."""
        return 1.0 / ratio if self._rotated and ratio else ratio

    def _rotor(self):
        """The turned group of a rotated axes (_Rotor), made when first needed and again after a clear detached it; None when the axes is not rotated."""
        if not self._rotated:
            return None
        r = self._rotor_item
        if r is not None and r.parentItem() is self.vb.childGroup:
            return r
        self._drop_rotor()
        r = _Rotor()
        r.setTransform(_SWAP)
        self.vb.addItem(r, ignoreBounds=True)
        self._rotor_item = r
        return r

    def _drop_rotor(self):
        r = self._rotor_item
        self._rotor_item = None
        if r is None:
            return
        try:
            self.vb.removeItem(r)
            if r.scene() is not None:
                r.scene().removeItem(r)
        except Exception:
            pass

    def _place_item(self, item):
        """An item just added to the view (Artist._add_item, a data-coordinate Text) goes under the rotor of a rotated axes."""
        r = self._rotor()
        if r is not None and item is not r:
            item.setParentItem(r)

    def get_rotated(self):
        return self._rotated

    def set_rotated(self, flag=True):
        """
        The axes turned 90 degrees anticlockwise on screen (data x up the vertical axis, data y
        along the horizontal axis increasing to the left: a rotation, never a mirror), or back.
        Everything drawn turns with it (the items sit under a turned group, _Rotor) and every
        value the axes takes or gives stays in data terms: the limits (set_xlim / set_ylim,
        get_xlim / get_ylim), the labels (set_xlabel on the vertical tick axis when turned), ticks
        and tick_params by data axis, the aspect, transData and the mouse events' xdata / ydata,
        the pickers.  The current limits, labels and aspect are carried over; fixed ticks
        (set_xticks) are not.  The panel's shape follows (its ratio turned: _aspect_want), so the
        window's rows and arrangement size it anew.  Used where a unit cell fits the window better
        the other way (the simulator's topographies and energy maps).
        """
        flag = bool(flag)
        if flag == self._rotated:
            return
        xlab, ylab = self._xlabel, self._ylabel
        xpt, ypt = self._axis_item('x')._label_pt, self._axis_item('y')._label_pt
        xi, yi = bool(self.vb.state['xInverted']), bool(self.vb.state['yInverted'])
        ar = list(self.vb.state['autoRange'])
        group = self.vb.childGroup
        if flag:
            self._rotated = True
            rotor = self._rotor()
            for it in list(group.childItems()):
                if it is not rotor:
                    it.setParentItem(rotor)
            self.vb.invertX(not yi)            # the view's x axis shows data y, to the left for ascending limits
            self.vb.invertY(xi)                # the view's y axis shows data x, as it ran
        else:
            rotor = self._rotor()
            self._rotated = False
            if rotor is not None:
                for it in list(rotor.childItems()):
                    it.setParentItem(group)
            self._drop_rotor()
            self.vb.invertX(yi)
            self.vb.invertY(not xi)
        self.vb.enableAutoRange(x=bool(ar[1]), y=bool(ar[0]))
        if self.vb.state['aspectLocked']:
            self.vb.setAspectLocked(True, 1.0 / float(self.vb.state['aspectLocked']))
        self._apply_targets()
        for item in (self.xaxis_item, self.yaxis_item):
            self._axis_label(item, '')
        self.set_xlabel(xlab, fontsize=xpt)
        self.set_ylabel(ylab, fontsize=ypt)
        self._notify_shape()
        self._request_place()

    def _home_view(self):
        if self._is_colorbar:
            return
        self._nav_clear()                      # Home: the view as the window set it, no view line
        cb = getattr(self, 'home_callback', None)
        if cb is not None:
            # An axes whose view a controller owns (home_callback set by the window):
            # the controller's own home (its full extent).
            cb()
            return
        if self._home is not None:
            xa, xb, ya, yb = self._home
            self.vb.setRange(xRange=(xa, xb), yRange=(ya, yb), padding=0)
        else:
            self.vb.enableAutoRange(x=True, y=True)

    def _apply_toolbar_mode(self, mode):
        if self._is_colorbar:
            self.vb.setMouseEnabled(False, False)
            return
        on = bool(mode)
        self.vb.setMouseEnabled(on, on)
        if mode == 'zoom rect':
            self.vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self.vb.setMouseMode(pg.ViewBox.PanMode)

    # --- plotting ---

    @staticmethod
    def _parse_fmt(fmt):
        style = {}
        s = fmt
        for ls in ('--', '-.', '-', ':'):
            if ls in s:
                style['ls'] = ls
                s = s.replace(ls, '', 1)
                break
        if len(s) >= 2 and s[0] == 'C' and s[1].isdigit():
            style['color'] = s[:2]
            s = s[2:]
        for ch in s:
            if ch in _BASE:
                style['color'] = ch
            elif ch in _MARKERS:
                style['marker'] = ch
        if 'marker' in style and 'ls' not in style:
            style['ls'] = 'none'
        return style

    def plot(self, *args, **kw):
        """Curves and markers (matplotlib's plot): x, y[, fmt] groups; a 2D y draws one curve per column."""
        groups = []
        a = list(args)
        while a:
            if len(a) >= 2 and not isinstance(a[1], str):
                x, y = a[0], a[1]
                a = a[2:]
            else:
                y = a[0]
                x = None
                a = a[1:]
            fmt = None
            if a and isinstance(a[0], str):
                fmt = a[0]
                a = a[1:]
            groups.append((x, y, fmt))
        out = []
        for x, y, fmt in groups:
            out.extend(self._plot_one(x, y, fmt, dict(kw)))
        return out

    def _plot_one(self, x, y, fmt, kw):
        style = self._parse_fmt(fmt) if fmt else {}
        color = kw.pop('color', kw.pop('c', style.get('color')))
        ls = kw.pop('ls', kw.pop('linestyle', style.get('ls', '-')))
        lw = kw.pop('lw', kw.pop('linewidth', 1.5))
        marker = kw.pop('marker', style.get('marker'))
        ms = kw.pop('ms', kw.pop('markersize', 6.0))
        mec = kw.pop('mec', kw.pop('markeredgecolor', None))
        mfc = kw.pop('mfc', kw.pop('markerfacecolor', None))
        mew = kw.pop('mew', kw.pop('markeredgewidth', 1.0))
        alpha = kw.pop('alpha', None)
        label = kw.pop('label', None)
        zorder = kw.pop('zorder', None)
        picker = kw.pop('picker', None)
        visible = kw.pop('visible', True)
        yv = np.asarray(np.ma.filled(np.ma.asarray(y, dtype=np.float64), np.nan), dtype=np.float64)
        if x is None:
            xv = np.arange(yv.shape[0], dtype=np.float64)
        else:
            xv = np.asarray(np.ma.filled(np.ma.asarray(x, dtype=np.float64), np.nan), dtype=np.float64)
        if yv.ndim <= 1:
            if color is None:
                color = self._next_color()
            ln = Line2D(self, xv.ravel(), yv.ravel(), color=color, lw=lw, ls=ls, marker=marker, ms=ms, mec=mec,
                        mfc=mfc, mew=mew, alpha=alpha, label=label, zorder=zorder, picker=picker, visible=visible)
            return [self._register(ln)]
        # A 2D y: one curve per column.  Columns of one colour are drawn as one item (NaN breaks), so a
        # grid's thousand spectra are ten items instead of a thousand.
        n = yv.shape[1]
        if n == 0:
            return []
        cols = [color] * n if color is not None else [None] * n
        if color is None:
            cols = [self._next_color() for _ in range(n)]
        by_color = {}
        for j, c in enumerate(cols):
            by_color.setdefault(c, []).append(j)
        out = []
        xcol = xv if xv.ndim == 1 else None
        for c, js in by_color.items():
            parts_x, parts_y = [], []
            for j in js:
                parts_x.append(xcol if xcol is not None else xv[:, j])
                parts_y.append(yv[:, j])
                parts_x.append([np.nan])
                parts_y.append([np.nan])
            ln = Line2D(self, np.concatenate(parts_x), np.concatenate(parts_y), color=c, lw=lw, ls=ls, marker=marker,
                        ms=ms, mec=mec, mfc=mfc, mew=mew, alpha=alpha, label=label, zorder=zorder, picker=picker,
                        visible=visible)
            out.append(self._register(ln))
        return out

    def scatter(self, x, y, s=None, c=None, marker='o', label=None, zorder=None, color=None, facecolors=None,
                facecolor=None, edgecolors='face', edgecolor=None, linewidths=1.0, linewidth=None, alpha=None, picker=None,
                cmap=None, norm=None, vmin=None, vmax=None, **kw):
        """
        matplotlib's scatter: a PathCollection of markers at (x, y), `s` their areas in points
        squared (36: the 6-point marker), `c` one colour, one colour per point, or numbers
        coloured through `cmap` and `norm` (vmin / vmax), `edgecolors` 'face', 'none' or a
        colour, `picker` a pick radius in pixels (True: 5).
        """
        x = np.asarray(np.ma.filled(np.ma.asarray(x, dtype=np.float64), np.nan), dtype=np.float64).ravel()
        y = np.asarray(np.ma.filled(np.ma.asarray(y, dtype=np.float64), np.nan), dtype=np.float64).ravel()
        n = len(x)
        fc = c if c is not None else (color if color is not None else (facecolors if facecolors is not None else facecolor))
        array = None
        if fc is not None and not isinstance(fc, str):
            arr = np.asarray(fc)
            if arr.dtype.kind in 'fiub' and arr.ndim == 1 and (n not in (3, 4) or len(arr) == n and n > 4):
                array, fc = arr.astype(np.float64), None          # numbers: through the colour map
            elif arr.dtype.kind in 'fiub' and arr.ndim == 1 and len(arr) == n and n in (3, 4) and cmap is not None:
                array, fc = arr.astype(np.float64), None
        if fc is None and array is None:
            fc = self._next_color()
        ec = edgecolor if edgecolor is not None else edgecolors
        lw = linewidth if linewidth is not None else linewidths
        nrm = norm
        if array is not None and nrm is None:
            nrm = Normalize(vmin, vmax)
        if picker is True:
            picker = 5.0
        coll = PathCollection(self, np.column_stack([x, y]) if n else np.zeros((0, 2)),
                              sizes=36.0 if s is None else s, facecolors=fc, edgecolors=ec, linewidths=lw,
                              marker=marker, alpha=alpha, label=label, zorder=zorder, picker=picker, cmap=cmap, norm=nrm,
                              array=array)
        return self._register(coll)

    def _lines_collection(self, segments, colors, linestyles, label, linewidths, color, alpha, zorder, picker, kw):
        lw = linewidths if linewidths is not None else kw.pop('linewidth', kw.pop('lw', 1.5))
        col = color if color is not None else colors
        lc = LineCollection(segments, colors=col, linewidths=lw, linestyles=linestyles, alpha=alpha, label=label,
                            zorder=zorder, picker=picker)
        return self.add_collection(lc)

    def vlines(self, x, ymin, ymax, colors='k', linestyles='solid', label=None, linewidths=None, color=None, alpha=None,
               zorder=None, picker=None, **kw):
        """matplotlib's vlines: vertical segments from ymin to ymax at each x (a LineCollection)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        lo = np.broadcast_to(np.asarray(ymin, dtype=np.float64), x.shape)
        hi = np.broadcast_to(np.asarray(ymax, dtype=np.float64), x.shape)
        segs = [np.array([[xi, a], [xi, b]]) for xi, a, b in zip(x, lo, hi)]
        return self._lines_collection(segs, colors, linestyles, label, linewidths, color, alpha, zorder, picker, kw)

    def hlines(self, y, xmin, xmax, colors='k', linestyles='solid', label=None, linewidths=None, color=None, alpha=None,
               zorder=None, picker=None, **kw):
        """matplotlib's hlines: horizontal segments from xmin to xmax at each y (a LineCollection)."""
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        lo = np.broadcast_to(np.asarray(xmin, dtype=np.float64), y.shape)
        hi = np.broadcast_to(np.asarray(xmax, dtype=np.float64), y.shape)
        segs = [np.array([[a, yi], [b, yi]]) for yi, a, b in zip(y, lo, hi)]
        return self._lines_collection(segs, colors, linestyles, label, linewidths, color, alpha, zorder, picker, kw)

    def add_line(self, line):
        """A Line2D put on this axes (matplotlib's add_line), or put back after a clear: its items are built here."""
        if line in self._children:
            return line
        line.axes, line.figure = self, self.figure
        line.item = None
        line._items = []
        line._build()
        return self._register(line)

    def tricontourf(self, x, y, z, levels=10, cmap=None, norm=None, vmin=None, vmax=None, zorder=None, alpha=None,
                    offsets=None, res=256, **kw):
        """Filled contours of scattered points (TriContourSet): linear interpolation onto a raster, banded by the levels; `offsets` tiles the raster."""
        return self._register(TriContourSet(self, x, y, z, levels=levels, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                                            zorder=zorder, alpha=alpha, offsets=offsets, res=res))

    def axhline(self, y=0.0, xmin=0, xmax=1, **kw):
        return self._register(_RefLine(self, y, True, color=kw.get('color', kw.get('c', 'C0')),
                                       lw=kw.get('lw', kw.get('linewidth', 1.5)), ls=kw.get('ls', kw.get('linestyle', '-')),
                                       alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder'),
                                       picker=kw.get('picker')))

    def axvline(self, x=0.0, ymin=0, ymax=1, **kw):
        return self._register(_RefLine(self, x, False, color=kw.get('color', kw.get('c', 'C0')),
                                       lw=kw.get('lw', kw.get('linewidth', 1.5)), ls=kw.get('ls', kw.get('linestyle', '-')),
                                       alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder'),
                                       picker=kw.get('picker')))

    def axvspan(self, xmin, xmax, **kw):
        return self._register(_Span(self, xmin, xmax, True, color=kw.get('color', kw.get('fc', 'C0')),
                                    alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder')))

    def axhspan(self, ymin, ymax, **kw):
        return self._register(_Span(self, ymin, ymax, False, color=kw.get('color', kw.get('fc', 'C0')),
                                    alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder')))

    def imshow(self, X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None,
               origin=None, extent=None, zorder=None, label=None, **kw):
        """
        An image (AxesImage) as matplotlib's imshow: `aspect` None or 'equal' locks the aspect,
        'auto' leaves it; without an `extent` and with origin 'upper' the y axis is inverted (row 0
        at the top).  `interpolation` is accepted and ignored: pyqtgraph draws each data pixel as a
        block, and mplexport passes 'none' to matplotlib.
        """
        im = AxesImage(self, X, cmap=cmap, norm=norm, extent=extent, origin=origin or 'upper', alpha=alpha,
                       zorder=zorder, label=label, vmin=vmin, vmax=vmax)
        self._register(im)
        if aspect is None or aspect == 'equal':
            self.set_aspect('equal')
        elif aspect != 'auto':
            self.set_aspect(aspect)
        if extent is None and (origin or 'upper') == 'upper' and not self.vb.state['yInverted']:
            self.vb.invertY(True)
        return im

    def pcolormesh(self, *args, cmap=None, norm=None, vmin=None, vmax=None, shading='flat', zorder=None,
                   alpha=None, **kw):
        if len(args) == 1:
            C = np.asarray(np.ma.filled(np.ma.asarray(args[0], dtype=np.float64), np.nan))
            X = np.arange(C.shape[1] + 1, dtype=np.float64)
            Y = np.arange(C.shape[0] + 1, dtype=np.float64)
        else:
            X, Y, C = args[:3]
        return self._register(QuadMesh(self, X, Y, C, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, zorder=zorder,
                                       alpha=alpha, shading=shading))

    def contourf(self, X, Y, Z, levels=10, cmap=None, zorder=None, norm=None, vmin=None, vmax=None, alpha=None, **kw):
        return self._register(ContourSet(self, X, Y, Z, levels=levels, cmap=cmap, zorder=zorder, norm=norm, vmin=vmin,
                                         vmax=vmax, alpha=alpha))

    def add_collection(self, coll, autolim=True):
        """A collection put on this axes (matplotlib's add_collection), or put back after a clear: its items are built here."""
        if coll in self._children:
            return coll
        coll._attach(self)
        return self._register(coll)

    def add_artist(self, a):
        """
        matplotlib's add_artist as the windows use it: a legend made by legend() is kept beside
        the next one (two legends on one panel); any other artist is already drawn.
        """
        if isinstance(a, pg.LegendItem) and a is self._legend and getattr(self, '_legend_args', None) is not None:
            args, kw = self._legend_args
            self._extra_legends.append((a, args, dict(kw)))
            self._legend = None
            self._legend_args = None
        return a

    def _drop_extra_legends(self):
        for leg, _, _ in self._extra_legends:
            try:
                leg.setParentItem(None)
                if leg.scene() is not None:
                    leg.scene().removeItem(leg)
            except Exception:
                pass
        self._extra_legends = []

    def _rebuild_extra_legends(self):
        """The kept legends built again (a new zoom or width): each through legend() and kept again."""
        extra = self._extra_legends
        self._extra_legends = []
        for leg, args, kw in extra:
            try:
                leg.setParentItem(None)
                if leg.scene() is not None:
                    leg.scene().removeItem(leg)
            except Exception:
                pass
            cur, cur_args = self._legend, getattr(self, '_legend_args', None)
            self._legend = None
            new = self.legend(*args, **kw)
            if new is not None:
                self._extra_legends.append((new, args, dict(kw)))
            self._legend, self._legend_args = cur, cur_args

    def text(self, x, y, s, transform=None, **kw):
        coords = 'data'
        if transform is not None:
            kind = getattr(transform, 'kind', 'data')
            coords = {'axes': 'axes', 'figure': 'figure'}.get(kind, 'data')
        owner = self.figure if coords == 'figure' else self
        t = Text(owner, x, y, s, coords, **kw)
        if coords == 'figure':
            self.figure._texts.append(t)
            return t
        if coords == 'axes':
            self._axes_texts.append(t)
            t.item.setVisible(self._visible and t.get_visible())
        self._children.append(t)
        return t

    def annotate(self, text, xy, xytext=None, textcoords=None, arrowprops=None, xycoords='data', **kw):
        if xycoords in ('axes fraction', 'figure fraction') and arrowprops is None:
            # A note placed by axes (or figure) fractions: a Text in those coordinates (it is in ax.texts).
            tr = self.transAxes if xycoords == 'axes fraction' else self.figure.transFigure
            return self.text(float(xy[0]), float(xy[1]), text, transform=tr,
                             ha=kw.get('ha', kw.get('horizontalalignment', 'left')),
                             va=kw.get('va', kw.get('verticalalignment', 'baseline')), fontsize=kw.get('fontsize'),
                             color=kw.get('color', 'k'), fontweight=kw.get('fontweight'), bbox=kw.get('bbox'),
                             zorder=kw.get('zorder'))
        return self._register(Annotation(self, text, xy, xytext=xytext, textcoords=textcoords,
                                         arrowprops=arrowprops, color=kw.get('color', 'k'),
                                         fontsize=kw.get('fontsize'), fontweight=kw.get('fontweight'),
                                         zorder=kw.get('zorder'), ha=kw.get('ha', 'left'), va=kw.get('va', 'baseline')))

    def secondary_yaxis(self, location='right', functions=None, **kw):
        a = self._make_side_axis('right')
        if functions is not None:
            a._mode = ('func', functions[0], functions[1])
        return SecondaryAxis(self, a)

    def inset_axes(self, bounds, **kw):
        return Axes(self.figure, rect=bounds, inset_parent=self)

    # --- legend ---

    def get_legend_handles_labels(self):
        handles, labels = [], []
        for a in self._children:
            lab = a.get_label()
            if isinstance(a, (Text, AxesImage)) or not lab or lab.startswith('_'):
                continue
            handles.append(a)
            labels.append(lab)
        return handles, labels

    def _remove_legend(self):
        leg = self._legend
        self._legend = None
        if leg is None:
            return
        try:
            leg.setParentItem(None)
            if leg.scene() is not None:
                leg.scene().removeItem(leg)
        except Exception:
            pass

    def get_legend(self):
        return self._legend

    def legend(self, *args, loc='best', frameon=True, fontsize=None, ncol=1, handles=None, labels=None, title=None,
               **kw):
        """
        The legend as matplotlib's: legend(), legend(labels), legend(handles, labels), or the
        keywords handles= and labels= (proxies: Line2D and Patch made without an axes); labels
        starting with '_' are left out; `title` a row on top; `loc` a matplotlib location name or
        code ('best' and 0 / 1 are the upper right: the legend never moves to avoid curves);
        `fontsize` in points (10) at the figure's zoom, shrunk when the longest label would not
        fit the panel.  Returns the pyqtgraph LegendItem, None when nothing is to be listed.  The
        arguments are kept so that _refont and _place can rebuild it at a new zoom or width.
        """
        if handles is not None:
            handles = list(handles)
            labels = list(labels) if labels is not None else [h.get_label() for h in handles]
        elif len(args) >= 2:
            handles, labels = list(args[0]), list(args[1])
        elif len(args) == 1:
            labels = list(args[0])
            handles = [a for a in self._children if not isinstance(a, (Text, AxesImage))][:len(labels)]
        else:
            handles, labels = self.get_legend_handles_labels()
        if self._legend is not None:
            self._remove_legend()
        pairs = [(h, l) for h, l in zip(handles, labels) if l and not str(l).startswith('_')]
        if not pairs:
            return None
        self._legend_args = (args, dict(loc=loc, frameon=frameon, fontsize=fontsize, ncol=ncol,
                                        handles=handles if handles is not None and not args else None,
                                        labels=labels if handles is not None and not args else None, title=title))
        size = font_pt(fontsize, 10.0) * self.figure.font_zoom
        leg = pg.LegendItem(offset=None, verSpacing=0, horSpacing=4,
                            labelTextSize=f"{pt2px(size) * FONT_SCALE:.1f}px", colCount=int(ncol), frame=False,
                            pen=None if not frameon else pg.mkPen((204, 204, 204)),
                            brush=None if not frameon else pg.mkBrush(255, 255, 255, 204))
        leg.layout.setContentsMargins(2, 2, 2, 2)
        leg.setParentItem(self.vb)
        rows_h, widths = [], []
        # The legend's text is a real font at the figure's zoom (an HTML size is scaled differently by
        # Qt on high-DPI displays), shrunk when needed so the legend fits inside its panel.
        base_pt = font_pt(fontsize, 10.0)
        fnt = qfont(base_pt, zoom=self.figure.font_zoom)
        fm = QtGui.QFontMetricsF(fnt)
        text_w = max((fm.horizontalAdvance(str(l)) for _, l in pairs), default=0.0)
        avail = self._px_rect()[2] / max(int(ncol), 1) - 26 - 16
        if text_w > avail > 0:
            fnt = qfont(base_pt, zoom=self.figure.font_zoom * max(0.45, avail / text_w))
            fm = QtGui.QFontMetricsF(fnt)
        row_px = fm.height() * 1.15
        title_w = 0
        if title:
            # The title: one row over the samples and labels, in the legend's font, bold.
            tl = pg.LabelItem('', justify='left')
            tf = QtGui.QFont(fnt)
            tf.setBold(True)
            tl.item.setFont(tf)
            tl.item.setDefaultTextColor(QtGui.QColor('black'))
            tl.item.setPlainText(str(title))
            tl.item.document().setDocumentMargin(1)
            row_h = int(math.ceil(row_px))
            title_w = int(math.ceil(QtGui.QFontMetricsF(tf).horizontalAdvance(str(title)))) + 6
            tl.setFixedHeight(row_h)
            tl.setFixedWidth(title_w)
            leg.layout.addItem(tl, 0, 0, 1, 2 * max(int(ncol), 1))
            rows_h.append(row_h)
        for h, l in pairs:
            sample = h._legend_sample() if hasattr(h, '_legend_sample') else None
            if sample is None:
                continue
            lab = pg.LabelItem('', justify='left')
            lab.item.setFont(fnt)
            lab.item.setDefaultTextColor(QtGui.QColor('black'))
            lab.item.setPlainText(mathtext(str(l)))
            lab.item.document().setDocumentMargin(1)
            row_h = int(math.ceil(row_px))
            lab_w = int(math.ceil(fm.horizontalAdvance(str(l)))) + 6
            lab.setFixedHeight(row_h)                 # fixed rows: the legend never changes its spacing
            lab.setFixedWidth(lab_w)
            sample.setFixedHeight(row_h)
            rows_h.append(row_h)
            widths.append(lab_w + 26 + 4)
            row = leg.layout.rowCount()
            col = 0
            if leg.columnCount > 1:
                n = len(leg.items)
                row, col = n // leg.columnCount + (1 if title else 0), 2 * (n % leg.columnCount)
            leg.items.append((sample, lab))
            leg.layout.addItem(sample, row, col)
            leg.layout.addItem(lab, row, col + 1)
        # The legend's size fixed from its rows (pyqtgraph's updateSize re-measures rows that the
        # layout has stretched, so every call made the legend taller and its rows farther apart).
        leg.layout.setVerticalSpacing(0)
        leg.layout.setContentsMargins(2, 2, 2, 2)
        ncol = max(int(ncol), 1)
        n_items = len(rows_h) - (1 if title else 0)
        n_rows = int(math.ceil(n_items / ncol)) + (1 if title else 0) if rows_h else 0
        lw = max((max(widths) if widths else 0) * ncol, title_w) + 4
        lh = (max(rows_h) if rows_h else 0) * n_rows + 4
        leg.updateSize = lambda: None
        leg.setMinimumSize(lw, lh)
        leg.setMaximumSize(lw, lh)
        leg.setGeometry(QtCore.QRectF(0, 0, lw, lh))
        loc = {'best': 'upper right', 0: 'upper right', 1: 'upper right', 2: 'upper left', 3: 'lower left',
               4: 'lower right'}.get(loc, loc)
        anchors = {'upper right': ((1, 0), (-4, 4)), 'upper left': ((0, 0), (4, 4)),
                   'lower left': ((0, 1), (4, -4)), 'lower right': ((1, 1), (-4, -4)),
                   'center': ((0.5, 0.5), (0, 0)), 'upper center': ((0.5, 0), (0, 4)),
                   'lower center': ((0.5, 1), (0, -4)), 'center left': ((0, 0.5), (4, 0)),
                   'center right': ((1, 0.5), (-4, 0)), 'right': ((1, 0.5), (-4, 0))}
        pos, off = anchors.get(loc, anchors['upper right'])
        leg.anchor(itemPos=pos, parentPos=pos, offset=off)
        leg.setZValue(100)
        self._legend = leg
        return leg


# ---------------------------------------------------------------------- #
#  Grids of subplots                                                       #
# ---------------------------------------------------------------------- #

class SubplotParams:
    """fig.subplotpars: the margins and spacings of the subplot grids, matplotlib's defaults (subplots_adjust changes them)."""

    def __init__(self):
        self.left, self.right, self.bottom, self.top = 0.125, 0.9, 0.11, 0.88
        self.wspace, self.hspace = 0.2, 0.2


class _GridSpecBase:
    def __init__(self, nrows, ncols, width_ratios=None, height_ratios=None, wspace=None, hspace=None):
        self.nrows, self.ncols = int(nrows), int(ncols)
        self.width_ratios = list(width_ratios) if width_ratios is not None else [1.0] * self.ncols
        self.height_ratios = list(height_ratios) if height_ratios is not None else [1.0] * self.nrows
        self.wspace, self.hspace = wspace, hspace

    def get_geometry(self):
        return self.nrows, self.ncols

    def _bounds(self, fig):
        raise NotImplementedError

    def get_grid_positions(self, fig):
        """(bottoms, tops, lefts, rights) of the cells, matplotlib's algorithm."""
        left, right, bottom, top, wspace, hspace = self._bounds(fig)
        tot_w, tot_h = right - left, top - bottom
        cell_h = tot_h / (self.nrows + hspace * (self.nrows - 1))
        sep_h = hspace * cell_h
        norm = cell_h * self.nrows / sum(self.height_ratios)
        cell_heights = [r * norm for r in self.height_ratios]
        sep_heights = [0] + [sep_h] * (self.nrows - 1)
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)
        cell_w = tot_w / (self.ncols + wspace * (self.ncols - 1))
        sep_w = wspace * cell_w
        norm = cell_w * self.ncols / sum(self.width_ratios)
        cell_widths = [r * norm for r in self.width_ratios]
        sep_widths = [0] + [sep_w] * (self.ncols - 1)
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return fig_bottoms, fig_tops, fig_lefts, fig_rights

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r, c = key
        else:
            if isinstance(key, slice):
                idx = range(self.nrows * self.ncols)[key]
                rows = sorted({i // self.ncols for i in idx})
                cols = sorted({i % self.ncols for i in idx})
                return SubplotSpec(self, slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
            k = int(key)
            if k < 0:
                k += self.nrows * self.ncols
            r, c = divmod(k, self.ncols)
        rs = r if isinstance(r, slice) else slice(int(r) % self.nrows, int(r) % self.nrows + 1)
        cs = c if isinstance(c, slice) else slice(int(c) % self.ncols, int(c) % self.ncols + 1)
        return SubplotSpec(self, rs, cs)


class GridSpec(_GridSpecBase):
    def __init__(self, nrows, ncols, figure=None, left=None, bottom=None, right=None, top=None, wspace=None,
                 hspace=None, width_ratios=None, height_ratios=None):
        super().__init__(nrows, ncols, width_ratios, height_ratios, wspace, hspace)
        self.figure = figure
        self.left, self.bottom, self.right, self.top = left, bottom, right, top

    def _bounds(self, fig):
        sp = fig.subplotpars
        return (sp.left if self.left is None else self.left, sp.right if self.right is None else self.right,
                sp.bottom if self.bottom is None else self.bottom, sp.top if self.top is None else self.top,
                sp.wspace if self.wspace is None else self.wspace, sp.hspace if self.hspace is None else self.hspace)


class GridSpecFromSubplotSpec(_GridSpecBase):
    def __init__(self, nrows, ncols, subplot_spec, wspace=None, hspace=None, height_ratios=None, width_ratios=None):
        super().__init__(nrows, ncols, width_ratios, height_ratios, wspace, hspace)
        self._parent = subplot_spec

    def _bounds(self, fig):
        b = self._parent.get_position(fig)
        sp = fig.subplotpars
        return (b.x0, b.x1, b.y0, b.y1, sp.wspace if self.wspace is None else self.wspace,
                sp.hspace if self.hspace is None else self.hspace)


class SubplotSpec:
    def __init__(self, gridspec, rows, cols):
        self._gs, self.rows, self.cols = gridspec, rows, cols

    def get_gridspec(self):
        return self._gs

    def get_position(self, fig):
        bottoms, tops, lefts, rights = self._gs.get_grid_positions(fig)
        r = range(self._gs.nrows)[self.rows]
        c = range(self._gs.ncols)[self.cols]
        x0, x1 = min(lefts[list(c)]), max(rights[list(c)])
        y0, y1 = min(bottoms[list(r)]), max(tops[list(r)])
        return Bbox(x0, y0, x1 - x0, y1 - y0)


gridspec = types.SimpleNamespace(GridSpec=GridSpec, GridSpecFromSubplotSpec=GridSpecFromSubplotSpec,
                                 SubplotSpec=SubplotSpec)


# ---------------------------------------------------------------------- #
#  Events                                                                  #
# ---------------------------------------------------------------------- #

class MouseButton:
    LEFT, MIDDLE, RIGHT, BACK, FORWARD = 1, 2, 3, 8, 9


class MouseEvent:
    """matplotlib's MouseEvent: x, y in display pixels from the bottom left, inaxes, xdata, ydata, button, dblclick."""

    def __init__(self, name, canvas, x, y, button=None, dblclick=False, step=0, guiEvent=None, key=None):
        self.name = name
        self.canvas = canvas
        self.x, self.y = x, y
        self.button = button
        self.dblclick = dblclick
        self.step = step
        self.guiEvent = guiEvent
        self.key = key
        self.inaxes = None
        self.xdata = self.ydata = None


class KeyEvent:
    """
    matplotlib's KeyEvent: key (matplotlib's names: 'r', ' ', 'home', 'left', 'ctrl+left',
    'shift+left', 'tab', 'escape', 'delete', ...), x, y (the cursor in display pixels from the
    bottom left), inaxes, xdata, ydata (the axes under the cursor).  A callback that uses the key
    sets `handled = True`; the canvas then consumes the Qt event.
    """

    def __init__(self, name, canvas, key, x=None, y=None, guiEvent=None):
        self.name = name
        self.canvas = canvas
        self.key = key
        self.x, self.y = x, y
        self.guiEvent = guiEvent
        self.inaxes = None
        self.xdata = self.ydata = None
        self.handled = False


_QT_KEY_NAMES = None


def _key_name(ev):
    """A Qt key event as matplotlib's key name (None for a bare modifier)."""
    global _QT_KEY_NAMES
    K = Qt.Key
    if _QT_KEY_NAMES is None:
        _QT_KEY_NAMES = {K.Key_Space: ' ', K.Key_Home: 'home', K.Key_End: 'end', K.Key_Left: 'left',
                         K.Key_Right: 'right', K.Key_Up: 'up', K.Key_Down: 'down', K.Key_Tab: 'tab',
                         K.Key_Backtab: 'tab', K.Key_Escape: 'escape', K.Key_Delete: 'delete',
                         K.Key_Backspace: 'backspace', K.Key_Return: 'enter', K.Key_Enter: 'enter',
                         K.Key_PageUp: 'pageup', K.Key_PageDown: 'pagedown'}
    k = ev.key()
    if k in (K.Key_Shift, K.Key_Control, K.Key_Alt, K.Key_Meta):
        return None
    name = _QT_KEY_NAMES.get(k)
    if name is None:
        t = ev.text()
        if not t or not t.isprintable():
            return None
        name = t.lower()
    mods = ev.modifiers()
    M = Qt.KeyboardModifier
    prefix = ''
    if mods & M.ControlModifier:
        prefix += 'ctrl+'
    if mods & M.AltModifier:
        prefix += 'alt+'
    if (mods & M.ShiftModifier) and (len(name) > 1 or name == ' '):
        prefix += 'shift+'
    return prefix + name


class PickEvent:
    def __init__(self, canvas, mouseevent, artist):
        self.name = 'pick_event'
        self.canvas = canvas
        self.mouseevent = mouseevent
        self.artist = artist
        self.ind = []


class CloseEvent:
    def __init__(self, canvas):
        self.name = 'close_event'
        self.canvas = canvas


class _Toolbar:
    """The figure's navigation state: mode '' (the windows' own mouse handling), 'pan/zoom', or 'zoom rect'."""

    def __init__(self):
        self.mode = ''


_QT_BUTTONS = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.MiddleButton: 2, Qt.MouseButton.RightButton: 3,
               Qt.MouseButton.BackButton: 8, Qt.MouseButton.ForwardButton: 9}


class _PosOnly:
    """A stand-in event holding only a position (the canvas's _event reads position())."""

    def __init__(self, p):
        self._p = p

    def position(self):
        return self._p


class FigureCanvas(pg.GraphicsView):
    """
    The figure's drawing surface: one pyqtgraph scene in pixel coordinates,
    the Qt controls as child widgets, and matplotlib's event connection
    (mpl_connect / mpl_disconnect, draw_idle).  Mouse motion is coalesced:
    a burst of moves during a slow redraw reaches the handlers as one event
    at the latest position.
    """

    EVENTS = ('button_press_event', 'button_release_event', 'motion_notify_event', 'scroll_event',
              'pick_event', 'close_event', 'resize_event', 'key_press_event', 'key_release_event',
              'figure_enter_event', 'figure_leave_event', 'axes_enter_event', 'axes_leave_event', 'draw_event')

    def __init__(self, figure, w_px, h_px):
        super().__init__(background='w')
        self.figure = figure
        self._size_px = (int(w_px), int(h_px))
        self._cbs = {name: {} for name in self.EVENTS}
        self._cid = itertools.count(1)
        self.toolbar = _Toolbar()
        self.manager = None
        self._pending_motion = None
        self._motion_scheduled = False
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.resize(int(w_px), int(h_px))

    def sizeHint(self):
        return QtCore.QSize(*self._size_px)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        fig = getattr(self, 'figure', None)
        if fig is not None and getattr(fig, 'manager', None) is not None:
            fig._relayout()

    # --- matplotlib's connection API ---

    def mpl_connect(self, name, func):
        cid = next(self._cid)
        self._cbs.setdefault(name, {})[cid] = func
        return cid

    def mpl_disconnect(self, cid):
        for d in self._cbs.values():
            d.pop(cid, None)

    def _fire(self, name, event):
        for cid, func in list(self._cbs.get(name, {}).items()):
            try:
                func(event)
            except Exception:
                traceback.print_exc()

    def draw_idle(self):
        pass

    def draw(self):
        self.viewport().update()

    def flush_events(self):
        QtWidgets.QApplication.processEvents()

    def get_width_height(self):
        return self.width(), self.height()

    # --- Qt events -> matplotlib events ---

    def _event(self, name, ev, button=None, dblclick=False, step=0):
        pos = _evpos(ev)
        px, py = float(pos.x()), float(pos.y())
        H = self.viewport().height()
        e = MouseEvent(name, self, px, H - py, button=button, dblclick=dblclick, step=step, guiEvent=ev)
        ax = self.figure._axes_at(px, py)
        if ax is not None:
            e.inaxes = ax
            try:
                (xa, xb), (ya, yb) = ax.vb.viewRange()
                x0, y0, w, h = ax._px_rect()
                fx = (px - x0) / (w or 1.0)
                fy = (y0 + h - py) / (h or 1.0)
                if ax.vb.state['xInverted']:
                    fx = 1.0 - fx
                if ax.vb.state['yInverted']:
                    fy = 1.0 - fy
                e.xdata = xa + fx * (xb - xa)
                e.ydata = ya + fy * (yb - ya)
                if ax._rotated:
                    e.xdata, e.ydata = e.ydata, e.xdata      # the view shows (y, x)
            except Exception:
                e.inaxes = None
        return e

    def mousePressEvent(self, ev):
        self._flush_motion()
        b = _QT_BUTTONS.get(ev.button())
        e = self._event('button_press_event', ev, button=b)
        self.figure._pick(e)
        self._fire('button_press_event', e)
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self._flush_motion()
        b = _QT_BUTTONS.get(ev.button())
        e = self._event('button_press_event', ev, button=b, dblclick=True)
        self.figure._pick(e)
        self._fire('button_press_event', e)
        super().mouseDoubleClickEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._flush_motion()
        b = _QT_BUTTONS.get(ev.button())
        self._fire('button_release_event', self._event('button_release_event', ev, button=b))
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = _evpos(ev)
        self._pending_motion = (QtCore.QPointF(pos), ev.buttons())
        if not self._motion_scheduled:
            self._motion_scheduled = True
            QtCore.QTimer.singleShot(0, self._flush_motion)
        if self.manager is not None:
            self.manager._show_coords(float(pos.x()), float(pos.y()))
        super().mouseMoveEvent(ev)

    def _flush_motion(self):
        self._motion_scheduled = False
        pm = self._pending_motion
        self._pending_motion = None
        if pm is None:
            return
        pos, buttons = pm

        class _E:
            def __init__(self, p):
                self._p = p

            def position(self):
                return self._p
        b = None
        for qb, nb in _QT_BUTTONS.items():
            if buttons & qb:
                b = nb
                break
        self._fire('motion_notify_event', self._event('motion_notify_event', _E(pos), button=b))

    def event(self, ev):
        # Tab / Backtab reach the key callbacks first (Qt would move the focus with them); a key the
        # callbacks do not handle moves the focus as usual.
        if (ev.type() == QtCore.QEvent.Type.KeyPress and self._cbs.get('key_press_event')
                and ev.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)):
            self._key_handled = False
            self.keyPressEvent(ev)
            if self._key_handled:
                return True
        return super().event(ev)

    def keyPressEvent(self, ev):
        """matplotlib's key_press_event, with the axes under the cursor; a callback setting handled consumes the key."""
        self._key_handled = False
        key = _key_name(ev)
        if key is None or not self._cbs.get('key_press_event'):
            super().keyPressEvent(ev)
            return
        pos = self.viewport().mapFromGlobal(QtGui.QCursor.pos())
        m = self._event('motion_notify_event', _PosOnly(QtCore.QPointF(pos)))
        e = KeyEvent('key_press_event', self, key, m.x, m.y, guiEvent=ev)
        e.inaxes, e.xdata, e.ydata = m.inaxes, m.xdata, m.ydata
        self._fire('key_press_event', e)
        if e.handled:
            self._key_handled = True
            ev.accept()
        else:
            super().keyPressEvent(ev)

    def wheelEvent(self, ev):
        """matplotlib's scroll_event: button 'up' / 'down', step in wheel notches (120 units each, fractional on a touchpad)."""
        dy = ev.angleDelta().y()
        if dy:
            step = dy / 120.0
            e = self._event('scroll_event', ev, button='up' if dy > 0 else 'down', step=step)
            self._fire('scroll_event', e)
        super().wheelEvent(ev)

    def leaveEvent(self, ev):
        if self.manager is not None:
            self.manager._show_coords(None, None)
        super().leaveEvent(ev)


class _Window(QtWidgets.QMainWindow):
    def __init__(self, manager):
        super().__init__()
        self._manager = manager

    def closeEvent(self, ev):
        self._manager._closed()
        super().closeEvent(ev)


def _icon(kind, size=22):
    """Small toolbar icons drawn here (pan: four arrows, zoom: a magnifier), as matplotlib's toolbar shows them."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(30, 30, 30))
    pen.setWidthF(2.2)
    p.setPen(pen)
    c = size / 2.0
    if kind == 'pan':
        r = size * 0.4
        p.drawLine(QtCore.QPointF(c - r, c), QtCore.QPointF(c + r, c))
        p.drawLine(QtCore.QPointF(c, c - r), QtCore.QPointF(c, c + r))
        h = size * 0.14
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tip = QtCore.QPointF(c + dx * r, c + dy * r)
            p.drawLine(tip, QtCore.QPointF(tip.x() - dx * h - dy * h, tip.y() - dy * h - dx * h))
            p.drawLine(tip, QtCore.QPointF(tip.x() - dx * h + dy * h, tip.y() - dy * h + dx * h))
    elif kind == 'zoom':
        r = size * 0.26
        p.drawEllipse(QtCore.QPointF(c - 2, c - 2), r, r)
        pen.setWidthF(3.2)
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(c - 2 + r * 0.7, c - 2 + r * 0.7), QtCore.QPointF(size - 3, size - 3))
    p.end()
    return QtGui.QIcon(pm)


class NavigationToolbar(QtWidgets.QToolBar):
    """
    A figure's navigation toolbar (matplotlib's): Home (the views as the
    window set them), Pan (left drag pans, right drag / wheel zooms), Zoom
    (a rectangle), Save, and the cursor's coordinates.  Works for a figure
    in its own window and for a figure embedded in a layout.
    """

    def __init__(self, figure, parent=None, coordinates=True):
        super().__init__('navigation', parent)
        self.figure = figure
        self.setMovable(False)
        k = int(round(22 * screen_scale()))
        self.setIconSize(QtCore.QSize(k, k))
        st = self.style()
        sp = QtWidgets.QStyle.StandardPixmap
        self.a_home = a = self.addAction(st.standardIcon(sp.SP_DirHomeIcon), 'Home')
        a.setToolTip('Reset the views to where the window put them')
        a.triggered.connect(self._home)
        self.addSeparator()
        self.a_pan = self.addAction(_icon('pan'), 'Pan')
        self.a_pan.setCheckable(True)
        self.a_pan.setToolTip('Pan with the left button, zoom with the right button or the wheel')
        self.a_pan.toggled.connect(lambda on: self._mode('pan/zoom' if on else ''))
        self.a_zoom = self.addAction(_icon('zoom'), 'Zoom')
        self.a_zoom.setCheckable(True)
        self.a_zoom.setToolTip('Zoom to a rectangle drawn with the left button')
        self.a_zoom.toggled.connect(lambda on: self._mode('zoom rect' if on else ''))
        self.addSeparator()
        a = self.addAction(st.standardIcon(sp.SP_DialogSaveButton), 'Save')
        a.setToolTip('Save the figure: PDF / EPS / SVG for publication (matplotlib), PNG / TIFF / SVG as on screen, or the curves as CSV')
        a.triggered.connect(self._save)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)
        self.coords = QtWidgets.QLabel('')
        self.coords.setMinimumWidth(200)
        self.coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if coordinates:
            self.addWidget(self.coords)
        figure._toolbars.append(self)

    @property
    def mode(self):
        """The navigation mode as matplotlib's toolbar.mode: '' (the windows' own mouse handling), 'pan/zoom' or 'zoom rect'."""
        return self.figure.canvas.toolbar.mode

    def _home(self):
        for ax in self.figure._axes:
            ax._home_view()

    def _mode(self, mode):
        for a, m in ((self.a_pan, 'pan/zoom'), (self.a_zoom, 'zoom rect')):
            if m != mode and a.isChecked():
                a.blockSignals(True)
                a.setChecked(False)
                a.blockSignals(False)
        self.figure.canvas.toolbar.mode = mode
        for ax in self.figure._axes:
            ax._apply_toolbar_mode(mode)

    # (dialog filter, extension, how) for Figure.savefig; every entry saves exactly what this window shows.
    # 'vector': PDF / EPS / SVG through matplotlib (mplexport): the plots at their on-screen places, sizes
    # and text sizes, the canvas at its on-screen size and shape, the controls left out with their places
    # empty, and the full info panel (the settings of the controls among it) below, wrapped as on
    # screen.  'qt-svg': the whole window as Qt draws it, controls included, as shown (export_vector).
    # 'image': the window rendered at WINDOW_IMAGE_DPI (Figure.window_image), the info panel as shown.
    # 'csv': the curves' data (export_csv).
    SAVE_FORMATS = [
        ("PDF, vector, editable: the plots as on screen and the full info panel (*.pdf)", '.pdf', 'vector'),
        ("EPS, vector, editable: the plots as on screen and the full info panel (*.eps)", '.eps', 'vector'),
        ("SVG, vector, editable: the plots as on screen and the full info panel (*.svg)", '.svg', 'vector'),
        ("SVG, the whole window as Qt draws it, controls included, as shown (*.svg)", '.svg', 'qt-svg'),
        ("PNG, the window at 720 dpi (*.png)", '.png', 'image'),
        ("TIFF, the window at 720 dpi (*.tif)", '.tif', 'image'),
        ("CSV, the curves' data (*.csv)", '.csv', 'csv'),
    ]

    def _save(self):
        title = self.figure.manager.get_window_title() if self.figure.manager is not None else ''
        name = ''.join(ch if ch.isalnum() or ch in '._- ' else '_' for ch in (title or 'figure')).strip() or 'figure'
        formats = list(self.SAVE_FORMATS)
        last = getattr(NavigationToolbar, '_last_filter', None)
        filters = [f[0] for f in formats]
        path, flt = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save figure', name, ';;'.join(filters), last if last in filters else filters[0])
        if not path:
            return
        NavigationToolbar._last_filter = flt
        _, ext, how = next((f for f in formats if f[0] == flt), formats[0])
        # The chosen type decides: a missing or different extension is replaced by the type's.
        root, given = os.path.splitext(path)
        if given.lower() not in (ext, '.tiff' if ext == '.tif' else ext):
            path = (root if given.lower() in {f[1] for f in formats} else path) + ext
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = self.figure.savefig(path, how=how)
            print(f"saved: {out}")
        except Exception as exc:
            traceback.print_exc()
            print(f"save failed: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def show_coords(self, px, py):
        fig = self.figure
        if px is None:
            self.coords.setText('')
            return
        ax = fig._axes_at(px, py)
        if ax is None or ax._is_colorbar:
            self.coords.setText('')
            return
        (xa, xb), (ya, yb) = ax.vb.viewRange()
        x0, y0, w, h = ax._px_rect()
        fx, fy = (px - x0) / (w or 1), (y0 + h - py) / (h or 1)
        if ax.vb.state['yInverted']:
            fy = 1 - fy
        if ax.vb.state['xInverted']:
            fx = 1 - fx
        x, y = xa + fx * (xb - xa), ya + fy * (yb - ya)
        if ax._rotated:
            x, y = y, x
        self.coords.setText(f"x={x:.5g}  y={y:.5g}")


class FigureManager:
    """
    The figure's window (matplotlib's Qt window): the navigation toolbar on top, the canvas, and
    the info panel docked at the bottom (set_info: Figure.info_text, the details cut from the titles,
    the window's own lines and the settings of its controls; hidden while
    empty, and once closed by hand it stays closed).  The window opens at the canvas's size plus
    the toolbar, limited to the screen's available area (98 % of its width, 95 % of its height).
    """

    def __init__(self, figure, canvas, w_px, h_px):
        self.figure = figure
        self.canvas = canvas
        self.window = _Window(self)
        self.window.setCentralWidget(canvas)
        self.toolbar = NavigationToolbar(figure, self.window)
        self.window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        th = self.toolbar.sizeHint().height()
        scr = QtGui.QGuiApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            w_px = min(int(w_px), int(av.width() * 0.98))
            h_px = min(int(h_px), int(av.height() * 0.95) - th)
        # The info panel: the details cut from the panel titles, the window's own lines and the
        # settings of its controls (Figure.info_text), docked at the bottom, out of the way, wrapped
        # at its width (closable; hidden only until its first text, pushed when the window is shown).
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.info.setFont(qfont(7.5))
        self.info.setFixedHeight(int(pt2px(7.5) * FONT_SCALE * 4.2))
        self.dock = QtWidgets.QDockWidget('info', self.window)
        self.dock.setWidget(self.info)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                              | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock)
        self.dock.hide()
        self._dock_closed = False
        self.dock.visibilityChanged.connect(self._dock_vis)
        self.window.resize(int(w_px), int(h_px) + th)
        self._title = f"Figure {figure.number}"
        self.window.setWindowTitle(self._title)
        # The hook a control of this window calls when its setting changes (controls.Control._changed):
        # the info panel's settings section updated.
        self.window.sm4_settings_changed = figure._settings_changed

    def set_window_title(self, title):
        self._title = str(title)
        self.window.setWindowTitle(self._title)

    def get_window_title(self):
        return self._title

    def _dock_vis(self, visible):
        try:
            if not visible and self.info.toPlainText() and self.window.isVisible() and not self._setting:
                self._dock_closed = True          # closed by hand: stays closed
        except RuntimeError:                      # the window is being destroyed
            pass

    _setting = False

    def set_info(self, text):
        if text == self.info.toPlainText():
            return
        self.info.setPlainText(text)
        want = bool(text) and not self._dock_closed
        if want != self.dock.isVisible():
            # The window grows / shrinks by the panel so the figure keeps its size.
            self._setting = True
            h = self.dock.sizeHint().height()
            self.dock.setVisible(want)
            if self.figure._shown:
                self.window.resize(self.window.width(), self.window.height() + (h if want else -h))
            self._setting = False

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.figure._shown = True
        self.figure._settings_changed()        # the info panel's settings section, once the controls are on screen

    def _closed(self):
        self.figure._on_close()

    def _show_coords(self, px, py):
        for tb in self.figure._toolbars:
            tb.show_coords(px, py)


class _EmbeddedManager:
    """The manager of a figure embedded in another window's layout: no window of its own."""

    def __init__(self, figure, canvas):
        self.figure = figure
        self.canvas = canvas
        self.window = None
        self._title = ''

    @property
    def toolbar(self):
        """The figure's toolbar (NavigationToolbar(fig) made by its owner), else the canvas's navigation state (mode '')."""
        return self.figure._toolbars[0] if self.figure._toolbars else self.canvas.toolbar

    def set_window_title(self, title):
        self._title = str(title)

    def get_window_title(self):
        return self._title

    def show(self):
        self.figure._shown = True

    def _show_coords(self, px, py):
        for tb in self.figure._toolbars:
            tb.show_coords(px, py)


# ---------------------------------------------------------------------- #
#  Figure                                                                  #
# ---------------------------------------------------------------------- #

_figures = {}

# controls.settings_text, set by controls.py when it is imported: bound to the qtplot module that
# controls.py imported, so a window's info panel lists its controls' settings through the copy of
# controls that made them (a module reloader that drops the package from sys.modules cannot split
# them); None until controls.py is imported (no controls, no settings).
_settings_text = None


class Figure:
    """
    A figure: axes in the cells of its window's Qt layout (or at figure fractions), figure texts,
    the Qt controls of controls.py in the same layout, and (in its own window) a navigation toolbar
    and an info panel.  Its text zoom (font_zoom) and shape
    (portrait) follow the canvas (_relayout, on every resize); the design size it was opened with
    (_design_px) is the reference of both.
    """

    def __init__(self, figsize=(6.4, 4.8), num=None, embed=False, fill_screen=False):
        """
        embed=True: the figure lives in another window's layout (fig.canvas is the widget;
        NavigationToolbar(fig) its toolbar).  fill_screen=True: a working window that opens on
        the screen's available area (rule 11), its text zoom still relative to `figsize`.
        """
        _app()
        self._toolbars = []
        self._embedded = bool(embed)
        self.number = int(num) if num is not None else (max(_figures) + 1 if _figures else 1)
        w_px, h_px = float(figsize[0]) * DPI, float(figsize[1]) * DPI
        # The design size (matplotlib's figsize at DPI; every window declares it as a DESIGN
        # constant): the text is at FONT_SCALE x its points when the canvas has this size and
        # grows / shrinks with the canvas (font_zoom = window_zoom(...), ZOOM_MIN to ZOOM_MAX, the
        # one rule of every window), so it stays readable on any monitor and in a maximized window.
        # On a high-DPI screen the window opens proportionally larger.
        self._design_px = (w_px, h_px)
        self.font_zoom = 1.0
        self.portrait = False          # the window taller than wide: short labels and stamp titles (set_portrait)
        if not embed:
            k = screen_scale()
            w_px, h_px = w_px * k, h_px * k
        if fill_screen and not embed:
            scr = QtGui.QGuiApplication.primaryScreen()
            if scr is not None:
                av = scr.availableGeometry()
                w_px, h_px = av.width() * 0.98, av.height() * 0.95
        self._fill_screen = bool(fill_screen)
        self.canvas = FigureCanvas(self, w_px, h_px)
        if not self._embedded:
            # A window of its own: white panels on the window colour.
            self.canvas.setBackground(QtGui.QColor(WINDOW_BG))
        self.canvas.scene().setItemIndexMethod(QtWidgets.QGraphicsScene.ItemIndexMethod.NoIndex)
        self._axes = []
        self._texts = []
        self._qt_controls = []         # the controls of widgets.py on this figure (short labels, text zoom)
        self._layout = None            # the window's Qt layout on the canvas's viewport (set_layout)
        self._size_override = None     # (W, H) while the layout is applied at another size (at_design_size)
        self._title_cell = None        # the layout cell of the figure's title (set_title_cell)
        self._fitting = False
        self._fit_zoom = None          # the control zoom the last fit settled on
        self.control_zoom = 1.0        # the controls' text zoom: font_zoom, reduced to fit (_fit_controls)
        self._suptitle = None
        self._shown = False
        self._closed = False
        self.patch = _FigPatch()
        self.subplotpars = SubplotParams()
        self.transFigure = _Transform('figure', self)
        self.dpi = DPI
        if self._embedded:
            self.manager = _EmbeddedManager(self, self.canvas)
            self._shown = True
        else:
            self.manager = FigureManager(self, self.canvas, w_px, h_px)
            _figures[self.number] = self
        self.canvas.manager = self.manager
        # The info panel whose font and width the saved files' info text takes (mplexport._info_lines):
        # the window's own panel; a figure inside another window is given that window's.
        self.export_info_panel = getattr(self.manager, 'info', None)

    # --- the info panel: the details cut from the titles ---

    def _set_info(self, key, text):
        info = self.__dict__.setdefault('_info', {})
        text = str(text or '').strip()
        if info.get(key, '') == text:
            return
        if text:
            info[key] = text
        else:
            info.pop(key, None)
        self._settings_changed()

    def _settings_changed(self):
        """The info panel brought up to date once (_push_info), however many changes come in one event: a line set here, or a control's setting (controls.Control._changed, through the window's sm4_settings_changed)."""
        if not getattr(self, '_info_pending', False):
            self._info_pending = True
            QtCore.QTimer.singleShot(50, self._push_info)

    def set_info_line(self, key, text):
        """A window's own line in the information panel, under the panels' lines (the Line window's E-marker readout), by a name of its own; '' removes it."""
        self._set_info(('line', str(key)), text)

    def info_text(self):
        """
        The details of every panel's title, one line each (panels in their order on the figure), then
        the window's own lines (set_info_line) in the order they were first set; in a window of its
        own (a FigureManager), then an empty line, the line 'settings' and the settings of the
        window's controls (controls.settings_text).  A figure inside another window leaves the
        settings to that window.
        """
        info = getattr(self, '_info', {})
        order = ((['suptitle'] if 'suptitle' in info else []) + [ax for ax in self._axes if ax in info]
                 + [k for k in info if isinstance(k, tuple) and k[0] == 'line'])
        text = "\n".join(" ".join(info[k].split()) for k in order)
        if isinstance(self.manager, FigureManager):
            settings = _settings_text(self.manager.window) if _settings_text is not None else ''
            text = (text + "\n" if text else "") + "\nsettings" + ("\n" + settings if settings else "")
        return text

    def _push_info(self):
        self._info_pending = False
        try:
            text = self.info_text()
        except RuntimeError:
            return
        cb = getattr(self, 'on_info', None)
        if cb is not None:
            try:
                cb(text)
            except Exception:
                traceback.print_exc()
        if self.manager is not None and hasattr(self.manager, 'set_info') and not self._closed:
            try:
                self.manager.set_info(text)
            except RuntimeError:               # the window is already gone (closing, or Python exiting)
                pass

    # --- size ---

    def _canvas_size(self):
        if self._size_override is not None:
            return float(self._size_override[0]), float(self._size_override[1])
        if self._shown and (not self._embedded or self.canvas.isVisible()):
            vp = self.canvas.viewport()
            return float(max(vp.width(), 1)), float(max(vp.height(), 1))
        return float(self.canvas._size_px[0]), float(self.canvas._size_px[1])

    def get_size_inches(self):
        W, H = self._canvas_size()
        return np.array([W / DPI, H / DPI])

    def set_size_inches(self, w, h=None, forward=True):
        if h is None:
            w, h = w
        tw, th = int(round(float(w) * DPI)), int(round(float(h) * DPI))
        cw, ch = self._canvas_size()
        self.canvas._size_px = (tw, th)
        win = self.manager.window
        if win is not None:
            win.resize(win.width() + tw - int(cw), win.height() + th - int(ch))
            if self._layout is not None and self._shown:
                # The window's layout passes run now (its resize applied as matplotlib's forward=True does).
                QtWidgets.QApplication.sendPostedEvents()
        self._relayout()

    @property
    def axes(self):
        return list(self._axes)

    def get_axes(self):
        return list(self._axes)

    # --- layout ---

    def _relayout(self):
        new_measure_pass()
        if self._shown and (not self._embedded or self.canvas.isVisible()):
            vp = self.canvas.viewport()
            self.canvas._size_px = (vp.width(), vp.height())
        W0, H0 = self._design_px
        W, H = self._canvas_size()
        if not self._embedded and self._shown:
            self.set_portrait(H > W, relayout=False)   # an embedded figure takes its window's shape from its owner
        z = window_zoom(W, H, W0, H0)        # the one rule: the size relative to the design size, ZOOM_MIN to ZOOM_MAX
        if abs(z - self.font_zoom) > 0.03 * self.font_zoom:
            self._apply_zoom(z)
        if self._layout is not None:
            self._fit_controls()
            self._schedule_fit()
        for ax in self._axes:
            ax._place()
        for t in self._texts:
            t._place()
        W, H = self._canvas_size()
        for c in self._qt_controls:
            place = getattr(c, '_place', None)
            if place is None:
                continue                       # a control in the window's layout: Qt places it
            try:
                place(W, H)
            except Exception:
                traceback.print_exc()

    def _apply_zoom(self, z):
        """Every text of the figure at the zoom z: the toolbars, the axes, the figure texts, the info panel, the controls."""
        self.font_zoom = z
        for tb in self._toolbars:
            try:
                k = int(round(22 * screen_scale() * z))
                tb.setIconSize(QtCore.QSize(k, k))
                tb.coords.setFont(qfont(9, zoom=z))
            except RuntimeError:
                pass
        for ax in list(self._axes):
            ax._refont(z)
            cell = ax._cell
            if cell is not None and cell.inner_w is not None:
                cell.updateGeometry()          # a narrow cell's hint (inner_w at the zoom) is announced to its layout
        for ax in list(self._axes):
            # A placement during the loop above (a legend refitted, a row member placed) measured the room of
            # an axes not yet re-fonted, at the new zoom with the old fonts, and _decor_px would keep that as
            # the zoom's maximum: the new zoom measures afresh once every text is at it.
            ax._decor_max = None
        for t in self._texts:
            t._apply()
        if hasattr(self.manager, 'info'):
            try:
                self.manager.info.setFont(qfont(7.5, zoom=z))
                self.manager.info.setFixedHeight(int(pt2px(7.5) * FONT_SCALE * z * 4.2))
            except RuntimeError:
                pass
        if self._layout is not None:
            self._set_control_zoom(z)

    def _set_control_zoom(self, z):
        self.control_zoom = z
        for c in list(self._qt_controls):
            f = getattr(c, '_set_zoom', None)
            if f is not None:
                try:
                    f(z)
                except RuntimeError:
                    pass
        new_measure_pass()                             # the controls' size hints follow the font

    # --- automatic layout: the one rule for every window (nothing placed by hand) ---

    def set_layout(self, layout):
        """
        The window's Qt layout, on the canvas's viewport: its controls (Qt widgets) and its panels'
        PanelCells, arranged by Qt together (automatic layout: the one rule for every window).  The
        window cannot shrink below what the layout needs at the smallest control text
        (_fit_controls), Qt's own rule; a row of controls that wraps (controls.FlowLayout) lets a
        narrow window keep everything without overlap.
        """
        vp = self.canvas.viewport()
        layout.setContentsMargins(6, 6, 6, 6)
        vp.setLayout(layout)
        self._layout = layout
        self._set_control_zoom(self.font_zoom)
        self._fit_controls()

    def cell(self, min_px=(40, 40), stretch=(True, True), inner_w=None):
        """A new PanelCell of this figure for the window's layout (an axes or a text goes in it with cell=...); inner_w for a narrow panel (a colour bar)."""
        return PanelCell(self, min_px=min_px, stretch=stretch, inner_w=inner_w)

    def set_title_cell(self, cell):
        """The layout cell the figure's title (suptitle) is centred in from now on."""
        self._title_cell = cell
        if self._suptitle is not None:
            self.suptitle(getattr(self, '_suptitle_full', self._suptitle.get_text()), fontsize=self._suptitle.get_fontsize())

    def _fit_controls(self):
        """
        The controls' text is never cut and the window never scrolls: at the window's text zoom the
        layout is measured; when it needs more room than the canvas has, the controls' font is
        reduced step by step, to at most CONTROL_FIT_MIN of the zoom, until it fits; when even that
        does not fit, the
        canvas's smallest size is raised to what the layout needs, so nothing overlaps.
        """
        lay = self._layout
        if lay is None or self._fitting or self._size_override is not None:
            return
        if not (self._shown and self.canvas.isVisible()):
            return                             # measured once the window is on screen (its viewport has its size)
        self._fitting = True
        try:
            # The controls' size hints follow a new font only once their layouts measure again: every
            # layout inside the canvas is invalidated here, so the window's layout is measured as it now
            # is (only this canvas's layouts: the rest of the window is left to its own events).
            vp = self.canvas.viewport()
            for w in vp.findChildren(QtWidgets.QWidget):
                sub = w.layout()
                if sub is not None:
                    sub.invalidate()
            W, H = vp.width(), vp.height()
            z0 = self.font_zoom
            zc = z0
            m = None
            for _ in range(12):
                if abs(zc - self.control_zoom) > 1e-6:
                    self._set_control_zoom(zc)
                    QtWidgets.QApplication.sendPostedEvents()
                lay.invalidate()
                lay.activate()
                m = lay.minimumSize()
                if lay.hasHeightForWidth():
                    # A row of controls that wraps: the smallest height the layout needs at this width (the
                    # panel rows count their minimum, not their natural height: a window may be shorter than
                    # its panels ask, they shrink; rule 11's fit is Figure.fit_window's).
                    m = QtCore.QSize(m.width(), max(m.height(), total_min_hfw(lay, max(W, m.width()))))
                if fits_controls((m.width(), m.height()), (W, H), zc, self._fit_zoom):
                    self._fit_zoom = zc
                    return
                if zc <= z0 * CONTROL_FIT_MIN + 1e-9:
                    break
                zc = max(zc * 0.96, z0 * CONTROL_FIT_MIN)
            if m is not None:
                fw = self.canvas.width() - W
                fh = self.canvas.height() - H
                self.canvas.setMinimumSize(max(self.canvas.minimumWidth(), m.width() + fw),
                                           max(self.canvas.minimumHeight(), m.height() + fh))
            self._fit_zoom = zc
        finally:
            self._fitting = False

    @staticmethod
    def _panel_rows(lay):
        """Every PanelRow in a layout tree (the automatic grid's rows, a window's own rows)."""
        out = []
        for i in range(lay.count()):
            it = lay.itemAt(i)
            sub = it.layout()
            if isinstance(sub, PanelRow):
                out.append(sub)
            elif sub is not None:
                out += Figure._panel_rows(sub)
            elif it.widget() is not None and it.widget().layout() is not None and not isinstance(it.widget(), PanelCell):
                out += Figure._panel_rows(it.widget().layout())
        return out

    @staticmethod
    def _panel_items(lay):
        """
        Every panel-holding item of a layout tree as (item, shaped): a PanelRow (shaped when it has a
        shaped item), a cell of an axes or a slot (shaped when its axes keeps a ratio); the free
        ones absorb the window's height.  Cells without an axes (a title cell) and controls are not
        panels.
        """
        out = []
        for i in range(lay.count()):
            it = lay.itemAt(i)
            sub = it.layout()
            w = it.widget()
            if isinstance(sub, PanelRow):
                out.append((sub, sub.hasHeightForWidth()))
            elif sub is not None:
                out += Figure._panel_items(sub)
            elif isinstance(w, PanelCell):
                if any(isinstance(o, Axes) for o in w._bound) and not w.inner_w:
                    out.append((w, w.shape_ratio() is not None))
            elif isinstance(w, _Slot):
                out.append((w, w.cell.shape_ratio() is not None))
            elif w is not None and w.layout() is not None:
                out += Figure._panel_items(w.layout())
        return out

    @staticmethod
    def _find_pane(lay):
        """The first widget of a layout tree that answers fit_delta (a controls.OrientedPanes), or None."""
        for i in range(lay.count()):
            it = lay.itemAt(i)
            sub, w = it.layout(), it.widget()
            if w is not None and hasattr(w, 'fit_delta'):
                return w
            found = None
            if sub is not None:
                found = Figure._find_pane(sub)
            elif w is not None and w.layout() is not None:
                found = Figure._find_pane(w.layout())
            if found is not None:
                return found
        return None

    def _fit_pane(self):
        """The window's oriented panes (controls.OrientedPanes) when its layout holds one: the window's fit is theirs (fit_delta)."""
        return self._find_pane(self._layout) if self._layout is not None else None

    def _shaped_layout(self):
        """Whether the window's height follows its panels: every panel row and panel of its layout is shaped (none is free: a free panel absorbs the height), and there is one."""
        lay = self._layout
        if lay is None:
            return False
        items = self._panel_items(lay)
        return bool(items) and all(shaped for _, shaped in items)

    def _schedule_fit(self):
        """The fit after a resize (rule 11: the window always adjusts), a moment after the last resize event."""
        if self._embedded or not self._shown or getattr(self, '_fitting_window', False) or self._size_override is not None:
            return
        t = getattr(self, '_fit_timer', None)
        if t is None:
            t = self._fit_timer = QtCore.QTimer()
            t.setSingleShot(True)
            t.timeout.connect(self.fit_window)
        t.start(250)

    def fit_window(self):
        """
        The window sized to its panels (rule 11 of the guide).  For a window whose panel rows are
        all shaped (_shaped_layout: maps, images, slices; a window with a free row of curves
        absorbs any height and is left alone): when the layout's natural height for the canvas's
        width (every row as tall as its shaped panels ask) is less than the canvas's height, the
        window is made that much shorter; when it is more (the panels are limited by the height),
        the window is made narrower until the natural height fits.  For a window of oriented
        panes (controls.OrientedPanes, _fit_pane) the panes say what the window can lose
        (fit_delta: the height beside a main panel at the left, the width beside one on top).
        Done at the first show (qtplot.show), a moment after every resize (_schedule_fit: the
        height follows the width the user sets), and by a window after it rebuilt its panels.
        Returns True when the window was resized.
        """
        lay = self._layout
        win = getattr(self.manager, 'window', None)
        if lay is None or win is None or self._embedded or not self._shown or getattr(self, '_fitting_window', False):
            return False
        if self._size_override is not None or getattr(self, '_closed', False):
            return False
        pane = self._fit_pane()
        try:
            if pane is None and (not lay.hasHeightForWidth() or not self._shaped_layout()):
                return False
        except Exception:
            return False
        self._fitting_window = True
        resized = False
        try:
            # A few passes: a resize re-measures the ticks' room and the controls' wrapping, which can
            # move the natural height a little; the loop ends when the canvas meets it (2 px).
            for _ in range(4):
                QtWidgets.QApplication.sendPostedEvents()
                vp = self.canvas.viewport()
                W, H = vp.width(), vp.height()
                dw = dh = 0
                if pane is not None:
                    pane.rearrange()
                    pw, ph = pane.fit_delta()
                    dw, dh = (pw if pw < -2 else 0), (ph if ph < -2 else 0)
                    nat = H
                else:
                    nat = lay.heightForWidth(W)
                if pane is not None:
                    pass
                elif nat < H - 2:
                    dh = nat - H
                elif nat > H + 2:
                    lo, hi = float(max(lay.minimumSize().width(), 50)), float(W)
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        if lay.heightForWidth(int(mid)) <= H:
                            lo = mid
                        else:
                            hi = mid
                    dw = int(round(lo)) - W
                if dw == 0 and dh == 0:
                    break
                before = (win.width(), win.height())
                win.resize(max(win.width() + dw, win.minimumWidth()), max(win.height() + dh, win.minimumHeight()))
                QtWidgets.QApplication.sendPostedEvents()
                self._relayout()
                QtWidgets.QApplication.sendPostedEvents()
                if (win.width(), win.height()) == before:
                    break                          # the window cannot change (its minimum): done
                resized = True
            return resized
        finally:
            self._fitting_window = False

    @contextlib.contextmanager
    def at_design_size(self):
        """
        The window's layout applied at its design size with every text at zoom 1, while the block
        runs (the publication export lays a window out at its design size, whatever its size on
        screen); the window's own size, zoom and layout come back afterwards.
        """
        if self._layout is None:
            yield
            return
        W0, H0 = self._design_px
        z_old, zc_old = self.font_zoom, self.control_zoom
        old_geo = self._layout.geometry()
        self._size_override = (float(W0), float(H0))
        try:
            self._apply_zoom(1.0)
            self._set_control_zoom(1.0)
            QtWidgets.QApplication.sendPostedEvents()
            self._layout.invalidate()
            self._layout.setGeometry(QtCore.QRect(0, 0, int(W0), int(H0)))
            for ax in self._axes:
                ax._place()
            for t in self._texts:
                t._place()
            yield
        finally:
            self._size_override = None
            self._apply_zoom(z_old)
            self._set_control_zoom(zc_old)
            QtWidgets.QApplication.sendPostedEvents()
            self._layout.invalidate()
            self._layout.setGeometry(old_geo)
            for ax in self._axes:
                ax._place()
            for t in self._texts:
                t._place()

    def _axes_at(self, px, py):
        for ax in reversed(self._axes):
            if not ax._visible:
                continue
            x0, y0, w, h = ax._px_rect()
            if x0 <= px <= x0 + w and y0 <= py <= y0 + h:
                return ax
        return None

    def _pick(self, e):
        ax = e.inaxes
        if ax is None:
            return
        hits = [(a, getattr(a, '_hit_index', None)) for a in ax._children if a.pickable() and a.contains_px(e.x, e.y)]
        for a, ind in hits:
            ev = PickEvent(self.canvas, e, a)
            ev.ind = [ind] if ind is not None else []
            self.canvas._fire('pick_event', ev)

    def _register_control(self, c):
        self._qt_controls.append(c)
        place = getattr(c, '_place', None)
        if place is not None:
            W, H = self._canvas_size()
            place(W, H)
            return
        # A control in the window's layout (controls.py): Qt places it; its text follows the figure.
        f = getattr(c, '_set_zoom', None)
        if f is not None:
            f(self.control_zoom)
        s = getattr(c, 'set_short', None)
        if s is not None and self.portrait:
            s(True)

    def _unregister_control(self, c):
        try:
            self._qt_controls.remove(c)
        except ValueError:
            pass

    # --- axes ---

    def add_axes(self, rect=None, sharex=None, sharey=None, cell=None, **kw):
        """An axes in a layout cell (`cell`, a PanelCell: automatic layout) or at a figure-fraction rectangle `rect`."""
        if isinstance(rect, Axes):
            return rect
        if cell is not None:
            ax = Axes(self, rect=None, sharex=sharex, sharey=sharey, cell=cell)
        else:
            b = rect.bounds if hasattr(rect, 'bounds') else rect
            ax = Axes(self, rect=b, sharex=sharex, sharey=sharey)
        if kw.get('facecolor') is not None:
            ax.patch.set_facecolor(kw['facecolor'])
        return ax

    # --- automatic layout of matplotlib-style subplots (the one rule for every window) ---

    def _auto_grid(self):
        """
        The figure's automatic grid, made at the first subplot: a Qt grid on the canvas (set_layout)
        whose slots hold the subplots' cells, its column and row stretches the gridspec's width and
        height ratios; None when the window has a layout of its own (its panels are its own cells).
        """
        auto = getattr(self, '_auto', None)
        if auto is not None:
            return auto['rows_lay']
        if self._layout is not None:
            return None
        root = QtWidgets.QVBoxLayout()
        rows_lay = self._new_rows_layout()
        root.addLayout(rows_lay, 1)
        self._auto = {'root': root, 'rows_lay': rows_lay, 'rows': {}, 'grid': None}
        self.set_layout(root)
        return rows_lay

    @staticmethod
    def _new_rows_layout():
        """The column of PanelRows of the automatic grid (rule 11: every row as tall as its shaped panels ask)."""
        lay = QtWidgets.QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        return lay

    def _auto_row(self, r, gs):
        """The PanelRow of gridspec row `r`, made and put in order when missing, its stretch the row's height ratio."""
        auto = self._auto
        rows = auto['rows']
        if r not in rows:
            row = PanelRow(spacing=12)
            rows[r] = row
            pos = sorted(rows).index(r)
            stretch = max(int(round(1000 * gs.height_ratios[r] / max(sum(gs.height_ratios), 1e-12))), 1)
            auto['rows_lay'].insertLayout(pos, row, stretch)
        return rows[r]

    def _auto_span_grid(self):
        """The Qt grid a subplot spanning rows is put in (the one layout the PanelRows cannot hold), made once, after the rows."""
        auto = self._auto
        if auto['grid'] is None:
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(10)
            auto['rows_lay'].addLayout(grid, 1)
            auto['grid'] = grid
        return auto['grid']

    def layout_root(self):
        """The automatic grid's outer layout (made now if needed): a script adds its own controls under (or over) its subplots here, so they are laid out with them."""
        self._auto_grid()
        auto = getattr(self, '_auto', None)
        return auto['root'] if auto is not None else self._layout

    def _auto_axes(self, spec, sharex=None, sharey=None):
        """An axes for a SubplotSpec in the automatic grid (its slot spanning the spec's rows and columns), or None (a nested spec, or a window with a layout of its own: the spec's rectangle)."""
        gs = spec.get_gridspec()
        if not isinstance(gs, GridSpec):
            return None
        rows_lay = self._auto_grid()
        if rows_lay is None:
            return None
        rows = range(gs.nrows)[spec.rows]
        cols = range(gs.ncols)[spec.cols]
        slot = _Slot(self)
        cell = slot.cell
        slot._col = cols[0]
        width_stretch = max(int(round(1000 * sum(gs.width_ratios[c] for c in cols) / max(sum(gs.width_ratios), 1e-12))), 1)
        if len(rows) > 1:
            # A subplot spanning rows: the Qt grid (its stretches the ratios), the one case a row of panels cannot hold.
            grid = self._auto_span_grid()
            for c in range(gs.ncols):
                grid.setColumnStretch(c, max(int(round(1000 * gs.width_ratios[c] / max(sum(gs.width_ratios), 1e-12))), 1))
            for r in range(gs.nrows):
                grid.setRowStretch(r, max(int(round(1000 * gs.height_ratios[r] / max(sum(gs.height_ratios), 1e-12))), 1))
            grid.addWidget(slot, rows[0], cols[0], len(rows), len(cols))
            slot._row = grid
        else:
            # The row of the gridspec row: the slot at its column position among the row's slots (automatic layout; rule 11).
            row = self._auto_row(rows[0], gs)
            idx = sum(1 for i in range(row.count())
                      if row.itemAt(i).widget() is not None and getattr(row.itemAt(i).widget(), '_col', -1) < cols[0])
            row.insertWidget(idx, slot, width_stretch)
            slot._row = row
        slot.show()
        return Axes(self, cell=cell, spec=spec, sharex=sharex, sharey=sharey)

    def _auto_dispose(self, cell):
        """A cell of the automatic grid left without axes: its slot goes when none of its cells holds one."""
        slot = getattr(cell, '_slot', None)
        if slot is None:
            return
        if cell is not slot._cells[0] and not cell._bound:
            # A colour bar's cell beside a panel: it goes with its bar.
            slot._cells.remove(cell)
            slot.layout().removeWidget(cell)
            cell.setParent(None)
            cell.deleteLater()
        if any(c._bound for c in slot._cells):
            return
        row = getattr(slot, '_row', None)
        if row is not None:
            row.removeWidget(slot)
            auto = getattr(self, '_auto', None)
            if auto is not None and isinstance(row, PanelRow) and row.count() == 0:
                # An emptied row leaves the column (its room goes to the rows left).
                for r, lay in list(auto['rows'].items()):
                    if lay is row:
                        del auto['rows'][r]
                auto['rows_lay'].removeItem(row)
                row.deleteLater()
        slot.setParent(None)
        slot.deleteLater()

    def add_subplot(self, *args, sharex=None, sharey=None, **kw):
        if len(args) == 1 and isinstance(args[0], SubplotSpec):
            spec = args[0]
        else:
            if len(args) == 1:
                n = int(args[0])
                args = (n // 100, (n // 10) % 10, n % 10)
            if not args:
                args = (1, 1, 1)
            nrows, ncols, index = args
            gs = GridSpec(nrows, ncols)
            if isinstance(index, tuple):
                a, b = index
                spec = gs[(a - 1):b]
            else:
                spec = gs[int(index) - 1]
        ax = self._auto_axes(spec, sharex=sharex, sharey=sharey)
        return ax if ax is not None else Axes(self, spec=spec, sharex=sharex, sharey=sharey)

    def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, width_ratios=None,
                 height_ratios=None, gridspec_kw=None, **kw):
        """matplotlib's subplots, each in a slot of the figure's automatic grid (automatic layout: the one rule for every window; the gridspec's ratios its stretches, the spacing Qt's)."""
        gk = dict(gridspec_kw or {})
        gs = GridSpec(nrows, ncols, width_ratios=width_ratios or gk.get('width_ratios'),
                      height_ratios=height_ratios or gk.get('height_ratios'),
                      wspace=gk.get('wspace'), hspace=gk.get('hspace'))
        axs = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                shx = axs[0, 0] if (sharex in (True, 'all') and (r or c)) else None
                shy = axs[0, 0] if (sharey in (True, 'all') and (r or c)) else None
                ax = self._auto_axes(gs[r, c], sharex=shx, sharey=shy)
                axs[r, c] = ax if ax is not None else Axes(self, spec=gs[r, c], sharex=shx, sharey=shy)
        if squeeze:
            if nrows == 1 and ncols == 1:
                return axs[0, 0]
            if nrows == 1 or ncols == 1:
                return axs.ravel()
        return axs

    def add_gridspec(self, nrows=1, ncols=1, **kw):
        return GridSpec(nrows, ncols, figure=self, **kw)

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        sp = self.subplotpars
        for k, v in (('left', left), ('bottom', bottom), ('right', right), ('top', top), ('wspace', wspace),
                     ('hspace', hspace)):
            if v is not None:
                setattr(sp, k, float(v))
        for ax in self._axes:
            if ax._spec is not None:
                ax._rect = None
                ax._place()

    def delaxes(self, ax):
        ax.remove()

    def sca(self, ax):
        return ax

    def gca(self):
        return self._axes[-1] if self._axes else self.add_subplot(111)

    # --- texts ---

    def text(self, x, y, s, cell=None, **kw):
        """A figure text at (x, y): fractions of `cell` (a PanelCell of the window's layout) when given, else of the figure."""
        kw.pop('transform', None)
        t = Text(self, x, y, s, 'figure', cell=cell, **kw)
        self._texts.append(t)
        return t

    def _forget_text(self, t):
        try:
            self._texts.remove(t)
        except ValueError:
            pass
        if self._suptitle is t:
            self._suptitle = None

    def suptitle(self, t, fontsize=None, y=0.98, x=0.5, ha='center', va='top', **kw):
        """
        The figure's title (a figure text at x, y): shown as short_title(t, portrait), the full
        text in the info panel when they differ; 12 pt unless given.
        """
        if self._suptitle is not None:
            self._suptitle.remove()
        full = '' if t is None else str(t)
        self._suptitle_full = full
        t = short_title(full, self.portrait)
        self._set_info('suptitle', full if full.strip() != t else '')
        auto = getattr(self, '_auto', None)
        if self._title_cell is None and auto is not None:
            # The automatic grid's title cell, made at the first title, above the grid.
            cell = PanelCell(self, min_px=(40, 24), stretch=(True, False))
            auto['root'].insertWidget(0, cell)
            self._title_cell = cell
        if self._title_cell is not None:
            # The title in its own cell of the window's layout (automatic layout), centred in it; an
            # empty title hides the cell (no empty band stays when a window drops its title).
            self._title_cell.setVisible(bool(t.strip()))
            self._suptitle = self.text(0.5, 0.5, t, cell=self._title_cell, fontsize=font_pt(fontsize, 12.0),
                                       ha='center', va='center', fontweight=kw.get('fontweight'))
            return self._suptitle
        self._suptitle = self.text(x, y, t, fontsize=font_pt(fontsize, 12.0), ha=ha, va=va,
                                   fontweight=kw.get('fontweight'))
        return self._suptitle

    def set_portrait(self, flag, relayout=True):
        """
        The figure's shape for the short forms: True when its window is taller than it is wide.
        A figure in its own window sets it from its canvas (_relayout, with relayout=False); an
        embedded figure is told by the window that holds it.  On a change every
        control with a short form switches (controls.Control.set_short) and every file-name title
        to its stamp and back, and the window's layout sizes the controls to their new text.
        """
        flag = bool(flag)
        if flag == self.portrait:
            return
        self.portrait = flag
        for ax in list(self._axes):
            ax._retitle()
        if self._suptitle is not None and getattr(self, '_suptitle_full', ''):
            self._suptitle.set_text(short_title(self._suptitle_full, flag))
        for c in self._qt_controls:
            f = getattr(c, 'set_short', None)
            if f is not None:
                try:
                    f(flag)
                except Exception:
                    traceback.print_exc()
        new_measure_pass()                             # the labels' size hints follow their text
        if relayout:
            self._relayout()

    # --- colour bars ---

    def colorbar(self, mappable, cax=None, ax=None, fraction=0.15, pad=0.05, label=None, shrink=1.0, aspect=20,
                 location='right', **kw):
        """
        A Colorbar for `mappable` (an AxesImage / QuadMesh, or an object with a `mappable`) in
        `cax`; without `cax`, beside `ax` (the mappable's own axes when None; a list of axes: one
        bar for the row they share) in a narrow cell of the automatic grid, on its right, or on
        the left of the first panel with location='left' (matplotlib's keyword).  An axes placed
        by figure fractions gives up `fraction` + `pad` of its width on the right, and the bar
        takes `fraction` of it (narrower when `aspect`, height over width, asks), `shrink` of the
        height, centred; Colorbar.remove restores the axes.
        """
        mappable = getattr(mappable, 'mappable', mappable) if not isinstance(mappable, (AxesImage, QuadMesh)) else mappable
        parent_rect = None
        group = None
        if isinstance(ax, (list, tuple, np.ndarray)):
            # One bar for several panels (matplotlib's ax=[...]): beside the row they share.
            group = [a for a in np.ravel(ax) if a is not None]
            ax = (group[0] if location == 'left' else group[-1]) if group else None
        if cax is None and ax is None:
            ax = getattr(mappable, 'axes', None)          # matplotlib's default: the mappable's axes
        if cax is None and ax is not None and getattr(getattr(ax, '_cell', None), '_slot', None) is not None:
            # A subplot of the automatic grid: the bar in a narrow cell of its own beside the panel, in the
            # panel's slot (automatic layout), as tall as the panel (align_row).
            slot = ax._cell._slot
            bar = PanelCell(self, inner_w=15)
            bar._slot = slot
            if location == 'left':
                slot.layout().insertWidget(0, bar)
            else:
                slot.layout().addWidget(bar)
            slot._cells.append(bar)
            cax = Axes(self, cell=bar)
            row = getattr(ax, '_row', None)
            if group:
                self.align_row(*(list(group) + [cax]))
            elif row is not None:
                row.append(cax)
                cax._row = row
                cax._request_place()
            else:
                self.align_row(ax, cax)
            return Colorbar(self, mappable, cax, label=label, parent_axes=ax, parent_rect=None)
        if cax is None:
            if ax is None:
                ax = self._axes[0] if self._axes else self.add_subplot(111)
            x0, y0, w, h = ax.get_position().bounds
            parent_rect = (x0, y0, w, h)
            W, H = self._canvas_size()
            cb_w = w * fraction
            pad_w = w * pad
            ax.set_position([x0, y0, w - cb_w - pad_w, h])
            hh = h * shrink
            cw = min(cb_w, (hh * H / float(aspect)) / W)
            cax = Axes(self, rect=[x0 + w - cb_w, y0 + (h - hh) / 2.0, cw, hh])
            # The bar is as tall as the axes it belongs to, whatever aspect the axes' data give it (align_row).
            row = getattr(ax, '_row', None)
            if row is not None:
                row.append(cax)
                cax._row = row
                cax._request_place()
            else:
                self.align_row(ax, cax)
        cb = Colorbar(self, mappable, cax, label=label, parent_axes=ax, parent_rect=parent_rect)
        return cb

    def align_row(self, *axes):
        """
        One row of panels: every axes given takes the same height, the smallest an adjustable-'box'
        locked aspect among the row's visible panels asks for (Axes._own_height; a spatial panel
        keeps its aspect, its width following; a colour bar, a ramp or a curve panel stretches or
        squishes to the height), each centred on its own cell, so that a row stays neat whatever
        shape the data give its spatial panels.  A limits change on one re-places them all; an axes
        belongs to one row, and a removed axes leaves its row.  Returns the row (a list).
        """
        row = [ax for ax in axes if ax is not None]
        for ax in row:
            old = getattr(ax, '_row', None)
            if old is not None and ax in old:
                old.remove(ax)
            ax._row = row
        for ax in row:
            ax._request_place()
        return row

    # --- output, closing ---

    def savefig(self, path, dpi=None, how=None, **kw):
        """
        Save exactly what the window shows, at the window's size and shape, nothing laid out again
        and nothing cropped.  Returns the path written.  `facecolor` and `transparent` are passed
        on to the vector files (mplexport.export: the file's background); `dpi` and any other
        keyword (matplotlib's savefig arguments) are accepted and ignored: the image formats are
        always at WINDOW_IMAGE_DPI.  how:
          'vector' (default for .pdf .eps .ps .svg): written by matplotlib (mplexport.export): every
                   panel, curve, heatmap, stripe, marker, legend, colour bar and figure text at its
                   on-screen place and size, text at its on-screen size and editable; the controls
                   left out with their places empty; the full info panel (its settings section
                   among it) below the plots, wrapped as on screen: below the canvas, or below the
                   bottom of an axes drawn at its full height (Axes.export_extent) when that reaches
                   past the canvas;
          'qt-svg' : the whole window as Qt draws it, controls included, as shown (export_vector;
                   PDF and, through Ghostscript, EPS as well);
          'image'  (default for every other extension, e.g. .png .tif .tiff .jpg .bmp): the window
                   rendered at WINDOW_IMAGE_DPI, the info panel as shown (window_image);
          'csv'    (default for .csv): the curves' data, one column pair per curve (export_csv;
                   an axes with csv False left out).
        """
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if how is None:
            if ext in ('.pdf', '.eps', '.ps', '.svg'):
                how = 'vector'
            elif ext == '.csv':
                how = 'csv'
            else:
                how = 'image'
        if how == 'csv':
            return export_csv(self, path)
        if how == 'vector':
            from qtgui import mplexport
            return mplexport.export(self, path, facecolor=kw.get('facecolor'),
                                    transparent=kw.get('transparent', False))
        if how == 'qt-svg':
            return export_vector(self, path)
        return self.window_image(path)

    def export_root(self):
        """
        The widget the image and Qt routes render (window_image, export_vector): the widget a window
        set as fig.export_root_widget, else this figure's window, of which the central area (the
        canvas with its controls) and the info panel's dock as shown are saved, the toolbar not
        (_export_rect); a figure without a window of its own and no export_root_widget: its canvas.
        """
        w = getattr(self, 'export_root_widget', None)
        if w is not None:
            return w
        return self.manager.window if self.manager.window is not None else self.canvas

    def _export_rect(self):
        """
        The part of export_root() that is saved, in its own coordinates: for this figure's own
        window, the canvas (the central widget, below the toolbar) and, when it is shown docked,
        the info panel's dock; else the whole widget.
        """
        root = self.export_root()
        win = self.manager.window
        if win is None or root is not win:
            return root.rect()
        r = self.canvas.geometry()
        dock = self.manager.dock
        if dock.isVisible() and not dock.isFloating():
            r = r.united(dock.geometry())
        return r

    def export_info(self):
        """
        The full text of the info for the saved files: the text a window gives through
        fig.export_info_fn, else this window's info panel as it is (which holds the settings
        section, info_text); '' when there is none.
        """
        fn = getattr(self, 'export_info_fn', None)
        if fn is not None:
            return str(fn() or '')
        info = getattr(self.manager, 'info', None)
        return info.toPlainText() if info is not None else ''

    def window_image(self, path, dpi=WINDOW_IMAGE_DPI):
        """
        A high-definition screenshot of the window: export_root() (its saved part, _export_rect)
        rendered with its children into a QImage of its size times dpi / DPI (a QPainter scaled by
        that factor), the info panel as shown, saved by Qt's image writer (the format from the
        extension) with dpi as its resolution.  Returns the path.
        """
        root, rect = self.export_root(), self._export_rect()
        k = float(dpi) / DPI
        img = QtGui.QImage(max(1, int(round(rect.width() * k))), max(1, int(round(rect.height() * k))),
                           QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor('white'))
        p = QtGui.QPainter(img)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            p.scale(k, k)
            root.render(p, QtCore.QPoint(0, 0), QtGui.QRegion(rect),
                        QtWidgets.QWidget.RenderFlag.DrawWindowBackground | QtWidgets.QWidget.RenderFlag.DrawChildren)
        finally:
            p.end()
        dpm = int(round(float(dpi) / 0.0254))         # dots per metre
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if not img.save(path):
            raise OSError(f"Qt could not write {path} (no image writer for {os.path.splitext(path)[1] or 'that type'})")
        return path

    def show(self):
        self.manager.show()

    def clf(self):
        """Every axes and figure text removed; the automatic grid's title cell is hidden until a suptitle is set again."""
        for ax in list(self._axes):
            ax.remove()
        for t in list(self._texts):
            t.remove()
        if self._title_cell is not None and getattr(self, '_auto', None) is not None:
            self._title_cell.setVisible(False)
        auto = getattr(self, '_auto', None)
        if auto is not None:
            # The automatic grid starts afresh: a window rebuilt with fewer rows or columns must not keep
            # the rows (room) it had.
            root, old = auto['root'], auto['rows_lay']
            idx = root.indexOf(old)
            root.removeItem(old)
            old.deleteLater()
            rows_lay = self._new_rows_layout()
            root.insertLayout(max(idx, 0), rows_lay, 1)
            auto.update(rows_lay=rows_lay, rows={}, grid=None)

    clear = clf

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        self.canvas._fire('close_event', CloseEvent(self.canvas))
        if _figures.get(self.number) is self:
            del _figures[self.number]

    def tight_layout(self, *a, **k):
        pass

    def legend(self, *a, **k):
        return None


def export_csv(fig, path):
    """Every curve of every visible panel as two columns (x, y), headed by panel title and label; an axes with csv False (Axes.csv) is left out."""
    cols, heads = [], []
    for k, ax in enumerate(fig._axes):
        if not ax._visible or not ax.csv:
            continue
        title = ax._title.get_text() if ax._title is not None else f"panel {k + 1}"
        for a in ax._children:
            if isinstance(a, Line2D) and a.get_visible() and len(a._x):
                lab = a.get_label()
                lab = '' if lab.startswith('_') else lab
                heads += [f"{title} | {lab} | {ax.get_xlabel() or 'x'}", f"{title} | {lab} | {ax.get_ylabel() or 'y'}"]
                cols += [a._x, a._y]
    n = max((len(c) for c in cols), default=0)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(','.join('"' + h.replace('"', "'") + '"' for h in heads) + '\n')
        for i in range(n):
            fh.write(','.join(('' if i >= len(c) or not np.isfinite(c[i]) else repr(float(c[i]))) for c in cols) + '\n')
    return path


def _render_scene(fig, painter):
    """
    Draw the saved part of the window (fig.export_root(), its area fig._export_rect(): the canvas
    with its controls and the info panel as shown) into a painter at (0, 0), with its children,
    the scene's item caches off (every item drawn as vectors, none from a cached pixmap) and
    QuadItem.EXPORTING set (each cell edged by a hairline of its colour).
    """
    scene = fig.canvas.scene()
    cached = []
    for it in scene.items():
        if it.cacheMode() != QtWidgets.QGraphicsItem.CacheMode.NoCache:
            cached.append((it, it.cacheMode()))
            it.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)
    QuadItem.EXPORTING = True
    try:
        fig.export_root().render(painter, QtCore.QPoint(0, 0), QtGui.QRegion(fig._export_rect()),
                                 QtWidgets.QWidget.RenderFlag.DrawWindowBackground
                                 | QtWidgets.QWidget.RenderFlag.DrawChildren)
    finally:
        QuadItem.EXPORTING = False
        for it, mode in cached:
            it.setCacheMode(mode)


@contextlib.contextmanager
def _pixel_fonts(fig):
    """
    While the block runs, every font of the saved part of the window given as a pixel size equal to
    its size on screen; afterwards every font as it was, so the window is unchanged (export_vector's
    SVG).  The fonts: those of export_root() and of every widget in it, and in the scene of every
    graphics view among them the text items' fonts (QGraphicsTextItem, QGraphicsSimpleTextItem: the
    titles, texts, axis labels and legend entries) and the tick axes' tick fonts.  Qt lays a
    point-size font out at qRound(points * dpi / 72) pixels of the device it lays it out for, and its
    SVG writer writes it as points * resolution / 72 (qfontToSvg), so in a 720 dpi file every such
    text would come out about 7.5 times its size beside the panels; a pixel-size font is laid out
    and written at its pixel size on every device.  Its size on screen: QFontInfo's pixel size of
    the font where it is drawn on screen, a widget's font as Qt keeps it (at the widget's screen
    resolution), a text item's font as it is (its document lays the text out at the font's own
    resolution), a tick font at the view's viewport (the device its painter draws on).  A font
    already given in pixels is left as it is; a widget whose font was inherited inherits it again
    afterwards (setFont(QFont())).
    """
    def in_px(f, device=None):
        g = QtGui.QFont(f)
        g.setPixelSize(max(1, int(QtGui.QFontInfo(QtGui.QFont(f, device) if device is not None else f).pixelSize())))
        return g

    root = fig.export_root()
    own = Qt.WidgetAttribute.WA_SetFont
    widgets, texts, ticks, scenes = [], [], [], []
    # Every font is read before any is changed (a widget's new font reaches the children that inherit it).
    for w in [root] + root.findChildren(QtWidgets.QWidget):
        f = QtGui.QFont(w.font())
        if f.pixelSize() <= 0:
            widgets.append((w, f, w.testAttribute(own), in_px(f)))
    for v in [root] + root.findChildren(QtWidgets.QGraphicsView):
        sc = v.scene() if isinstance(v, QtWidgets.QGraphicsView) else None
        if sc is None or any(sc is s for s in scenes):
            continue
        scenes.append(sc)
        for it in sc.items():
            if isinstance(it, (QtWidgets.QGraphicsTextItem, QtWidgets.QGraphicsSimpleTextItem)):
                f = QtGui.QFont(it.font())
                if f.pixelSize() <= 0:
                    texts.append((it, f, in_px(f)))
            elif isinstance(it, pg.AxisItem):
                f = it.style.get('tickFont')
                if f is not None and f.pixelSize() <= 0:
                    ticks.append((it, f, in_px(f, v.viewport())))
    try:
        for w, f, mine, g in widgets:
            w.setFont(g)
        for it, f, g in texts:
            it.setFont(g)
        for it, f, g in ticks:
            it.setTickFont(g)
        yield
    finally:
        for w, f, mine, g in widgets:
            w.setFont(f if mine else QtGui.QFont())
        for it, f, g in texts:
            it.setFont(f)
        for it, f, g in ticks:
            it.setTickFont(f)


def _have_matplotlib():
    """Whether matplotlib is installed (without importing it: the Save dialog's publication formats)."""
    try:
        import importlib.util
        return importlib.util.find_spec('matplotlib') is not None
    except Exception:
        return False


def _ghostscript():
    """The Ghostscript executable on the PATH (gswin64c, gswin32c or gs), or None."""
    import shutil
    for name in ('gswin64c', 'gswin32c', 'gs'):
        p = shutil.which(name)
        if p:
            return p
    return None


def export_vector(fig, path):
    """
    The whole window as Qt draws it, exactly as shown, as a vector file (the Save dialog's
    'qt-svg'): export_root() (its saved part, fig._export_rect(): the canvas with its controls and
    the info panel as shown) rendered through Qt's own vector writers (_render_scene), nothing
    added below it, so the page is the saved part's size, W / DPI by H / DPI inches, the physical
    size of the window images (window_image).  SVG (QSvgGenerator at WINDOW_IMAGE_DPI, as the window
    images, with every font drawn as its pixel size on screen, _pixel_fonts, so the text keeps its
    size beside the panels), PDF (QPdfWriter: the window
    recorded in a QPicture first and the picture played into the PDF, since Qt paints widgets into
    a pixmap when they are rendered straight into a printer-type device), or EPS (the PDF
    converted by Ghostscript, which must be installed; without it a PDF is written beside the
    requested name and the reason printed: Illustrator opens the PDF and SVG directly with every
    object editable).
    """
    rect = fig._export_rect()
    W, H = float(rect.width()), float(rect.height())
    ext = os.path.splitext(path)[1].lower()
    if ext == '.svg':
        from pyqtgraph.Qt import QtSvg
        gen = QtSvg.QSvgGenerator()
        gen.setFileName(path)
        k = float(WINDOW_IMAGE_DPI) / DPI          # the window images' pixels per logical pixel (window_image)
        gen.setSize(QtCore.QSize(int(round(W * k)), int(round(H * k))))
        gen.setViewBox(QtCore.QRectF(0, 0, W, H))
        gen.setResolution(int(WINDOW_IMAGE_DPI))
        gen.setTitle(fig.manager.get_window_title() if fig.manager is not None else 'figure')
        with _pixel_fonts(fig):
            p = QtGui.QPainter(gen)
            try:
                _render_scene(fig, p)
            finally:
                p.end()
        return path
    pic = QtGui.QPicture()
    p = QtGui.QPainter(pic)
    try:
        _render_scene(fig, p)
    finally:
        p.end()
    pdf = path if ext == '.pdf' else os.path.splitext(path)[0] + '.pdf'
    w = QtGui.QPdfWriter(pdf)
    w.setResolution(int(DPI))
    w.setPageSize(QtGui.QPageSize(QtCore.QSizeF(W / DPI * 72.0, H / DPI * 72.0), QtGui.QPageSize.Unit.Point))
    w.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    p = QtGui.QPainter(w)
    try:
        sx = w.width() / W
        p.scale(sx, sx)
        p.drawPicture(QtCore.QPointF(0, 0), pic)
    finally:
        p.end()
    if ext == '.pdf':
        return path
    gs = _ghostscript()
    if gs is None:
        print(f"EPS needs Ghostscript (gswin64c / gs on the PATH), which was not found; the figure was "
              f"written as {pdf} instead (Illustrator opens PDF and SVG with every object editable).")
        return pdf
    import subprocess
    subprocess.run([gs, '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=eps2write',
                    f'-sOutputFile={path}', pdf], check=True)
    try:
        os.remove(pdf)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------- #
#  pyplot functions                                                        #
# ---------------------------------------------------------------------- #

_RUNNING = [False]
_INTERACTIVE = [False]      # matplotlib's interactive mode (ion / ioff): show() returns at once while it is on


def _app():
    """The QApplication, made on the first call (pg.mkQApp) with its default font scaled by FONT_SCALE."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        # OpenCV's Linux wheels point Qt at their own (Qt5) platform plugins when cv2 is imported
        # anywhere in the process; PySide6 must use its own.
        p = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', '')
        if 'cv2' in p:
            os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        app = pg.mkQApp('VASP_for_SPM')
        f = app.font()
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() * FONT_SCALE)
            app.setFont(f)
    return app


def figure(num=None, figsize=None, dpi=None, fill_screen=False, **kw):
    """
    pyplot's figure: the Figure given, the open figure numbered `num`, or a new window of
    `figsize` inches (6.4 x 4.8); `dpi` is ignored (DPI); fill_screen=True opens a working
    window on the screen's available area (rule 11), `figsize` staying its design size.
    """
    if isinstance(num, Figure):
        return num
    if num is not None and num in _figures:
        return _figures[num]
    return Figure(figsize=figsize or (6.4, 4.8), num=num, fill_screen=fill_screen)


def subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False, squeeze=True, **kw):
    fig = figure(figsize=figsize)
    axs = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze,
                       gridspec_kw=kw.get('gridspec_kw'))
    return fig, axs


def gcf():
    if _figures:
        return _figures[max(_figures)]
    return figure()


def fignum_exists(num):
    """
    Whether a figure is open: for a Figure object, that this very object is still registered (an
    embedded one: not closed), never just its number, which a later figure reuses after a close;
    for a number, that some figure holds it.  The windows' is_open tests pass the object.
    """
    if isinstance(num, Figure):
        if num._embedded:
            return not num._closed
        return _figures.get(num.number) is num
    return num in _figures


def get_fignums():
    return sorted(_figures)


def close(fig=None):
    """
    pyplot's close: the current figure, 'all', a Figure, or a number; the window is closed and
    the figure unregistered (close_event fired).
    """
    if fig is None:
        targets = [gcf()] if _figures else []
    elif fig == 'all':
        targets = list(_figures.values())
    elif isinstance(fig, Figure):
        targets = [fig]
    else:
        targets = [_figures[fig]] if fig in _figures else []
    for f in targets:
        try:
            f.manager.window.close()
        except Exception:
            pass
        f._on_close()


def _ipython_gui():
    """True when an IPython kernel (Spyder, Jupyter qtconsole) runs a Qt event loop or one could be turned on."""
    try:
        ip = get_ipython()                       # noqa: F821 (defined inside IPython)
    except NameError:
        return False
    if ip is None:
        return False
    loop = getattr(ip, 'active_eventloop', None)
    if loop and 'qt' in str(loop):
        return True
    try:
        ip.enable_gui('qt')
        return True
    except Exception:
        return False


def show(block=None):
    """
    Show every figure not yet shown.  Outside IPython the first call runs
    the Qt event loop until every window is closed (matplotlib's blocking
    show); inside a running loop, or in an IPython console with its Qt
    event loop (Spyder), it returns at once.
    """
    app = _app()
    for f in list(_figures.values()):
        if not f._shown:
            f.manager.show()
            f._relayout()
            app.processEvents()
            f.fit_window()                 # the window sized to its panels (rule 11)
    if block is False or (block is None and _INTERACTIVE[0]):
        # block=False, or matplotlib's interactive mode (ion): the windows are shown and show returns.
        app.processEvents()
        return
    if _RUNNING[0] or QtCore.QThread.currentThread().loopLevel() > 0:
        return
    if block is None and _ipython_gui():
        return
    if not _figures and not any(w.isVisible() for w in _figures_main_windows):
        return
    _RUNNING[0] = True
    try:
        (app.exec() if hasattr(app, 'exec') else app.exec_())
    finally:
        _RUNNING[0] = False


def pause(interval):
    app = _app()
    show(block=False)
    deadline = QtCore.QDeadlineTimer(int(interval * 1000))
    while not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)


def draw():
    pass


def ion():
    """matplotlib's interactive mode on: show() returns right after showing the windows (a script holds the process with ioff() and show())."""
    _INTERACTIVE[0] = True


def ioff():
    """matplotlib's interactive mode off: show() runs the event loop until every window is closed."""
    _INTERACTIVE[0] = False


def isinteractive():
    return _INTERACTIVE[0]


class _Cycler:
    def __init__(self, colors):
        self._colors = list(colors)

    def by_key(self):
        return {'color': list(self._colors)}


# The entries of matplotlib's rcParams the package reads ('axes.prop_cycle' for a colour per curve).
rcParams = {'axes.prop_cycle': _Cycler(TAB10), 'figure.dpi': DPI, 'font.size': 10.0, 'lines.markersize': 6.0,
            'lines.linewidth': 1.5, 'figure.figsize': (6.4, 4.8), 'savefig.dpi': 300}

# The main windows that embed qtplot figures (a QMainWindow appended by its owner): kept
# referenced, and show() blocks while one of them is visible.
_figures_main_windows = []
