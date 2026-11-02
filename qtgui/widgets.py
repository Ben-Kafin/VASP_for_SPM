"""
The control tables and the canvas interaction every window of the suite shares.

  - SPECS: one specification per control name (its kind, its labels, its font size, a box's
    width and limits), LABELS (a name's label text where it differs from the name), SHORT_LABELS
    and short_form (the short labels a window taller than wide shows), so that a control of a
    given name is the same in every window.  The controls themselves are built from these
    tables in controls.py (controls.make and the composites), the one control set of every
    window.
  - The controls' fonts (_font, indicator_css, _refont_widgets).
  - HoverLabels: the hover label of a panel, the one replacement of the mplcursors cursors the
    old windows made (the matcher's spread and decomposition plots, the band structure, the
    simulator's orbital curves, the DOS plotter).
  - figure_is_open.
A window owns its data and state, draws through qtplot, and receives events from here and from
controls.py.

automatic layout: the one rule for every window.  No control is placed by hand: every control is
one Qt widget in its window's Qt layout (controls.py, qtplot.Figure.set_layout), so nothing here
holds a position.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from qtgui import qtplot as plt
from qtgui.qtplot import QtWidgets, QtGui, qfont


def figure_is_open(fig):
    """True while `fig` itself is still open (a closed figure's number may be reused by a later figure)."""
    try:
        return plt.fignum_exists(fig)
    except Exception:
        return False


# ---------------------------------------------------------------------- #
#  Specifications: name -> kind, labels, font size, width, limits          #
# ---------------------------------------------------------------------- #
# Kinds: 'check' (one toggle row), 'checks' (a box of toggle rows), 'radio', 'button', 'text'
# (a text or number box), 'value' (a number box with limits, decimals), 'count' (a whole-number
# box with limits), 'slider' (the one slider kept: scrubbing a list).  The layout sizes every
# control to its text (automatic layout: the one rule for every window); of the sizes only a
# box's width (w, inches) is read (controls.Text).

def _spec(kind, w=1.0, h=0.3, fs=8, labels=None, **extra):
    d = {'kind': kind, 'w': float(w), 'h': float(h), 'fs': fs}
    if labels is not None:
        d['labels'] = list(labels)
    d.update(extra)
    return d


SPECS = {
    # --- toggles (one row) ---
    'Orbital': _spec('check', 0.9, 0.3),                            # Band_structure_plot: colour by orbital (on) or by element (off)
    'Metal curve': _spec('check', 1.2, 0.3),                        # decomp_window: the metal curve drawn
    'Segmentation': _spec('check', 1.2, 0.3),                       # spread_collapse: under Horizontal
    'Norm': _spec('check', 0.7, 0.3),                               # state_decomposition_plot: per-curve normalization
    # --- toggle boxes (several rows) ---
    'sim toggles': _spec('checks', 1.2, 1.4, labels=['Atoms', 'Decay', '0-clamp', 'Norm', 'Mag', 'Cell', 'Extend']),   # the simulator ('0-clamp' indented under 'Decay')
    'broadening': _spec('checks', 0.6, 0.6, labels=None),           # the simulator: one row per atom type (labels per file)
    'collapse': _spec('checks', 1.1, 0.9, labels=['Vertical', 'Horizontal', 'Amplitude']),   # decomp_window; spread_collapse adds 'Curve' (labels per window)
    'spread view': _spec('checks', 1.0, 0.6, labels=['Curve', 'Amplitude']),               # spread_plot, state_decomposition_plot
    'topo view': _spec('checks', 1.0, 0.6, labels=['Amplitude', 'Norm']),                  # topo_decomp_window
    # --- radios ---
    'mode': _spec('radio', 1.1, 1.0, labels=['Single Point', 'Line', 'Path', 'Map'], title=True),   # the simulator's view
    # --- buttons ---
    'RUN': _spec('button', 0.6, 0.4, fs=9),                         # the simulator: RUN / STOP
    'Rerun': _spec('button', 0.6, 0.3),                             # the simulator: apply a tip height
    'Crop': _spec('button', 0.6, 0.26, fs=7),                       # the simulator's placement topo: the view to the cursors' ranges
    'Uncrop': _spec('button', 0.7, 0.26, fs=7),                     # ... back to the full extent
    'Save data': _spec('button', 1.0, 0.36),
    'topo': _spec('button', 0.5, 0.36),                             # the simulator wrapper's 'Send to decomp' box
    'ldos': _spec('button', 0.5, 0.36),
    # --- whole-number boxes (the counts that were sliders) ---
    'Cells': _spec('count', 0.4, 0.24, vmin=0, vmax=4),            # the simulator: periodic images drawn each way
    'Markers': _spec('count', 0.4, 0.24, vmin=1, vmax=10),
    'Path Pts': _spec('count', 0.4, 0.24, vmin=2, vmax=12),
    'E Pts': _spec('count', 0.4, 0.24, vmin=1, vmax=20),
    # --- number boxes (the values that were sliders) ---
    'E Min': _spec('value', 0.6, 0.24, vmin=-5.0, vmax=5.0, decimals=3),       # the simulator's energy window (eV)
    'E Max': _spec('value', 0.6, 0.24, vmin=-5.0, vmax=5.0, decimals=3),
    'Topo h': _spec('value', 0.6, 0.24, vmin=0.0, vmax=None, decimals=3),      # tip heights (Å); the window passes vmax
    'LDOS h': _spec('value', 0.6, 0.24, vmin=0.0, vmax=None, decimals=3),
    'σ broad': _spec('value', 0.5, 0.22, fs=7, vmin=0.0, vmax=1.0, decimals=2),   # the simulator: extra broadening per atom type (eV), labelled 'σ <type>'
    'σ (eV)': _spec('value', 0.6, 0.24, decimals=3),                           # SigmaBox: the window passes the limits
    'γ': _spec('text', 0.45, 0.24, fs=7),                                      # GammaBox (plane slice)
    'LDOS γ': _spec('text', 0.45, 0.24, fs=7),                                 # GammaBox (the simulator's heatmaps)
    'x lo': _spec('text', 0.55, 0.22, fs=7),                                   # the simulator's crop cursors: the boxes riding with them,
    'x hi': _spec('text', 0.55, 0.22, fs=7),                                   # ... the x ones along the placement topo's top edge,
    'y lo': _spec('text', 0.55, 0.22, fs=7),                                   # ... the y ones along its right edge (Å, two decimals)
    'y hi': _spec('text', 0.55, 0.22, fs=7),
    'area ≥': _spec('text', 0.7, 0.24),                                        # decomp_window: the MO area threshold
    # --- sliders ---
    'window start': _spec('slider', 3.0, 0.2, fs=8, vmin=0, vmax=1),           # forces.py: scrubbing the ionic steps (the one slider kept)
}

# Widget names whose label text differs from the name (every other widget is labeled with its name).
LABELS = {'sim toggles': None, 'broadening': None, 'collapse': None, 'spread view': None, 'topo view': None,
          'mode': 'mode', 'σ broad': 'σ', 'x lo': '', 'x hi': '', 'y lo': '', 'y hi': ''}

# The short form of every label that has one, used when a window is taller than it is wide (the
# figure's portrait flag, qtplot.Figure.set_portrait): a control whose label is a key here shows
# the value instead, and the full label when the window is wider than tall again.  Keys are the
# label texts as shown (LABELS applied), stripped.  A label not listed keeps its text in both
# modes.  One table for every window's controls (controls.Control.set_short).
SHORT_LABELS = {
    # toggles and boxes
    'Vertical': 'Vert', 'Horizontal': 'Horiz', 'Amplitude': 'Amp', 'Segmentation': 'Seg', 'Metal curve': 'Metal',
    'Orbital': 'Orb', 'broadening': 'broad',
    # radio options
    'Single Point': 'Point',
    # buttons
    'Save data': 'Save', 'Send to decomp': 'Send', 'Rerun': 'Re',
    # text-box and slider labels
    'Path Pts': 'Pts', 'Markers': 'mk', 'E Pts': 'E pts', 'E Min': 'Emin', 'E Max': 'Emax', 'LDOS γ': 'γ',
    'Topo h': 'Topo', 'LDOS h': 'LDOS', 'σ (eV)': 'σ', 'area ≥': '≥', 'window start': 'start',
}


def short_form(text):
    """The short form of a label text (SHORT_LABELS), or the text itself when it has none."""
    t = str(text)
    return SHORT_LABELS.get(t.strip(), t)


# ---------------------------------------------------------------------- #
#  Qt helpers                                                              #
# ---------------------------------------------------------------------- #

WIDGET_FONT_SCALE = 1.1      # the controls' text: a little larger than matplotlib's, smaller than the plots'


def _font(fs, zoom=1.0):
    return qfont(fs * WIDGET_FONT_SCALE / plt.FONT_SCALE, zoom=zoom)


def indicator_css(zoom):
    """Check-box and radio indicators as large as the text around them."""
    n = max(13, int(round(13 * zoom)))
    return (f"QCheckBox::indicator, QRadioButton::indicator {{ width: {n}px; height: {n}px; }}"
            if zoom > 1.15 else "")


def _refont_widgets(widgets_fs, zoom):
    """(Qt widget, font size) pairs at a figure's zoom: the widget and every child widget."""
    css = indicator_css(zoom)
    for qw, fs in widgets_fs:
        f = _font(fs, zoom)
        for w in [qw] + qw.findChildren(QtWidgets.QWidget):
            w.setFont(f)
            if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                base = w.property('base_css') or ''
                w.setStyleSheet(base + css)


# ---------------------------------------------------------------------- #
#  The hover label                                                         #
# ---------------------------------------------------------------------- #

class HoverLabels:
    """
    The hover label of one panel: the one function behind every hover cursor the old windows
    made with mplcursors (mplcursors.cursor(artists, hover=True), its HoverMode.Persistent:
    the label of the artist under the pointer, kept after the pointer leaves it until another
    artist is hovered).  The label's text comes from `texts`: per artist a string, or a function
    of the index of the point (a curve's sample, a scatter marker, a segment of a collection)
    hit; an artist without one shows its label (artist.get_label(), the simulator's and the DOS
    plotter's orbital curves).

    Arming (the windows' click rule): with `arm_by_click` (the default) the labels start off;
    a left click on the panel turns them on and labels the artist under the click at once, a
    right click removes the label and turns them off.  `enabled=True` starts them on (the
    simulator's orbital curves, which the old window never armed).  One HoverLabels per axes:
    a new one made for the same axes replaces the old (a window that redraws its curves makes a
    new one); remove() takes it off.  The figure holds one motion and one click handler for all
    its panels (_install), so a window never wires a hover itself.

    The label is a qtplot Text in axes coordinates at the hit point (a pale box offset from it,
    anchored toward the panel's interior): an axes text sits in the window's scene, not inside
    the panel's view box, so a tall label is never clipped by the panel and may overlap its
    neighbours, as the old annotations did.  The same item is moved and retexted on every hit
    (created once, then updated; made anew when its anchor side changes).  It keeps its place in
    the panel, not in the data, until the next hit.
    """

    PICK_PX = 6.0          # the pick radius (pixels) around a curve or marker without a picker of its own

    def __init__(self, ax, artists=(), texts=None, enabled=False, arm_by_click=True, mono=False):
        self.ax = ax
        self.fig = ax.figure
        self.artists = []
        self.texts = {}
        self.enabled = bool(enabled)
        self.arm_by_click = bool(arm_by_click)
        self.mono = bool(mono)           # a monospace label (a text laid out in columns)
        self._label = None
        self._current = None
        texts = texts or {}
        for a in artists:
            self.add(a, texts.get(a) if isinstance(texts, dict) else None)
        reg = self.fig.__dict__.setdefault('_hover_labels', {})
        old = reg.get(ax)
        if old is not None and old is not self:
            old.remove()
        reg[ax] = self
        self._install(self.fig)

    # --- the registry and the figure's handlers ---

    @staticmethod
    def _install(fig):
        if getattr(fig, '_hover_cids', None) is not None:
            return
        fig._hover_cids = (fig.canvas.mpl_connect('motion_notify_event', HoverLabels._on_motion),
                           fig.canvas.mpl_connect('button_press_event', HoverLabels._on_click))

    @staticmethod
    def of(ax):
        """The HoverLabels of an axes, or None."""
        return getattr(ax.figure, '_hover_labels', {}).get(ax)

    @staticmethod
    def prune(fig):
        """The entries of axes no longer on the figure dropped (a window that rebuilds its panels calls it after the rebuild)."""
        reg = getattr(fig, '_hover_labels', None)
        if not reg:
            return
        live = {id(a) for a in fig.axes}
        for ax in [a for a in reg if id(a) not in live]:
            reg[ax].remove()

    @staticmethod
    def _on_motion(event):
        hl = event.inaxes and HoverLabels.of(event.inaxes)
        if hl and hl.enabled:
            hl._hover(event)

    @staticmethod
    def _on_click(event):
        hl = event.inaxes and HoverLabels.of(event.inaxes)
        if not hl or not hl.arm_by_click:
            return
        if event.button == 1:
            if not hl.enabled:
                hl.set_enabled(True)
            hl._hover(event)              # the artist under the click labelled at once
        elif event.button == 3:
            hl.clear()
            hl.set_enabled(False)

    # --- the artists and their texts ---

    def add(self, artist, text=None):
        """An artist to label: `text` a string, a function of the point index, or None for the artist's label."""
        if artist not in self.artists:
            self.artists.append(artist)
        if text is not None:
            self.texts[artist] = text

    def text_for(self, artist, index):
        t = self.texts.get(artist)
        if t is None:
            return str(artist.get_label())
        if callable(t):
            return str(t(index))
        return str(t)

    def set_enabled(self, flag):
        self.enabled = bool(flag)
        if not self.enabled:
            self.clear()

    # --- hits and the label ---

    def _hit(self, x, y):
        """The topmost visible artist under display pixel (x, y) and the index of its point, or None."""
        for a in reversed(self.artists):
            if not a.get_visible() or a.axes is not self.ax:
                continue
            picker = getattr(a, '_picker', None)
            if picker is None:
                a._picker = self.PICK_PX
                hit = a.contains_px(x, y)
                a._picker = None
            else:
                hit = a.contains_px(x, y)
            if hit:
                return a, getattr(a, '_hit_index', None)
        return None

    def _point(self, artist, index, event):
        """Where the label sits: the point hit (a curve's sample, a marker), else the pointer."""
        try:
            if index is not None:
                if hasattr(artist, 'get_offsets'):
                    xy = artist.get_offsets()
                    return float(xy[index, 0]), float(xy[index, 1])
                if hasattr(artist, 'get_xdata') and hasattr(artist, 'get_segments') is False:
                    xs, ys = artist.get_xdata(), artist.get_ydata()
                    return float(xs[index]), float(ys[index])
                if hasattr(artist, 'get_segments'):
                    sg = artist.get_segments()[index]
                    return float(np.mean(sg[:, 0])), float(np.mean(sg[:, 1]))
        except Exception:
            pass
        return float(event.xdata), float(event.ydata)

    def _fractions(self, x, y):
        """Data point (x, y) as fractions of the panel's box (from its left and its bottom)."""
        x0, y0, w, h = self.ax._px_rect()
        px, py_up = self.ax.transData.transform([[x, y]])[0]
        H = self.fig._canvas_size()[1]
        return (px - x0) / (w or 1.0), (y0 + h - (H - py_up)) / (h or 1.0)

    @staticmethod
    def _anchor(fx, fy):
        """The label's alignment at panel fractions (fx, fy): ('right', ...) in the right 40 % of the panel, (..., 'top') in its upper 40 %."""
        return ('right' if fx > 0.6 else 'left'), ('top' if fy > 0.6 else 'bottom')

    def _hover(self, event):
        if event.xdata is None:
            return
        hit = self._hit(event.x, event.y)
        if hit is None:
            return                         # persistent: the last label stays until another artist is hit
        artist, index = hit
        if self._current == (artist, index) and self._label is not None and self._label in self.ax._children:
            return
        self._current = (artist, index)
        x, y = self._point(artist, index, event)
        text = self.text_for(artist, index)
        try:
            fx, fy = self._fractions(x, y)
        except Exception:
            return
        # The label grows toward the panel's interior: to the left of a point in the right part of
        # the panel, downward from a point in its upper part.
        ha, va = self._anchor(fx, fy)
        if self._label is not None and self._label in self.ax._children and (self._label._ha, self._label._va) != (ha, va):
            self.clear()
        if self._label is None or self._label not in self.ax._children:
            self._label = self.ax.text(fx, fy, text, transform=self.ax.transAxes, fontsize=8, ha=ha, va=va, color='k',
                                       zorder=60, offset=(-10.0 if ha == 'right' else 10.0, -10.0 if va == 'top' else 10.0),
                                       family='monospace' if self.mono else None,
                                       bbox=dict(facecolor='#ffffe0', edgecolor='#808080', alpha=0.95, lw=0.8))
        else:
            self._label.set_text(text)
            self._label.set_position((fx, fy))
        self._label.set_visible(True)
        self.fig.canvas.draw_idle()

    def clear(self):
        """The label removed (a right click; the hover stays armed or not as it was)."""
        self._current = None
        if self._label is not None:
            try:
                self._label.remove()
            except Exception:
                pass
            self._label = None

    def remove(self):
        """This HoverLabels taken off its axes (its label removed, its registry entry dropped)."""
        self.clear()
        reg = getattr(self.fig, '_hover_labels', None)
        if reg is not None and reg.get(self.ax) is self:
            del reg[self.ax]


if __name__ == '__main__':
    # Every control of the specification built once by name (controls.make), each in a box frame
    # titled with its name, in one wrapping row of the window's layout (automatic layout: the one
    # rule for every window), so that sizes and labels can be checked at a glance.
    from qtgui import controls
    fig = plt.figure(figsize=(16, 9))
    flow = controls.FlowLayout()
    keep = []
    for name in SPECS:
        kw = {'labels': ['Au', 'N', 'C', 'H']} if name == 'broadening' else {}
        if SPECS[name]['kind'] == 'slider':
            kw.update(vmin=0, vmax=10)
        if name in ('Topo h', 'LDOS h'):
            kw.update(vmax=10.0)
        if name == 'σ (eV)':
            kw.update(vmin=0.01, vmax=1.0)
        c = controls.make(fig, name, **kw)
        keep.append(c)
        flow.addWidget(controls.box(c, title=name))
    root = QtWidgets.QVBoxLayout()
    root.addLayout(flow)
    root.addStretch(1)
    fig.set_layout(root)
    plt.show()
