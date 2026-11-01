"""
The one set of controls of every window of the suite, so that a shared feature (a σ box, a γ box,
a number box, the step buttons, ...) works the same everywhere: one class per kind (Check,
Checks, Radio, Button, Text, Value, Count, Slider, TextPanel, Label) with one interface (value,
set (never fires), on_change, show, enable, set_short), and the composite controllers several
windows share (SigmaBox, GammaBox, StepButtons, RunControls).  widgets.SPECS names every control
(its kind, its labels, its font size, a box's width and limits) and widgets.SHORT_LABELS its short
label.

automatic layout: the one rule for every window (nothing placed by hand).  Every control is one Qt
widget, `qt`, that the window puts in its Qt layout (qtplot.Figure.set_layout); a control never
knows a position.  Groups of controls sit in box frames (box), the same frame in every window.  A
panel that holds only text scrolls (TextPanel); a window never does.

Text size: a figure window's controls follow its text zoom (qtplot: the window's size relative to
its design size, reduced to fit the window, Figure._fit_controls); a control made without a figure
(fig=None) takes its window's font.  When the window is taller than wide every label with a short
form shows it (set_short).

Number boxes (Text, Value, Count): select one or more digits and press Up or Down: the selected
digits go up or down by one in the place of the last selected digit, carrying into the digits to
the left as arithmetic does, and the same places stay selected; with nothing selected the digit
just left of the cursor steps.  Every number in a box steps on its own (a box holding "3,3"); a
Value or a Count stops at its limits.  Each press applies the new value at once, as Enter does
(rule 8 of the guide: number boxes instead of sliders; the one slider kept scrubs a list).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import re
import weakref
from decimal import Decimal, InvalidOperation
import numpy as np
from qtgui import qtplot as plt
from qtgui.qtplot import QtCore, QtGui, QtWidgets, qfont, qcolor
from qtgui import widgets
from qtgui.widgets import SPECS, LABELS, short_form

Qt = QtCore.Qt

# The box frame of every group of controls in every window (the light frame of the figure windows).
FRAME_CSS = "QFrame#box { background: #f0f0f0; border: 1px solid #c8c8c8; border-radius: 3px; }"
COLOR_FRAME_CSS = "QFrame#count { border: 2px solid %s; border-radius: 3px; }"


def _font(fs, zoom=1.0):
    """A control's font: its specification's points, the controls' scale, at a zoom (widgets._font)."""
    return widgets._font(fs, zoom)


def _label_of(name):
    """The label text of a control name (widgets.LABELS; the name itself without an entry), stripped."""
    lab = LABELS.get(name, name)
    return name if lab is None else str(lab).strip()


def _qw(item):
    """The Qt widget of a control, a composite (its `qt`) or a Qt widget."""
    return item.qt if hasattr(item, 'qt') else item


def _fig_of(item):
    """The figure a control or composite belongs to (its `fig`, or its first control's), else None."""
    for cand in (item, getattr(item, 'box', None), getattr(item, 'left', None)):
        fig = getattr(cand, 'fig', None)
        if fig is not None:
            return fig
    return None


# ---------------------------------------------------------------------- #
#  Layout helpers: frames, rows, columns                                    #
# ---------------------------------------------------------------------- #

def box(*items, title=None, orient='h', spacing=4, margins=(4, 2, 4, 2), stretch=False):
    """
    A box frame holding `items` (controls, composites, Qt widgets or Qt layouts) in a row
    (orient='h') or a column ('v'), with a small title on top when given (automatic layout: the
    frame and its contents are placed by Qt).  The title is a Label control of the items' figure,
    so it follows the window's control zoom and shows its short form in a portrait window like
    every other label; without a figure among the items it is a plain Qt label.  A trailing
    stretch keeps the items together.
    """
    frame = QtWidgets.QFrame()
    frame.setObjectName('box')
    frame.setStyleSheet(FRAME_CSS)
    outer = QtWidgets.QVBoxLayout(frame)
    outer.setContentsMargins(*margins)
    outer.setSpacing(2)
    if title:
        fig = next((f for f in (_fig_of(it) for it in items) if f is not None), None)
        if fig is not None:
            lab = Label(fig, 'box title', text=title, fs=8)
            t = lab.qt
            frame._title_control = lab
        else:
            t = QtWidgets.QLabel(title)
        t.setObjectName('box_title')
        frame._title_label = t
        outer.addWidget(t)
    inner = QtWidgets.QHBoxLayout() if orient == 'h' else QtWidgets.QVBoxLayout()
    inner.setContentsMargins(0, 0, 0, 0)
    inner.setSpacing(spacing)
    for it in items:
        if isinstance(it, QtWidgets.QLayout):
            inner.addLayout(it)
        elif it is not None:
            inner.addWidget(_qw(it))
    if stretch:
        inner.addStretch(1)
    outer.addLayout(inner)
    return frame


def row(*items, spacing=4, stretch=False, align=None):
    """A plain row (no frame) of items: a QHBoxLayout (automatic layout)."""
    lay = QtWidgets.QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for it in items:
        if it == 'stretch':
            lay.addStretch(1)
        elif isinstance(it, QtWidgets.QLayout):
            lay.addLayout(it)
        elif it is not None:
            if align is not None:
                lay.addWidget(_qw(it), 0, align)
            else:
                lay.addWidget(_qw(it))
    if stretch:
        lay.addStretch(1)
    return lay


def column(*items, spacing=4, stretch=False, align=None):
    """A plain column (no frame) of items: a QVBoxLayout (automatic layout)."""
    lay = QtWidgets.QVBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for it in items:
        if it == 'stretch':
            lay.addStretch(1)
        elif isinstance(it, QtWidgets.QLayout):
            lay.addLayout(it)
        elif it is not None:
            if align is not None:
                lay.addWidget(_qw(it), 0, align)
            else:
                lay.addWidget(_qw(it))
    if stretch:
        lay.addStretch(1)
    return lay


class FlowLayout(QtWidgets.QLayout):
    """
    A row of items that wraps onto more rows when the window is too narrow for one (automatic
    layout: the one rule for every window; a window never scrolls and nothing overlaps): items
    keep their preferred sizes, left to right, each row's items aligned at its bottom; with
    `push_last` the last item sits at the right end of its row (a window's right-hand group).
    Its height follows its width (heightForWidth).
    """

    def __init__(self, hspacing=6, vspacing=4, push_last=False):
        super().__init__()
        self._items = []
        self._hs = int(hspacing)
        self._vs = int(vspacing)
        self._push_last = bool(push_last)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def addWidget(self, w, *args):
        super().addWidget(_qw(w))

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        m = self.contentsMargins()
        w = h = 0
        for it in self._visible():
            s = it.sizeHint()
            w += s.width() + (self._hs if w else 0)
            h = max(h, s.height())
        return QtCore.QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._visible():
            size = size.expandedTo(it.sizeHint())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _visible(self):
        return [it for it in self._items if not (it.widget() is not None and it.widget().isHidden())]

    def _arrange(self, rect, apply):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        items = self._visible()
        rows, cur, cur_w = [], [], 0
        for it in items:
            w = it.sizeHint().width()
            if cur and cur_w + self._hs + w > r.width():
                rows.append(cur)
                cur, cur_w = [], 0
            cur_w += (self._hs if cur else 0) + w
            cur.append(it)
        if cur:
            rows.append(cur)
        y = r.y()
        for k, rw in enumerate(rows):
            rh = max(it.sizeHint().height() for it in rw)
            if apply:
                x = r.x()
                last = rw[-1]
                for it in rw:
                    s = it.sizeHint()
                    if it is last and self._push_last and it is items[-1] and len(rw) > 1:
                        x = r.x() + r.width() - s.width()
                    it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y + rh - s.height()), s))
                    x += s.width() + self._hs
            y += rh + (self._vs if k < len(rows) - 1 else 0)
        return y - r.y() + m.top() + m.bottom()


class UnderPanel(QtWidgets.QWidget):
    """
    Items kept centred under a panel's box (automatic layout: the one rule for every window): put
    in the same layout column as the panel's cell, its margins follow the room the panel's tick
    axes take in the cell (Axes.on_placed), so the items centre under the box itself, not under
    the cell with its axis labels.
    """

    def __init__(self, ax, *items, spacing=3, follow=True):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(spacing)
        lay.addStretch(1)
        for it in items:
            lay.addWidget(_qw(it))
        lay.addStretch(1)
        self._ax = ax
        if follow:
            ax.on_placed(self._follow)         # follow=False: centred on the whole cell (a narrow cell, whose box is centred in it)

    def _follow(self):
        ax = self._ax
        cell = getattr(ax, '_cell', None)
        if cell is None:
            return
        cx, cy, cw, ch = cell.rect_px()
        x0, y0, w, h = ax._px_rect()
        left = max(int(round(x0 - cx)), 0)
        right = max(int(round(cx + cw - (x0 + w))), 0)
        m = self.layout().contentsMargins()
        if (m.left(), m.right()) != (left, right):
            self.layout().setContentsMargins(left, 0, right, 0)


class AbovePanel(UnderPanel):
    """Items kept centred over a panel's box (a tip-height box over its topography): UnderPanel's centring, placed above the panel in its column (PanelWithBox)."""


class PanelWithBox(QtWidgets.QWidget):
    """
    A shaped panel with a control box over it, one item of a qtplot.PanelRow or an OrientedPanes
    (rule 11 of the guide): the box centred over the panel's box (AbovePanel), the panel's cell
    below it; height_for_width and width_for_height are the cell's plus the box's height, so the
    pair is sized as one shaped item and the box never sits between panels.
    """

    def __init__(self, ax, cell, *items, spacing=3):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(spacing)
        narrow = getattr(cell, 'inner_w', None) is not None
        self.above = AbovePanel(ax, *items, follow=not narrow)
        lay.addWidget(self.above, 0)
        lay.addWidget(cell, 1)
        self.cell = cell
        self._sp = int(spacing)
        if narrow:
            # Over a narrow cell (a tip-height stripe): the pair is as wide as the wider of the box and
            # the cell, a fixed-width item of its row like the cell itself (PanelRow._fixed).
            pol = QtWidgets.QSizePolicy.Policy
            self.setSizePolicy(pol.Fixed, pol.Expanding)

    def _box_h(self):
        return 0.0 if self.above.isHidden() else float(self.above.sizeHint().height()) + self._sp

    def height_for_width(self, w):
        h = self.cell.height_for_width(w)
        return None if h is None else h + self._box_h()

    def width_for_height(self, h):
        return self.cell.width_for_height(max(float(h) - self._box_h(), 1.0))

    def hasHeightForWidth(self):
        return self.cell.hasHeightForWidth()

    def heightForWidth(self, w):
        h = self.height_for_width(w)
        return -1 if h is None else int(round(h))


class OrientedPanes(QtWidgets.QWidget):
    """
    Two panes arranged by the main panel's shape (rule 11 of the guide: the panel the user works
    in takes the largest size its ratio allows): `main` is a shaped widget (a PanelCell, a
    PanelWithBox, a qtplot.RowWidget), `rest` the widget holding the other panels.  Of the two
    arrangements, the main pane at the left with the window's full height and the main pane
    across the top with its full width, the one that gives the main panel the larger area is
    taken; the main pane takes what its ratio needs to fill the window's height (or width), never
    more than `main_share` of the window along the split (0.6: the other panels keep two fifths);
    the rest fills what is left.  Chosen again on every resize and whenever the main panel's
    shape changes (its cell's updateGeometry reaches this widget as a layout request);
    `hysteresis` keeps the arrangement unless the other is that much larger.  `orientation` is
    'h' (main at the left) or 'v' (main on top).  When the cap binds (a window too narrow or too
    low for the main panel's ratio), the main panel is as large as the cap allows and the room
    it cannot use stays beside it; that room is not an empty band of a row (the render check
    skips the main pane).
    """

    def __init__(self, main, rest, main_share=0.6, spacing=8, hysteresis=0.1):
        super().__init__()
        self.main, self.rest = main, rest
        for w in (main, rest):
            w.setParent(self)
            w.show()
        main._pane_main = True                 # the render check's mark: this panel's room follows the pane rule, not a row's
        self.main_share, self._sp, self._hys = float(main_share), int(spacing), float(hysteresis)
        self.orientation = None
        pol = QtWidgets.QSizePolicy.Policy
        self.setSizePolicy(pol.Expanding, pol.Expanding)
        self._cb = None
        self._pending = False
        main.installEventFilter(self)          # the main panel's shape change (its cell's updateGeometry) reaches it as a layout request

    def eventFilter(self, obj, ev):
        if obj is self.main and ev.type() == QtCore.QEvent.Type.LayoutRequest and not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self._arrange_later)
        return False

    def _arrange_later(self):
        self._pending = False
        self._arrange()

    def on_arranged(self, callback):
        """callback(orientation) after every arrangement (a window that logs or adapts to it)."""
        self._cb = callback

    def _shape(self):
        hfw = getattr(self.main, 'height_for_width', None)
        wfh = getattr(self.main, 'width_for_height', None)
        if hfw is None or wfh is None:
            return None
        try:
            return (hfw, wfh) if hfw(100.0) is not None else None
        except Exception:
            return None

    def _options(self, W, H):
        """[(area, orientation, main width, main height)] of the two arrangements at W x H."""
        shape = self._shape()
        if shape is None:
            w = self.main_share * W
            return [(w * H, 'h', w, H)]
        hfw, wfh = shape
        w = min(self.main_share * W, wfh(H) or W)
        h = min(H, hfw(w) or H)
        w = min(w, wfh(h) or w)
        h2 = min(self.main_share * H, hfw(W) or H)
        w2 = min(W, wfh(h2) or W)
        h2 = min(h2, hfw(w2) or h2)
        return [(w * h, 'h', w, h), (w2 * h2, 'v', w2, h2)]

    def _choose(self, W, H):
        """(area, orientation, main width, main height) for the widget's size W x H: the larger area, the current arrangement kept within the hysteresis."""
        opts = self._options(W - self._sp, H - self._sp)
        best = max(opts, key=lambda o: o[0])
        cur = next((o for o in opts if o[1] == self.orientation), None)
        if cur is not None and best[1] != cur[1] and best[0] <= cur[0] * (1.0 + self._hys):
            best = cur
        return best

    def fit_delta(self):
        """
        (dw, dh), each <= 0 px: how much narrower or shorter this widget could be with its main panel
        keeping its size (rule 11: the window follows the panel; Figure.fit_window applies it to the
        window).  With the main pane at the left ('h') the main panel spans the height the window
        gives, unless the share cap holds it shorter: the window can lose that height (the rest pane,
        free panels, follows).  With the main pane on top ('v') it spans the width, unless the cap
        holds it narrower: the window can lose that width.  (0, 0) when the main panel has no shape.
        """
        W, H = float(self.width()), float(self.height())
        if W <= 0 or H <= 0 or self._shape() is None:
            return (0, 0)
        area, orient, mw, mh = self._choose(W, H)
        if orient == 'h':
            return (0, min(0, int(round(mh - (H - self._sp)))))
        return (min(0, int(round(mw - (W - self._sp)))), 0)

    def _arrange(self):
        W, H = float(self.width()), float(self.height())
        if W <= 0 or H <= 0:
            return
        area, orient, mw, mh = self._choose(W, H)
        self.orientation = orient
        if orient == 'h':
            mw = int(round(mw))
            self.main.setGeometry(0, 0, mw, int(H))
            self.rest.setGeometry(mw + self._sp, 0, max(int(W) - mw - self._sp, 1), int(H))
        else:
            mh = int(round(mh))
            self.main.setGeometry(0, 0, int(W), mh)
            self.rest.setGeometry(0, mh + self._sp, int(W), max(int(H) - mh - self._sp, 1))
        if self._cb is not None:
            self._cb(orient)

    def rearrange(self):
        """The arrangement chosen again now (a window that changed its main panel's data)."""
        self._arrange()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._arrange()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._arrange()

    def event(self, ev):
        if ev.type() == QtCore.QEvent.Type.LayoutRequest:
            self._arrange()
        return super().event(ev)

    def minimumSizeHint(self):
        a, b = self.main.minimumSizeHint(), self.rest.minimumSizeHint()
        return QtCore.QSize(max(a.width(), b.width(), 100), max(a.height(), b.height(), 100))

    def sizeHint(self):
        return QtCore.QSize(900, 600)


class RideBand(QtWidgets.QWidget):
    """
    A band of the window's layout beside a panel whose items ride along it at positions the window
    sets from its data (the Line window's crop-cursor boxes on their cursors): the band takes its
    place from the layout (automatic layout: the one rule for every window), as thick as its
    largest item, so the items never reach another control; each item centres on the canvas
    coordinate given (set_pos: x for a horizontal band, y for a vertical one), kept inside the band.
    """

    def __init__(self, orient='h'):
        super().__init__()
        self.orient = orient
        self._items = {}                  # key -> [widget, canvas coordinate or None]
        pol = QtWidgets.QSizePolicy.Policy
        if orient == 'h':
            self.setSizePolicy(pol.Expanding, pol.Fixed)
        else:
            self.setSizePolicy(pol.Fixed, pol.Expanding)

    def add(self, key, item):
        w = _qw(item)
        w.setParent(self)
        w.show()
        self._items[key] = [w, None]
        self.updateGeometry()

    def sizeHint(self):
        hints = [it[0].sizeHint() for it in self._items.values()]
        if self.orient == 'h':
            return QtCore.QSize(0, max([h.height() for h in hints] + [0]))
        return QtCore.QSize(max([h.width() for h in hints] + [0]), 0)

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_pos(self, key, coord_px):
        """Centre item `key` at canvas coordinate `coord_px` along the band (x for 'h', y for 'v')."""
        self._items[key][1] = float(coord_px)
        self._apply(key)

    def _apply(self, key):
        w, c = self._items[key]
        if c is None:
            return
        s = w.sizeHint()
        vp = self.parentWidget()
        while vp is not None and vp.parentWidget() is not None and not isinstance(vp.parentWidget(), QtWidgets.QGraphicsView):
            vp = vp.parentWidget()
        origin = self.mapTo(vp, QtCore.QPoint(0, 0)) if vp is not None else QtCore.QPoint(0, 0)
        if self.orient == 'h':
            x = int(round(c - origin.x() - 0.5 * s.width()))
            x = max(0, min(x, self.width() - s.width()))
            w.setGeometry(x, 0, s.width(), self.height())
        else:
            y = int(round(c - origin.y() - 0.5 * s.height()))
            y = max(0, min(y, self.height() - s.height()))
            w.setGeometry(0, y, self.width(), s.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        for k in self._items:
            self._apply(k)


def widget_of(layout):
    """A plain QWidget holding a layout (to put a row or a column where a widget is needed)."""
    w = QtWidgets.QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


# ---------------------------------------------------------------------- #
#  The base                                                                #
# ---------------------------------------------------------------------- #

class _LabelProxy:
    """A control's label as matplotlib's Text (set_color, set_text, get_text, set_fontsize)."""

    def __init__(self, qwidget, setter=None):
        self._w = qwidget
        self._setter = setter
        self._color = None

    def set_color(self, c):
        col = qcolor(c).name()
        if col != self._color:
            self._color = col
            self._w.setStyleSheet(f"color: {col};")

    def set_text(self, s):
        if self._setter is not None:
            self._setter(str(s))
        else:
            self._w.setText(str(s))

    def get_text(self):
        return self._w.text()

    def set_fontsize(self, fs):
        self._w.setFont(_font(fs))


class _Anchor:
    """A control's place for the code that asks where it is (AveragePicker): its Qt widget."""

    def __init__(self, control):
        self._c = control

    def _qwidget(self):
        return self._c.qt

    def set_visible(self, flag):
        self._c.show(flag)

    def get_visible(self):
        return self._c._visible


class Control:
    """
    One control: its Qt widgets in one widget, `qt`, which the window's Qt layout places
    (automatic layout: the one rule for every window).  `fig` is the figure whose window holds it
    (its text zoom and short labels follow the figure), or None (its window sets them).
    The interface every kind shares: value, set (never fires), on_change(callback), show, enable,
    set_short, remove; `widget` is the main Qt widget and unknown attributes go to it.
    """

    _all = weakref.WeakSet()      # every control made (a window without a figure switches their short labels)

    def __init__(self, fig, name, label=None):
        self.fig = fig
        self.name = name
        self.spec = dict(SPECS.get(name, {'kind': None, 'w': 1.0, 'h': 0.3, 'fs': 8}))
        self.label = _label_of(name) if label is None else str(label)
        self._short = False
        self._cb = None
        self._quiet = False
        self._visible = True
        self._enabled = True
        self._zoom = None
        self.eventson = True
        self.qt = None
        self.widget = None
        Control._all.add(self)

    def _finish(self):
        if self.fig is not None:
            self.fig._register_control(self)

    # --- text size and short labels ---

    def _fs(self):
        return self.spec.get('fs', 8)

    def _set_zoom(self, z):
        """The control's text at the figure's control zoom z (a control without a figure takes its window's font)."""
        if self.fig is None or self.qt is None:
            return
        if self._zoom is not None and abs(z - self._zoom) < 1e-6:
            return
        self._zoom = z
        widgets._refont_widgets([(self.qt, self._fs())] + [(w, self._fs()) for w in getattr(self, '_font_also', [])], z)
        self._after_zoom(z)

    def _after_zoom(self, z):
        return None

    def shown(self, text):
        """The text a label shows: its short form in the short mode, else as it is."""
        return short_form(text) if self._short else str(text)

    def _apply_labels(self):
        return None

    def set_short(self, flag):
        flag = bool(flag)
        if flag != self._short:
            self._short = flag
            self._apply_labels()

    # --- the shared interface ---

    def show(self, flag):
        self._visible = bool(flag)
        self.qt.setVisible(self._visible)

    def enable(self, flag):
        """The control takes events (True) or ignores them (False)."""
        self._enabled = bool(flag)
        self.qt.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self._enabled)

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, *args):
        if self._cb is not None and not self._quiet and self.eventson:
            self._cb(*args)

    @property
    def ax(self):
        """The control's place (for AveragePicker's anchor): its Qt widget."""
        return _Anchor(self)

    def remove(self):
        if self.fig is not None:
            self.fig._unregister_control(self)
        if self.qt is not None:
            self.qt.hide()
            self.qt.deleteLater()

    def __getattr__(self, item):
        w = self.__dict__.get('widget')
        if w is None:
            raise AttributeError(item)
        return getattr(w, item)


# ---------------------------------------------------------------------- #
#  The kinds                                                               #
# ---------------------------------------------------------------------- #

class Check(Control):
    """A one-row toggle: value is its state; set(flag) never fires; callback(flag)."""

    def __init__(self, fig, name, value=False, label=None):
        super().__init__(fig, name, label)
        cb = QtWidgets.QCheckBox(self.label)
        cb.setChecked(bool(value))
        cb.toggled.connect(lambda flag: self._fire(bool(flag)))
        self.widget = self.qt = cb
        self._labels = [_LabelProxy(cb, self._set_label)]
        self._finish()

    def _set_label(self, text):
        self.label = str(text)
        self._apply_labels()

    def _apply_labels(self):
        self.widget.setText(self.shown(self.label))

    @property
    def labels(self):
        return self._labels

    @property
    def value(self):
        return bool(self.widget.isChecked())

    def get_status(self):
        return [self.value]

    def set(self, flag):
        if self.value != bool(flag):
            self._quiet = True
            blocked = self.widget.blockSignals(True)
            try:
                self.widget.setChecked(bool(flag))
            finally:
                self.widget.blockSignals(blocked)
                self._quiet = False

    def set_label_color(self, color):
        self._labels[0].set_color(color)


class Checks(Control):
    """
    A box frame of toggle rows: value is the list of states; set(i, flag) never fires;
    callback(label, states) after a click (the clicked row already flipped).  `indents` gives a
    left indent per row: pixels, or a text whose width (in the box's font) is the indent (the
    calibrate box's rows under 'calibrate').
    """

    def __init__(self, fig, name, values=None, labels=None, title=None, indents=None):
        super().__init__(fig, name)
        labs = list(labels if labels is not None else self.spec.get('labels') or [])
        self.spec['labels'] = labs
        vals = [False] * len(labs) if values is None else [bool(v) for v in values]
        self._title = title or ''
        frame = QtWidgets.QFrame()
        frame.setObjectName('box')
        frame.setStyleSheet(FRAME_CSS)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(1)
        self._title_w = None
        if self._title:
            self._title_w = QtWidgets.QLabel(self._title)
            lay.addWidget(self._title_w)
        self.boxes = []
        for k, (lab, v) in enumerate(zip(labs, vals)):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(v)
            cb.toggled.connect(lambda _f, lab=lab: self._fire(lab, self.value))
            ind = indents[k] if indents is not None and k < len(indents) else 0
            if isinstance(ind, str) and ind:
                # An indent given as text: a label holding it, so the indent follows the box's font.
                r = QtWidgets.QHBoxLayout()
                r.setContentsMargins(0, 0, 0, 0)
                r.setSpacing(0)
                pad = QtWidgets.QLabel(ind)
                pad.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                r.addWidget(pad)
                r.addWidget(cb)
                r.addStretch(1)
                lay.addLayout(r)
            elif ind:
                r = QtWidgets.QHBoxLayout()
                r.setContentsMargins(int(ind), 0, 0, 0)
                r.addWidget(cb)
                r.addStretch(1)
                lay.addLayout(r)
            else:
                lay.addWidget(cb)
            self.boxes.append(cb)
        self.widget = self.qt = frame
        self._label_proxies = [_LabelProxy(cb, lambda s, i=i: self._set_label(i, s)) for i, cb in enumerate(self.boxes)]
        self._finish()

    def _set_label(self, i, text):
        self.spec['labels'][i] = str(text)
        self._apply_labels()

    def _apply_labels(self):
        for cb, lab in zip(self.boxes, self.spec['labels']):
            cb.setText(self.shown(lab))
        if self._title_w is not None:
            self._title_w.setText(self.shown(self._title))

    @property
    def labels(self):
        return list(self.spec['labels'])

    @property
    def label_texts(self):
        return self._label_proxies

    def get_status(self):
        return self.value

    @property
    def value(self):
        return [bool(cb.isChecked()) for cb in self.boxes]

    def set(self, i, flag):
        cb = self.boxes[i]
        if cb.isChecked() != bool(flag):
            self._quiet = True
            blocked = cb.blockSignals(True)
            try:
                cb.setChecked(bool(flag))
            finally:
                cb.blockSignals(blocked)
                self._quiet = False

    def set_all(self, flags):
        for i, f in enumerate(flags):
            self.set(i, f)


class Radio(Control):
    """
    A box frame of radio rows: value is the selected label; set(label) never fires;
    callback(label).  `title` (or a specification with title=True: its label) sits on top.
    """

    def __init__(self, fig, name, value=None, labels=None, title=None):
        super().__init__(fig, name)
        labs = list(labels if labels is not None else self.spec.get('labels') or [])
        self.spec['labels'] = labs
        if title is None and self.spec.get('title') and str(self.label).strip():
            title = self.label
        self._title = title or ''
        frame = QtWidgets.QFrame()
        frame.setObjectName('box')
        frame.setStyleSheet(FRAME_CSS)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(1)
        self._title_w = None
        if self._title:
            self._title_w = QtWidgets.QLabel(self._title)
            lay.addWidget(self._title_w)
        self.group = QtWidgets.QButtonGroup(frame)
        self.group.setExclusive(True)
        self.buttons = []
        active = labs.index(value) if value in labs else 0
        for k, lab in enumerate(labs):
            rb = QtWidgets.QRadioButton(lab)
            rb.setChecked(k == active)
            self.group.addButton(rb, k)
            lay.addWidget(rb)
            rb.toggled.connect(lambda on, lab=lab: on and self._fire(lab))
            self.buttons.append(rb)
        self.widget = self.qt = frame
        self._finish()

    def _apply_labels(self):
        for rb, lab in zip(self.buttons, self.spec['labels']):
            rb.setText(self.shown(lab))
        if self._title_w is not None:
            self._title_w.setText(self.shown(self._title))

    @property
    def labels(self):
        return list(self.spec['labels'])

    @property
    def value(self):
        k = self.group.checkedId()
        return self.spec['labels'][k] if k >= 0 else None

    value_selected = value

    def set(self, label):
        if label in self.labels and self.value != label:
            self._quiet = True
            rb = self.buttons[self.labels.index(label)]
            for b in self.buttons:
                b.blockSignals(True)
            try:
                rb.setChecked(True)
            finally:
                for b in self.buttons:
                    b.blockSignals(False)
                self._quiet = False


class Button(Control):
    """A button: callback() on a click; set_text changes the shown text (the name stays)."""

    def __init__(self, fig, name, label=None):
        super().__init__(fig, name, label)
        self._text = self.label
        b = QtWidgets.QPushButton(self.label)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.clicked.connect(lambda checked=False: self._fire())
        self.widget = self.qt = b
        self._finish()

    @property
    def value(self):
        return self.widget.text()

    def set_text(self, text):
        self._text = str(text)
        self._apply_labels()

    def _apply_labels(self):
        self.widget.setText(self.shown(self._text))


# The digit stepping of every number box -------------------------------------------------------

# A number in a box's text: a mantissa (sign, digits, point) not inside a word, or an exponent's digits with their sign.
_NUM = re.compile(r'(?<=[eE])[-+]?\d+|(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)')


def _places(tok):
    """Per character of a number token its power of ten (None for the sign and the point)."""
    body = tok.lstrip('+-')
    lead = len(tok) - len(body)
    dot = body.find('.')
    int_len = len(body) if dot < 0 else dot
    out = [None] * lead
    for i, ch in enumerate(body):
        if ch == '.':
            out.append(None)
        elif i < int_len:
            out.append(int_len - 1 - i)
        else:
            out.append(-(i - int_len))
    return out


def step_text(text, s0, s1, d, vmin=None, vmax=None, integer=False):
    """
    The text after one Up (d = +1) or Down (d = -1) press, and the new selection (a0, a1), or
    None when no digit is there.  [s0, s1) is the selection; with none (s0 == s1, the cursor at
    s0) the digit just left of the cursor steps (a point in between is skipped) and the cursor
    stays right of it.  The selected digits go up or down by one in the place of the last selected
    digit, carrying into the digits to the left as arithmetic does, and the same places stay
    selected; the number keeps its decimals; vmin / vmax clamp it, `integer` keeps it whole.
    """
    cursor = s0 == s1
    if cursor:
        k = s1 - 1
        if k >= 0 and text[k] == '.':
            k -= 1
        if k < 0 or not text[k].isdigit():
            return None
        s0, s1 = k, k + 1
    for m in _NUM.finditer(text):
        a, b = m.start(), m.end()
        if not (a <= s0 and s1 <= b and s0 < b):
            continue
        tok = m.group(0)
        pl = _places(tok)
        sel = [pl[i - a] for i in range(s0, s1) if pl[i - a] is not None]
        if not sel:
            return None
        p_lo, p_hi = min(sel), max(sel)
        try:
            v = Decimal(tok)
        except InvalidOperation:
            return None
        body = tok.lstrip('+-')
        ndec = len(body) - body.find('.') - 1 if '.' in body else 0
        nv = v + d * (Decimal(10) ** p_lo)
        if vmin is not None and nv < Decimal(str(vmin)):
            nv = Decimal(str(vmin))
        if vmax is not None and nv > Decimal(str(vmax)):
            nv = Decimal(str(vmax))
        if integer:
            nv = nv.to_integral_value()
        new = format(nv.quantize(Decimal(1).scaleb(-ndec)), 'f') if ndec else format(nv.to_integral_value(), 'f')
        if body.endswith('.') and '.' not in new:
            new += '.'
        if new.startswith('-') and Decimal(new) == 0:
            new = new[1:]
        if tok.startswith('+') and not new.startswith('-'):
            new = '+' + new
        npl = _places(new)
        idx = [i for i, p in enumerate(npl) if p is not None and p_lo <= p <= p_hi]
        if not idx:                       # the places selected are gone (10 -> 9): the units
            idx = [i for i, p in enumerate(npl) if p == 0] or [len(new) - 1]
        out = text[:a] + new + text[b:]
        if cursor:
            return out, (a + idx[-1] + 1, a + idx[-1] + 1)
        return out, (a + idx[0], a + idx[-1] + 1)
    return None


class _DigitEdit(QtWidgets.QLineEdit):
    """A line edit whose Up / Down keys step the selected digits (step_text); `owner._stepped` applies the result."""

    def __init__(self, text, owner):
        super().__init__(text)
        self._owner = owner

    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            d = 1 if key == Qt.Key.Key_Up else -1
            if self.hasSelectedText():
                s0 = self.selectionStart()
                s1 = s0 + len(self.selectedText())
            else:
                s0 = s1 = self.cursorPosition()
            r = step_text(self.text(), s0, s1, d, *self._owner._limits())
            if r is not None:
                text, (a0, a1) = r
                self.setText(text)
                if a1 > a0:
                    self.setSelection(a0, a1 - a0)
                else:
                    self.setCursorPosition(a0)
                self._owner._stepped(text)
            ev.accept()
            return
        super().keyPressEvent(ev)


class Text(Control):
    """
    A text box with its label on the left: value is its text; set(text) never fires;
    callback(text) on Enter, or on leaving the box after an edit, and at once on every Up / Down
    digit step (step_text).  The box's width is its specification's (inches), at the text zoom.
    """

    def __init__(self, fig, name, value='', label=None, width=None):
        super().__init__(fig, name, label)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self._lab = QtWidgets.QLabel(self.label)
        lay.addWidget(self._lab)
        if not self.label.strip():
            self._lab.hide()
        e = _DigitEdit(str(value), self)
        e.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._base_w = int(width) if width is not None else int(round(self.spec.get('w', 0.6) * plt.DPI))
        if fig is None:
            e.setMaximumWidth(self._base_w)          # its window scales it
        else:
            e.setFixedWidth(self._base_w)
            # A fixed width with a fixed policy: a line edit's own policy is 'expanding', which would make
            # every row or box holding this one claim the spare width of its window's layout.
            e.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, e.sizePolicy().verticalPolicy())
        e.editingFinished.connect(self._submit)
        e.returnPressed.connect(self._enter)
        lay.addWidget(e)
        self._last = str(value)
        self._entered = False
        self.widget = e
        self.qt = w
        self._labels = [_LabelProxy(self._lab, self._set_label)]
        self._finish()

    def _limits(self):
        return None, None, False

    def _after_zoom(self, z):
        self.widget.setFixedWidth(int(round(self._base_w * z)))

    def _enter(self):
        self._entered = True

    def _submit(self):
        """Enter, or the box left after an edit: the text applied (_accept) unless it is unchanged and nothing was entered."""
        text = self.widget.text()
        entered, self._entered = self._entered, False
        if text == self._last and not entered:
            return
        self._accept(text)

    def _stepped(self, text):
        """A digit step (Up / Down): the new text applied at once."""
        self._accept(text)

    def _accept(self, text):
        """The text taken as the box's value and reported; Value and Count parse and clamp it first."""
        self._last = text
        self._fire(text)

    @property
    def label_text(self):
        return self._labels[0]

    def _set_label(self, text):
        self.label = str(text)
        self._apply_labels()
        self._lab.setVisible(bool(self.label.strip()))

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def set_label_color(self, color):
        self._labels[0].set_color(color)

    @property
    def value(self):
        return self.widget.text()

    @property
    def text(self):
        return self.widget.text()

    def set(self, text):
        text = str(text)
        self._last = text
        if self.widget.text() != text:
            self._quiet = True
            blocked = self.widget.blockSignals(True)
            try:
                self.widget.setText(text)
            finally:
                self.widget.blockSignals(blocked)
                self._quiet = False

    set_val = set


class Count(Text):
    """
    A whole-number box with limits (the marker and group counts that were sliders): value and val
    are the number; set(n) never fires; callback(n) on Enter, on leaving the box after an edit, and
    on every Up / Down step, which stops at vmin / vmax.  An entry that is not a whole number within
    the limits is reported on the console and the box put back.  set_color(c) draws the box's frame
    in c (a marker group's colour, so the box tells which average it belongs to).
    """

    def __init__(self, fig, name, value=0, vmin=None, vmax=None, label=None, width=None):
        spec = SPECS.get(name, {})
        self.vmin = int(spec.get('vmin', 0) if vmin is None else vmin)
        self.vmax = int(spec.get('vmax', 10) if vmax is None else vmax)
        super().__init__(fig, name, str(int(round(float(value)))), label=label,
                         width=width if width is not None else 34)
        self._n = int(round(float(value)))
        self._color = None
        self.widget.setObjectName('count')

    def _limits(self):
        return self.vmin, self.vmax, True

    def _accept(self, text):
        try:
            v = float(str(text).strip())
            if v != int(v) or not (self.vmin <= v <= self.vmax):
                raise ValueError
        except ValueError:
            print(f"{self.label or self.name}: '{text}' is not a whole number from {self.vmin} to {self.vmax}; "
                  f"the box is put back to {self._n}.")
            self.set(self._n)
            return
        self._n = int(v)
        self._last = str(self._n)
        if self.widget.text() != self._last:
            self.set(self._n)
        self._fire(self._n)

    @property
    def value(self):
        return self._n

    @property
    def val(self):
        return float(self._n)

    def set(self, n):
        self._n = int(round(float(n)))
        super().set(str(self._n))

    set_val = set

    def set_color(self, color):
        """The box's frame in `color` (a marker group's colour)."""
        if color != self._color:
            self._color = color
            self.widget.setStyleSheet(f"QLineEdit#count {{ border: 2px solid {qcolor(color).name()}; border-radius: 3px; }}")


class Value(Text):
    """
    A number box with limits (the values that were sliders: E Min / E Max, σ, the tip heights):
    value and val are the number; set(v) never fires; callback(v) on Enter, on leaving the box
    after an edit, and on every Up / Down step, which stops at vmin / vmax (None: no limit on
    that side).  An entry that is not a number is reported on the console and the box put back; a
    number outside the limits is clamped to them (as a slider's set_val clamped it) and the
    clamped value applied.  `decimals` digits are shown (the specification's, 3 by default).
    """

    def __init__(self, fig, name, value=0.0, vmin=None, vmax=None, decimals=None, label=None, width=None):
        spec = SPECS.get(name, {})
        self.vmin = spec.get('vmin') if vmin is None else vmin
        self.vmax = spec.get('vmax') if vmax is None else vmax
        self.vmin = None if self.vmin is None else float(self.vmin)
        self.vmax = None if self.vmax is None else float(self.vmax)
        self.decimals = int(spec.get('decimals', 3) if decimals is None else decimals)
        self._v = self._clamp(float(value))
        super().__init__(fig, name, self._fmt(self._v), label=label, width=width)
        self.widget.setObjectName('value')

    def _fmt(self, v):
        return f"{v:.{self.decimals}f}"

    def _clamp(self, v):
        if self.vmin is not None and v < self.vmin:
            v = self.vmin
        if self.vmax is not None and v > self.vmax:
            v = self.vmax
        return v

    def _limits(self):
        return self.vmin, self.vmax, False

    def _accept(self, text):
        try:
            v = float(str(text).strip())
            if not np.isfinite(v):
                raise ValueError
        except ValueError:
            lo = 'any' if self.vmin is None else f"{self.vmin:g}"
            hi = 'any' if self.vmax is None else f"{self.vmax:g}"
            print(f"{self.label or self.name}: '{text}' is not a number (limits {lo} to {hi}); "
                  f"the box is put back to {self._fmt(self._v)}.")
            self.set(self._v)
            return
        self._v = self._clamp(v)
        self._last = self._fmt(self._v)
        if self.widget.text() != self._last:
            self.set(self._v)
        self._fire(self._v)

    @property
    def value(self):
        return self._v

    @property
    def val(self):
        return self._v

    def set(self, v):
        self._v = self._clamp(float(v))
        super().set(self._fmt(self._v))

    set_val = set

    def set_limits(self, vmin=None, vmax=None):
        """New limits (a tip height's top after a file loads); the value is clamped to them without firing."""
        self.vmin = None if vmin is None else float(vmin)
        self.vmax = None if vmax is None else float(vmax)
        self.set(self._v)


class Slider(Control):
    """
    An integer slider with its label left of it and its value right of it (the one slider kept:
    scrubbing a list, forces.py's window start over the ionic steps): callback(value) while it
    moves; release(callback) registers callback() for the end of a move (the mouse released after
    a drag, or at once for a keyboard / page-click step: a slow load happens then).
    """

    def __init__(self, fig, name, value=0, vmin=0, vmax=1, label=None):
        super().__init__(fig, name, label)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lab = QtWidgets.QLabel(self.label)
        lay.addWidget(self._lab)
        sl = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        sl.setRange(int(vmin), int(vmax))
        sl.setValue(int(value))
        sl.setPageStep(max(1, (int(vmax) - int(vmin)) // 20))
        self.slider = sl
        self.readout = QtWidgets.QLabel(str(int(value)))
        self.readout.setMinimumWidth(36)
        lay.addWidget(sl, 1)
        lay.addWidget(self.readout)
        self._release = None
        sl.valueChanged.connect(self._changed)
        sl.sliderReleased.connect(self._released)
        self.widget = sl
        self.qt = w
        self._finish()

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def _changed(self, v):
        self.readout.setText(str(v))
        if self._quiet:
            return
        self._fire(v)
        if not self.slider.isSliderDown():          # keyboard, wheel, or page click: no drag to wait for
            self._released()

    def _released(self):
        if self._release is not None and not self._quiet:
            self._release()

    def release(self, callback):
        self._release = callback

    @property
    def value(self):
        return self.slider.value()

    def set(self, value):
        v = int(round(value))
        if self.slider.value() != v:
            self._quiet = True
            blocked = self.slider.blockSignals(True)
            try:
                self.slider.setValue(v)
                self.readout.setText(str(v))
            finally:
                self.slider.blockSignals(blocked)
                self._quiet = False

    def set_range(self, vmin, vmax):
        self._quiet = True
        blocked = self.slider.blockSignals(True)
        try:
            self.slider.setRange(int(vmin), int(vmax))
            self.slider.setPageStep(max(1, (int(vmax) - int(vmin)) // 20))
        finally:
            self.slider.blockSignals(blocked)
            self._quiet = False


class TextPanel(Control):
    """
    A panel that holds only text (a window's readout or help text): it scrolls when the text needs
    more room than the layout gives it (a window never scrolls); the text can be selected and
    copied.  The interface of a matplotlib text: set_text, get_text, set_visible, get_visible,
    set_color, set_fontsize; `mono` for a monospace readout.
    """

    def __init__(self, fig, name='readout', text='', fs=7, mono=False, lines=None, fixed=False):
        super().__init__(fig, name, label='')
        self._fixed = bool(fixed)
        self.spec['fs'] = fs
        e = QtWidgets.QPlainTextEdit()
        e.setReadOnly(True)
        e.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        e.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap if mono
                          else QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        e.setStyleSheet("QPlainTextEdit { background: transparent; }")
        e.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        e.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        pol = QtWidgets.QSizePolicy.Policy
        e.setSizePolicy(pol.Expanding, pol.Expanding)
        self._mono = bool(mono)
        self._lines = lines
        self._text = ''
        self.widget = self.qt = e
        self._finish()
        self.set_text(text)

    def _set_zoom(self, z):
        if self.fig is None:
            return
        self._zoom = z
        f = _font(self._fs(), z)
        if self._mono:
            f.setFamily('monospace')
            f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.widget.setFont(f)
        if self._lines:
            # `lines` lines of text tall: exactly (fixed) or at least; more text scrolls.
            fm = QtGui.QFontMetrics(f)
            h = int(fm.lineSpacing() * self._lines + 2 * self.widget.document().documentMargin() + 4)
            if self._fixed:
                self.widget.setFixedHeight(h)
            else:
                self.widget.setMinimumHeight(h)

    def sizeHint(self):
        return self.widget.sizeHint()

    def set_text(self, s):
        s = '' if s is None else str(s)
        if s != self._text:
            self._text = s
            self.widget.setPlainText(s)

    def get_text(self):
        return self._text

    def set_visible(self, flag):
        self.show(flag)

    def get_visible(self):
        return self._visible

    def set_color(self, c):
        self.widget.setStyleSheet(f"QPlainTextEdit {{ background: transparent; color: {qcolor(c).name()}; }}")

    def set_fontsize(self, fs):
        self.spec['fs'] = fs
        self._zoom = None
        if self.fig is not None:
            self._set_zoom(self.fig.control_zoom)


class Label(Control):
    """A short text beside controls (a row's name): set_text, get_text, set_visible, set_color, as a matplotlib text."""

    def __init__(self, fig, name='label', text='', fs=8, wrap=False):
        super().__init__(fig, name, label='')
        self.spec['fs'] = fs
        self._text = str(text)
        self.widget = self.qt = QtWidgets.QLabel(self._text)
        if wrap:
            # A long line wraps rather than widening the window; the text can be selected and copied.
            self.widget.setWordWrap(True)
            self.widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._finish()

    def _apply_labels(self):
        self.widget.setText(self.shown(self._text))

    def set_text(self, s):
        self._text = '' if s is None else str(s)
        self._apply_labels()

    def get_text(self):
        return self._text

    def set_visible(self, flag):
        self.show(flag)

    def get_visible(self):
        return self._visible

    def set_color(self, c):
        self.widget.setStyleSheet(f"color: {qcolor(c).name()};")


_KINDS = {'check': Check, 'checks': Checks, 'radio': Radio, 'button': Button, 'text': Text, 'value': Value,
          'count': Count, 'slider': Slider}


def make(fig, name, **kw):
    """The control named `name`, of the kind widgets.SPECS gives it ('count' and 'value' are the boxes that were sliders; 'window start' the one Slider)."""
    return _KINDS[SPECS[name]['kind']](fig, name, **kw)


# ---------------------------------------------------------------------- #
#  Composite controllers                                                   #
# ---------------------------------------------------------------------- #

class StepButtons:
    """
    A pair of step buttons (names such as '< line' / 'line >') in one row: on_change(cb(d)) with
    d = -1 for the left one, +1 for the right.  chevrons=True shows them as '‹' and '›' shapes.
    """

    CHEVRONS = ('‹', '›')

    def __init__(self, fig, left_name, right_name, chevrons=False):
        self.left = Button(fig, left_name, label=self.CHEVRONS[0] if chevrons else None)
        self.right = Button(fig, right_name, label=self.CHEVRONS[1] if chevrons else None)
        if chevrons:
            for b in (self.left, self.right):
                b.spec['fs'] = 13
                b.widget.setStyleSheet("QPushButton { font-weight: bold; padding: 0px 10px; }")
        self._cb = None
        self.left.on_change(lambda: self._fire(-1))
        self.right.on_change(lambda: self._fire(+1))
        self.qt = widget_of(row(self.left, self.right, spacing=3))

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, d):
        if self._cb is not None:
            self._cb(d)

    def show(self, flag):
        self.qt.setVisible(bool(flag))


class RunControls:
    """Named buttons ('run', 'clear', 'restore', 'save', 'Send to fit', ...): on(name, cb) wires one; the window places buttons[name]."""

    def __init__(self, fig, names):
        self.buttons = {name: Button(fig, name) for name in names}

    def on(self, name, callback):
        self.buttons[name].on_change(callback)

    def __getitem__(self, name):
        return self.buttons[name]


class SigmaBox:
    """
    The 'σ (eV)' box of the spread, state-decomposition and global-decomposition windows and the
    simulator wrapper: the Gaussian width of the MO spread curves, a Value box with the window's
    limits (SIGMA_MIN .. SIGMA_MAX), three decimals.  on_change(callback) runs callback(sigma) on
    Enter, on leaving the box after an edit, and on every digit step (each step applies at once);
    set(sigma) puts a value in without firing.  Replaces the σ slider with its mirrored text box
    that every one of those windows built by hand.
    """

    def __init__(self, fig, sigma, vmin, vmax, name='σ (eV)'):
        self.box = Value(fig, name, sigma, vmin=vmin, vmax=vmax)
        self._cb = None
        self.box.on_change(self._changed)
        self.qt = self.box.qt

    def on_change(self, callback):
        self._cb = callback

    def show(self, flag):
        self.box.show(flag)

    def enable(self, flag):
        self.box.enable(flag)

    @property
    def value(self):
        return self.box.value

    sigma = value

    def set(self, sigma):
        self.box.set(sigma)

    def _changed(self, v):
        if self._cb is not None:
            self._cb(float(v))


class GammaBox:
    """
    The 'γ' box of a heatmap: the power-law exponent of its colour map (1 = linear).  A finite
    number > 0 is accepted; anything else is reported and the box put back.  on_change(callback)
    runs callback(gamma).  The window puts `qt` under its heatmap in its layout.  `name` picks the
    specification ('γ', or the simulator's 'LDOS γ'); `fmt` the digits shown ('g', or '.2f').
    """

    def __init__(self, fig, gamma=1.0, name='γ', fmt='g'):
        self._fmt = fmt
        self.box = Text(fig, name, format(float(gamma), fmt))
        self.gamma = float(gamma)
        self._cb = None
        self.box.on_change(self._on_submit)
        self.qt = self.box.qt

    def on_change(self, callback):
        self._cb = callback

    def show(self, flag):
        self.box.show(flag)

    @property
    def value(self):
        return self.gamma

    @property
    def text(self):
        return self.box.value

    @property
    def val(self):
        """The gamma as a number (the old slider's `val`)."""
        return self.gamma

    def set(self, gamma):
        self.gamma = float(gamma)
        self.box.set(format(self.gamma, self._fmt))

    set_val = set

    def _on_submit(self, text):
        try:
            g = float(str(text).strip())
        except ValueError:
            g = -1.0
        if not np.isfinite(g) or g <= 0:
            print(f"γ: '{text}' is not a positive number; the box is put back to {format(self.gamma, self._fmt)}.")
            self.box.set(format(self.gamma, self._fmt))
            return
        if format(g, self._fmt) != self.box.value:
            self.box.set(format(g, self._fmt))
        if g == self.gamma:
            return
        self.gamma = g
        if self._cb is not None:
            self._cb(g)
