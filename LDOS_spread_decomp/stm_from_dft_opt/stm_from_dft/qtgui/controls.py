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

import re
import weakref
from decimal import Decimal, InvalidOperation
import numpy as np
from . import qtplot as plt
from .qtplot import QtCore, QtGui, QtWidgets, qfont, qcolor
from . import widgets
from .widgets import SPECS, LABELS, short_form

Qt = QtCore.Qt

# The box frame of every group of controls in every window (the light frame of the figure windows):
# its border width and radius are filled in where it is set (1 and 3 px at the old fixed 100 DPI, by em_px).
FRAME_CSS = "QFrame#box { background: #f0f0f0; border: %dpx solid #c8c8c8; border-radius: %dpx; }"
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
    frame.setStyleSheet(FRAME_CSS % (round(1 / (1000 / 72) * plt.em_px()), round(3 / (1000 / 72) * plt.em_px())))
    outer = QtWidgets.QVBoxLayout(frame)
    outer.setContentsMargins(*(round(m / (1000 / 72) * plt.em_px()) for m in margins))
    outer.setSpacing(round(2 / (1000 / 72) * plt.em_px()))
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
    inner.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
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
    lay.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
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
    lay.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
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
        self._hs = int(round(hspacing / (1000 / 72) * plt.em_px()))
        self._vs = int(round(vspacing / (1000 / 72) * plt.em_px()))
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
        lay.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
        lay.addStretch(1)
        for it in items:
            lay.addWidget(_qw(it))
        lay.addStretch(1)
        self._ax = ax
        if follow:
            ax.on_placed(self._follow)         # follow=False: centred on the whole cell (a narrow cell, whose box is centred in it)

    def _items_w(self):
        """The width the items need (their size hints, whole labels, and the spacing between them)."""
        lay = self.layout()
        ws = [lay.itemAt(i).widget() for i in range(lay.count())]
        ws = [w for w in ws if w is not None and not w.isHidden()]
        return sum(max(w.sizeHint().width(), w.minimumSizeHint().width()) for w in ws) + lay.spacing() * max(len(ws) - 1, 0)

    def need_w(self):
        """The width the panel under this box must have for the box's labels to stay whole."""
        return float(self._items_w())

    def _follow(self):
        ax = self._ax
        cell = getattr(ax, '_cell', None)
        if cell is None:
            return
        cx, cy, cw, ch = cell.rect_px()
        x0, y0, w, h = ax._px_rect()
        left = max(int(round(x0 - cx)), 0)
        right = max(int(round(cx + cw - (x0 + w))), 0)
        # The items keep their width: the margins never close the room under the box below what
        # they need (a box wider than a small panel's box is centred on it as far as the cell allows).
        room = max(int(cw) - self._items_w(), 0)
        left = min(left, room)
        right = min(right, room - left)
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
        lay.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
        narrow = getattr(cell, 'inner_w', None) is not None
        self.above = AbovePanel(ax, *items, follow=not narrow)
        lay.addWidget(self.above, 0)
        lay.addWidget(cell, 1)
        self.cell = cell
        self._sp = int(round(spacing / (1000 / 72) * plt.em_px()))
        if narrow:
            # Over a narrow cell (a tip-height stripe): the pair is as wide as the wider of the box and
            # the cell, a fixed-width item of its row like the cell itself (PanelRow._fixed).
            pol = QtWidgets.QSizePolicy.Policy
            self.setSizePolicy(pol.Fixed, pol.Expanding)

    def _box_h(self):
        hidden = self.above.isHidden()
        return plt.pass_cached(self, ('box_h', hidden), lambda: 0.0 if hidden else float(self.above.sizeHint().height()) + self._sp)

    def height_for_width(self, w):
        h = self.cell.height_for_width(w)
        return None if h is None else h + self._box_h()

    def width_for_height(self, h):
        w = self.cell.width_for_height(max(float(h) - self._box_h(), 1.0))
        if w is None:
            return None
        return max(w, self.above.need_w())     # never narrower than the box over the panel

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
        self.main_share, self._sp, self._hys = float(main_share), int(round(spacing / (1000 / 72) * plt.em_px())), float(hysteresis)
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
        if _panel_shapes(self.main) != getattr(self, '_last_shapes', None):
            self._arrange()                    # the main panel's shape changed; a label's text changes nothing

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
        keeping its size (Figure.fit_window, which nothing calls: the window is never trimmed to its
        panels).  With the main pane at the left ('h') the main panel spans the height the window
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
        self._last_shapes = _panel_shapes(self.main)     # a layout request arranges again only when it changes
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
        if ev.type() == QtCore.QEvent.Type.LayoutRequest and _panel_shapes(self.main) != getattr(self, '_last_shapes', None):
            self._arrange()
        return super().event(ev)

    def minimumSizeHint(self):
        a, b = self.main.minimumSizeHint(), self.rest.minimumSizeHint()
        return QtCore.QSize(max(a.width(), b.width(), round(100 / (1000 / 72) * plt.em_px())), max(a.height(), b.height(), round(100 / (1000 / 72) * plt.em_px())))

    def sizeHint(self):
        return QtCore.QSize(round(900 / (1000 / 72) * plt.em_px()), round(600 / (1000 / 72) * plt.em_px()))


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
        plt.new_measure_pass()

    def sizeHint(self):
        return plt.pass_cached(self, ('hint', len(self._items)), self._hint_now)      # once per layout pass

    def _hint_now(self):
        # As thick as the thickest item will really be: the larger of its hint and its minimum (a
        # number box's minimum width is often the larger), so that a shape counting the band is right.
        hints = [(it[0].sizeHint(), it[0].minimumSizeHint()) for it in self._items.values()]
        if self.orient == 'h':
            return QtCore.QSize(0, max([max(h.height(), m.height()) for h, m in hints] + [0]))
        return QtCore.QSize(max([max(h.width(), m.width()) for h, m in hints] + [0]), 0)

    def minimumSizeHint(self):
        return self.sizeHint()

    def take(self, key):
        """Item `key` taken out of the band: its widget, hidden and unparented (for another band's add); None when the band has no such item."""
        it = self._items.pop(key, None)
        if it is None:
            return None
        w = it[0]
        w.hide()
        w.setParent(None)
        self.updateGeometry()
        plt.new_measure_pass()
        return w

    def set_pos(self, key, coord_px):
        """Centre item `key` at canvas coordinate `coord_px` along the band (x for 'h', y for 'v')."""
        self._items[key][1] = float(coord_px)
        self._apply(key)

    def _apply(self, key):
        w, c = self._items[key]
        s = w.sizeHint()
        if c is None:
            # Not placed yet: at the band's start, the band's thickness (never wider than the band).
            if self.orient == 'h':
                w.setGeometry(0, 0, s.width(), self.height())
            else:
                w.setGeometry(0, 0, self.width(), s.height())
            return
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

    def event(self, ev):
        if ev.type() == QtCore.QEvent.Type.LayoutRequest:
            # An item's size hint changed (a text zoom): the band's thickness follows, and the layout
            # holding the band is told (the band has no layout of its own to pass it on).
            self.updateGeometry()
            for k in self._items:
                self._apply(k)
        return super().event(ev)


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
    stateful = False              # a Button whose label shows a state: setting() gives its label

    def __init__(self, fig, name, label=None):
        self.fig = fig
        self.name = name
        self.spec = dict(SPECS.get(name, {'kind': None, 'w': 1.0, 'h': 0.3, 'fs': 8}))
        self.label = _label_of(name) if label is None else str(label)
        # The name of this control's line in the info panel's settings section (setting, settings_text):
        # its label, or its name when the label is empty; a composite box or a window sets another
        # where a label is empty or repeated in its window.
        self.setting_label = self.label if self.label.strip() else str(name)
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
        self._changed()

    def enable(self, flag):
        """The control takes events (True) or ignores them (False)."""
        self._enabled = bool(flag)
        self.qt.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self._enabled)

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, *args):
        self._changed()                # the info panel shows the change, with or without a callback
        if self._cb is not None and not self._quiet and self.eventson:
            self._cb(*args)

    def setting(self):
        """
        This control's line in the info panel's settings section: (setting_label, the value as
        shown), or None for a control that holds no setting (a Button, unless stateful, whose label
        then is the setting; a Label; a TextPanel).  A Checks gives a list, one (row label, on/off)
        per row.  Each kind gives its own (Check, Checks, Radio, Text, Value, Count, Slider, Button).
        """
        return None

    def _changed(self):
        """
        A setting of this control changed (or the control was shown, hidden or removed): its window's
        hook (sm4_settings_changed: a figure window's coalesced info push) is called.  Never raises:
        a control without a window, or a closed one, is ignored.
        """
        try:
            if self.qt is None:
                return
            hook = getattr(self.qt.window(), 'sm4_settings_changed', None)
            if hook is not None:
                hook()
        except Exception:              # the window or its Qt objects are already gone
            pass

    @property
    def ax(self):
        """The control's place (for AveragePicker's anchor): its Qt widget."""
        return _Anchor(self)

    def remove(self):
        self._changed()
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

    def setting(self):
        return (self.setting_label, 'on' if self.value else 'off')

    def set(self, flag):
        if self.value != bool(flag):
            self._quiet = True
            blocked = self.widget.blockSignals(True)
            try:
                self.widget.setChecked(bool(flag))
            finally:
                self.widget.blockSignals(blocked)
                self._quiet = False
        self._changed()

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
        frame.setStyleSheet(FRAME_CSS % (round(1 / (1000 / 72) * plt.em_px()), round(3 / (1000 / 72) * plt.em_px())))
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(round(4 / (1000 / 72) * plt.em_px()), round(2 / (1000 / 72) * plt.em_px()),
                               round(4 / (1000 / 72) * plt.em_px()), round(2 / (1000 / 72) * plt.em_px()))
        lay.setSpacing(round(1 / (1000 / 72) * plt.em_px()))
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
                r.setContentsMargins(int(round(ind / (1000 / 72) * plt.em_px())), 0, 0, 0)
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

    def setting(self):
        return [(lab, 'on' if cb.isChecked() else 'off') for cb, lab in zip(self.boxes, self.spec['labels'])]

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
        self._changed()

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
        frame.setStyleSheet(FRAME_CSS % (round(1 / (1000 / 72) * plt.em_px()), round(3 / (1000 / 72) * plt.em_px())))
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(round(4 / (1000 / 72) * plt.em_px()), round(2 / (1000 / 72) * plt.em_px()),
                               round(4 / (1000 / 72) * plt.em_px()), round(2 / (1000 / 72) * plt.em_px()))
        lay.setSpacing(round(1 / (1000 / 72) * plt.em_px()))
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

    def setting(self):
        v = self.value
        return (self.setting_label, '' if v is None else str(v))

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
        self._changed()


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

    def setting(self):
        return (self.setting_label, self._text) if self.stateful else None

    def set_text(self, text):
        self._text = str(text)
        self._apply_labels()
        self._changed()

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
        lay.setSpacing(round(3 / (1000 / 72) * plt.em_px()))
        self._lab = QtWidgets.QLabel(self.label)
        lay.addWidget(self._lab)
        if not self.label.strip():
            self._lab.hide()
        e = _DigitEdit(str(value), self)
        e.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._base_w = int(round(width / (1000 / 72) * plt.em_px())) if width is not None else int(round(self.spec.get('w', 0.6) * 72 * plt.px_per_pt()))
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

    def setting(self):
        return (self.setting_label, self.widget.text())

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
        self._changed()

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
            self.widget.setStyleSheet(f"QLineEdit#count {{ border: {round(2 / (1000 / 72) * plt.em_px())}px solid {qcolor(color).name()}; "
                                      f"border-radius: {round(3 / (1000 / 72) * plt.em_px())}px; }}")


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
        self.readout.setMinimumWidth(round(36 / (1000 / 72) * plt.em_px()))
        lay.addWidget(sl, 1)
        lay.addWidget(self.readout)
        self._release = None
        sl.valueChanged.connect(self._moved)
        sl.sliderReleased.connect(self._released)
        self.widget = sl
        self.qt = w
        self._finish()

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def _moved(self, v):
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

    def setting(self):
        return (self.setting_label, self.readout.text())

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
        self._changed()

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
            fm = QtGui.QFontMetricsF(f)
            h = int(fm.lineSpacing() * self._lines + 2 * self.widget.document().documentMargin() + 4 / (1000 / 72) * plt.em_px())
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


def settings_text(root):
    """
    The settings section of a window's info panel: every live control (Control._all) whose Qt widget
    is visible and inside `root` (a window), with a setting (setting()), ordered by its place on
    screen (its widget's top left mapped to `root`: top to bottom, then left to right), one line
    each, 'label: value' (a Checks one line per row).  '' when there is none.
    """
    if root is None:
        return ''
    found = []
    for c in list(Control._all):
        qt = c.qt
        if qt is None:
            continue
        try:
            if not qt.isVisible() or not root.isAncestorOf(qt):
                continue
            s = c.setting()
            if s is None:
                continue
            pos = qt.mapTo(root, QtCore.QPoint(0, 0))
        except RuntimeError:           # its Qt widget is already deleted
            continue
        found.append(((pos.y(), pos.x()), s))
    found.sort(key=lambda item: item[0])
    lines = []
    for _, s in found:
        for lab, val in (s if isinstance(s, list) else [s]):
            lines.append(f"{lab}: {val}")
    return "\n".join(lines)


# qtplot.Figure.info_text lists a figure window's settings through this module's settings_text,
# bound to the qtplot this module imported: a window's info panel then always reads the controls of
# the copy of this module that made them (a module reloader that drops the package from sys.modules
# cannot split them).
plt._settings_text = settings_text


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
                b.widget.setStyleSheet(f"QPushButton {{ font-weight: bold; padding: 0px {round(10 / (1000 / 72) * plt.em_px())}px; }}")
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


# ---------------------------------------------------------------------- #
#  Rule 11, the arrangement: plots fill the window, blank space minimized  #
# ---------------------------------------------------------------------- #

def _shaped_size(widget, W, H):
    """
    The largest size (w, h) a shaped widget takes inside W x H: its own `fit(W, H)` when it has one
    (a wrapping block chooses its columns), else from height_for_width / width_for_height (a cell,
    a PanelWithBox, a RowWidget); None for a widget without a shape.
    """
    W, H = max(float(W), 1.0), max(float(H), 1.0)
    fit = getattr(widget, 'fit', None)
    if fit is not None:
        r = fit(W, H)
        if r is not None:
            return (float(r[0]), float(r[1]))
    hfw, wfh = getattr(widget, 'height_for_width', None), getattr(widget, 'width_for_height', None)
    try:
        h = hfw(W) if hfw is not None else None
        if h is None:
            # No shape yet (a row with no panel in it: a control box waiting for RUN): the widget's own
            # size hint, so that the group keeps its place and stays reachable.
            hint = widget.sizeHint()
            return (min(W, float(max(hint.width(), 1))), min(H, float(max(hint.height(), 1))))
        if h <= H:
            return (W, float(h))
        w = wfh(H)
        if w is None:
            return None
        w = min(float(w), W)
        h2 = hfw(w)
        return (w, min(float(h2 if h2 is not None else H), H))
    except Exception:
        return None


def _panel_shapes(widget):
    """
    The shapes of the panels in `widget` (the widget itself when it is a panel cell), in order: each
    cell's height over width (PanelCell.shape_ratio, rounded as Axes._notify_shape rounds it; None
    for a free panel).  A crop or a turn of a panel changes it, a panel added or removed too; the text
    of a label or a tick value never does (a container's layout request re-places only on a change).
    """
    cells = [widget] if isinstance(widget, plt.PanelCell) else []
    cells += [c for c in widget.findChildren(QtWidgets.QWidget) if isinstance(c, plt.PanelCell)]
    out = []
    for c in cells:
        r = c.shape_ratio()
        out.append(None if r is None else round(float(r), 4))
    return tuple(out)


class Titled(QtWidgets.QWidget):
    """
    A shaped group with a label over it (the 'LDOS' label over the heatmap row or the energy maps):
    the label centred above `body`, the pair one shaped item (height_for_width, width_for_height,
    fit forward to the body plus the label's height).  `label` is a controls.Label or a QWidget.
    """

    def __init__(self, body, label, spacing=2):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
        self.label = _qw(label)
        lay.addWidget(self.label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(body, 1)
        self.body = body
        self._sp = int(round(spacing / (1000 / 72) * plt.em_px()))

    def _label_h(self):
        return 0.0 if self.label.isHidden() else float(self.label.sizeHint().height()) + self._sp

    def fit(self, W, H):
        r = _shaped_size(self.body, W, max(float(H) - self._label_h(), 1.0))
        return None if r is None else (r[0], r[1] + self._label_h())

    def height_for_width(self, w):
        hfw = getattr(self.body, 'height_for_width', None)
        h = hfw(w) if hfw is not None else None
        return None if h is None else h + self._label_h()

    def width_for_height(self, h):
        wfh = getattr(self.body, 'width_for_height', None)
        return None if wfh is None else wfh(max(float(h) - self._label_h(), 1.0))

    def hasHeightForWidth(self):
        return bool(getattr(self.body, 'hasHeightForWidth', lambda: False)())

    def heightForWidth(self, w):
        h = self.height_for_width(w)
        return -1 if h is None else int(round(h))


class FlipBox(QtWidgets.QWidget):
    """
    Free panels side by side when their space is wider than tall, stacked when it is taller than
    wide (rule 11: the curve panels of the simulator's Line / Path views beside or under the heatmap
    row).  `items` are widgets with a stretch each: FlipBox((cell, 25), (cell, 10)) or plain
    widgets (stretch 1).  The direction flips at a width-to-height ratio of `wide` (to a row) and
    `tall` (to a column), the gap between keeping it steady.
    """

    def __init__(self, *items, spacing=6, wide=1.3, tall=0.9):
        super().__init__()
        self._lay = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.LeftToRight, self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(round(spacing / (1000 / 72) * plt.em_px()))
        for it in items:
            w, st = (it if isinstance(it, tuple) else (it, 1))
            self._lay.addWidget(_qw(w), int(st))
        self._wide, self._tall = float(wide), float(tall)
        pol = QtWidgets.QSizePolicy.Policy
        self.setSizePolicy(pol.Expanding, pol.Expanding)

    @property
    def direction(self):
        return 'row' if self._lay.direction() == QtWidgets.QBoxLayout.Direction.LeftToRight else 'column'

    def _flip(self):
        w, h = float(self.width()), float(max(self.height(), 1))
        r = w / h
        want = None
        if r > self._wide:
            want = QtWidgets.QBoxLayout.Direction.LeftToRight
        elif r < self._tall:
            want = QtWidgets.QBoxLayout.Direction.TopToBottom
        if want is not None and want != self._lay.direction():
            self._lay.setDirection(want)

    def resizeEvent(self, ev):
        self._flip()
        super().resizeEvent(ev)

    def minimumSizeHint(self):
        return QtCore.QSize(round(120 / (1000 / 72) * plt.em_px()), round(90 / (1000 / 72) * plt.em_px()))

    def sizeHint(self):
        return QtCore.QSize(round(400 / (1000 / 72) * plt.em_px()), round(300 / (1000 / 72) * plt.em_px()))


class CropPanel(QtWidgets.QWidget):
    """
    The placement panel with its crop: a control box over it (AbovePanel: the tip-height box, the
    Crop / Uncrop buttons), a band under that box where the x-cursor boxes ride along the panel's
    top edge (RideBand 'h', `band_x`), the panel's cell, and a band at its right where the
    y-cursor boxes ride along its right edge (RideBand 'v', `band_y`).  One shaped item of an
    Arrangement: height_for_width and width_for_height count the box and the bands, so the group
    is sized as one and the boxes never sit between panels.  The window adds the riding boxes
    (`band_x.add(key, box)`, `band_y.add(key, box)`) and moves them (`set_pos`) when the panel is
    placed (Axes.on_placed); when the panel is turned (Axes.set_rotated) the window moves each box
    to the band its cursor's line now runs along (`take` and `add`).
    """

    def __init__(self, ax, cell, *items, spacing=3):
        super().__init__()
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(round(spacing / (1000 / 72) * plt.em_px()))
        grid.setVerticalSpacing(round(spacing / (1000 / 72) * plt.em_px()))
        self.above = AbovePanel(ax, *items)
        self.band_x = RideBand('h')
        self.band_y = RideBand('v')
        grid.addWidget(self.above, 0, 0)       # over the cell's column only: the y band keeps its own width whatever the box needs
        grid.addWidget(self.band_x, 1, 0)
        grid.addWidget(cell, 2, 0)
        grid.addWidget(self.band_y, 2, 1)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(2, 1)
        self.cell = cell
        self._sp = int(round(spacing / (1000 / 72) * plt.em_px()))

    def _box_h(self):
        # Measured once per layout pass (plt.pass_cached): the arrangement asks it hundreds of times in one.
        key = ('box_h', self.above.isHidden(), self.band_x.isHidden())
        return plt.pass_cached(self, key, self._box_h_now)

    def _box_h_now(self):
        h = 0.0 if self.above.isHidden() else float(self.above.sizeHint().height()) + self._sp
        if not self.band_x.isHidden():
            h += float(self.band_x.sizeHint().height()) + self._sp
        return h

    def _band_w(self):
        hidden = self.band_y.isHidden()
        return plt.pass_cached(self, ('band_w', hidden), lambda: 0.0 if hidden else float(self.band_y.sizeHint().width()) + self._sp)

    def height_for_width(self, w):
        h = self.cell.height_for_width(max(float(w) - self._band_w(), 1.0))
        return None if h is None else h + self._box_h()

    def width_for_height(self, h):
        w = self.cell.width_for_height(max(float(h) - self._box_h(), 1.0))
        if w is None:
            return None
        return max(w + self._band_w(), self.above.need_w())       # never narrower than the boxes over the panel

    def hasHeightForWidth(self):
        return self.cell.hasHeightForWidth()

    def heightForWidth(self, w):
        h = self.height_for_width(w)
        return -1 if h is None else int(round(h))


class MapBlock(QtWidgets.QWidget):
    """
    A block of equal-shaped map panels (the simulator's energy maps) that wraps: partition by
    partition, the maps go into rows of `cols` columns, and `cols` is chosen for the largest maps
    that fit the block's region (`fit`); each partition may have a colour-bar cell at its right
    spanning its rows, or one bar serves the whole block at its right.  The cells are placed by
    the block itself (rule 11: the row of each visual line shares one band, Figure.align_row).
    `set_groups([(cells, bar_or_None), ...], shared_bar=None, extra=None, cell_bars=None,
    extra_side=None)` gives it the cells (the block takes them as children; the old ones are the
    window's to remove);
    `extra` is one more cell of a map's size (the simulator's spectrum) placed in the free slot of
    the last row, or in a row of its own when the last row is full (alone in the block before
    there are maps); it counts in the block's shape but not in the maps' room for titles and tick
    axes.  `extra_side` is a widget of fixed width beside the extra cell (the simulator's legend
    panel; taken only with `extra`): drawn one spacing right of the extra cell, at that row's top
    and height (the extra cell keeps a map's size), its fixed width counted in the block's width
    and in the maps' fit (_size_at), the colour bars placed right of it when it reaches past the
    maps' columns (_right); it sets no row's height and is not aligned with the row's panels.
    `cell_bars` ({cell: bar cell}) puts a narrow colour-bar cell right of each map named,
    inside the map's column (the simulator's 'cmap norm': every map its own scale and bar, as the
    sm4_Browser grid view).
    """

    def __init__(self, fig, spacing=4):
        super().__init__()
        self.fig = fig
        self._sp = spacing / (1000 / 72) * plt.em_px()
        self._groups = []             # [(cells, bar)]
        self._shared = None
        self._extra = None            # the extra cell (set_groups), one map's size at the first free slot
        self._extra_side = None       # the extra cell's side item (set_groups extra_side), right of it at its row's height
        self._cell_bars = {}          # {cell: bar cell} beside each map named (set_groups cell_bars)
        self._metrics_cache = None    # the cells' Qt-measured room (_metrics), kept for one layout pass (_invalidate)
        self._best_cache = {}         # _best per (W, H) within that pass
        self.cols = 1
        pol = QtWidgets.QSizePolicy.Policy
        self.setSizePolicy(pol.Expanding, pol.Expanding)

    def _invalidate(self):
        """The cells' measured room forgotten: measured again at the next size question (a new placement of the block: set_groups, a resize, rearrange, a layout request with a map's shape changed)."""
        self._metrics_cache = None
        self._best_cache = {}

    def _metrics(self):
        """
        The cells' room measured once per layout pass (Qt calls: the tick axes, titles and bar
        widths; _invalidate drops it): 'lr' the room left and right of a map's box, 'tb' the room
        above and below per partition, 'title_w' the widest title, 'bar_w' the partition / shared
        bars' width, 'cell_bar_w' a map's own bar's, 'ratio' a map's height over its width,
        'tb_cells' the room above and below of every cell per partition and 'tb_extra' the extra
        cell's (_row_tb: each row is as tall as its own cells ask), 'side_w' the extra cell's side
        item's fixed width (None without one: _side_w).  The
        arrangement asks the block its size hundreds of times in one pass (the candidates, their
        bisections): without this each question measured every cell again.
        """
        m = self._metrics_cache
        if m is None:
            m = {'lr': self._decor()[0], 'tb': [self._tb_of(cells) for cells, _ in self._groups],
                 'tb_cells': [[self._tb_of([c]) for c in cells] for cells, _ in self._groups],
                 'tb_extra': self._tb_of([self._extra]) if self._extra is not None else 0.0,
                 'title_w': self._title_w(), 'bar_w': self._bar_w(), 'cell_bar_w': self._cell_bar_w(),
                 'side_w': self._side_w(), 'ratio': self._ratio()}
            self._metrics_cache = m
        return m

    def set_groups(self, groups, shared_bar=None, extra=None, cell_bars=None, extra_side=None):
        self._invalidate()
        self._groups = [(list(cells), bar) for cells, bar in groups]
        self._shared = shared_bar
        self._extra = extra
        self._extra_side = extra_side if extra is not None else None   # a side item only beside an extra cell
        self._cell_bars = dict(cell_bars or {})
        for cells, bar in self._groups:
            for c in cells + ([bar] if bar is not None else []):
                c.setParent(self)
                c.show()
        for b in self._cell_bars.values():
            b.setParent(self)
            b.show()
        if shared_bar is not None:
            shared_bar.setParent(self)
            shared_bar.show()
        if extra is not None:
            extra.setParent(self)
            extra.show()
        if self._extra_side is not None:
            self._extra_side.setParent(self)
            self._extra_side.show()
        self.cols = max(1, max([len(c) for c, _ in self._groups] + [1]))
        self.updateGeometry()
        self._arrange()

    # --- the shape ---

    def _ratio(self):
        for cells, _ in self._groups:
            for c in cells:
                r = c.shape_ratio()
                if r is not None:
                    return float(r)
        return 1.0

    def _decor(self):
        """(left+right, top+bottom) the widest room around a map in the block (left+right is one for every column)."""
        lr = tb = 0.0
        for cells, _ in self._groups:
            for c in cells:
                l, t, r, b = c._decor()
                lr, tb = max(lr, l + r), max(tb, t + b)
        return lr, tb

    def _tb_of(self, cells):
        """top+bottom the widest among `cells` (a partition's own rows: a row without titles is no taller than its maps need)."""
        tb = 0.0
        for c in cells:
            l, t, r, b = c._decor()
            tb = max(tb, t + b)
        return tb

    def _bar_w(self):
        bars = [bar for _, bar in self._groups if bar is not None] + ([self._shared] if self._shared is not None else [])
        if not bars:
            return 0.0
        return float(max(b.sizeHint().width() for b in bars)) + self._sp

    def _cell_bar_w(self):
        """The room a map's own colour bar takes in its column (the widest bar cell plus the spacing); 0 without cell bars."""
        if not self._cell_bars:
            return 0.0
        return float(max(b.sizeHint().width() for b in self._cell_bars.values())) + self._sp

    def _side_w(self):
        """The extra cell's side item's fixed width (its minimumWidth, which QWidget.setFixedWidth sets); None without a side item."""
        if self._extra_side is None:
            return None
        return float(self._extra_side.minimumWidth())

    def _extra_col(self, n):
        """The extra cell's column at `n` columns: after the last partition's maps in its last row, 0 in a row of its own (or alone in the block)."""
        return len(self._groups[-1][0]) % n if self._groups else 0

    def _right(self, n, cw, cbw, side_w):
        """
        The right edge of the block's cells at `n` columns of `cw`-wide cells (each column with `cbw`
        for a map's own bar): the maps' columns, or the extra cell's side item's right edge (`side_w`
        wide, one spacing right of the extra cell; None without one) when it reaches past them.  The
        partition and shared colour bars go one spacing right of it, so they stay clear of the side item.
        """
        right = n * (cw + cbw) + (n - 1) * self._sp
        if side_w is not None:
            right = max(right, self._extra_col(n) * (cw + cbw + self._sp) + cw + self._sp + side_w)
        return right

    def _lines(self, k, cols):
        """The rows partition k takes at `cols` columns (the extra cell counted with the last partition)."""
        n = len(self._groups[k][0]) + (1 if self._extra is not None and k == len(self._groups) - 1 else 0)
        return -(-n // cols) if n else 0

    def _rows(self, cols):
        if not self._groups:
            return 1 if self._extra is not None else 0
        return sum(self._lines(k, cols) for k in range(len(self._groups)))

    def _title_w(self):
        """The widest map title: a map's cell is never narrower than its title."""
        tw = 0.0
        for cells, _ in self._groups:
            for c in cells:
                for o in c._bound:
                    t = getattr(o, '_title', None)
                    if t is not None and t.get_text():
                        try:
                            tw = max(tw, float(t.item.boundingRect().width()))
                        except Exception:
                            pass
        return tw

    def _row_tb(self, cols):
        """
        The room above and below each row at `cols` columns, in order: the widest among the row's
        own cells (the extra cell counted in the last partition's last row).  A row of maps alone
        is no taller than its maps need; the row holding the spectrum (its tick labels and label
        below) is taller, and only that row (rule 11: the row of each visual line shares one band).
        """
        m = self._metrics()
        out = []
        for k, (cells, _) in enumerate(self._groups):
            tbs = list(m['tb_cells'][k]) + ([m['tb_extra']] if self._extra is not None and k == len(self._groups) - 1 else [])
            for i in range(0, len(tbs), cols):
                out.append(max(tbs[i:i + cols]))
        if not self._groups and self._extra is not None:
            out.append(m['tb_extra'])
        return out

    def _size_at(self, cols, W, H):
        """(w, h, box_w) of the block at `cols` columns inside W x H: the largest maps that fit by width (the extra cell's side item counted), then by height."""
        n = max(1, min(cols, max([len(c) for c, _ in self._groups] + [1])))
        R = self._rows(n)
        if R == 0:
            return (1.0, 1.0, 0.0)
        m = self._metrics()
        lr, bar_w, cbw = m['lr'], m['bar_w'], m['cell_bar_w']   # cbw: a map's own colour bar beside it, in its column
        sw = m['side_w']
        r = m['ratio']
        # Every row carries its own room (_row_tb: titles on the first partition's rows, the spectrum's labels on its row).
        tb_sum = sum(self._row_tb(n))
        cw = (W - bar_w - (n - 1) * self._sp) / n - cbw
        if sw is not None:
            # The side item right of the extra cell (in its column c) inside W as well, the bars right of it (_right).
            c = self._extra_col(n)
            cw = min(cw, (W - bar_w - self._sp - sw - c * (cbw + self._sp)) / (c + 1))
        box_w = max(cw - lr, 1.0)
        box_h = box_w * r
        total = R * box_h + tb_sum + (R - 1) * self._sp
        if total > H:
            box_h = max((H - (R - 1) * self._sp - tb_sum) / R, 1.0)
            box_w = box_h / r
        cw = max(box_w + lr, m['title_w'] + 4.0 / (1000 / 72) * plt.em_px())                 # a cell at least as wide as its title
        w = self._right(n, cw, cbw, sw) + bar_w
        h = R * (box_w * r) + tb_sum + (R - 1) * self._sp
        return (w, h, box_w)

    def _best(self, W, H):
        """(w, h, box_w, cols) inside W x H over every column count: the largest maps whose block fits the width (memoized per (W, H) within a layout pass)."""
        key = (round(float(W), 2), round(float(H), 2))
        hit = self._best_cache.get(key)
        if hit is not None:
            return hit
        n_max = max([len(c) for c, _ in self._groups] + [1])
        best = None
        for n in range(1, n_max + 1):
            w, h, box_w = self._size_at(n, W, H)
            if w > W + 0.5 and n > 1:
                continue                                 # the titles do not fit side by side at this count
            if best is None or box_w > best[2] + 0.5:
                best = (w, h, box_w, n)
        if len(self._best_cache) > 2048:
            self._best_cache = {}
        self._best_cache[key] = best
        return best

    def fit(self, W, H):
        """The block's size (w, h) inside W x H with the columns chosen for the largest maps (no side effect: the block chooses again for the size it is given)."""
        if not self._groups and self._extra is None:
            return (1.0, 1.0)
        w, h, box_w, n = self._best(W, H)
        return (w, h)

    def map_box(self, W, H):
        """One map's plot box (w, h) when the block is given W x H (the columns chosen as fit chooses them); None without maps."""
        if not self._groups:
            return None
        w, h, box_w, n = self._best(W, H)
        return box_w, box_w * self._metrics()['ratio']

    def _has_cells(self):
        return bool(self._groups) or self._extra is not None

    def height_for_width(self, w):
        return self._best(w, 1.0e9)[1] if self._has_cells() else None

    def width_for_height(self, h):
        return self._best(1.0e9, h)[0] if self._has_cells() else None

    def hasHeightForWidth(self):
        return self._has_cells()

    def heightForWidth(self, w):
        h = self.height_for_width(w)
        return -1 if h is None else int(round(h))

    # --- the placement ---

    def _arrange(self):
        if not self._has_cells() or self.width() <= 0 or self.height() <= 0:
            return
        W, H = float(self.width()), float(self.height())
        self._invalidate()                                       # a placement: the cells measured anew
        self._last_shapes = _panel_shapes(self)                  # a layout request places again only when it changes
        w, h, box_w, n = self._best(W, H)
        self.cols = n
        m = self._metrics()
        lr = m['lr']
        r = m['ratio']
        cw = max(box_w + lr, m['title_w'] + 4.0 / (1000 / 72) * plt.em_px())
        cbw = m['cell_bar_w']
        bar_w = m['bar_w'] - self._sp if m['bar_w'] > 0 else 0.0
        sw = m['side_w']
        bx = self._right(n, cw, cbw, sw) + self._sp              # the colour bars' left: right of the maps and of the side item
        y = 0.0
        rows_for_align = []
        groups = self._groups or [([], None)]
        row_tb = self._row_tb(n)                                 # each row's own room above and below
        line0 = 0                                                # the first row of this partition in row_tb
        for k, (cells, bar) in enumerate(groups):
            # The extra cell after the last partition's maps: the free slot of its last row, or a new row.
            placed = list(cells) + ([self._extra] if self._extra is not None and k == len(groups) - 1 else [])
            y0 = y
            lines = -(-len(placed) // n)
            heights = [box_w * r + row_tb[line0 + j] for j in range(lines)]   # this partition's row heights
            tops = [y0 + sum(heights[:j]) + j * self._sp for j in range(lines)]
            row = []
            for i, c in enumerate(placed):
                col, line = i % n, i // n
                if col == 0 and i > 0:
                    rows_for_align.append(row)
                    row = []
                x = col * (cw + cbw + self._sp)
                cy, ch = tops[line], heights[line]
                c.setGeometry(int(round(x)), int(round(cy)), int(round(cw)), int(round(ch)))
                row.append(c)
                b = self._cell_bars.get(c)
                if b is not None:                                # the map's own colour bar, right of it
                    b.setGeometry(int(round(x + cw + self._sp)), int(round(cy)), int(round(max(cbw - self._sp, 1.0))), int(round(ch)))
                    row.append(b)
                if c is self._extra and sw is not None:          # the extra cell's side item, right of it at the row's top and height
                    self._extra_side.setGeometry(int(round(x + cw + self._sp)), int(round(cy)), int(round(sw)), int(round(ch)))
            if row:
                rows_for_align.append(row)
            line0 += lines
            gh = sum(heights) + (lines - 1) * self._sp
            if bar is not None:
                bar.setGeometry(int(round(bx)), int(round(y0)), int(round(bar_w)), int(round(gh)))
            y = y0 + gh + self._sp
        if self._shared is not None:
            self._shared.setGeometry(int(round(bx)), 0, int(round(bar_w)), int(round(max(y - self._sp, 1.0))))
        for row in rows_for_align:
            axes = [o for c in row for o in c._bound if isinstance(o, plt.Axes)]
            if len(axes) > 1:
                self.fig.align_row(*axes)

    def rearrange(self):
        self._arrange()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._arrange()

    def event(self, ev):
        if ev.type() == QtCore.QEvent.Type.LayoutRequest and _panel_shapes(self) != getattr(self, '_last_shapes', None):
            # A cell's layout request places the block again only when a map's shape changed (a turn, a
            # crop): a title, a colour bar's or a tick axis's text changes nothing (the cells stay).
            self._arrange()
        return super().event(ev)

    def minimumSizeHint(self):
        return QtCore.QSize(round(120 / (1000 / 72) * plt.em_px()), round(80 / (1000 / 72) * plt.em_px()))

    def sizeHint(self):
        return QtCore.QSize(round(600 / (1000 / 72) * plt.em_px()), round(300 / (1000 / 72) * plt.em_px()))


class Arrangement(QtWidgets.QWidget):
    """
    The rule of every window with spatial panels (rule 11 of the guide, as the owner stated it):
    the plots fill the window, spatial panels keep their ratio, and the heat-map data comes first:
    the shaped groups are sized as large as the window allows, so the blank space is what they
    leave; the curve panels (the spectrum) never grow at their expense, yet stay readable at any
    size.  `shaped` is one or two shaped groups (a CropPanel or PanelWithBox, a RowWidget, a
    MapBlock: each tells its size inside a region, _shaped_size), `free` the widget of the curve
    panels.  Of every candidate's room the free widget is given only its readable minimum
    (`free_min()`, (w, h) px, and its own minimum) before the groups are sized; it then takes the
    rest of its slot up to `free_cap(rects)` ((max w, max h) for the groups' rects in that
    candidate, never below the readable minimum; the simulator sets none: its curve panels take
    the room the groups leave), at the top left of its slot, and the part of the slot it leaves
    counts as blank.  With `free_scored` (the simulator's Single Point view) the free widget is a
    first-class panel instead: it is scored as a group (its scale against the whole window) and
    the one group and it are sized to one scale.  `group_cap` (A, B, k, box_a, box_b): group A's
    plot box never more than k times group B's in width or in height (box_a(rect), box_b(rect):
    their plot boxes (w, h) in a region; the simulator: the topographies at most twice a heatmap);
    a candidate that breaks it is sized again with A limited (_candidates).  `fill_slot`: a shaped
    group (the simulator's Line / Path heatmap row) sized and scored as every group, then placed in
    its whole slot of the chosen candidate instead of its shaped size at the slot's top left
    (_slot), its fill panels stretching to it (qtplot.PanelCell.fill_as).  `best_score()` is
    the best candidate's score at the widget's size, nothing placed (the simulator compares its
    unit-cell panels' two orientations with it).  The candidate arrangements for two groups A, B and the free widget F:

        row_top   A and B in one row across the top, F below
        a_left    A at the left with the window's height; B over F at its right
        a_top     A across the top; B at the left of F below it
        ab_left   A over B at the left; F at the right
        abf_row   A, B and F side by side
        abf_col   A, B and F stacked

    and for one group: `h` (A at the left, F at the right) and `v` (A on top, F below).  Each is
    sized at the widget's size and scored: every group's scale is its size against the size it
    would have alone in the whole window (the square root of the area ratio), and the score is the
    smallest scale less the blank fraction, so that no group is squeezed for another and blank
    space costs; the best score is taken (the current arrangement kept unless another scores
    better by more than `hysteresis`; ties go to the larger plotted area).  Chosen again on every
    resize, on `rearrange()` (a window that rebuilt a group) and when a group's shape changes (its
    layout request).  `fit_delta` is the blank strip at the right and bottom edges that no region
    reaches (Figure.fit_window, which nothing calls: the window is never trimmed to its panels);
    `layout_name` the candidate taken;
    `blank_fraction()` the blank share of the area; `scales()` the groups' scales.
    """

    def __init__(self, shaped, free=None, spacing=8, hysteresis=0.05, free_min=None, free_cap=None,
                 free_scored=False, group_cap=None, fill_slot=None):
        super().__init__()
        self.shaped = [w for w in shaped if w is not None]
        self.free = free
        self._fill_slot = fill_slot       # the group placed in its whole slot once a candidate is chosen (_slot)
        for w in self.shaped + ([free] if free is not None else []):
            w.setParent(self)
            w.show()
            w.installEventFilter(self)
        self._sp, self._hys = spacing / (1000 / 72) * plt.em_px(), float(hysteresis)
        self._free_min_fn, self._free_cap_fn, self._free_scored = free_min, free_cap, bool(free_scored)
        self._group_cap = group_cap       # (A, B, k, box_a, box_b): group A's plot box at most k times group B's
        self._lim = {}                    # {group: (max w, max h)} while a candidate is sized under group_cap
        self.layout_name = None
        self._placed = None               # {widget: QRectF} of the last arrangement
        self._blank = 0.0
        self._cb = None
        self._pending = False
        pol = QtWidgets.QSizePolicy.Policy
        self.setSizePolicy(pol.Expanding, pol.Expanding)

    # --- events ---

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.LayoutRequest and not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self._later)
        return False

    def _later(self):
        self._pending = False
        # A child's layout request arranges again only when a group's shape changed (a crop): a label's
        # width, a tick value or a colour bar's label changes nothing (the panels stay where they are).
        # The window's text zoom, control zoom or portrait setting changed since the last arrangement
        # arranges again too (they follow a resize: the figure applies them after this widget's
        # resizeEvent, Figure._relayout).
        if self._shapes() != getattr(self, '_last_shapes', None) or self._window_state() != getattr(self, '_last_state', None):
            self._arrange()

    def _shapes(self):
        """The groups' panel shapes (_panel_shapes of each shaped group)."""
        return tuple(_panel_shapes(w) for w in self.shaped)

    def _window_state(self):
        """The text zoom, control zoom and portrait setting of the figure whose canvas holds this widget (font_zoom, control_zoom, portrait), None outside a figure's canvas."""
        w = self.parentWidget()
        while w is not None and not isinstance(w, QtWidgets.QGraphicsView):
            w = w.parentWidget()
        fig = getattr(w, 'figure', None)
        if fig is None:
            return None
        return (getattr(fig, 'font_zoom', None), getattr(fig, 'control_zoom', None), getattr(fig, 'portrait', None))

    def on_arranged(self, callback):
        """callback(layout_name) after every arrangement."""
        self._cb = callback

    def rearrange(self):
        self._arrange()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._arrange()
        if not self._pending:
            # The figure applies a new text zoom, control zoom or portrait setting after this (Figure._relayout):
            # arranged again then when one changed (_later).
            self._pending = True
            QtCore.QTimer.singleShot(0, self._later)

    def showEvent(self, ev):
        super().showEvent(ev)
        self._arrange()

    def event(self, ev):
        if ev.type() == QtCore.QEvent.Type.LayoutRequest and not self._pending:
            # The requests of one event-loop turn (every child placed posts one) make one arrangement.
            self._pending = True
            QtCore.QTimer.singleShot(0, self._later)
        return super().event(ev)

    def minimumSizeHint(self):
        return QtCore.QSize(round(200 / (1000 / 72) * plt.em_px()), round(150 / (1000 / 72) * plt.em_px()))

    def sizeHint(self):
        return QtCore.QSize(round(1000 / (1000 / 72) * plt.em_px()), round(700 / (1000 / 72) * plt.em_px()))

    # --- the candidates ---

    def _free_min(self, W, H):
        """The free widget's readable minimum (free_min(), and its own minimum size hint): all the room it is given before the groups are sized."""
        if self.free is None:
            return 0.0, 0.0
        m = self.free.minimumSizeHint()
        rw, rh = self._free_min_fn() if self._free_min_fn is not None else (0.0, 0.0)
        return max(float(m.width()), float(rw)), max(float(m.height()), float(rh))

    def _capped(self, rects, blank, fw, fh):
        """The free widget's region cut to free_cap (never below the readable minimum fw x fh), at the top left of its slot; the part of the slot it leaves is added to the blank."""
        F = self.free
        if F is None or F not in rects or self._free_cap_fn is None:
            return rects, blank
        cap = self._free_cap_fn({w: r for w, r in rects.items() if w is not F})
        if cap is None:
            return rects, blank
        x, y, w, h = rects[F]
        cw = min(w, max(float(cap[0]), fw))
        ch = min(h, max(float(cap[1]), fh))
        rects = dict(rects)
        rects[F] = (x, y, cw, ch)
        return rects, blank + w * h - cw * ch

    def _balanced(self, A, horizontal, W, H, fw, fh):
        """
        free_scored, one group: the group's size and the free widget's slot at one scale (the
        group's area against its size alone, the free widget's against the whole window), by
        bisection on the room the group may take; (a, slot) or None.  `horizontal`: the group at
        the left and the free widget at its right, else the group on top.
        """
        sp = self._sp
        solo = self._sz(A, W, H)
        if not solo:
            return None
        s_area = solo[0] * solo[1]
        lo, hi = 1.0, (W - fw - sp) if horizontal else (H - fh - sp)
        best = None
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            a = self._sz(A, mid, H) if horizontal else self._sz(A, W, mid)
            if not a:
                return best
            slot = (W - a[0] - sp, H) if horizontal else (W, H - a[1] - sp)
            sa = (a[0] * a[1] / s_area) ** 0.5
            sf = (max(slot[0], 0.0) * max(slot[1], 0.0) / (W * H)) ** 0.5
            best = (a, slot)
            if sa < sf:
                lo = mid
            else:
                hi = mid
        return best

    def _row_shared(self, A, B, W, H):
        """A and B in one row of width W, height at most H: the largest common height whose widths fit; ((wA, hA), (wB, hB))."""
        lo, hi = 1.0, float(H)
        best = None
        for _ in range(16):                                  # to 1/65536 of the range: well under a pixel
            h = 0.5 * (lo + hi)
            a, b = self._sz(A, W, h), self._sz(B, W, h)
            if a is None or b is None:
                return None
            if a[0] + b[0] + self._sp <= W:
                best, lo = (a, b), h
            else:
                hi = h
        return best

    def _col_shared(self, A, B, W, H):
        """A over B in one column of height H, width at most W: the largest common width whose heights fit."""
        lo, hi = 1.0, float(W)
        best = None
        for _ in range(16):
            w = 0.5 * (lo + hi)
            a, b = self._sz(A, w, H), self._sz(B, w, H)
            if a is None or b is None:
                return None
            if a[1] + b[1] + self._sp <= H:
                best, lo = (a, b), w
            else:
                hi = w
        return best

    def _sz(self, w, W, H):
        """_shaped_size of a group inside W x H, within its limit while one is set (_lim, group_cap)."""
        lim = self._lim.get(w)
        if lim is not None:
            W, H = min(float(W), lim[0]), min(float(H), lim[1])
        return _shaped_size(w, W, H)

    def _within(self, cand):
        """Whether a candidate keeps group_cap: group A's plot box at most k times group B's in every dimension (box_a, box_b of their rects give (w, h), a side None when it is not known; a box not known yet counts as kept)."""
        A, B, k, box_a, box_b = self._group_cap
        rects = cand[2]
        if A not in rects or B not in rects:
            return True
        a, b = box_a(rects[A]), box_b(rects[B])
        if a is None or b is None:
            return True
        return all(x is None or y is None or x <= k * y + 0.5 for x, y in zip(a, b))

    def _candidates(self, W, H):
        """
        [(name, blank, {widget: (x, y, w, h)}, valid)] at the size W x H (_candidates_at); with
        group_cap, a candidate whose group A is more than k times group B (their plot boxes) is
        sized again with A limited to the largest share of its size that keeps the cap (a
        bisection of 7 steps on that candidate alone; the room A gives up goes to B and the free
        widget as the candidate's geometry gives it).
        """
        self._lim = {}
        fmin = self._free_min(W, H)                 # measured once per search (Qt calls behind it)
        base = self._candidates_at(W, H, fmin=fmin)
        if self._group_cap is None or len(self.shaped) < 2:
            return base
        A = self._group_cap[0]
        out = []
        for cand in base:
            if self._within(cand):
                out.append(cand)
                continue
            name, ra = cand[0], cand[2][A]
            lo, hi, best = 0.05, 1.0, None
            for _ in range(7):
                f = 0.5 * (lo + hi)
                self._lim = {A: (ra[2] * f, ra[3] * f)}
                c = next((c for c in self._candidates_at(W, H, only=name, fmin=fmin) if c[0] == name), None)
                if c is not None and self._within(c):
                    best, lo = c, f
                else:
                    hi = f
            self._lim = {}
            out.append(best if best is not None else cand)
        return out

    def _candidates_at(self, W, H, only=None, fmin=None):
        """[(name, blank, {widget: (x, y, w, h)}, valid)] at the size W x H, the groups within their limits (_lim); `only`: that candidate alone (two groups); `fmin`: the free widget's minimum, measured by the caller."""
        want = lambda name: only is None or only == name
        sp = self._sp
        fw, fh = self._free_min(W, H) if fmin is None else fmin
        F = self.free
        out = []
        R = QtCore.QRectF

        def put(name, rects, blank, valid=True):
            rects, blank = self._capped(rects, blank, fw, fh)
            out.append((name, float(blank), rects, bool(valid)))

        if len(self.shaped) == 1:
            A = self.shaped[0]
            if F is None:
                a = self._sz(A, W, H)
                if a:
                    put('a', {A: (0, 0, a[0], a[1])}, W * H - a[0] * a[1])
                return out
            if self._free_scored:                                 # the group and the free widget at one scale
                b = self._balanced(A, True, W, H, fw, fh)
                a = b[0] if b else None
            else:
                a = self._sz(A, W - fw - sp, H)
            if a:                                                 # h: A at the left
                wA = a[0]
                put('h', {A: (0, 0, a[0], a[1]), F: (wA + sp, 0, W - wA - sp, H)}, wA * (H - a[1]),
                    W - wA - sp >= fw - 1)
            if self._free_scored:
                b = self._balanced(A, False, W, H, fw, fh)
                a = b[0] if b else None
            else:
                a = self._sz(A, W, H - fh - sp)
            if a:                                                 # v: A on top
                hA = a[1]
                put('v', {A: (0, 0, a[0], a[1]), F: (0, hA + sp, W, H - hA - sp)}, hA * (W - a[0]),
                    H - hA - sp >= fh - 1)
            return out
        A, B = self.shaped[0], self.shaped[1]
        if F is None:
            fw = fh = 0.0
        # row_top
        rs = self._row_shared(A, B, W, H - fh - sp if F is not None else H) if want('row_top') else None
        if rs:
            (wA, hA), (wB, hB) = rs
            row_h = max(hA, hB)
            rects = {A: (0, 0, wA, hA), B: (wA + sp, 0, wB, hB)}
            if F is not None:
                rects[F] = (0, row_h + sp, W, H - row_h - sp)
            put('row_top', rects, W * row_h - wA * hA - wB * hB, F is None or H - row_h - sp >= fh - 1)
        # a_left
        a = self._sz(A, W - fw - sp, H) if want('a_left') else None
        if a:
            wA, hA = a
            Wr = W - wA - sp
            b = self._sz(B, Wr, H - fh - sp if F is not None else H)
            if b:
                wB, hB = b
                rects = {A: (0, 0, wA, hA), B: (wA + sp, 0, wB, hB)}
                if F is not None:
                    rects[F] = (wA + sp, hB + sp, Wr, H - hB - sp)
                put('a_left', rects, wA * (H - hA) + hB * (Wr - wB), F is None or (Wr >= fw - 1 and H - hB - sp >= fh - 1))
        # a_top
        a = self._sz(A, W, H - fh - sp) if want('a_top') else None
        if a:
            wA, hA = a
            Hb = H - hA - sp
            b = self._sz(B, W - fw - sp if F is not None else W, Hb)
            if b:
                wB, hB = b
                rects = {A: (0, 0, wA, hA), B: (0, hA + sp, wB, hB)}
                if F is not None:
                    rects[F] = (wB + sp, hA + sp, W - wB - sp, Hb)
                put('a_top', rects, hA * (W - wA) + wB * (Hb - hB), F is None or (W - wB - sp >= fw - 1 and Hb >= fh - 1))
        # ab_left
        cs = self._col_shared(A, B, W - fw - sp if F is not None else W, H) if want('ab_left') else None
        if cs:
            (wA, hA), (wB, hB) = cs
            col_w = max(wA, wB)
            rects = {A: (0, 0, wA, hA), B: (0, hA + sp, wB, hB)}
            if F is not None:
                rects[F] = (col_w + sp, 0, W - col_w - sp, H)
            put('ab_left', rects, col_w * H - wA * hA - wB * hB, F is None or W - col_w - sp >= fw - 1)
        if F is not None:
            # abf_row
            a = self._sz(A, W - fw - 2 * sp, H) if want('abf_row') else None
            if a:
                wA, hA = a
                b = self._sz(B, W - wA - fw - 2 * sp, H)
                if b:
                    wB, hB = b
                    x = wA + sp + wB + sp
                    put('abf_row', {A: (0, 0, wA, hA), B: (wA + sp, 0, wB, hB), F: (x, 0, W - x, H)},
                        wA * (H - hA) + wB * (H - hB), W - x >= fw - 1)
            # abf_col
            a = self._sz(A, W, H - fh - 2 * sp) if want('abf_col') else None
            if a:
                wA, hA = a
                b = self._sz(B, W, H - hA - fh - 2 * sp)
                if b:
                    wB, hB = b
                    y = hA + sp + hB + sp
                    put('abf_col', {A: (0, 0, wA, hA), B: (0, hA + sp, wB, hB), F: (0, y, W, H - y)},
                        hA * (W - wA) + hB * (W - wB), H - y >= fh - 1)
        return out

    def _score(self, cand, solo, W, H):
        """(score, plotted area) of a candidate: the smallest group scale (with free_scored the free widget's too, against the whole window) less the blank fraction."""
        name, blank, rects, _ = cand
        scales, area = [], 0.0
        for w, s_area in zip(self.shaped, solo):
            r = rects.get(w)
            a = r[2] * r[3] if r else 0.0
            area += a
            scales.append(min(1.0, (a / s_area) ** 0.5) if s_area > 0 else 0.0)
        if self._free_scored and self.free is not None and self.free in rects:
            r = rects[self.free]
            scales.append(min(1.0, (r[2] * r[3] / (W * H)) ** 0.5))
        return (min(scales) if scales else 0.0) - blank / (W * H), area, scales

    def _scored(self, W, H):
        """[((score, area, scales), candidate)] of the valid candidates at W x H (every candidate when none is valid)."""
        cands = self._candidates(W, H)
        if not cands:
            return []
        valid = [c for c in cands if c[3]] or cands
        solo = []
        for w in self.shaped:
            s = _shaped_size(w, W, H)
            solo.append(s[0] * s[1] if s else 0.0)
        return [(self._score(c, solo, W, H), c) for c in valid]

    def best_score(self):
        """(score, name) of the best candidate at the widget's size now, nothing placed; None before the widget has a size."""
        W, H = float(self.width()), float(self.height())
        if W <= 0 or H <= 0 or not self.shaped:
            return None
        plt.new_measure_pass()
        scored = self._scored(W, H)
        if not scored:
            return None
        best = max(scored, key=lambda t: (round(t[0][0], 3), t[0][1]))
        return float(best[0][0]), best[1][0]

    def _choose(self, W, H):
        scored = self._scored(W, H)
        if not scored:
            return None
        best = max(scored, key=lambda t: (round(t[0][0], 3), t[0][1]))
        cur = next((t for t in scored if t[1][0] == self.layout_name), None)
        if cur is not None and cur is not best and cur[0][0] >= best[0][0] - self._hys:
            best = cur
        self._scales = best[0][2]
        return best[1]

    def scales(self):
        """The groups' scales in the current arrangement (1.0: as large as alone in the window)."""
        return list(getattr(self, '_scales', []))

    def explain(self):
        """[(name, score, blank fraction, scales, valid)] of every candidate at the current size, best first (a check's diagnostic)."""
        W, H = float(self.width()), float(self.height())
        if W <= 0 or H <= 0 or not self.shaped:
            return []
        solo = []
        for w in self.shaped:
            s = _shaped_size(w, W, H)
            solo.append(s[0] * s[1] if s else 0.0)
        out = []
        for c in self._candidates(W, H):
            score, area, scales = self._score(c, solo, W, H)
            out.append((c[0], round(score, 3), round(c[1] / (W * H), 3), [round(v, 2) for v in scales], c[3]))
        return sorted(out, key=lambda t: -t[1])

    def _arrange(self):
        W, H = float(self.width()), float(self.height())
        if W <= 0 or H <= 0 or not self.shaped:
            return
        plt.new_arrangement()
        # The same size, the same group shapes and the same free minimum as the last arrangement give the
        # same arrangement: it is applied again without the search (rearrange() or a show at the same size).
        sig = (round(W, 1), round(H, 1), tuple(_shaped_size(w, W, H) for w in self.shaped), self._free_min(W, H),
               self._free_scored, self.layout_name)
        if self._placed and sig == getattr(self, '_last_sig', None):
            rects = self._placed
        else:
            best = self._choose(W, H)
            if best is None:
                return
            name, blank, rects, _ = best
            self.layout_name, self._blank, self._placed = name, blank, rects
            self._last_sig = (sig[0], sig[1], sig[2], sig[3], sig[4], name)
        self._last_shapes = self._shapes()          # a child's layout request arranges again only when they change
        self._last_state = self._window_state()     # arranged again when the window's text zoom, control zoom or portrait setting changes (_later)
        for w, (x, y, ww, hh) in rects.items():
            if w is self._fill_slot:
                # fill_slot: sized and scored at its shaped size, placed in its whole slot.
                s = self._slot(self.layout_name, rects, W, H, w)
                if s is not None:
                    x, y, ww, hh = s
            w.setGeometry(int(round(x)), int(round(y)), max(int(round(ww)), 1), max(int(round(hh)), 1))
        if self._cb is not None:
            self._cb(self.layout_name)

    def _slot(self, name, rects, W, H, w):
        """
        Group `w`'s whole slot (x, y, w, h) in candidate `name` (rects: its regions): the part of the
        widget the candidate's geometry gives the group, whose top left its shaped size takes; it ends
        at the other group's or the free widget's region (less the spacing), else at the widget's
        edge, so it never takes room from the other group's slot or the free widget's.  None (the
        group keeps its shaped rect) without two groups and a free widget, or while the group has no
        shape (a row with no panel in it yet).
        """
        F = self.free
        if len(self.shaped) != 2 or w not in self.shaped or F is None or F not in rects:
            return None
        hfw = getattr(w, 'height_for_width', None)
        if hfw is None or hfw(float(rects[w][2])) is None:
            return None
        A, B = self.shaped
        sp = self._sp
        a, b, f = rects[A], rects[B], rects[F]
        if w is A:
            ends = {'row_top': (b[0] - sp, f[1] - sp), 'a_left': (b[0] - sp, H), 'a_top': (W, b[1] - sp),
                    'ab_left': (f[0] - sp, b[1] - sp), 'abf_row': (b[0] - sp, H), 'abf_col': (W, b[1] - sp)}
        else:
            ends = {'row_top': (W, f[1] - sp), 'a_left': (W, f[1] - sp), 'a_top': (f[0] - sp, H),
                    'ab_left': (f[0] - sp, H), 'abf_row': (f[0] - sp, H), 'abf_col': (W, f[1] - sp)}
        if name not in ends:
            return None
        x, y, sw, sh = rects[w]
        right, bottom = ends[name]
        return (x, y, max(right - x, sw), max(bottom - y, sh))

    # --- what the window reads ---

    def blank_fraction(self):
        """The blank share of this widget's area in the current arrangement (the free widget counts as filled)."""
        W, H = float(self.width()), float(self.height())
        return 0.0 if W <= 0 or H <= 0 else self._blank / (W * H)

    def fit_delta(self):
        """
        (dw, dh), each <= 0 px: the strip at the right and the strip at the bottom that no region
        reaches (the free widget's region counts as reached: the curves fill it), which the window
        can lose.  (0, 0) before the first arrangement.
        """
        if not self._placed:
            return (0, 0)
        W, H = float(self.width()), float(self.height())
        right = max(x + ww for (x, y, ww, hh) in self._placed.values())
        bottom = max(y + hh for (x, y, ww, hh) in self._placed.values())
        return (min(0, int(round(right - W))), min(0, int(round(bottom - H))))
