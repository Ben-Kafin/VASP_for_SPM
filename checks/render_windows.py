"""
The render-and-measure helper of every converted window (plan 3.6): a window opened offscreen
is resized to its design size, half, double and a shape taller than wide, and at each size the
layer's rules are measured: the text zoom follows the one rule (qtplot.window_zoom), every
control's text fits its widget (no text cut), no two controls overlap, no control enters a
panel's box, and in portrait every label with a short form shows it.  A screenshot of every size
goes to `shots_dir` for a look by eye.

    from checks import render_windows as rw
    report = rw.check_window(fig, 'simulator', shots_dir)     # fig: an open qtplot Figure
    rw.print_report(report); assert not report['problems']

Run with QT_QPA_PLATFORM=offscreen (the module sets it when nothing is set).  check_window
leaves the window at its design size.
"""

import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qtgui import qtplot as plt                                   # noqa: E402
from qtgui.qtplot import QtCore, QtGui, QtWidgets                 # noqa: E402
from qtgui import widgets                                         # noqa: E402

SIZES = (('design', 1.0, 1.0), ('half', 0.5, 0.5), ('double', 2.0, 2.0), ('portrait', 0.45, 1.0))


def pump(ms=120):
    """Run the event loop for a moment (the layout passes, the info dock's timer)."""
    from PySide6.QtTest import QTest
    QTest.qWait(ms)


def resize(fig, w_in, h_in):
    """
    The figure at w_in x h_in inches, applied twice (the toolbar and the info dock take window
    height first), then the moment the window's own fit takes after a resize (rule 11: a window of
    shaped panels follows them; its size is then its own, not the one asked for).
    """
    for _ in range(2):
        fig.set_size_inches(w_in, h_in)
        pump()
    pump(350)


def control_rects(fig):
    """(name, QRect) of every visible control of the figure, in canvas pixels."""
    vp = fig.canvas.viewport()
    out = []
    for c in fig._qt_controls:
        w = getattr(c, 'qt', None)
        if w is None or not w.isVisible():
            continue
        p = w.mapTo(vp, QtCore.QPoint(0, 0))
        out.append((c.name, QtCore.QRect(p, w.size())))
    return out


def text_cut(fig):
    """[(text, needed px, width px)] of every visible label, check box, radio or button whose text is wider than it."""
    bad = []
    vp = fig.canvas.viewport()
    for w in vp.findChildren(QtWidgets.QWidget):
        if not isinstance(w, (QtWidgets.QLabel, QtWidgets.QCheckBox, QtWidgets.QRadioButton, QtWidgets.QPushButton)):
            continue
        if not w.isVisible() or not w.text():
            continue
        if isinstance(w, QtWidgets.QLabel) and w.wordWrap():
            need_h = w.heightForWidth(w.width())
            if need_h > w.height() + 1:
                bad.append((w.text(), need_h, w.height()))
            continue
        fm = QtGui.QFontMetrics(w.font())
        extra = 0 if isinstance(w, QtWidgets.QLabel) else (22 if not isinstance(w, QtWidgets.QPushButton) else 10)
        need = fm.horizontalAdvance(w.text()) + extra
        if need > w.width() + 2:
            bad.append((w.text(), need, w.width()))
    return bad


def overlaps(fig):
    """Pairs of controls whose rectangles intersect (neither inside the other), and controls inside a panel's box."""
    rects = control_rects(fig)
    out = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i][1], rects[j][1]
            if a.intersects(b) and not (a.contains(b) or b.contains(a)):
                out.append((rects[i][0], rects[j][0]))
    for name, r in rects:
        rf = QtCore.QRectF(r)
        for ax in fig.axes:
            if not ax._visible:
                continue
            x0, y0, w, h = ax._px_rect()
            box = QtCore.QRectF(x0, y0, w, h)
            inter = box.intersected(rf)
            if inter.width() > 2 and inter.height() > 2:
                out.append((name, 'inside panel ' + (ax.get_title() or f'#{fig.axes.index(ax)}')))
    return out


def text_rule(fig, factor, requested=None):
    """
    The fonts follow the one rule at this size factor: the figure's zoom, the tick fonts, the
    titles, the controls.  `requested` is the canvas size (W, H) px the window was asked for: a
    window that sized itself to its shaped panels instead (rule 11, fitted_to_panels) is at its own
    zoom, which the fonts must follow all the same.
    """
    bad = []
    W, H = fig._canvas_size()
    z = plt.window_zoom(W, H, *fig._design_px)
    want = min(max(factor, plt.ZOOM_MIN), plt.ZOOM_MAX)
    if abs(z - want) > 0.06 and not (requested is not None and fitted_to_panels(fig, requested)):
        bad.append(('zoom', factor, (W, H), fig._design_px, z))
    if abs(fig.font_zoom - z) > 0.03 * z + 1e-9:
        bad.append(('font_zoom', fig.font_zoom, z))
    z = fig.font_zoom
    for ax in fig.axes:
        if not ax._visible:
            continue
        for item in (ax.xaxis_item, ax.yaxis_item):
            tf = item.style.get('tickFont')
            if tf is None or tf.pixelSize() != plt.qfont(item._tick_pt, zoom=z).pixelSize():
                bad.append(('tick font', ax.get_title()[:30], tf and tf.pixelSize()))
        if ax._title is not None:
            got, want_px = ax._title.item.textItem.font().pixelSize(), plt.qfont(ax._title._size, zoom=z).pixelSize()
            if got != want_px:
                bad.append(('title font', ax._title_full[:40], got, want_px))
    for c in fig._qt_controls:
        if getattr(c, 'qt', None) is None or not c._visible:
            continue
        base = widgets._font(c.spec['fs'], z).pixelSize()
        for w in [c.qt] + c.qt.findChildren(QtWidgets.QWidget):
            if not isinstance(w, (QtWidgets.QLabel, QtWidgets.QCheckBox, QtWidgets.QRadioButton, QtWidgets.QPushButton,
                                  QtWidgets.QLineEdit)) or not w.isVisible():
                continue
            px = w.font().pixelSize()
            if not (0.75 * base - 1.0 <= px <= base + 1.0):
                bad.append(('control font', c.name, getattr(w, 'text', lambda: '')(), px, base))
    return bad


BAND_PX = 10          # the empty band a shaped panel may leave in its cell (rounding of the ticks' room)


def _in_main_pane(widget):
    """Whether a widget sits in the main pane of a controls.OrientedPanes (marked `_pane_main`)."""
    w = widget
    while w is not None:
        if getattr(w, '_pane_main', False):
            return True
        w = w.parentWidget()
    return False


def empty_bands(fig):
    """
    Rule 11: [(title, band px)] of every visible shaped panel (an aspect-locked or box-aspect axes
    in a layout cell) whose cell is taller than the panel's box plus its ticks and title (the row's
    tallest, in an aligned row) by more than BAND_PX; a free panel fills its cell by construction.
    """
    bad = []
    for ax in fig.axes:
        cell = getattr(ax, '_cell', None)
        if cell is None or not ax._visible or getattr(ax, '_is_colorbar', False):
            continue
        if cell.shape_ratio() is None:
            continue
        if _in_main_pane(cell):
            continue                        # the main pane of an OrientedPanes: its room follows the pane rule (the cap)
        cx, cy, cw, ch = cell.rect_px()
        x0, y0, w, h = ax._px_rect()
        left, top, right, bottom = cell._decor()          # the row's band when the panel is in an aligned row
        band = ch - (h + top + bottom)
        if band > BAND_PX:
            bad.append((ax.get_title() or f'#{fig.axes.index(ax)}', round(band, 1)))
    return bad


def window_fit(fig):
    """
    Rule 11: how far the window is from its panels' size, in px, negative when it is larger: the
    layout's natural height for the canvas's width less the canvas's height, for a window whose
    panel rows are all shaped; for a window of oriented panes the smaller of the panes' fit_delta
    (the height or width the window could lose); None for a window with a free row.
    """
    lay = fig._layout
    pane = fig._fit_pane()
    if pane is not None:
        dw, dh = pane.fit_delta()
        return min(dw, dh)
    try:
        if lay is None or not lay.hasHeightForWidth() or not fig._shaped_layout():
            return None
    except Exception:
        return None
    vp = fig.canvas.viewport()
    return lay.heightForWidth(vp.width()) - vp.height()


def fitted_to_panels(fig, requested):
    """
    Rule 11: whether a window asked for the canvas size `requested` (W, H) px has sized itself to
    its shaped panels instead (Figure.fit_window): its canvas no wider than asked and its layout's
    natural height met (window_fit within BAND_PX below, 2 px above).  A window with a free row
    never fits itself and answers False.
    """
    fit = window_fit(fig)
    if fit is None:
        return False
    W, H = fig._canvas_size()
    return W <= requested[0] + 2 and -BAND_PX <= fit <= 2


def short_labels(fig):
    """In portrait every control label with a short form shows it: the labels that do not."""
    bad = []
    for c in fig._qt_controls:
        if getattr(c, 'qt', None) is None or not c._visible:
            continue
        for w in [c.qt] + c.qt.findChildren(QtWidgets.QWidget):
            if isinstance(w, (QtWidgets.QLabel, QtWidgets.QCheckBox, QtWidgets.QRadioButton, QtWidgets.QPushButton)) and w.isVisible():
                t = w.text().strip()
                if t and t in widgets.SHORT_LABELS and widgets.SHORT_LABELS[t] != t:
                    bad.append((c.name, t))
    return bad


def check_window(fig, name, shots_dir=None, sizes=SIZES):
    """
    The window at every size of SIZES: the measurements above, a screenshot per size when
    `shots_dir` is given.  Returns {'name', 'problems': [(size, kind, detail), ...],
    'counts': {'axes': n, 'controls': n}}.  The window is left at its design size.  A window of
    shaped panels may answer a size with its own (rule 11, fitted_to_panels): the text rule and
    the portrait flag then hold for the size it took.
    """
    plt.show(block=False)
    pump(150)
    W0, H0 = fig._design_px
    problems = []
    for label, fx, fy in sizes:
        if label == 'portrait':
            # A shape taller than wide for every window, a wide design included: the width is the
            # smaller of 0.45 of the design width and half the design height.
            fx = min(fx, 0.5 * H0 / W0)
        resize(fig, W0 / plt.DPI * fx, H0 / plt.DPI * fy)
        pump(100)
        factor = min(fx, fy)
        requested = (W0 * fx, H0 * fy)
        for item in text_rule(fig, factor, requested):
            problems.append((label, 'text rule', item))
        for item in text_cut(fig):
            problems.append((label, 'text cut', item))
        for item in overlaps(fig):
            problems.append((label, 'overlap', item))
        for item in empty_bands(fig):
            problems.append((label, 'empty band', item))
        if label == 'portrait':
            # A window of shaped panels asked for a tall narrow size takes the height its panels give
            # that width (rule 11), which may not be portrait at all; its labels then stay long.
            fitted = fitted_to_panels(fig, requested)
            if not fig.portrait and not fitted:
                problems.append((label, 'portrait flag', fig._canvas_size()))
            if fig.portrait or not fitted:
                for item in short_labels(fig):
                    problems.append((label, 'short label', item))
        if shots_dir:
            os.makedirs(shots_dir, exist_ok=True)
            fig.savefig(os.path.join(shots_dir, f'{name}_{label}.png'), how='screenshot')
    resize(fig, W0 / plt.DPI, H0 / plt.DPI)
    # The window's fit at its design size (rule 11): a window of shaped panels has followed them.
    fit = window_fit(fig)
    if fit is not None and fit < -BAND_PX:
        problems.append(('design', 'window fit', fit))
    return {'name': name, 'problems': problems,
            'counts': {'axes': len([a for a in fig.axes if a._visible]), 'controls': len(fig._qt_controls)}}


RESULTS = []          # (name, ok) of every check made through check(), in order
FAILS = []            # the names of the failed ones (a script may append a failure of its own)


def check(name, cond, info=''):
    """One PASS / FAIL line per check, the result kept (RESULTS, FAILS); the one check function of every check script."""
    ok = bool(cond)
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'   {info}'))
    RESULTS.append((name, ok))
    if not ok:
        FAILS.append(name)
    return ok


def summary():
    """The closing lines of a check script: 'n of m checks passed', the failures; returns the exit code."""
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks passed" + (f"; FAILED: {FAILS}" if FAILS else ""))
    return 1 if FAILS else 0


def print_report(report):
    n = len(report['problems'])
    print(f"{'PASS' if n == 0 else 'FAIL'}  {report['name']}: {report['counts']['axes']} panels, "
          f"{report['counts']['controls']} controls, {n} problem(s)")
    for size, kind, detail in report['problems'][:60]:
        print(f"      {size:9s} {kind:12s} {detail}")
