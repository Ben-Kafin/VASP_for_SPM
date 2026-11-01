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
    """The figure at w_in x h_in inches, applied twice (the toolbar and the info dock take window height first)."""
    for _ in range(2):
        fig.set_size_inches(w_in, h_in)
        pump()


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


def text_rule(fig, factor):
    """The fonts follow the one rule at this size factor: the figure's zoom, the tick fonts, the titles, the controls."""
    bad = []
    W, H = fig._canvas_size()
    z = plt.window_zoom(W, H, *fig._design_px)
    want = min(max(factor, plt.ZOOM_MIN), plt.ZOOM_MAX)
    if abs(z - want) > 0.06:
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
    'counts': {'axes': n, 'controls': n}}.  The window is left at its design size.
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
        for item in text_rule(fig, factor):
            problems.append((label, 'text rule', item))
        for item in text_cut(fig):
            problems.append((label, 'text cut', item))
        for item in overlaps(fig):
            problems.append((label, 'overlap', item))
        if label == 'portrait':
            if not fig.portrait:
                problems.append((label, 'portrait flag', fig._canvas_size()))
            for item in short_labels(fig):
                problems.append((label, 'short label', item))
        if shots_dir:
            os.makedirs(shots_dir, exist_ok=True)
            fig.savefig(os.path.join(shots_dir, f'{name}_{label}.png'), how='screenshot')
    resize(fig, W0 / plt.DPI, H0 / plt.DPI)
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
    for size, kind, detail in report['problems'][:40]:
        print(f"      {size:9s} {kind:12s} {detail}")
