"""
Offscreen checks of the matcher's interactive plots (plan step 4.6) on a synthetic matcher output.

    QT_QPA_PLATFORM=offscreen python checks/check_matcher_plots.py [shots_dir]

A small matcher output (band_matches_rectangular.txt and its _all.txt: two molecule components, a
metal, sixteen full-system states, no INCAR so SOC is inferred from the spectra) is written to a
temporary directory; every window is opened on it: RectAEPAWColorPlotter.plot (horizontal and
vertical), plot_combined_runs (both orientations), spread_plot.plot_spread, spread_collapse's
SpreadPlotter and MultiSpreadPlotter (grid, collapsed and curve layouts), plot_state_decomposition
(sticks and curves).  Each window goes through checks.render_windows.check_window (the text rule,
no text cut, no overlap, short labels, screenshots) and the interactions of its inventory are
scripted: every check, the σ box with a valid and an out-of-range value, the Segmentation and Norm
checks, the hover arming by a left click and its clearing by a right click.  One line PASS / FAIL
per check; exits non-zero when any failed.

The `matcher` package is stubbed as its modules' own __main__ blocks stub it (a module object with
__path__), so the physics stack behind matcher/__init__.py (pymatgen, pySBT) is not needed here.
"""
import os
import sys
import types
import tempfile
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_MOPM = os.path.join(_ROOT, 'LDOS_spread_decomp', 'MOPmatcher_mem')
if _MOPM not in sys.path:
    sys.path.insert(0, _MOPM)
_pkg = types.ModuleType("matcher")
_pkg.__path__ = [os.path.join(_MOPM, 'matcher')]
sys.modules.setdefault("matcher", _pkg)

import numpy as np                                                       # noqa: E402
from qtgui import qtplot as plt, controls, widgets                       # noqa: E402
from qtgui.qtplot import QtCore                                          # noqa: E402
from checks import render_windows as rw                                  # noqa: E402
from matcher.vacupot_plotter import RectAEPAWColorPlotter, PlotConfig    # noqa: E402
from matcher.spread_plot import plot_spread as plot_spread_single        # noqa: E402
from matcher.spread_collapse import plot_spread, plot_spread_multi, SIGMA_MAX   # noqa: E402
from matcher.state_decomposition_plot import plot_state_decomposition    # noqa: E402
from matcher.combined_vacupot_plotter import plot_combined_runs          # noqa: E402

SHOTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), 'matcher_shots')
check = rw.check           # the one check function of every check script (checks.render_windows)
fails = rw.FAILS


def section(name, fn):
    try:
        fn()
    except Exception as exc:
        traceback.print_exc()
        check(f'{name} (raised {type(exc).__name__}: {exc})', False)


# ---------------------------------------------------------------------- #
#  The synthetic matcher output                                            #
# ---------------------------------------------------------------------- #

COMPS = {
    'NHC_a': [-9.0, -7.5, -6.4, -5.3, -2.1, -1.0],
    'NHC_b': [-8.9, -7.4, -6.3, -5.2, -2.0, -0.9],
    'metal': [-8.2, -7.1, -6.0, -4.9, -3.8, -2.7, -1.6, -0.5],
}
HOMOS = {'NHC_a': 4, 'NHC_b': 4, 'metal': 4}
FERMIS = {'full': -4.6, 'NHC_a': -5.0, 'NHC_b': -4.9, 'metal': -4.7}
VACUUM = 0.8


def write_synthetic_run(d, n_full=16):
    """The two matcher text outputs of a made-up run in `d`; returns the main file's path."""
    rng = np.random.default_rng(3)
    E_full = np.linspace(-9.3, 0.2, n_full)
    labels = list(COMPS)
    rows, all_lines = [], []
    for j, Ef in enumerate(E_full, start=1):
        raw = {}
        for lbl, Es in COMPS.items():
            w = 0.45 if lbl == 'metal' else 0.3
            raw[lbl] = w * np.exp(-((Ef - np.asarray(Es)) / 0.7) ** 2) * (0.7 + 0.6 * rng.random(len(Es)))
        tot = sum(v.sum() for v in raw.values())
        scale = min(1.0, 0.9 / tot) if tot > 0 else 1.0            # a residual of at least 0.1
        blocks = [f"{j} {Ef:.4f}"]
        kept = 0.0
        for lbl in labels:
            ovs = raw[lbl] * scale
            Es = COMPS[lbl]
            ib = int(np.argmax(ovs))
            ws = float(ovs.sum())
            kept += ws
            blocks.append(f"{ib + 1} {Es[ib]:.4f} {Ef - Es[ib]:.4f} {ovs[ib]:.5f} {ws:.5f}")
            for i, (E, ov) in enumerate(zip(Es, ovs), start=1):
                if ov > 1e-5:
                    all_lines.append(f"{lbl} {j} {i} {E:.4f} {Ef - E:.4f} {ov:.5f} {ws:.5f}")
        blocks.append(f"{max(0.0, 1.0 - kept):.5f}")
        rows.append(" | ".join(blocks))
    head = ["# HOMOS: " + "; ".join(f"{k}={v}" for k, v in HOMOS.items()),
            "# FERMIS: " + "; ".join(f"{k}={v}" for k, v in FERMIS.items()),
            f"# VACUUMS: aligned={VACUUM}",
            "# full_idx E_full | " + " | ".join(f"{lbl}_idx E dE ov_best w_span" for lbl in labels) + " | residual"]
    main = os.path.join(d, 'band_matches_rectangular.txt')
    with open(main, 'w') as fh:
        fh.write("\n".join(head + rows) + "\n")
    with open(os.path.join(d, 'band_matches_rectangular_all.txt'), 'w') as fh:
        fh.write("# comp full_idx comp_idx E dE ov w_span\n" + "\n".join(all_lines) + "\n")
    return main


# ---------------------------------------------------------------------- #
#  Event helpers                                                           #
# ---------------------------------------------------------------------- #

def _canvas_xy(fig, ax, x, y):
    px, py_up = ax.transData.transform([[x, y]])[0]
    return float(px), float(fig.canvas.viewport().height() - py_up)


def press(fig, ax, x, y, button=1):
    """A mouse press at data (x, y) of `ax`, fired to every handler of the figure (the layer's and the window's)."""
    px, py = _canvas_xy(fig, ax, x, y)
    ev = fig.canvas._event('button_press_event', plt._PosOnly(QtCore.QPointF(px, py)), button=button)
    fig.canvas._fire('button_press_event', ev)
    rw.pump(30)
    return ev


def move(fig, ax, x, y):
    px, py = _canvas_xy(fig, ax, x, y)
    ev = fig.canvas._event('motion_notify_event', plt._PosOnly(QtCore.QPointF(px, py)))
    fig.canvas._fire('motion_notify_event', ev)
    rw.pump(30)
    return ev


def set_sigma(sigma_box, text):
    """A value typed into a SigmaBox and entered."""
    sigma_box.box.widget.setText(text)
    sigma_box.box._enter()
    sigma_box.box._submit()
    rw.pump(30)


def label_text(ax):
    hl = widgets.HoverLabels.of(ax)
    return hl._label.get_text() if (hl is not None and hl._label is not None) else None


def curves(ax):
    return [a for a in ax.lines if isinstance(a, plt.Line2D)]


def render(fig, name):
    rep = rw.check_window(fig, name, SHOTS)
    rw.print_report(rep)
    check(f'{name}: layout rules at every size', not rep['problems'])


def seg_mid(coll, k=0):
    sg = coll.get_segments()[k]
    return float(sg[:, 0].mean()), float(sg[:, 1].mean())


# ---------------------------------------------------------------------- #
#  The windows                                                             #
# ---------------------------------------------------------------------- #

def t_vacupot(d, main):
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    plotter = RectAEPAWColorPlotter(cfg)
    fig, axes = plotter.plot(main, bonding=True)
    plt.show()
    rw.pump(150)
    check('vacupot: four panels (metal, two molecules, full)', len(axes) == 4, len(axes))
    check('vacupot: hover labels on every panel', all(widgets.HoverLabels.of(a) is not None for a in axes))
    check('vacupot: behaviour summaries written', os.path.isfile(os.path.join(d, 'NHC_a_behavior.txt')))
    ax_f = axes[-1]
    segs = [c for c in ax_f.collections if isinstance(c, plt.LineCollection)]
    check('vacupot: the full panel holds segments', len(segs) >= 16, len(segs))
    lo, hi = ax_f.get_xlim()
    check('vacupot: the energy axis spans the sticks and the vacuum line (old autoscale)', lo < -9.0 and hi > VACUUM, (lo, hi))
    hl = widgets.HoverLabels.of(ax_f)
    check('vacupot: labels start disarmed', not hl.enabled)
    x, y = seg_mid(segs[0])
    press(fig, ax_f, x, y, 1)
    check('vacupot: a left click arms and labels the segment under it', hl.enabled and (label_text(ax_f) or '').startswith('full_idx'), label_text(ax_f))
    press(fig, ax_f, x, y, 3)
    check('vacupot: a right click clears and disarms', not hl.enabled and label_text(ax_f) is None)
    ax_m = axes[0]
    hm = widgets.HoverLabels.of(ax_m)
    stick = [c for c in ax_m.collections if isinstance(c, plt.LineCollection)][0]
    x, y = seg_mid(stick)
    press(fig, ax_m, x, y, 1)
    check('vacupot: a component stick labels with its band text', 'band' in (label_text(ax_m) or ''), label_text(ax_m))
    render(fig, 'vacupot_horizontal')
    plt.close(fig)

    cfg_v = PlotConfig(shared_molecule_color=True, orientation='vertical', energy_range=(-10.0, 1.0), energy_max_vacuum=True)
    fig2, axes2 = RectAEPAWColorPlotter(cfg_v).plot(main, bonding=False)
    plt.show()
    rw.pump(150)
    lo, hi = axes2[-1].get_ylim()
    check('vacupot vertical: energy on y up to the vacuum', abs(hi - VACUUM) < 1e-6 and abs(lo + 10.0) < 1e-6, (lo, hi))
    render(fig2, 'vacupot_vertical')
    plt.close(fig2)


def t_combined(d, main):
    for orient in ('horizontal', 'vertical'):
        cfg = PlotConfig(shared_molecule_color=True, orientation=orient, energy_range=(-10.0, 1.0),
                         energy_max_vacuum=True, pick_primary=False)
        fig, axes = plot_combined_runs(d, d, cfg=cfg, bonding=True, title_molecule='NHC')
        plt.show()
        rw.pump(150)
        check(f'combined {orient}: six panels', len(axes) == 6, len(axes))
        mol_axes = [a for a in axes if a.get_title() == 'NHC']
        check(f'combined {orient}: the molecule panels carry the two-column (monospace) hover', mol_axes and all(widgets.HoverLabels.of(a) is not None and widgets.HoverLabels.of(a).mono for a in mol_axes))
        render(fig, f'combined_{orient}')
        plt.close(fig)


def t_spread_single(d, main):
    cfg = PlotConfig(shared_molecule_color=True)
    fig, axes = plot_spread_single(d, 'NHC_a', [-2, -1, 0, 1], cfg=cfg)
    plt.show()
    rw.pump(150)
    chk, sld = fig._spread_widgets
    check('spread: four panels, sticks first', len(axes) == 4 and not curves(axes[0]) and len(axes[0].collections) >= 2)
    check('spread: σ box hidden in stick mode', sld.qt.parentWidget().isHidden())
    check('spread: title short name shown, details in the info panel', axes[0].get_title().startswith('NHC_a HOMO') and 'E =' in fig.info_text(), (axes[0].get_title(), fig.info_text()[:80]))
    render(fig, 'spread_sticks')
    # the MO marker's hover
    mo = [c for c in axes[0].collections if isinstance(c, plt.LineCollection)][-1]
    x, y = seg_mid(mo)
    press(fig, axes[0], x, y, 1)
    check('spread: left click on the marker labels it', (label_text(axes[0]) or '').startswith('NHC_a'), label_text(axes[0]))
    press(fig, axes[0], x, y, 3)
    check('spread: right click clears', label_text(axes[0]) is None)
    chk.boxes[0].click()          # Curve on
    rw.pump(60)
    check('spread: curve mode draws one curve per panel', all(len(curves(a)) == 1 for a in axes))
    check('spread: σ box shown in curve mode', not sld.qt.parentWidget().isHidden())
    x0 = curves(axes[0])[0].get_xdata().copy()
    y0 = curves(axes[0])[0].get_ydata().copy()
    set_sigma(sld, '0.020')
    x1 = curves(axes[0])[0].get_xdata()
    check('spread: σ = 0.020 refines the grid and recomputes the curve', len(x1) > len(x0), (len(x0), len(x1)))
    set_sigma(sld, '5')
    check('spread: σ above its limit is clamped to SIGMA_MAX and applied', sld.value == SIGMA_MAX and sld.box.text == '1.000', (sld.value, sld.box.text))
    y2 = curves(axes[0])[0].get_ydata()
    check('spread: the clamped σ changed the curve', len(y2) != len(y0) or not np.allclose(y2, y0))
    set_sigma(sld, 'abc')
    check('spread: a bad σ entry is put back', sld.box.text == '1.000')
    chk.boxes[1].click()          # Amplitude on
    rw.pump(60)
    check('spread: Amplitude on shows the sqrt label', 'sqrt' in axes[0].get_ylabel(), axes[0].get_ylabel())
    render(fig, 'spread_curve')
    plt.close(fig)


def t_spread_collapse(d, main):
    cfg = PlotConfig(shared_molecule_color=True)
    fig, sp = plot_spread(d, ['NHC_a', 'NHC_b'], [-2, -1, 0, 1], cfg=cfg)
    plt.show()
    rw.pump(150)
    W = fig._canvas_size()[0]
    check('collapse: 4 x 2 grid', len(fig.axes) == 8, len(fig.axes))
    check('collapse: Segmentation and σ hidden at start', sp._seg_box.isHidden() and sp._sigma_box.isHidden())
    render(fig, 'collapse_grid')
    sp.chk.boxes[1].click()       # Horizontal on
    rw.pump(60)
    check('collapse: Horizontal on gives one column', len(fig.axes) == 4 and not sp._seg_box.isHidden(), len(fig.axes))
    w_one = fig.axes[0]._px_rect()[2]
    check('collapse: the one column takes the width (no stale grid stretch)', w_one > 0.75 * W, (w_one, W))
    n_before = sum(len(a.collections) for a in fig.axes)
    sp.chk_seg.widget.click()     # Segmentation on
    rw.pump(60)
    n_after = sum(len(a.collections) for a in fig.axes)
    check('collapse: Segmentation splits the bars into component parts', sp.collapse_segmentation and n_after > n_before, (n_before, n_after))
    sp.chk.boxes[0].click()       # Vertical on
    rw.pump(60)
    check('collapse: both on gives one panel titled Spread', len(fig.axes) == 1 and fig.axes[0].get_title() == 'Spread', (len(fig.axes), fig.axes[0].get_title()))
    sp.chk.boxes[3].click()       # Curve on
    rw.pump(60)
    check('collapse: curve mode registers curves and shows σ', len(sp._curve_entries) == 4 and not sp._sigma_box.isHidden(), len(sp._curve_entries))
    x0 = sp._curve_entries[0]['line'].get_xdata().copy()
    set_sigma(sp.sld_sigma, '0.020')
    check('collapse: σ recomputes the curves in place', len(sp._curve_entries[0]['line'].get_xdata()) > len(x0))
    # hover on a curve
    ax = fig.axes[0]
    ln = sp._curve_entries[0]['line']
    k = int(np.argmax(ln.get_ydata()))
    press(fig, ax, float(ln.get_xdata()[k]), float(ln.get_ydata()[k]), 1)
    check('collapse: left click on a curve labels it', 'Gaussian-broadened spread' in (label_text(ax) or ''), label_text(ax))
    render(fig, 'collapse_curve')
    plt.close(fig)


def t_multi(d, main):
    cfg = PlotConfig(shared_molecule_color=True)
    fig, msp = plot_spread_multi([(d, ['NHC_a', 'NHC_b']), (d, ['NHC_a'])], [-1, 0, 1], cfg=cfg)
    plt.show()
    rw.pump(150)
    check('multi: MO row plus one row per run', len(fig.axes) == 3, len(fig.axes))
    check('multi: MO row title short', fig.axes[0].get_title() == 'Original MOs', fig.axes[0].get_title())
    render(fig, 'multi_collapsed')
    msp.chk.boxes[0].click()      # Horizontal off -> split
    rw.pump(60)
    check('multi: split grid 3 x 2 with the empty slot off', len(fig.axes) == 6 and sum(not a._axison for a in fig.axes) == 1, len(fig.axes))
    msp.chk.boxes[2].click()      # Curve on
    rw.pump(60)
    n = sum(len(sp._curve_entries) for sp in msp.plotters)
    check('multi: curve mode registers curves on the sub-plotters and shows σ', n > 0 and not msp._sigma_box.isHidden(), n)
    x0 = msp.plotters[0]._curve_entries[0]['line'].get_xdata().copy()
    set_sigma(msp.sld_sigma, '0.020')
    check('multi: σ recomputes the sub-plotters\' curves', len(msp.plotters[0]._curve_entries[0]['line'].get_xdata()) > len(x0))
    render(fig, 'multi_split_curve')
    plt.close(fig)


def t_state_decomp(d, main):
    cfg = PlotConfig(shared_molecule_color=True)
    fig, axes = plot_state_decomposition(d, sum_threshold=0.05, cfg=cfg)
    plt.show()
    rw.pump(150)
    chk, sld, chk_norm = fig._decomp_widgets
    check('state: four columns (metal states, metal spreads, molecule spreads, molecule states)', len(axes) == 4, len(axes))
    titles = [a.get_title() for a in axes]
    check('state: titles short (shown exactly as written: a label is data)',
          titles == ['metal states', 'metal spreads', 'molecule spreads', 'molecule states'], titles)
    lo, hi = axes[0].get_ylim()
    # the energies carry the Fermi shift (-FERMIS['full']): the lowest drawn line is the lowest
    # full state, the highest the vacuum, each with the 5 % margin beyond it
    check('state: the energy axis spans the drawn lines (old autoscale)',
          lo < -9.3 - FERMIS['full'] and hi > VACUUM - FERMIS['full'], (lo, hi))
    check('state: Norm and σ hidden in stick mode', sld.qt.parentWidget().isHidden() and chk_norm.qt.parentWidget().isHidden())
    ax_s = axes[1]
    hl = widgets.HoverLabels.of(ax_s)
    check('state: spreads panel hover armed by the window, not by the figure click', hl is not None and not hl.arm_by_click and not hl.enabled)
    n_all = len(hl.artists)
    stacks = [c for c in ax_s.collections if isinstance(c, plt.LineCollection)]
    longest = max(stacks, key=lambda c: abs(c.get_segments()[0][1, 0] - c.get_segments()[0][0, 0]))
    x, y = seg_mid(longest)
    press(fig, ax_s, x, y, 1)
    check('state: left click on a stack arms that state\'s segments alone', hl.enabled and 0 < len(hl.artists) < n_all, (hl.enabled, len(hl.artists), n_all))
    move(fig, ax_s, x, y)
    check('state: hovering the stack shows its segment text', 'full_idx' in (label_text(ax_s) or ''), label_text(ax_s))
    press(fig, ax_s, x, y, 3)
    check('state: right click clears and disarms', not hl.enabled and label_text(ax_s) is None)
    ax_og = axes[0]
    rl = [a for a in ax_og.lines if isinstance(a, plt._RefLine)][0]
    press(fig, ax_og, 0.5, float(rl.get_ydata()[0]), 1)
    check('state: left click on an original-states line labels it', 'band' in (label_text(ax_og) or ''), label_text(ax_og))
    render(fig, 'state_sticks')
    chk.boxes[0].click()          # Curve on
    rw.pump(60)
    ents = fig._decomp_curve_entries
    check('state: curve mode gives two system curves and shows Norm and σ', len(ents) == 2 and not sld.qt.parentWidget().isHidden() and not chk_norm.qt.parentWidget().isHidden(), len(ents))
    check('state: curve titles short', [a.get_title() for a in axes] == ['metal states', 'metal curve', 'molecule curve', 'molecule states'], [a.get_title() for a in axes])
    y0 = [e['y'].copy() for e in ents]
    chk_norm.widget.click()       # Norm on: per-curve normalization
    rw.pump(60)
    check('state: Norm changes the curves\' heights', any(not np.allclose(a, b) for a, b in zip(y0, [e['y'] for e in ents])))
    x0 = ents[0]['x'].copy()
    set_sigma(sld, '0.020')
    check('state: σ refines the grid', len(ents[0]['x']) > len(x0))
    lc = ents[0]['lc']
    k = int(np.argmax(ents[0]['y']))
    press(fig, ents[0]['ax'], float(ents[0]['y'][k]), float(ents[0]['x'][k]), 1)
    check('state: left click on a curve labels it', 'total decomposition curve' in (label_text(ents[0]['ax']) or ''), label_text(ents[0]['ax']))
    render(fig, 'state_curve')
    plt.close(fig)


def main():
    plt._RUNNING[0] = True            # plt.show() returns at once: the windows are driven from here
    os.makedirs(SHOTS, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        main_path = write_synthetic_run(d)
        for name, fn in (('vacupot', t_vacupot), ('combined', t_combined), ('spread', t_spread_single),
                         ('collapse', t_spread_collapse), ('multi', t_multi), ('state', t_state_decomp)):
            section(name, lambda fn=fn: fn(d, main_path))
    print('screenshots in', SHOTS)
    return rw.summary()


if __name__ == '__main__':
    sys.exit(main())
