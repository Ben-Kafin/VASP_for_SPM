"""
The checks of the three stand-alone windows of plan step 4.7, offscreen on synthetic inputs:

  - PROCAR_tools/Band_structure_plot.py (CPUOrbitalBandPlotter): element mode, the 'Orbital'
    toggle to orbital mode and back, the hover labels armed by a left click and cleared by a
    right click, an NCL/SOC case with Kramers-grouped bands;
  - convergence/forces.py: the window on a five-step OUTCAR (the marker dragged on the
    convergence panel, the arrow keys, the wheel on the 3D panel, the window-start slider) and
    on a one-step OUTCAR (the slider hidden);
  - charge_density/lv_slice_w_atoms_2system.py: the two-system window with its left-hand colour
    bar, the cropped view and the single slice.

Every window is measured at its design size, half, double and portrait
(checks.render_windows.check_window); screenshots go to shots_dir.  Run:

    QT_QPA_PLATFORM=offscreen python checks/check_misc_windows.py [shots_dir]
"""

import os
import sys
import tempfile
import types
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ('', 'PROCAR_tools', 'convergence', 'charge_density'):
    p = os.path.join(_ROOT, sub) if sub else _ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                 # noqa: E402
from qtgui import qtplot as plt, widgets                           # noqa: E402
from qtgui.qtplot import QtCore                                    # noqa: E402
from checks import render_windows as rw                            # noqa: E402

plt._RUNNING[0] = True          # plt.show() returns at once: the checks drive the windows themselves
SHOTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), 'misc_shots')
check = rw.check           # the one check function of every check script (checks.render_windows)
fails = rw.FAILS


def sizes_for(fig):
    """The render sizes of a window: the standard ones, the portrait factor narrowed for a design wider than 2:1 so the window is taller than wide."""
    W0, H0 = fig._design_px
    fx = min(0.45, 0.8 * H0 / W0)
    return (('design', 1.0, 1.0), ('half', 0.5, 0.5), ('double', 2.0, 2.0), ('portrait', fx, 1.0))


def press_event(fig, ax, x, y, button=1, name='button_press_event'):
    """A canvas event at data (x, y) of `ax`, as the canvas makes one from a Qt event."""
    px, py_bottom = ax.transData.transform([[x, y]])[0]
    H = fig.canvas.viewport().height()
    return fig.canvas._event(name, plt._PosOnly(QtCore.QPointF(px, H - py_bottom)), button=button)


# ------------------------------------------------------------------------------------------- #
#  Band_structure_plot                                                                          #
# ------------------------------------------------------------------------------------------- #

ORBS = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2']


def write_band_inputs(d, soc=False, nb=4, nk=6):
    """POSCAR (Si2 O1), DOSCAR (9 or 36 site columns), BAND.dat (one band in zigzag order), KLABELS."""
    with open(os.path.join(d, 'POSCAR'), 'w') as f:
        f.write("Si2O\n1.0\n 5.4 0 0\n 0 5.4 0\n 0 0 5.4\n Si O\n 2 1\nDirect\n 0 0 0\n 0.25 0.25 0.25\n 0.5 0.5 0.5\n")
    ncol = 36 if soc else 9
    with open(os.path.join(d, 'DOSCAR'), 'w') as f:
        f.write(" 3 3 1 0\n x\n x\n x\n x\n 10.0 -5.0 7 2.5 1.0\n")
        for i in range(7):
            f.write(f" {-5 + 2 * i:.3f} 1.0 1.0\n")
        f.write(" 10.0 -5.0 7 2.5 1.0\n")
        f.write(" -5.0 " + " ".join(["0.1"] * ncol) + "\n")
    with open(os.path.join(d, 'BAND.dat'), 'w') as f:
        for b in range(nb):
            f.write(f"# Band-Index: {b + 1}\n")
            ks = np.linspace(0, 1, nk)
            if soc:
                base = -2.0 + 1.0 * (b // 2) + 0.5 * np.sin(np.pi * ks)
                es = base + (1e-4 if b % 2 else 0.0)        # Kramers pairs: within the 1 meV threshold
            else:
                es = -2.0 + 0.8 * b + 0.5 * np.sin(np.pi * ks)
            order = range(nk - 1, -1, -1) if b == 1 else range(nk)     # VASPKIT's zigzag on band 2
            for i in order:
                f.write(f" {ks[i]:.4f} {es[i]:.4f}\n")
    with open(os.path.join(d, 'KLABELS'), 'w') as f:
        f.write("K-Label    K-Coordinate\nGAMMA 0.000\nX 0.500\nM 1.000\n* note\n")


def make_band_plotter(d, soc=False):
    import Band_structure_plot as bsp
    p = bsp.CPUOrbitalBandPlotter(directory=d, filter_types=['Si', 'O'])
    rng = np.random.default_rng(3)
    nb, nk = 4, 6
    p.states = {(0, k, b): rng.random((3, 9)).astype(np.float32) for k in range(nk) for b in range(nb)}
    p.ispin, p.norb, p.orbital_names = 1, 9, list(ORBS)
    if soc:
        p.spin_mode = bsp.SpinMode.NCL_SOC
    return p, bsp


def check_band_plot():
    print("\n== Band_structure_plot")
    d = tempfile.mkdtemp(prefix='bands_')
    write_band_inputs(d)
    p, bsp = make_band_plotter(d)
    p.plot_colored_bands()
    fig = p.fig
    check('band 2 canonicalised to ascending k', p.band_raw[1]['k'][0] < p.band_raw[1]['k'][-1])
    axes = [a for a in fig.axes if a._visible]
    check('element mode: one panel', len(axes) == 1, len(axes))
    ax = axes[0]
    scs = [a for a in ax.collections if isinstance(a, plt.PathCollection)]
    check('one scatter per band', len(scs) == 4, len(scs))
    check("the 'Orbital' control is a check of SPECS", p._chk_orbital.name == 'Orbital' and p._chk_orbital.value is False)
    check('KLABELS guides and ticks', len(ax.lines) >= 4)
    tick_texts = [t for level in ax.xaxis_item._tickLevels for _, t in level]
    check('KLABELS labels applied to the ticks (Gamma as a Greek letter)', tick_texts == ['\u0393', 'X', 'M'], tick_texts)
    check('legend of the element proxies', ax.get_legend() is not None)
    check('title is "Short: details"', ax.get_title() == 'Fatbands' and 'dynamic normalized' in fig.info_text(), (ax.get_title(), fig.info_text()))
    rep = rw.check_window(fig, 'bands_element', SHOTS)
    rw.print_report(rep)
    check('render check: element mode', not rep['problems'])
    # hover: a left click on the third marker of band 1 arms the labels and labels it
    hl = widgets.HoverLabels.of(ax)
    check('hover labels registered, off', hl is not None and not hl.enabled)
    xy = scs[0].get_offsets()[2]
    ev = press_event(fig, ax, xy[0], xy[1], button=1)
    check('click lands in the panel', ev.inaxes is ax, (ev.inaxes, ev.xdata, ev.ydata))
    widgets.HoverLabels._on_click(ev)
    check('left click arms and labels "band 1 / k-point 3"', hl.enabled and hl._label is not None and hl._label.get_text() == 'band 1\nk-point 3',
          (hl.enabled, hl._label and hl._label.get_text()))
    widgets.HoverLabels._on_click(press_event(fig, ax, xy[0], xy[1], button=3))
    check('right click clears and disarms', not hl.enabled and hl._label is None)
    # the Orbital toggle: two element panels, a suptitle; and back
    p._chk_orbital.widget.click()
    check('toggle -> orbital mode', p.color_by == 'orbital')
    axes = [a for a in fig.axes if a._visible]
    check('orbital mode: one panel per element', len(axes) == 2 and [a.get_title() for a in axes] == ['Si', 'O'], [a.get_title() for a in axes])
    check('suptitle "Short: details"', fig._suptitle is not None and fig._suptitle.get_text() == 'Orbital-resolved fatbands', fig._suptitle and fig._suptitle.get_text())
    check('every panel has hover labels', all(widgets.HoverLabels.of(a) is not None for a in axes))
    rep = rw.check_window(fig, 'bands_orbital', SHOTS)
    rw.print_report(rep)
    check('render check: orbital mode', not rep['problems'])
    p._chk_orbital.widget.click()
    axes = [a for a in fig.axes if a._visible]
    check('toggle back -> element mode, one panel, no title band', p.color_by == 'element' and len(axes) == 1
          and (fig._title_cell is None or not fig._title_cell.isVisible()),
          (p.color_by, len(axes), fig._title_cell and fig._title_cell.isVisible()))
    plt.close(fig)
    # NCL/SOC: Kramers pairs grouped, one scatter per group, a two-line hover text
    d2 = tempfile.mkdtemp(prefix='bands_soc_')
    write_band_inputs(d2, soc=True)
    p2, _ = make_band_plotter(d2, soc=True)
    check('DOSCAR with 36 site columns -> NCL_SOC', p2.spin_mode is bsp.SpinMode.NCL_SOC)
    p2.plot_colored_bands()
    groups = p2._soc_groups
    check('two Kramers groups of two bands at the 1 meV threshold', [g['members'] for g in groups] == [[0, 1], [2, 3]], [g['members'] for g in groups])
    ax2 = [a for a in p2.fig.axes if a._visible][0]
    scs2 = [a for a in ax2.collections if isinstance(a, plt.PathCollection)]
    check('one scatter per group', len(scs2) == 2, len(scs2))
    xy = scs2[1].get_offsets()[0]
    widgets.HoverLabels._on_click(press_event(p2.fig, ax2, xy[0], xy[1], button=1))
    hl2 = widgets.HoverLabels.of(ax2)
    check('SOC hover text lists both members', hl2._label is not None and hl2._label.get_text() == 'band 3 k-point 1\nband 4 k-point 1',
          hl2._label and hl2._label.get_text())
    p2._chk_orbital.widget.click()
    axes2 = [a for a in p2.fig.axes if a._visible]
    check('SOC orbital mode: grouped scatters per element panel',
          len(axes2) == 2 and all(len([a for a in ax.collections if isinstance(a, plt.PathCollection)]) == 2 for ax in axes2))
    plt.close(p2.fig)


# ------------------------------------------------------------------------------------------- #
#  forces                                                                                       #
# ------------------------------------------------------------------------------------------- #

def write_forces_inputs(d, nsteps=5):
    rng = np.random.default_rng(5)
    pos0 = np.array([[0.5, 0.5, 1.0], [2.5, 0.5, 1.2], [1.5, 2.5, 3.0]])
    with open(os.path.join(d, 'POSCAR'), 'w') as f:
        f.write("test\n1.0\n 4.0 0 0\n 0 4.0 0\n 0 0 8.0\n Au C\n 2 1\nSelective dynamics\nDirect\n")
        f.write(" 0.125 0.125 0.125 F F F\n 0.625 0.125 0.150 T T T\n 0.375 0.625 0.375 T T T\n")
    with open(os.path.join(d, 'OUTCAR'), 'w') as f:
        f.write(" header\n   IBRION =      2    ionic relax: 0-MD 1-quasi-New 2-CG\n   POTIM  =   0.5000    time-step for ionic-motion\n")
        f.write("   EDIFFG = -.2E-01   stopping-criterion for IOM\n   NIONS =      3\n")
        f.write("  direct lattice vectors                 reciprocal lattice vectors\n")
        f.write("     4.000000000  0.000000000  0.000000000     0.25 0 0\n     0.000000000  4.000000000  0.000000000     0 0.25 0\n     0.000000000  0.000000000  8.000000000     0 0 0.125\n")
        for s in range(nsteps):
            f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n -----------------------------------------------------------------------------------\n")
            for i in range(3):
                p = pos0[i] + 0.05 * s * np.array([1, 0, 0])
                fo = (rng.random(3) - 0.5) * (0.4 / (s + 1))
                f.write(f"  {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}   {fo[0]:.6f} {fo[1]:.6f} {fo[2]:.6f}\n")
            f.write(" -----------------------------------------------------------------------------------\n")
    return os.path.join(d, 'OUTCAR'), os.path.join(d, 'POSCAR')


def check_forces():
    print("\n== forces")
    import forces as fmod
    d = tempfile.mkdtemp(prefix='forces_')
    outcar, poscar = write_forces_inputs(d, 5)
    positions, fo, time, tol, mode = fmod.parse_forces(outcar, seldyn=['FFF', 'TTT', 'TTT'])
    check('five steps read in time mode, EDIFFG 0.02', len(time) == 5 and mode == 'time' and abs(tol - 0.02) < 1e-12, (len(time), mode, tol))
    r = fmod.forces(outcar, poscar)
    fig = r.fig
    check('the 3D panel is a matplotlib canvas in the layout', r.canvas3d.parentWidget() is fig.canvas.viewport() and r.ax.name == '3d')
    check('the convergence legend sits in its own cell at the right', r.ax_leg.get_legend() is not None and r.axf.get_legend() is None
          and r.ax_leg._cell.inner_w is not None)
    check('the y label reads with its units', r.axf.get_ylabel().startswith('$F_{total}$') and 'Å' in plt.mathtext(r.axf.get_ylabel()))
    check('the convergence panel is a qtplot axes with its curves, the convergence line and the marker',
          r.axf in fig.axes and len([c for c in r.axf.collections if isinstance(c, plt.PathCollection)]) == 5 and r.marker in r.axf.lines)
    check("the slider is 'window start' of SPECS, shown", r.slider.name == 'window start' and r.slider._visible and r.slider.slider.maximum() == 4)
    check('readout shows the window start time', r.readout.get_text() == f'window start {time[0]:.3f} fs', r.readout.get_text())
    rw.pump(150)
    leg_w, box_w = r.ax_leg._legend.geometry().width(), r.ax_leg._px_rect()[2]
    check('the legend fits its cell at the design size', leg_w + 8 <= box_w, (leg_w, box_w))
    rep = rw.check_window(fig, 'forces', SHOTS)
    rw.print_report(rep)
    check('render check: forces', not rep['problems'])
    # the time marker: press at step 4, drag to step 2, release
    r.on_press(press_event(fig, r.axf, time[3], float(np.mean(r.axf.get_ylim()))))
    check('press on the convergence panel steps to the nearest step', r.state['step'] == 3 and r.state['pressed'], r.state)
    check('marker moved', abs(r.marker.get_xdata()[0] - time[3]) < 1e-9)
    r.on_motion(press_event(fig, r.axf, time[1], float(np.mean(r.axf.get_ylim())), name='motion_notify_event'))
    check('drag moves the step', r.state['step'] == 1, r.state['step'])
    r.on_release(None)
    check('release ends the drag', not r.state['pressed'])
    check('the 3D title follows the step', r.ax.get_title().startswith('step 2 of 5'), r.ax.get_title())
    # arrow keys on the window's canvas
    ev = plt.KeyEvent('key_press_event', fig.canvas, 'right')
    r.on_key(ev)
    check('right arrow steps forward and keeps the key', r.state['step'] == 2 and ev.handled)
    r.on_key(plt.KeyEvent('key_press_event', fig.canvas, 'left'))
    check('left arrow steps back', r.state['step'] == 1)
    # the wheel on the 3D panel
    before = [list(l) for l in r.state['lims']]
    r.on_scroll(types.SimpleNamespace(inaxes=r.ax, button='up'))
    after = r.state['lims']
    check('scroll up zooms the 3D view in by 0.9', all(abs((a[1] - a[0]) - 0.9 * (b[1] - b[0])) < 1e-9 for a, b in zip(after, before)))
    r.on_scroll(types.SimpleNamespace(inaxes=r.axf, button='up'))
    check('scroll elsewhere leaves the 3D view', all(abs((a[1] - a[0]) - 0.9 * (b[1] - b[0])) < 1e-9 for a, b in zip(r.state['lims'], before)))
    # the slider: window start at step 3 (index 2)
    r.slider.slider.setValue(2)
    check('slider sets the window start', r.state['window'] == 2 and r.state['step'] == 2, (r.state['window'], r.state['step']))
    check('x limits follow the window', abs(r.axf.get_xlim()[0] - time[2]) < 1e-9 and abs(r.axf.get_xlim()[1] - time[-1]) < 1e-9, r.axf.get_xlim())
    check('readout follows', r.readout.get_text() == f'window start {time[2]:.3f} fs', r.readout.get_text())
    r.on_key(plt.KeyEvent('key_press_event', fig.canvas, 'left'))
    check('left arrow stops at the window start', r.state['step'] == 2)
    r.slider.slider.setValue(4)
    check('window start at the last step pads the x limits', r.state['window'] == 4 and r.axf.get_xlim()[0] < time[-1] < r.axf.get_xlim()[1])
    try:
        out = fig.savefig(os.path.join(SHOTS, 'forces_pub.png'), dpi=72)
        check('publication export of the window', os.path.getsize(out) > 500)
    except Exception as exc:
        traceback.print_exc()
        check('publication export of the window', False, exc)
    plt.close(fig)
    # one step: the slider hidden
    d1 = tempfile.mkdtemp(prefix='forces1_')
    outcar1, poscar1 = write_forces_inputs(d1, 1)
    r1 = fmod.forces(outcar1, poscar1)
    check('one step: the slider is hidden', not r1.slider._visible and r1.state['step'] == 0)
    rep = rw.check_window(r1.fig, 'forces_one_step', SHOTS)
    rw.print_report(rep)
    check('render check: one step', not rep['problems'])
    plt.close(r1.fig)


# ------------------------------------------------------------------------------------------- #
#  lv_slice_w_atoms_2system                                                                     #
# ------------------------------------------------------------------------------------------- #

def write_chg_dir(d, shift=0.0, dims=(10, 10, 14)):
    lat = np.diag([5.0, 5.0, 7.0])
    with open(os.path.join(d, 'POSCAR'), 'w') as f:
        f.write("slab\n1.0\n 5 0 0\n 0 5 0\n 0 0 7\n Au C H\n 2 1 1\nDirect\n")
        f.write(" 0.2 0.5 0.3\n 0.7 0.5 0.3\n 0.45 0.5 0.6\n 0.5 0.5 0.8\n")
    nx, ny, nz = dims
    fx = (np.arange(nx) + 0.5) / nx
    fy = (np.arange(ny) + 0.5) / ny
    fz = (np.arange(nz) + 0.5) / nz
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
    val = 2.0 * np.exp(-(((FX - 0.45 - shift) * 5) ** 2 + ((FY - 0.5) * 5) ** 2 + ((FZ - 0.6) * 7) ** 2) / 0.5) \
        - 1.5 * np.exp(-(((FX - 0.5) * 5) ** 2 + ((FY - 0.5) * 5) ** 2 + ((FZ - 0.8) * 7) ** 2) / 0.5)
    vals = val.flatten(order='F') * abs(np.linalg.det(lat))
    with open(os.path.join(d, 'CHGCAR_diff.vasp'), 'w') as f:
        f.write("slab\n1.0\n 5 0 0\n 0 5 0\n 0 0 7\n Au C H\n 2 1 1\nDirect\n")
        f.write(" 0.2 0.5 0.3\n 0.7 0.5 0.3\n 0.45 0.5 0.6\n 0.5 0.5 0.8\n\n")
        f.write(f" {nx} {ny} {nz}\n")
        for k in range(0, len(vals), 5):
            f.write(" ".join(f"{v:.6E}" for v in vals[k:k + 5]) + "\n")


def check_slices():
    print("\n== lv_slice_w_atoms_2system")
    import lv_slice_w_atoms_2system as mod
    da, db = tempfile.mkdtemp(prefix='sysA_'), tempfile.mkdtemp(prefix='sysB_')
    write_chg_dir(da)
    write_chg_dir(db, shift=0.1)
    colors = {'Au': 'orange', 'N': 'mediumaquamarine', 'C': 'saddlebrown', 'H': 'yellow'}
    fig, axes, pcms = mod.plot_two_slices_side_by_side(da, db, axis=1, fraction=0.5, filetype='CHGCAR', use_interpolation=True,
                                                       cmap='seismic', x_half_width1=2.0, x_half_width2=2.0,
                                                       lower_cut1=1.0, upper_cut1=6.5, lower_cut2=1.0, upper_cut2=6.5,
                                                       tol=1.275, element_colors=colors, show_legend=True)
    check('two panels and one colour bar', len(fig.axes) == 3 and len(pcms) == 2, len(fig.axes))
    cax = [a for a in fig.axes if a._is_colorbar][0]
    slot0 = axes[0]._cell._slot
    check('the bar is on the left of the first panel', cax._cell._slot is slot0 and slot0.layout().indexOf(cax._cell) == 0)
    check('the bar is as tall as the row (one row)', cax._row is not None and all(a in cax._row for a in axes))
    check('SymLogNorm meshes', all(isinstance(p.norm, plt.SymLogNorm) for p in pcms))
    check('atoms drawn and the element legend', all(len([c for c in a.collections if isinstance(c, plt.PathCollection)]) >= 1 for a in axes)
          and axes[0].get_legend() is not None)
    check('titles "Short: details"', [a.get_title() for a in axes] == ['System 1', 'System 2'] and 'fraction 0.500' in fig.info_text(),
          ([a.get_title() for a in axes], fig.info_text()))
    check('x limits median-centred', abs((axes[0].get_xlim()[1] - axes[0].get_xlim()[0]) - 4.0) < 0.3, axes[0].get_xlim())
    rep = rw.check_window(fig, 'slices_two', SHOTS, sizes=sizes_for(fig))
    rw.print_report(rep)
    check('render check: two systems', not rep['problems'])
    try:
        out = fig.savefig(os.path.join(SHOTS, 'slices_pub.png'), dpi=72)
        check('publication export with SymLogNorm', os.path.getsize(out) > 500)
    except Exception as exc:
        traceback.print_exc()
        check('publication export with SymLogNorm', False, exc)
    plt.close(fig)
    fig2, ax2, pcm2 = mod.run_cropped_view(da, 1, 0.5, filetype='CHGCAR', use_interpolation=True, cmap='seismic', x_half_width=1.5,
                                           lower_cut=1.0, upper_cut=6.0)
    check('cropped view: panel, bar, title', len(fig2.axes) == 2 and ax2.get_title() == 'Cropped view' and abs(ax2.get_ylim()[0] - 1.0) < 1e-9)
    rep = rw.check_window(fig2, 'slices_cropped', SHOTS, sizes=sizes_for(fig2))
    rw.print_report(rep)
    check('render check: cropped view', not rep['problems'])
    plt.close(fig2)
    ps = mod.PlaneSlice(da, filetype='CHGCAR', cmap='seismic', tol=1.275, element_colors=colors)
    fig3, ax3, pcm3 = ps.plot_plane_slice(1, 0.5, use_interpolation=True)
    check('single slice: panel, bar, title', len(fig3.axes) == 2 and ax3.get_title() == 'Slice')
    rep = rw.check_window(fig3, 'slices_single', SHOTS, sizes=sizes_for(fig3))
    rw.print_report(rep)
    check('render check: single slice', not rep['problems'])
    plt.close(fig3)
    # LOCPOT-style label and formatter path (the potential branch of the colour bar)
    ps.filetype = 'LOCPOT'
    fig4, ax4, pcm4 = ps.plot_plane_slice(1, 0.5, use_interpolation=True)
    cax4 = [a for a in fig4.axes if a._is_colorbar][0]
    check('potential: formatter on the bar axis', cax4._colorbar._axis._formatter is not None and cax4._colorbar._label.startswith('electrostatic'))
    plt.close(fig4)


if __name__ == '__main__':
    os.makedirs(SHOTS, exist_ok=True)
    for fn in (check_band_plot, check_forces, check_slices):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            fails.append(fn.__name__ + ' (exception)')
    sys.exit(rw.summary())
