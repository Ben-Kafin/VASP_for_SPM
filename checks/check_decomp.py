"""
The offscreen checks of the decomposition windows and the plane slice (plan step 4.5): the
windows built on synthetic inputs (no VASP run needed), every size of checks.render_windows
measured, and every interaction of the old windows' inventory scripted:

  DecompWindow        the 'collapse' toggles (all four layouts), Amplitude, 'Metal curve', the
                      'area ≥' box (a valid and an invalid entry), Save data, refresh, the hover
                      labels armed by a left click and cleared by a right click.
  TopoDecompWindow    the 'topo view' toggles (Amplitude, Norm), the 'not found' cell, one colour
                      bar per drawn cell, the transparent figure patch, Save data, refresh.
  plot_global_decomposition   the σ box (a value, a clamped value, a bad entry), Save data.
  run_wfn_plane_slice.run     log and linear modes, the γ box (1.5, 2.75, bad entries), the PNG.

    QT_QPA_PLATFORM=offscreen python3 checks/check_decomp.py [shots_dir]

The static MO data of the two decomposition windows (load_static_mo_data) and the matcher
results of the global decomposition come from small synthetic sets built here, so the checks run
without a matcher run; the mathematics of the windows is untouched by that (the same functions
run on the synthetic inputs).  When pymatgen is not installed, a stand-in module takes its place
for the imports alone (the matcher package imports its LOCPOT reader at import time; nothing
here reads a LOCPOT).  Prints PASS / FAIL per check and exits non-zero when any failed.
"""

import os
import sys
import tempfile
import types
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_MO = os.path.join(_ROOT, 'LDOS_spread_decomp', 'mo_ldos_decomp')
for p in (_ROOT, _MO):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pymatgen.io.vasp          # noqa: F401
except ImportError:
    # The matcher package imports pymatgen's Locpot reader at import time; a stand-in lets the
    # plotting modules import here (nothing in these checks reads a LOCPOT).
    _pm = types.ModuleType('pymatgen'); _io = types.ModuleType('pymatgen.io'); _v = types.ModuleType('pymatgen.io.vasp')
    _v.Locpot = type('Locpot', (), {})
    _pm.io, _io.vasp = _io, _v
    sys.modules.update({'pymatgen': _pm, 'pymatgen.io': _io, 'pymatgen.io.vasp': _v})

import numpy as np
from qtgui import qtplot as plt, widgets
from qtgui.qtplot import QtCore
from checks import render_windows as rw

plt._RUNNING[0] = True                 # a window's plt.show() returns at once

from mo_ldos_projection import decomp_window as dw                 # noqa: E402
from mo_ldos_projection import topo_decomp_window as tdw           # noqa: E402
from mo_ldos_projection import global_decomposition as gd          # noqa: E402
from mo_ldos_projection._imports import Entity, PlotConfig         # noqa: E402
import run_wfn_plane_slice as rps                                  # noqa: E402

SHOTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), 'decomp_shots')
check = rw.check           # the one check function of every check script (checks.render_windows)
fails = rw.FAILS


def press(fig, ax, fx, fy, button=1):
    """A synthesised button press at axes fractions (fx, fy), delivered to the hover handler."""
    x0, y0, w, h = ax._px_rect()
    ev = fig.canvas._event('button_press_event', plt._PosOnly(QtCore.QPointF(x0 + fx * w, y0 + (1 - fy) * h)),
                           button=button)
    widgets.HoverLabels._on_click(ev)
    return ev


# ---------------------------------------------------------------------- #
#  Synthetic static MO data (what load_static_mo_data returns)             #
# ---------------------------------------------------------------------- #

def synthetic_static(seed=0):
    rng = np.random.default_rng(seed)
    n_full = 40
    E_sticks = np.linspace(-3.0, 0.4, n_full)
    labels = ['molA', 'molB']
    colors = {'molA': np.array([0.12, 0.47, 0.71]), 'molB': np.array([0.84, 0.15, 0.16])}
    entities_by_comp, rows = {}, []
    for k, lab in enumerate(labels):
        offmap = {}
        for offset in (-1, 0, 1):
            idx = 10 + 8 * (offset + 1) + 3 * k
            ent = Entity((idx,), float(E_sticks[idx]), idx)
            offmap[offset] = ent
            # the LUMO entities carry an area below the threshold (5e-2): the filter has something to drop
            w = np.exp(-0.5 * ((np.arange(n_full) - idx) / (1.5 + k)) ** 2) * (0.9 if offset != 1 else 0.005)
            rows.append(dict(comp=lab, offset=offset, ent=ent, weights=w, color=colors[lab],
                             label=f"{lab} {dw._offset_name(offset)} ({idx})"))
        entities_by_comp[lab] = offmap
    metal_grid = np.abs(rng.normal(size=(6, n_full))) * 0.3
    return dict(P=types.SimpleNamespace(), metal_grid=metal_grid, e_raw_win=E_sticks + 1.0, ef_doscar=1.0,
                E_sticks=E_sticks, mol_labels=labels, entities_by_comp=entities_by_comp, entity_rows=rows)


def ldos_snapshot(sigma=0.2):
    eg = np.arange(-3.0, 0.5 + 1e-9, 0.005)
    ld1 = 0.6 * np.exp(-((eg + 1.0) / 0.5) ** 2) + 0.2 * np.exp(-((eg + 2.2) / 0.3) ** 2) + 0.01
    ld2 = 0.9 * np.exp(-((eg + 0.4) / 0.4) ** 2) + 0.02
    return dict(sigma=sigma, eg=eg, markers=[(1.0, 2.0), (3.0, 4.0)], ldos=[ld1, ld2],
                marker_colors=['#1f77b4', '#2ca02c'])


def topo_snapshot(sigma=0.2, bias=0.2, n=7):
    eg = np.arange(-0.3, bias + 1e-9, 0.005)
    xs, ys = np.meshgrid(np.linspace(0.0, 10.0, n), np.linspace(0.0, 10.0, n))
    grid_xy = np.column_stack([xs.ravel(), ys.ravel()])
    t = np.linspace(0.0, 1.0, n * n)
    ldos = (0.05 + np.outer(t, np.exp(-((eg - 0.05) / 0.08) ** 2))
            + np.outer(1.0 - t, np.exp(-((eg + 0.2) / 0.1) ** 2)))
    return dict(sigma=sigma, eg=eg, grid_xy=grid_xy, ldos=ldos, topo_bias=bias, topo_height=2.5,
                lv=np.diag([10.0, 10.0, 20.0]))


static = synthetic_static()
dw.load_static_mo_data = lambda run_dir, cfg: static
tdw.load_static_mo_data = lambda run_dir, cfg: static
run_dir = tempfile.mkdtemp(prefix='decomp_check_')
cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)

# ---------------------------------------------------------------------- #
#  DecompWindow                                                            #
# ---------------------------------------------------------------------- #
try:
    win = dw.DecompWindow(run_dir, ldos_snapshot(), cfg=cfg)
    plt.show(block=False); rw.pump(150)
    data_axes = lambda w: [a for a in w.fig.axes if a.get_subplotspec() is not None]
    check('decomp: opens collapsed, one panel per marker', len(data_axes(win)) == 2 and win.collapse_vertical and win.collapse_horizontal)
    check('decomp: the weak entities are below the threshold', len(win.kept) == 4 and all(r['offset'] != 1 for r in win.kept), [r['label'] for r in win.kept])
    check('decomp: every panel has a legend and the LDOS', all(a.get_legend() is not None and any(l.get_label().startswith('LDOS') for l in a.get_lines()) for a in data_axes(win)))
    n_ctrl = len(win.fig._qt_controls)
    # the four layouts through the real check boxes (Vertical, Horizontal)
    counts = {}
    for v, h in ((True, False), (False, False), (False, True), (True, True)):
        if win.chk.value[0] != v:
            win.chk.boxes[0].click()
        if win.chk.value[1] != h:
            win.chk.boxes[1].click()
        rw.pump(60)
        counts[(v, h)] = len(data_axes(win))
    R, C = len(win.kept_offsets), len(win.kept_comps)
    check('decomp: the layouts give n, nC, nR, nRC panels', counts == {(True, True): 2, (True, False): 2 * C, (False, True): 2 * R, (False, False): 2 * R * C}, (counts, R, C))
    check('decomp: the controls persist through rebuilds', len(win.fig._qt_controls) == n_ctrl)
    # Amplitude: the LDOS drawn as the sqrt
    win.chk.boxes[2].click(); rw.pump(60)
    ldos_line = [l for l in data_axes(win)[0].get_lines() if l.get_label().startswith('LDOS')][0]
    check('decomp: Amplitude draws the sqrt', win.amplitude and np.allclose(ldos_line.get_ydata(), np.sqrt(np.abs(win.ldos_curves[0]))))
    win.chk.boxes[2].click(); rw.pump(60)
    # Metal curve off and on
    n_lines_on = len(data_axes(win)[0].get_lines())
    win.chk_metal.widget.click(); rw.pump(60)
    n_lines_off = len(data_axes(win)[0].get_lines())
    check('decomp: Metal curve toggle removes the dashed curve', not win.show_metal and n_lines_off == n_lines_on - 1, (n_lines_on, n_lines_off))
    win.chk_metal.widget.click(); rw.pump(60)
    check('decomp: Metal curve back on', win.show_metal and len(data_axes(win)[0].get_lines()) == n_lines_on)
    # the area box: a huge threshold empties the kept set (annotation), an invalid entry is put back
    win.tb_thresh.widget.setText('1e9'); win.tb_thresh._enter(); win.tb_thresh._submit(); rw.pump(60)
    annos = [t.get_text() for a in win.fig.axes for t in a.texts]
    check('decomp: area 1e9 keeps nothing and annotates', win.kept == [] and any('no MOs above threshold' in t for t in annos), annos)
    win.tb_thresh.widget.setText('abc'); win.tb_thresh._enter(); win.tb_thresh._submit(); rw.pump(30)
    check('decomp: a bad area entry is put back', win.tb_thresh.value == '1.00e+09' and win.threshold == 1e9, win.tb_thresh.value)
    win.tb_thresh.widget.setText('5e-2'); win.tb_thresh._enter(); win.tb_thresh._submit(); rw.pump(60)
    check('decomp: the threshold back, four kept', len(win.kept) == 4)
    # hover: a left click arms and labels, a right click clears
    ax0 = data_axes(win)[0]
    hl = widgets.HoverLabels.of(ax0)
    check('decomp: a hover label is registered per panel', hl is not None and not hl.enabled)
    (xa, xb), (ya, yb) = ax0.vb.viewRange()
    # the LDOS peak of marker 0 at E = -1.0: click there
    fx = (-1.0 - xa) / (xb - xa); fy = (float(win.ldos_curves[0].max()) - ya) / (yb - ya)
    press(win.fig, ax0, fx, fy, button=1); rw.pump(30)
    check('decomp: left click arms and labels the LDOS', hl.enabled and hl._label is not None and 'LDOS marker 0' in hl._label.get_text(), (hl.enabled, hl._label and hl._label.get_text()))
    press(win.fig, ax0, fx, fy, button=3); rw.pump(30)
    check('decomp: right click clears and disarms', not hl.enabled and hl._label is None)
    # Save data
    out = win.save_data()
    check('decomp: Save data writes the record', os.path.isfile(out) and open(out, encoding='utf-8').read().startswith('# SIGMA: 0.200000'))
    win.btn_save.widget.click()
    # refresh with a new snapshot
    snap2 = ldos_snapshot(); snap2['ldos'] = snap2['ldos'][::-1]
    win.refresh(snap2); rw.pump(60)
    check('decomp: refresh recomputes from the new press', np.allclose(win.ldos_curves[0], snap2['ldos'][0]) and len(data_axes(win)) == 2)
    rep = rw.check_window(win.fig, 'decomp_window', SHOTS); rw.print_report(rep)
    check('decomp: render check clean', not rep['problems'])
    # the empty case at every size too
    win.threshold = 1e9; win._recompute_kept(); win._build_ui(); rw.pump(60)
    rep = rw.check_window(win.fig, 'decomp_window_empty', SHOTS); rw.print_report(rep)
    check('decomp (empty): render check clean', not rep['problems'])
    win.threshold = 5e-2; win._recompute_kept(); win._build_ui()
    # the full grid at every size
    win.chk.boxes[0].click(); win.chk.boxes[1].click(); rw.pump(60)
    rep = rw.check_window(win.fig, 'decomp_window_grid', SHOTS); rw.print_report(rep)
    check('decomp (grid): render check clean', not rep['problems'])
    plt.close(win.fig)
except Exception:
    traceback.print_exc(); check('decomp window', False)

# ---------------------------------------------------------------------- #
#  TopoDecompWindow                                                        #
# ---------------------------------------------------------------------- #
try:
    twin = tdw.TopoDecompWindow(run_dir, topo_snapshot(), offsets=[-1, 0, 500], cfg=cfg)
    plt.show(block=False); rw.pump(150)
    data_axes = lambda w: [a for a in w.fig.axes if a.get_subplotspec() is not None]
    check('topo: a 3 x 2 grid of cells', len(twin.cells) == 6 and len(data_axes(twin)) == 6)
    check('topo: the missing offset draws not found', all(c['p'] is None for c in twin.cells[-2:]) and any('not found' in t.get_text() for a in data_axes(twin)[-2:] for t in a.texts))
    # the title rule shows the short name exactly as written (a component label is data); compared case-blind here
    check('topo: the not-found title is Short: details', data_axes(twin)[-1].get_title().lower() == 'molb lumo+499' and 'not found' in data_axes(twin)[-1]._title_full)
    check('topo: one colour bar per drawn cell', len(twin.cbars) == 6 and sum(b is not None for b in twin.cbars) == 4 and all(b.ax.get_subplotspec() is None for b in twin.cbars if b))
    check('topo: the figure patch is transparent', twin.fig.patch.get_facecolor()[3] == 0.0)
    cell_ax = data_axes(twin)[0]
    tc = cell_ax.collections[0]
    fc = tc.get_facecolor()
    check('topo: one single-colour alpha-ramp layer per cell', len(cell_ax.patches) == 0 and np.allclose(fc[:, :3], twin.cells[0]['row']['color']) and fc[0, 3] < 0.05 and fc[-1, 3] > 0.95 and cell_ax.get_facecolor()[3] == 0.0)
    check('topo: the title is the entity label', cell_ax.get_title().lower() == twin.cells[0]['row']['label'].lower())
    p_before = [c['p'].copy() for c in twin.cells if c['p'] is not None]
    twin.chk.boxes[0].click(); rw.pump(60)
    check('topo: Amplitude is display-only', twin.amplitude and all(np.array_equal(c['p'], p0) for c, p0 in zip([c for c in twin.cells if c['p'] is not None], p_before)) and twin.cbars[0].mappable.norm.vmin == twin._reference_range(twin.cells[0])[0])
    twin.chk.boxes[1].click(); rw.pump(60)
    bars = [b for b in twin.cbars if b is not None]
    check('topo: Norm shares one range over every bar', twin.norm_shared and len({(b.mappable.norm.vmin, b.mappable.norm.vmax) for b in bars}) == 1)
    out = twin.save_data()
    body = np.array([[float(x) for x in l.split()] for l in open(out, encoding='utf-8') if not l.startswith('#')])
    check('topo: Save data writes x, y and one raw p column per found cell', body.shape == (49, 2 + 4))
    twin.chk.boxes[0].click(); twin.chk.boxes[1].click(); rw.pump(60)
    snap2 = topo_snapshot(); snap2['ldos'] = snap2['ldos'][::-1].copy()
    twin.refresh(snap2); rw.pump(60)
    check('topo: refresh recomputes the projections', any(not np.allclose(c['p'], p0) for c, p0 in zip([c for c in twin.cells if c['p'] is not None], p_before)))
    rep = rw.check_window(twin.fig, 'topo_decomp_window', SHOTS); rw.print_report(rep)
    check('topo: render check clean', not rep['problems'])
    try:
        tdw.TopoDecompWindow(run_dir, topo_snapshot(), offsets=[], cfg=cfg)
        check('topo: empty offsets rejected', False)
    except ValueError:
        check('topo: empty offsets rejected', True)
    plt.close(twin.fig)
except Exception:
    traceback.print_exc(); check('topo window', False)

# ---------------------------------------------------------------------- #
#  plot_global_decomposition                                               #
# ---------------------------------------------------------------------- #
try:
    rng = np.random.default_rng(1)
    n_full, n_met = 30, 5
    E_full = np.linspace(-6.0, 3.0, n_full) + 0.5
    comp_pairs = {'molA': [(1, -4.0), (2, -2.5), (3, -0.5)]}
    colors = {'molA': {1: (0.1, 0.3, 0.8, 1.0), 2: (0.2, 0.6, 0.3, 1.0), 3: (0.8, 0.2, 0.2, 1.0)}}
    P_fake = types.SimpleNamespace(comp_pairs=comp_pairs, is_soc=False, known_homos={'molA': 2}, component_colors=colors,
                                   known_fermis={'full': 0.5}, rows=[], by_full={}, v_aligned=5.0)
    # small amplitudes: the per-state weights sum below 1, so the residual channel stays non-negative
    c_mo = 0.2 * np.exp(-0.5 * ((np.arange(n_full)[None, :] - np.array([[6], [12], [20]])) / 3.0) ** 2)
    c_met = 0.1 * rng.uniform(-1.0, 1.0, size=(n_met, n_full))
    data_fake = dict(comp_labels=['molA', 'metal'], E_full_raw=E_full, fermi_full=0.5, v_aligned=5.0,
                     comp_energies={'molA': np.array([-4.0, -2.5, -0.5]), 'metal': np.zeros(n_met)},
                     homo_indices={'molA': 2}, grid={'molA': np.abs(c_mo) ** 2, 'metal': np.abs(c_met) ** 2},
                     extended=True, pre_amendment_cross=False, grid_amp={'molA': c_mo, 'metal': c_met},
                     B=np.zeros((n_met, 3)), S_MO=np.eye(3), S_mm=np.eye(n_met),
                     mo_axis=[['molA', 1], ['molA', 2], ['molA', 3]], stored_diagnostics={})
    gd.RectAEPAWColorPlotter = lambda cfg: types.SimpleNamespace(_prepare=lambda *a, **k: P_fake)
    gd.load_results = lambda path: data_fake
    gdir = tempfile.mkdtemp(prefix='global_check_')
    fig, state = gd.plot_global_decomposition(gdir, mo_curves={'molA': [-1, 0, 1, 7]}, cfg=cfg, sigma=0.2)
    plt.show(block=False); rw.pump(150)
    ax = state['ax']
    check('global: total, three channels and three MO curves', len(state['lines']) == 7, list(state['lines']))
    check('global: legend and reference lines', ax.get_legend() is not None and len([l for l in ax.lines if isinstance(l, plt._RefLine)]) == 2)
    sigma_box, btn = state['widgets']
    y0 = state['lines']['total'].get_ydata().copy()
    sigma_box.box.widget.setText('0.350'); sigma_box.box._enter(); sigma_box.box._submit(); rw.pump(30)
    check('global: the σ box applies 0.350', state['sigma'] == 0.35 and not np.array_equal(state['lines']['total'].get_ydata(), y0))
    sigma_box.box.widget.setText('5'); sigma_box.box._enter(); sigma_box.box._submit(); rw.pump(30)
    check('global: a σ above the range is clamped to SIGMA_MAX', state['sigma'] == gd.SIGMA_MAX and sigma_box.box.text == '1.000')
    sigma_box.box.widget.setText('x'); sigma_box.box._enter(); sigma_box.box._submit(); rw.pump(30)
    check('global: a bad σ is put back', state['sigma'] == gd.SIGMA_MAX and sigma_box.box.text == '1.000')
    sigma_box.set(0.2); check('global: set without firing', sigma_box.value == 0.2 and state['sigma'] == gd.SIGMA_MAX)
    out = state['save']()
    check('global: Save data writes the record', os.path.isfile(out) and '# SIGMA: 1.000000' in open(out, encoding='utf-8').read())
    btn.widget.click()
    rep = rw.check_window(fig, 'global_decomposition', SHOTS); rw.print_report(rep)
    check('global: render check clean', not rep['problems'])
    plt.close(fig)
except Exception:
    traceback.print_exc(); check('global decomposition', False)

# ---------------------------------------------------------------------- #
#  run_wfn_plane_slice                                                     #
# ---------------------------------------------------------------------- #
try:
    from mo_ldos_projection.tests.test_wfn_plane_slice import write_multiatom_chg
    tmp = tempfile.mkdtemp(prefix='slice_check_')
    dims = (8, 10, 12)
    A = np.broadcast_to((np.arange(8) / 8.0)[:, None, None], dims)
    path = os.path.join(tmp, 'ramp.vasp')
    write_multiatom_chg(path, A.flatten(order='F'), dims)
    path2 = os.path.join(tmp, 'ramp2x.vasp')
    write_multiatom_chg(path2, 2.0 * A.flatten(order='F'), dims)
    for mode in ('log', 'linear'):
        fig, axes, panels = rps.run([path, path2], [1, 3], n_s=21, n_z=5, norm_mode=mode, save_png=True)
        plt.show(block=False); rw.pump(150)
        png = os.path.splitext(path)[0] + '_plane_slice_compare.png'
        check(f'slice ({mode}): PNG written through the publication export', os.path.isfile(png) and os.path.getsize(png) > 1000)
        pcm = axes[0].collections[0]
        check(f'slice ({mode}): one shared norm, a colour bar beside the last panel', axes[1].collections[0].norm is pcm.norm and pcm.colorbar is not None and pcm.colorbar.ax._row is not None)
        gb = fig._gamma_widgets
        gb.box.widget.setText('1.5'); gb.box._enter(); gb.box._submit(); rw.pump(30)
        check(f'slice ({mode}): γ 1.5 applied to every mesh', all(a.collections[0].norm.gamma == 1.5 for a in axes) and gb.text == '1.50')
        gb.box.widget.setText('2.75'); gb.box._enter(); gb.box._submit(); rw.pump(30)
        check(f'slice ({mode}): γ above 2 accepted', pcm.norm.gamma == 2.75 and pcm.colorbar._axis._mode[1] is pcm.norm)
        for bad in ('x', 'nan', '-1', '0'):
            gb.box.widget.setText(bad); gb.box._enter(); gb.box._submit()
        check(f'slice ({mode}): bad γ entries put back', pcm.norm.gamma == 2.75 and gb.text == '2.75')
        if mode == 'log':
            ticks = pcm.colorbar._axis.tickValues(0, 1, 300)
            check('slice (log): the colour bar ticks decades', bool(ticks) and len(ticks[0][1]) >= 2, ticks)
        rep = rw.check_window(fig, f'plane_slice_{mode}', SHOTS); rw.print_report(rep)
        check(f'slice ({mode}): render check clean', not rep['problems'])
        plt.close(fig)
except Exception:
    traceback.print_exc(); check('plane slice', False)

sys.exit(rw.summary())
