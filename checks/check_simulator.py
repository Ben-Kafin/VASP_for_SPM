"""
The simulator window, the decomposition wrapper's control boxes and the DOS plotter, built
offscreen on synthetic inputs and measured (plan 3.6):

    QT_QPA_PLATFORM=offscreen python3 checks/check_simulator.py

There is no GPU here: `cupy` and `cupyx.scipy.ndimage` are numpy / scipy stand-ins registered in
sys.modules before the simulator is imported (the same array calls, on the CPU), and
`pymatgen.io.vasp.Locpot` is a stand-in too (the LOCPOT is given to the simulator as its
LOCPOT.npy cache, which LocpotManager reads without pymatgen).  The inputs are a four-atom slab
(POSCAR), an ISPIN=1 DOSCAR with nine orbital columns and a smooth potential, written into a
temporary directory.  The decomposition windows belong to another part of the suite: the
wrapper's `_imports`, `decomp_window` and `topo_decomp_window` are stand-ins here with the
names the wrapper takes from them, so the wrapper's own controls are what is checked.

Every window is opened, resized through checks.render_windows.check_window (design size, half,
double, portrait: the text rule, no text cut, no overlap, short labels; screenshots in
shots_dir), and the interactions of the inventory are scripted: RUN, the counts and value boxes,
the view radio, a marker drag, the endpoint drag, the energy-marker drag, the spectrum levels,
the γ box, a broadening toggle, the wrapper's σ box and its two buttons, the DOS plotter's
levels, hover and 'm' key.  One line PASS / FAIL per check; exit code 1 when any failed.
"""
import json
import os
import sys
import tempfile
import traceback
import types

_cur = os.environ.get('QT_QPA_PLATFORM', '')
if _cur in ('', 'offscreen'):
    # A screen large enough for the simulator at twice its design size (the plugin's own screen is 800 x 800).
    _path = os.path.join(tempfile.gettempdir(), 'vasp_for_spm_offscreen_screen.json')
    with open(_path, 'w') as fh:
        json.dump({'screens': [{'name': 'checks', 'x': 0, 'y': 0, 'width': 4400, 'height': 3200,
                                'logicalDpi': 96, 'logicalBaseDpi': 96, 'dpr': 1.0}]}, fh)
    os.environ['QT_QPA_PLATFORM'] = f"offscreen:configfile={_path}"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STM = os.path.join(ROOT, 'LDOS_spread_decomp', 'stm_from_dft_opt')
MO = os.path.join(ROOT, 'LDOS_spread_decomp', 'mo_ldos_decomp')
for p in (ROOT, STM, MO, os.path.join(STM, 'stm_from_dft')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

SHOTS = os.environ.get('SHOTS_DIR', '/tmp/claude-0/-home-user/e027c064-e45e-5745-8c9f-d6260f223483/scratchpad/shots/simulator')
SHOTS_SHAPE = os.path.join(SHOTS, 'shape')          # the shape checks' screenshots (a tall and a wide unit cell)
CACHE = os.path.join(ROOT, 'checks', '_cache', 'simulator')   # the inputs and the simulator's topography caches, kept between runs (git-ignored)


# ---------------------------------------------------------------------- #
#  Stand-ins: cupy on numpy, pymatgen's Locpot                             #
# ---------------------------------------------------------------------- #

class GArray(np.ndarray):
    """A numpy array with cupy's .get() (the wrapper's send_topo reads arrays with it)."""

    def get(self):
        return np.asarray(self)


def _g(x):
    return x.view(GArray) if isinstance(x, np.ndarray) and not isinstance(x, GArray) else x


def _wrap(f):
    def w(*a, **k):
        return _g(f(*a, **k))
    w.__name__ = getattr(f, '__name__', 'f')
    return w


def install_cupy_stub():
    cp = types.ModuleType('cupy')
    for name in dir(np):
        if name.startswith('_'):
            continue
        obj = getattr(np, name)
        if callable(obj) and not isinstance(obj, type):
            setattr(cp, name, _wrap(obj))
        else:
            setattr(cp, name, obj)
    cp.asnumpy = lambda a: np.asarray(a)
    cp.trapz = _wrap(getattr(np, 'trapezoid', None) or np.trapz)
    cp.cuda = types.SimpleNamespace(Device=lambda i=0: types.SimpleNamespace(use=lambda: None))
    sys.modules['cupy'] = cp
    import scipy.ndimage as nd
    cpx = types.ModuleType('cupyx')
    cpx_sc = types.ModuleType('cupyx.scipy')
    cpx_nd = types.ModuleType('cupyx.scipy.ndimage')
    cpx_nd.map_coordinates = _wrap(nd.map_coordinates)
    cpx_nd.gaussian_filter1d = _wrap(nd.gaussian_filter1d)
    cpx_sc.ndimage = cpx_nd
    cpx.scipy = cpx_sc
    sys.modules['cupyx'] = cpx
    sys.modules['cupyx.scipy'] = cpx_sc
    sys.modules['cupyx.scipy.ndimage'] = cpx_nd


def install_pymatgen_stub():
    try:
        import pymatgen.io.vasp  # noqa: F401
        return False
    except ImportError:
        pass
    pm = types.ModuleType('pymatgen')
    io = types.ModuleType('pymatgen.io')
    vasp = types.ModuleType('pymatgen.io.vasp')

    class Locpot:
        @staticmethod
        def from_file(path):
            raise RuntimeError("pymatgen is not installed here: the check gives the simulator its LOCPOT.npy cache")
    vasp.Locpot = Locpot
    io.vasp = vasp
    pm.io = io
    sys.modules['pymatgen'] = pm
    sys.modules['pymatgen.io'] = io
    sys.modules['pymatgen.io.vasp'] = vasp
    return True


install_cupy_stub()
PYMATGEN_STUBBED = install_pymatgen_stub()

from qtgui import qtplot as plt                       # noqa: E402
from qtgui.qtplot import QtCore                       # noqa: E402
from qtgui import widgets                             # noqa: E402
from checks import render_windows as rw               # noqa: E402

plt._RUNNING[0] = True          # plt.show() returns at once (no event loop here)

RESULTS = rw.RESULTS       # (name, ok) of every check, kept by checks.render_windows


check = rw.check           # the one check function of every check script (checks.render_windows)


# ---------------------------------------------------------------------- #
#  Synthetic inputs                                                        #
# ---------------------------------------------------------------------- #

def inputs_dir(name, a=5.0, b=5.0):
    """
    The input directory of one check, `checks/_cache/simulator/<name>/`: the synthetic inputs
    written afresh, the simulator's topography caches seeded (seed_caches) and kept between runs,
    so that no run computes a topography.  The simulator changes into this directory and names
    its caches relative to it.
    """
    d = os.path.join(CACHE, name)
    os.makedirs(d, exist_ok=True)
    write_inputs(d, a, b)
    seed_caches(d)
    return d


def seed_caches(d, grid_res=10, z_top=6.3):
    """
    Ready-made topography data in place of the simulator's computations: the global topography
    at the check's bias and height, the LDOS topography at the check's bias, the Topo h rerun
    heights the checks apply (3.0 and the cell top above the highest atom), and the decay-off
    model, each a smooth z-map on the grid_res x grid_res fractional grid (one value per grid
    point, the simulator's own layout), written with the names the simulator looks for
    (`global_topo_<bias>V_<height>A_<grid>px<broadening><unit cell><decay>.npy`,
    `ldos_topo_...`).  The names do not depend on the cell's shape, so the same data serves the
    5 x 5, the tall and the wide cell: the layout check needs the shape, not the physics.
    Existing files are left alone.
    """
    g = np.linspace(0.0, 1.0, grid_res)
    fx, fy = np.meshgrid(g, g)
    bump = 0.3 * np.sin(2 * np.pi * fx.ravel()) * np.cos(2 * np.pi * fy.ravel())
    tip_max = 20.0 - z_top                      # the cell top above the highest atom (the Topo h box's limit)
    names = [f"global_topo_0.2V_{h}A_{grid_res}px_uc1_{model}.npy"
             for h in (2.5, 3.0, tip_max) for model in ('decay_on', 'decay_off', 'decay_on_noclamp')]
    names += [f"ldos_topo_-3.0V_1.25A_{grid_res}px_uc1_{model}.npy" for model in ('decay_on', 'decay_off', 'decay_on_noclamp')]
    names += [f"ldos_topo_-5.0V_1.25A_{grid_res}px_uc1_{model}.npy" for model in ('decay_on', 'decay_off', 'decay_on_noclamp')]
    for name in names:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            h = float(name.split('_')[3][:-1]) if name.startswith('global') else 1.25
            np.save(path, (z_top + h + bump).astype(np.float32))


def write_inputs(d, a=5.0, b=5.0):
    """A four-atom slab in an a x b x 20 Å cell (the atoms scaled into it), an ISPIN=1 DOSCAR (9 orbital columns), a LOCPOT cache."""
    fa, fb = a / 5.0, b / 5.0
    atoms = [(0.0, 0.0, 2.0), (2.5, 2.5, 4.0), (1.0, 1.0, 6.0), (3.5, 3.5, 6.3)]
    with open(os.path.join(d, 'POSCAR'), 'w') as f:
        f.write(f"synthetic slab\n1.0\n  {a:.1f} 0.0 0.0\n  0.0 {b:.1f} 0.0\n  0.0 0.0 20.0\nAu N C\n2 1 1\nCartesian\n"
                + "".join(f"  {x * fa:.3f} {y * fb:.3f} {z:.1f}\n" for x, y, z in atoms))
    ef = -2.0
    nedos = 81
    E = np.linspace(-6.0, 2.0, nedos)
    rng = np.random.RandomState(3)
    with open(os.path.join(d, 'DOSCAR'), 'w') as f:
        f.write("   4   4   1   0\n  0.1 0.2 0.3 0.4 0.5\n  1.0\n  CAR\n synthetic\n")
        f.write(f"  2.0  -6.0  {nedos}  {ef:.4f}  1.0000\n")
        total = np.zeros(nedos)
        atoms = []
        for a in range(4):
            cols = np.zeros((nedos, 9))
            for c in range(9):
                centre = rng.uniform(-5.0, 1.0)
                cols[:, c] = (0.2 + 0.8 * rng.rand()) * np.exp(-0.5 * ((E - centre) / 0.35) ** 2) + 0.02
            atoms.append(cols)
            total += cols.sum(axis=1)
        integ = np.cumsum(total) * (E[1] - E[0])
        for e, t, i in zip(E, total, integ):
            f.write(f"  {e:.4f}  {t:.6f}  {i:.6f}\n")
        for cols in atoms:
            f.write(f"  2.0  -6.0  {nedos}  {ef:.4f}  1.0000\n")
            for e, row in zip(E, cols):
                f.write("  " + f"{e:.4f}  " + "  ".join(f"{v:.6f}" for v in row) + "\n")
    with open(os.path.join(d, 'LOCPOT'), 'w') as f:
        f.write("synthetic (the simulator reads LOCPOT.npy here)\n")
    gx, gy, gz = np.meshgrid(np.arange(8), np.arange(8), np.arange(16), indexing='ij')
    pot = ef + 4.5 + 0.3 * np.sin(2 * np.pi * gx / 8) * np.cos(2 * np.pi * gy / 8) - 0.5 * np.exp(-((gz - 5) / 2.0) ** 2)
    np.save(os.path.join(d, 'LOCPOT.npy'), pot.astype(np.float32))


# ---------------------------------------------------------------------- #
#  Event helpers                                                           #
# ---------------------------------------------------------------------- #

def data_to_canvas(ax, x, y):
    """Canvas pixels (from the top left) of a data point of `ax`."""
    px, py = ax.transData.transform([[x, y]])[0]
    H = ax.figure.canvas.viewport().height()
    return float(px), float(H - py)


def press(fig, ax, x, y, button=1):
    px, py = data_to_canvas(ax, x, y)
    return fig.canvas._event('button_press_event', plt._PosOnly(QtCore.QPointF(px, py)), button=button)


def motion(fig, ax, x, y):
    px, py = data_to_canvas(ax, x, y)
    return fig.canvas._event('motion_notify_event', plt._PosOnly(QtCore.QPointF(px, py)))


def type_into(box, text):
    """A value typed into a number box and applied with Enter."""
    box.widget.setText(text)
    box._enter()
    box._submit()


def apart(lines):
    """The x and y of the first curve where it is farthest from the others (a pick there hits one curve only)."""
    xs = lines[0].get_xdata()
    ys = [l.get_ydata() for l in lines]
    gap = np.min([np.abs(ys[0] - y) for y in ys[1:]], axis=0) if len(ys) > 1 else np.abs(ys[0])
    k = int(np.argmax(gap))
    return float(xs[k]), float(ys[0][k])


def report(rep):
    rw.print_report(rep)
    return check(f"render {rep['name']}: no problem at any size", not rep['problems'])


# ---------------------------------------------------------------------- #
#  The simulator                                                           #
# ---------------------------------------------------------------------- #

def check_simulator(d):
    from stm_from_dft import simulator as simmod
    cmap = plt.LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])
    sim = simmod.Interactive_STM_Simulator(filepath=d, erange=[-3.0, 0.5], ldos_height=1.25, cmap_topo=cmap, unit_cell_num=1)
    sim.run_interactive(grid_res=10, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', use_decay=True,
                        show_decay_toggle=True, reuse_cache=True, line_endpoints=([0.5, 0.5], [4.5, 4.5]),
                        marker_positions=[[1.0, 1.0], [3.0, 3.0]], extra_broadening={'N': 0.0},
                        path=[[0.5, 0.5], [2.5, 2.5], [4.5, 4.5]], path_extend=False)
    fig = sim.fig
    rw.pump(150)
    check('simulator window opens with its controls', fig is not None and len(fig._qt_controls) > 10 and sim.mode == 'Single Point')
    check('no matplotlib on screen', 'matplotlib.pyplot' not in sys.modules or not any(
        getattr(m, '__file__', '') and 'backend_qt' in getattr(m, '__file__', '') for m in sys.modules.values()))
    check('title cell: the spin mode', fig._suptitle is not None and fig._suptitle.get_text() == sim.spin_mode.name)
    check('mode-specific boxes hidden in Single Point', not sim._strip_boxes['Path Pts'].isVisible() and not sim._strip_boxes['E Pts'].isVisible()
          and not sim._box_ldos_h.isVisible() and sim._box_topo_h.isVisible())
    check('σ box of an unchecked type hidden', not sim.s_broad['N'].qt.isVisible() and not sim.s_broad['Au'].qt.isVisible())
    report(rw.check_window(fig, 'simulator_single_point', SHOTS))

    # RUN: the spectra computed and drawn
    sim.btn_run.widget.click()
    rw.pump(50)
    check('RUN computes the marker spectra', sim.is_running and sim.cached_spec_ldos is not None and sim.btn_run.widget.text() == 'STOP'
          and len([l for l in sim.ax_spec.get_lines() if l.get_label().startswith('marker_')]) == 2)
    check('Partitioned LDOS title is short with its details in the info panel',
          sim.ax_spec.get_title() == 'Partitioned LDOS' and 'marker 0' in fig.info_text())
    check('Global Topo title short', sim.ax_map.get_title() == 'Global Topo' and 'bias 0.2 V' in fig.info_text())
    # Markers count box
    type_into(sim.s_num_marks, '3')
    check('Markers box adds a marker', len(sim.marker_coords) == 3 and len(sim.marks.get_offsets()) == 3)
    type_into(sim.s_num_marks, '2')
    check('Markers box removes a marker', len(sim.marker_coords) == 2)
    type_into(sim.s_num_marks, '15')
    check('Markers box refuses 15 (limit 10) and keeps 2', len(sim.marker_coords) == 2 and sim.s_num_marks.value == 2)
    # a marker drag: pick on marker 1, move it, release
    x0, y0 = sim.marker_coords[1]
    ev = press(fig, sim.ax_map, x0, y0)
    fig._pick(ev)
    check('pick on a marker selects it', sim.active_obj == ('mark_map', 1), sim.active_obj)
    sim._on_motion(motion(fig, sim.ax_map, x0 + 0.4, y0 - 0.3))
    check('drag moves the marker and recomputes its spectrum', abs(sim.marker_coords[1][0] - (x0 + 0.4)) < 1e-6
          and np.allclose(sim.cached_marker_coords[1], [x0 + 0.4, y0 - 0.3]))
    sim._on_rel(ev)
    sim.active_obj = None
    # spectrum levels: a pick on a marker curve descends to level 1, a press beside the curves ascends
    curves = [l for l in sim.ax_spec.get_lines() if l.get_label().startswith('marker_')]
    x, y = apart(curves)
    ev = press(fig, sim.ax_spec, x, y)
    sim.active_obj = None
    fig._pick(ev)
    check('pick on a spectrum curve descends to level 1', sim.plot_level == 1, sim.plot_level)
    # The press handler follows the pick (matplotlib's order too): it keeps the level when a
    # pickable curve of the new level lies under the press, else it ascends one level.
    hit = any(l.contains(ev)[0] for l in sim.ax_spec.get_lines() if l.get_picker())
    sim._on_press(ev)
    check('press after the pick keeps the level iff a curve is under it', sim.plot_level == (1 if hit else 0), (hit, sim.plot_level))
    sim.plot_level = 1
    sim._update_all()
    xl, yl = sim.ax_spec.get_xlim(), sim.ax_spec.get_ylim()
    ev = press(fig, sim.ax_spec, xl[0] + 0.02 * (xl[1] - xl[0]), yl[1] - 0.01 * (yl[1] - yl[0]))
    sim._on_press(ev)
    check('press beside the curves ascends to level 0', sim.plot_level == 0, sim.plot_level)
    # γ box recolours only
    cached = sim.cached_spec_ldos
    type_into(sim.s_ldos_gamma.box, '0.50')
    check('LDOS γ box applies 0.50 without recomputing', sim.ldos_gamma == 0.5 and sim.cached_spec_ldos is cached)
    type_into(sim.s_ldos_gamma.box, '-2')
    check('LDOS γ box refuses -2', sim.ldos_gamma == 0.5 and sim.s_ldos_gamma.text == '0.50')
    # E Min box clamps to its limit
    type_into(sim.s_emin, '-9')
    check('E Min clamps to -5 and recomputes', sim.s_emin.val == -5.0 and sim.cached_emin == -5.0)
    type_into(sim.s_emin, '-3.000')
    # broadening: the check shows the σ box, the σ box applies
    sim.chk_broad.boxes[1].click()
    rw.pump(30)
    check('broadening check shows the σ box of N', sim.s_broad['N'].qt.isVisible() and sim.chk_broad.value[1])
    type_into(sim.s_broad['N'], '0.10')
    check('σ N box sets the extra broadening', abs(sim.extra_broadening['N'] - 0.10) < 1e-9)
    sim.chk_broad.boxes[1].click()
    rw.pump(30)
    check('unchecking hides the σ box and clears the broadening', not sim.s_broad['N'].qt.isVisible() and sim.extra_broadening['N'] == 0.0)
    # Cells count: 0 .. 4
    type_into(sim.s_cell, '0')
    check('Cells 0 draws one image', sim.display_cells == 0 and any(isinstance(a, plt.TriContourSet) and len(a._items) == 1 for a in sim.ax_map.collections))
    type_into(sim.s_cell, '1')
    check('Cells 1 draws nine images (one raster moved)', any(isinstance(a, plt.TriContourSet) and len(a._items) == 9 for a in sim.ax_map.collections))
    # Topo h: a value then Rerun
    type_into(sim.s_topo_h, '3.0')
    check('Topo h box holds 3.0 and the topo is unchanged until Rerun', sim.s_topo_h.val == 3.0 and sim.topo_height == 2.5)
    sim.btn_topo_rerun.widget.click()
    rw.pump(50)
    check('Rerun applies the tip height', sim.topo_height == 3.0)
    type_into(sim.s_topo_h, '99')
    check('Topo h clamps to the cell top above the highest atom', abs(sim.s_topo_h.val - sim.tip_height_max) < 1e-9)
    type_into(sim.s_topo_h, '3.0')
    # the toggles
    sim.chk.boxes[3].click()          # Norm
    rw.pump(30)
    check('Norm toggle normalizes the spectra', sim.normalize)
    sim.chk.boxes[3].click()
    rw.pump(30)
    sim.chk.boxes[1].click()          # Decay off: recompute with the other model
    rw.pump(50)
    check('Decay toggle off recomputes (decay_off model)', not sim.use_decay and sim.cached_d_ldos == 'decay_off')
    report(rw.check_window(fig, 'simulator_single_running', SHOTS))

    # ---- Line view ----
    sim.ui_radio.buttons[1].click()
    rw.pump(100)
    check('Line view built', sim.mode == 'Line' and sim.ax_line_topo is not None and sim.ax_stripe is not None
          and sim._box_ldos_h.isVisible() and sim.ax_prof in fig.axes and sim.cax is None)
    heats = [a for a in sim.line_decomp_axes if a is not sim.ax_line_topo and a is not sim.ax_stripe and not a._is_colorbar]
    bars = [a for a in sim.line_decomp_axes if a._is_colorbar]
    check('Line cluster: Topo box, stripe, one heatmap, one colour bar', len(heats) == 1 and len(bars) == 1)
    check('cluster label LDOS', sim._cluster_label.get_text() == 'LDOS' and sim._cluster_label.qt.isVisible())
    check('the stripe is a coloured LineCollection', any(isinstance(a, plt.LineCollection) for a in sim.ax_stripe.collections))
    check('the heatmap row shares one height (align_row)', sim.ax_line_topo._row is not None and sim.ax_stripe in sim.ax_line_topo._row
          and heats[0] in sim.ax_line_topo._row)
    check('heatmap title is the partition', heats[0].get_title() == 'Total')
    check('colour bar exponent title', bars[0].get_title().startswith('1e'))
    report(rw.check_window(fig, 'simulator_line', SHOTS))
    # the endpoint drag
    p1 = sim.p1.copy()
    ev = press(fig, sim.ax_map, float(p1[0]), float(p1[1]))
    fig._pick(ev)
    check('pick on a line end selects it', sim.active_obj == ('end', 0), sim.active_obj)
    sim._on_motion(motion(fig, sim.ax_map, float(p1[0]) + 0.3, float(p1[1]) + 0.2))
    check('dragging the end moves the line and recomputes', abs(sim.p1[0] - (p1[0] + 0.3)) < 1e-6 and np.allclose(sim.cached_p1, sim.p1))
    sim.active_obj = None
    # a marker dragged along the profile panel
    ev = press(fig, sim.ax_prof, 0.0, 0.0)
    sim.active_obj = ('mark_dynamic', 0)
    p_len = float(np.linalg.norm(sim.p2 - sim.p1))
    sim._on_motion(motion(fig, sim.ax_prof, 0.5 * p_len, float(np.mean(sim.current_z_line))))
    check('marker dragged along the profile', abs(sim.marker_ratios[0] - 0.5) < 0.02, sim.marker_ratios)
    sim.active_obj = None
    # the orbital levels and the hover labels (level 4)
    sim.plot_level = 2
    sim.active_element = 'Au'
    sim.plot_level = 3
    sim.active_atom = 1
    sim.plot_level = 4
    sim._update_all()
    rw.pump(50)
    hl = widgets.HoverLabels.of(sim.ax_spec)
    check('level 4: orbital curves with hover labels on', hl is not None and hl.enabled and len(hl.artists) == 9)
    check('level 4: two legends (atom types and orbitals)', sim.ax_spec._legend is not None and len(sim.ax_spec._extra_legends) == 1)
    orb = hl.artists[0]
    xs, ys = orb.get_xdata(), orb.get_ydata()
    k = int(np.argmax(ys))
    widgets.HoverLabels._on_motion(motion(fig, sim.ax_spec, float(xs[k]), float(ys[k])))
    check('hover over an orbital curve shows its label', hl._label is not None and hl._label.get_text() == orb.get_label(), hl._label and hl._label.get_text())
    check('cluster label names the element at the atom levels', sim._cluster_label.get_text() == 'LDOS: Au')
    sim.plot_level = 0
    sim._update_all()

    # ---- Path view ----
    sim.ui_radio.buttons[2].click()
    rw.pump(100)
    check('Path view built', sim.mode == 'Path' and sim.ax_line_topo is None and sim._strip_boxes['Path Pts'].isVisible()
          and len(sim.path_nodes) == 3 and sim.s_path_pts.value == 3)
    type_into(sim.s_path_pts, '4')
    check('Path Pts box adds a node', len(sim.path_nodes) == 4)
    nd = sim.path_nodes[1].copy()
    ev = press(fig, sim.ax_map, float(nd[0]), float(nd[1]))
    fig._pick(ev)
    check('pick on a path node selects it', sim.active_obj == ('path_node', 1), sim.active_obj)
    sim._on_motion(motion(fig, sim.ax_map, float(nd[0]) + 0.2, float(nd[1]) - 0.2))
    check('dragging a node moves it', abs(sim.path_nodes[1][0] - (nd[0] + 0.2)) < 1e-6)
    sim.active_obj = None
    sim.chk.boxes[6].click()      # Extend
    rw.pump(50)
    check('Extend toggle recomputes the path', sim.path_extend and sim.cached_path_extend)
    sim.chk.boxes[6].click()
    rw.pump(50)
    report(rw.check_window(fig, 'simulator_path', SHOTS))

    # ---- Map view (first while stopped: the Map Topo title, which a running refresh clears with the panel as before) ----
    sim.btn_run.widget.click()
    rw.pump(30)
    sim.ui_radio.buttons[3].click()
    rw.pump(150)
    check('Map view built (stopped)', sim.mode == 'Map' and sim._strip_boxes['E Pts'].isVisible() and sim.ax_map_global in fig.axes
          and not sim.is_running)
    check('Map Topo title short', sim.ax_map.get_title() == 'Map Topo' and 'height 1.25' in fig.info_text(), (sim.ax_map.get_title(), fig.info_text()))
    sim.btn_run.widget.click()
    rw.pump(150)
    check('Map view running: the map rows', sim.is_running and len(sim.map_axes) == 5 and len(sim.cax_list) == 1)
    type_into(sim.s_nepts, '3')
    check('E Pts box rebuilds the map rows', len(sim.map_axes) == 3 and len(sim.map_e_targets) == 3)
    em = [l for l in sim.ax_spec.get_lines() if l.get_label().startswith('emarker_')]
    check('energy markers on the spectrum', len(em) == 3)
    e1 = float(sim.map_e_targets[1])
    # The press is in the band above the curves (the y limits leave 10 % over the highest curve): a
    # press on the marker where a curve crosses it would descend a level as well, as it always did.
    yl = sim.ax_spec.get_ylim()
    ev = press(fig, sim.ax_spec, e1, yl[1] - 0.03 * (yl[1] - yl[0]))
    fig._pick(ev)
    check('pick on an energy marker selects it', sim.active_obj == ('emarker', 1), sim.active_obj)
    sim._on_motion(motion(fig, sim.ax_spec, e1 + 0.3, yl[1] - 0.03 * (yl[1] - yl[0])))
    check('dragging the energy marker moves it', abs(sim.map_e_targets[1] - (e1 + 0.3)) < 1e-6)
    before = sim.cached_ld_up[:, 1].copy()
    try:
        sim._on_rel(ev)
        rel_ok = True
    except Exception:
        traceback.print_exc()
        rel_ok = False
    check('release recomputes that slice and ends the drag', rel_ok and sim.active_obj is None and not np.array_equal(before, sim.cached_ld_up[:, 1]),
          (rel_ok, sim.active_obj))
    sim.active_obj = None
    report(rw.check_window(fig, 'simulator_map', SHOTS))

    # ---- back to Single Point: the window rebuilt without leaks ----
    sim.ui_radio.buttons[0].click()
    rw.pump(100)
    check('back to Single Point', sim.mode == 'Single Point' and len(fig.axes) == 2 and not sim._box_ldos_h.isVisible())
    # STOP
    sim.btn_run.widget.click()
    check('STOP', not sim.is_running and sim.btn_run.widget.text() == 'RUN')
    return sim


# ---------------------------------------------------------------------- #
#  The wrapper's boxes                                                     #
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
#  The shape rule: a tall and a wide unit cell (plan 6.4, guide rule 11)   #
# ---------------------------------------------------------------------- #

def rect_in(fig, w):
    """A widget's rectangle in the canvas's coordinates."""
    vp = fig.canvas.viewport()
    p = w.mapTo(vp, QtCore.QPoint(0, 0))
    return QtCore.QRect(p, w.size())


def above(fig, box, cell):
    """Whether the control box `box` is shown above the panel cell `cell`: its bottom at or over the cell's top, their x ranges overlapping."""
    if box is None or cell is None or not box.isVisible():
        return False
    b, c = rect_in(fig, box), rect_in(fig, cell)
    return b.bottom() <= c.top() + 1 and b.left() < c.right() and b.right() > c.left()


def inside(w, ancestor):
    """Whether `ancestor` is on the parent chain of the widget `w`."""
    while w is not None:
        if w is ancestor:
            return True
        w = w.parentWidget()
    return False


def check_shape(a, b, name, want):
    """
    The simulator on a unit cell a x b Å, RUN on, every view in turn (plan 6.4, the owner's rule:
    the plots fill the window, nothing plotted is squeezed, blank space minimized): no render
    problem at any size (empty bands included); at the design size the arrangement leaves little
    blank (< 12 %), no shaped group is below 45 % of the size it would have alone (the
    arrangement's scales), the Topo h box and the crop buttons sit above the placement topo, the
    LDOS h box above the panel it belongs to (Line: the Line Topo panel; Path: the stripe; Map: the
    Map Topo), the heatmaps of Line / Path and the energy maps of Map are not small (their boxes at
    least 22 % of the canvas height); `want` is the arrangement expected for the cell at the
    design size in the Line view ('a_left' for a tall cell, 'ab_left' for a wide one).  Then the
    crop in the Line view: a cursor dragged on the topo moves and its box follows, a typed box
    moves its cursor, the shade covers the outside, Crop applies the ranges as the topo's limits
    and the arrangement follows the new shape, Uncrop restores the extent.
    """
    from stm_from_dft import simulator as simmod
    d = inputs_dir(name, a, b)
    cmap = plt.LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])
    sim = simmod.Interactive_STM_Simulator(filepath=d, erange=[-3.0, 0.5], ldos_height=1.25, cmap_topo=cmap, unit_cell_num=1)

    def at(fx, fy):
        return [fx * a, fy * b]
    sim.run_interactive(grid_res=10, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', use_decay=True,
                        show_decay_toggle=True, reuse_cache=True, line_endpoints=(at(0.1, 0.1), at(0.9, 0.9)),
                        marker_positions=[at(0.2, 0.2), at(0.6, 0.6)], extra_broadening={'N': 0.0},
                        path=[at(0.1, 0.1), at(0.5, 0.5), at(0.9, 0.9)], path_extend=False)
    fig = sim.fig
    rw.pump(150)
    sim.btn_run.widget.click()                 # RUN: the Line / Path heatmap row and the Map block exist
    rw.pump(100)
    for i, view in enumerate(('Single Point', 'Line', 'Path', 'Map')):
        sim.ui_radio.buttons[i].click()
        rw.pump(150)
        tag = f"{name} cell, {view}"
        report(rw.check_window(fig, f"sim_{name}_{view.replace(' ', '_').lower()}", SHOTS_SHAPE))
        arr = sim._arrangement
        H = fig.canvas.viewport().height()
        print(f"   {tag}: arrangement {arr.layout_name}, blank {arr.blank_fraction():.3f}, scales {[round(v, 2) for v in arr.scales()]}, canvas {fig._canvas_size()}")
        print(f"   {tag}: candidates {arr.explain()}")
        check(f"{tag}: little blank space (< 15 %)", arr.blank_fraction() < 0.15, round(arr.blank_fraction(), 3))
        check(f"{tag}: no group below 45 % of its size alone", all(v >= 0.45 for v in arr.scales()), arr.scales())
        if view == 'Line':
            check(f"{tag}: the arrangement is {want!r}", arr.layout_name == want, arr.layout_name)
        topo_ax = sim.ax_map_global if view == 'Map' else sim.ax_map
        check(f"{tag}: Topo h box and the crop buttons above the placement topo",
              above(fig, sim._box_topo_h, topo_ax._cell) and above(fig, sim._box_crop, topo_ax._cell) and inside(sim._box_topo_h, sim._crop_panel),
              (rect_in(fig, sim._box_topo_h), rect_in(fig, sim._box_crop), rect_in(fig, topo_ax._cell)))
        check(f"{tag}: the cursor boxes ride in the bands of the crop panel", all(inside(b.qt, sim._crop_panel) and b.qt.isVisible() for b in sim._cursor_boxes.values()))
        if view != 'Single Point':
            owner = {'Line': sim.ax_line_topo, 'Path': sim.ax_stripe, 'Map': sim.ax_map}[view]
            check(f"{tag}: LDOS h box above the panel it belongs to", owner is not None and above(fig, sim._box_ldos_h, owner._cell),
                  None if owner is None else (rect_in(fig, sim._box_ldos_h), rect_in(fig, owner._cell)))
            if view == 'Map':
                hs = [ax._px_rect()[3] for ax in sim.map_axes]
                print(f"   {tag}: the map block wraps {sim._map_block.cols} per row; map boxes {round(min(hs))} px tall in a {H} px canvas")
            else:
                hs = [a_._px_rect()[3] for a_ in sim.line_decomp_axes if a_.get_title() == 'Total']
                check(f"{tag}: the heatmap not small (box >= 22 % of the canvas height)", bool(hs) and min(hs) >= 0.22 * H, (hs and round(min(hs)), H))
        else:
            check(f"{tag}: LDOS h box hidden", not sim._box_ldos_h.isVisible())
    # The crop, in the Line view.
    sim.ui_radio.buttons[1].click()
    rw.pump(150)
    ax = sim._crop_ax
    (ex0, ex1), (ey0, ey1) = sim._full_extent()
    check(f"{name}: the cursors start at the full extent of the drawn cells", sim._cursors['x'] == [ex0, ex1] and sim._cursors['y'] == [ey0, ey1]
          and len(sim._cursor_artists) == 4 and not sim._shade_artists, (sim._cursors, (ex0, ex1, ey0, ey1)))
    check(f"{name}: a cursor box shows its cursor's value", sim._cursor_boxes[('x', 0)].text == f"{ex0:.2f}", sim._cursor_boxes[('x', 0)].text)
    # drag the upper x cursor: pick, motion on the topo, release
    art = next(a_ for a_ in sim._cursor_artists if a_.get_label() == '_cursor_x_1')
    ev = types.SimpleNamespace(artist=art, ind=[0], mouseevent=None)
    sim._on_pick(ev)
    check(f"{name}: a pick on a cursor selects it before any marker", sim.active_obj == ('cursor', 'x', 1), sim.active_obj)
    target = ex0 + 0.4 * (ex1 - ex0)
    sim._on_motion(motion(fig, ax, target, 0.5 * (ey0 + ey1)))
    sim._on_rel(None)
    check(f"{name}: the dragged cursor moved and its box followed", abs(sim._cursors['x'][1] - target) < 1e-6 and sim._cursor_boxes[('x', 1)].text == f"{target:.2f}"
          and len(sim._shade_artists) == 1, (sim._cursors['x'], sim._cursor_boxes[('x', 1)].text, len(sim._shade_artists)))
    # a typed box moves the y cursor
    y_hi = ey0 + 0.7 * (ey1 - ey0)
    sim._on_cursor_box('y', 1, f"{y_hi:.4f}")
    check(f"{name}: a typed box moves its cursor", abs(sim._cursors['y'][1] - y_hi) < 1e-6 and len(sim._shade_artists) == 2, (sim._cursors['y'], len(sim._shade_artists)))
    sim._on_cursor_box('y', 1, "abc")
    check(f"{name}: a box text that is not a number is refused", abs(sim._cursors['y'][1] - y_hi) < 1e-6 and sim._cursor_boxes[('y', 1)].text == f"{y_hi:.2f}")
    # Crop: the limits, the cursors at the new edges, the arrangement follows the new shape
    before = sim._arrangement.layout_name
    sim._on_crop()
    rw.pump(300)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    check(f"{name}: Crop applies the cursors' ranges as the topo's limits", abs(xl[1] - target) < 1e-6 and abs(yl[1] - y_hi) < 1e-6
          and np.allclose(sim.crop_x, (ex0, target)) and np.allclose(sim.crop_y, (ey0, y_hi)), (xl, yl, sim.crop_x, sim.crop_y))
    check(f"{name}: after Crop the cursors sit at the new edges with no shade", np.allclose(sim._cursors['x'], [ex0, target]) and not sim._shade_artists, sim._cursors)
    print(f"   {name}: arrangement before crop {before}, after {sim._arrangement.layout_name}; topo box {tuple(round(v) for v in ax._px_rect())}")
    check(f"{name}: no render problem at the design size after the crop", not [p_ for p_ in rw.check_window(fig, f'sim_{name}_line_cropped', SHOTS_SHAPE)['problems'] if p_[0] == 'design'])
    sim._on_uncrop()
    rw.pump(200)
    check(f"{name}: Uncrop restores the full extent", sim.crop_x is None and np.allclose(ax.get_xlim(), (ex0, ex1)) and np.allclose(ax.get_ylim(), (ey0, ey1))
          and sim._cursors['x'] == [ex0, ex1], (ax.get_xlim(), ax.get_ylim(), sim._cursors))
    sim.btn_run.widget.click()                 # STOP
    return sim


def install_decomp_stubs():
    """The wrapper's siblings as stand-ins: PlotConfig, resolve_stm, the two decomposition windows."""
    import mo_ldos_projection  # the real package (its __init__)
    imports = types.ModuleType('mo_ldos_projection._imports')

    class PlotConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def resolve_stm():
        from stm_from_dft import simulator, doscar_parser
        return simulator, doscar_parser
    imports.PlotConfig = PlotConfig
    imports.resolve_stm = resolve_stm
    # synthetic_doscar takes the matcher's Gaussian convention from here (the synthetic-DOSCAR route, not exercised)
    imports._gauss_sum = lambda x, states, sigma: np.zeros_like(x)
    imports._curve_grid = lambda lo, hi, sigma: np.linspace(lo, hi, 2)
    sys.modules['mo_ldos_projection._imports'] = imports
    mo_ldos_projection._imports = imports
    opened = []

    class _Win:
        def __init__(self, run_dir, snapshot, *a, **k):
            self.snapshot = snapshot
            self.fig = plt.figure(figsize=(4, 3))
            opened.append(self)

        def refresh(self, snapshot):
            self.snapshot = snapshot
    dw = types.ModuleType('mo_ldos_projection.decomp_window')
    dw.DecompWindow = type('DecompWindow', (_Win,), {})
    dw.MO_AREA_NOISE_THRESHOLD = 1e-3
    tw = types.ModuleType('mo_ldos_projection.topo_decomp_window')
    tw.TopoDecompWindow = type('TopoDecompWindow', (_Win,), {})
    sys.modules['mo_ldos_projection.decomp_window'] = dw
    sys.modules['mo_ldos_projection.topo_decomp_window'] = tw
    mo_ldos_projection.decomp_window = dw
    mo_ldos_projection.topo_decomp_window = tw
    return opened


def check_wrapper(d):
    opened = install_decomp_stubs()
    from mo_ldos_projection import sim_wrapper as sw
    plt.ioff()
    session = sw.StmDecompSession(d, erange=(-3.0, 0.5), ldos_height=1.25, unit_cell_num=1, sigma=0.2,
                                  sim_kwargs=dict(grid_res=10, topo_bias=0.2, topo_height=2.5, reuse_cache=True,
                                                  marker_positions=[[1.0, 1.0], [3.0, 3.0]]),
                                  use_synthetic_doscar=False)
    sim = session.launch()
    check('launch under ion returns with the window up', sim is not None and sim.fig is not None and plt.isinteractive())
    fig = sim.fig
    w = session._widgets
    check('wrapper boxes in the left column', set(w) == {'sigma_box', 'send_ldos_btn', 'send_topo_btn'} and len(sim._extra_boxes) == 2
          and all(b.isVisible() for b in sim._extra_boxes))
    check('σ box holds the session σ', abs(w['sigma_box'].value - 0.2) < 1e-12)
    # the mode change keeps the boxes (the left column is not rebuilt)
    sim.ui_radio.buttons[1].click()
    rw.pump(100)
    check('view change keeps the wrapper boxes', all(b.isVisible() for b in sim._extra_boxes) and sim.mode == 'Line')
    sim.ui_radio.buttons[0].click()
    rw.pump(100)
    # a σ commit in real-DOSCAR mode: no teardown, σ recorded (on the next event-loop turn)
    type_into(w['sigma_box'].box, '0.300')
    rw.pump(50)
    check('σ commit (real DOSCAR) records the new σ without a teardown', abs(session.sigma - 0.3) < 1e-12 and session.sim is sim)
    type_into(w['sigma_box'].box, '5')
    rw.pump(50)
    check('σ box clamps to SIGMA_MAX', abs(session.sigma - sw.SIGMA_MAX) < 1e-12 and w['sigma_box'].value == sw.SIGMA_MAX)
    # the send buttons: the Decay guard first, then a send
    w['send_ldos_btn'].widget.click()
    check('ldos send refused while Decay is on', not opened)
    sim.chk.boxes[1].click()            # Decay off
    rw.pump(50)
    sim.btn_run.widget.click()          # RUN
    rw.pump(50)
    w['send_ldos_btn'].widget.click()
    check('ldos send opens the decomposition window with the marker spectra',
          len(opened) == 1 and len(opened[0].snapshot['ldos']) == 2 and len(opened[0].snapshot['eg']) > 0)
    w['send_topo_btn'].widget.click()
    check('topo send opens the topo window with every grid point',
          len(opened) == 2 and opened[1].snapshot['ldos'].shape[0] == len(sim.grid_xy))
    report(rw.check_window(fig, 'simulator_with_wrapper', SHOTS))
    # the capture / restore of a σ commit in synthetic mode is the same code path as before; the
    # state capture itself is exercised here
    st = session.capture_state()
    check('capture_state reads the boxes', st['display_cells'] == 1 and st['ldos_gamma'] == 1.0 and st['ctor_kwargs']['erange'] == [-3.0, 0.5])
    plt.close(fig)
    for win in opened:
        plt.close(win.fig)


# ---------------------------------------------------------------------- #
#  The DOS plotter                                                         #
# ---------------------------------------------------------------------- #

def check_dos_plotter(d):
    import plot_dos_cursors_levels as pdl
    plotter = pdl.DosPlotter(d)
    plotter.plot_dos_cursors()
    fig, ax = plotter.fig, plotter.ax
    rw.pump(100)
    check('DOS window: level 0 total DOS', plotter.plot_level == 0 and len(ax.get_lines()) == 1 and ax.get_title() == 'DOS')
    report(rw.check_window(fig, 'dos_plotter', SHOTS))
    total = ax.get_lines()[0]
    xs, ys = total.get_xdata(), total.get_ydata()
    k = int(np.argmax(ys))
    ev = press(fig, ax, float(xs[k]), float(ys[k]))
    fig._pick(ev)
    check('pick descends to level 1 (Au / molecule)', plotter.plot_level == 1 and ax._legend is not None)

    def pick_curve(label):
        # the pick where the named curve is farthest from the other pickable curves
        picks = [l for l in ax.get_lines() if l.get_picker()]
        target = next(l for l in picks if l.get_label() == label)
        x, y = apart([target] + [l for l in picks if l is not target])
        fig._pick(press(fig, ax, x, y))
    pick_curve('Au')
    check('level 2 (atom types)', plotter.plot_level == 2)
    pick_curve('Au')
    check('level 3 (atoms of Au)', plotter.plot_level == 3 and plotter.active_element == 'Au')
    pick_curve('Au1')
    hl = widgets.HoverLabels.of(ax)
    check('level 4 (orbitals of Au1) with hover labels on and two legends',
          plotter.plot_level == 4 and hl is not None and hl.enabled and len(hl.artists) == 9 and len(ax._extra_legends) == 1)
    orb = hl.artists[2]
    xs, ys = orb.get_xdata(), orb.get_ydata()
    k = int(np.argmax(ys))
    widgets.HoverLabels._on_motion(motion(fig, ax, float(xs[k]), float(ys[k])))
    check('hover shows the orbital label', hl._label is not None and hl._label.get_text() == orb.get_label())
    ke = plt.KeyEvent('key_press_event', fig.canvas, 'm')
    fig.canvas._fire('key_press_event', ke)
    check("'m' key does nothing outside SOC mode", plotter.plot_level == 4 and not plotter.show_mag)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ev = press(fig, ax, xl[0] + 0.01 * (xl[1] - xl[0]), yl[1] - 0.01 * (yl[1] - yl[0]))
    fig.canvas._fire('button_press_event', ev)
    check('click beside the curves ascends to level 3', plotter.plot_level == 3)
    report(rw.check_window(fig, 'dos_plotter_level3', SHOTS))
    plt.close(fig)


def main(shape_only=False):
    """Every check; with `shape_only` (the --shape argument, or SHAPE_ONLY=1) the two shape checks alone, for iterating on the layout."""
    d = inputs_dir('base')
    cwd = os.getcwd()
    print(f"inputs in {d}; pymatgen stand-in: {PYMATGEN_STUBBED}")
    steps = [] if shape_only else [(check_simulator, (d,)), (check_wrapper, (d,)), (check_dos_plotter, (d,))]
    steps += [(check_shape, (4.0, 10.0, 'tall', 'a_left')), (check_shape, (10.0, 4.0, 'wide', 'ab_left'))]
    for fn, args in steps:
        try:
            os.chdir(cwd)
            fn(*args)
        except Exception as exc:
            traceback.print_exc()
            check(f'{fn.__name__}{args[1:] if fn is check_shape else ""} runs without an exception', False, exc)
        os.chdir(cwd)
    return rw.summary()


if __name__ == '__main__':
    sys.exit(main(shape_only='--shape' in sys.argv[1:] or os.environ.get('SHAPE_ONLY', '') not in ('', '0')))
