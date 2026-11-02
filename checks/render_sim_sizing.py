"""
The simulator's sizing on synthetic processed data (checks/sim_synthetic.py): every view (Single
Point, Line, Path, Map) at the design size and in a wide and a tall window, one screenshot each, and
one line each: the arrangement chosen, the spectrum panel's size, one energy map's size (Map), the
heatmap row's height (Line, Path) and the blank fraction.

    QT_QPA_PLATFORM=offscreen python3 -m checks.render_sim_sizing <out_dir> [tag]

run from the repository's root.
"""
import os
import sys

from checks import sim_synthetic as ss
from checks import render_windows as rw
from qtgui import qtplot as plt

VIEWS = ('Single Point', 'Line', 'Path', 'Map')
SIZES = (('design', 20, 14), ('wide', 28, 11), ('tall', 14, 18))


def _rect(w):
    g = w.geometry()
    return g.width(), g.height()


def measure(sim):
    """(arrangement, spectrum (w, h), one map (w, h) or None, heatmap row height or None, blank fraction)."""
    arr = sim._arrangement
    spec = sim.ax_spec._px_rect() if getattr(sim, 'ax_spec', None) is not None else None
    one_map = None
    if sim.mode == 'Map' and getattr(sim, 'map_axes', None):
        r = sim.map_axes[0]._px_rect()
        one_map = (round(r[2]), round(r[3]))
    row_h = None
    if sim.mode in ('Line', 'Path') and getattr(sim, 'line_decomp_axes', None):
        row_h = round(sim.line_decomp_axes[0]._px_rect()[3])
    return (arr.layout_name if arr is not None else None,
            (round(spec[2]), round(spec[3])) if spec else None, one_map, row_h,
            round(arr.blank_fraction(), 3) if arr is not None else None)


def main(out, tag='now'):
    os.makedirs(out, exist_ok=True)
    sim = ss.open_simulator()
    rw.pump(200)
    sim.btn_run.widget.click()
    rw.pump(200)
    for view in VIEWS:
        sim._on_mode_change(view)
        rw.pump(200)
        for name, w_in, h_in in SIZES:
            rw.resize(sim.fig, w_in, h_in)
            rw.pump(250)
            lay, spec, one_map, row_h, blank = measure(sim)
            win = sim.fig.canvas.window()
            print(f"{tag:6s} {view:12s} {name:6s} window {win.width()}x{win.height()}  arrangement {lay}  "
                  f"spectrum {spec}  one map {one_map}  heatmap row h {row_h}  blank {blank}", flush=True)
            sim.fig.savefig(os.path.join(out, f"{tag}_{view.replace(' ', '_').lower()}_{name}.png"))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'shots_sizing', sys.argv[2] if len(sys.argv) > 2 else 'now')
