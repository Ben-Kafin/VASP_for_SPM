# -*- coding: utf-8 -*-
"""Section 10.8 — wrapper: σ commit produces a new tagged dir,
re-instantiates, and restores every captured UI element and marker;
Decay-on press warns and does nothing. The live-simulator parts need a
CUDA GPU and the full example inputs; they skip cleanly elsewhere."""
import os

import numpy as np
import pytest

from .conftest import gpu_available


def test_decay_on_press_warns_and_does_nothing(capsys):
    """The Decay gate needs no GPU: exercise send_to_decomposition on a
    minimal stand-in simulator object."""
    from mo_ldos_projection.sim_wrapper import StmDecompSession

    class _FakeSim:
        use_decay = True
        cached_eg = None

    session = StmDecompSession.__new__(StmDecompSession)  # no heavy init
    session.sim = _FakeSim()
    session.decomp_window = None
    out = session.send_to_decomposition()
    assert out is None
    assert session.decomp_window is None
    text = capsys.readouterr().out
    assert "projection LDOS requires energy decay off" in text


def test_topo_decay_on_press_warns_and_does_nothing(capsys):
    """The topo send shares the Decay gate: warns and sends nothing."""
    from mo_ldos_projection.sim_wrapper import StmDecompSession

    class _FakeSim:
        use_decay = True

    session = StmDecompSession.__new__(StmDecompSession)  # no heavy init
    session.sim = _FakeSim()
    session.topo_window = None
    out = session.send_topo_to_decomposition()
    assert out is None
    assert session.topo_window is None
    text = capsys.readouterr().out
    assert "projection LDOS requires energy decay off" in text


def test_topo_stale_topography_warns_and_does_nothing(capsys):
    """A dirty (decay/broadening-stale) global topography is refused."""
    from mo_ldos_projection.sim_wrapper import StmDecompSession

    class _FakeSim:
        use_decay = False
        _broadening_topo_dirty = True

    session = StmDecompSession.__new__(StmDecompSession)  # no heavy init
    session.sim = _FakeSim()
    session.topo_window = None
    out = session.send_topo_to_decomposition()
    assert out is None
    assert session.topo_window is None
    assert "stale" in capsys.readouterr().out


def test_real_doscar_mode_skips_synthetic_machinery(example_dir):
    """use_synthetic_doscar=False (session input, not a figure toggle):
    no PROCAR parse, the simulator's working directory IS the run
    directory, and a σ commit updates only the spread-curve σ with no
    teardown/re-instantiation."""
    from mo_ldos_projection.sim_wrapper import StmDecompSession

    session = StmDecompSession(
        example_dir, sigma=0.20, use_synthetic_doscar=False)
    assert session.procar is None
    assert os.path.normcase(session._prepare_workdir()) == \
        os.path.normcase(os.path.abspath(example_dir))

    sentinel = object()
    session.sim = sentinel            # any teardown would replace this
    session.commit_sigma(0.35)
    assert session.sigma == 0.35
    assert session._sigma_pending == 0.35
    assert session.sim is sentinel    # no teardown, no re-instantiation


def test_synthetic_mode_still_parses_procar(example_dir):
    from mo_ldos_projection.sim_wrapper import StmDecompSession
    session = StmDecompSession(
        example_dir, sigma=0.20, use_synthetic_doscar=True)
    assert session.procar is not None
    wd = session._prepare_workdir()
    assert wd.endswith(f"sigma_{0.20:.4f}")
    assert os.path.isfile(os.path.join(wd, "DOSCAR"))


@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU required")
def test_sigma_commit_new_dir_reinstantiate_restore(example_dir):
    """Full σ-commit round trip on the example run with a small grid.
    Verifies: a new σ-tagged dir exists, the simulator was re-instantiated
    on it, and every captured UI element and the markers were restored."""
    import matplotlib.pyplot as plt
    from mo_ldos_projection.sim_wrapper import StmDecompSession
    from mo_ldos_projection._imports import PlotConfig

    session = StmDecompSession(
        example_dir,
        erange=(-2.0, 0.5),
        ldos_height=1.25,
        unit_cell_num=2,
        sigma=0.20,
        cfg=PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005),
        sim_kwargs=dict(
            grid_res=12, topo_bias=0.2, topo_height=2.5,
            use_decay=False, show_decay_toggle=True, reuse_cache=True,
            marker_positions=[[12.8, 9.7], [21.5, 17.8]],
        ),
    )
    sim1 = session.launch()
    wd1 = os.path.join(example_dir, "mo_ldos_work", "sigma_0.2000")
    assert os.path.isdir(wd1)
    assert os.path.isfile(os.path.join(wd1, "DOSCAR"))
    assert os.path.normcase(os.path.abspath(sim1.filepath)) == \
        os.path.normcase(os.path.abspath(wd1))

    # Adjust some UI state, then commit a new σ.
    sim1.s_emin.set_val(-1.5)
    sim1.s_ldos_gamma.set_val(0.5)
    sim1.normalize = True
    markers_before = [list(pt) for pt in sim1.marker_coords]

    session.commit_sigma(0.30)
    sim2 = session.sim
    wd2 = os.path.join(example_dir, "mo_ldos_work", "sigma_0.3000")
    assert sim2 is not sim1
    assert os.path.isdir(wd2)
    assert os.path.isfile(os.path.join(wd2, "DOSCAR"))
    assert os.path.normcase(os.path.abspath(sim2.filepath)) == \
        os.path.normcase(os.path.abspath(wd2))
    # restored state
    assert abs(sim2.s_emin.val - (-1.5)) < 1e-9
    assert abs(sim2.s_ldos_gamma.val - 0.5) < 1e-9
    assert sim2.normalize is True
    assert sim2.use_decay is False
    assert [list(pt) for pt in sim2.marker_coords] == markers_before
    plt.close("all")


@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU required")
def test_send_to_decomposition_snapshot(example_dir):
    """Decay off + computed spectra ⇒ the window opens with one LDOS curve
    per marker over the current energy window."""
    import matplotlib.pyplot as plt
    from mo_ldos_projection.sim_wrapper import StmDecompSession
    from mo_ldos_projection._imports import PlotConfig

    session = StmDecompSession(
        example_dir,
        erange=(-2.0, 0.5),
        ldos_height=1.25,
        unit_cell_num=2,
        sigma=0.20,
        cfg=PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005),
        sim_kwargs=dict(
            grid_res=12, topo_bias=0.2, use_decay=False,
            marker_positions=[[12.8, 9.7], [21.5, 17.8]],
        ),
    )
    sim = session.launch()
    sim._toggle_run(None)          # RUN: compute spectra
    win = session.send_to_decomposition()
    assert win is not None
    assert len(win.markers) == len(sim.marker_coords)
    assert len(win.ldos_curves) == len(win.markers)
    assert win.eg.shape == np.asarray(sim.cached_eg).shape
    assert win.sigma == 0.20
    # snapshot semantics: pressing again regenerates the same window object
    win2 = session.send_to_decomposition()
    assert win2 is win
    plt.close("all")


@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU required")
def test_send_topo_to_decomposition_snapshot(example_dir):
    """Decay off ⇒ the topo window opens straight from launch (no RUN, no
    markers): one cell per input offset per molecule component, per-point
    projections over the full topo grid on the topo bias window."""
    import matplotlib.pyplot as plt
    from mo_ldos_projection.sim_wrapper import StmDecompSession
    from mo_ldos_projection._imports import PlotConfig

    session = StmDecompSession(
        example_dir,
        erange=(-2.0, 0.5),
        ldos_height=1.25,
        unit_cell_num=2,
        sigma=0.20,
        cfg=PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005),
        sim_kwargs=dict(
            grid_res=12, topo_bias=0.2, use_decay=False,
            marker_positions=[[12.8, 9.7], [21.5, 17.8]],
        ),
        topo_decomp_offsets=[-1, 0, 1],
    )
    sim = session.launch()
    win = session.send_topo_to_decomposition()
    assert win is not None
    assert len(win.cells) == 3 * len(win.mol_labels)
    assert win.grid_xy.shape[0] == 12 * 12
    assert win.topo_bias == sim.global_topo_bias
    lo, hi = sorted([0.0, sim.global_topo_bias])
    assert win.eg[0] >= lo - 1e-6 and win.eg[-1] <= hi + 1e-6
    found = [c for c in win.cells if c["p"] is not None]
    assert found
    for c in found:
        assert c["p"].shape == (12 * 12,)
        assert np.all(np.isfinite(c["p"]))
    # snapshot semantics: pressing again regenerates the same window object
    win2 = session.send_topo_to_decomposition()
    assert win2 is win
    plt.close("all")
