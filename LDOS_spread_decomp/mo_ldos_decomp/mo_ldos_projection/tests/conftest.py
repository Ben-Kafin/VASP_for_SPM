# -*- coding: utf-8 -*-
"""Shared fixtures: paths, parsed PROCAR, matcher results (session-scoped).

The windows draw with the shared Qt layer (qtgui at the repository root): the suite runs on
Qt's offscreen platform (QT_QPA_PLATFORM=offscreen, set here when nothing is set) and the
layer's event loop is marked running so a window's plt.show() returns at once."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# Make the package importable from the repository/distribution root.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)
_ROOT = os.path.dirname(_PKG_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# The shared Qt layer: the folder holding qtgui/ (the repository root), found by walking up.
_QT_ROOT = _ROOT
while not os.path.isdir(os.path.join(_QT_ROOT, "qtgui")) and os.path.dirname(_QT_ROOT) != _QT_ROOT:
    _QT_ROOT = os.path.dirname(_QT_ROOT)
if _QT_ROOT not in sys.path:
    sys.path.insert(0, _QT_ROOT)
from qtgui import qtplot as _qtplot          # noqa: E402
_qtplot._RUNNING[0] = True                   # plt.show() returns at once under pytest

EXAMPLE_DIR = os.environ.get(
    "MO_LDOS_EXAMPLE_DIR",
    os.path.join(_ROOT, "MOPM_VASP_Example", "wvchg"))


def _require(path: str) -> str:
    if not os.path.exists(path):
        pytest.skip(f"example file not present: {path}")
    return path


@pytest.fixture(scope="session")
def example_dir():
    if not os.path.isdir(EXAMPLE_DIR):
        pytest.skip(f"example directory not present: {EXAMPLE_DIR} "
                    "(set MO_LDOS_EXAMPLE_DIR)")
    return EXAMPLE_DIR


@pytest.fixture(scope="session")
def procar(example_dir):
    from mo_ldos_projection.procar_dos import parse_procar
    return parse_procar(_require(os.path.join(example_dir, "PROCAR")))


@pytest.fixture(scope="session")
def doscar_header(example_dir):
    from mo_ldos_projection.synthetic_doscar import read_doscar_header
    return read_doscar_header(_require(os.path.join(example_dir, "DOSCAR")))


@pytest.fixture(scope="session")
def results(example_dir):
    from mo_ldos_projection.global_decomposition import load_results
    path = _require(os.path.join(example_dir, "matcher_results.h5"))
    data = load_results(path)
    if data is None:
        pytest.skip("matcher_results.h5 unreadable")
    return data


@pytest.fixture(scope="session")
def prepared(example_dir):
    """RectAEPAWColorPlotter._prepare bundle with the example's run config."""
    from mo_ldos_projection._imports import (
        RectAEPAWColorPlotter, PlotConfig)
    cfg = PlotConfig(shared_molecule_color=True, soc_pair_thresh_eV=0.005)
    plotter = RectAEPAWColorPlotter(cfg)
    main = os.path.join(example_dir, "band_matches_rectangular.txt")
    _require(main)
    P = plotter._prepare(main, bonding=False, write_summaries=False)
    return P, cfg, plotter


@pytest.fixture(scope="session")
def synth_doscar(tmp_path_factory, procar, doscar_header):
    """A sigma=0.2 synthetic DOSCAR built once for the session."""
    from mo_ldos_projection.synthetic_doscar import build_synthetic_doscar
    out = tmp_path_factory.mktemp("synth") / "DOSCAR"
    info = build_synthetic_doscar(procar, doscar_header, 0.2, str(out))
    return str(out), info


def gpu_available():
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False
