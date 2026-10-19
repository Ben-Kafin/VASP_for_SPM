# -*- coding: utf-8 -*-
"""Shared fixtures: paths, parsed PROCAR, matcher results (session-scoped)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

# Make the package importable from the repository/distribution root.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)
_ROOT = os.path.dirname(_PKG_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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
