# -*- coding: utf-8 -*-
"""
Internal import resolution for the VASP_for_SPM suite layout.

In the suite the package has no vendored copies of the matcher or the
simulator: the single working copies MOPmatcher_mem/ and stm_from_dft_opt/
sit next to the wrapper folder that holds mo_ldos_projection/
(LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection), i.e. two levels
above this package. The vendored-copy imports (mo_ldos_projection/mopm,
mo_ldos_projection/stm) are still tried first, so a standalone copy with
vendored packages keeps working, and otherwise the sibling copies two
levels up are used; every other module in the package imports from ONE
place and never duplicates the logic.

_gauss_sum and _curve_grid are re-exported from the matcher's spread_plot —
the single shared implementation every consumer must use (one Gaussian
convention, never reimplemented).
"""
from __future__ import annotations

import os
import sys

from ._resolve import resolve_stm, resolve_qt, resolve_doscar_parser

try:
    # Standalone layout: vendored copies inside the package (absent in the suite).
    from .mopm.matcher.vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
    )
    from .mopm.matcher.spread_plot import _gauss_sum, _curve_grid
    from .mopm.matcher.kramers import (
        Entity,
        pair_entities,
        homo_entity_offsets,
        combine_pair_weights,
    )
    from .mopm.matcher.builder import detect_ncl_from_incar
except ImportError:
    # Suite layout: MOPmatcher_mem/ two levels up, next to the wrapper folder.
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _mopm_dir = os.path.join(_root, "MOPmatcher_mem")
    if os.path.isdir(_mopm_dir) and _mopm_dir not in sys.path:
        sys.path.insert(0, _mopm_dir)
    from matcher.vacupot_plotter import (
        RectAEPAWColorPlotter,
        PlotConfig,
    )
    from matcher.spread_plot import _gauss_sum, _curve_grid
    from matcher.kramers import (
        Entity,
        pair_entities,
        homo_entity_offsets,
        combine_pair_weights,
    )
    from matcher.builder import detect_ncl_from_incar


def resolve_run_match():
    """
    Return the matcher entry point run_match (builder + matcher pipeline).
    Deferred import: pulling in matcher_vacupot drags the whole physics
    stack (vaspwfc, aewfc, pymatgen, ...) along, so only the callers that
    actually run the pipeline pay for it.
    """
    try:
        from .mopm.matcher.matcher_vacupot import run_match
    except ImportError:
        from matcher.matcher_vacupot import run_match
    return run_match


def resolve_blochstates_check():
    """
    Return (TrueBlochStateBuilder, whitener_uid_from_file) for the
    _blochstates_valid reuse check (deferred, same reason as above).
    """
    try:
        from .mopm.matcher.builder import (
            TrueBlochStateBuilder,
            whitener_uid_from_file,
        )
    except ImportError:
        from matcher.builder import (
            TrueBlochStateBuilder,
            whitener_uid_from_file,
        )
    return TrueBlochStateBuilder, whitener_uid_from_file
