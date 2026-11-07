# -*- coding: utf-8 -*-
"""
Import resolution of the stm_from_dft package (the simulator, its
doscar_parser and its qtgui copy) for the VASP_for_SPM suite layout.

At module level this imports nothing beyond os and sys; each resolver
imports only inside its function. So a caller that needs only the Qt
layer (run_wfn_plane_slice.py) does not load the matcher, which
_imports.py imports at module level. As in _imports.py, the vendored copy
(mo_ldos_projection/stm) is tried first, else stm_from_dft is taken from
stm_from_dft_opt/ two levels above this package, put on sys.path.
_imports.py re-exports the three resolvers, so its callers are unchanged.
"""
import os
import sys


def resolve_stm():
    """
    Return the (simulator, doscar_parser) modules of the stm_from_dft
    package. Deferred import: simulator.py imports cupy and initializes a
    CUDA device at class construction, so only Module B callers pay for it.
    """
    try:
        from .stm import simulator as _simulator
        from .stm import doscar_parser as _doscar_parser
        return _simulator, _doscar_parser
    except ImportError:
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _stm_dir = os.path.join(_root, "stm_from_dft_opt")
        if os.path.isdir(_stm_dir) and _stm_dir not in sys.path:
            sys.path.insert(0, _stm_dir)
        from stm_from_dft import simulator as _simulator
        from stm_from_dft import doscar_parser as _doscar_parser
        return _simulator, _doscar_parser


def resolve_doscar_parser():
    """
    Machinery: return the doscar_parser module of the stm_from_dft package
    (read_doscar_header, SpinAwareDosParser). Found the same way
    resolve_stm() finds it: the vendored copy (mo_ldos_projection/stm)
    first, else stm_from_dft with stm_from_dft_opt/ two levels up on
    sys.path. simulator.py is not imported (stm_from_dft/__init__.py
    imports its submodules only on first access), so neither cupy nor a
    CUDA device comes with it.
    """
    try:
        from .stm import doscar_parser as _doscar_parser
        return _doscar_parser
    except ImportError:
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _stm_dir = os.path.join(_root, "stm_from_dft_opt")
        if os.path.isdir(_stm_dir) and _stm_dir not in sys.path:
            sys.path.insert(0, _stm_dir)
        from stm_from_dft import doscar_parser as _doscar_parser
        return _doscar_parser


def resolve_qt():
    """
    Return the (qtplot, controls, widgets, mplexport) modules of the
    simulator's qtgui copy (stm_from_dft/qtgui), the Qt layer every window
    of this package draws with. Found the same way resolve_stm() finds the
    simulator: the vendored copy (mo_ldos_projection/stm/qtgui) first, else
    stm_from_dft.qtgui with stm_from_dft_opt/ two levels up on sys.path.
    simulator.py is not imported (stm_from_dft/__init__.py imports its
    submodules only on first access), so neither cupy nor a CUDA device
    comes with it.
    """
    try:
        from .stm.qtgui import qtplot as _qtplot
        from .stm.qtgui import controls as _controls
        from .stm.qtgui import widgets as _widgets
        from .stm.qtgui import mplexport as _mplexport
        return _qtplot, _controls, _widgets, _mplexport
    except ImportError:
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _stm_dir = os.path.join(_root, "stm_from_dft_opt")
        if os.path.isdir(_stm_dir) and _stm_dir not in sys.path:
            sys.path.insert(0, _stm_dir)
        from stm_from_dft.qtgui import qtplot as _qtplot
        from stm_from_dft.qtgui import controls as _controls
        from stm_from_dft.qtgui import widgets as _widgets
        from stm_from_dft.qtgui import mplexport as _mplexport
        return _qtplot, _controls, _widgets, _mplexport
