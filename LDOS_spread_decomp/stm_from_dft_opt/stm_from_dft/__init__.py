# -*- coding: utf-8 -*-
"""
stm_from_dft: Unified GPU-accelerated STM simulator from DFT outputs.
Auto-detects ISPIN=1 (collinear unpolarized), ISPIN=2 (collinear polarized),
and LSORBIT/LNONCOLLINEAR (non-collinear SOC) calculations.

wrong_decays/ holds four UNUSED reference files (fully commented out) whose
decay factor is the WKB average over a linearly tilted barrier, the expression
of simulator.py's Decay-on weight (gpu_exact_factor) applied within
the earlier models those files hold; they are not imported here.

Author: Benjamin Kafin
"""

import importlib

__all__ = [
    "SpinAwareDosParser",
    "SpinMode",
    "LocpotManager",
    "Unified_STM_Simulator",
    "Interactive_STM_Simulator",
]

# Each exported name and the submodule that defines it, imported on first access (PEP 562 module
# __getattr__): importing the package, or its qtgui copy (stm_from_dft.qtgui), does not import
# simulator.py (and so not CuPy).
_SUBMODULE_OF = {
    "SpinAwareDosParser": "doscar_parser",
    "SpinMode": "doscar_parser",
    "LocpotManager": "locpot_manager",
    "Unified_STM_Simulator": "simulator",
    "Interactive_STM_Simulator": "simulator",
}


def __getattr__(name):
    submodule = _SUBMODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value
