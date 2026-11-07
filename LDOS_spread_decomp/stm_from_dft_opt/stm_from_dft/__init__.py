# -*- coding: utf-8 -*-
"""
stm_from_dft: Unified GPU-accelerated STM simulator from DFT outputs.
Auto-detects ISPIN=1 (collinear unpolarized), ISPIN=2 (collinear polarized),
and LSORBIT/LNONCOLLINEAR (non-collinear SOC) calculations.

wrong_decays/ holds four UNUSED reference files (fully commented out) whose
decay factor is the WKB average over a linearly tilted barrier, not the model
of Chen's book used by simulator.py; they are not imported here.

Author: Benjamin Kafin
"""

from .doscar_parser import SpinAwareDosParser, SpinMode
from .locpot_manager import LocpotManager
from .simulator import Unified_STM_Simulator, Interactive_STM_Simulator

__all__ = [
    "SpinAwareDosParser",
    "SpinMode",
    "LocpotManager",
    "Unified_STM_Simulator",
    "Interactive_STM_Simulator",
]
