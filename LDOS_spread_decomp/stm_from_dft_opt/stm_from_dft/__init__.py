# -*- coding: utf-8 -*-
"""
stm_from_dft: Unified GPU-accelerated STM simulator from DFT outputs.
Auto-detects ISPIN=1 (collinear unpolarized), ISPIN=2 (collinear polarized),
and LSORBIT/LNONCOLLINEAR (non-collinear SOC) calculations.

tier3_decay.py and sim_decay_full.py in this package are UNUSED reference
files (fully commented out, see their banners) for the Tier 3 tunneling-
barrier decay work; they are not imported here.

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
