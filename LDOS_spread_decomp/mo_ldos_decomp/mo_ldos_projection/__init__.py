# -*- coding: utf-8 -*-
"""
mo_ldos_projection: MO→LDOS projection package.

Attributes features of simulated STM LDOS spectra to original gas-phase
molecular orbitals using MOPMatcher's PAW-correct full-wavefunction overlaps,
without any atom/orbital PDOS partitioning of the attribution.

Modules
-------
global_decomposition  Module A: global tri-partition figure (MO curves,
                      orthogonalized-metal channel, residual, total).
procar_dos            Module B: LORBIT=14 NCL PROCAR parser.
synthetic_doscar      Module B: σ-smeared DOSCAR writer + σ-tagged work dirs.
sim_wrapper           Module B: launcher around the unmodified simulator.
decomp_window         Module B: per-marker LDOS + og-MO overlay figure.
topo_decomp_window    Module B: MO topo-decomposition maps (single-color
                      alpha-ramp panels per input offset).

The heavy modules are imported lazily by their consumers; importing this
package is cheap.
"""

__all__ = [
    "global_decomposition",
    "procar_dos",
    "synthetic_doscar",
    "sim_wrapper",
    "decomp_window",
    "topo_decomp_window",
]
