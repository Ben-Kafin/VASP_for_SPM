# LDOS_spread_decomp

Three packages that work together on the same VASP calculations:

| folder | package | run from |
|---|---|---|
| `MOPmatcher_mem/` | MOPMatcher, the Bloch-state overlap matcher (`matcher/` package + physics modules) | `MOPmatcher_mem/run_MOPMatcher.py` and the other `run_*.py` there |
| `stm_from_dft_opt/` | the GPU STM / LDOS simulator (`stm_from_dft/` package) | `stm_from_dft_opt/run_DFT_STM.py` |
| `mo_ldos_decomp/` | the MO to LDOS projection (`mo_ldos_projection/` package) | `mo_ldos_decomp/run_stm_decomp.py` and the other `run_*.py` there |

The three folders must stay side by side with these names: `mo_ldos_projection/_imports.py`
finds `MOPmatcher_mem/` and `stm_from_dft_opt/` two levels above itself (next to the
`mo_ldos_decomp/` folder). `CHANGE_REPORT.md` and `CHANGES_DIFFS.patch` document the changes made
to the matcher and the decomposition when the extended matcher record (grid amplitudes and cross
overlaps) was introduced; they are history, not instructions.

Each of the three folders has its own README.
