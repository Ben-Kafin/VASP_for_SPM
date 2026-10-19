# mo_ldos_decomp (MO to LDOS projection)

Attributes features of simulated STM LDOS spectra and topographies to the original gas-phase
molecular orbitals, using MOPMatcher's PAW-correct full-wavefunction overlaps (no atom/orbital PDOS
partitioning). Package `mo_ldos_projection/` plus the run scripts next to it.

The package reaches the matcher and the simulator through `mo_ldos_projection/_imports.py`, which
expects `../MOPmatcher_mem/` and `../stm_from_dft_opt/` next to this folder (two levels above the
package). Keep the three folders side by side.

## Run scripts (edit the settings at the top of each `__main__` block)

| script | does |
|---|---|
| `run_stm_decomp.py` | single entry point: checks for an extended `matcher_results.h5` in the full directory (runs `run_match` if needed), then launches the simulator wrapper with the σ slider and the "Send to decomposition" box that opens the LDOS and topo decomposition windows. Settings: `molecule_dirs`, `metal_dir`, `full_dir`, `run_kwargs`, `sigma`, `mo_area_threshold`, `topo_decomp_offsets` |
| `run_mo_band_lists.py` | classifies full-system bands by one MO's standing among its component's MOs and writes the vaspkit job that generates the qualifying real-space squared wavefunctions (the SLURM account and mail lines in the job template are placeholders in the clean copy) |
| `run_wfn_mo_sum.py` | MO-weighted sum of the vaspkit real-space squared wavefunctions, written as one CHGCAR-format probability map |
| `run_wfn_plane_slice.py` | vertical plane slice through that sum (or any CHGCAR-format file) |

## Package modules

| module | role |
|---|---|
| `global_decomposition.py` | Module A: global tri-partition figure (MO subspace, MO-orthogonalized metal, residual) |
| `procar_dos.py` | Module B: strict LORBIT=14 noncollinear single-Γ PROCAR reader with validation (used only here; the general reader lives in `PROCAR_tools`) |
| `synthetic_doscar.py` | σ-smeared synthetic DOSCAR writer and σ-tagged working directories |
| `sim_wrapper.py` | launcher around the unmodified simulator with the σ slider and the decomposition buttons |
| `decomp_window.py` | per-marker LDOS + MO decomposition figure |
| `topo_decomp_window.py` | MO topo-decomposition maps |
| `_imports.py` | import resolution of the matcher and simulator packages |
| `tests/` | pytest suite (`conftest.py` takes the example directory from `MO_LDOS_EXAMPLE_DIR`) |

Inputs: the MOPMatcher outputs (`matcher_results.h5`), PROCAR (LORBIT=14), DOSCAR, LOCPOT,
POSCAR/CONTCAR of the full run, vaspkit real-space wavefunction files. Outputs: decomposition
figures, synthetic DOSCARs in σ-tagged working directories, CHGCAR-format maps.

Dependencies: those of `MOPmatcher_mem` and `stm_from_dft_opt`, plus pytest for the tests.
