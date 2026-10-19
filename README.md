# VASP_for_SPM

VASP post-processing codes for scanning probe work (STM/STS on molecule/metal systems), collected
as a suite of independent packages. Each top-level folder is its own package with its own README:
run a package's scripts from inside that folder (or with that folder on `sys.path`). Nothing in one
package imports another, except the MO to LDOS decomposition, which uses the matcher and the
simulator through fixed sibling paths (see `LDOS_spread_decomp/README.md`).

| Package | Purpose |
|---|---|
| `LDOS_spread_decomp/MOPmatcher_mem` | MOPMatcher: PAW-correct Bloch-state overlap matching of molecule and metal states against a full system, with the vacuum-aligned, Kramers-paired plots |
| `LDOS_spread_decomp/stm_from_dft_opt` | GPU (CuPy) STM / LDOS simulator from DFT output (constant-current topography, dI/dV maps, line scans) |
| `LDOS_spread_decomp/mo_ldos_decomp` | MO to LDOS projection: attributes simulated LDOS features to the gas-phase molecular orbitals using the two packages above |
| `dIdV_tip_sample` | dI/dV simulation from a sample LDOS and a tip LDOS |
| `PROCAR_tools` | one shared PROCAR reader and the tools built on it: fatband plot, state filter/export, phase-sensitive DOS, GPU broadened orbital curves, band unfolding plot |
| `PROJCAR_tools` | PROJCAR parser and LOCPROJ line generator |
| `WAVECAR_tools` | WAVECAR coefficient parser, G-point facet-duplicate checker, wavefunction orthonormality test |
| `charge_density` | CHGCAR / PARCHG / LOCPOT slicing and plotting, PARCHG naming and projections, planar averages |
| `POSCAR_tools` | POSCAR creation and manipulation (layers, molecules from xyz, dimers, pyramids and tips, subsets, selective dynamics, supercells, unit-cell reshaping) |
| `convergence` | relaxation monitoring from OUTCAR (forces, convergence plot, lowest-force structure) |
| `LOCPOT_tools` | electric field from LOCPOT as volumetric files |

## Dependencies

Python 3 with numpy, scipy and matplotlib everywhere. In addition: pymatgen (`dIdV_tip_sample`,
`LOCPOT_tools`, `WAVECAR_tools`, `MOPmatcher_mem`, `stm_from_dft_opt`), cupy with a CUDA GPU
(`stm_from_dft_opt`, `mo_ldos_decomp`, `PROCAR_tools/procar_spin.py`), h5py (`MOPmatcher_mem`,
`mo_ldos_decomp`), pySBT (`MOPmatcher_mem`, from github.com/QijingZheng/pySBT), mplcursors
(optional, hover labels in `PROCAR_tools/Band_structure_plot.py` and the simulator), pytest
(`mo_ldos_decomp/mo_ldos_projection/tests`). VASPKIT output (BAND.dat, KLABELS) is read by the
band-structure plotters.

## Third-party code

`MOPmatcher_mem` bundles the physics modules of QijingZheng's VaspBandUnfolding (vaspwfc.py,
paw.py, aewfc.py, sph_harm.py, vasp_constant.py; aewfc.py carries a spinor patch for SOC and
sph_harm.py a shim for scipy >= 1.17). `charge_density/lib.py` and the parsers inside the charge
density scripts, `convergence/plot_forces.py` and `dIdV_tip_sample/lib.py` come from Ben McDowell's
repositories (github.com/benwmcdowell) with Benjamin Kafin's edits. Everything else is Benjamin
Kafin's own code.

## Paths in the run blocks

Every script keeps its settings in the `if __name__ == "__main__":` block at the bottom. In the
clean distribution the directory strings there are placeholders that say what the path must point at:

| placeholder | points at |
|---|---|
| `dir/run` | one VASP calculation directory (the files named in the package README) |
| `dir/run/<file>` | a specific file inside such a directory |
| `dir/run_1`, `dir/run_2`, ... | several calculation directories of the same kind |
| `dir/run_complex`, `dir/run_surface`, `dir/run_sam` | the complex, bare-surface and SAM runs of a combined plot |
| `dir/molecule_N` | the N-th isolated-molecule (component) calculation |
| `dir/metal` | the bare metal / adatom-surface calculation |
| `dir/full` | the full (molecule + surface) calculation |
| `dir/sample`, `dir/tip` | the sample and the tip calculation of the dI/dV simulation |
| `dir/system_1`, `dir/system_2` | the two systems of a side-by-side comparison |
| `dir/output`, `dir/output/<file>` | where a script writes its result |
| `dir/package` | the folder of this package when a script adds it to `sys.path` |
