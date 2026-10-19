# WAVECAR_tools

| script | does | run block settings |
|---|---|---|
| `wavecar_parser_force.py` | `WavecarParser`: parses WAVECAR + EIGENVAL into plane-wave coefficients, G-points and energies, caches them to `.npz` (with a `force` re-parse flag), writes `Gpoints.txt`, and can generate synthetic all-ones or delta coefficient sets for tests | `directory`, `vasp_version` |
| `gpoint_checker_gpts.py` | `GpointChecker`: loads the two spin blocks of a `Gpoints.txt` (or takes an array), finds the periodic-facet duplicates on each axis and removes them, keeping the removed indices and the filtered blocks | the `Gpoints.txt` path |
| `wave_orthotest_gpts_force_norm.py` | `WavefunctionOrthonormality`: runs the parser, removes the facet duplicates with the checker, filters the coefficients, builds the |<psi_m|psi_n>|^2 overlap matrices per block (optionally normalized), prints a summary and saves npz/tsv | `directory`, `vasp_version`, `cache`, `use_delta`, `normalize`, `out_npz` |

Reads WAVECAR, EIGENVAL, `Gpoints.txt`. Dependencies: numpy, pymatgen (parser).
