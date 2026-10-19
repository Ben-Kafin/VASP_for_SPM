# PROJCAR_tools

| script | does | run block settings |
|---|---|---|
| `PROJCAR_parser.py` | `ProjcarParser`: parses PROJCAR (site markers with `R=` fractional coordinates, k-point / spin blocks, per-band projection rows), finds and parses EIGENVAL, and writes `final_output_by_kpoint.txt` with band energies and per-site projections | `projcar_directory` |
| `locproj_aggregator.py` | `LocprojAggregator`: uses the parser to collect the non-zero projected orbitals per site, groups sites by identical orbital sets and prints LOCPROJ lines for INCAR | `projcar_dir` |

Reads PROJCAR and EIGENVAL. Dependencies: numpy.
