# dIdV_tip_sample

dI/dV simulation from a sample LDOS and a tip LDOS (`dIdV_sim.py`, class `dIdV_sim`), with
`lib.py` (parsers from Ben McDowell's code: `parse_doscar`, `parse_poscar`, `parse_potcar`,
`parse_bader_ACF`, `tunneling_factor`).

For each of the two calculations the class parses DOSCAR/POSCAR, computes the LDOS at a chosen
position (optionally written to and re-read from an LDOS file), smears it, aligns the Fermi
levels, and the tip object's `calculate_didv` convolves tip and sample LDOS over the bias range
into a dI/dV curve, with optional zero-bias shift, resampling and normalization.

Run block settings: `filepath_sys` (sample run: DOSCAR, POSCAR, LOCPOT), `filepath_tip` (tip
run), the sampling position and energy window of each LDOS, `vmin`, `vmax`, `npts` for the bias
range. The script adds its own folder to `sys.path` to import `lib.py`.

Dependencies: numpy, scipy, matplotlib, pymatgen.
