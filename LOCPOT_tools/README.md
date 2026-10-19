# LOCPOT_tools

`efield_from_LOCPOT.py` (`ElectricFieldCalculator`): loads a LOCPOT with pymatgen, computes the
electric field E = -grad(V) on the LOCPOT grid by finite differences with the voxel spacings, and
writes it as LOCPOT/CHGCAR-style volumetric files that VESTA can open: the three components as
separate files (`write_locpot_format_fields`) or one combined vector-field file
(`write_combined_vector_field`), optionally downsampled (`ds`). The header with the atomic
positions is taken from the structure in the LOCPOT; output goes next to the input LOCPOT.

Run block settings: the LOCPOT path given to `ElectricFieldCalculator(...)`, `ds_factor`.
Dependencies: numpy, pymatgen.
