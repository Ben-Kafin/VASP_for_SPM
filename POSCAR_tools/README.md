# POSCAR_tools

| script | does | run block settings |
|---|---|---|
| `Add_layer.py` | adds one bulk-spaced Au(111) layer to the bottom of a slab POSCAR: detects layers and adatoms from z-gaps, derives the surface lattice and ABC stacking from the bottom layer, places the new layer at the Materials Project bulk spacing, rebalances the vacuum so the molecule top sits at c/2, and writes selective dynamics (the two bottom layers fixed) | input/output files, lattice constant and tolerances at the top of the run block |
| `add_xyz_to_poscar.py` | `PoscarModifier`: inserts a molecule from an `.xyz` file into a POSCAR/CONTCAR, centered on one xyz atom and anchored to one POSCAR atom with a translation offset | `xyz_file`, `poscar_file`, `output_file`, `subtract_index`, `anchor_index`, `translation_offset`, `merge_order` |
| `flatten_NHC_iPr_dimers.py` | flattens NHC-Au-NHC dimers on Au(111): backbone planes, carbene planarization, iPr wingtip flap and twist, dimer tilt and bend, optional selective dynamics by height | `poscar_in`, `poscar_out`, the angles, `sd_enable`, `sd_z_height` |
| `Make_pyramid_any_number_sides.py` | `PyramidFilter`: carves a pyramid with any number of base vertices out of a slab by per-layer cross sections | input/output POSCAR, apex and base coordinates |
| `Make_trigonal_pyramid_POSCAR.py` | `PyramidFilter`: carves a trigonal pyramid by plane cuts from apex and base | input/output POSCAR, apex and base coordinates |
| `make_subsets.py` | from a complete slab + adatoms + NHC calculation directory, detects the molecules and adatoms by composition and connectivity and creates one subdirectory per subset (`adatom_surface`, `lone_adatom_k`, `NHC_k`, `NHC<m>Au_complex_k`), each a complete input set: POSCAR, INCAR (MAGMOM rewritten), KPOINTS, POTCAR, job.sh | `calc_dir` and the chemistry constants in the run block |
| `SeldynAdder.py` | `SeldynAdder`: adds selective dynamics flags by z window (atoms between `z_minimum` and `z_maximum` free) | `ifile`, `ofile`, `z_minimum`, `z_maximum` |
| `supercell_layer_remover.py` | `SupercellLayerRemover`: builds an (nx, ny, nz) supercell keeping selective dynamics flags, then removes atoms given as (layer index, x, y) | `input_file`, `output_file`, `supercell`, `remove_list` |
| `tip_maker.py` | crystalline pyramid tip generator: bulk layers plus a nearest-neighbour-cascade pyramid under an apex atom centered in the cell | the constants in its run block |
| `unitcell.py` | `POSCARTransformer`: recasts a structure into a new unit cell (new lattice vectors, atoms outside [0,1) dropped, counts updated) keeping selective dynamics flags; `mode='origin'` or `'centered'`; `fixed_width=True` writes the fixed-width VASP column format | input/output files, lattice vectors, `mode`, `retain_selective_dynamics`, `fixed_width` |

All scripts read and write POSCAR / CONTCAR (plus `.xyz` for `add_xyz_to_poscar.py`).
Dependencies: numpy (scipy for some).
