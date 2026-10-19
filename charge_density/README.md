# charge_density

Slicing and plotting of VASP volumetric data. `lib.py` is Ben McDowell's parser library
(`parse_CHGCAR`, `parse_LOCPOT`, `parse_poscar`, `parse_doscar`, `parse_eigenval`, ...) that the
scripts import; keep it in this folder.

| script | does | run block settings |
|---|---|---|
| `lv_slice.py` | `PlaneSlice`: 2D slice of CHGCAR (or a difference density) on a plane fixed along one lattice vector at a fractional position, with interpolation | `vasp_dir`, `axis`, `fraction`, `filetype` |
| `lv_slice_w_atoms_2system.py` | the same plane slice with the atoms drawn, for one system or two systems side by side | `vasp_dir_a`, `vasp_dir_b`, `axis`, `fraction`, half-widths, `elem_colors` |
| `path_slice_horizontal.py` | `slice_path`: slice of charge density or electrostatic potential along a user path through atom indices (as in VESTA), vertical, horizontal and draped views | path atoms, `filetype`, `dipole_correction` |
| `point_cloud_slicer_horiz_vert.py` | `PointCloudSlicer`: horizontal (refined around the path) and linear vertical slices for two systems compared side by side | `vasp_dir1`, `path1`, `vasp_dir2`, `path2`, `tol`, `padding`, colors |
| `add_or_remove_energy_to_or_from_PARCHG_name.py` | renames `PARCHG.NNNN.*` files to carry the band energy relative to the Fermi level (from EIGENVAL and DOSCAR), or strips it again | the run directory |
| `calculate_PARCHG_projections.py` | for every energy-named PARCHG of a reference run, computes the normalized overlap with each PARCHG of a second run (plus its mean z position) and prints the matches above 0.9 with their energy shift | `ref`, `fp` |
| `chgcar_locpot_zavg.py` | planar-averaged density ρ(z) from CHGCAR / AECCAR0 / AECCAR2 (total, and spin or magnetization blocks when present), with an optional planar-averaged LOCPOT potential on a second axis | the paths and options in the run block |

Reads CHGCAR, PARCHG, AECCAR, LOCPOT, POSCAR/CONTCAR, DOSCAR, EIGENVAL. Dependencies: numpy,
scipy, matplotlib.
