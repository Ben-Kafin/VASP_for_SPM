# convergence

Relaxation monitoring from VASP's OUTCAR. Built on Ben McDowell's monitor_VASP_convergence
(`plot_forces.py` is his script with Benjamin Kafin's edits; `forces.py` and
`lowest_force_poscar.py` are Benjamin Kafin's).

plot_forces.py:
Plots the evolution of forces from VASP's OUTCAR file relative to the convergence criteria. Useful for analyzing optimization progression between or during job submissions.

By default, forces will be read from 'OUTCAR' and plotted. Alternative files can be specified with -o, --outcar [OUTCAR path] or -p, --poscar [POSCAR path]. The x axis is the optimization time in fs. For molecular dynamics runs (IBRION=0) every step is POTIM long. For conjugate-gradient relaxations (IBRION=2) the time step is not constant: each point is placed using the actual trial and line-minimum step sizes recorded in the OUTCAR (in units of POTIM), so the spacing reflects the step each ionic move really took. For relaxations without per-step line-search output (e.g. IBRION=1), POTIM per step is used. If multiple job submissions are contained within the OUTCAR file (i.e. the VASP run was continued without purging output files from previous runs), the convergence criteria plotted will correspond to the EDIFFG parameter specified in the most recent job. If selective dynamics is employed in the POSCAR file, only forces along optimizable degrees of freedom will be plotted, since frozen degrees of freedom do not contribute to the force convergence threshold.

Example of intended usage: python plot_forces or python plot_forces -o ./my_OUTCAR -p ./my_POSCAR
(-q, --quiet suppresses the quartile curves). `default_dir` in the run block sets the directory used when no options are given.

forces.py:
Interactive 3D view of the atoms and their force vectors for every ionic step of the OUTCAR, with the force-convergence plot of plot_forces as a second panel and a slider (or the time marker on the convergence panel) to step through the relaxation. Arrow lengths follow a single asinh mapping of the force magnitude. If selective dynamics is enabled, only optimizable degrees of freedom are shown. Options: -o, --outcar, -p, --poscar, -a, --all (show all forces, not only those above EDIFFG). `default_dir` in the run block sets the default directory.

lowest_force_poscar.py:
Reads every ionic step of the OUTCAR, finds the step with the lowest force by the chosen metric ('max', 'average' or 'sum' over the free degrees of freedom), and writes that geometry as a new POSCAR (selective dynamics flags kept). Settings in the run block: `calc_dir`, `metric`; output is `POSCAR_lowest_force` in that directory.

Compatible with VASP 5.4.4 and later OUTCAR formats; Python 3 with numpy and matplotlib.
