import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Patch
from matplotlib.colors import SymLogNorm

from lib import parse_CHGCAR, parse_LOCPOT, parse_poscar

class PlaneSlice:
    """
    Load a VASP directory containing:
      • CONTCAR or POSCAR         (parsed by parse_poscar)
      • CHGCAR_diff*              (parsed by parse_CHGCAR)
      • or LOCPOT                 (parsed by parse_LOCPOT)
    Then extract and plot a 2D slice at a fractional position along one axis,
    overlaying atoms that lie within a small fractional tolerance.
    """

    def __init__(self,
                 vasp_dir,
                 filetype='CHGCAR',
                 cmap='seismic',
                 tol=0.02,                 # now in Å, not fraction
                 element_colors=None):
        self.cmap      = cmap
        self.filetype  = filetype
        self.tol_ang   = tol       # store tolerance in Å

        # 1) parse lattice + atom positions from CONTCAR or POSCAR
        contcar = os.path.join(vasp_dir, "CONTCAR")
        if not os.path.isfile(contcar):
            contcar = os.path.join(vasp_dir, "POSCAR")
            if not os.path.isfile(contcar):
                raise FileNotFoundError(f"No CONTCAR or POSCAR in {vasp_dir}")
        self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(contcar)[:4]

        # 2) parse volumetric data
        if 'CHG' in filetype:
            # find CHGCAR_diff*.vasp preferentially, else any CHGCAR*diff*
            all_diff = [
                f for f in os.listdir(vasp_dir)
                if f.upper().startswith("CHGCAR") and "DIFF" in f.upper()
            ]
            vasp_only = [f for f in all_diff if f.lower().endswith(".vasp")]
            candidates = vasp_only or all_diff
            if not candidates:
                raise FileNotFoundError(f"No CHGCAR_diff file in {vasp_dir}")
            chgcar_path = os.path.join(vasp_dir, candidates[0])
            parsed = parse_CHGCAR(chgcar_path)
            # parse_CHGCAR may return (data, lv, coords, types, nums, ...)
            if isinstance(parsed, tuple) and len(parsed) >= 5:
                self.e, lv2, coord2, types2, nums2 = parsed[:5]
                self.lv, self.coord, self.atomtypes, self.atomnums = lv2, coord2, types2, nums2
            else:
                self.e = parsed

        elif filetype == "LOCPOT":
            loc = os.path.join(vasp_dir, "LOCPOT")
            if not os.path.isfile(loc):
                raise FileNotFoundError(f"No LOCPOT in {vasp_dir}")
            self.e, self.lv, self.coord, self.atomtypes, self.atomnums = parse_LOCPOT(loc)

        else:
            raise ValueError("filetype must be 'CHGCAR' or 'LOCPOT'")

        self.dim = self.e.shape  # (Nx, Ny, Nz)

        # 3) build element→color map (custom or default)
        if element_colors is None:
            palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
            self._element_colors = {}
            idx = 0
            for el in self.atomtypes:
                if el not in self._element_colors:
                    self._element_colors[el] = palette[idx % len(palette)]
                    idx += 1
        else:
            self._element_colors = element_colors

    def plane_slice(self, axis, fraction, use_interpolation=False, Na=None, Nb=None):
        """
        Extract a 2D slice at `fraction` along `axis`.
        Returns (X_cart, Y_cart, data2D) using the two free lattice vectors.
        """
        Nx, Ny, Nz = self.dim
        free = [0, 1, 2]
        free.remove(axis)
        a, b = free

        Na = Na or self.dim[a]
        Nb = Nb or self.dim[b]
        fa = (np.arange(Na) + 0.5) / Na
        fb = (np.arange(Nb) + 0.5) / Nb
        FA, FB = np.meshgrid(fa, fb, indexing='ij')

        lv_a = np.array(self.lv[a], float)
        lv_b = np.array(self.lv[b], float)
        lv_axis = np.array(self.lv[axis], float)
        origin = fraction * lv_axis

        # Cartesian sampling grid
        R = origin[None, None, :] \
            + FA[..., None] * lv_a[None, None, :] \
            + FB[..., None] * lv_b[None, None, :]

        # project onto a and b
        X_cart = (R @ lv_a) / np.linalg.norm(lv_a)
        Y_cart = (R @ lv_b) / np.linalg.norm(lv_b)

        # pick nearest grid‐plane or interpolate
        dim_map = {0: Nx, 1: Ny, 2: Nz}
        idxf = fraction * dim_map[axis]
        idxn = int(round(idxf)) % dim_map[axis]
        if not use_interpolation and abs(idxf - idxn) < 1e-8:
            if axis == 0:
                data2D = self.e[idxn, :, :].T
            elif axis == 1:
                data2D = self.e[:, idxn, :].T
            else:
                data2D = self.e[:, :, idxn].T
            if data2D.shape != X_cart.shape:
                use_interpolation = True

        if use_interpolation:
            gi = (np.arange(Nx) + 0.5) / Nx
            gj = (np.arange(Ny) + 0.5) / Ny
            gk = (np.arange(Nz) + 0.5) / Nz
            interp = RegularGridInterpolator((gi, gj, gk),
                                             self.e,
                                             bounds_error=False,
                                             fill_value=0.0)
            lat_mat = np.array(self.lv, float).T
            inv_lat = np.linalg.inv(lat_mat)
            R_flat = R.reshape(-1, 3).T
            fracs = (inv_lat @ R_flat).T % 1.0
            data_flat = interp(fracs)
            data2D = data_flat.reshape(Na, Nb)

        return X_cart, Y_cart, data2D

    def atoms_in_slab(self, axis, fraction):
        """
        Select atoms whose Cartesian distance from the slice‐plane
        (along `axis`) is < tol_ang, then project them.
        """
        # fractional positions of atoms
        lat_mat = np.array(self.lv, float).T
        inv_lat = np.linalg.inv(lat_mat)
        coords  = np.array(self.coord, float)
        if np.nanmax(np.abs(coords)) > 1.1:
            cfrac = (inv_lat @ coords.T).T % 1.0
        else:
            cfrac = coords % 1.0

        # fractional distance from slice‐plane
        frac_axis = cfrac[:, axis]
        d_frac = (frac_axis - fraction + 0.5) % 1.0 - 0.5

        # convert to Cartesian distances along that lattice vector
        lv_axis   = np.array(self.lv[axis], float)
        L_axis    = np.linalg.norm(lv_axis)
        d_cart    = d_frac * L_axis

        # now mask by Cartesian tolerance in Å
        mask = np.abs(d_cart) < self.tol_ang

        # build projected atom positions exactly as before
        lv      = np.array(self.lv, float)
        origin  = fraction * lv[axis]
        a, b    = [i for i in (0,1,2) if i != axis]
        xa      = lv[a] / np.linalg.norm(lv[a])
        xb      = lv[b] / np.linalg.norm(lv[b])

        offsets = np.cumsum([0] + self.atomnums)
        sel     = []
        for i in np.where(mask)[0]:
            ua, ub = cfrac[i, a], cfrac[i, b]
            R_i    = ua * lv[a] + ub * lv[b] + origin
            x       = float(R_i @ xa)
            y       = float(R_i @ xb)
            atom_idx = np.searchsorted(offsets, i, side='right') - 1
            elem     = self.atomtypes[atom_idx]
            sel.append({
                'index'    : i,
                'elem'     : elem,
                'x'        : x,
                'y'        : y,
                'dist_axis': float(d_cart[i])   # now in Å
            })
        return sel

    def plot_plane_slice(self, axis, fraction,
                         use_interpolation=False, clim=None, ax=None,
                         show_atoms=True, show_atoms_within=None,
                         atom_marker='o', atom_size=60, atom_edgecolor='k'):
        """
        Draw the CHGCAR_diff slice and overlay atoms.
        """
        X, Y, data = self.plane_slice(axis, fraction, use_interpolation)

        newfig = False
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
            newfig = True
        else:
            fig = ax.figure

        if clim is None:
            m = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
            vmin, vmax = -m, m
        else:
            v = abs(clim[1])
            vmin, vmax = -v, v

        linthresh = 0.01  # pick a threshold that makes small features visible
        norm = SymLogNorm(linthresh=linthresh,
                          linscale=0.5,
                          vmin=vmin,
                          vmax=vmax)

        pcm = ax.pcolormesh(X, Y, data,
                            cmap=self.cmap,
                            shading='nearest',
                            norm=norm)
        cbar = fig.colorbar(pcm)
        if 'CHG' in self.filetype:
            cbar.set_label(r'electron density / electrons $\AA^{-3}$')
        else:
            cbar.set_label('electrostatic potential / eV')
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.4f'))

        if show_atoms:
            atoms = self.atoms_in_slab(axis, fraction)
            if show_atoms_within is not None:
                atoms = [a for a in atoms
                         if abs(a['dist_axis']) <= show_atoms_within]
            handles, seen = [], set()
            for a in atoms:
                color = self._element_colors.get(a['elem'], 'grey')
                ax.scatter(a['x'], a['y'],
                           c=color, marker=atom_marker,
                           s=atom_size, edgecolor=atom_edgecolor,
                           zorder=5)
                if a['elem'] not in seen:
                    handles.append(Patch(color=color, label=a['elem']))
                    seen.add(a['elem'])
            if handles:
                ax.legend(handles=handles, title='Elements')

        ax.set_aspect('equal')
        ax.set_xlabel(r'X / $\AA$')
        ax.set_ylabel(r'Z / $\AA$')
        ax.set_title(f'Plane slice (axis={axis}, fraction={fraction:.3f})')

        if newfig:
            fig.show()
        return fig, ax, pcm


def run_cropped_view(vasp_dir,
                     axis,
                     fraction,
                     filetype="CHGCAR",
                     use_interpolation=True,
                     cmap='seismic',
                     x_half_width=None,
                     lower_cut=None,
                     upper_cut=None,
                     show_legend=True):
    """
    1) Extract the full 2D slice (X,Y,D).
    2) Plot it intact.
    3) Compute your desired X‐window: center ± x_half_width (or full span).
    4) Use ax.set_xlim and ax.set_ylim to zoom in on [left, right] × [lower_cut, upper_cut].
    5) Overlay atoms + legend.
    """

    # 1) build slice
    ps = PlaneSlice(
        vasp_dir,
        filetype=filetype,
        cmap=cmap,
        tol=1.275,
        element_colors={'Au':'orange','N':'mediumaquamarine','C':'saddlebrown','H':'yellow'}
    )
    X, Y, D = ps.plane_slice(axis, fraction, use_interpolation)

    # 2) full‐data plot
    fig, ax = plt.subplots(tight_layout=True)
    m = max(abs(np.nanmin(D)), abs(np.nanmax(D)))
    norm = SymLogNorm(linthresh=0.01, linscale=0.5, vmin=-m, vmax=m)

    pcm = ax.pcolormesh(X, Y, D,
                        cmap=cmap,
                        shading='nearest',
                        norm=norm)

    cbar = fig.colorbar(pcm)
    if 'CHG' in filetype:
        cbar.set_label(r'electron density / electrons $\AA^{-3}$')
    else:
        cbar.set_label('electrostatic potential / eV')
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.4f'))

    # 3) overlay atoms + legend
    atoms = ps.atoms_in_slab(axis, fraction)
    handles, seen = [], set()
    for a in atoms:
        color = ps._element_colors.get(a['elem'], 'grey')
        ax.scatter(a['x'], a['y'],
                   c=color, marker='o',
                   s=60, edgecolor='k',
                   zorder=5)
        if a['elem'] not in seen:
            handles.append(Patch(color=color, label=a['elem']))
            seen.add(a['elem'])

    if show_legend and handles:
        ax.legend(handles=handles, title='Elements')

    # 4) compute X‐limits (median-centred ± x_half_width like older code)
    x_center = float(np.nanmedian(X))

    if x_half_width is not None:
        left_cut  = x_center - x_half_width
        right_cut = x_center + x_half_width
    else:
        left_cut, right_cut = float(np.nanmin(X)), float(np.nanmax(X))

    # clamp to mesh extents
    left_cut  = max(left_cut,  np.nanmin(X))
    right_cut = min(right_cut, np.nanmax(X))

    # --- Y-limits: default to full mesh if you passed None ---
    y_vals = Y[:, 0]
    if lower_cut is None:
        lower_cut = float(np.nanmin(y_vals))
    if upper_cut is None:
        upper_cut = float(np.nanmax(y_vals))

    # --- now just zoom the axes to that rectangle ---
    ax.set_xlim(left_cut,  right_cut)
    ax.set_ylim(lower_cut, upper_cut)

    ax.set_aspect('equal')
    ax.set_xlabel(r'X / $\AA$')
    ax.set_ylabel(r'Z / $\AA$')
    ax.set_title(f'Cropped view (axis={axis}, frac={fraction:.3f})')

    return fig, ax, pcm


def plot_two_slices_side_by_side(vasp_dir1,
                                 vasp_dir2,
                                 axis=1,
                                 fraction=0.5,
                                 filetype="CHGCAR",
                                 use_interpolation=True,
                                 cmap='seismic',
                                 x_half_width1=None,
                                 x_half_width2=None,
                                 lower_cut1=None,
                                 upper_cut1=None,
                                 lower_cut2=None,
                                 upper_cut2=None,
                                 tol=1.275,
                                 element_colors=None,
                                 show_legend=True,
                                 linthresh=0.01,
                                 linscale=0.5):
    """
    Plot two plane slices side-by-side using the same symmetric color scale.

    - x_half_width1 and x_half_width2 are per-system half-widths applied as
      median(X) ± half-width for each subplot. If None, full X span is used.
    - lower_cut1/upper_cut1 and lower_cut2/upper_cut2 control Y-limits per subplot.
    - axis and fraction may be scalars (applied to both) or length-2 tuples to specify per-system.
    """

    # allow scalar or pair for axis/fraction
    if isinstance(axis, (list, tuple)):
        axis1, axis2 = axis
    else:
        axis1 = axis2 = axis
    if isinstance(fraction, (list, tuple)):
        frac1, frac2 = fraction
    else:
        frac1 = frac2 = fraction

    # build PlaneSlice objects
    ps1 = PlaneSlice(vasp_dir1, filetype=filetype, cmap=cmap, tol=tol, element_colors=element_colors)
    ps2 = PlaneSlice(vasp_dir2, filetype=filetype, cmap=cmap, tol=tol, element_colors=element_colors)

    # get slices
    X1, Y1, D1 = ps1.plane_slice(axis1, frac1, use_interpolation)
    X2, Y2, D2 = ps2.plane_slice(axis2, frac2, use_interpolation)

    # compute global symmetric v for color scale
    m1_min, m1_max = np.nanmin(D1), np.nanmax(D1)
    m2_min, m2_max = np.nanmin(D2), np.nanmax(D2)
    v = max(abs(m1_min), abs(m1_max), abs(m2_min), abs(m2_max))
    vmin, vmax = -v, v

    norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)

    # prepare figure with two axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    pcs = []
    for ax, X, Y, D, ps, axis_idx, frac, wh, lcut, ucut in [
            (axes[0], X1, Y1, D1, ps1, axis1, frac1, x_half_width1, lower_cut1, upper_cut1),
            (axes[1], X2, Y2, D2, ps2, axis2, frac2, x_half_width2, lower_cut2, upper_cut2),
    ]:
        pcm = ax.pcolormesh(X, Y, D, cmap=cmap, shading='nearest', norm=norm)
        pcs.append(pcm)

        # overlay atoms
        atoms = ps.atoms_in_slab(axis_idx, frac)
        handles, seen = [], set()
        for a in atoms:
            color = ps._element_colors.get(a['elem'], 'grey')
            ax.scatter(a['x'], a['y'], c=color, marker='o', s=60, edgecolor='k', zorder=5)
            if a['elem'] not in seen:
                handles.append(Patch(color=color, label=a['elem']))
                seen.add(a['elem'])
        if show_legend and handles:
            ax.legend(handles=handles, title='Elements')

        # compute X limits using median-centred ± wh (per-subplot)
        xmin, xmax = float(np.nanmin(X)), float(np.nanmax(X))
        x_center = float(np.nanmedian(X))
        if wh is not None:
            left = x_center - wh
            right = x_center + wh
        else:
            left, right = xmin, xmax

        # Y limits: accept per-plot provided cuts or default to full span
        ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))
        lower = ymin if lcut is None else max(lcut, ymin)
        upper = ymax if ucut is None else min(ucut, ymax)

        # clamp to mesh extents
        left = max(left, xmin)
        right = min(right, xmax)
        lower = max(lower, ymin)
        upper = min(upper, ymax)

        ax.set_xlim(left, right)
        ax.set_ylim(lower, upper)

        ax.set_aspect('equal')
        ax.set_xlabel(r'X / $\AA$')
        ax.set_ylabel(r'Z / $\AA$')
        ax.set_title(f'axis={axis_idx}, frac={frac:.3f}')

    # create a dedicated left-side colorbar axes (values are [left, bottom, width, height] in figure coords)
    cbar_ax = fig.add_axes([0.04, 0.12, 0.02, 0.76])   # tweak these numbers to taste
    cbar = fig.colorbar(pcs[0], cax=cbar_ax)
    if 'CHG' in filetype:
        cbar.set_label(r'electron density / electrons $\AA^{-3}$')
    else:
        cbar.set_label('electrostatic potential / eV')
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.4f'))

    return fig, axes, pcs


if __name__ == "__main__":
    # Replace these with your two directories
    vasp_dir_a = r"dir/system_1"
    vasp_dir_b = r'dir/system_2'

    # Slicing parameters (same axis and fraction for both here; can pass tuples)
    axis = 1
    fraction = 0.50

    # Per-system horizontal half-widths (median-centered ± value)
    x_half_width1 = 5.0   # system A: ±5.0 Å around its slice center
    x_half_width2 = 5.0   # system B: ±6.5 Å around its slice center

    # Per-subplot vertical cuts (lower, upper). Use None to take full Y span.
    lower_cut1, upper_cut1 = 1.74, 15.26
    lower_cut2, upper_cut2 = 1.74, 15.26

    # Shared element colors so legends match
    elem_colors = {'Au':'orange','N':'mediumaquamarine','C':'saddlebrown','H':'yellow'}

    fig, axes, pcms = plot_two_slices_side_by_side(
        vasp_dir_a,
        vasp_dir_b,
        axis=axis,
        fraction=fraction,
        filetype="CHGCAR",
        use_interpolation=True,
        cmap='seismic',
        x_half_width1=x_half_width1,
        x_half_width2=x_half_width2,
        lower_cut1=lower_cut1,
        upper_cut1=upper_cut1,
        lower_cut2=lower_cut2,
        upper_cut2=upper_cut2,
        tol=1.275,
        element_colors=elem_colors,
        show_legend=False
    )

    plt.show()
 