import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from lib import parse_CHGCAR, parse_LOCPOT, parse_poscar

from scipy.spatial import cKDTree


def debug_atoms_vs_mesh(ps, axis, fraction, Na=None, Nb=None, tol_frac=0.02, n_print=80):
    """
    Diagnostic: compare per-atom projected (x,y) to the nearest mesh cell center
    produced by ps.plane_slice(axis, fraction, Na=Na, Nb=Nb).

    - ps: PlaneSlice instance
    - axis, fraction: same as you pass to plane_slice/plot_plane_slice
    - Na, Nb: optional in-plane sampling (defaults to ps.dim for the two free axes)
    - tol_frac: fractional tolerance along the sliced axis used to mark atoms 'in-plane'
    - n_print: number of atoms to always print (prints additionally any that pass tol_frac)
    """
    # Build mesh used by plane_slice (no interpolation)
    Xmesh, Ymesh, _ = ps.plane_slice(axis, fraction, use_interpolation=False, Na=Na, Nb=Nb)
    Nx, Ny, Nz = ps.dim
    free = [0, 1, 2]; free.remove(axis); a, b = free
    Na = Na or ps.dim[a]; Nb = Nb or ps.dim[b]

    # voxel centers in fractional for mesh
    fa = (np.arange(Na) + 0.5) / Na
    fb = (np.arange(Nb) + 0.5) / Nb
    FA, FB = np.meshgrid(fa, fb, indexing='ij')  # shape (Na,Nb)

    # lattice vectors and matrices
    lv = np.array(ps.lv, dtype=float)     # rows = lattice vectors (Å)
    inv_lv = np.linalg.inv(lv)
    lv_a = lv[a]; lv_b = lv[b]; lv_axis = lv[axis]
    origin_cart = float(fraction) * lv_axis

    # Build mesh Cartesian points R for nearest lookup
    R = origin_cart[None, None, :] + FA[..., None] * lv_a[None, None, :] + FB[..., None] * lv_b[None, None, :]
    Xmesh_flat = (R @ lv_a) / np.linalg.norm(lv_a)
    Ymesh_flat = (R @ lv_b) / np.linalg.norm(lv_b)
    mesh_points = np.column_stack([Xmesh_flat.ravel(), Ymesh_flat.ravel()])
    tree = cKDTree(mesh_points)

    # get atom fractional coords robustly
    if np.nanmax(np.abs(ps.coord)) <= 1.1:
        coord_frac = np.array(ps.coord, dtype=float)
    else:
        atom_cart = np.array(ps.coord, dtype=float)
        coord_frac = (inv_lv @ atom_cart.T).T

    outputs = []
    for i, afr in enumerate(coord_frac):
        # nearest-image fractional difference along sliced axis
        d_frac_axis = (afr[axis] - fraction + 0.5) % 1.0 - 0.5
        near_frac_pass = abs(d_frac_axis) <= tol_frac

        # atom Cartesian and wrap relative to origin
        atom_cart = afr @ lv
        d_cart = atom_cart - origin_cart
        d_frac = (inv_lv @ d_cart.T)
        d_frac_wrapped = (d_frac + 0.5) % 1.0 - 0.5
        d_cart_wr = (d_frac_wrapped @ lv)

        # in-plane projection using lv_a, lv_b units (same convention used in the cleaned plane_slice)
        lv_a_norm = np.linalg.norm(lv_a); lv_b_norm = np.linalg.norm(lv_b)
        axis_len = np.linalg.norm(lv_axis)
        axis_unit = lv_axis / axis_len if axis_len > 0 else np.array([0.,0.,1.])
        R_rel = d_cart_wr - np.dot(d_cart_wr, axis_unit) * axis_unit
        x_proj = float(np.dot(R_rel, lv_a) / lv_a_norm)
        y_proj = float(np.dot(R_rel, lv_b) / lv_b_norm)
        dist_axis = float(np.dot(d_cart_wr, axis_unit))

        # nearest mesh point and distances
        dist_mesh, idx_mesh = tree.query([x_proj, y_proj])
        j_idx, k_idx = divmod(idx_mesh, Nb)
        fa_near = FA[j_idx, k_idx]; fb_near = FB[j_idx, k_idx]
        mesh_cart = R[j_idx, k_idx]
        atom_cart_proj_cart = origin_cart + fa_near * lv_a + fb_near * lv_b
        cart_diff = np.linalg.norm(atom_cart_proj_cart - (origin_cart + R_rel))

        # element lookup (1-based element mapping)
        running = 0; elem = None
        for t, n in zip(ps.atomtypes, ps.atomnums):
            if i < running + n:
                elem = t; break
            running += n

        entry = {
            'index': i+1,
            'elem': elem,
            'afr_axis': float(afr[axis]),
            'd_frac_axis': float(d_frac_axis),
            'near_frac_pass': bool(near_frac_pass),
            'x_proj': x_proj, 'y_proj': y_proj,
            'mesh_idx': (int(j_idx), int(k_idx)),
            'fa_near': float(fa_near), 'fb_near': float(fb_near),
            'scalar_dist_to_mesh': float(dist_mesh),
            'cart_dist_to_mesh': float(cart_diff),
            'dist_axis': float(dist_axis)
        }
        outputs.append(entry)

    # Print header and a controlled set of lines
    print("DEBUG atoms vs mesh (first rows plus any near_frac_pass).")
    printed = 0
    for info in outputs:
        if printed < n_print or info['near_frac_pass']:
            printed += 1
            print(
                f"atom {info['index']:3d} elem {info['elem']:<3} afr_axis {info['afr_axis']:.6f} "
                f"d_frac {info['d_frac_axis']:+.6f} pass {info['near_frac_pass']} "
                f"dist_axis {info['dist_axis']:+.4f}Å mesh_idx {info['mesh_idx']} "
                f"scalar_dx {info['scalar_dist_to_mesh']:.4f} cart_dx {info['cart_dist_to_mesh']:.4f} "
                f"Xproj {info['x_proj']:.4f} Yproj {info['y_proj']:.4f}"
            )
    print(f"... printed {printed} records. tol_frac={tol_frac}, Na={Na}, Nb={Nb}")
    return outputs

class PlaneSlice:
    """
    Get and plot a 2D plane slice parallel to two lattice vectors,
    positioned at a fractional position along the third lattice vector.

    Clean, consistent implementation:
      - plane_slice builds the 2D mesh from the two free lattice vectors and a Cartesian origin.
      - atoms_in_slab uses a simple fractional test along the sliced axis (nearest-image) and
        projects selected atoms into the same X/Y coordinates used by plane_slice so overlays align.
      - plot_plane_slice can filter atoms by physical distance from the slice via show_atoms_within (Å).
    """

    def __init__(self, ifile, filetype='CHGCAR', read_data_from_file=False, cmap='seismic', **kwargs):
        self.cmap = cmap
        self.filetype = filetype

        def _find_chgcar_diff_in_dir(d):
            patterns = [
                os.path.join(d, "*diff*CHGCAR*"),
                os.path.join(d, "CHGCAR*diff*"),
                os.path.join(d, "*CHGCAR*diff*"),
                os.path.join(d, "*_diff*"),
                os.path.join(d, "*diff.vasp"),
                os.path.join(d, "*_diff.vasp"),
                os.path.join(d, "CHGCAR_diff"),
                os.path.join(d, "CHGCAR_diff*"),
                os.path.join(d, "*CHGCAR_diff*"),
            ]
            matches = []
            for p in patterns:
                matches.extend(glob.glob(p))
            try:
                matches.extend([os.path.join(d, f) for f in os.listdir(d) if "diff" in f.lower()])
            except PermissionError:
                pass
            matches = [m for m in dict.fromkeys(matches) if os.path.isfile(m)]
            if not matches:
                return None
            for m in matches:
                if "chgc" in os.path.basename(m).lower():
                    return m
            return matches[0]

        # Loading logic
        if read_data_from_file:
            self.e = np.load(read_data_from_file)
            poscar_dir = ifile
            try:
                self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(os.path.join(poscar_dir, "CONTCAR"))[:4]
            except Exception:
                self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar_dir)[:4]
        else:
            if isinstance(ifile, str) and os.path.isfile(ifile):
                path_to_use = ifile
            else:
                path_to_use = None

            if "CHG" in self.filetype:
                if path_to_use is None and isinstance(ifile, str) and os.path.isdir(ifile):
                    candidate = _find_chgcar_diff_in_dir(ifile)
                    if candidate:
                        path_to_use = candidate
                if path_to_use is None and isinstance(ifile, str):
                    parent = os.path.dirname(ifile)
                    if parent and os.path.isdir(parent):
                        candidate = _find_chgcar_diff_in_dir(parent)
                        if candidate:
                            path_to_use = candidate

                try:
                    if path_to_use:
                        parsed = parse_CHGCAR(path_to_use)
                    else:
                        parsed = parse_CHGCAR(ifile)
                except Exception:
                    if isinstance(ifile, str) and os.path.isdir(ifile):
                        tried = False
                        for name in ["CHGCAR_diff", "CHGCAR_diff.vasp", "CHGCAR", "CHGCAR.vasp"]:
                            p = os.path.join(ifile, name)
                            if os.path.isfile(p):
                                parsed = parse_CHGCAR(p)
                                tried = True
                                break
                        if not tried:
                            raise
                    else:
                        raise

                if isinstance(parsed, tuple) and len(parsed) >= 1:
                    self.e = parsed[0]
                else:
                    self.e = parsed

                if isinstance(parsed, tuple) and len(parsed) >= 5:
                    self.lv, self.coord, self.atomtypes, self.atomnums = parsed[1], parsed[2], parsed[3], parsed[4]
                else:
                    poscar_dir = None
                    if path_to_use:
                        poscar_dir = os.path.dirname(path_to_use) or '.'
                    elif isinstance(ifile, str) and os.path.isdir(ifile):
                        poscar_dir = ifile
                    else:
                        poscar_dir = os.path.dirname(ifile) or '.'

                    try:
                        self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(os.path.join(poscar_dir, 'CONTCAR'))[:4]
                    except Exception:
                        try:
                            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar_dir)[:4]
                        except Exception:
                            raise RuntimeError(f"Failed to obtain lattice/atom data from parse_CHGCAR and CONTCAR in {poscar_dir}")

            elif self.filetype == "LOCPOT":
                if path_to_use:
                    self.e, self.lv, self.coord, self.atomtypes, self.atomnums = parse_LOCPOT(path_to_use)
                else:
                    self.e, self.lv, self.coord, self.atomtypes, self.atomnums = parse_LOCPOT(ifile)

            elif self.filetype == "none":
                npts = 200
                self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(ifile)[:4]
                self.e = np.zeros((npts, npts, npts))

            else:
                raise ValueError("Unrecognized filetype or read_data_from_file usage.")

        self.dim = self.e.shape  # (Nx, Ny, Nz)

        # Build element color map for overlays
        palette = plt.rcParams.get('axes.prop_cycle').by_key().get('color', plt.cm.tab10.colors)
        self._element_colors = {}
        idx = 0
        for el in getattr(self, 'atomtypes', []):
            if el not in self._element_colors:
                self._element_colors[el] = palette[idx % len(palette)]
                idx += 1

    def plane_slice(self, axis, fraction, use_interpolation=False, Na=None, Nb=None):
        """
        Build 2D mesh. Uses lattice_matrix (columns = lattice vectors) for all
        fractional<->Cartesian conversions so it matches atoms_in_slab below.
        """
        assert 0.0 <= fraction <= 1.0, "fraction must be in [0,1]"
        Nx, Ny, Nz = self.dim
    
        free = [0, 1, 2]
        free.remove(axis)
        a, b = free
    
        Na = Na or self.dim[a]
        Nb = Nb or self.dim[b]
    
        fa = (np.arange(Na) + 0.5) / Na
        fb = (np.arange(Nb) + 0.5) / Nb
        FA, FB = np.meshgrid(fa, fb, indexing='ij')  # shape (Na, Nb)
    
        # lattice vectors (rows in self.lv) but use lattice_matrix for conversions
        lv_a = np.array(self.lv[a], dtype=float)
        lv_b = np.array(self.lv[b], dtype=float)
        lv_axis = np.array(self.lv[axis], dtype=float)
    
        # lattice_matrix: columns are lattice vectors
        lattice_matrix = np.array(self.lv, dtype=float).T
        inv_lat = np.linalg.inv(lattice_matrix)
    
        # Cartesian origin for the slice (same convention used below)
        origin = float(fraction) * lv_axis
    
        # Build R directly in Cartesian using lv_a and lv_b
        R = origin[None, None, :] + FA[..., None] * lv_a[None, None, :] + FB[..., None] * lv_b[None, None, :]
    
        # Use lv_a and lv_b consistently for scalar projection
        if 2 in free:
            other = a if b == 2 else b
            x_vec = self.lv[other]
            y_vec = self.lv[2]
        else:
            x_vec = lv_a
            y_vec = lv_b
        x_cart = (R @ x_vec) / np.linalg.norm(x_vec)
        y_cart = (R @ y_vec) / np.linalg.norm(y_vec)
    
        dim_map = {0: Nx, 1: Ny, 2: Nz}
        idx_float = fraction * dim_map[axis]
        idx_nearest = int(round(idx_float)) % dim_map[axis]
    
        if (not use_interpolation) and abs(idx_float - idx_nearest) < 1e-8:
            if axis == 0:
                data2D = self.e[idx_nearest, :, :].T
            elif axis == 1:
                data2D = self.e[:, idx_nearest, :].T
            else:
                data2D = self.e[:, :, idx_nearest].T
            if data2D.shape != x_cart.shape:
                use_interpolation = True
    
        if use_interpolation:
            gi = (np.arange(Nx) + 0.5) / Nx
            gj = (np.arange(Ny) + 0.5) / Ny
            gk = (np.arange(Nz) + 0.5) / Nz
            interp = RegularGridInterpolator((gi, gj, gk), self.e, bounds_error=False, fill_value=0.0)
    
            # build fractional coords for each R point using inv_lat
            R_flat = R.reshape(-1, 3).T
            frac_coords = (inv_lat @ R_flat).T
            frac_coords %= 1.0
            data_flat = interp(frac_coords)
            data2D = data_flat.reshape(Na, Nb)
    
        return x_cart, y_cart, data2D
    
    def atoms_in_slab(self, axis, fraction, tol_frac=0.02, debug=False):
        """
        Select atoms by simple fractional test along `axis` and project them using
        the same projection rule plane_slice uses.
    
        - axis: sliced lattice vector index (0,1,2)
        - fraction: fractional plane position (0..1)
        - tol_frac: fractional tolerance along the sliced axis
        - debug: if True prints diagnostics for selected atoms
        """
        # lattice_matrix: columns = lattice vectors (Å)
        lattice_matrix = np.array(self.lv, dtype=float).T   # shape (3,3)
        inv_lat = np.linalg.inv(lattice_matrix)
    
        # free indices and their lattice vectors (same ordering as plane_slice)
        free = [0, 1, 2]; free.remove(axis)
        a, b = free
        lv_a = np.array(self.lv[a], dtype=float)
        lv_b = np.array(self.lv[b], dtype=float)
        lv_axis = np.array(self.lv[axis], dtype=float)
    
        # Choose projection vectors exactly the same way plane_slice does
        if 2 in free:
            other = a if b == 2 else b
            x_vec = np.array(self.lv[other], dtype=float)
            y_vec = np.array(self.lv[2], dtype=float)
        else:
            x_vec = lv_a.copy()
            y_vec = lv_b.copy()
        nx = np.linalg.norm(x_vec); ny = np.linalg.norm(y_vec)
        if nx == 0 or ny == 0:
            if debug: print("atoms_in_slab: zero norm for projection vectors"); return []
    
        # Get fractional coordinates reliably
        if np.nanmax(np.abs(self.coord)) <= 1.1:
            coord_frac = np.array(self.coord, dtype=float)
        else:
            atom_cart = np.array(self.coord, dtype=float)
            coord_frac = (inv_lat @ atom_cart.T).T
    
        origin_cart = float(fraction) * lv_axis
        axis_len = np.linalg.norm(lv_axis)
        axis_unit = lv_axis / axis_len if axis_len > 0 else np.array([0.0, 0.0, 1.0])
    
        results = []
        selected_info = []
        for i, afr in enumerate(coord_frac):
            # fractional difference along axis (nearest-image)
            d = afr[axis] - float(fraction)
            d = (d + 0.5) % 1.0 - 0.5
    
            if abs(d) <= float(tol_frac):
                # Build Cartesian atom position consistently (lattice_matrix @ frac)
                atom_cart_i = (lattice_matrix @ afr.T)
                atom_cart_i = np.asarray(atom_cart_i).reshape(3,)
                d_cart = atom_cart_i - origin_cart
    
                # fractional displacement relative to origin (use inv_lat)
                d_frac = (inv_lat @ d_cart)
                d_frac = (d_frac + 0.5) % 1.0 - 0.5
                d_cart_wrapped = (lattice_matrix @ d_frac)
    
                # remove axis component to get in-plane Cartesian vector
                R_rel = d_cart_wrapped - np.dot(d_cart_wrapped, axis_unit) * axis_unit
    
                # scalar projection coordinates using chosen x_vec / y_vec
                x_coord = float(np.dot(R_rel, x_vec) / nx)
                y_coord = float(np.dot(R_rel, y_vec) / ny)
                dist_axis = float(np.dot(d_cart_wrapped, axis_unit))
    
                # element lookup
                running = 0; elem = None
                for t, n in zip(self.atomtypes, self.atomnums):
                    if i < running + n:
                        elem = t; break
                    running += n
                if elem is None:
                    elem = self.atomtypes[-1] if len(self.atomtypes) else f'X{i}'
    
                results.append({'index': i, 'elem': elem, 'x': x_coord, 'y': y_coord, 'dist_axis': dist_axis})
                selected_info.append((i+1, elem, float(afr[axis]), d, dist_axis))
    
        if debug:
            print(f"atoms_in_slab: axis={axis} fraction={fraction} tol_frac={tol_frac} selected={len(results)}")
            for idx, el, frac_val, delta_frac, distA in selected_info[:200]:
                print(f" atom {idx}: elem={el} frac={frac_val:.6f} delta_frac={delta_frac:+.6f} dist_axis={distA:+.4f} Å")
    
        return results
    def plot_plane_slice(self, axis, fraction, use_interpolation=False, Na=None, Nb=None, clim=None, ax=None,
                         show_atoms=True, show_atoms_within=None, atom_marker='o', atom_size=60, atom_edgecolor='k',
                         atom_tol_frac=0.02, debug_atoms=False):
        """
        Plot a plane slice with a diverging colormap centered at zero.
        Optionally overlay atoms. To restrict atoms to a thin slab, set show_atoms_within (Å).
        atom_tol_frac: fractional tolerance used for selecting atoms along the sliced axis (default 0.02).
        debug_atoms: if True prints diagnostic lines from atoms_in_slab.
        """
        X, Y, data = self.plane_slice(axis, fraction, use_interpolation=use_interpolation, Na=Na, Nb=Nb)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True)
            created_fig = True
        else:
            fig = ax.figure

        if clim is None:
            m = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
            vmin, vmax = -m, m
        else:
            vmin, vmax = clim[0], clim[1]
            if abs(vmin) != abs(vmax):
                m = max(abs(vmin), abs(vmax))
                vmin, vmax = -m, m

        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(X, Y, data, cmap=self.cmap, shading='nearest', norm=norm)
        cbar = fig.colorbar(pcm)

        if 'CHG' in self.filetype:
            cbar.set_label(r'electron density / electrons $\AA^{-3}$')
        if self.filetype == 'LOCPOT':
            cbar.set_label('electrostatic potential / eV')
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.4f'))

        # atom overlay via simple fractional selection (nearest-image)
        if show_atoms:
            atoms = self.atoms_in_slab(axis, fraction, tol_frac=atom_tol_frac, debug=debug_atoms)
            if show_atoms_within is not None:
                atoms = [a for a in atoms if abs(a['dist_axis']) <= float(show_atoms_within)]
            legend_handles = []
            seen = set()
            for a in atoms:
                color = self._element_colors.get(a['elem'], None)
                if color is None:
                    palette = plt.rcParams.get('axes.prop_cycle').by_key().get('color', plt.cm.tab10.colors)
                    self._element_colors = {el: palette[i % len(palette)] for i, el in enumerate(self.atomtypes)}
                    color = self._element_colors.get(a['elem'], 'grey')
                ax.scatter(a['x'], a['y'], c=color, marker=atom_marker, s=atom_size,
                           edgecolor=atom_edgecolor, zorder=5)
                if a['elem'] not in seen:
                    legend_handles.append(Patch(color=color, label=a['elem']))
                    seen.add(a['elem'])
            if legend_handles:
                ax.legend(handles=legend_handles, title='Elements')

        ax.set_aspect('equal')
        ax.set_xlabel(r'X / $\AA$')
        ax.set_ylabel(r'Z / $\AA$')
        title = f'Change in Electron Density (axis={axis}, fraction={fraction:.3f})'
        ax.set_title(title)

        if created_fig:
            fig.show()
        return fig, ax, pcm

    

# -----------------------
# run helper + if-main
# -----------------------
def run(vasp_dir, axis, fraction, filetype="CHGCAR", npy_path=None, use_interpolation=True, cmap='seismic'):
    """
    Simple entrypoint: construct PlaneSlice and show the requested plane slice.
    """
    ps = PlaneSlice(ifile=vasp_dir, filetype=filetype, read_data_from_file=npy_path, cmap=cmap)
    fig, ax, pcm = ps.plot_plane_slice(axis=axis, fraction=fraction, use_interpolation=use_interpolation)
    return fig, ax, pcm


if __name__ == "__main__":
    # Edit these three variables in Spyder before pressing Run:
    vasp_dir = r"dir/run"   # directory containing CHGCAR_diff.* or path to the diff file
    axis     = 1      # lattice vector index to FIX: 0=a, 1=b, 2=c
    fraction = 0.50   # fractional position along that lattice vector (0..1)

    # Single-line action to run and show
    run(vasp_dir, axis, fraction, filetype="CHGCAR", npy_path=None, use_interpolation=True); plt.show()
    
    ps = PlaneSlice(ifile=vasp_dir, filetype="CHGCAR", read_data_from_file=None, cmap='seismic')
    debug_atoms_vs_mesh(ps, axis, fraction, Na=None, Nb=None, tol_frac=0.02, n_print=120)
    fig, ax, pcm = ps.plot_plane_slice(axis=axis, fraction=fraction, use_interpolation=True)