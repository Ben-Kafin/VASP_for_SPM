# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 23:13:22 2025

@author: Benjamin Kafin
"""

# =============================================================================
# point_cloud_slicer.py
#
# A self-contained script for quantitatively comparing charge density slices
# from two VASP calculations, with options for:
#  - 'horizontal': A top-down "draped sheet" view.
#  - 'vertical': A full-cell side-view along a best-fit line.
#
# Instructions:
#  1. Ensure this file is in the same directory as your 'lib.py'.
#  2. Modify the variables in the "USER CONFIGURATION" section at the bottom.
#  3. Run the script from your terminal: python point_cloud_slicer.py
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import SymLogNorm
from copy import deepcopy
import os
from lib import parse_CHGCAR, parse_poscar

# =============================================================================
# CLASS 1: Data Loader and Slice Generator
# =============================================================================
class slice_path():
    def __init__(self, ifile, path_atoms, **args):
        self.ifile = ifile
        self.path_atoms = path_atoms
        
        try:
            contcar_path = os.path.join(ifile, "CONTCAR")
            if not os.path.exists(contcar_path):
                contcar_path = os.path.join(ifile, "POSCAR")
            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(contcar_path)[:4]
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find CONTCAR or POSCAR in directory: {ifile}")

        all_diff = [f for f in os.listdir(ifile) if "DIFF" in f.upper() and f.upper().startswith("CHGCAR")]
        if not all_diff:
            raise FileNotFoundError(f"No CHGCAR*diff* file found in {ifile}")
        chgcar_path = os.path.join(ifile, all_diff[0])
        self.e, _, _, _, _ = parse_CHGCAR(chgcar_path)
        self.dim = self.e.shape

    def _select_atoms_near_horizontal_path(self, atom_tol):
        """Selects atoms for the 'horizontal' draped-sheet view."""
        initial_path_coords = np.array([deepcopy(self.coord[i[0] - 1]) for i in self.path_atoms])
        if len(initial_path_coords) < 2:
            path_segments_2d, path_segments_3d = [], []
        else:
            path_segments_2d = [ (initial_path_coords[i, :2], initial_path_coords[i+1, :2]) for i in range(len(initial_path_coords)-1) ]
            path_segments_3d = [ (initial_path_coords[i], initial_path_coords[i+1]) for i in range(len(initial_path_coords)-1) ]
        
        selected_atoms = []
        offsets = np.cumsum([0] + self.atomnums)
        for i, atom_pos_3d in enumerate(self.coord):
            p_2d = atom_pos_3d[:2]
            min_dist_sq_atom, sheet_z_at_atom = np.inf, 0.0
            
            if not path_segments_2d:
                sheet_z_at_atom = initial_path_coords[0, 2] if len(initial_path_coords) > 0 else 0
            else:
                for (a2, b2), (a3, b3) in zip(path_segments_2d, path_segments_3d):
                    v = b2 - a2; v_dot_v = np.dot(v, v)
                    if v_dot_v < 1e-9: continue
                    t = max(0, min(1, np.dot(p_2d - a2, v) / v_dot_v))
                    dist_sq = np.sum((p_2d - (a2 + t * v))**2)
                    if dist_sq < min_dist_sq_atom:
                        min_dist_sq_atom = dist_sq
                        sheet_z_at_atom = a3[2] * (1 - t) + b3[2] * t
            
            if abs(atom_pos_3d[2] - sheet_z_at_atom) <= atom_tol:
                atom_type_idx = np.searchsorted(offsets, i, side='right') - 1
                selected_atoms.append({'elem': self.atomtypes[atom_type_idx], 'coords': atom_pos_3d})
        
        return selected_atoms

    def _generate_refined_horizontal_data(self, padding, n_points, atom_tol):
        selected_atoms = self._select_atoms_near_horizontal_path(atom_tol)
        if not selected_atoms: return None
        
        refined_atom_coords = np.array([a['coords'] for a in selected_atoms])
        min_xy = np.min(refined_atom_coords[:, :2], axis=0) - padding
        max_xy = np.max(refined_atom_coords[:, :2], axis=0) + padding
        
        x_range = np.linspace(min_xy[0], max_xy[0], n_points)
        y_range = np.linspace(min_xy[1], max_xy[1], n_points)
        
        z_height_map = np.zeros((n_points, n_points))
        refined_atoms_xy = refined_atom_coords[:, :2]
        refined_atoms_z = refined_atom_coords[:, 2]

        for j, y_val in enumerate(y_range):
            for i, x_val in enumerate(x_range):
                p = np.array([x_val, y_val])
                distances = np.sqrt(np.sum((refined_atoms_xy - p)**2, axis=1))
                weights = 1.0 / (distances**2 + 1e-6)
                z_height_map[j, i] = np.sum(refined_atoms_z * weights) / np.sum(weights)

        plot_data = np.zeros((n_points, n_points))
        dim, inv_lv = np.shape(self.e), np.linalg.inv(self.lv)
        for j, y_val in enumerate(y_range):
            for i, x_val in enumerate(x_range):
                point_3d = np.array([x_val, y_val, z_height_map[j, i]])
                point_frac = np.dot(point_3d, inv_lv)
                ix, iy, iz = [int(round((point_frac[k] % 1.0) * dim[k])) % dim[k] for k in range(3)]
                plot_data[j, i] = self.e[ix, iy, iz]
        
        return {"plot_data": plot_data, "extent": [min_xy[0], max_xy[0], min_xy[1], max_xy[1]], "selected_atoms": selected_atoms}

    def _generate_linear_vertical_data(self, n_points_path, n_points_z, atom_tol):
        # 1. Determine the best-fit line in the XY plane from initial path atoms
        path_atom_coords = np.array([deepcopy(self.coord[i[0] - 1]) for i in self.path_atoms])
        path_xy = path_atom_coords[:, :2]
        
        if len(path_xy) < 2:
            raise ValueError("Vertical slice requires at least two atoms to define a best-fit line.")
            
        centroid = np.mean(path_xy, axis=0)
        _, _, vh = np.linalg.svd(path_xy - centroid)
        direction_vector = vh[0]
        
        # 2. Define the full-cell grid for the plot
        cell_corners_xy = np.array([[0,0], self.lv[0,:2], self.lv[1,:2], self.lv[0,:2]+self.lv[1,:2]])
        projected_distances = np.dot(cell_corners_xy - centroid, direction_vector)
        dist_min, dist_max = np.min(projected_distances), np.max(projected_distances)
        
        path_axis = np.linspace(dist_min, dist_max, n_points_path)
        z_axis = np.linspace(0, np.linalg.norm(self.lv[2]), n_points_z)
        
        # 3. Sample the 3D data onto the unrolled vertical plane
        plot_data = np.zeros((n_points_z, n_points_path))
        inv_lv = np.linalg.inv(self.lv)
        
        for i, dist in enumerate(path_axis):
            for j, z_val in enumerate(z_axis):
                point_xy = centroid + dist * direction_vector
                sample_point = np.array([point_xy[0], point_xy[1], z_val])
                point_frac = np.dot(sample_point, inv_lv)
                ix, iy, iz = [int(round((pf % 1.0) * dim_size)) % dim_size for pf, dim_size in zip(point_frac, self.dim)]
                plot_data[j, i] = self.e[ix, iy, iz]

        # 4. Select ALL atoms near the slice and calculate their plot coordinates
        selected_atoms = []
        offsets = np.cumsum([0] + self.atomnums)

        for i, atom_coord in enumerate(self.coord):
            vec_from_centroid = atom_coord[:2] - centroid
            dist_to_line = np.linalg.norm(vec_from_centroid - np.dot(vec_from_centroid, direction_vector) * direction_vector)
            
            if dist_to_line <= atom_tol:
                plot_x = np.dot(vec_from_centroid, direction_vector)
                atom_type_idx = np.searchsorted(offsets, i, side='right') - 1
                selected_atoms.append({
                    'elem': self.atomtypes[atom_type_idx],
                    'plot_x': plot_x,
                    'plot_y': atom_coord[2]
                })
        
        return {"plot_data": plot_data, "extent": [dist_min, dist_max, 0, z_axis[-1]], "selected_atoms": selected_atoms}


# =============================================================================
# CLASS 2: The High-Level Controller Class
# =============================================================================
class PointCloudSlicer():
    def __init__(self, vasp_dir1, path1, vasp_dir2, path2, **kwargs):
        print("--- Initializing System 1 ---")
        self.system1 = slice_path(ifile=vasp_dir1, path_atoms=path1)
        print("--- Initializing System 2 ---")
        self.system2 = slice_path(ifile=vasp_dir2, path_atoms=path2)
        
        self.params = {
            'cmap': kwargs.get('cmap', 'seismic'),
            'tol': kwargs.get('tol', 0.75),
            'padding': kwargs.get('padding', 4.0),
            'element_colors': kwargs.get('element_colors', None),
            'show_legend': kwargs.get('show_legend', True)
        }

    def plot_comparison(self, orientation='horizontal'):
        if orientation == 'horizontal':
            print("\n--- Processing System 1 (Horizontal Top-Down View) ---")
            data_dict1 = self.system1._generate_refined_horizontal_data(self.params['padding'], 300, self.params['tol'])
            print("\n--- Processing System 2 (Horizontal Top-Down View) ---")
            data_dict2 = self.system2._generate_refined_horizontal_data(self.params['padding'], 300, self.params['tol'])
        elif orientation == 'vertical':
            print("\n--- Processing System 1 (Vertical Best-Fit Line View) ---")
            data_dict1 = self.system1._generate_linear_vertical_data(500, 300, self.params['tol'])
            print("\n--- Processing System 2 (Vertical Best-Fit Line View) ---")
            data_dict2 = self.system2._generate_linear_vertical_data(500, 300, self.params['tol'])
        else:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'")

        if data_dict1 is None or data_dict2 is None:
            print("Data could not be generated. Aborting plot.")
            return

        v = max(abs(np.nanmin(d["plot_data"])) for d in [data_dict1, data_dict2] if d is not None)
        norm = SymLogNorm(linthresh=v/100.0 if v > 0 else 0.01, linscale=0.5, vmin=-v, vmax=v, base=10)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        plot_params = [(axes[0], data_dict1, self.system1), (axes[1], data_dict2, self.system2)]
        
        colors_to_use = self.params['element_colors'] or {el: c for i, (el, c) in enumerate(zip(set(self.system1.atomtypes) | set(self.system2.atomtypes), plt.rcParams['axes.prop_cycle'].by_key()['color']))}

        for ax, data_dict, system_obj in plot_params:
            if data_dict is None:
                ax.text(0.5, 0.5, 'Could not generate plot', ha='center', va='center', transform=ax.transAxes)
                continue
                
            im = ax.imshow(data_dict["plot_data"], origin='lower', aspect='auto', extent=data_dict["extent"], cmap=self.params['cmap'], norm=norm)
            
            if orientation == 'horizontal':
                for a in data_dict["selected_atoms"]:
                    ax.scatter(a['coords'][0], a['coords'][1], c=colors_to_use.get(a['elem'], 'grey'), s=60, edgecolor='black', zorder=5)
                ax.set_xlabel(r'X-coordinate / $\AA$')
                ax.set_ylabel(r'Y-coordinate / $\AA$')
                ax.set_aspect('equal')
            
            elif orientation == 'vertical':
                for a in data_dict["selected_atoms"]:
                    ax.scatter(a['plot_x'], a['plot_y'], c=colors_to_use.get(a['elem'], 'grey'), s=60, edgecolor='black', zorder=5)
                ax.set_xlabel(r'Distance along Best-Fit Line / $\AA$')
                ax.set_ylabel(r'Z-coordinate / $\AA$')

            handles, seen_elems = [], set()
            # The structure of selected_atoms is different, so we unify it here for the legend
            atom_list_for_legend = data_dict["selected_atoms"]
            for a in atom_list_for_legend:
                if a['elem'] not in seen_elems:
                    handles.append(Patch(color=colors_to_use.get(a['elem'], 'grey'), label=a['elem']))
                    seen_elems.add(a['elem'])
            if self.params['show_legend'] and handles:
                ax.legend(handles=handles, title='Elements')
            
            clean_path = system_obj.ifile.rstrip('/\\')
            ax.set_title(f'System: {os.path.basename(clean_path)}')

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='bottom', pad=0.1, shrink=0.6, aspect=30)
        cbar.set_label(r'Electron Density Difference / $e\AA^{-3}$')
        plt.show()

# =============================================================================
# --- USER CONFIGURATION AND EXECUTION ---
# =============================================================================
if __name__ == "__main__":
    # --- Define System 1 ---
    vasp_dir1 = r"dir/system_1"
    #path1 = [[139], [109], [126]]
    path1=[[154], [119], [118], [122], [110], [126], [109], [139], [113], [136], [134], [133], [175]] # Example path for system 1
    

    # --- Define System 2 ---
    vasp_dir2 = r"dir/system_2"
    #path2 = [[31], [1], [18]]
    path2 = [[46], [11], [10], [14], [2], [18], [1], [31], [5], [28], [26], [25], [67]] # Example path for system 2

    # --- Define Shared Plotting Settings ---
    shared_colors = {'Au':'orange','N':'mediumaquamarine','C':'saddlebrown','H':'yellow'}

    # --- Create an instance of the slicer ---
    comparative_slicer = PointCloudSlicer(
        vasp_dir1=vasp_dir1, path1=path1,
        vasp_dir2=vasp_dir2, path2=path2,
        #tol=0.1, # Tolerance (in Å) for selecting atoms near the slice
        tol=0.65, # Tolerance (in Å) for selecting atoms near the slice
        padding=2.0, # Padding for the 'horizontal' view only
        element_colors=shared_colors
    )
    
    # --- Generate the plot ---
    # Choose your orientation: 'horizontal' or 'vertical'

    #comparative_slicer.plot_comparison(orientation='vertical')
    comparative_slicer.plot_comparison(orientation='horizontal') 