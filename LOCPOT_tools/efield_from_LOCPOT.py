#!/usr/bin/env python3
"""
This script defines a class that:
 • Loads a VASP LOCPOT file using pymatgen.
 • Computes the 3D electric field E = -∇V from the LOCPOT data.
 • Writes the electric field as volumetric data in VASP (LOCPOT/CHGCAR) style.
   In this version, the three components (Ex, Ey, Ez) are combined into one file.
   The header (including atomic positions) is taken from the POSCAR output,
   and the final file is written into the same directory as the input LOCPOT.
   
The final output file, e.g. “E_field_vector_LOCPOT”, can be opened in VESTA (or similar)
and should represent a vector field at each grid point.
Before running, ensure pymatgen is installed:
    pip install pymatgen
"""

import os
import numpy as np
from pymatgen.io.vasp.outputs import Locpot
from pymatgen.io.vasp.inputs import Poscar

class ElectricFieldCalculator:
    def __init__(self, locpot_filename="LOCPOT"):
        """
        Initialize by loading the LOCPOT file (via pymatgen) and extracting
        the potential, the lattice (cell) and the grid dimensions.
        Also, store the directory of the LOCPOT file so that output files are placed there.
        """
        self.locpot_filename = locpot_filename
        self.locpot_path = os.path.dirname(os.path.abspath(locpot_filename))
        try:
            self.lp = Locpot.from_file(locpot_filename)
        except Exception as e:
            raise Exception("Error loading LOCPOT file: " + str(e))
        
        # Extract potential data from the LOCPOT.
        self.potential = self.lp.data.get("total", None)
        if self.potential is None:
            self.potential = self.lp.data
        
        # Extract lattice (cell) information from the structure.
        self.cell = self.lp.structure.lattice.matrix  # 3x3 numpy array (each row is a lattice vector)
        self.ng = self.potential.shape  # Grid dimensions: (nx, ny, nz)
        
        # Compute voxel spacing (delta) vectors along each lattice direction.
        self.dv = []
        for i in range(3):
            cell_vec = np.array(self.cell[i])
            delta_vec = cell_vec / float(self.ng[i])
            self.dv.append(delta_vec)
        
        # Placeholders for computed electric field components.
        self.Ex = None
        self.Ey = None
        self.Ez = None

    def compute_electric_field(self):
        """
        Compute the electric field from the potential via finite differences:
            E = -∇V.
        The spacing is determined by the norms of the dv vectors.
        """
        dx = np.linalg.norm(self.dv[0])
        dy = np.linalg.norm(self.dv[1])
        dz = np.linalg.norm(self.dv[2])
        dVdx, dVdy, dVdz = np.gradient(self.potential, dx, dy, dz)
        self.Ex = -dVdx
        self.Ey = -dVdy
        self.Ez = -dVdz
        print("Electric field computed successfully.")

    def _write_volumetric_file(self, field, filename, ds=1):
        """
        Write a single scalar field volumetric file in VASP format.
        (This is the original routine for a single component.)
        
        Parameters:
          field    : 3D numpy array (one component, e.g. Ex).
          filename : Name of the output file (will be placed in the LOCPOT file's directory).
          ds       : Downsampling factor (ds = 1 gives the full resolution).
        """
        if ds < 1:
            ds = 1
        field_ds = field[::ds, ::ds, ::ds]
        ng_ds = field_ds.shape  # New grid dimensions (nx, ny, nz)
        nx, ny, nz = ng_ds
        
        # Get the header from the structure using pymatgen's Poscar (includes atom positions).
        poscar = Poscar(self.lp.structure)
        header_str = poscar.get_string()
        
        out_path = os.path.join(self.locpot_path, filename)
        with open(out_path, "w") as f:
            f.write(header_str.strip() + "\n")
            f.write("\n")
            # Write grid dimensions.
            f.write(" {:4d} {:4d} {:4d}\n".format(nx, ny, nz))
            # Flatten the data in Fortran order (x changes fastest), and write 5 numbers per line.
            flat_data = field_ds.ravel(order="F")
            numbers_per_line = 5
            count = 0
            for val in flat_data:
                f.write("{: 13.5E}".format(val))
                count += 1
                if count % numbers_per_line == 0:
                    f.write("\n")
            if count % numbers_per_line != 0:
                f.write("\n")
        print(f"Volumetric file '{out_path}' written successfully with grid dimensions {ng_ds} (ds={ds}).")
        
    def write_locpot_format_fields(self, ds=1):
        """
        Write the electric field components (Ex, Ey, Ez) separately in VASP volumetric file format.
        Three files will be created in the LOCPOT file's directory:
         E_field_Ex_LOCPOT, E_field_Ey_LOCPOT, and E_field_Ez_LOCPOT.
        """
        if any(comp is None for comp in (self.Ex, self.Ey, self.Ez)):
            raise Exception("Electric field has not been computed. Run compute_electric_field() first.")
        
        self._write_volumetric_file(self.Ex, "E_field_Ex_LOCPOT", ds)
        self._write_volumetric_file(self.Ey, "E_field_Ey_LOCPOT", ds)
        self._write_volumetric_file(self.Ez, "E_field_Ez_LOCPOT", ds)
    
    def write_combined_vector_field(self, ds=1):
        """
        Write a single volumetric file that combines the three electric field components
        into one file, in a VASP-like format.
        
        The file will have a header (as from a full POSCAR, including atom positions),
        then a blank line followed by a line with the grid dimensions.
        For each grid point (flattened in Fortran order) the three components (Ex, Ey, Ez)
        are written together.
        
        Parameters:
          ds : Downsampling factor (ds = 1 gives full resolution).
          
        The output file is named "E_field_vector_LOCPOT" and is stored in the same directory
        as the input LOCPOT.
        """
        if any(comp is None for comp in (self.Ex, self.Ey, self.Ez)):
            raise Exception("Electric field has not been computed. Run compute_electric_field() first.")
        if ds < 1:
            ds = 1
        
        # Downsample the arrays.
        Ex_ds = self.Ex[::ds, ::ds, ::ds]
        Ey_ds = self.Ey[::ds, ::ds, ::ds]
        Ez_ds = self.Ez[::ds, ::ds, ::ds]
        ng_ds = Ex_ds.shape  # New grid dimensions (nx, ny, nz)
        nx, ny, nz = ng_ds
        
        # Get the header from the structure via Poscar.
        poscar = Poscar(self.lp.structure)
        header_str = poscar.get_string()
        
        out_filename = os.path.join(self.locpot_path, "E_field_vector_LOCPOT")
        with open(out_filename, "w") as f:
            f.write(header_str.strip() + "\n")
            f.write("\n")
            # Write the new grid dimensions.
            f.write(" {:4d} {:4d} {:4d}\n".format(nx, ny, nz))
            
            # Flatten each downsampled array in Fortran order.
            Ex_flat = Ex_ds.ravel(order="F")
            Ey_flat = Ey_ds.ravel(order="F")
            Ez_flat = Ez_ds.ravel(order="F")
            total_points = Ex_flat.size
            
            # We'll write two vector triples per line (i.e. two sets of 3 numbers, total 6 numbers per line).
            vectors_per_line = 2
            count = 0
            for i in range(total_points):
                # Format each component with scientific notation.
                f.write("{: 13.5E}{: 13.5E}{: 13.5E} ".format(Ex_flat[i], Ey_flat[i], Ez_flat[i]))
                count += 1
                if count % vectors_per_line == 0:
                    f.write("\n")
            if count % vectors_per_line != 0:
                f.write("\n")
        print(f"Combined vector field file '{out_filename}' written successfully with grid dimensions {ng_ds} (ds={ds}).")


# Example run code
if __name__ == "__main__":
    print("Initializing ElectricFieldCalculator with LOCPOT (using pymatgen)...")
    ef_calc = ElectricFieldCalculator('dir/molecule/LOCPOT')
    
    print("Computing the electric field (E = -∇V)...")
    ef_calc.compute_electric_field()
    
    # Optionally, write separate components:
    # ds_factor = 1  # Full resolution
    # ef_calc.write_locpot_format_fields(ds=ds_factor)
    
    # Write the combined vector field into a single file.
    ds_factor = 1
    ef_calc.write_combined_vector_field(ds=ds_factor)