# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:50:29 2026

@author: Benjamin Kafin
"""

import numpy as np

CERTIFIED_DIGITS = 12
SNAP_TOL = 5e-13

class POSCARTransformer:
    def __init__(self, input_file, output_file, new_lattice_vectors, mode='centered', retain_selective_dynamics=True,
                 fixed_width=False):
        self.input_file = input_file
        self.output_file = output_file
        self.new_lattice_vectors = np.array(new_lattice_vectors)
        # Round new lattice vectors to certified precision so the matrix inverse
        # is computed from clean values and printed values match used values exactly.
        self.new_lattice_vectors = np.round(self.new_lattice_vectors, CERTIFIED_DIGITS)
        self.mode = mode
        self.retain_selective_dynamics = retain_selective_dynamics
        # fixed_width=True writes the fixed-width VASP column format (ported from
        # reshape_unitcell.py: CRLF line endings, '1.0' scaling, 20/21/21-wide
        # lattice columns, 5-wide species/count columns, 16/20/20-wide coordinates).
        # fixed_width=False keeps the 12-decimal writer below.
        self.fixed_width = fixed_width
        self.old_lattice_vectors = None
        self.atom_types = None
        self.atom_numbers = None
        self.cartesian_coordinates = None  # Cartesian coordinates remain fixed in space
        self.selective_flags = None
        self._input_has_seldyn = False

    def read_poscar(self):
        with open(self.input_file, 'r') as file:
            lines = file.readlines()
        
        scaling_factor = float(lines[1])
        self.old_lattice_vectors = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scaling_factor
        self.atom_types = lines[5].split()
        self.atom_numbers = [int(x) for x in lines[6].split()]
        total_atoms = sum(self.atom_numbers)

        # Detect optional "Selective dynamics" line at index 7
        line7_stripped = lines[7].strip()
        if line7_stripped and line7_stripped[0].lower() == 's':
            self._input_has_seldyn = True
            coord_start = 9  # skip Selective dynamics (7) and coord-mode (8)
        else:
            self._input_has_seldyn = False
            coord_start = 8  # skip only coord-mode (7)

        fractional_coordinates = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(coord_start, coord_start + total_atoms)])

        if self._input_has_seldyn:
            # Capture per-atom T/F flags verbatim (tokens 3, 4, 5 of each coord line)
            self.selective_flags = [lines[i].split()[3:6] for i in range(coord_start, coord_start + total_atoms)]

        if self.mode == 'centered':
            # Zero the center in the xy-plane
            fractional_coordinates[:, :2] -= 0.5  # Shift xy-coordinates to center (subtract 0.5)
        
        self.cartesian_coordinates = self.fractional_to_cartesian(fractional_coordinates, self.old_lattice_vectors)

    def write_poscar(self):
        # Convert Cartesian coordinates back to fractional using the new lattice
        fractional_coordinates = self.cartesian_to_fractional(self.cartesian_coordinates, self.new_lattice_vectors)
        
        if self.mode == 'centered':
            # Restore the center in the xy-plane
            fractional_coordinates[:, :2] += 0.5  # Add 0.5 to shift back to center in xy

        # Snap near-zero values (within float64 round-trip noise) to exactly 0.0
        # so boundary atoms (frac == 0 in input) survive the mask instead of being
        # dropped due to sub-ulp negative noise from the lattice round-trip.
        fractional_coordinates[np.abs(fractional_coordinates) < SNAP_TOL] = 0.0

        # Identify atoms that are within the [0, 1) range in all three dimensions
        mask = np.all((fractional_coordinates >= 0) & (fractional_coordinates < 1), axis=1)
        
        # Filter coordinates
        filtered_coords = fractional_coordinates[mask]

        # Filter selective dynamics flags by the same mask, if we'll be writing them
        write_seldyn = self.retain_selective_dynamics and self._input_has_seldyn
        if write_seldyn:
            filtered_flags = [self.selective_flags[i] for i in range(len(self.selective_flags)) if mask[i]]

        # Update atom counts and types based on survivors
        new_atom_numbers = []
        new_atom_types = []
        current_idx = 0
        
        for i, count in enumerate(self.atom_numbers):
            # Check how many atoms in this species group survived the mask
            species_mask = mask[current_idx : current_idx + count]
            surviving_count = np.sum(species_mask)
            
            if surviving_count > 0:
                new_atom_numbers.append(surviving_count)
                new_atom_types.append(self.atom_types[i])
            
            current_idx += count

        if self.fixed_width:
            with open(self.output_file, 'w', newline='\r\n') as file:
                file.write("Generated POSCAR\n")
                file.write("1.0\n")
                for vec in self.new_lattice_vectors:
                    file.write('{:>20.10f}{:>21.10f}{:>21.10f}\n'.format(vec[0], vec[1], vec[2]))
                for at in new_atom_types:
                    file.write('{:>5}'.format(str(at)))
                file.write('\n')
                for num in new_atom_numbers:
                    file.write('{:>5}'.format(num))
                file.write('\n')
                if write_seldyn:
                    file.write("Selective dynamics\n")
                file.write("Direct\n")
                if write_seldyn:
                    for coord, flags in zip(filtered_coords, filtered_flags):
                        file.write('{:>16.9f}{:>20.9f}{:>20.9f}  {}  {}  {}\n'.format(coord[0], coord[1], coord[2], *flags))
                else:
                    for coord in filtered_coords:
                        file.write('{:>16.9f}{:>20.9f}{:>20.9f}\n'.format(coord[0], coord[1], coord[2]))
            return

        with open(self.output_file, 'w') as file:
            file.write("Generated POSCAR\n")
            file.write("  {:.12f}\n".format(1.0))
            for vec in self.new_lattice_vectors:
                file.write("  {:.12f}  {:.12f}  {:.12f}\n".format(*vec))
            file.write("  " + "  ".join(new_atom_types) + "\n")
            file.write("  " + "  ".join(map(str, new_atom_numbers)) + "\n")
            if write_seldyn:
                file.write("Selective dynamics\n")
            file.write("Direct\n")
            if write_seldyn:
                for coord, flags in zip(filtered_coords, filtered_flags):
                    file.write("  {:.12f}  {:.12f}  {:.12f}  {}  {}  {}\n".format(*coord, *flags))
            else:
                for coord in filtered_coords:
                    file.write("  {:.12f}  {:.12f}  {:.12f}\n".format(*coord))

    def fractional_to_cartesian(self, fractional_coords, lattice_vectors):
        return np.dot(fractional_coords, lattice_vectors)

    def cartesian_to_fractional(self, cartesian_coords, lattice_vectors):
        inverse_lattice = np.linalg.inv(lattice_vectors)
        return np.dot(cartesian_coords, inverse_lattice)

    def transform(self):
        self.read_poscar()  # Read old POSCAR and zero-center
        self.write_poscar()  # Transform and filter atoms outside bounds
        print(f"Transformation complete. New POSCAR written to {self.output_file}")


# Example usage
if __name__ == "__main__":

    # Set mode to 'origin' to keep the absolute origin, or 'centered' to use previous behavior
    transformer = POSCARTransformer(
        'dir/full/CONTCAR',
        'dir/output/POSCAR', 
        [
            [17.9802889999999991,    0.0000000000000000,    0.0000000000000000],
            [-8.9901499999999999,   15.5713910000000002,    0.0000000000000000],
            [0.0000000000000000,    0.0000000000000000,   25.0]
        ],
        mode='origin',
        retain_selective_dynamics=True
    )
    transformer.transform()