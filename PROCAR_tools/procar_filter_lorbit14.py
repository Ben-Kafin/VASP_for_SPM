import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from procar_parser import ProcarParser

class ProcarStatePlotter:
    def __init__(self, directory="path/to/directory", fermi_level=0.0, filter_types=None):
        """
        Initializes the state plotter.
        
        Parameters:
          directory    : Folder containing the PROCAR and POSCAR files.
                         Update this with your actual folder path.
          fermi_level  : Fermi level (in eV); energies will be plotted as (energy - fermi_level).
          filter_types : A list of element symbols.
        """
        self.directory = directory
        self.procar_path = os.path.join(directory, "PROCAR")
        self.poscar_path = os.path.join(directory, "POSCAR")
        self.fermi_level = fermi_level
        self.filter_types = filter_types  # For optional filtering.
        self.states = []  # Will store state dictionaries from PROCAR.
        self.states_combined = []  # Will store combined states (one per unique band).
        self.parser = None  # Shared ProcarParser instance (set by parse_procar).
        self.ispin = 1
        # Build a mapping from ion (atom) index to its element symbol.
        self.atom_type_map = self.parse_poscar()  

    def parse_poscar(self):
        """
        Parses the POSCAR file to build a mapping from atom (ion) indices to element symbols.
        Assumes the VASP 5+ format:
          - Line 1: Comment/title.
          - Line 2: Scaling factor.
          - Lines 3-5: Lattice vectors.
          - Line 6: Element symbols (space separated).
          - Line 7: Number of atoms for each element (space separated).
          
        Returns:
          A dictionary mapping atom numbers (starting at 1) to element symbols.
        """
        mapping = {}
        if not os.path.exists(self.poscar_path):
            print("POSCAR file not found in", self.directory)
            return mapping
        with open(self.poscar_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        if len(lines) < 7:
            print("POSCAR file is too short; cannot determine atom types.")
            return mapping
        elements = lines[5].split()
        try:
            counts = [int(x) for x in lines[6].split()]
        except Exception as e:
            print("Error parsing atom counts in POSCAR:", e)
            return mapping
        current_index = 1
        for i, elem in enumerate(elements):
            num = counts[i] if i < len(counts) else 0
            for _ in range(num):
                mapping[current_index] = elem  # Save as is.
                current_index += 1
        return mapping

    def parse_procar(self):
        """
        Reads PROCAR through the package's shared ProcarParser (procar_parser.py)
        and stores one state dictionary per (spin, k-point, band) in self.states.

        Each state carries the keys the rest of this class uses: "spin", "kpoint",
        "kweight", "band", "band_name", "energy" (raw eigenvalue; plotting subtracts
        self.fermi_level), "occ", "orbital_headers" (the file header without "ion"),
        "orbital_matrix" (ions x columns including the per-ion tot column, with the
        file's tot row appended when present), "contrib_by_ion" and "orbital_totals".
        ISPIN=1, ISPIN=2, noncollinear, LORBIT=10/11/12/14 and any orbital column set
        are handled by the shared parser.
        """
        self.parser = ProcarParser(self.directory, filter_types=self.filter_types)
        self.parser.parse()
        self.ispin = self.parser.ispin
        self.states = self.parser.get_states()
        print("Parsed", len(self.states), "states from PROCAR.")

    def build_state_types(self):
        """
        (Optional) Adds a new key 'types' to each state.
        Iterates over each ion in a state and uses the atom_type_map to determine the element;
        only ions with nonzero total contribution are included.
        """
        for state in self.states:
            types_set = set()
            tot_key = state.get("orbital_headers", ["tot"])[-1]
            for ion_index, contrib in state.get("contrib_by_ion", {}).items():
                if contrib.get(tot_key, 0.0) != 0:
                    elem = self.atom_type_map.get(ion_index, "Unknown")
                    types_set.add(elem)
            state["types"] = sorted(list(types_set))

    def filter_states(self):
        """
        (Optional) Filters out states that consist exclusively of atom types listed in self.filter_types.
        """
        if self.filter_types is None:
            return
        allowed_set = set(x.upper() for x in self.filter_types)
        filtered_states = []
        for state in self.states:
            state_types = {x.upper() for x in state.get("types", [])}
            if state_types and state_types.issubset(allowed_set):
                continue
            filtered_states.append(state)
        self.states = filtered_states
        print("After filtering, {} states remain.".format(len(self.states)))

    def plot_states(self):
        """
        Plots each state's aggregated total contribution (using the last column of orbital_matrix)
        per ion as vertical delta lines for visualization.
        """
        if not self.states:
            print("No states available for plotting; run parse_procar() first.")
            return
        plt.figure(figsize=(10, 6))
        for state in self.states:
            base_x = state["energy"] - self.fermi_level
            tot_array = state.get("orbital_matrix", np.array([]))[:, -1]
            for value in tot_array:
                plt.vlines(base_x, 0, value, colors="black", linewidth=1)
        plt.xlabel("Energy (eV) relative to Fermi level")
        plt.ylabel("Total contribution")
        plt.title("PROCAR States: Aggregated total contributions per ion")
        plt.grid(True)
        plt.show()

    def combine_kpoint_bands(self):
        """
        Combines states from different k-points that correspond to the same band.
        A weighted average (using k-point weights) of the orbital_matrix, energy, and occupancy is computed.
        For ISPIN=2 the two spin channels are combined separately (grouped by (spin, band)).
        """
        band_groups = defaultdict(list)
        for state in self.states:
            band_groups[(state.get("spin", 1), state["band"])].append(state)
        
        combined_states = []
        for (spin, band), states in band_groups.items():
            total_weight = sum(s["kweight"] for s in states)
            combined_matrix = sum(s["orbital_matrix"] * s["kweight"] for s in states) / total_weight
            avg_energy = sum(s["energy"] * s["kweight"] for s in states) / total_weight
            avg_occ = sum(s["occ"] * s["kweight"] for s in states) / total_weight
            
            combined_states.append({
                "spin": spin,
                "band": band,
                "band_name": states[0].get("band_name", f"band {band}"),
                "energy": avg_energy,
                "occ": avg_occ,
                "orbital_matrix": combined_matrix,
                "total_weight": total_weight
            })
        
        combined_states = sorted(combined_states, key=lambda s: s["energy"])
        self.states_combined = combined_states
        return combined_states

    def export_combined_bands(self, out_filename="combined_composition_matrices.txt"):
        """
        Exports the combined composition matrix for each band into a tab-delimited file.
        The file is written into the provided directory.
        """
        combined_states = self.combine_kpoint_bands()
        out_lines = []
        for new_index, state in enumerate(combined_states, start=1):
            M = state["orbital_matrix"]
            norm_val = np.linalg.norm(M)
            norm_M = M / norm_val if norm_val != 0 else M
            spin_tag = f"\tSpin: {state['spin']}" if self.ispin == 2 else ""
            out_lines.append(f"Band {new_index}{spin_tag}\tOriginalBand: {state['band']}\tEnergy: {state['energy']:.4f} eV")
            num_mapped = len(self.atom_type_map)
            n_rows = state["orbital_matrix"].shape[0]
            mapped_indices = sorted(self.atom_type_map.keys())
            restricted_rows = []
            for idx in mapped_indices:
                if idx <= n_rows:
                    restricted_rows.append(state["orbital_matrix"][idx-1])
                else:
                    restricted_rows.append(np.zeros(M.shape[1]))
            restricted_matrix = np.array(restricted_rows)
            fro_norm = np.linalg.norm(restricted_matrix)
            norm_restricted = restricted_matrix / fro_norm if fro_norm != 0 else restricted_matrix
            for row in norm_restricted:
                row_str = "\t".join(f"{val:.4e}" for val in row)
                out_lines.append(row_str)
            out_lines.append("")
        # Build the full file path in the provided directory.
        full_out_filename = os.path.join(self.directory, out_filename)
        with open(full_out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported combined band composition matrices to {full_out_filename}")

    def run(self):
        self.parse_procar()
        self.build_state_types()
        self.filter_states()
        self.plot_states()
        self.export_combined_bands()

# ----- Run the Code -----
if __name__ == "__main__":
    directory = 'dir/run'
    fermi_level = -1.9849  # Example value; adjust as needed.
    filter_types = []
    
    plotter = ProcarStatePlotter(directory, fermi_level, filter_types)
    plotter.run()