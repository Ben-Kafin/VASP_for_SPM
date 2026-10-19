import os
import re
import numpy as np
import cupy as cp
from collections import defaultdict
from procar_parser import ProcarParser

class GPUComplexProcarParser:
    """
    GPU workflow on the complex PROCAR coefficients (LORBIT=12/13/14).
    PROCAR, POSCAR and DOSCAR are read by the package's shared ProcarParser
    (procar_parser.py), which handles ISPIN=1, ISPIN=2 and noncollinear files;
    this class moves the per-state coefficient matrices to the GPU, pairs the
    two spin channels per k-point and band (incoherent k-point separation,
    spin-polarized summation), applies square-root Gaussian broadening with
    square-root k-weighting, and saves Broadened_Orbital_Curves.npy.
    """
    def __init__(self, directory="path/to/directory", filter_types=None):
        self.directory       = directory
        self.filter_types    = filter_types or []
        self.parser          = ProcarParser(directory, filter_types=self.filter_types)  # shared PROCAR reader
        self.fermi_level     = self.parser.fermi_level
        self.atom_type_map   = self.parser.atom_type_map
        self.states          = []  
        self.states_combined = []
        self.ispin           = 1 

    def parse_procar(self):
        """Reads PROCAR through the shared ProcarParser and moves the complex
        coefficients (both spins, all k-points, collinear or noncollinear) to GPU."""
        self.parser.parse()
        if not self.parser.has_phase:
            raise ValueError("procar_spin needs the complex phase block (LORBIT=12/13/14); "
                             f"the PROCAR in {self.directory} has none.")
        if self.fermi_level is None:
            raise ValueError(f"DOSCAR not found in {self.directory}; the Fermi energy is required.")
        self.ispin = self.parser.ispin
        self.states = []
        for st in self.parser.get_states(fermi_relative=True):
            self.states.append({
                "spin": st["spin"], "kpoint": st["kpoint"], "kweight": st["kweight"],
                "band": st["band"], "energy": st["energy"], "phase_headers": st["phase_headers"],
                "coeff_matrix": cp.array(st["coeff_matrix"], dtype=cp.complex64)
            })
        
        print(f"Parsed {len(self.states)} states (ISPIN={self.ispin}) to GPU memory.")

    def combine_kpoint_bands(self):
        """Pairs ISPIN=2 matrices by orthogonal k-point & band. Severed unphysical averaging."""
        groups = defaultdict(lambda: defaultdict(dict))
        for st in self.states:
            groups[st["kpoint"]][st["band"]][st["spin"]] = st

        comb = []
        for kpt in sorted(groups.keys()):
            for band in sorted(groups[kpt].keys()):
                spins = groups[kpt][band]
                kwt = spins[1]["kweight"]
                if self.ispin == 2:
                    C_spin_combined = cp.stack([spins[1]["coeff_matrix"], spins[2]["coeff_matrix"]])
                    E_final = (spins[1]["energy"] + spins[2]["energy"]) / 2.0
                else:
                    C_spin_combined = spins[1]["coeff_matrix"]
                    E_final = spins[1]["energy"]

                comb.append({
                    "kpoint": kpt, "kweight": kwt, "band": band, "energy": E_final,
                    "coeff_matrix": C_spin_combined, 
                    "phase_headers": spins[1]["phase_headers"]
                })
                
        self.states_combined = comb
        return self.states_combined

    def apply_broadening_and_save(self, sigmas, e_min=-5.0, e_max=5.0, num_points=1000):
        """
        Applies Square Root Gaussian and Square Root K-Weighting.
        Extends output tensor shape to preserve independent K-Points.
        """
        print(f"Applying Gaussian Broadening. Generating {num_points} energy points...")
        energy_grid = cp.linspace(e_min, e_max, num_points, dtype=cp.float32)
        
        N_atoms = max(self.atom_type_map.keys()) if self.atom_type_map else 0
        if not self.states_combined: return
        M_orbs = self.states_combined[0]["coeff_matrix"].shape[-1]
        
        unique_kpts = sorted(list(set(st["kpoint"] for st in self.states_combined)))
        kpt_idx_map = {k: i for i, k in enumerate(unique_kpts)}
        N_kpts = len(unique_kpts)
        
        # Extended tensor structure: (ISPIN, NKPTS, NATOMS, NORBS, NENERGIES)
        if self.ispin == 2:
            broadened_coeffs = cp.zeros((2, N_kpts, N_atoms, M_orbs, num_points), dtype=cp.complex64)
        else:
            broadened_coeffs = cp.zeros((N_kpts, N_atoms, M_orbs, num_points), dtype=cp.complex64)

        sigma_array = cp.zeros(N_atoms, dtype=cp.float32)
        for idx, elem in self.atom_type_map.items():
            sigma_array[idx-1] = sigmas.get(elem, 0.1)

        for state in self.states_combined:
            k_idx = kpt_idx_map[state["kpoint"]]
            E_state = state["energy"]
            C_state = state["coeff_matrix"] 
            kwt = state["kweight"]
            
            sig_reshaped = sigma_array[:, None, None]
            exponent = -0.25 * ((energy_grid[None, None, :] - E_state) / sig_reshaped)**2
            prefactor = cp.sqrt(kwt / (sig_reshaped * cp.sqrt(2.0 * cp.pi)))
            weights = prefactor * cp.exp(exponent)
            
            if self.ispin == 2:
                C_expanded = C_state[..., None] 
                broadened_coeffs[:, k_idx, ...] += C_expanded * weights[None, ...]
            else:
                C_expanded = C_state[..., None]
                broadened_coeffs[k_idx, ...] += C_expanded * weights
                
        output_data = {
            'energy_grid': cp.asnumpy(energy_grid),
            'broadened_coeffs': cp.asnumpy(broadened_coeffs),
            'ef': self.fermi_level,
            'headers': self.states_combined[0]["phase_headers"]
        }
        
        out_path = os.path.join(self.directory, "Broadened_Orbital_Curves.npy")
        np.save(out_path, output_data)
        print(f"Cache generated successfully at: {out_path}")
        print(f"Tensor Shape: {output_data['broadened_coeffs'].shape}")

    def run_gpu_workflow(self, sigmas=None):
        self.parse_procar()
        self.combine_kpoint_bands()
        print("GPU Spin-Polarized Integration Complete.")
        
        if sigmas is not None:
            self.apply_broadening_and_save(sigmas)

if __name__ == "__main__":
    target_dir = 'dir'
    my_sigmas = {'C': 0.1, 'H': 0.1, 'N': 0.1, 'Au': 0.2}
    parser = GPUComplexProcarParser(target_dir)
    parser.run_gpu_workflow(sigmas=my_sigmas)