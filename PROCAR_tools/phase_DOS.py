import numpy as np
import os
import sys
from procar_parser import ProcarParser

class PhaseSensitiveSiteProjectedDOS:
    """
    Generates an atom-specific and orbital-specific DOS using coherent
    summation with verbose console updates for tracking integration progress.
    """
    def __init__(self, directory, energy_range=(-10, 5), n_points=2000, sigma=0.05):
        self.directory = directory
        self.energy_grid = np.linspace(energy_range[0], energy_range[1], n_points)
        self.sigma = sigma
        self.parser = ProcarParser(directory=self.directory)  # shared PROCAR reader
        
        # Data storage
        self.site_dos = None
        self.total_dos = np.zeros(n_points)

    def run(self):
        print("\n" + "="*60)
        print(" PHASE-SENSITIVE DOS INTEGRATION PROTOCOL ".center(60, "#"))
        print("="*60)
        
        # 1. Parse data with status updates
        print(f"[*] Accessing VASP directory: {self.directory}")
        self.parser.parse()
        if not self.parser.has_phase:
            raise ValueError("phase_DOS needs the complex phase block (LORBIT=12/13/14); "
                             f"the PROCAR in {self.directory} has none.")
        # Energies relative to the DOSCAR Fermi level, as before.
        states = self.parser.get_states(fermi_relative=True)
        
        if not states:
            print("[!] ERROR: No states found. Check PROCAR and LORBIT settings.")
            return

        # 2. Extract System Metadata
        orbital_headers = states[0]["phase_headers"]
        num_ions = max(self.parser.atom_type_map.keys())
        num_orbs = len(orbital_headers)
        total_states = len(states)
        
        # Initialize 3D matrix: [Ion][Orbital][EnergyGridPoint]
        self.site_dos = np.zeros((num_ions, num_orbs, len(self.energy_grid)))

        print(f"[*] Fermi Level (from DOSCAR): {self.parser.fermi_level:.4f} eV") #
        print(f"[*] Energy Range: {self.energy_grid[0]:.2f} to {self.energy_grid[-1]:.2f} eV")
        print(f"[*] Resolution: {len(self.energy_grid)} points | Sigma: {self.sigma} eV")
        print(f"[*] System: {num_ions} ions detected with {num_orbs} orbitals each.")
        print(f"[*] Total states to process (k-points * bands): {total_states}")
        print("-" * 60)

        # 3. Full Integration Loop with Progress Tracking
        print("[>] Starting Coherent Integration...")
        
        for idx, st in enumerate(states):
            Cc = st["coeff_matrix"]
            kweight = st["kweight"]
            energy = st["energy"]
            
            # Progress bar update every 10% or 500 states
            if idx % max(1, (total_states // 10)) == 0:
                percent = (idx / total_states) * 100
                print(f"    Progress: {percent:3.0f}% | Processing state {idx}/{total_states}...")

            # Gaussian Broadening Kernel
            energy_diff = self.energy_grid - energy
            gaussian = (1.0 / (self.sigma * np.sqrt(2 * np.pi))) * \
                       np.exp(-0.5 * (energy_diff / self.sigma)**2)

            # --- Total Coherent DOS (System-wide Interference) ---
            total_amp = np.sum(Cc)
            t_weight = (np.abs(total_amp)**2) * kweight
            self.total_dos += t_weight * gaussian

            # --- Site-Projected DOS ---
            for i in range(num_ions):
                for o in range(num_orbs):
                    val = Cc[i, o]
                    weight = (np.abs(val)**2) * kweight
                    self.site_dos[i, o, :] += weight * gaussian

        print(f"    Progress: 100% | Integration complete.")
        print("-" * 60)
        self.export_verbose_output(orbital_headers)

    def export_verbose_output(self, orbital_headers):
        out_name = "PHASE_SENSITIVE_DOSCAR.dat"
        out_path = os.path.join(self.directory, out_name)
        
        print(f"[*] Writing verbose output to: {out_name}")
        
        with open(out_path, "w") as f:
            f.write(f"VASP Phase-Sensitive DOS (Coherent Summation)\n")
            f.write(f"Ions: {self.site_dos.shape[0]} | Sigma: {self.sigma}\n")
            f.write(f"Energy relative to E_Fermi: {self.parser.fermi_level:.6f}\n\n")

            # Block 1: Total Phase-Sensitive DOS
            f.write("TOTAL COHERENT DOS\n")
            f.write("Energy\tDOS\n")
            for e_idx, energy in enumerate(self.energy_grid):
                f.write(f"{energy:.6f}\t{self.total_dos[e_idx]:.8e}\n")
            f.write("\n")

            # Block 2: Site-Projected Blocks
            for i in range(self.site_dos.shape[0]):
                elem = self.parser.atom_type_map.get(i+1, "X") #
                f.write(f"SITE: Ion {i+1} ({elem})\n")
                header = "Energy\t" + "\t".join(orbital_headers) + "\n"
                f.write(header)
                
                for e_idx, energy in enumerate(self.energy_grid):
                    row = [f"{energy:.6f}"]
                    for o_idx in range(len(orbital_headers)):
                        row.append(f"{self.site_dos[i, o_idx, e_idx]:.8e}")
                    f.write("\t".join(row) + "\n")
                f.write("\n")

        print("[!] SUCCESS: Phase-sensitive DOS integration finished.")
        print("="*60 + "\n")

if __name__ == "__main__":
    # Benjamin, ensure your directory path is correct for the terminal environment
    target_dir = r'dir/run'
    
    integrator = PhaseSensitiveSiteProjectedDOS(
        directory=target_dir,
        energy_range=(-8, 4),
        n_points=3000,
        sigma=0.05
    )
    integrator.run()