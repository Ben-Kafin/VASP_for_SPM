import os
import re

class ProjcarParser:
    r"""
    A parser for VASP PROJCAR files that extracts k-point values, fractional coordinates,
    spin, header, and projection data.

    Expected (but flexible) structure (extra blanks allowed):
      • A line containing "ISITE:" or "projection for atom" (case‑insensitive) indicates an atomic site.
        The first number found after "ISITE:" is taken as the site number.
      • In the same line, if the string "R=" appears, the three numbers immediately following "R="
        are interpreted as the fractional coordinates for that site.
      • A line containing both "k-point:" and "spin:" indicates a new spin block.
        The k-point is extracted using the regex pattern: r"k[-]?point:\s*(\S+)"
        and the spin is extracted using the regex pattern: r"spin:\s+(\d+)".
        For example, the line:
            k-point:     1  spin:   2
        sets the k-point to "1" and the spin channel to "spin2".
      • Immediately after a k-point/spin marker, the next nonblank line is examined.
        If its first token (split on whitespace) is "Band" (case‑insensitive) then it is taken as the header;
        otherwise, that line is skipped and the following nonblank line is used.
      • The header line must have tokens: the first is "Band" (case‑insensitive), and the remainder
        is one token per orbital.
      • Each subsequent nonblank line (until a new marker) is a data row.
        Each row starts with the band number followed by 2×(number of orbitals) numeric values
        (two coefficients per orbital).

    The parsed projection data is stored in a nested dictionary using k-point values as the top-level key:
         final_data[kpt][band][site][spin] = {
             "header": [orbital names],
             "orbitals": {orbital: (val1, val2), ...}
         }
         
    In addition, fractional coordinates are stored in self.site_coords as:
         self.site_coords[site] = (x, y, z)
    """
    def __init__(self, directory):
        self.directory = directory
        self.site_coords = {}  # Maps site number (int) to fractional coordinates (tuple of 3 floats)
        self.filepath = self.find_projcar_file()
        
    def find_projcar_file(self):
        potential_filenames = ["PROJCAR.txt", "PROJCAR"]
        for fname in potential_filenames:
            path = os.path.join(self.directory, fname)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No PROJCAR file found in directory '{self.directory}'. Expected one of: {', '.join(potential_filenames)}")
    
    def parse(self):
        final_data = {}
        current_kpoint = None       
        current_site = None         
        current_spin = None         
        current_spin_header = None  
        current_spin_rows = []      
        state = "global"
        
        def commit_spin_block():
            nonlocal final_data, current_kpoint, current_site, current_spin, current_spin_header, current_spin_rows
            if current_kpoint is None:
                print("Warning: k-point value not set for this spin block; skipping block commit.")
                return
            if current_site is None or current_spin_header is None or not current_spin_rows:
                return
            tokens = current_spin_header
            if tokens[0].strip().lower() != "band":
                print(f"Warning: Expected header's first token 'Band'; got '{tokens[0]}'")
            n_orb = len(tokens) - 1
            orbital_names = [tokens[i].strip() for i in range(1, len(tokens))]
            for row in current_spin_rows:
                row_tokens = re.split(r"\s+", row.strip())
                if not row_tokens:
                    continue
                try:
                    band_num = int(row_tokens[0])
                except ValueError:
                    print(f"Warning: Cannot parse band number from '{row_tokens[0]}'")
                    continue
                if len(row_tokens) < 1 + 2 * n_orb:
                    print("Warning: Incomplete data row for", n_orb, "orbitals")
                    continue
                orb_dict = {}
                for i, orb in enumerate(orbital_names):
                    try:
                        v1 = float(row_tokens[1 + 2*i])
                        v2 = float(row_tokens[1 + 2*i + 1])
                    except ValueError:
                        v1 = v2 = None
                    orb_dict[orb] = (v1, v2)
                if current_kpoint not in final_data:
                    final_data[current_kpoint] = {}
                if band_num not in final_data[current_kpoint]:
                    final_data[current_kpoint][band_num] = {}
                if current_site not in final_data[current_kpoint][band_num]:
                    final_data[current_kpoint][band_num][current_site] = {}
                final_data[current_kpoint][band_num][current_site][current_spin] = {
                    "header": orbital_names,
                    "orbitals": orb_dict
                }
            current_spin_header = None
            current_spin_rows.clear()
        
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        for ln in lines:
            line = ln.strip()
            if line == "":
                continue

            # If line contains both "ISITE:" and "R=", extract both isite number and coords.
            if "isite" in line.lower() and "r=" in line.lower():
                m_site = re.search(r"isite:\s*(\d+)", line, flags=re.IGNORECASE)
                m_coords = re.search(r"r=\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)", line, flags=re.IGNORECASE)
                if m_site and m_coords:
                    current_site = int(m_site.group(1))
                    coords = (float(m_coords.group(1)), float(m_coords.group(2)), float(m_coords.group(3)))
                    self.site_coords[current_site] = coords
                continue

            # Detect isite marker (if not processed above).
            if "isite" in line.lower() or "projection for atom" in line.lower():
                if state in ("reading_data", "expect_header", "expect_skip"):
                    commit_spin_block()
                    current_spin_header = None
                    current_spin_rows.clear()
                    state = "global"
                nums = re.findall(r"\d+", line)
                if nums:
                    current_site = int(nums[0])
                else:
                    print("Warning: Could not extract site number")
                state = "global"
                continue

            # Detect k-point + spin marker.
            if re.search(r"k[-]?point:", line.lower()) and re.search(r"spin:", line.lower()):
                if state in ("reading_data", "expect_header", "expect_skip"):
                    commit_spin_block()
                    current_spin_header = None
                    current_spin_rows.clear()
                m_kpt = re.search(r"k[-]?point:\s*(\S+)", line.lower())
                if m_kpt:
                    current_kpoint = m_kpt.group(1)
                else:
                    print("Warning: Could not parse k-point")
                m_spin = re.search(r"spin:\s+(\d+)", line.lower())
                if m_spin:
                    spin_num = m_spin.group(1)
                    if spin_num == "1":
                        current_spin = "spin1"
                    elif spin_num == "2":
                        current_spin = "spin2"
                    else:
                        current_spin = "unknown"
                else:
                    current_spin = "spin1"
                state = "expect_skip"
                continue

            if state == "expect_skip":
                tokens = re.split(r"\s+", line)
                if tokens and tokens[0].strip().lower() == "band":
                    current_spin_header = tokens
                    state = "reading_data"
                    continue
                else:
                    state = "expect_header"
                    continue

            if state == "expect_header":
                current_spin_header = re.split(r"\s+", line)
                state = "reading_data"
                continue

            if state == "reading_data":
                if ("isite" in line.lower()) or re.search(r"k[-]?point:", line.lower()) or ("spin" in line.lower()):
                    commit_spin_block()
                    current_spin_header = None
                    current_spin_rows.clear()
                    state = "global"
                    continue
                else:
                    current_spin_rows.append(line)
                    continue
        
        if state in ("reading_data", "expect_header", "expect_skip"):
            commit_spin_block()
        return final_data

def write_output_by_kpoint(final_data, eigenval_energies, site_coords, output_file):
    out_lines = []
    out_lines.append("Site Coordinates:")
    for site in sorted(site_coords.keys()):
        coords = site_coords[site]
        out_lines.append(f"  Site: {site}\tCoordinates: {coords[0]}\t{coords[1]}\t{coords[2]}")
    out_lines.append("")
    try:
        sorted_kpts = sorted(final_data.keys(), key=lambda k: float(k))
    except Exception:
        sorted_kpts = sorted(final_data.keys())
    for kpt in sorted_kpts:
        out_lines.append(f"K-Point: {kpt}")
        for band in sorted(final_data[kpt].keys()):
            energy = eigenval_energies.get(band, None)
            if energy is not None:
                out_lines.append(f"  Band: {band}\tEnergy: {energy}")
            else:
                out_lines.append(f"  Band: {band}")
            for spin in ["spin1", "spin2"]:
                out_lines.append(f"    Spin: {spin}")
                rows_info = []
                for site in sorted(final_data[kpt][band].keys()):
                    if spin in final_data[kpt][band][site]:
                        block = final_data[kpt][band][site][spin]
                        rows_info.append((site, block["header"], block["orbitals"]))
                if not rows_info:
                    out_lines.append("      No data")
                    out_lines.append("")
                    continue
                union_orbs = []
                for (site, hdr, orb_dict) in rows_info:
                    for orb in hdr:
                        if orb not in union_orbs:
                            union_orbs.append(orb)
                header_line = "      Site\t" + "\t".join(union_orbs)
                out_lines.append(header_line)
                for (site, hdr, orb_dict) in rows_info:
                    row_cells = [str(site)]
                    for orb in union_orbs:
                        if orb in orb_dict:
                            v1, v2 = orb_dict[orb]
                            row_cells.append(f"{v1},{v2}")
                        else:
                            row_cells.append("")
                    out_lines.append("      " + "\t".join(row_cells))
                out_lines.append("")
            out_lines.append("")
        out_lines.append("")
    with open(output_file, "w") as f_out:
        f_out.write("\n".join(out_lines))

def find_eigenval_file(directory):
    potential_filenames = ["EIGENVAL.txt", "EIGENVAL"]
    for fname in potential_filenames:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No EIGENVAL file found in directory '{directory}'.")

def parse_eigenval(eigenval_path):
    r"""
    Parse an EIGENVAL file to extract the energy for each band.
    Assumes that the first 8 lines are header, then each subsequent line contains three numbers:
      - Band index
      - Energy
      - Occupation
    Returns a dictionary mapping band number (int) to energy (float).
    """
    energies = {}
    with open(eigenval_path, 'r') as f:
        lines = f.readlines()
    for ln in lines[8:]:
        ln = ln.strip()
        if ln == "":
            continue
        tokens = re.split(r"\s+", ln)
        if len(tokens) < 2:
            continue
        try:
            band = int(tokens[0])
            energy = float(tokens[1])
            energies[band] = energy
        except Exception:
            continue
    return energies

if __name__ == "__main__":
    projcar_directory = 'dir/run'
    eigenval_path = find_eigenval_file(projcar_directory)
    try:
        parser = ProjcarParser(projcar_directory)
        final_data = parser.parse()
        eigenval_energies = parse_eigenval(eigenval_path)
        site_coords = parser.site_coords
        if not final_data:
            print("No projection data was parsed. Please verify that the file format meets the expected structure.")
        else:
            output_filename = "final_output_by_kpoint.txt"
            output_path = os.path.join(projcar_directory, output_filename)
            write_output_by_kpoint(final_data, eigenval_energies, site_coords, output_path)
            print(f"Final output organized by k-point has been written to '{output_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")