import numpy as np


class SupercellLayerRemover:
    """
    Build an (nx, ny, nz) supercell from a POSCAR with selective dynamics,
    preserving each atom's T/F flags through replication, then remove a list
    of atoms specified by (layer_index, x, y).

    Layers are identified automatically by the large empty z-gaps between them
    (the small intra-layer z variance is ignored). Layers are indexed top-down:
    the topmost layer is 1, the bottommost is n for n detected layers.

    Removal target (layer_index, x, y): within the named layer, the atom whose
    in-plane position is closest to Cartesian (x, y) is removed (plain Euclidean
    distance, no minimum-image convention).

    Internals are Cartesian, matching reshape_unitcell.py / SeldynAdder.py.
    """

    def __init__(self, input_file, output_file, supercell, remove_list):
        self.input_file = input_file
        self.output_file = output_file
        self.supercell = np.array(supercell, dtype=int)      # (nx, ny, nz)
        self.remove_list = remove_list                        # list of (layer_index, x, y) Cartesian
        self.lattice = None                                   # (3,3) Cartesian, scaled
        self.atom_types = None
        self.atom_numbers = None
        self.cart_coords = None                               # (N,3) Cartesian
        self.flags = None                                     # list of [str,str,str] per atom
        self.labels = None                                    # species string per atom

    def parse_poscar(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()

        sf = float(lines[1])
        self.lattice = np.array(
            [[float(x) for x in lines[i].split()] for i in range(2, 5)]
        ) * sf
        self.atom_types = lines[5].split()
        self.atom_numbers = [int(x) for x in lines[6].split()]
        N = sum(self.atom_numbers)

        # Detect optional Selective dynamics line, then the coordinate mode line.
        if lines[7].strip().lower().startswith('s'):
            mode_line = 8
            coord_start = 9
            has_seldyn = True
        else:
            mode_line = 7
            coord_start = 8
            has_seldyn = False

        mode = lines[mode_line].strip().lower()
        is_cartesian = mode.startswith('c') or mode.startswith('k')

        coords = []
        flags = []
        for i in range(coord_start, coord_start + N):
            parts = lines[i].split()
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_seldyn:
                flags.append([parts[3], parts[4], parts[5]])
            else:
                flags.append(['T', 'T', 'T'])
        coords = np.array(coords)

        if not is_cartesian:
            coords = coords @ self.lattice                    # fractional -> Cartesian

        self.cart_coords = coords
        self.flags = flags

    def build_supercell(self):
        nx, ny, nz = self.supercell
        a, b, c = self.lattice[0], self.lattice[1], self.lattice[2]

        # Cartesian offsets for every periodic image.
        offsets = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offsets.append(i * a + j * b + k * c)
        offsets = np.array(offsets)
        n_img = len(offsets)

        # Replicate per species so species stay grouped (valid POSCAR ordering).
        idx = 0
        new_coords_blocks = []
        new_flags = []
        for count in self.atom_numbers:
            cblock = self.cart_coords[idx:idx + count]
            fblock = self.flags[idx:idx + count]
            for off in offsets:
                new_coords_blocks.append(cblock + off)
                new_flags.extend([list(fr) for fr in fblock])
            idx += count

        self.cart_coords = np.vstack(new_coords_blocks)
        self.flags = new_flags
        self.atom_numbers = [count * n_img for count in self.atom_numbers]
        self.lattice = np.array([nx * a, ny * b, nz * c])

        # Species label per atom (for layer printout).
        self.labels = []
        for t, count in zip(self.atom_types, self.atom_numbers):
            self.labels.extend([t] * count)

    def detect_layers(self):
        """Return (layer_id per atom, n_layers). Layer 1 = top."""
        z = self.cart_coords[:, 2].copy()
        c = self.lattice[2, 2]
        N = len(z)

        # Unwrap across the vacuum gap so the slab is contiguous in z.
        # The vacuum is the largest gap in the periodic arrangement of z.
        order = np.argsort(z)
        zs = z[order]
        gaps = np.diff(zs)
        wrap_gap = (zs[0] + c) - zs[-1]
        all_gaps = np.append(gaps, wrap_gap)
        vac_idx = int(np.argmax(all_gaps))

        z_un = z.copy()
        if vac_idx != N - 1:
            # Vacuum sits between zs[vac_idx] and zs[vac_idx+1]; the atoms above it
            # are physically the bottom (wrapped) -> drop them by one cell length.
            seam = zs[vac_idx + 1]
            z_un[z >= seam] -= c
        # else: vacuum is the wrap-around gap; slab already contiguous, no shift.

        # Identify layers by the large empty gaps between them. The intra-layer
        # gaps are tiny and the inter-layer gaps are large; the split between the
        # two is the single largest jump in the sorted gap values.
        order2 = np.argsort(z_un)
        zs2 = z_un[order2]
        gaps2 = np.diff(zs2)
        sorted_gaps = np.sort(gaps2)
        threshold = sorted_gaps[int(np.argmax(np.diff(sorted_gaps)))]
        is_boundary = gaps2 > threshold

        # Number ascending from the bottom, then invert to top-down.
        lay_sorted = np.empty(N, dtype=int)
        lay_sorted[0] = 1
        cur = 1
        for i in range(1, N):
            if is_boundary[i - 1]:
                cur += 1
            lay_sorted[i] = cur
        n_layers = cur

        layer_id = np.empty(N, dtype=int)
        layer_id[order2] = lay_sorted
        layer_id = n_layers - layer_id + 1                    # top-down: 1 = top
        return layer_id, n_layers

    def print_layers(self, layer_id, n_layers):
        z = self.cart_coords[:, 2]
        print(f"{n_layers} layers detected (layer 1 = top, layer {n_layers} = bottom):")
        for L in range(1, n_layers + 1):
            mask = layer_id == L
            zs = z[mask]
            species = {}
            for lab in np.array(self.labels)[mask]:
                species[lab] = species.get(lab, 0) + 1
            breakdown = ' '.join(f"{k}:{v}" for k, v in species.items())
            print(f"  layer {L:>2}: z = [{zs.min():8.4f}, {zs.max():8.4f}] A   "
                  f"{int(mask.sum()):>3} atoms   ({breakdown})")

    def remove_atoms(self, layer_id):
        keep = np.ones(len(self.cart_coords), dtype=bool)
        xy = self.cart_coords[:, :2]
        for (L, x, y) in self.remove_list:
            in_layer = np.where(layer_id == L)[0]
            target = np.array([x, y])
            d = np.linalg.norm(xy[in_layer] - target, axis=1)
            chosen = in_layer[int(np.argmin(d))]
            keep[chosen] = False
            cc = self.cart_coords[chosen]
            print(f"  removing layer {L} atom {self.labels[chosen]} "
                  f"at Cartesian ({cc[0]:.4f}, {cc[1]:.4f}, {cc[2]:.4f})")
        return keep

    def write_poscar(self, keep):
        # Recount species from survivors (mask-and-recount, as in reshape_unitcell.py).
        new_types = []
        new_numbers = []
        idx = 0
        for t, count in zip(self.atom_types, self.atom_numbers):
            surv = int(keep[idx:idx + count].sum())
            if surv > 0:
                new_types.append(t)
                new_numbers.append(surv)
            idx += count

        coords = self.cart_coords[keep]
        flags = [self.flags[i] for i in range(len(keep)) if keep[i]]
        frac = coords @ np.linalg.inv(self.lattice)           # Cartesian -> fractional

        with open(self.output_file, 'w', newline='\r\n') as f:
            f.write("Generated supercell POSCAR\n")
            f.write("1.0\n")
            for vec in self.lattice:
                f.write('{:>20.10f}{:>21.10f}{:>21.10f}\n'.format(vec[0], vec[1], vec[2]))
            for t in new_types:
                f.write('{:>5}'.format(str(t)))
            f.write('\n')
            for n in new_numbers:
                f.write('{:>5}'.format(n))
            f.write('\n')
            f.write("Selective Dynamics\n")
            f.write("Direct\n")
            for fr, fl in zip(frac, flags):
                f.write('{:>16.9f}{:>20.9f}{:>20.9f}'.format(fr[0], fr[1], fr[2]))
                for j in range(3):
                    f.write(' ' * 9)
                    f.write(fl[j])
                f.write('\n')

    def run(self):
        self.parse_poscar()
        self.build_supercell()
        layer_id, n_layers = self.detect_layers()
        self.print_layers(layer_id, n_layers)
        keep = self.remove_atoms(layer_id)
        self.write_poscar(keep)
        print(f"Supercell POSCAR written to {self.output_file}")


if __name__ == "__main__":
    input_file = 'dir/run/POSCAR'
    output_file = 'dir/output/POSCAR_supercell.vasp'
    supercell = (2, 2, 1)

    # Each entry: (layer_index_from_top, x, y) in Cartesian Angstroms.
    remove_list = [
        (1, 12.188, 12.265),
    ]

    remover = SupercellLayerRemover(input_file, output_file, supercell, remove_list)
    remover.run()
