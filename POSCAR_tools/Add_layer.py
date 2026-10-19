# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:22:28 2026

@author: Benjamin Kafin
"""

#!/usr/bin/env python3
"""
add_bottom_layer.py
====================

Adds one bulk-spaced Au(111) layer to the BOTTOM of a slab POSCAR and
rebalances the vacuum so the top of the molecule sits exactly at c/2.

Purpose: the existing slab layers are surface-relaxed (compressed spacing);
the new bottom layer replicates bulk material beneath them using the
Materials Project bulk Au (mp-81) interlayer spacing.

What it does
------------
1. Reads the input POSCAR (Selective Dynamics optional on input).
2. Clusters Au atoms into z-layers; identifies slab layers (modal atom
   count) and adatoms (small clusters above the slab).
3. Derives the surface primitive vectors from the bottom layer's in-plane
   nearest neighbors, measures the ABC stacking vector by reducing
   layer-1 positions modulo the primitive lattice, and snaps it to the
   ideal (1/3, 1/3) or (2/3, 2/3).
4. Builds the new bottom layer: bottom-layer xy positions shifted DOWN
   the stacking sequence (registry = ABC continuation), placed D_NEW
   below the current bottom layer.
5. Shifts all atoms up by D_NEW so the new layer sits at exactly z = 0.
6. Sets c_z = 2 * H, where H = z of the highest atom after the shift,
   so the vacuum above the molecule equals the slab+molecule height and
   the molecule top sits exactly at c/2.
7. Inserts the new Au atoms into the Au block immediately after the
   current bottom layer (preserving the file's top-to-bottom ordering).
8. Writes Selective Dynamics: bottom two layers (new + old bottom)
   F F F; every other atom (upper slab layers, adatoms, molecules)
   T T T.

Hard-fail gates (no fallbacks)
------------------------------
- a or b lattice vector has a z-component, or c is not purely along z
- Au species missing
- bottom slab layer not at z = 0 (within 1e-4 A)
- fewer than 2 slab layers
- ambiguous cluster size (neither modal slab size nor small adatom cluster)
- adatom cluster at or below the top slab layer
- bottom-layer NN analysis fails (six equal-length in-plane neighbors
  with a 60-degree pair not found)
- measured stacking vector does not snap to (1/3,1/3) or (2/3,2/3)
  within tolerance
- bottom-layer atoms not contiguous in the Au block (insertion ambiguous)

Constants
---------
MP_A   : Materials Project mp-81 conventional lattice constant (4.171 A)
D_NEW  : bulk Au(111) interlayer spacing = MP_A / sqrt(3) = 2.4081 A

parse_poscar / write_poscar / frac_to_cart / cart_to_frac lifted verbatim
from make_subsets.py (validated versions, Unix newline output).

Usage: edit the paths in the __main__ block and run.
"""

import os
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

MP_A = 4.171                       # Å, Materials Project mp-81 conventional lattice constant
D_NEW = MP_A / np.sqrt(3)          # Å, bulk Au(111) interlayer spacing (2.4081...)

LAYER_GAP = 0.5                    # Å, z-gap threshold separating layers
ADATOM_MAX = 5                     # clusters with <= this many atoms are adatoms
BOTTOM_Z_TOL = 1e-4                # Å, bottom layer must sit at z = 0 within this
NN_DIST_TOL = 0.01                 # relative tolerance for the six equal NN distances
NN_ANGLE_TOL = 5.0                 # deg, tolerance for the 60-degree NN pair
STACK_SNAP_TOL = 0.08              # primitive frac, max deviation from ideal stacking


# ═══════════════════════════════════════════════════════════════════
# POSCAR I/O   (lifted verbatim from make_subsets.py)
# ═══════════════════════════════════════════════════════════════════

def parse_poscar(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    comment = lines[0].rstrip("\n")
    scale = float(lines[1].strip())

    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    species_names = lines[5].split()
    species_counts = [int(x) for x in lines[6].split()]
    natoms = sum(species_counts)

    line_idx = 7
    selective_dynamics = False
    if lines[line_idx].strip()[0].upper() == "S":
        selective_dynamics = True
        line_idx += 1

    coord_type = lines[line_idx].strip()
    line_idx += 1

    frac_coords = np.zeros((natoms, 3))
    sd_flags = []
    for i in range(natoms):
        parts = lines[line_idx + i].split()
        frac_coords[i] = [float(parts[j]) for j in range(3)]
        if selective_dynamics and len(parts) >= 6:
            sd_flags.append(parts[3:6])

    tail_start = line_idx + natoms
    tail_lines = lines[tail_start:]

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "species_names": species_names,
        "species_counts": species_counts,
        "selective_dynamics": selective_dynamics,
        "coord_type": coord_type,
        "frac_coords": frac_coords,
        "sd_flags": sd_flags,
        "tail_lines": tail_lines,
    }


def write_poscar(filepath, data):
    with open(filepath, "w", newline="\r\n") as f:
        f.write(data["comment"] + "\n")
        f.write("1.0\n")
        for i in range(3):
            f.write('{:>20.10f}{:>21.10f}{:>21.10f}\n'.format(
                data["lattice"][i, 0], data["lattice"][i, 1], data["lattice"][i, 2]))
        for sp in data["species_names"]:
            f.write('{:>5}'.format(str(sp)))
        f.write("\n")
        for c in data["species_counts"]:
            f.write('{:>5}'.format(c))
        f.write("\n")
        if data["selective_dynamics"]:
            f.write("Selective Dynamics\n")
        f.write("Direct\n")
        for i in range(len(data["frac_coords"])):
            x, y, z = data["frac_coords"][i]
            f.write('{:>16.9f}{:>20.9f}{:>20.9f}'.format(x, y, z))
            if data["selective_dynamics"] and i < len(data["sd_flags"]):
                for flag in data["sd_flags"][i]:
                    f.write(' ' * 9)
                    f.write(flag)
            f.write("\n")
        for line in data["tail_lines"]:
            f.write(line)


def frac_to_cart(frac, lattice):
    return frac @ lattice


def cart_to_frac(cart, lattice):
    return cart @ np.linalg.inv(lattice)


# ═══════════════════════════════════════════════════════════════════
# Geometry analysis
# ═══════════════════════════════════════════════════════════════════

def cluster_au_layers(cart, au_indices):
    """Cluster Au atoms into z-layers separated by gaps > LAYER_GAP.

    Returns (slab_layers, adatom_indices):
      slab_layers   : list of lists of global atom indices, bottom-first,
                      all with the modal (largest) cluster size
      adatom_indices: flat list of global indices in small clusters
    Hard-fails on ambiguous cluster sizes or adatoms at/below the slab top.
    """
    zs = np.array([cart[i, 2] for i in au_indices])
    order = np.argsort(zs)
    clusters = []
    current = [au_indices[order[0]]]
    for k in order[1:]:
        i = au_indices[k]
        if cart[i, 2] - cart[current[-1], 2] < LAYER_GAP:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    sizes = [len(c) for c in clusters]
    modal = max(sizes)
    slab_layers = []
    adatoms = []
    for c in clusters:
        if len(c) == modal:
            slab_layers.append(c)
        elif len(c) <= ADATOM_MAX:
            adatoms.extend(c)
        else:
            raise RuntimeError(
                f"Ambiguous Au cluster of size {len(c)} (modal slab size is "
                f"{modal}, adatom max is {ADATOM_MAX}). Cluster z ≈ "
                f"{np.mean([cart[i,2] for i in c]):.3f} Å."
            )

    if len(slab_layers) < 2:
        raise RuntimeError(
            f"Need at least 2 slab layers to determine stacking; found "
            f"{len(slab_layers)}."
        )

    top_slab_z = max(cart[i, 2] for i in slab_layers[-1])
    for a in adatoms:
        if cart[a, 2] <= top_slab_z:
            raise RuntimeError(
                f"Adatom Au #{a+1} (z={cart[a,2]:.3f} Å) is at or below the "
                f"top slab layer (z={top_slab_z:.3f} Å)."
            )

    return slab_layers, adatoms


def surface_primitive_vectors(frac, lattice, layer_indices):
    """Derive the two surface primitive vectors (60-degree pair) from the
    in-plane nearest neighbors of the first atom in the layer."""
    f0 = frac[layer_indices]
    d = f0 - f0[0]
    d -= np.round(d)
    dc = (d @ lattice)[:, :2]
    dists = np.linalg.norm(dc, axis=1)
    order = np.argsort(dists)

    if len(order) < 7:
        raise RuntimeError("Layer too small for NN analysis (< 7 atoms).")
    nn = order[1:7]
    d_nn = dists[nn]
    if (d_nn.max() - d_nn.min()) / d_nn.min() > NN_DIST_TOL:
        raise RuntimeError(
            f"Six nearest in-plane neighbors are not equal-length: "
            f"{np.round(d_nn, 4)} Å."
        )
    if len(order) > 7 and dists[order[7]] < 1.3 * d_nn.max():
        raise RuntimeError(
            f"No clear gap after six nearest neighbors "
            f"(7th at {dists[order[7]]:.4f} Å vs NN {d_nn.max():.4f} Å)."
        )

    nn_vecs = dc[nn]
    v1 = nn_vecs[0]

    def angle(a, b):
        return np.degrees(np.arctan2(a[0]*b[1] - a[1]*b[0], np.dot(a, b)))

    v2 = None
    for v in nn_vecs[1:]:
        if abs(abs(angle(v1, v)) - 60.0) < NN_ANGLE_TOL:
            v2 = v
            break
    if v2 is None:
        raise RuntimeError(
            "No nearest neighbor at 60 degrees from the first NN vector; "
            "bottom layer is not a triangular lattice."
        )
    return v1, v2


def measure_stacking_shift(frac, lattice, layer_from, layer_to, M, Minv):
    """Mean position of layer_to relative to layer_from's sublattice,
    reduced modulo the primitive surface lattice. Handles hollow-site
    nearest-atom ties by construction. Returns primitive-frac 2-vector."""
    ref = frac[layer_from[0]]
    ts = []
    for i in layer_to:
        d = frac[i] - ref
        d -= np.round(d)
        dxy = (d @ lattice)[:2]
        t = dxy @ Minv
        t -= np.floor(t)
        ts.append(t)
    ts = np.array(ts)
    ts_adj = ts - np.round(ts - ts[0])
    return ts_adj.mean(axis=0) % 1.0


def snap_stacking(s_prim):
    """Snap measured primitive-frac stacking shift to (1/3,1/3) or
    (2/3,2/3). Hard-fail if neither is within STACK_SNAP_TOL."""
    candidates = [np.array([1/3, 1/3]), np.array([2/3, 2/3])]
    best, best_dev = None, None
    for c in candidates:
        d = s_prim - c
        d -= np.round(d)
        dev = np.linalg.norm(d)
        if best_dev is None or dev < best_dev:
            best, best_dev = c, dev
    if best_dev > STACK_SNAP_TOL:
        raise RuntimeError(
            f"Measured stacking shift {np.round(s_prim, 4)} (primitive frac) "
            f"does not snap to (1/3,1/3) or (2/3,2/3) within "
            f"{STACK_SNAP_TOL} (deviation {best_dev:.4f})."
        )
    return best, best_dev


# ═══════════════════════════════════════════════════════════════════
# Main operation
# ═══════════════════════════════════════════════════════════════════

def run(input_path, output_path):
    full = parse_poscar(input_path)
    lat = full["lattice"]
    frac = full["frac_coords"]
    species = full["species_names"]
    counts = full["species_counts"]

    # ----- Lattice orientation gates -----
    if abs(lat[0, 2]) > 1e-8 or abs(lat[1, 2]) > 1e-8:
        raise RuntimeError("a or b lattice vector has a z-component.")
    if abs(lat[2, 0]) > 1e-8 or abs(lat[2, 1]) > 1e-8:
        raise RuntimeError("c lattice vector is not purely along z.")

    if 'Au' not in species:
        raise RuntimeError("No Au species in POSCAR.")

    cart = frac_to_cart(frac, lat)

    # Au block range
    au_sp = species.index('Au')
    au_lo = sum(counts[:au_sp])
    au_hi = au_lo + counts[au_sp]
    au_indices = list(range(au_lo, au_hi))

    # ----- Layer clustering -----
    slab_layers, adatoms = cluster_au_layers(cart, au_indices)
    n_per_layer = len(slab_layers[0])

    bottom = slab_layers[0]
    bottom_zmax = max(abs(cart[i, 2]) for i in bottom)
    if bottom_zmax > BOTTOM_Z_TOL:
        raise RuntimeError(
            f"Bottom slab layer is not at z = 0 (max |z| = {bottom_zmax:.6f} Å)."
        )

    # ----- Bottom-layer contiguity in the Au block (insertion rule) -----
    b_sorted = sorted(bottom)
    if b_sorted[-1] - b_sorted[0] + 1 != len(b_sorted):
        raise RuntimeError(
            "Bottom-layer atoms are not contiguous in the Au block; "
            "insertion position is ambiguous."
        )
    insert_after = b_sorted[-1]          # global index of last bottom-layer atom

    # ----- Stacking vector -----
    v1, v2 = surface_primitive_vectors(frac, lat, bottom)
    M = np.array([v1, v2])
    Minv = np.linalg.inv(M)

    s_meas = measure_stacking_shift(frac, lat, bottom, slab_layers[1], M, Minv)
    s_prim, s_dev = snap_stacking(s_meas)
    s_up = s_prim @ M                     # Cartesian xy shift going UP one layer

    # ----- New bottom layer -----
    # Registry: ABC continuation downward = bottom-layer xy minus s_up.
    # Placed at z = 0 in the FINAL frame (old atoms shift up by D_NEW).
    new_xy = cart[bottom][:, :2] - s_up
    new_cart = np.column_stack([new_xy, np.zeros(n_per_layer)])

    # ----- Shift old atoms, compute new c -----
    cart_shifted = cart.copy()
    cart_shifted[:, 2] += D_NEW

    H = max(cart_shifted[:, 2].max(), new_cart[:, 2].max())
    c_new = 2.0 * H

    lat_new = lat.copy()
    lat_new[2, 2] = c_new

    # ----- Assemble atom list: insert new Au after old bottom layer -----
    all_cart = []
    all_flags = []
    frozen = set(bottom)                  # old bottom layer -> F F F

    for i in range(len(cart_shifted)):
        all_cart.append(cart_shifted[i])
        all_flags.append(['F', 'F', 'F'] if i in frozen else ['T', 'T', 'T'])
        if i == insert_after:
            for j in range(n_per_layer):
                all_cart.append(new_cart[j])
                all_flags.append(['F', 'F', 'F'])

    all_cart = np.array(all_cart)

    counts_new = list(counts)
    counts_new[au_sp] += n_per_layer

    # ----- Fractional coordinates in the new cell -----
    frac_new = cart_to_frac(all_cart, lat_new)
    # Wrap only the inserted atoms into [0,1) in xy (their shifted xy can
    # leave the cell); existing atoms keep their exact frac values.
    n_before_insert = insert_after + 1
    for j in range(n_before_insert, n_before_insert + n_per_layer):
        frac_new[j, 0] %= 1.0
        frac_new[j, 1] %= 1.0

    out = {
        "comment": "",
        "scale": 1.0,
        "lattice": lat_new,
        "species_names": species,
        "species_counts": counts_new,
        "selective_dynamics": True,
        "coord_type": "Direct",
        "frac_coords": frac_new,
        "sd_flags": all_flags,
        "tail_lines": [],
    }
    write_poscar(output_path, out)

    # ----- Summary -----
    n_total = len(all_cart)
    n_frozen = sum(1 for f in all_flags if f == ['F', 'F', 'F'])
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"\nSlab layers found: {len(slab_layers)} x {n_per_layer} atoms, "
          f"adatoms: {len(adatoms)}")
    print(f"Stacking shift measured {np.round(s_meas, 4)} -> snapped to "
          f"{np.round(s_prim, 4)} (deviation {s_dev:.4f} prim frac)")
    print(f"D_NEW = MP_A/sqrt(3) = {D_NEW:.4f} Å  (MP_A = {MP_A} Å)")
    print(f"\nLayer table (after):")
    print(f"  new bottom layer : z = 0.0000, {n_per_layer} atoms, F F F")
    for li, layer in enumerate(slab_layers):
        zm = np.mean([cart_shifted[i, 2] for i in layer])
        flag = 'F F F' if li == 0 else 'T T T'
        print(f"  old layer {li}      : z = {zm:.4f}, {len(layer)} atoms, {flag}")
    for a in adatoms:
        print(f"  adatom Au        : z = {cart_shifted[a, 2]:.4f}, T T T")
    print(f"\nHighest atom H = {H:.4f} Å")
    print(f"c_z: {lat[2,2]:.4f} -> {c_new:.4f}  (= 2H)")
    print(f"Vacuum above molecule = {c_new - H:.4f} Å (= H: "
          f"{'OK' if abs(c_new - 2*H) < 1e-10 else 'MISMATCH'})")
    print(f"\nAtoms: {sum(counts)} -> {n_total}   "
          f"(Au {counts[au_sp]} -> {counts_new[au_sp]})")
    print(f"Selective dynamics: {n_frozen} x FFF, {n_total - n_frozen} x TTT")
    print(f"NOTE: update MAGMOM in any INCAR for this system to "
          f"{n_total}-atom counts.")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    input_path  = 'dir/run/Au111_1x1.vasp'
    output_path = 'dir/output/Au111_1x1_new.vasp'

    run(input_path, output_path)