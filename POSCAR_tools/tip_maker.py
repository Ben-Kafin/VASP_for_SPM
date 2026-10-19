# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:15:11 2026

@author: Benjamin Kafin
"""

import numpy as np

# =============================================================================
# Crystalline pyramid tip generator
#
# Builds a single-metal, single-crystal tip as a POSCAR:
#   - N_BULK full-width bulk layers at the bottom (bottom atom at z = 0)
#   - M_PYRAMID layers above, carved by a PURE NEAREST-NEIGHBOR CASCADE that
#     grows downward from a single apex atom at the lateral cell center.
#   - The apex atom sits at fractional (0.5, 0.5, 0.5): lateral center AND
#     vertical center. The cell c-axis is therefore twice the physical height
#     from the bottom atom to the apex, leaving an equal vacuum region above
#     the apex (upward-facing tip centered in the cell).
#   - Lateral cell width is the smallest whole number of surface unit cells
#     that is at least SPACING x (footprint width of the widest carved
#     pyramid layer, i.e. the pyramid layer adjacent to the bulk).
#
# Crystal geometry (lattice vectors, interlayer spacing, stacking offsets) is
# derived analytically from a small set of inputs. Default is FCC(111) silver.
# =============================================================================

# ----------------------------- INPUT VARIABLES -------------------------------
N_BULK      = 5          # number of full-width bulk layers
M_PYRAMID   = 3          # number of cascade-carved pyramid layers
SPACING     = 2.5        # lateral gap factor: cell width >= (1 + 2*?)...
                         #   interpreted per spec below as multiplier on pyramid
                         #   footprint width (see WIDTH note). SPACING = 2 means
                         #   "fit two pyramid-widths of clearance".
NN_TOL      = 1.10       # a layer-below atom counts as a nearest neighbor if its
                         #   in-3D distance to a parent <= NN_TOL * (min such dist)

A_LATTICE   = 4.0857     # FCC conventional lattice constant (Angstrom). Silver.
ELEMENT     = "Ag"       # element symbol written to POSCAR

OUTPUT_FILE = "POSCAR_crystal_tip"
# -----------------------------------------------------------------------------


def fcc111_geometry(a):
    """
    Return the FCC(111) surface description derived from conventional constant a.

    Returns
    -------
    a1, a2 : (2,) in-plane primitive surface lattice vectors (Angstrom)
    dz     : interlayer spacing along the surface normal (Angstrom)
    shifts : list of 3 in-plane offset vectors (Angstrom) for ABC stacking,
             indexed by (layer_index % 3)
    """
    # In-plane nearest-neighbor distance in a close-packed (111) plane = a/sqrt(2).
    d_nn = a / np.sqrt(2.0)
    # Hexagonal surface lattice vectors of length d_nn, 60 degrees apart.
    a1 = np.array([d_nn, 0.0])
    a2 = np.array([d_nn * 0.5, d_nn * np.sqrt(3.0) / 2.0])
    # (111) interlayer spacing = a / sqrt(3).
    dz = a / np.sqrt(3.0)
    # ABC stacking: each successive layer is shifted by the fcc-hollow offset.
    # A at (0,0); B at (a1+a2)/3; C at 2*(a1+a2)/3.
    hollow = (a1 + a2) / 3.0
    shifts = [np.array([0.0, 0.0]), hollow.copy(), (2.0 * hollow).copy()]
    return a1, a2, dz, shifts


def build_lattice_atoms(a1, a2, dz, shifts, n_layers, n_rep):
    """
    Build a full-width crystal block: n_layers stacked (111) layers, each layer
    an n_rep x n_rep tiling of the surface cell. Layer 0 is the BOTTOM (z=0),
    layer index increases upward.

    Returns a list of layers; each layer is an (n_atoms, 3) array of Cartesian
    coordinates. Coordinates are NOT yet centered; that is done later.
    """
    layers = []
    for L in range(n_layers):
        shift = shifts[L % 3]
        pts = []
        for i in range(n_rep):
            for j in range(n_rep):
                xy = i * a1 + j * a2 + shift
                pts.append([xy[0], xy[1], L * dz])
        layers.append(np.array(pts))
    return layers


def cascade_keep(layers, apex_xy, top_layer_idx, m_pyramid, nn_tol):
    """
    Pure nearest-neighbor cascade.

    Starting from the apex atom (the atom in layer `top_layer_idx` closest to
    apex_xy), walk DOWNWARD through m_pyramid layers. The frontier at each step
    is the set of nearest-neighbor atoms (in the layer immediately below) of the
    current frontier atoms. Atoms are tracked by (layer, index) identity so an
    atom reached via multiple parents is kept exactly once.

    Returns
    -------
    kept : dict mapping layer_index -> set of atom indices kept in that layer
    apex_idx : index (within top layer) of the apex atom
    """
    # Identify the apex: atom in the top pyramid layer nearest to apex_xy.
    top = layers[top_layer_idx]
    apex_idx = int(np.argmin(np.linalg.norm(top[:, :2] - apex_xy, axis=1)))

    kept = {L: set() for L in range(top_layer_idx - m_pyramid + 1, top_layer_idx + 1)}
    kept[top_layer_idx].add(apex_idx)
    frontier = [(top_layer_idx, apex_idx)]

    # Walk downward m_pyramid - 1 steps (the top layer is layer 1 of the pyramid).
    for _ in range(m_pyramid - 1):
        cur_layer = frontier[0][0]
        below_layer = cur_layer - 1
        if below_layer < 0:
            break
        below = layers[below_layer]
        new_frontier_idx = set()
        for (Lp, ip) in frontier:
            parent = layers[Lp][ip]
            d = np.linalg.norm(below - parent, axis=1)   # full 3D distance
            dmin = d.min()
            near = np.where(d <= nn_tol * dmin)[0]
            for k in near:
                new_frontier_idx.add(int(k))
        kept[below_layer] |= new_frontier_idx
        frontier = [(below_layer, k) for k in sorted(new_frontier_idx)]

    return kept, apex_idx


def footprint_width(layer_pts, idx_set):
    """Max pairwise distance among the kept atoms of a layer (footprint span)."""
    if len(idx_set) <= 1:
        return 0.0
    pts = layer_pts[sorted(idx_set)][:, :2]
    # O(n^2) is fine: pyramid base layers are small.
    dmax = 0.0
    for p in range(len(pts)):
        diff = pts[p + 1:] - pts[p]
        if len(diff):
            dmax = max(dmax, np.linalg.norm(diff, axis=1).max())
    return dmax


def write_poscar(path, lattice3, frac, element, count):
    with open(path, "w", newline="\r\n") as f:
        f.write("Crystalline pyramid tip (nearest-neighbor cascade)\n")
        f.write("1.0\n")
        for v in lattice3:
            f.write('{:>20.10f}{:>21.10f}{:>21.10f}\n'.format(v[0], v[1], v[2]))
        f.write('{:>5}'.format(str(element)))
        f.write('\n')
        f.write('{:>5}'.format(count))
        f.write('\n')
        f.write("Direct\n")
        for r in frac:
            f.write('{:>16.9f}{:>20.9f}{:>20.9f}\n'.format(r[0], r[1], r[2]))


def main():
    a1, a2, dz, shifts = fcc111_geometry(A_LATTICE)

    total_layers = N_BULK + M_PYRAMID
    top_layer_idx = total_layers - 1

    # Provisional large tiling so the cascade has room to grow; trimmed later.
    # The pyramid can widen by ~1 nn-ring per layer, so 2*M plus a margin suffices.
    n_rep_build = 2 * M_PYRAMID + 4

    layers = build_lattice_atoms(a1, a2, dz, shifts, total_layers, n_rep_build)

    # Place the apex near the geometric middle of the provisional block so the
    # cascade is not clipped by the block edge.
    center_xy = (n_rep_build / 2.0) * (a1 + a2)

    kept, apex_idx = cascade_keep(layers, center_xy, top_layer_idx, M_PYRAMID, NN_TOL)

    apex_xy = layers[top_layer_idx][apex_idx][:2].copy()

    # Widest carved pyramid layer = the lowest pyramid layer (adjacent to bulk).
    lowest_pyr_layer = top_layer_idx - (M_PYRAMID - 1)
    base_width = footprint_width(layers[lowest_pyr_layer], kept[lowest_pyr_layer])

    # Required lateral cell edge: SPACING multiplied onto the base footprint,
    # plus one base width for the pyramid itself -> total >= (1 + SPACING)*base.
    # "fit a whole other pyramid on either side, SPACING widths away" => the
    # center-to-center image distance must be >= (1 + SPACING) * base_width.
    required_edge = (1.0 + SPACING) * base_width if base_width > 0 else np.linalg.norm(a1)

    # Round UP to a whole number of surface cells (|a1| == |a2|).
    surf_edge = np.linalg.norm(a1)
    n_rep = int(np.ceil(required_edge / surf_edge))
    if n_rep < 1:
        n_rep = 1

    # Rebuild the FINAL full-width block at n_rep so bulk layers fill the cell.
    final_layers = build_lattice_atoms(a1, a2, dz, shifts, total_layers, n_rep)

    # Re-find the apex in the final block (nearest atom in top layer to the
    # final block center), and re-run the cascade on the final block so the
    # kept indices refer to the final atom arrays.
    final_center_xy = (n_rep / 2.0) * (a1 + a2)
    kept, apex_idx = cascade_keep(final_layers, final_center_xy,
                                  top_layer_idx, M_PYRAMID, NN_TOL)
    apex_xy = final_layers[top_layer_idx][apex_idx][:2].copy()

    # ---- Assemble Cartesian atom list ----
    cart = []
    for L in range(total_layers):
        if L < N_BULK:
            cart.append(final_layers[L])                 # full bulk layer
        else:
            idxs = sorted(kept[L])
            if idxs:
                cart.append(final_layers[L][idxs])       # carved pyramid layer
    cart = np.vstack(cart)

    # ---- Center laterally so apex_xy -> cell center; shift z so bottom = 0 ----
    apex_z = final_layers[top_layer_idx][apex_idx][2]    # physical apex height
    # bottom atom is at z=0 already by construction (layer 0 at z=0).

    # Lateral lattice (3x3) with c chosen so apex is at fractional z = 0.5,
    # i.e. c = 2 * apex_z (vacuum above equals crystal height below).
    c_axis = 2.0 * apex_z
    A1 = np.array([a1[0] * n_rep, a1[1] * n_rep, 0.0])
    A2 = np.array([a2[0] * n_rep, a2[1] * n_rep, 0.0])
    A3 = np.array([0.0, 0.0, c_axis])
    lattice3 = np.array([A1, A2, A3])

    # Shift atoms laterally so the apex lands at the center of (A1, A2).
    lateral_center = 0.5 * (A1[:2] + A2[:2])
    cart[:, :2] += (lateral_center - apex_xy)

    # Fractional coordinates; wrap lateral into [0,1) (z left as-is, 0..0.5).
    Linv = np.linalg.inv(lattice3.T)
    frac = (Linv @ cart.T).T
    frac[:, 0] %= 1.0
    frac[:, 1] %= 1.0

    write_poscar(OUTPUT_FILE, lattice3, frac, ELEMENT, len(frac))

    # ---- Report ----
    print(f"Element            : {ELEMENT}  (a = {A_LATTICE} A, FCC(111))")
    print(f"Bulk layers        : {N_BULK}")
    print(f"Pyramid layers     : {M_PYRAMID}")
    print(f"Total layers        : {total_layers}")
    print(f"Atoms per bulk layer: {n_rep * n_rep}")
    print(f"Pyramid kept/layer : "
          + ", ".join(str(len(kept[L])) for L in range(top_layer_idx, N_BULK - 1, -1))
          + "  (top -> bottom)")
    print(f"Base footprint width: {base_width:.4f} A")
    print(f"Required edge       : {required_edge:.4f} A  -> n_rep = {n_rep} "
          f"(cell edge {n_rep * surf_edge:.4f} A)")
    print(f"c-axis (2*apex_z)  : {c_axis:.4f} A   apex_z = {apex_z:.4f} A")
    print(f"Total atoms        : {len(frac)}")
    print(f"Written            : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()