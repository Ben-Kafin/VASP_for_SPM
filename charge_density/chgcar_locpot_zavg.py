#!/usr/bin/env python3
"""
chgcar_zavg.py — Plot planar-averaged electron density along Z from VASP CHGCAR.

Reads CHGCAR (or AECCAR0/AECCAR2), averages ρ(x,y) at each z-slice,
and plots ρ_avg(z) in electrons/Å³ vs. z in Å. Optionally overlays the
planar-averaged total potential from LOCPOT on a secondary right-side
y-axis (in eV).

Spin handling (CHGCAR):
  - Non-spin-polarized: 1 data block  → total ρ
  - Collinear ISPIN=2:   2 data blocks → total ρ, spin density (ρ↑−ρ↓)
  - Non-collinear SOC:   4 data blocks → total ρ, m_x, m_y, m_z

LOCPOT is always plotted as the total scalar potential (first data block)
regardless of its spin content.

Edit the configuration block at the bottom of this file (under
`if __name__ == "__main__":`) to set the paths and options,
then run:  python chgcar_zavg.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# CHGCAR parser
# ---------------------------------------------------------------------------

def _parse_poscar_header(lines):
    """
    Parse the POSCAR-style header at the top of a CHGCAR file.

    Returns
    -------
    lattice         : (3, 3) array in Å (row vectors)
    species_list    : list of element symbols per atom, or None (VASP 4 format)
    positions_cart  : (natoms, 3) Cartesian coordinates in Å
    next_idx        : int, first line index after the atomic positions
    """
    scale = float(lines[1].strip())
    lattice = np.array(
        [[float(x) for x in lines[2 + i].split()] for i in range(3)]
    ) * scale

    # VASP 4: counts on line 5, no species names.
    # VASP 5: species on line 5, counts on line 6.
    tokens5 = lines[5].split()
    try:
        [int(t) for t in tokens5]
        species_names = None
        counts_line = 5
    except ValueError:
        species_names = tokens5
        counts_line = 6

    counts = [int(x) for x in lines[counts_line].split()]
    natoms = sum(counts)

    if species_names is None:
        species_list = None
    else:
        species_list = []
        for name, count in zip(species_names, counts):
            species_list.extend([name] * count)

    idx = counts_line + 1

    # Optional selective-dynamics line
    if lines[idx].strip() and lines[idx].strip()[0].lower() == "s":
        idx += 1

    coord_line = lines[idx].strip()
    coord_mode = coord_line[0].lower() if coord_line else "d"
    idx += 1

    positions = np.array(
        [[float(x) for x in lines[idx + i].split()[:3]] for i in range(natoms)]
    )
    idx += natoms

    if coord_mode in ("c", "k"):
        # Cartesian POSCAR coordinates are multiplied by the universal scale
        positions_cart = positions * scale
    else:
        # Direct (fractional) — default
        positions_cart = positions @ lattice

    return lattice, species_list, positions_cart, idx


def _find_next_block_header(lines, idx, ngx, ngy, ngz):
    """
    Starting at idx, scan forward through blank lines and augmentation data
    to locate the next grid-dimensions line matching (NGX, NGY, NGZ).

    Returns (data_start_idx, found) where data_start_idx is the first line
    after the grid header, and found is True if one was located.
    """
    while idx < len(lines):
        line = lines[idx].strip()
        if line == "":
            idx += 1
            continue
        if line.lower().startswith("augmentation"):
            idx += 1
            continue
        tokens = line.split()
        if len(tokens) == 3:
            try:
                g = [int(t) for t in tokens]
                if g == [ngx, ngy, ngz]:
                    return idx + 1, True
            except ValueError:
                pass
        # Otherwise it is augmentation-occupancy data — skip.
        idx += 1
    return idx, False


def _read_data_block(lines, idx, npts, ngx, ngy, ngz):
    """
    Read npts floats starting at line idx. Returns (array, next_idx).
    Stops early if a non-numeric line is encountered.
    """
    vals = []
    i = idx
    while len(vals) < npts and i < len(lines):
        line = lines[i].strip()
        if line == "":
            i += 1
            continue
        try:
            tokens = [float(x) for x in line.split()]
        except ValueError:
            break
        vals.extend(tokens)
        i += 1
    if len(vals) < npts:
        raise ValueError(f"Expected {npts} grid values, got {len(vals)}")
    arr = np.array(vals[:npts]).reshape((ngz, ngy, ngx))
    return arr, i


def parse_chgcar(filepath):
    """
    Parse a CHGCAR-format file (also valid for LOCPOT, which uses the same
    header + grid layout).

    Returns
    -------
    lattice        : (3, 3) lattice vectors in Å (row vectors)
    species_list   : list of element symbols per atom, or None
    positions_cart : (natoms, 3) Cartesian atomic positions in Å
    grid           : (NGX, NGY, NGZ)
    blocks         : list of (NGZ, NGY, NGX) arrays. For CHGCAR these store
                     Q(r) = ρ(r) × V_cell. For LOCPOT they store V(r) in eV.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    lattice, species_list, positions_cart, idx = _parse_poscar_header(lines)

    # Skip blank line(s) before the first grid-dimensions line
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    ngx, ngy, ngz = (int(x) for x in lines[idx].split())
    npts = ngx * ngy * ngz
    idx += 1

    blocks = []
    block, idx = _read_data_block(lines, idx, npts, ngx, ngy, ngz)
    blocks.append(block)

    # Up to 3 more blocks (collinear → 2 total, SOC → 4 total)
    while len(blocks) < 4:
        idx, found = _find_next_block_header(lines, idx, ngx, ngy, ngz)
        if not found:
            break
        try:
            block, idx = _read_data_block(lines, idx, npts, ngx, ngy, ngz)
            blocks.append(block)
        except ValueError:
            break

    return lattice, species_list, positions_cart, (ngx, ngy, ngz), blocks


def cell_volume(lattice):
    return abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))


def z_height(lattice):
    """Projection of c onto the normal of the ab-plane."""
    n = np.cross(lattice[0], lattice[1])
    n = n / np.linalg.norm(n)
    return abs(np.dot(lattice[2], n))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def planar_average_z(block, vol):
    """ρ_avg(z) in electrons/Å³ from a raw CHGCAR data block."""
    return block.mean(axis=(1, 2)) / vol


def planar_average_z_potential(block):
    """
    V_avg(z) in eV from a LOCPOT data block.

    Unlike CHGCAR (which stores ρ × V_cell), LOCPOT stores V(r) directly
    in eV, so no volume division is applied.
    """
    return block.mean(axis=(1, 2))


def macro_average(rho, dz, width):
    """
    Macroscopic average via periodic convolution with a rectangular window
    of the given width (in Å). Returns an array the same length as rho.
    """
    n_win = max(1, int(round(width / dz)))
    n = len(rho)
    pad = n_win
    padded = np.concatenate([rho[-pad:], rho, rho[:pad]])
    window = np.ones(n_win) / n_win
    smoothed = np.convolve(padded, window, mode="same")
    return smoothed[pad:pad + n]


def cumulative_electrons(rho, dz, A_xy):
    """∫₀^z ρ(z') A_xy dz'  — electrons contained below the plane at z."""
    return np.cumsum(rho) * dz * A_xy


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(chgcar_path, component="total", macro=None, cumulative=False,
        atoms=False, no_show=False, output=None, locpot_path=None):
    """
    Parse a CHGCAR and plot planar-averaged electron density along z.
    Optionally overlays the planar-averaged total potential from LOCPOT.

    Parameters
    ----------
    chgcar_path : str or Path
        Path to the CHGCAR file.
    component : str
        Which CHGCAR trace(s) to plot. One of:
          "total"  — total ρ (default)
          "spin"   — spin density (ρ↑−ρ↓) (collinear only)
          "up"     — ρ↑ (collinear only)
          "down"   — ρ↓ (collinear only)
          "mx"     — m_x (SOC only)
          "my"     — m_y (SOC only)
          "mz"     — m_z (SOC only)
          "all"    — overlay all available components
    macro : float or None
        If set, overlay the macroscopic average using a periodic convolution
        with a rectangular window of this width (in Å). Applied to both the
        CHGCAR trace(s) and (if present) the LOCPOT trace. None disables it.
    cumulative : bool
        If True, add a second panel showing cumulative electrons below z.
        Applies to CHGCAR only; LOCPOT is never integrated.
    atoms : bool
        If True, overlay atomic z-positions as colored tick marks.
    no_show : bool
        If True, save the figure to a PNG instead of showing interactively.
    output : str or None
        Output PNG filename (only used if no_show=True). Defaults to
        "chgcar_zavg.png".
    locpot_path : str or Path or None
        If provided, also plot the planar-averaged total potential from this
        LOCPOT file on a secondary right-side y-axis (in eV). Only the total
        scalar potential (first data block) is used, regardless of spin
        content. None disables the overlay.
    """
    valid_components = {"total", "spin", "up", "down", "mx", "my", "mz", "all"}
    if component not in valid_components:
        raise ValueError(
            f"component must be one of {sorted(valid_components)}, "
            f"got {component!r}"
        )

    filepath = Path(chgcar_path)
    if not filepath.exists():
        print(f"Error: {filepath} not found.")
        sys.exit(1)

    print(f"Reading {filepath} ...")
    lattice, species_list, positions_cart, (ngx, ngy, ngz), blocks = parse_chgcar(filepath)
    vol = cell_volume(lattice)
    Lz = z_height(lattice)
    A_xy = vol / Lz
    dz = Lz / ngz
    nblocks = len(blocks)

    print(f"  Lattice volume: {vol:.4f} Å³")
    print(f"  Cell height (z): {Lz:.4f} Å")
    print(f"  Grid: {ngx} × {ngy} × {ngz}")
    print(f"  Data blocks: {nblocks}")

    if nblocks == 1:
        spin_mode = "none"
    elif nblocks == 2:
        spin_mode = "collinear"
    elif nblocks == 4:
        spin_mode = "soc"
    else:
        spin_mode = "unknown"
    print(f"  Spin mode: {spin_mode}")

    z_ang = (np.arange(ngz) + 0.5) * dz

    rho_total = planar_average_z(blocks[0], vol)

    # Assemble traces to plot
    to_plot = {}

    if component == "total" or component == "all":
        to_plot["total ρ"] = rho_total

    if spin_mode == "collinear":
        rho_spin = planar_average_z(blocks[1], vol)
        if component in ("spin", "all"):
            to_plot["spin density (ρ↑−ρ↓)"] = rho_spin
        if component in ("up", "all"):
            to_plot["ρ↑"] = 0.5 * (rho_total + rho_spin)
        if component in ("down", "all"):
            to_plot["ρ↓"] = 0.5 * (rho_total - rho_spin)
    elif component in ("spin", "up", "down"):
        print(f"  Warning: requested '{component}' but calculation is not "
              f"collinear spin-polarized. Falling back to total.")
        to_plot.setdefault("total ρ", rho_total)

    if spin_mode == "soc":
        soc_labels = {1: "m_x", 2: "m_y", 3: "m_z"}
        soc_map = {"mx": 1, "my": 2, "mz": 3}
        if component in soc_map:
            idx_b = soc_map[component]
            to_plot[soc_labels[idx_b]] = planar_average_z(blocks[idx_b], vol)
        elif component == "all":
            for idx_b in range(1, 4):
                to_plot[soc_labels[idx_b]] = planar_average_z(blocks[idx_b], vol)
    elif component in ("mx", "my", "mz"):
        print(f"  Warning: requested '{component}' but spin mode is "
              f"'{spin_mode}', not SOC. Falling back to total.")
        to_plot.setdefault("total ρ", rho_total)

    if not to_plot:
        to_plot["total ρ"] = rho_total

    # Electron count sanity check
    total_e = blocks[0].sum() / (ngx * ngy * ngz)
    print(f"  Total electrons (grid integral): {total_e:.4f}")

    # Optional LOCPOT parsing
    V_avg = None
    l_z_ang = None
    l_dz = None
    if locpot_path is not None:
        locpot_fp = Path(locpot_path)
        if not locpot_fp.exists():
            print(f"Error: LOCPOT {locpot_fp} not found.")
            sys.exit(1)
        print(f"Reading LOCPOT {locpot_fp} ...")
        l_lattice, _, _, (lngx, lngy, lngz), lblocks = parse_chgcar(locpot_fp)
        l_Lz = z_height(l_lattice)
        l_dz = l_Lz / lngz
        l_z_ang = (np.arange(lngz) + 0.5) * l_dz
        V_avg = planar_average_z_potential(lblocks[0])
        print(f"  LOCPOT grid: {lngx} × {lngy} × {lngz}")
        print(f"  V range: [{V_avg.min():.3f}, {V_avg.max():.3f}] eV")

    # --- Plot ---
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "font.size": 11,
    })

    if cumulative:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = None

    for label, rho in to_plot.items():
        line, = ax.plot(z_ang, rho, label=label, linewidth=1.3)
        if macro is not None and macro > 0:
            rho_macro = macro_average(rho, dz, macro)
            ax.plot(z_ang, rho_macro, color=line.get_color(),
                    linewidth=1.8, linestyle="--", alpha=0.85,
                    label=f"{label}, macro avg ({macro:.2f} Å)")

    ax.set_ylabel("ρ (e⁻ / Å³)")
    ax.set_title(f"Planar-averaged electron density — {filepath.name}")
    ax.set_xlim(0, Lz)

    # Optional LOCPOT overlay on twin axis
    ax_right = None
    if V_avg is not None:
        ax_right = ax.twinx()
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(True)
        loc_color = "C3"  # red
        ax_right.plot(l_z_ang, V_avg, color=loc_color,
                      linewidth=1.3, label="V (total)")
        if macro is not None and macro > 0:
            V_macro = macro_average(V_avg, l_dz, macro)
            ax_right.plot(l_z_ang, V_macro, color=loc_color,
                          linewidth=1.8, linestyle="--", alpha=0.85,
                          label=f"V, macro avg ({macro:.2f} Å)")
        ax_right.set_ylabel("V (eV)", color=loc_color)
        ax_right.tick_params(axis="y", labelcolor=loc_color)
        ax_right.grid(False)  # avoid double grid overlapping the left-axis grid

    if atoms and species_list is not None:
        trans = ax.get_xaxis_transform()
        z_atoms = positions_cart[:, 2] % Lz
        unique_species = sorted(set(species_list))
        cmap = plt.get_cmap("tab10")
        species_colors = {sp: cmap(i % 10) for i, sp in enumerate(unique_species)}
        for sp in unique_species:
            zs = [z_atoms[i] for i, s in enumerate(species_list) if s == sp]
            ax.scatter(zs, [0.015] * len(zs), marker="|", s=260,
                       linewidths=1.8, color=species_colors[sp],
                       label=sp, zorder=5, transform=trans, clip_on=False)
    elif atoms and species_list is None:
        print("  Warning: atoms=True but POSCAR header has no "
              "species line (VASP 4 format).")

    # Legend: combine both axes when LOCPOT twin is active
    handles_left, labels_left = ax.get_legend_handles_labels()
    if ax_right is not None:
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        ax.legend(handles_left + handles_right,
                  labels_left + labels_right,
                  fontsize=9, loc="best", framealpha=0.9)
    else:
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    if ax2 is not None:
        cum = cumulative_electrons(rho_total, dz, A_xy)
        ax2.plot(z_ang, cum, color="C3", linewidth=1.3)
        ax2.set_xlabel("z (Å)")
        ax2.set_ylabel("electrons below z")
        ax2.set_xlim(0, Lz)
    else:
        ax.set_xlabel("z (Å)")

    fig.tight_layout()

    if no_show:
        outname = output or "chgcar_zavg.png"
        fig.savefig(outname, dpi=200)
        print(f"  Saved to {outname}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point — edit the configuration below, then run this file.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ====== Configure the run ======
    chgcar_path = 'dir/metal/'    # path to the CHGCAR file
    locpot_path = 'dir/metal/'        # path to LOCPOT file, or None to skip the overlay

    component  = "total"     # "total", "spin", "up", "down", "mx", "my", "mz", "all"
    macro      = None        # macroscopic-average window in Å, or None to disable
    cumulative = False       # add a second panel showing cumulative electrons below z
    atoms      = False       # overlay atomic z-positions as colored tick marks
    no_show    = False       # True → save PNG instead of showing interactively
    output     = None        # output filename (only used if no_show=True)
    # ================================

    run(
        chgcar_path,
        component=component,
        macro=macro,
        cumulative=cumulative,
        atoms=atoms,
        no_show=no_show,
        output=output,
        locpot_path=locpot_path,
    )
