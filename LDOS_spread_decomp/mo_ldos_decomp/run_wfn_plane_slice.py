# -*- coding: utf-8 -*-
"""
run_wfn_plane_slice.py — vertical plane slice through the MO-weighted
squared-wavefunction sum (or any CHGCAR-format file).

Standalone runner next to run_wfn_mo_sum.py, in the conventions of the
charge_density_methods_VASP slicers (lv_slice / path_slice) but for the
slice mode neither of them offers: not the curtain draped over atoms and
not the polyline path — ONE flat plane that contains the vertical
(Cartesian z) direction and whose horizontal trace is the best-fit line
through the given atoms' xy positions (exact line for two atoms; VESTA
1-based indices, path_slice convention). The plane spans the full cell
footprint along that line and the full cell height.

Mechanics mirror the existing slicers: fractional-coordinate
RegularGridInterpolator sampling with periodic wrapping (the grid is
wrap-padded so cell edges interpolate periodically instead of clipping
to zero), pcolormesh plot with an element-colored atom overlay within a
tolerance in Å of the plane. Every input grid is unit-normalized to
integrate to 1 over its cell (Å^-3), so several inputs — e.g. an
original MO's |psi|^2 next to the weighted wavefunction sum — compare
directly on one shared color scale: one input gives the familiar single
panel, more give side-by-side panels in one figure with a single norm
and colorbar. The plane is defined once, from the FIRST input's atoms;
each panel overlays its own file's atoms near that shared plane.
Because the data is a non-negative probability density the colormap is
sequential, under a "log" or "linear" norm — not the diverging
seismic/SymLog scheme the CHGCAR_diff slicers use. Independent of that
choice, a gamma correction in the STM simulator's ldos_gamma spirit is
applied to the norm's 0-1 OUTPUT before the color lookup (gamma 1.0 =
the plain log/linear plot, < 1 stretches weak features, > 1 compresses
them; the data and the colorbar's labels are never transformed), with
a live gamma slider + text box on every figure in the simulator's
style, except gamma may also go above 1 (slider to 2.0, text box
unlimited).

The volumetric files are streamed with run_wfn_mo_sum's chunked reader
(one full grid in memory per input — needed for interpolation — but
never a second copy from parsing).

Run with the config block below, optionally overridden on the command
line: integer-like tokens are atom indices, everything else is an input
file, and "none" stands for the derived default file (it may sit among
other files):

    python run_wfn_plane_slice.py [file ...] [atom1] [atom2] [...]
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from run_wfn_mo_sum import (
    read_chgcar_header,
    accumulate_grid,
    _offset_name_ascii,
)

# ======================================================================
# Config. run_dir/cfg None → reuse full_dir/cfg from run_stm_decomp.py.
# input_files None → the weighted-sum output for mo_component/mo_offset
# (plus band_list tag when set) inside run_dir, as the single input.
# ======================================================================
run_dir: Optional[str] = None
cfg = None

# One panel per entry, e.g. ["mo.vasp", "HOMO_sum.vasp"]; a bare string
# means one file. None → the derived weighted-sum default below.
input_files: Optional[List[str]] = ['dir/full/HOMO_wvfx_squared_probability.vasp',
                                    "dir/molecule_1/HOMO_sum.vasp"]
mo_component = "clean_kp551"
mo_offset = 0
band_list: Optional[str] = None    # None | "top5" | "greatest" | "dominant"

atom_indices: List[int] = [110, 111]   # 1-based VESTA indices defining the plane
atom_tol_ang = 3.0               # overlay: max distance from the plane (Å)

n_s: Optional[int] = None          # in-plane horizontal samples (None → grid)
n_z: Optional[int] = None          # vertical samples (None → grid NGZ)
norm_mode = "log"                  # "log" | "linear"
gamma = 1.0                        # exponent on the norm's 0-1 output,
                                   # applied before the color lookup and
                                   # independent of norm_mode: 1.0 leaves
                                   # the log/linear scaling untouched, < 1
                                   # stretches weak features, > 1 compresses
                                   # them. Live slider 0.1-2.0 + text box
                                   # (any positive value) on every figure.
cmap = "inferno"
invert_cmap = False                # True → reversed colormap (matplotlib's
                                   # "_r" twin, toggled either way, so
                                   # cmap="bone_r" + invert gives "bone")
save_png = True

ELEMENT_COLORS = {"Au": "orange", "N": "mediumaquamarine",
                  "C": "saddlebrown", "H": "yellow"}


# ======================================================================
# CHGCAR structure + grid loading (streamed once).
# ======================================================================
def parse_structure(header: bytes) -> Dict[str, Any]:
    """
    Lattice (scale applied, rows = lattice vectors, Å), per-atom element
    labels, and Cartesian atom positions (Å) from a CHGCAR header as
    returned by read_chgcar_header. Direct and Cartesian modes are both
    handled; the VASP4 no-species-line variant labels atoms X1, X2, ...
    """
    lines = header.decode("latin-1").splitlines()
    sf = float(lines[1].split()[0])
    lv = np.array([[float(t) for t in lines[2 + i].split()[:3]]
                   for i in range(3)], dtype=np.float64) * sf
    toks = lines[5].split()
    if all(t.isdigit() for t in toks):
        counts = [int(t) for t in toks]
        species = [f"X{i}" for i in range(len(counts))]
        i_mode = 6
    else:
        species = toks
        counts = [int(t) for t in lines[6].split()]
        i_mode = 7
    mode = lines[i_mode].strip()
    if mode[:1] in ("S", "s"):
        i_mode += 1
        mode = lines[i_mode].strip()
    natoms = sum(counts)
    coords = np.array([[float(t) for t in lines[i_mode + 1 + i].split()[:3]]
                       for i in range(natoms)], dtype=np.float64)
    if mode[:1] in ("D", "d"):
        cart = coords @ lv
    else:
        cart = coords * sf
    elems: List[str] = []
    for el, n in zip(species, counts):
        elems += [el] * n
    return dict(lattice=lv, elements=elems, cart=cart,
                volume=float(np.dot(np.cross(lv[0], lv[1]), lv[2])))


def load_wfx_grid(path: str, divide_by_volume: bool = True
                  ) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Stream a CHGCAR-format file into ONE (NGX, NGY, NGZ) array (x
    fastest — CHGCAR text order, reshape order='F'), divided by the cell
    volume when requested (lib.parse_CHGCAR convention).
    """
    with open(path, "rb") as f:
        header, _, dims = read_chgcar_header(f)
        vals = np.zeros(int(np.prod(dims)), dtype=np.float64)
        accumulate_grid(f, 1.0, vals, 5_000_000)
    struct = parse_structure(header)
    data = vals.reshape(dims, order="F")
    if divide_by_volume:
        data /= struct["volume"]
    return struct, data


def unit_normalize(data: np.ndarray, volume: float) -> np.ndarray:
    """
    Scale a grid so it integrates to 1 over the cell (sum * dV = 1 with
    dV = volume / N_gridpoints) — a unit probability density in Å^-3.
    Scale-invariant in the input's prefactor, so volume-divided and raw
    CHGCAR grids normalize to the same thing.
    """
    dV = volume / data.size
    total = float(data.sum()) * dV
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "grid integrates to a non-finite or non-positive value — "
            "cannot unit-normalize")
    return data / total


# ======================================================================
# Plane geometry + sampling.
# ======================================================================
def plane_from_atoms(cart: np.ndarray, indices: Sequence[int]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    (origin_xy, unit direction u_xy) of the vertical plane's horizontal
    trace: the best-fit line through the atoms' xy positions (leading
    SVD axis of the centered coordinates — exact for two atoms).
    indices are 1-based (VESTA).
    """
    if len(indices) < 2:
        raise ValueError("need at least two atom indices to define the plane")
    bad = [i for i in indices if not 1 <= i <= cart.shape[0]]
    if bad:
        raise IndexError(
            f"atom indices {bad} out of range 1..{cart.shape[0]}")
    xy = cart[[i - 1 for i in indices], :2]
    origin = xy.mean(axis=0)
    centered = xy - origin
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-8:
        raise ValueError(
            "the given atoms coincide in xy — the vertical plane's "
            "direction is undefined")
    u = vt[0] / np.linalg.norm(vt[0])
    # Canonical orientation: s increases from the first given atom
    # toward the last, so the plot never mirrors with the input order.
    if (xy[-1] - xy[0]) @ u < 0:
        u = -u
    return origin, u


def sample_vertical_plane(data: np.ndarray, lattice: np.ndarray,
                          origin_xy: np.ndarray, u_xy: np.ndarray,
                          n_s: Optional[int] = None,
                          n_z: Optional[int] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the plane spanned by (u_xy, ẑ) through origin_xy over the
    full cell footprint (s range = the cell corners projected onto the
    line) and the full cell height. Periodic: fractional coordinates are
    wrapped and the grid is wrap-padded, matching the %1.0 convention of
    the existing slicers without their zero-filled edges.

    Returns (S, Z, V): (n_s, n_z) meshes of in-plane distance, height,
    and interpolated values.
    """
    from scipy.interpolate import RegularGridInterpolator

    dims = data.shape
    corners_xy = np.array([np.zeros(3), lattice[0], lattice[1],
                           lattice[0] + lattice[1]])[:, :2]
    proj = (corners_xy - origin_xy) @ u_xy
    s_min, s_max = float(proj.min()), float(proj.max())
    corners_z = np.array(
        [(i * lattice[0] + j * lattice[1] + k * lattice[2])[2]
         for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    z_min, z_max = float(min(corners_z.min(), 0.0)), float(corners_z.max())

    if n_s is None:
        n_s = int(max(dims[0], dims[1]))
    if n_z is None:
        n_z = int(dims[2])
    s = np.linspace(s_min, s_max, n_s)
    z = np.linspace(z_min, z_max, n_z)
    S, Z = np.meshgrid(s, z, indexing="ij")

    R = np.empty(S.shape + (3,), dtype=np.float64)
    R[..., 0] = origin_xy[0] + S * u_xy[0]
    R[..., 1] = origin_xy[1] + S * u_xy[1]
    R[..., 2] = Z
    frac = (R.reshape(-1, 3) @ np.linalg.inv(lattice)) % 1.0

    # VASP grid values sit at fractional positions i/N; wrap-pad so the
    # [1-1/N, 1] interval interpolates periodically.
    padded = np.pad(data, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    axes = tuple(np.arange(n + 1) / n for n in dims)
    interp = RegularGridInterpolator(axes, padded)
    V = interp(frac).reshape(S.shape)
    return S, Z, V


def atoms_near_plane(struct: Dict[str, Any], origin_xy: np.ndarray,
                     u_xy: np.ndarray, tol_ang: float,
                     s_range: Tuple[float, float]
                     ) -> List[Dict[str, Any]]:
    """
    Atoms (periodic xy images included) within tol_ang of the plane,
    projected to (s, z) for the overlay — the analogue of the existing
    slicers' atoms_in_slab, with the distance measured perpendicular to
    this arbitrary vertical plane instead of along a lattice vector.
    """
    n_xy = np.array([-u_xy[1], u_xy[0]])
    out: List[Dict[str, Any]] = []
    lat_a, lat_b = struct["lattice"][0][:2], struct["lattice"][1][:2]
    # a/b may carry z-components (tilted cells): a translated image's
    # height shifts with them, not just its xy.
    z_a, z_b = float(struct["lattice"][0][2]), float(struct["lattice"][1][2])
    for idx, (elem, p) in enumerate(zip(struct["elements"], struct["cart"]),
                                    start=1):
        for ia in (-1, 0, 1):
            for ib in (-1, 0, 1):
                q = p[:2] + ia * lat_a + ib * lat_b
                d = float((q - origin_xy) @ n_xy)
                if abs(d) > tol_ang:
                    continue
                s_val = float((q - origin_xy) @ u_xy)
                if not s_range[0] <= s_val <= s_range[1]:
                    continue
                out.append(dict(index=idx, elem=elem, s=s_val,
                                z=float(p[2] + ia * z_a + ib * z_b), dist=d))
    return out


# ======================================================================
# Plot + orchestration.
# ======================================================================
def _resolve_cmap(name: str, invert: bool) -> str:
    """
    The colormap name with matplotlib's "_r" reversal toggled when
    invert is set — stripping the suffix if it is already there, so the
    flag always means "the other direction of whatever cmap says".
    """
    if not invert:
        return name
    return name[:-2] if name.endswith("_r") else name + "_r"


_GammaNormCls = None


def _gamma_norm(base, gamma):
    """
    Wrap a base norm (log or linear) with the gamma correction: the
    base maps data to its usual 0-1 range, then t -> t**gamma warps
    that normalized value before the colormap lookup. The data itself
    is never transformed and colorbar labels stay true data values —
    gamma only moves where they sit along the bar. The wrapper keeps
    .base and .gamma so the widgets can re-wrap the same base with a
    new gamma. (Class built on first use — matplotlib stays a
    function-local import, matching the rest of this module.)
    """
    global _GammaNormCls
    if _GammaNormCls is None:
        from matplotlib.colors import Normalize

        class _GammaNorm(Normalize):
            def __init__(self, base, gamma):
                super().__init__(vmin=base.vmin, vmax=base.vmax)
                self.base = base
                self.gamma = float(gamma)

            def __call__(self, value, clip=None):
                # Clip to 0-1 first: below-vmin values saturate at the
                # low end like a plain norm draw, instead of a
                # fractional power of a negative masking them blank.
                t = np.ma.clip(self.base(value), 0.0, 1.0)
                return np.ma.power(t, self.gamma)

            def inverse(self, value):
                return self.base.inverse(np.ma.power(value,
                                                     1.0 / self.gamma))

        _GammaNormCls = _GammaNorm
    return _GammaNormCls(base, gamma)


def _style_gamma_colorbar(cbar, norm):
    """
    Decade ticks for a gamma-wrapped log norm: the wrapper is a plain
    Normalize subclass, so the colorbar falls back to linearly spaced
    ticks (which crowd at the top of a log scale) unless told
    otherwise. Must be re-applied after every widget norm swap —
    set_norm resets the colorbar's locator. A linear base keeps the
    default locator.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    if isinstance(norm.base, LogNorm):
        cbar.locator = LogLocator(base=10.0)
        cbar.formatter = LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()


def _shared_norm(V_list, norm_mode=norm_mode, gamma=gamma):
    """
    One norm for every panel so intensities compare directly: vmax is
    the max over all panels; log vmin is the smallest positive value
    across panels, floored at vmax * 1e-8, with non-positives masked
    per panel. Either base norm is wrapped with the output-side gamma
    correction (_gamma_norm) — gamma 1.0 is the pure log/linear plot.
    Returns (norm, C_list).
    """
    from matplotlib.colors import LogNorm, Normalize

    vmax = max(float(np.nanmax(V)) for V in V_list)
    if norm_mode == "log":
        pos_mins = [float(V[V > 0].min()) for V in V_list if (V > 0).any()]
        if not pos_mins or vmax <= 0:
            raise ValueError("no positive values to plot on a log norm")
        vmin = max(min(pos_mins), vmax * 1e-8)
        return (_gamma_norm(LogNorm(vmin=vmin, vmax=vmax), gamma),
                [np.ma.masked_less_equal(V, 0.0) for V in V_list])
    if norm_mode == "linear":
        return _gamma_norm(Normalize(vmin=0.0, vmax=vmax), gamma), list(V_list)
    raise ValueError(f"unknown norm_mode {norm_mode!r}")


def plot_slices(panels, atoms, suptitle, norm_mode=norm_mode, cmap=cmap,
                gamma=gamma,
                value_label=r"unit-normalized $|\psi|^2$ / $\AA^{-3}$"):
    """
    panels: [(S, Z, V, panel_title), ...] — one pcolormesh per panel,
    all under one shared norm and one colorbar (taken from a real mesh
    so it carries the gamma-wrapped norm). atoms is either one
    flat list of atom dicts (the same overlay on every panel) or a
    per-panel list of lists (each panel overlays its own structure's
    atoms — the multi-input path, where inputs may hold different
    structures on the same plane). The legend collects the elements of
    every panel and sits on the first. Returns (fig, axes, pcms) as
    plain lists.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    norm, C_list = _shared_norm([p[2] for p in panels],
                                norm_mode=norm_mode, gamma=gamma)
    n = len(panels)
    per_panel = bool(atoms) and isinstance(atoms[0], (list, tuple))
    atoms_lists = list(atoms) if per_panel else [list(atoms)] * n
    if len(atoms_lists) != n:
        raise ValueError("per-panel atoms must match the number of panels")
    fig, axes = plt.subplots(1, n, sharey=True, constrained_layout=True,
                             figsize=(4.8 * n + 1.6, 4.8), squeeze=False)
    axes = list(axes[0])

    pcms = []
    handles, seen = [], set()
    for ax, (S, Z, _, title), C, p_atoms in zip(axes, panels, C_list,
                                                atoms_lists):
        pcm = ax.pcolormesh(S, Z, C, cmap=cmap, shading="nearest", norm=norm)
        pcms.append(pcm)
        for a in p_atoms:
            color = ELEMENT_COLORS.get(a["elem"], "grey")
            ax.scatter(a["s"], a["z"], c=color, marker="o", s=60,
                       edgecolor="k", zorder=5)
            if a["elem"] not in seen:
                handles.append(Patch(color=color, label=a["elem"]))
                seen.add(a["elem"])
        ax.set_aspect("equal")
        ax.set_xlabel(r"distance along plane / $\AA$")
        ax.set_title(title)
    if handles:
        # Explicit loc: "best" scans the full mesh and is slow on
        # production-size grids (450x693 per panel).
        axes[0].legend(handles=handles, title="Elements", loc="upper right")
    axes[0].set_ylabel(r"height z / $\AA$")

    cbar = fig.colorbar(pcms[0], ax=axes)
    cbar.set_label(value_label)
    _style_gamma_colorbar(cbar, norm)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, axes, pcms


def plot_slice(S, Z, V, atoms, title, norm_mode=norm_mode, cmap=cmap,
               gamma=gamma,
               value_label=r"unit-normalized $|\psi|^2$ / $\AA^{-3}$"):
    """Single-panel wrapper around plot_slices; returns (fig, ax, pcm)."""
    fig, axes, pcms = plot_slices([(S, Z, V, title)], atoms, "",
                                  norm_mode=norm_mode, cmap=cmap,
                                  gamma=gamma,
                                  value_label=value_label)
    return fig, axes[0], pcms[0]


def attach_gamma_widgets(fig, pcms, gamma_init=gamma):
    """
    Live gamma control in the STM simulator's bottom-row style: a
    slider with an editable text box beside it (the slider's own value
    text hidden, partial/invalid entries silently reverted). Unlike the
    simulator the gamma is NOT capped at 1: the slider spans 0.1-2.0
    and the text box accepts any positive value, even above the slider
    range (the slider display just pegs). Mode-independent: each change
    re-wraps the meshes' SAME base norm (log or linear) with the new
    gamma via _gamma_norm and re-colors the existing meshes — nothing
    is recomputed; the colorbar follows the shared norm automatically
    (log decade ticks re-applied after the swap). Returns
    (slider, textbox), also kept on fig._gamma_widgets.
    """
    from matplotlib.widgets import Slider, TextBox

    fmt = "{:.2f}"

    # Reserve a bottom strip for the widgets — constrained layout keeps
    # managing only the region above it.
    try:
        fig.get_layout_engine().set(rect=(0.0, 0.10, 1.0, 0.90))
    except Exception:
        pass
    ax_sld = fig.add_axes([0.32, 0.035, 0.30, 0.03])
    ax_tb = fig.add_axes([0.66, 0.035, 0.08, 0.03])
    for a in (ax_sld, ax_tb):
        a.set_in_layout(False)
        a.set_navigate(False)
    sld = Slider(ax_sld, "γ", 0.1, 2.0,
                 valinit=min(max(gamma_init, 0.1), 2.0), valstep=0.05)
    try:
        sld.valtext.set_visible(False)
    except Exception:
        pass
    tb = TextBox(ax_tb, "", initial=fmt.format(gamma_init))

    def _apply(g):
        norm = _gamma_norm(pcms[0].norm.base, g)
        for pcm in pcms:
            pcm.set_norm(norm)
        cbar = pcms[0].colorbar
        if cbar is not None:
            _style_gamma_colorbar(cbar, norm)
        fig.canvas.draw_idle()

    def _set_text(text):
        tb.eventson = False
        try:
            tb.set_val(text)
        finally:
            tb.eventson = True

    def _on_slider(val):
        _set_text(fmt.format(float(val)))
        _apply(float(val))

    def _on_submit(text):
        # Guarded like the simulator's textboxes: an exception here must
        # never leave the TextBox holding the canvas mouse grab.
        try:
            try:
                gamma = float((text or "").strip())
            except (TypeError, ValueError):
                gamma = float("nan")    # invalid entry: revert below
            if not np.isfinite(gamma) or gamma <= 0.0:
                _set_text(fmt.format(float(pcms[0].norm.gamma)))
                return
            sld.eventson = False
            try:
                sld.set_val(min(max(gamma, sld.valmin), sld.valmax))
            finally:
                sld.eventson = True
            _set_text(fmt.format(gamma))
            _apply(gamma)
        except Exception:
            pass
        finally:
            try:
                cv = tb.ax.figure.canvas
                if getattr(cv, "mouse_grabber", None) is tb.ax:
                    cv.release_mouse(tb.ax)
            except Exception:
                pass

    sld.on_changed(_on_slider)
    tb.on_submit(_on_submit)
    fig._gamma_widgets = (sld, tb)
    return sld, tb


def default_input_file(run_dir: str, component: str, offset: int,
                       band_list: Optional[str]) -> str:
    tag = f"_{band_list}" if band_list else ""
    return os.path.join(
        run_dir,
        f"{component}_{_offset_name_ascii(offset)}{tag}"
        "_wvfx_squared_probability.vasp")


def run(input_paths, indices: Sequence[int],
        tol_ang: float = atom_tol_ang,
        n_s: Optional[int] = n_s, n_z: Optional[int] = n_z,
        norm_mode: str = norm_mode, gamma: float = gamma,
        cmap: str = cmap, invert_cmap: bool = invert_cmap,
        save_png: bool = save_png):
    """
    Load (each grid unit-normalized) → plane from the FIRST input's
    atoms → sample every grid on that plane → one shared-norm figure.
    A str input returns (fig, ax, (S, Z, V)) as before; a sequence of
    paths returns (fig, axes, [(S, Z, V), ...]), one panel per file.
    """
    single = isinstance(input_paths, (str, os.PathLike))
    paths = ([os.fspath(input_paths)] if single
             else [os.fspath(p) for p in input_paths])
    if not paths:
        raise ValueError("need at least one input file")

    structs, grids = [], []
    for k, p in enumerate(paths, start=1):
        print(f"[WFNSLICE] input ({k}/{len(paths)}): {p}")
        struct, data = load_wfx_grid(p, divide_by_volume=False)
        print(f"[WFNSLICE] grid {data.shape}, {len(struct['elements'])} "
              f"atoms, volume {struct['volume']:.2f} A^3")
        integral = float(data.sum()) * struct["volume"] / data.size
        data = unit_normalize(data, struct["volume"])
        check = float(data.sum()) * struct["volume"] / data.size
        print(f"[WFNSLICE] unit-normalized: raw integral {integral:.6e} "
              f"-> {check:.6g}")
        if structs and not np.allclose(struct["lattice"],
                                       structs[0]["lattice"], atol=1e-6):
            print(f"[WFNSLICE] WARNING: lattice of {os.path.basename(p)} "
                  "differs from the first input — plane geometry uses the "
                  "first input")
        structs.append(struct)
        grids.append(data)

    origin_xy, u_xy = plane_from_atoms(structs[0]["cart"], indices)
    picked = ", ".join(
        f"{i}:{structs[0]['elements'][i - 1]}" for i in indices)
    print(f"[WFNSLICE] plane through atoms [{picked}] — origin "
          f"({origin_xy[0]:.3f}, {origin_xy[1]:.3f}) Å, direction "
          f"({u_xy[0]:+.4f}, {u_xy[1]:+.4f})")

    # Resolve the sample counts once, from the first grid (matching
    # sample_vertical_plane's own defaults), so every panel shares one
    # shape even when the inputs' grids differ.
    dims0 = grids[0].shape
    n_s_eff = n_s if n_s is not None else int(max(dims0[0], dims0[1]))
    n_z_eff = n_z if n_z is not None else int(dims0[2])
    panels = []
    panel_atoms = []
    for p, struct, grid in zip(paths, structs, grids):
        S, Z, V = sample_vertical_plane(grid, struct["lattice"], origin_xy,
                                        u_xy, n_s=n_s_eff, n_z=n_z_eff)
        panels.append((S, Z, V, os.path.basename(p)))
        # Each panel overlays ITS OWN structure's atoms near the shared
        # plane, so inputs from different systems mark their own atoms.
        panel_atoms.append(
            atoms_near_plane(struct, origin_xy, u_xy, tol_ang,
                             (float(S.min()), float(S.max()))))
    print(f"[WFNSLICE] sampled {n_s_eff}x{n_z_eff} points per panel; "
          f"atom marker(s) within {tol_ang:g} Å of the plane: "
          + ", ".join(str(len(a)) for a in panel_atoms))

    suptitle = f"vertical plane through atoms {list(indices)}"
    fig, axes, pcms = plot_slices(panels, panel_atoms, suptitle,
                                  norm_mode=norm_mode,
                                  cmap=_resolve_cmap(cmap, invert_cmap),
                                  gamma=gamma)
    if save_png:
        suffix = "_plane_slice.png" if single else "_plane_slice_compare.png"
        png = os.path.splitext(paths[0])[0] + suffix
        fig.savefig(png, dpi=300)
        print(f"[WFNSLICE] wrote {png}")
    # After the PNG save, so saved images stay widget-free.
    attach_gamma_widgets(fig, pcms, gamma)
    results = [(S, Z, V) for S, Z, V, _ in panels]
    if single:
        return fig, axes[0], results[0]
    return fig, axes, results


def _resolve_config():
    """
    Fill run_dir (and cfg, which rides along for parity with the sibling
    runners — nothing in this module consumes it) from run_stm_decomp.py
    when run_dir is left as None. A user-set run_dir skips the import
    entirely, keeping this runner decoupled from that script's config.
    """
    global run_dir, cfg
    if run_dir is None:
        import run_stm_decomp as _rc
        run_dir = _rc.full_dir
        if cfg is None:
            cfg = _rc.cfg
    return run_dir, cfg


def _parse_cli_args(argv: Sequence[str]
                    ) -> Tuple[List[Optional[str]], List[int]]:
    """
    Split raw CLI tokens: integer-like tokens are atom indices,
    "none"/"" is the derived-default placeholder (None), anything else
    is an input file path. (A file literally named like an integer is
    the one ambiguity — pass it as "./<name>".)
    """
    paths: List[Optional[str]] = []
    idx: List[int] = []
    for tok in argv:
        try:
            idx.append(int(tok))
            continue
        except ValueError:
            pass
        paths.append(None if tok.lower() in ("none", "") else tok)
    return paths, idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cli_paths, cli_idx = _parse_cli_args(sys.argv[1:])
    if not cli_paths:
        if isinstance(input_files, (str, os.PathLike)):
            cli_paths = [os.fspath(input_files)]   # bare string = one file
        elif input_files:
            cli_paths = list(input_files)
        else:
            cli_paths = [None]
    if any(p is None for p in cli_paths):
        rd, _ = _resolve_config()
        default = default_input_file(rd, mo_component, mo_offset, band_list)
        cli_paths = [default if p is None else p for p in cli_paths]
    indices = cli_idx if cli_idx else atom_indices
    run(cli_paths[0] if len(cli_paths) == 1 else cli_paths, indices)
    plt.show()
