# -*- coding: utf-8 -*-
"""
run_wfn_mo_sum.py — MO-weighted sum of vaspkit real-space SQUARED
wavefunctions, written as one CHGCAR-format probability map.

Standalone runner next to run_stm_decomp.py, operating on the same run
directory and the same extended matcher record:

  1. Resolve the requested MO as a Kramers entity exactly the way the
     decomposition windows do (load_static_mo_data → pair_entities +
     homo_entity_offsets), selected by component label + HOMO-relative
     offset (spread_collapse comp_indices convention: 0 = HOMO,
     -1 = HOMO-1, +1 = LUMO, ...). The entity's per-band weight
     w_n = |<MO|band n>|^2 is the sum of grid/{comp} rows over the
     entity's member MOs — probabilities straight from the matcher.
  2. Scan the run directory for vaspkit files named
     WFN_SOC_SQUARED_B####_K####_UP/DW.vasp (WFN_SQUARED_... and a DN
     suffix are accepted too). Only files at the matcher's k-point
     (meta/run_kwargs_json k_index, default 1) are used — the stored
     overlaps are valid at that k-point only. UP and DW are the two
     spinor components of one band: both stream with the band's single
     weight, so their sum reconstructs w_n * |psi_n|^2.
  3. Bands whose weight passes weight_threshold AND have files on disk
     form the worklist; everything else is reported (below threshold,
     above the matcher band window, or passing but missing from disk)
     before any volumetric data is touched.
  4. Stream each massive file exactly once: parse its header, then read
     the grid in fixed-size chunks and accumulate acc += w * chunk
     immediately. Peak memory is one accumulator grid plus one read
     chunk — never a second full grid, never all files at once.
  5. Write <comp>_<OFFSETNAME>_wvfx_squared_probability.vasp into the
     run directory: the first input file's header verbatim plus the
     accumulated grid — the output is the size of ONE input file.

There is deliberately NO amplitude output: the inputs are already
|psi|^2 densities, so the only physically meaningful combination is the
probability-weighted sum sum_n |<MO|n>|^2 * |psi_n|^2.

Run with the config block below, optionally overridden on the command
line (band_list: none | top5 | greatest | dominant — restricts the sum
to that run_mo_band_lists classification and tags the output filename;
with a band_list active the classification is energy-gated exactly like
run_mo_band_lists, default the LDOS erange of run_stm_decomp.py, "none"
as the single energy argument disabling the gate):

    python run_wfn_mo_sum.py [component] [offset] [threshold] [band_list] [emin emax | none]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import h5py

# ======================================================================
# Config. run_dir/cfg None → reuse full_dir/cfg from run_stm_decomp.py,
# so both runners always point at the same run and the same pairing
# threshold (soc_pair_thresh_eV drives the Kramers entities).
# ======================================================================
run_dir: Optional[str] = None
cfg = None

mo_component = "clean_kp551"          # molecule component label
mo_offset = 0                      # HOMO-relative Kramers-entity offset
weight_threshold = 5e-3            # min |<MO|band>|^2 for a band to count

# Optional band-list restriction (run_mo_band_lists classification, ranked
# within the MO's own component): None sums every discovered band that
# passes the threshold; "top5" / "greatest" / "dominant" recomputes that
# list from the matcher record and sums only bands on it.
band_list: Optional[str] = 'top5'

# Energy window for the band-list classification (Fermi-referenced eV,
# DOSCAR E_F = 0 — the LDOS erange convention), so the recomputed list
# matches what run_mo_band_lists wrote. None → reuse run_stm_decomp's
# erange when a band_list is active. Only consulted with band_list set.
energy_range: Optional[List[float]] = None

# Streaming granularity: grid values per read chunk (5e6 float64 ≈ 40 MB
# plus text buffers) and per formatted write block. Multiples of 5 keep
# the output's 5-per-line layout aligned with the write blocks.
read_chunk_values = 5_000_000
write_chunk_values = 1_000_000

VALUES_PER_LINE = 5
VALUE_FMT = "%18.11E"

# vaspkit real-space squared-wavefunction files. DN is normalized to DW.
_WFN_RE = re.compile(
    r"^WFN(?:_SOC)?_SQUARED_B(\d+)_K(\d+)_(UP|DW|DN)\.vasp$",
    re.IGNORECASE)


# ======================================================================
# MO entity + weights (no volumetric data touched here).
# ======================================================================
def _offset_name_ascii(offset: int) -> str:
    """decomp_window's offset naming with the U+2212 minus made ASCII,
    so HOMO-2/LUMO+1 are filesystem-plain."""
    from mo_ldos_projection.decomp_window import _offset_name
    return _offset_name(offset).replace("−", "-")


def resolve_entity_weights(run_dir: str, cfg, component: str,
                           offset: int) -> Dict[str, Any]:
    """
    Per-band probability weights for one MO Kramers entity, through the
    decomp windows' single load path (load_static_mo_data) so the entity
    set and weights can never diverge from the figures.

    Returns dict(weights (n_full,), ent, title, label, n_full,
    comp_rows — every entity row of the component, for band-list
    classification within the component — and energies, the E_sticks
    per-band energies (Fermi-referenced eV) for the energy gate).
    """
    from mo_ldos_projection.decomp_window import load_static_mo_data
    d = load_static_mo_data(run_dir, cfg)
    if not d["P"].is_soc:
        print("[WFNSUM] WARNING: the run is NOT flagged SOC (INCAR "
              "missing or LSORBIT/LNONCOLLINEAR not set) — entities are "
              "SINGLE MOs, not Kramers pairs. For WFN_SOC_SQUARED files "
              "this halves the entity: check the full-system INCAR.")
    offmap = d["entities_by_comp"].get(component)
    if offmap is None:
        raise KeyError(
            f"component {component!r} not in the matcher record; "
            f"available: {sorted(d['entities_by_comp'])}")
    if not offmap:
        raise KeyError(
            f"component {component!r} has no resolvable Kramers entities "
            "(no HOMO anchor and no occupied entity in the matcher "
            "record) — cannot select an offset")
    if offset not in offmap:
        lo, hi = min(offmap), max(offmap)
        raise KeyError(
            f"offset {offset} not available for {component!r}; "
            f"entity offsets span [{lo}, {hi}]")
    row = next(r for r in d["entity_rows"]
               if r["comp"] == component and r["offset"] == offset)
    return dict(
        weights=np.asarray(row["weights"], dtype=np.float64),
        ent=row["ent"],
        title=f"{component}_{_offset_name_ascii(offset)}",
        label=row["label"],
        n_full=int(row["weights"].shape[0]),
        comp_rows=[r for r in d["entity_rows"] if r["comp"] == component],
        energies=(np.asarray(d["E_sticks"], dtype=np.float64)
                  if d.get("E_sticks") is not None else None),
    )


def matcher_k_index(run_dir: str) -> int:
    """The k-point (1-based) the matcher record's overlaps were computed
    at — the only k whose wavefunction files these weights apply to."""
    path = os.path.join(run_dir, "matcher_results.h5")
    with h5py.File(path, "r") as hf:
        meta = hf.get("meta")
        if meta is not None and "run_kwargs_json" in meta:
            try:
                kwargs = json.loads(meta["run_kwargs_json"][()])
                return int(kwargs.get("k_index", 1))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
    return 1


# ======================================================================
# Discovery + worklist (still no volumetric data).
# ======================================================================
def discover_wfn_files(run_dir: str, k_expected: int
                       ) -> Tuple[Dict[int, Dict[str, str]], List[str]]:
    """
    Map band → {"UP"/"DW": path} for every squared-wavefunction file in
    run_dir at k_expected; files at other k-points are returned as the
    skipped list (their weights are unknown at the matcher's k-point).
    """
    by_band: Dict[int, Dict[str, str]] = {}
    skipped_k: List[str] = []
    for name in sorted(os.listdir(run_dir)):
        m = _WFN_RE.match(name)
        if not m:
            continue
        band, k, spinor = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        if spinor == "DN":
            spinor = "DW"
        if k != k_expected:
            skipped_k.append(name)
            continue
        by_band.setdefault(band, {})[spinor] = os.path.join(run_dir, name)
    return by_band, skipped_k


def build_worklist(by_band: Dict[int, Dict[str, str]],
                   weights: np.ndarray,
                   threshold: float,
                   allowed_bands: Optional[set] = None) -> Dict[str, Any]:
    """
    Split the discovered bands against the weight vector:
      included       [(band, w, {spinor: path})] — w ≥ threshold (and on
                     the band list when one is active)
      below          [(band, w)] — files on disk but w < threshold
      outside_list   [(band, w)] — files on disk, w ≥ threshold, but not
                     on the active band list
      above_window   [band] — files beyond the matcher band window
      missing        [(band, w)] — should be summed but no files on disk
    plus the weight totals that quantify coverage. allowed_bands=None
    means no band-list restriction.
    """
    n_full = int(weights.shape[0])
    included, below, above_window, outside_list = [], [], [], []
    for band in sorted(by_band):
        if band > n_full:
            above_window.append(band)
            continue
        w = float(weights[band - 1])
        if w < threshold:
            below.append((band, w))
        elif allowed_bands is not None and band not in allowed_bands:
            outside_list.append((band, w))
        else:
            included.append((band, w, by_band[band]))
    passing = np.nonzero(weights >= threshold)[0] + 1
    if allowed_bands is not None:
        passing = np.array(
            [b for b in passing if int(b) in allowed_bands], dtype=int)
    missing = [(int(b), float(weights[b - 1]))
               for b in passing if int(b) not in by_band]
    w_total = float(weights.sum())
    w_passing = float(weights[passing - 1].sum()) if passing.size else 0.0
    w_included = float(sum(w for _, w, _ in included))
    return dict(included=included, below=below, above_window=above_window,
                outside_list=outside_list, missing=missing, w_total=w_total,
                w_passing=w_passing, w_included=w_included)


# ======================================================================
# CHGCAR-format streaming.
# ======================================================================
def read_chgcar_header(f) -> Tuple[bytes, int, Tuple[int, int, int]]:
    """
    Read the CHGCAR header from a binary handle, leaving the position at
    the first grid value. Returns (header_bytes incl. the dims line,
    natoms, (ngx, ngy, ngz)). Handles the VASP4 no-species-line variant
    and vaspkit's spacer line that holds a single blank before the dims.
    """
    lines: List[bytes] = []

    def rl() -> bytes:
        line = f.readline()
        if not line:
            raise ValueError("unexpected end of file inside CHGCAR header")
        lines.append(line)
        return line

    rl()                      # comment
    rl()                      # universal scale factor
    for _ in range(3):        # lattice vectors
        rl()
    line = rl()               # species names (VASP5) or counts (VASP4)
    toks = line.split()
    if toks and all(t.isdigit() for t in toks):
        counts = [int(t) for t in toks]
    else:
        counts = [int(t) for t in rl().split()]
    natoms = sum(counts)
    mode = rl()               # Direct/Cartesian (Selective dynamics: skip)
    if mode.strip()[:1] in (b"S", b"s"):
        mode = rl()
    for _ in range(natoms):   # positions (trailing atom labels are fine)
        rl()
    while True:               # spacer line(s), then the grid dimensions
        toks = rl().split()
        if not toks:
            continue
        if len(toks) == 3:
            try:
                dims = tuple(int(t) for t in toks)
                break
            except ValueError:
                pass
        raise ValueError(
            f"expected the 3-integer grid-dims line, got {toks!r}")
    if any(d <= 0 for d in dims):
        raise ValueError(f"non-positive grid dimensions {dims}")
    return b"".join(lines), natoms, dims  # type: ignore[return-value]


def accumulate_grid(f, weight: float, acc: np.ndarray,
                    chunk_values: int) -> None:
    """
    Stream the grid block at f's current position into acc (in place):
    acc[i:j] += weight * values, chunk by chunk. Only one chunk of file
    data is ever alive; the file is read exactly once, forward-only.
    """
    npts = acc.size
    filled = 0
    while filled < npts:
        want = min(chunk_values, npts - filled)
        vals = np.fromfile(f, dtype=np.float64, count=want, sep=" ")
        if vals.size == 0:
            raise ValueError(
                f"grid truncated: read {filled} of {npts} values")
        acc[filled:filled + vals.size] += weight * vals
        filled += vals.size


def write_chgcar(path: str, header: bytes, acc: np.ndarray,
                 chunk_values: int = write_chunk_values) -> None:
    """
    Write header (verbatim, dims line included) + the accumulated grid,
    VALUES_PER_LINE per line, formatted in blocks so the full grid is
    never duplicated as one giant string.
    """
    if chunk_values % VALUES_PER_LINE:
        chunk_values -= chunk_values % VALUES_PER_LINE
    header_text = "\n".join(
        header.decode("latin-1").splitlines()) + "\n"
    n = acc.size
    n_body = n - (n % VALUES_PER_LINE)
    with open(path, "w", encoding="latin-1", newline="\n") as f:
        f.write(header_text)
        for start in range(0, n_body, chunk_values):
            block = acc[start:min(start + chunk_values, n_body)]
            np.savetxt(f, block.reshape(-1, VALUES_PER_LINE),
                       fmt=VALUE_FMT, delimiter=" ")
        if n_body < n:
            f.write(" ".join(VALUE_FMT % v for v in acc[n_body:]) + "\n")


# ======================================================================
# Orchestration.
# ======================================================================
def run(run_dir: str, cfg, component: str, offset: int,
        threshold: float,
        chunk_values: int = read_chunk_values,
        band_list: Optional[str] = None,
        energy_range: Optional[List[float]] = None) -> Optional[str]:
    """Full pipeline; returns the output path (None if nothing summed).
    energy_range (Fermi-referenced eV) gates the band_list
    classification exactly like run_mo_band_lists, so the two tools
    agree on the lists; it is ignored when band_list is None."""
    info = resolve_entity_weights(run_dir, cfg, component, offset)
    weights, ent, title = info["weights"], info["ent"], info["title"]
    k_expected = matcher_k_index(run_dir)
    print(f"[WFNSUM] MO entity: {info['label']}  "
          f"(members {ent.indices}, E_mean {ent.E_mean:+.4f} eV)")
    print(f"[WFNSUM] matcher k-point: {k_expected}  |  "
          f"band window: {info['n_full']} bands  |  "
          f"threshold |<MO|n>|^2 >= {threshold:g}")

    allowed: Optional[set] = None
    if band_list is not None:
        if band_list not in ("top5", "greatest", "dominant"):
            raise KeyError(
                f"band_list {band_list!r} unknown; choose top5, "
                "greatest, or dominant (or None)")
        from run_mo_band_lists import classify_bands
        energies = info.get("energies")
        if energy_range is not None:
            if energies is None:
                raise KeyError(
                    "energy_range given but the matcher record provides "
                    "no E_sticks band energies")
            lo, hi = sorted(float(x) for x in energy_range)
            print(f"[WFNSUM] energy gate: {lo:g} <= E - E_F(DOSCAR) <= "
                  f"{hi:g} eV (LDOS erange)")
        cls = classify_bands(info["comp_rows"], component, offset,
                             threshold=threshold, energies=energies,
                             energy_range=energy_range)
        allowed = set(cls[band_list])
        title = f"{title}_{band_list}"
        print(f"[WFNSUM] band list '{band_list}': {len(allowed)} band(s) "
              "qualify (ranking within the MO's own component).")

    by_band, skipped_k = discover_wfn_files(run_dir, k_expected)
    if skipped_k:
        print(f"[WFNSUM] ignoring {len(skipped_k)} file(s) at other "
              f"k-points (weights only exist at k={k_expected}).")
    if not by_band:
        print(f"[WFNSUM] no WFN*_SQUARED_B*_K{k_expected:04d}_* files in "
              f"{run_dir} — nothing to do.")
        return None

    wl = build_worklist(by_band, weights, threshold, allowed_bands=allowed)

    # ---- coverage report (before any volumetric data is read) ----
    for band in wl["above_window"]:
        print(f"[WFNSUM]   band {band}: file(s) beyond the matcher band "
              "window — no overlap recorded, skipped.")
    for band, w in wl["below"]:
        print(f"[WFNSUM]   band {band}: weight {w:.3e} < threshold, "
              "skipped.")
    for band, w in wl["outside_list"]:
        print(f"[WFNSUM]   band {band}: weight {w:.3e} but not on the "
              f"'{band_list}' band list, skipped.")
    if wl["missing"]:
        top = sorted(wl["missing"], key=lambda t: -t[1])
        shown = top[:30]
        print(f"[WFNSUM] {len(top)} band(s) pass the threshold but have "
              "NO files on disk (weight coverage is incomplete):")
        for band, w in shown:
            print(f"[WFNSUM]   band {band}: weight {w:.3e} — missing")
        if len(top) > len(shown):
            print(f"[WFNSUM]   ... and {len(top) - len(shown)} more "
                  "missing band(s).")
    if not wl["included"]:
        print("[WFNSUM] no discovered band passes the threshold — "
              "nothing to sum.")
        return None

    print(f"[WFNSUM] summing {len(wl['included'])} band(s):")
    for band, w, files in wl["included"]:
        have = "+".join(s for s in ("UP", "DW") if s in files)
        note = "" if len(files) == 2 else \
            "  [WARNING: spinor partner file missing]"
        print(f"[WFNSUM]   band {band}: weight {w:.6e}  ({have}){note}")
    print(f"[WFNSUM] weight coverage: included {wl['w_included']:.6e} of "
          f"{wl['w_passing']:.6e} above-threshold "
          f"(entity total {wl['w_total']:.6e}).")

    # ---- streaming accumulation: one file at a time, read once ----
    acc: Optional[np.ndarray] = None
    ref_header: Optional[bytes] = None
    ref_dims: Optional[Tuple[int, int, int]] = None
    ref_lattice: Optional[bytes] = None
    n_files = sum(len(files) for _, _, files in wl["included"])
    i_file = 0
    for band, w, files in wl["included"]:
        for spinor in ("UP", "DW"):
            path = files.get(spinor)
            if path is None:
                continue
            i_file += 1
            t0 = time.time()
            with open(path, "rb") as f:
                header, natoms, dims = read_chgcar_header(f)
                lattice = b"".join(header.splitlines(keepends=True)[2:5])
                if acc is None:
                    ref_header, ref_dims, ref_lattice = header, dims, lattice
                    acc = np.zeros(int(np.prod(dims)), dtype=np.float64)
                elif dims != ref_dims:
                    raise ValueError(
                        f"{os.path.basename(path)}: grid {dims} != "
                        f"reference {ref_dims} — files are not summable")
                elif lattice != ref_lattice:
                    print(f"[WFNSUM]   WARNING: {os.path.basename(path)} "
                          "lattice block differs from the reference "
                          "header (dims match; summing anyway).")
                accumulate_grid(f, w, acc, chunk_values)
            print(f"[WFNSUM]   [{i_file}/{n_files}] "
                  f"{os.path.basename(path)}  (w={w:.4e}, "
                  f"{time.time() - t0:.1f} s)")

    assert acc is not None and ref_header is not None
    out_path = os.path.join(run_dir, f"{title}_wvfx_squared_probability.vasp")
    print(f"[WFNSUM] writing {out_path} "
          f"({ref_dims[0]}x{ref_dims[1]}x{ref_dims[2]} grid) ...")
    write_chgcar(out_path, ref_header, acc)
    print(f"[WFNSUM] done: sum={acc.sum():.6e}, max={acc.max():.6e}, "
          f"min={acc.min():.6e} over {acc.size} points.")
    return out_path


def _resolve_config():
    """Fill run_dir/cfg/energy_range from run_stm_decomp.py when left as
    None, so the two runners share one directory + PlotConfig definition
    (energy_range ← the LDOS erange, as in run_mo_band_lists)."""
    global run_dir, cfg, energy_range
    if run_dir is None or cfg is None or energy_range is None:
        import run_stm_decomp as _rc
        if run_dir is None:
            run_dir = _rc.full_dir
        if cfg is None:
            cfg = _rc.cfg
        if energy_range is None:
            energy_range = list(_rc.erange)
    return run_dir, cfg, energy_range


if __name__ == "__main__":
    component = sys.argv[1] if len(sys.argv) > 1 else mo_component
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else mo_offset
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else weight_threshold
    blist = sys.argv[4] if len(sys.argv) > 4 else band_list
    if isinstance(blist, str) and blist.lower() in ("none", ""):
        blist = None
    rd, c, erange_ = _resolve_config()
    if len(sys.argv) == 6:
        if sys.argv[5].lower() in ("none", "nogate"):
            erange_ = None          # classify over the whole matcher window
        else:
            sys.exit(
                "usage: python run_wfn_mo_sum.py [component] [offset] "
                "[threshold] [band_list|none] [emin emax | none] — emin "
                "and emax must be given together ('none' disables the "
                "energy gate)")
    elif len(sys.argv) > 6:
        erange_ = [float(sys.argv[5]), float(sys.argv[6])]
    print(f"[WFNSUM] run dir: {rd}")
    run(rd, c, component, offset, threshold, band_list=blist,
        energy_range=erange_ if blist is not None else None)
