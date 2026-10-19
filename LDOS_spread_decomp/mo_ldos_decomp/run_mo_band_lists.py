# -*- coding: utf-8 -*-
"""
run_mo_band_lists.py — classify full-system bands by one MO's standing
among its own component's MOs, and emit the talapas vaspkit job that
generates the qualifying real-space squared wavefunctions.

Standalone runner next to run_stm_decomp.py / run_wfn_mo_sum.py, on the
same run directory and matcher record. For the input MO (component +
HOMO-relative Kramers offset) every band n in the matcher window is
classified by the component-internal ranking of w = |<MO|band n>|^2
(the ranking pool is ONLY the component's own Kramers entities — other
molecules and the metal never participate):

  top5      the MO ranks in the top five contributors of its component
  greatest  the MO is the single largest contributor (rank 1; ties tie)
  dominant  the MO's weight exceeds the SUM of all other entities'
            weights for that band

dominant ⊆ greatest ⊆ top5. Every list is additionally gated by
weight_threshold on the MO's own weight, so a band the MO technically
"wins" with negligible overlap never earns a 2 GB vaspkit run, and by
the LDOS energy window (energy_range, default the erange of
run_stm_decomp.py; Fermi-referenced eV) so out-of-window bands never
earn one either.

Outputs, written into the run directory:
  <comp>_<OFFSET>_band_lists.txt   all three lists in one file
  vaspkitjob.sh                    SLURM batch job (LF line endings)
                                   running vaspkit 51 → 515 → k → band
                                   sequentially over the top5 bands,
                                   skipping bands whose UP+DW files
                                   already exist

mo_component = None processes EVERY molecule component at the given
offset (each ranked within its own component, one lists file each) and
writes ONE union vaspkitjob.sh over all their top5 bands — each band's
files are generated once and serve every component's weighted sum.

Run with the config block below, optionally overridden on the command
line ("all" as component = every component; "none" as the single energy
argument disables the energy gate, i.e. the whole matcher window):

    python run_mo_band_lists.py [component|all] [offset] [threshold] [emin emax | none]
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from run_wfn_mo_sum import matcher_k_index, _offset_name_ascii

# ======================================================================
# Config. run_dir/cfg None → reuse full_dir/cfg from run_stm_decomp.py.
# ======================================================================
run_dir: Optional[str] = None
cfg = None

mo_component: Optional[str] = None  # component label; None → all components
mo_offset = 0                       # HOMO-relative Kramers-entity offset
weight_threshold = 5.0E-3             # gate on the MO's own |<MO|band>|^2
top_n = 80                           # "among the top five" pool depth

# Fermi-referenced eV (DOSCAR E_F = 0), same convention as the LDOS
# slider range. None → reuse erange from run_stm_decomp.py.
energy_range: Optional[List[float]] = None

JOB_NAME = "vaspkitjob.sh"


# ======================================================================
# Classification (pure record math — no files touched).
# ======================================================================
def classify_bands(entity_rows: List[Dict[str, Any]], component: str,
                   offset: int, threshold: float = weight_threshold,
                   top_n: int = top_n,
                   energies: Optional[np.ndarray] = None,
                   energy_range: Optional[List[float]] = None
                   ) -> Dict[str, Any]:
    """
    Rank one MO entity against the other entities of ITS OWN component
    for every band, and split the bands into the three lists.

    entity_rows: load_static_mo_data()["entity_rows"] (any superset —
    filtered to `component` here). Rank is competition style: 1 + the
    number of entities with strictly greater weight, so exact ties share
    the better rank.

    energies + energy_range (both or neither): per-band energies indexed
    like the weights (element i ↔ band i+1) and a window in the same
    frame; bands outside the window are dropped from every list. Ranks
    are per-band across entities, so the window never changes a rank.

    Returns dict(top5, greatest, dominant — sorted 1-based band lists —
    plus detail {band: (weight, rank, others_sum)} for reporting).
    """
    rows = [r for r in entity_rows if r["comp"] == component]
    if not rows:
        raise KeyError(
            f"component {component!r} has no entity rows; available: "
            f"{sorted(set(r['comp'] for r in entity_rows))}")
    target = next((r for r in rows if r["offset"] == offset), None)
    if target is None:
        offs = sorted(r["offset"] for r in rows)
        raise KeyError(
            f"offset {offset} not available for {component!r}; "
            f"entity offsets span [{offs[0]}, {offs[-1]}]")

    W = np.stack([np.asarray(r["weights"], dtype=np.float64)
                  for r in rows])                      # (n_ent, n_full)
    w = np.asarray(target["weights"], dtype=np.float64)  # (n_full,)
    others_sum = W.sum(axis=0) - w
    rank = 1 + (W > w[None, :]).sum(axis=0)            # target row never > w

    gate = w >= threshold
    if energies is not None and energy_range is not None:
        lo, hi = sorted(float(x) for x in energy_range)
        e = np.asarray(energies, dtype=np.float64)
        if e.shape != w.shape:
            raise ValueError(
                f"energies shape {e.shape} does not match the "
                f"{w.shape[0]}-band weight arrays")
        gate &= (e >= lo) & (e <= hi)
    top5_mask = gate & (rank <= top_n)
    greatest_mask = gate & (rank == 1)
    dominant_mask = gate & (w > others_sum)

    def bands(mask: np.ndarray) -> List[int]:
        return [int(b) for b in (np.nonzero(mask)[0] + 1)]

    detail = {int(b): (float(w[b - 1]), int(rank[b - 1]),
                       float(others_sum[b - 1]))
              for b in np.nonzero(top5_mask | dominant_mask)[0] + 1}
    return dict(top5=bands(top5_mask), greatest=bands(greatest_mask),
                dominant=bands(dominant_mask), detail=detail,
                label=target["label"], n_entities=len(rows))


# ======================================================================
# Outputs.
# ======================================================================
def write_band_lists(path: str, cls: Dict[str, Any], run_dir: str,
                     component: str, offset: int, k_index: int,
                     threshold: float, top_n: int = top_n,
                     energies: Optional[np.ndarray] = None,
                     energy_range: Optional[List[float]] = None) -> None:
    """One text file holding all three lists, LF-terminated, with the
    band/energy/weight/rank/others columns so the choice is auditable."""
    names = {
        "top5": f"MO among the top {top_n} MO contributors",
        "greatest": "MO is the single largest MO contributor",
        "dominant": "MO weight > sum of ALL other MO weights",
    }
    # utf-8: entity labels carry the U+2212 minus of the offset names.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# MO band lists - {cls['label']}\n")
        f.write(f"# run dir: {run_dir}\n")
        f.write(f"# matcher k-point: {k_index} (WFN files K{k_index:04d})\n")
        f.write(f"# ranking pool: the {cls['n_entities']} Kramers "
                f"entities of component '{component}' only "
                "(other molecules and metal excluded)\n")
        f.write(f"# gate: |<MO|band>|^2 >= {threshold:g} in every list\n")
        if energy_range is not None:
            lo, hi = sorted(float(x) for x in energy_range)
            f.write(f"# energy gate: {lo:g} <= E - E_F(DOSCAR) <= {hi:g} "
                    "eV (LDOS erange)\n")
        f.write("# columns: band  E-Ef_eV  weight  rank  sum_other_MOs\n")
        for key in ("top5", "greatest", "dominant"):
            bands = cls[key]
            f.write(f"\n[{key}]  # {names[key]} ({len(bands)} bands)\n")
            for b in bands:
                w, rank, others = cls["detail"][b]
                e_col = (f"{float(energies[b - 1]):+10.4f}"
                         if energies is not None else f"{'n/a':>10s}")
                f.write(f"{b:6d}  {e_col}  {w:.6e}  {rank:3d}  "
                        f"{others:.6e}\n")


def write_vaspkit_job(path: str, bands: List[int], k_index: int) -> None:
    """
    SLURM job generating each band's squared SOC wavefunction with
    vaspkit (interactive sequence 51 → 515 → k-point → band as a
    heredoc), sequentially, skipping bands whose UP+DW files already
    exist. LF line endings, enforced by a binary write plus read-back
    audit — written on Windows, submitted on talapas.
    """
    lines = [
        "#!/bin/bash",
        "#SBATCH --account=ACCOUNT",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=user@example.edu",
        "#SBATCH --job-name=vaspkit_wfn",
        "#SBATCH --partition=compute # 40 nodes with 128 cores and 512GB ea.",
        "#SBATCH --time=1-00:00:00",
        "#SBATCH --ntasks=1               # vaspkit is a serial post-processing tool",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=64G                # one SOC band grid in memory at a time",
        "#SBATCH -o output_vaspkit",
        "",
        "# vaspkit comes from ~/.bashrc (module loaded at login).",
        "# Submit from the directory holding the full-system WAVECAR:",
        "#     sbatch vaspkitjob.sh",
        'cd "${SLURM_SUBMIT_DIR}"',
        "",
        f"KPOINT={k_index}",
    ]
    band_words = [str(b) for b in bands]
    lines.append("BANDS=(")
    for i in range(0, len(band_words), 10):
        lines.append("    " + " ".join(band_words[i:i + 10]))
    lines.append(")")
    lines += [
        "",
        'for BAND in "${BANDS[@]}"; do',
        '    BPAD=$(printf "%04d" "${BAND}")',
        '    KPAD=$(printf "%04d" "${KPOINT}")',
        '    if [ -f "WFN_SOC_SQUARED_B${BPAD}_K${KPAD}_UP.vasp" ] && \\',
        '       [ -f "WFN_SOC_SQUARED_B${BPAD}_K${KPAD}_DW.vasp" ]; then',
        '        echo "[vaspkitjob] band ${BAND}: UP+DW already on disk, skipping"',
        "        continue",
        "    fi",
        '    echo "[vaspkitjob] generating band ${BAND} (k-point ${KPOINT}) ..."',
        "    vaspkit <<EOF",
        "51",
        "515",
        "${KPOINT}",
        "${BAND}",
        "EOF",
        "done",
        'echo "[vaspkitjob] all bands done"',
    ]
    # Raw binary write: no text-mode/encoding layer that could ever
    # inject \r. The read-back audit makes every generation prove it.
    blob = ("\n".join(lines) + "\n").encode("ascii")
    if b"\r" in blob:
        raise ValueError("vaspkit job content contains a CR byte")
    with open(path, "wb") as f:
        f.write(blob)
    with open(path, "rb") as f:
        back = f.read()
    if back != blob or b"\r" in back:
        raise ValueError(
            f"{path}: read-back does not match the LF-only ASCII content "
            "just written — refusing to hand over a corrupt job script")
    n_lf = back.count(b"\n")
    print(f"[BANDLISTS] {os.path.basename(path)}: {len(back)} bytes, "
          f"{n_lf} LF-terminated lines, 0 CR bytes — verified on write")


# ======================================================================
# Orchestration.
# ======================================================================
def run(run_dir: str, cfg, component: Optional[str], offset: int,
        threshold: float, top_n: int = top_n,
        energy_range: Optional[List[float]] = None) -> Dict[str, Any]:
    """Classify (one component, or every molecule component when
    component is None), write the lists file(s) and ONE union
    vaspkitjob.sh over all top5 bands. energy_range (Fermi-referenced
    eV, the LDOS erange) restricts every list to bands inside the
    window; None applies no energy gate. Returns the written paths."""
    from mo_ldos_projection.decomp_window import load_static_mo_data
    d = load_static_mo_data(run_dir, cfg)
    if not d["P"].is_soc:
        print("[BANDLISTS] WARNING: run not flagged SOC (INCAR missing "
              "or LSORBIT/LNONCOLLINEAR unset) — entities are single "
              "MOs, not Kramers pairs.")
    energies = d.get("E_sticks")
    if energy_range is not None:
        if energies is None:
            raise KeyError(
                "energy_range given but the matcher record provides no "
                "E_sticks band energies")
        lo, hi = sorted(float(x) for x in energy_range)
        print(f"[BANDLISTS] energy gate: {lo:g} <= E - E_F(DOSCAR) <= "
              f"{hi:g} eV (LDOS erange)")
    comps = [component] if component else list(d["mol_labels"])
    k_index = matcher_k_index(run_dir)

    list_paths: List[str] = []
    union: set = set()
    skipped: List[str] = []
    for comp in comps:
        # A component without a resolvable HOMO anchor (or without this
        # offset) must not abort the others in all-components mode: skip
        # it loudly and still deliver the union job for the rest.
        try:
            cls = classify_bands(d["entity_rows"], comp, offset,
                                 threshold=threshold, top_n=top_n,
                                 energies=energies,
                                 energy_range=energy_range)
        except KeyError as e:
            if component is not None:
                raise
            print(f"[BANDLISTS] SKIPPING component {comp!r}: {e}")
            skipped.append(comp)
            continue
        print(f"[BANDLISTS] {cls['label']}: "
              f"top{top_n} {len(cls['top5'])} band(s), "
              f"greatest {len(cls['greatest'])}, "
              f"dominant {len(cls['dominant'])}.")
        path = os.path.join(
            run_dir, f"{comp}_{_offset_name_ascii(offset)}_band_lists.txt")
        write_band_lists(path, cls, run_dir, comp, offset, k_index,
                         threshold, top_n, energies=energies,
                         energy_range=energy_range)
        print(f"[BANDLISTS] wrote {path}")
        list_paths.append(path)
        union.update(cls["top5"])
    if not list_paths:
        raise RuntimeError(
            f"no component produced a band list (skipped: {skipped}) — "
            "nothing to write a vaspkit job for")

    job_path = os.path.join(run_dir, JOB_NAME)
    job_bands = sorted(union)
    write_vaspkit_job(job_path, job_bands, k_index)
    print(f"[BANDLISTS] wrote {job_path}: {len(job_bands)} band(s) "
          f"({2 * len(job_bands)} UP/DW files, sequential vaspkit runs "
          f"at k-point {k_index}).")
    if skipped:
        print(f"[BANDLISTS] NOTE: {len(skipped)} component(s) skipped "
              f"({', '.join(skipped)}) — the union job covers only the "
              "components listed above.")
    return dict(list_paths=list_paths, job_path=job_path,
                job_bands=job_bands, skipped=skipped)


def _resolve_config():
    """Fill run_dir/cfg/energy_range from run_stm_decomp.py when left
    as None (energy_range ← the LDOS erange)."""
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
    if isinstance(component, str) and component.lower() in ("all", "none", ""):
        component = None
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else mo_offset
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else weight_threshold
    rd, c, erange_ = _resolve_config()
    if len(sys.argv) == 5:
        if sys.argv[4].lower() in ("none", "nogate"):
            erange_ = None          # whole matcher window, no energy gate
        else:
            sys.exit(
                "usage: python run_mo_band_lists.py [component|all] "
                "[offset] [threshold] [emin emax | none] — emin and emax "
                "must be given together ('none' disables the energy gate)")
    elif len(sys.argv) > 5:
        erange_ = [float(sys.argv[4]), float(sys.argv[5])]
    print(f"[BANDLISTS] run dir: {rd}")
    run(rd, c, component, offset, threshold, energy_range=erange_)
