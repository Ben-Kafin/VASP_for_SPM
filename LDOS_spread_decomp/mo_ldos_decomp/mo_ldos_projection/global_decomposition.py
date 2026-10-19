# -*- coding: utf-8 -*-
"""
Module A — global tri-partition decomposition figure.

Decomposes every full-system state's unit weight into three channels using
the exact-projector (generalized-Woodbury) variant of Option 4:

    w_MO_sub(j)  = Re[ c_MO(j)† · S_MO⁻¹ · c_MO(j) ]     MO-subspace weight
    w_met⊥(j)    = Re[ chi(j)† · S_mt⁻¹ · chi(j) ]       MO-orthogonalized
                                                          metal weight
    residual(j)  = 1 − w_MO_sub(j) − w_met⊥(j)

with chi(j) = c_met(j) − B·S_MO⁻¹·c_MO(j) and (approved amendment) the
EXACT block form S_mt = S_mm − B·S_MO⁻¹·B†, factored once (it is
j-independent). The injected metal states are not mutually orthonormal in
the full-cell S-metric (S_mm ≠ I), so the identity-Gram Woodbury form
S_mt⁻¹ = I + B·(S_MO − B†B)⁻¹·B† — to which this reduces when S_mm = I —
does not satisfy residual ≥ 0 on real data; the exact form does.
molecules + metal⊥ + residual = total holds exactly at every σ.

Displayed/exported per-MO quantities are always the raw grid values
p_a(j) = |⟨a|ψ_j⟩|² (Kramers-combined per the shared kramers utility) —
never rescaled by S_MO⁻¹.

Energy frame: meta/E_full − final_fermis["full"] (full-system Fermi at 0);
reference lines are the Fermi at 0 (dotted black) and v_aligned + shift
(solid black), spread_plot conventions.

Legacy mode: when grid_amp/cross are absent (an old matcher_results.h5) or
only the txt outputs exist, the figure shows the per-MO curves and the
total only, with a console note that metal⊥/residual require the extended
matcher run. The channels are never emulated with scalar arithmetic.

All broadening uses the one shared _gauss_sum implementation at the live σ
on a _curve_grid-style grid. Standalone-runnable via the stub-package
__main__ pattern at the bottom.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

if __name__ != "__main__":
    from ._imports import (
        RectAEPAWColorPlotter,
        PlotConfig,
        _gauss_sum,
        _curve_grid,
        pair_entities,
        homo_entity_offsets,
    )

# σ slider range / default (eV), VASP SIGMA convention.
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
SIGMA_DEFAULT = 0.2

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _offset_name(offset: int) -> str:
    """Display name of a HOMO-relative entity offset: 0 = HOMO, -1 = HOMO-1,
    +1 = LUMO, +2 = LUMO+1, ... (unicode minus, matching the spread codes)."""
    if offset == 0:
        return "HOMO"
    if offset < 0:
        return f"HOMO−{-offset}"
    if offset == 1:
        return "LUMO"
    return f"LUMO+{offset - 1}"


def _entity_label(offset: int, ent) -> str:
    """Label showing both forms, e.g. 'HOMO−1 (74+75)'."""
    if ent is None:
        return _offset_name(offset)
    return f"{_offset_name(offset)} ({'+'.join(str(i) for i in ent.indices)})"


# ======================================================================
# Full-precision data loading (Module-A math reads h5 datasets directly;
# the rounded reconstructions are only for the legacy txt path and colors)
# ======================================================================
def load_results(results_path: str) -> Optional[Dict[str, Any]]:
    """
    Read matcher_results.h5 at full precision. Returns None when the file
    is absent (txt-only legacy run). The returned dict always carries the
    grid weights and energy metadata; the amplitude/cross datasets are
    included only when the extended matcher run wrote them ("extended").
    """
    if not os.path.isfile(results_path):
        return None
    out: Dict[str, Any] = {}
    with h5py.File(results_path, "r") as hf:
        meta = hf["meta"]
        out["comp_labels"] = json.loads(meta["comp_labels_json"][()])
        out["E_full_raw"] = meta["E_full"][...].astype(np.float64)
        out["fermi_full"] = float(meta["final_fermis"]["full"][()])
        out["v_aligned"] = float(meta["v_aligned"][()])
        out["comp_energies"] = {
            l: meta["comp_energies"][l][...].astype(np.float64)
            for l in out["comp_labels"]
        }
        out["homo_indices"] = {
            k: int(meta["homo_indices"][k][()]) for k in meta["homo_indices"]
        }
        out["grid"] = {l: hf["grid"][l][...] for l in out["comp_labels"]}
        has_cross = ("grid_amp" in hf) and ("cross" in hf)
        out["extended"] = has_cross and ("S_mm" in hf["cross"])
        out["pre_amendment_cross"] = has_cross and not out["extended"]
        if out["extended"]:
            out["grid_amp"] = {l: hf["grid_amp"][l][...]
                               for l in out["comp_labels"]}
            out["B"] = hf["cross"]["B"][...]
            out["S_MO"] = hf["cross"]["S_MO"][...]
            out["S_mm"] = hf["cross"]["S_mm"][...]
            out["mo_axis"] = json.loads(hf["cross"]["mo_axis_json"][()])
            out["stored_diagnostics"] = {
                k: float(hf["cross"]["diagnostics"][k][()])
                for k in hf["cross"]["diagnostics"]
            }
    return out


def compute_decomposition(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-state tri-partition weights from the extended datasets (general,
    non-orthonormal form for BOTH the MO set and the metal set; reduces to
    the spec's simple forms when S_MO = I and S_mm = I). Factors S_MO and
    the exact S_mt = S_mm − B·S_MO⁻¹·B† once each (both j-independent).

    Returns dict with w_mo_sub, w_met_perp, residual (clamped),
    clamp_count, mo_sum_gap_max, and the stored diagnostics.
    """
    from scipy.linalg import cho_factor, cho_solve

    mo_axis = data["mo_axis"]
    # Stack the molecule amplitude rows in mo_axis order (molecule
    # components in comp_labels order, bands ascending within each — the
    # order grid_amp rows already have, asserted here).
    mol_order: List[str] = []
    for label, _idx in mo_axis:
        if not mol_order or mol_order[-1] != label:
            mol_order.append(label)
    blocks = [data["grid_amp"][label] for label in mol_order]
    c_mo = np.vstack(blocks)
    if c_mo.shape[0] != len(mo_axis):
        raise ValueError(
            f"mo_axis length {len(mo_axis)} does not match stacked molecule "
            f"amplitude rows {c_mo.shape[0]} — matcher output inconsistent.")
    c_met = data["grid_amp"]["metal"]
    B = data["B"]
    S_MO = data["S_MO"]
    n_mo = S_MO.shape[0]
    if B.shape != (c_met.shape[0], n_mo):
        raise ValueError(
            f"cross/B shape {B.shape} inconsistent with metal rows "
            f"{c_met.shape[0]} and n_MO {n_mo}.")

    # w_MO_sub(j) = Re[c_MO(j)† S_MO⁻¹ c_MO(j)], S_MO factored once.
    S_MO_h = 0.5 * (S_MO + S_MO.conj().T)
    cf_smo = cho_factor(S_MO_h)
    X = cho_solve(cf_smo, c_mo)                       # S_MO⁻¹ · c_MO
    w_mo_sub = np.einsum("aj,aj->j", c_mo.conj(), X).real

    # chi(j) = c_met(j) − B · S_MO⁻¹ · c_MO(j)
    chi = c_met - B @ X

    # Exact block form (approved amendment): S_mt = S_mm − B·S_MO⁻¹·B†,
    # factored once. S_mt is positive definite exactly when min_eig_S_mt > 0
    # (the matcher's linear-dependence guard); with S_mm = I this reduces to
    # the spec's Woodbury form S_mt⁻¹ = I + B·(S_MO − B†B)⁻¹·B†.
    S_mm = data["S_mm"]
    S_mt = 0.5 * (S_mm + S_mm.conj().T) - B @ cho_solve(cf_smo, B.conj().T)
    S_mt = 0.5 * (S_mt + S_mt.conj().T)
    try:
        cf_mt = cho_factor(S_mt)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            "S_mt = S_mm − B·S_MO⁻¹·B† is not positive definite "
            "(min_eig_S_mt ≤ 0): the metal window is linearly dependent on "
            "the MO set. The inputs are inconsistent — inspect "
            "cross/diagnostics/min_eig_S_mt."
        ) from e
    w_met_perp = np.einsum(
        "mj,mj->j", chi.conj(), cho_solve(cf_mt, chi)).real

    residual = 1.0 - w_mo_sub - w_met_perp
    min_res = float(residual.min()) if residual.size else 0.0
    if min_res < -1e-10:
        raise RuntimeError(
            f"residual(j) ≥ −1e−10 violated: min residual {min_res:.3e}. "
            "The decomposition identities do not hold on this input — "
            "inputs are inconsistent, not the constraint.")
    clamp_mask = residual < 0.0
    clamp_count = int(np.count_nonzero(clamp_mask))
    residual = np.where(clamp_mask, 0.0, residual)

    # Diagnostic gap between the raw per-MO weights (displayed/exported)
    # and the S_MO-metric subspace weight: max over j of Σ_a p_a − w_MO_sub.
    sum_p = np.einsum("aj,aj->j", c_mo.conj(), c_mo).real
    mo_sum_gap_max = float(np.max(np.abs(sum_p - w_mo_sub))) if sum_p.size else 0.0

    return dict(
        w_mo_sub=w_mo_sub,
        w_met_perp=w_met_perp,
        residual=residual,
        clamp_count=clamp_count,
        mo_sum_gap_max=mo_sum_gap_max,
        stored_diagnostics=data.get("stored_diagnostics", {}),
    )


# ======================================================================
# Figure
# ======================================================================
def plot_global_decomposition(
    output_dir: str,
    mo_curves: Optional[Dict[str, List[int]]] = None,
    cfg: Optional[PlotConfig] = None,
    sigma: float = SIGMA_DEFAULT,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Build the Module-A figure for one matcher run directory.

    Parameters
    ----------
    output_dir : str
        The matcher run directory (full_dir) holding matcher_results.h5
        and/or the band_matches_rectangular*.txt outputs.
    mo_curves : {comp_label: [homo_offsets]} or None
        Per-MO curves to draw: combined-entity HOMO-relative offsets per
        molecule component (0 = HOMO entity, −1 = HOMO−1, +1 = LUMO).
        Curves are UNNORMALIZED (area = combined captured weight), colored
        with the _prepare colors, labeled 'HOMO−1 (74+75)' style.
    cfg : PlotConfig or None
        Colors / thresholds / reference-line configuration.
    sigma : float
        Initial smearing width in eV (VASP SIGMA convention).

    Returns
    -------
    fig, state
        The figure and the live state dict (curves, widgets, export path).
    """
    cfg = cfg if cfg is not None else PlotConfig()
    mo_curves = mo_curves or {}

    # Colors / Fermis / HOMOs / is_soc / txt fallback via _prepare.
    plotter = RectAEPAWColorPlotter(cfg)
    main_path = os.path.join(output_dir, "band_matches_rectangular.txt")
    P = plotter._prepare(main_path, bonding=False, write_summaries=False)

    results_path = os.path.join(output_dir, "matcher_results.h5")
    data = load_results(results_path)

    # ------------------------------------------------------------------
    # Energy axis and per-state weights (full precision when h5 present).
    # ------------------------------------------------------------------
    if data is not None:
        fermi_full = data["fermi_full"]
        E = data["E_full_raw"] - fermi_full           # Fermi at 0
        v_shifted = data["v_aligned"] - fermi_full
        extended = data["extended"]
    else:
        # txt-only legacy: rounded energies via _prepare's rows.
        fermi_full = P.known_fermis.get("full", 0.0)
        E = np.array([float(r["E_full"]) for r in P.rows]) - fermi_full
        v_shifted = (P.v_aligned - fermi_full) if P.v_aligned is not None else None
        extended = False

    decomp = None
    if extended:
        decomp = compute_decomposition(data)
        print("[MODULE A] Decomposition diagnostics: "
              f"S_MO_dev_max={decomp['stored_diagnostics'].get('S_MO_dev_max', float('nan')):.3e}; "
              f"S_mm_dev_max={decomp['stored_diagnostics'].get('S_mm_dev_max', float('nan')):.3e}; "
              f"min_eig_S_mt={decomp['stored_diagnostics'].get('min_eig_S_mt', float('nan')):.6f}; "
              f"residual clamps={decomp['clamp_count']}; "
              f"MO-sum gap (max over j)={decomp['mo_sum_gap_max']:.3e}")
    elif data is not None and data.get("pre_amendment_cross"):
        print("[MODULE A] cross/ present but cross/S_mm missing (record "
              "predates the exact-metal-Gram amendment): showing MO curves "
              "and the total only — rerun the extended matcher to add "
              "S_mm and enable the metal⊥ and residual channels.")
    else:
        print("[MODULE A] grid_amp/cross not present (legacy run record): "
              "showing MO curves and the total only — the metal⊥ and "
              "residual channels require the extended matcher rerun.")

    # ------------------------------------------------------------------
    # Kramers entities per molecule component (colors path: comp_pairs,
    # known_homos, is_soc from _prepare; one pairing rule everywhere).
    # ------------------------------------------------------------------
    mo_entries: List[Dict[str, Any]] = []
    for label, offsets in mo_curves.items():
        ents = pair_entities(
            P.comp_pairs.get(label, []), P.is_soc, cfg.soc_pair_thresh_eV,
            center_idx=P.known_homos.get(label),
        )
        try:
            offmap = homo_entity_offsets(ents, P.known_homos.get(label))
        except ValueError:
            offmap = {}
        for offset in offsets:
            ent = offmap.get(offset)
            if ent is None:
                print(f"[MODULE A] {label} {_offset_name(offset)}: entity "
                      "not found — skipped.")
                continue
            # Combined per-entity weights over full states: full-precision
            # grid rows when available, else txt-derived recs.
            if data is not None:
                G = data["grid"][label]
                w = np.zeros(G.shape[1], dtype=np.float64)
                for i in ent.indices:
                    w += G[i - 1, :]
            else:
                w = np.zeros(E.shape[0], dtype=np.float64)
                for full_idx, comps in P.by_full.items():
                    for rec in comps.get(label, []):
                        if rec["comp_idx"] in ent.indices:
                            w[full_idx - 1] += rec["ov"]
            color = P.component_colors.get(label, {}).get(
                ent.rep, (0.4, 0.4, 0.4, 1.0))
            mo_entries.append(dict(
                label=f"{label} {_entity_label(offset, ent)}",
                comp=label, offset=offset, ent=ent,
                weights=w, color=color,
            ))

    # ------------------------------------------------------------------
    # Curve definitions: (name, weights or None for unit, color, style).
    # ------------------------------------------------------------------
    curve_defs: List[Dict[str, Any]] = [
        dict(key="total", label="Total", weights=None,
             color="black", lw=2.0, ls="-"),
    ]
    if extended:
        curve_defs += [
            dict(key="molecules", label="Molecules (MO subspace)",
                 weights=decomp["w_mo_sub"], color="tab:blue", lw=1.8, ls="-"),
            dict(key="metal_perp", label="Metal⊥ (MO-orthogonalized)",
                 weights=decomp["w_met_perp"], color="tab:orange", lw=1.8, ls="-"),
            dict(key="residual", label="Residual",
                 weights=decomp["residual"], color="tab:gray", lw=1.5, ls="-"),
        ]
    for m in mo_entries:
        curve_defs.append(dict(
            key=f"mo:{m['comp']}:{m['offset']}", label=m["label"],
            weights=m["weights"], color=m["color"], lw=1.5, ls="-"))

    # ------------------------------------------------------------------
    # Figure + widgets (σ slider with a paired two-way-synced textbox in
    # the stm_from_dft_opt style, plus the Save-data button).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(cfg.figsize[0], 4.8))
    fig.subplots_adjust(bottom=0.24)

    state: Dict[str, Any] = {
        "sigma": float(sigma), "lines": {}, "E": E,
        "curve_defs": curve_defs, "extended": extended,
        "decomp": decomp, "mo_entries": mo_entries,
        "output_dir": output_dir, "v_shifted": v_shifted,
    }

    def _xrange() -> Tuple[float, float]:
        if cfg.energy_range:
            lo, hi = cfg.energy_range
        else:
            lo = float(E.min()) - 3.0
            hi = v_shifted if v_shifted is not None else float(E.max()) + 3.0
        lo, hi = (min(lo, hi), max(lo, hi))
        if hi <= lo:
            lo, hi = lo - 1.0, hi + 1.0
        return float(lo), float(hi)

    def _compute(x: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
        if weights is None:
            states = [(float(e), 1.0) for e in E]
        else:
            states = [(float(e), float(w)) for e, w in zip(E, weights)]
        return _gauss_sum(x, states, state["sigma"])

    def _draw(initial: bool = False):
        lo, hi = _xrange()
        x = _curve_grid(lo, hi, state["sigma"])
        ymax = 0.0
        for cd in curve_defs:
            y = _compute(x, cd["weights"])
            ymax = max(ymax, float(y.max()) if y.size else 0.0)
            line = state["lines"].get(cd["key"])
            if line is None:
                (line,) = ax.plot(x, y, color=cd["color"], lw=cd["lw"],
                                  ls=cd["ls"], label=cd["label"])
                state["lines"][cd["key"]] = line
            else:
                line.set_data(x, y)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
        if initial:
            if cfg.show_fermi_line:
                ax.axvline(0.0, color=cfg.fermi_line_color,
                           linestyle=cfg.fermi_line_style, alpha=0.7)
            if cfg.show_vacuum_line and v_shifted is not None:
                ax.axvline(v_shifted, color=cfg.vacuum_line_color,
                           linewidth=cfg.vacuum_line_width,
                           linestyle=cfg.vacuum_line_style, zorder=10)
            ax.set_xlabel(cfg.xlabel)
            ax.set_ylabel("DOS-like weight (states/eV)")
            ax.set_title("Global MO/metal⊥/residual decomposition"
                         if extended else
                         "Global decomposition — legacy (MO curves + total)")
            ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    # --- σ slider + two-way-synced textbox (stm_from_dft style) ---
    ax_sld = fig.add_axes([0.28, 0.075, 0.36, 0.03])
    sld = Slider(ax_sld, r"$\sigma$ (eV)", SIGMA_MIN, SIGMA_MAX,
                 valinit=state["sigma"], valfmt="%.3f")
    ax_tb = fig.add_axes([0.70, 0.07, 0.075, 0.04])
    tb = TextBox(ax_tb, "", initial=f"{state['sigma']:.3f}")
    guard = {"busy": False}

    def _on_sigma(val):
        if guard["busy"]:
            return
        guard["busy"] = True
        try:
            state["sigma"] = float(val)
            tb.set_val(f"{state['sigma']:.3f}")
            _draw()
        finally:
            guard["busy"] = False

    def _on_text(text):
        if guard["busy"]:
            return
        try:
            v = float(text)
        except ValueError:
            tb.set_val(f"{state['sigma']:.3f}")
            return
        v = min(max(v, SIGMA_MIN), SIGMA_MAX)
        guard["busy"] = True
        try:
            sld.set_val(v)          # updates slider display
            state["sigma"] = v
            tb.set_val(f"{v:.3f}")
            _draw()
        finally:
            guard["busy"] = False

    sld.on_changed(_on_sigma)
    tb.on_submit(_on_text)

    # --- Save data button (matcher-style commented-header txt) ---
    ax_btn = fig.add_axes([0.82, 0.065, 0.12, 0.05])
    btn = Button(ax_btn, "Save data")

    def _save(_event=None) -> str:
        return save_decomposition_txt(state, P, cfg)

    btn.on_clicked(_save)

    _draw(initial=True)
    fig._mo_ldos_widgets = (sld, tb, btn)
    state["widgets"] = (sld, tb, btn)
    state["fig"], state["ax"] = fig, ax
    state["save"] = _save
    return fig, state


def save_decomposition_txt(state: Dict[str, Any], P, cfg) -> str:
    """
    Write the matcher-style commented-header txt into the run directory:
    header = σ, entity/pairing map, selected offsets, diagnostics; columns:
    E, total, molecules, metal⊥, residual, one column per selected MO curve
    (legacy runs: only the columns that exist). Returns the written path.
    """
    E = state["E"]
    sigma = state["sigma"]
    decomp = state["decomp"]
    extended = state["extended"]
    mo_entries = state["mo_entries"]

    lo, hi = state["ax"].get_xlim()
    x = _curve_grid(float(lo), float(hi), sigma)

    def curve(weights):
        if weights is None:
            states = [(float(e), 1.0) for e in E]
        else:
            states = [(float(e), float(w)) for e, w in zip(E, weights)]
        return _gauss_sum(x, states, sigma)

    cols: List[Tuple[str, np.ndarray]] = [("total", curve(None))]
    if extended:
        cols += [
            ("molecules", curve(decomp["w_mo_sub"])),
            ("metal_perp", curve(decomp["w_met_perp"])),
            ("residual", curve(decomp["residual"])),
        ]
    for m in mo_entries:
        cols.append((m["label"].replace(" ", "_"), curve(m["weights"])))

    out_path = os.path.join(state["output_dir"], "global_decomposition.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# SIGMA: {sigma:.6f}\n")
        # Entity/pairing map for every component with selected curves.
        for label in sorted({m["comp"] for m in mo_entries}):
            ents = pair_entities(
                P.comp_pairs.get(label, []), P.is_soc,
                cfg.soc_pair_thresh_eV, center_idx=P.known_homos.get(label))
            try:
                offmap = homo_entity_offsets(ents, P.known_homos.get(label))
            except ValueError:
                offmap = {}
            pair_str = "; ".join(
                f"{off:+d}={_entity_label(off, ent)},E_mean={ent.E_mean:.5f}"
                for off, ent in sorted(offmap.items()))
            f.write(f"# ENTITIES {label}: {pair_str}\n")
        sel_str = "; ".join(
            f"{m['comp']}:{m['offset']:+d}" for m in mo_entries)
        f.write(f"# SELECTED: {sel_str}\n")
        if extended:
            sd = decomp["stored_diagnostics"]
            f.write("# DIAGNOSTICS: "
                    f"S_MO_dev_max={sd.get('S_MO_dev_max', float('nan')):.6e}; "
                    f"S_mm_dev_max={sd.get('S_mm_dev_max', float('nan')):.6e}; "
                    f"min_eig_S_mt={sd.get('min_eig_S_mt', float('nan')):.6e}; "
                    f"residual_clamp_count={decomp['clamp_count']}; "
                    f"mo_sum_gap_max={decomp['mo_sum_gap_max']:.6e}\n")
        else:
            f.write("# DIAGNOSTICS: legacy run record — metal⊥/residual "
                    "require the extended matcher run\n")
        f.write("# " + " ".join(["E"] + [name for name, _ in cols]) + "\n")
        arr = np.column_stack([x] + [y for _, y in cols])
        for row in arr:
            f.write(" ".join(f"{v:.8e}" for v in row) + "\n")
    print(f"[MODULE A] Wrote decomposition data to '{out_path}'")
    return out_path


# ======================================================================
if __name__ == "__main__":

    import sys
    import types

    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_pkg_dir))
    # Stub package so relative imports resolve without installation.
    _pkg = types.ModuleType("mo_ldos_projection")
    _pkg.__path__ = [_pkg_dir]
    sys.modules.setdefault("mo_ldos_projection", _pkg)
    from mo_ldos_projection.global_decomposition import (   # noqa: F811
        plot_global_decomposition,
    )
    from mo_ldos_projection._imports import PlotConfig      # noqa: F811

    # ---- Matcher run directory (full_dir) ----
    output_dir = r"dir/output"

    # ---- Per-MO curves: {component: [HOMO-relative entity offsets]} ----
    mo_curves = {
        "NHC_c1m1": [-1, 0, 1],
        "NHC_c2m1": [-1, 0, 1],
    }

    cfg = PlotConfig(
        shared_molecule_color=True,
        soc_pair_thresh_eV=0.005,
    )

    fig, state = plot_global_decomposition(
        output_dir=output_dir,
        mo_curves=mo_curves,
        cfg=cfg,
        sigma=0.2,
    )
    plt.show()
