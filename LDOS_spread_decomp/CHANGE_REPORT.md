# MO→LDOS Projection — Delivery Change Report

Per spec §0.7: two lists. Every approved change is traceable to a spec
section, to one of the two user-approved amendments, or to the
user-directed packaging layout. Unified diffs for every modified existing
file are in `CHANGES_DIFFS.patch` (generated against the pristine
originals in `..\MOPmatcher_mem` / `..\stm_from_dft_opt`).

---

## List 1 — Approved changes

### §5 — Shared Kramers/HOMO-relative utility
* **NEW `MOPmatcher_mem/matcher/kramers.py`** — `pair_entities` (the
  pairing walk transplanted verbatim from
  `_build_colors_with_soc_pairing`; greedy adjacent ΔE < thresh; triplet →
  pair + single; `is_soc` False ⇒ singles; rep = closer-to-pivot member via
  the optional shared `center_idx` argument), `homo_entity_offsets`
  (offset 0 = entity containing the HOMO index; fallback = highest entity
  with E_mean ≤ 0 matching `get_center_idx` logic, with an optional `fermi`
  kwarg; raises when unanchorable), `combine_pair_weights`
  (p_ent(j) = Σ member rows).
* **`MOPmatcher_mem/matcher/vacupot_plotter.py`** —
  `_build_colors_with_soc_pairing` now calls `kramers.pair_entities`
  (walk extracted, not duplicated). **Regression verified: component
  colors on the example run are byte-identical before and after the
  refactor** (both the run config and default `PlotConfig`).

### §4 — Matcher extension (rerun on existing builder outputs)
* **`MOPmatcher_mem/matcher/matcher_vacupot.py`** —
  * `grid_amp/{label}` (complex128, every label incl. `metal`), written
    tile-wise inside `_stream_overlaps_and_write` from the same `S` the
    existing pass already computes, chunked/filtered identically to
    `grid`; zero additional resident memory (contiguous label-run writes).
    `grid/{label} == |grid_amp/{label}|²` verified exactly (0 deviation).
  * New `_write_cross_amplitudes`: `cross/B` (metal window × concatenated
    molecule-MO set), `cross/S_MO` (molecule-MO Gram), `cross/mo_axis_json`
    (molecule components in comp_labels order, bands ascending,
    window-local 1-based indices matching grid rows / txt comp_idx),
    `cross/diagnostics/*` — same stored blochstates, same block-sum
    S-metric GEMM, same streaming reader machinery and RAM-budget
    convention (§4.3 band scopes unchanged). Diagnostics echoed at the end
    of the matcher run.
  * **Every existing dataset and text output written exactly as before:**
    `band_matches_rectangular.txt` and `band_matches_rectangular_all.txt`
    verified **byte-identical** after the extended rerun.
* **AMENDMENT (user-approved — exact metal Gram):** the spec's
  `S_mt = I − B·S_MO⁻¹·B†` presumes orthonormal metal states; measured
  `S_mm` deviates from identity by 4.7×10⁻², driving `residual(j)` to
  −2.3×10⁻² on 1200/1794 states. The matcher therefore additionally stores
  `cross/S_mm` (metal-window Gram, same streaming pass, Hermitian-mirrored
  tile pairs) plus `S_mm_dev_max`/`S_mm_dev_fro` diagnostics, and
  `min_eig_S_mt` is computed from the exact
  `S_mt = S_mm − B·S_MO⁻¹·B†` (reduces to the original form when
  S_mm = I). With the exact form, residual ∈ [+8.3×10⁻⁶, 0.948] on the
  example — zero clamps, constraint at full strength.

### §6 — spread_plot.py / spread_collapse.py
* Kramers combining **always on**, gated only by `detect_ncl_from_incar`
  (via `P.is_soc`, same as colors); data-level pair-sum at weight
  assembly, keyed by full state (`full_idx`), strictly upstream of the
  untouched combine → normalize → amplitude pipeline. Stick mode draws
  summed heights per full state; curve mode broadens combined weights;
  Horizontal order: pair-sum within component → component-sum → pipeline.
* `comp_indices` are combined-entity HOMO-relative offsets everywhere
  (0 = HOMO entity, −1 = HOMO−1, +1 = LUMO), computed per component
  including `metal`; absolute band indices no longer accepted.
* One MO marker per entity at `E_mean`, rep-member color; titles and hover
  show both forms (`HOMO−1 (77+78), E = −0.5948 eV`) with both member
  energies in hover; missing offsets keep the existing "not found"
  behavior. Multi-run top MO row: one stick per entity at the averaged
  entity energy under the existing averaging rules.
* Both `__main__` example configs updated to the offset convention.
* Data loading, colors, Fermi/vacuum lines, pipeline math, and widgets
  untouched.

### §7 — Module A `mo_ldos_projection/global_decomposition.py` (NEW)
* Full-precision h5 reads for weights/amplitudes; colors/Fermis/HOMOs/
  is_soc/txt-fallback via `RectAEPAWColorPlotter._prepare`; energy frame
  `meta/E_full − final_fermis["full"]`; Fermi dotted at 0 and
  `v_aligned + shift` solid black (spread_plot conventions).
* Exact-projector decomposition with the approved exact-`S_mm` amendment;
  `S_MO` and `S_mt` factored once; `residual ≥ −1e−10` enforced (raise),
  negatives clamped and counted; per-MO curves are always the raw
  Kramers-combined `grid` weights (never rescaled by S_MO⁻¹); the MO-sum
  gap (max over j) reported in console and export header.
* Verified on the example: Woodbury identities ≤ 1e−12 (random S_MO=I and
  general cases), production solve vs direct-inverse ≤ 2.5e−13,
  `molecules + metal⊥ + residual = total` at σ = 0.05 and 0.35 to ~1e−15,
  `w_MO_sub + w_met⊥ ≤ 1 + 1e−10` per state.
* Legacy mode (missing `grid_amp`/`cross`, pre-amendment `cross`, or
  txt-only): MO curves + total only, console note; never emulated.
* σ slider (0.01–1.0, default 0.2) + two-way-synced textbox; in-place
  recompute; "Save data" → matcher-style commented-header txt
  (`global_decomposition.txt`, UTF-8) with σ, entity/pairing map, selected
  offsets, diagnostics, and the E/total/molecules/metal⊥/residual/per-MO
  columns. No figure-image saving code. Standalone-runnable via the
  stub-package `__main__` pattern.

### §8.1 — `mo_ldos_projection/procar_dos.py` (NEW)
* Format verified against the real files BEFORE writing (per §1.2):
  per-band block = band line + blank + orbital header + 4×(NI+1)
  projection rows (total/mx/my/mz, each NI ions + `tot`) + phase header +
  NI phase rows (ion + 9×(Re, Im) + per-ion Σ|c|²) + `charge` row + blank;
  file length 5 + NB×block asserted (1729005 lines on the example).
* Γ-only asserted (`# of k-points == 1`; multi-k fails with a clear
  message, §9.2). Energies raw; Fermi from the real DOSCAR header only.
* Hard validations at print-precision tolerances: header counts vs
  structure; per band Σ_ions(total block) ≈ the file's `tot` row; ion-row
  `tot` column spot-checks.
* **AMENDMENT (user-approved — internal phase check):** the file's phase
  sub-block uses VASP's interval-optimized projector
  (`opt. projector for intervall`), so |c|² ≠ total-block values
  (measured band-dependent scale 0.54–2.22). The consistency check
  hard-asserts the phase block's OWN identities (per-ion trailing column =
  Σ_orb|c|²; `charge` row = Σ_ions|c|²) and reports the total/|c|² scale
  range as informational. Downstream numbers come only from the
  total/mx/my/mz blocks, as spec'd.

### §8.2 — `mo_ldos_projection/synthetic_doscar.py` (NEW)
* Mirrors the real DOSCAR structure: 5 header lines copied verbatim,
  line-6 `EMAX EMIN NEDOS EFERMI 1.0000` (our grid span/length, real E_F,
  VASP field widths), per-ion repeated header line, NCL stride-4 site
  columns (total/mx/my/mz per orbital from their own parsed blocks).
  Implementation note: the energy column is printed at 5 decimals (the
  real file prints 3) so sub-5-meV grids at small σ stay single-valued;
  the whitespace-splitting parser is unaffected.
* Grid = `_curve_grid(EMIN, EMAX, σ)` (spacing min(0.005, max(σ/6,
  0.001)), 40000-point cap); NEDOS = grid length. Block 0 = unit-weight
  Σ_bands g_σ + cumulative-trapezoid integrated column (verified: parser
  auto-detects NCL, energies Fermi-shift correctly, block 0 integrates to
  exactly 1820 = n_bands on the example). Every Gaussian is a literal
  `_gauss_sum` evaluation (per-band kernel rows), never a reimplementation.
* σ-tagged dirs `<run>/mo_ldos_work/sigma_<σ>/` with the synthetic DOSCAR
  plus the enumerated simulator inputs (CONTCAR, POSCAR, LOCPOT,
  LOCPOT.npy) hardlink-or-copied; the simulator's `global_topo_*` /
  `ldos_topo_*` caches are deliberately NOT provisioned (fresh per σ ⇒ no
  stale-σ cache can ever be served; caches regenerated inside a σ-dir
  stay valid for that σ).

### §8.3 — `mo_ldos_projection/sim_wrapper.py` (NEW)
* `simulator.py` untouched (vendored copy verified **byte-identical**);
  the wrapper wraps only the instance's `_build_ui` attribute to re-attach
  its two widgets after the simulator's own `fig.clf()` rebuilds.
* σ slider + paired synced textbox; commit = slider mouse-release or
  textbox Enter only. On commit: re-smear → new σ-tagged dir → capture UI
  state → tear down → re-instantiate → restore (energy range, LDOS/topo
  heights, bias sign, all six toggles + T2 clamp, γ, Cells, E Pts,
  broadening σ per type, colormap, mode, plot level, active marker, RUN
  state, and all marker positions/ratios/path nodes/line endpoints).
* "Send to decomposition": Decay ON ⇒ prints "projection LDOS requires
  energy decay off" and does nothing (never auto-toggles). Otherwise
  collects ALL markers' spectra exactly as the spectra panel computes them
  (`cached_spec_ldos`/`cached_eg` for Single Point/Map; the
  `_combine_ldos_display` line arrays sampled at marker ratios for
  Line/Path; the panel's normalize branch included) and opens/refreshes
  the decomposition window. Snapshot semantics; never live-tracks.

### §8.4 — `mo_ldos_projection/decomp_window.py` (NEW)
* Marker blocks stacked vertically, order preserved, shared energy axis;
  one shared Vertical/Horizontal/Amplitude CheckButtons set with
  spread_collapse curve-mode semantics; V+H default ON, Amplitude OFF;
  no Curve toggle, no stick mode; persistent-figure `_build_ui()` /
  `fig.clf()` rebuild pattern; toggles re-layout from snapshot data only.
* Energy frame `meta/e_raw_full − E_F(DOSCAR header)`; automatic
  Kramers-combined MO set over all molecule components; W_ent on the LDOS
  window grid at the committed σ; threshold `MO_AREA_NOISE_THRESHOLD`
  (default 1e-4) with textbox; per-block common
  `α = A_LDOS / Σ_kept a_ent`; LDOS on every axis at unchanged scale;
  component-qualified labels; per-component lightness offsets on shared
  axes; empty-set "no MOs above threshold" annotation; display-only
  Amplitude sqrt applied after α to LDOS and MO curves alike; exports
  always probability-scale with unscaled W_ent + α in the header
  (`decomp_window_data.txt`, UTF-8).
* Verified: Σ kept drawn areas == A_LDOS to ~1e−16; α-ratio preservation
  exact; identical curve data across all four layouts; export/α
  reconstruction to 4.5e−14.

### §10 — Test suite `mo_ldos_projection/tests/` (NEW)
* 46 tests covering all nine §10 groups, runnable via
  `python -m pytest mo_ldos_projection/tests -v`; example directory
  injectable via `MO_LDOS_EXAMPLE_DIR`; GPU-dependent wrapper tests skip
  cleanly without CUDA. Full suite passes on the example (the two live-GPU
  tests require conda's `Library\bin` on PATH for CuPy's NVRTC, as in a
  normal conda/Spyder session).

### User-directed packaging layout (supersedes the spec's in-place layout)
* `mo_ldos_projection/` is standalone: vendored slimmed copies
  `mopm/` (matcher subpackage + builder physics deps: vaspwfc, aewfc,
  paw, sph_harm, vasp_constant, proj_batched — keep-list traced from real
  imports) and `stm/` (simulator.py byte-identical, doscar_parser,
  locpot_manager). `_imports.py` resolves vendored copies first, sibling
  development copies as fallback. `mopm/__init__.py` reproduces the
  original repo-root sys.path arrangement for the physics modules'
  absolute imports; the vendored `mopm/matcher/__init__.py` omits the
  exports of the two non-vendored plotters (state_decomposition_plot,
  combined_vacupot_plotter) and is otherwise identical.
* **`run_stm_decomp.py`** (sole top-level run script, per user direction):
  checks for the extended matcher record (grid_amp + cross incl. S_mm);
  if absent, runs the full builder+matcher pipeline via `run_match`
  (virgin VASP directory sets fully supported — the builder's
  `_blochstates_valid` gate decides per directory); then launches the
  Module B session. Module A and the spread plotters remain
  standalone-runnable as modules.
* **Decomposition-window projection scaling (user-directed, 3rd request —
  supersedes the spec's §8.4 common-α rule and pointwise-sqrt Amplitude;
  projection definition revised once after user feedback):** each kept
  MO spread curve is scaled by its OWN normalized overlap with the
  marker's LDOS, p = ⟨W_a, LDOS_m⟩/(‖W_a‖·‖LDOS_m‖), computed
  independently per (entity, marker) — a joint NNLS decomposition was
  tried first and rejected because competitive zeroing drew
  threshold-surviving entities flat. Drawn curves: probability view
  β·p²·W_a, magnitude view (Amplitude ON) β'·p·W_a, with one block
  factor per view keeping Σ kept areas = A_LDOS in BOTH views; the LDOS
  is drawn unchanged in both views (the earlier pointwise-sqrt display
  is replaced). Every kept entity draws with strictly positive weight.
  Projections computed once per press; layout toggles never recompute.
  Exports stay unscaled probability-scale, with the per-marker
  projections and both β factors in the header (both views exactly
  reconstructable). Per-block legends and mplcursors hover (left click
  enables, right click clears/disables; hover shows the MO's p² or p per
  the toggle) added at user direction. Tests updated accordingly.
* **DOSCAR-source input option (user-directed, 2nd request):**
  `use_synthetic_doscar` in `run_stm_decomp.py` /
  `StmDecompSession(..., use_synthetic_doscar=True)`. Default True keeps
  the spec's synthetic route unchanged. False runs the simulator on the
  run directory's REAL DOSCAR: no PROCAR parse, no σ-tagged dirs, the
  directory's own topo caches apply, and a σ commit only sets the MO
  spread-curve smearing used by the decomposition window at the next
  Send press (no teardown — the real DOSCAR does not depend on σ). The
  §9.1 one-σ guarantee between site DOS and spread curves intentionally
  does not hold in this mode (the real DOSCAR carries VASP's own
  ISMEAR/SIGMA smearing); this is a launch-time input, not a figure
  toggle, per the user's instruction.

---

## List 2 — Additional changes

**Empty.** No functional change was made beyond the spec, the two
user-approved amendments (exact metal Gram; internal phase-block
consistency check), and the user-directed packaging layout. Unspecified
implementation details (export file names, Module A aggregate curve
colors, wrapper widget rects, the 5-decimal synthetic-DOSCAR energy
column) are disclosed inline above under their sections.

---

## Verification summary (example run, `MOPM_VASP_Example/wvchg`)

| Check | Result |
|---|---|
| Colors before/after §5 refactor | byte-identical |
| txt outputs after extended matcher rerun | byte-identical (both files) |
| `grid == \|grid_amp\|²` | exact (0 deviation) |
| S_MO within-component blocks | ≈ I (max dev 1.34e−2) |
| S_mm deviation / min_eig_S_mt (exact) | 4.71e−2 / 0.4157 |
| Residual bounds (exact form) | [+8.3e−6, 0.948], 0 clamps |
| Curve identity at σ = 0.05, 0.35 | ≤ 6e−15 relative |
| PROCAR line count / block grammar | 1729005 = 5 + 1820×950, asserted |
| Synthetic DOSCAR round trip | NCL auto-detect, ∫DOS = 1820.000 |
| σ commit (GPU) | new dir + reinstantiation + state restored |
| Decay-on send | warns, does nothing |
| pytest | 46/46 pass (2 GPU tests need conda PATH) |
| `stm_from_dft_opt` working copy | identical to original (simulator.py never modified) |
