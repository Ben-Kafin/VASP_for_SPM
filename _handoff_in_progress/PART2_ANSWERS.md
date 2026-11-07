# Answers to the part 2 pre-flight

Accept every R except the items below. B is approved as a block. C is noted, with no change.

## Group A

- **A1, the guard's tolerance:** neither (a) nor (b). The tolerance is half a unit of the last decimal the printed energy column carries, read from the file's own text, plus 1e-6 eV. For 3 decimals that is 0.0005 + 1e-6 eV. Keep the strict `>`, and keep the error message naming the largest difference and its file line.
- **A2, V_b at or across 0 V:** (a). I will never give a bias within one DOS step of 0, so there is no check for it.
- **A4, repeated Map samples:** (b). Map energies that snap to the same sample show the same map. The Map Norm integrates over each distinct sample once.
- **A9, the `global_topo_bias` round trip:** moot. The synthetic DOSCAR is removed (below), so a σ commit never re-instantiates the simulator and `global_topo_bias` is never fed back.
- **A11, SciPy:**
  - Transcribe SciPy 1.18.1's `simpson` with its own signature, `simpson(y, x=None, *, dx=1.0, axis=-1)` (`dx` keyword-only), not the plan's positional one.
  - Keep the guards, and use the two float32 forms you gave (the two-neighbour interpolation and the `cp.where` division).
  - `quadrature.py` is a transcription of 1.18.1, the newest release, and imports nothing from SciPy. So the simulator's integrals are the same whatever SciPy version is installed.
- **A15, the texts:** update the unnamed texts in the same terms, and adjust both wordings. On (1), Chen's chapter 2 gives κ ≈ κ0(1 − ε/2φ̄) (Eq. 2.37), with ε measured from the middle of the bias interval. So "first order in (E − V/2)/φ and V/φ" is right: the V correction starts at second order.

## Group C

C1 is moot with the synthetic DOSCAR removed.

## Part 1, 3.2: `qtgui/__init__.py`

Don't add `from . import qtplot as plt, controls, widgets` or any other import. Keep `__init__.py` docstring-only, exporting nothing, as now. Every caller imports the submodules explicitly (part 1, 3.4 to 3.6).

In both copies, change only the docstring. Replace the paragraph about walking up to the repository root, and its example code block, with one saying that this is the package's own copy and is imported relatively. Use `from .qtgui import qtplot as plt, controls, widgets` from inside the package as the example. Keep the module list. The root `qtgui/__init__.py` is not changed.

## Added to part 2: the synthetic DOSCAR is removed

The simulator always runs on the run directory's real DOSCAR, as `use_synthetic_doscar=False` does today. Do this after part 2's other sections. Paths are in `LDOS_spread_decomp/`.

- **`stm_from_dft_opt/stm_from_dft/doscar_parser.py`:**
  - Move `read_doscar_header(path)` here from `mo_ldos_projection/synthetic_doscar.py`, as a module-level function with its body unchanged and the imports it needs.
  - Drop "Module B" from its docstring.
  - `SpinAwareDosParser` keeps its own line-6 reading (section 2).
- **`mo_ldos_projection/_imports.py`:** add `resolve_doscar_parser()`.
  - It returns the `doscar_parser` module, found the way `resolve_stm()` finds it: the vendored `.stm` first, else `stm_from_dft` with the same `stm_from_dft_opt` path insert.
  - It must not import `simulator.py`. Part 1's lazy `stm_from_dft/__init__.py` makes sure of that.
- **`git rm mo_ldos_projection/synthetic_doscar.py`**, after the move.
- **`mo_ldos_projection/sim_wrapper.py`:**
  - The `procar_dos` and `synthetic_doscar` imports go (about lines 90 to 91). `read_doscar_header` comes from `resolve_doscar_parser()`.
  - The class docstring (about line 102) no longer says the PROCAR is parsed.
  - The `use_synthetic_doscar` parameter, its docstring entry and its attribute go (about lines 117, 121 to 126 and 137).
  - In `__init__`, keep `self.header = read_doscar_header(...)` as the launch check of the DOSCAR. The PROCAR parse, `self.procar` and the real-DOSCAR-mode print go. The comment above says only that the header is read early to check the DOSCAR and carries Module B's E_F.
  - `_prepare_workdir` goes, and `launch` instantiates on `self.run_dir`.
  - `commit_sigma` keeps only today's real-DOSCAR branch: the clamp, the print, and setting `self.sigma` and `self._sigma_pending`.
  - These go too, since only the removed re-instantiation used them:
    - `capture_state`
    - `_pending_prerun_state` and `_pending_postrun_state`
    - their restore in `_instantiate`
    - part 1's `t2_clamp` edits inside them
  - The module docstring:
    - A σ commit only sets the MO spread smearing for the next "Send to decomposition" press, with no teardown.
    - The σ-tagged working directory, the re-smear and the "DOSCAR source option" paragraph go.
    - Keep the note that the real DOSCAR carries VASP's own smearing, so the site DOS and the spread curves are not smeared with the same σ.
- **`mo_ldos_projection/decomp_window.py`:** take `read_doscar_header` from `resolve_doscar_parser()` (about line 151). The docstring at about line 40 no longer says σ built the synthetic DOSCAR.
- **`mo_ldos_projection/__init__.py`:** drop the `synthetic_doscar` docstring line and its `__all__` entry.
- **`mo_ldos_projection/procar_dos.py`:** moves to `PROCAR_tools/` (next section).
- **`mo_ldos_decomp/run_stm_decomp.py`:**
  - Remove my `use_synthetic_doscar = False` setting and its comment block (about lines 113 to 120), and the `use_synthetic_doscar=` keyword (about line 218).
  - Drop the DOSCAR-source part of the launch print (about line 208).
  - The docstring's step 3 (about lines 13 to 19) describes the simulator on the run directory's real DOSCAR.
  - Nothing else in the script changes.
- **READMEs:**
  - `mo_ldos_decomp/README.md`: drop the `synthetic_doscar.py` and `procar_dos.py` rows (about lines 27 to 28). The inputs line (about line 63) drops "PROCAR (LORBIT=14)", since nothing in the package reads the PROCAR any more. The outputs line (about line 65) drops "synthetic DOSCARs in σ-tagged working directories".
  - `stm_from_dft_opt/README.md`, package table: the `doscar_parser.py` row also names `read_doscar_header`, which reads only the header and gives the decomposition package its E_F.

The `mo_ldos_work/` folders that earlier synthetic runs left in run directories are outputs on my disk, not in the repository. Nothing touches them.

## Added to part 2: `procar_dos.py` moves to `PROCAR_tools/`

- **The move:** `git mv LDOS_spread_decomp/mo_ldos_decomp/mo_ldos_projection/procar_dos.py PROCAR_tools/procar_dos.py`. It imports only the standard library and NumPy, so no import changes. After this part, nothing imports it.
- **Its docstring:** "(Module B)" goes. The pointer to `synthetic_doscar.read_doscar_header` (about line 43) becomes `read_doscar_header` in `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/doscar_parser.py`. Nothing else in the file changes.
- **`mo_ldos_projection/__init__.py`:** drop the `procar_dos` docstring line and its `__all__` entry.
- **`PROCAR_tools/README.md`:**
  - The opening says the folder holds the one shared reader, `procar_parser.py`, and the tools built on it. It also holds `procar_dos.py`, a separate strict reader for LORBIT=14 noncollinear single-Γ PROCARs with validation, which the tools here do not use.
  - Add a short section for `procar_dos.py`: `parse_procar(path, spot_check_bands=12)` returns `ProcarData`, from the file's own docstring.
- **Root `README.md`, the `PROCAR_tools` row (about line 17):** also names the strict LORBIT=14 reader `procar_dos.py`.

## Added to part 2: the Map panels always in increasing energy

When an energy marker is released (`_on_rel`, the `emarker` branch), the dragged map is computed for its new snapped sample, as now. Then the maps are put back into increasing snapped energy:

- **The permutation:** a stable sort of the snapped Map energies, so maps on the same sample keep their relative order.
- **What it is applied to, everywhere together:**
  - the snapped Map energies and indices (A5)
  - the cached columns `cached_ld_up` / `cached_ld_dn` and their GPU copies
  - `cached_current` / `cached_current_E`
  - the drag cache (A8)
  - the energy markers and their labels, so marker i sits at map i's energy
- **The panels:** every map panel whose map changed position is redrawn in place with its new map: image data, title, colour limits and iso-lines, for every partition row of `plot_level`. Use the existing in-place update, `_redraw_map_slice`.
- **No arrangement or rebuild:** the number of panels and their shapes don't change, so this triggers no rearrangement, no resize and no rebuild of the panels (part 1, section 5).
- **No extra GPU work:** only the dragged map is computed. The rest is reordering what is already cached.
- **The Map Norm:** unchanged. It integrates over sorted energies, with each distinct sample once (A4).
