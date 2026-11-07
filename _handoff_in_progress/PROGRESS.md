# LDOS_spread_decomp handoff: progress log (temporary, deleted by the final commit)

Plans: `handoff_part1_qt_layout_text_drawing.md`, `handoff_part2_energy_integration_gpu.md` (this folder).
The owner's `disk/` files of the handoff zip are already in the repository (commit b051547).
Branch: `claude/jolly-babbage-ppmbdj`. Base of part 1's work: b051547.

## Owner's rulings (override the plan where they differ)

1. Part 1 section 3.2, `qtgui/__init__.py`: the file is only a docstring and exports nothing; keep it so.
   Add no import code. In both copies change only the docstring: the paragraph about walking up to the
   repository root and its example block are replaced by one saying this is the package's own copy,
   imported relatively, with the example `from .qtgui import qtplot as plt, controls, widgets` from
   inside the package. The module list above it stays unchanged.

## Conventions given to the implementing agents (part 1)

- Size API in the simulator's qtgui copy (`qtplot.py`): `base_font()`, `em_px()`, `px_per_pt()`, `pt2px(pt)`;
  never called at import time.
- A pixel constant c becomes `c / (1000 / 72) * em_px()`, written in that form with the old literal;
  rounded only where Qt needs an int. Named pixel constants (e.g. `CURVE_BOX_MIN_PX`) keep name and value
  (pixels at the old 100 DPI) and are converted where used. Inch sizes `x * DPI` become `x * 72 * px_per_pt()`.
- `if __package__ in (None, ''):` blocks of `controls.py` / `widgets.py` are the "sys.path insert" removed in 3.2.
- Only the simulator's copy (`stm_from_dft/qtgui/`) is edited; it is copied byte for byte over
  `matcher/qtgui/` at the end. The root `qtgui/` is unchanged (except `widgets.py` from plan 1.2).

## Status

- [x] Part 1 section 0: Claude-written checks and tests deleted (b051547)
- [x] 1.1: `wrong_decays/tilted_barrier_decay_github.py` written; `sim_decay_full.py`, `tier3_decay.py` moved (b051547)
- [x] 1.2: owner's files copied; `__init__.py` says four files; README `wrong_decays/` row names the GitHub file (b051547)
- [x] 3.1: qtgui copies made (b051547)
- [~] 3.2-3.6, 4, 5, 6, 7: in progress (workflow; uncommitted in the working tree)
- [ ] Owner's ruling 1 applied to `stm_from_dft/qtgui/__init__.py`
- [ ] Copy `stm_from_dft/qtgui/` over `matcher/qtgui/`; check byte-identical
- [ ] Part 1: py_compile every changed file; commit; per-file change list (rule 7)
- [ ] Part 2: pre-flight (read-only mapping of every bullet onto the code) done; owner questions asked
- [ ] Part 2 sections 2-11; py_compile; commit; per-file change list
- [ ] Delete this folder in the final commit

## Open questions for the owner

(none outstanding yet; filled in as the workflows report)
