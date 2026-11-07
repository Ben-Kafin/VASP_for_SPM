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

2. Part 2: `PART2_ANSWERS.md` — every R of `PART2_QUESTIONS.md` accepted except A1, A2, A4, A9, A11, A15 as
   answered there; group B approved as a block; C noted. Plus the three added part-2 sections.

3. Part 2 section 11, the Decay row: no Rejali. Drop "Cite R. Rejali et al., Phys. Rev. B 107, 035406 (2023), as
   wrong_decays/tilted_barrier_decay.py does". Cite Chen only, for the frozen-bias topography and the swept dI/dV.
4. Nothing in `wrong_decays/` is written again: its files (and their Rejali / Goes mentions) and the README's
   `wrong_decays/` row stay exactly as they are now. Part 2 (including A15's text updates) does not touch them.
5. New requirement (owner): the README, the comments and the layout of the defs make clear what is the actual
   decay logic (the physics, with its citations) and what is machinery needed to integrate and to run on the
   GPU. Layout not yet ruled; default until ruled: no new module; simulator.py opens with a block "THE DECAY
   MODEL (physics)" (constants, the weight and its derivative, the definitions of I and dI/dV, citations), the
   rest under "NUMERICS AND GPU (no physics)"; each def's docstring starts with "Physics:" or "Machinery:";
   the kernel's comment says which lines transcribe the physics; the README separates the model from how it
   is computed.
6. Checks in part 2, until ruled otherwise: agents read files and run `python -m py_compile` only (no AST scans,
   no scratch maths scripts).
7. Every agent runs on Opus 5.5 or higher (workflow agents inherit the session model, claude-opus-5-5; any
   agent type with its own default model is given model: opus).
8. Part 3 section 6, the info panel (owner, 2026-11-07): "the info panel here should be like the info panel in
   the browser. it should say all the settings of the widgets or toggles and whatever the settings of the
   window is along with any information in the titles that was dropped for the sake of space".
   Part 3's plan is `handoff_part3_spectrum_decomp.md` (this folder). Its rule 2: report the design (ldos_decomp.py's
   functions, simulator.py's calls, the legend panel per view, the sm4_browser functions brought into qtgui)
   and wait for approval before any code. Source for section 6: Ben-Kafin/sm4_browser at eb9bb6279e70
   (read-only clone at /home/user/ben-kafin/sm4_browser; HEAD is that commit).
9. Part 3 rule 2 is dropped (owner, 2026-11-07): "you have my approval on part 3" / "I don't care about getting your
   approval as the plan says, that was the last thing I was trying to edit before that conversation compacted".
   No design report is waited on before code. Rules 1 (stop and report where the code does not match the plan or
   a step needs a change the plan does not name), 3, 4, 5 and 7 stand. The pre-flight's mapping serves as the
   implementing agents' brief; its open questions go to the owner in one batch, and only the work they block waits.
10. The owner's response document (2026-11-07): `RESPONSE_PART1_PART3.md` (this folder). It answers `PART1_OPEN.md`
    (every R except P2, which takes (a) with the height-sharing rule; P13 (b) as a new `mo_ldos_projection/_resolve.py`;
    P10 changed for `export_vector`) and `PART3_QUESTIONS.md` (Q1 a, Q2 a, Q3 a with the scroll bar's strip in the fixed
    width, Q4 as its sections 2 and 3, Q5 a, Q6 a, Q7 the label, Q8 a), and adds: what is taken from sm4_browser
    (settled things stand, the rest as it is), 720 dpi for every saved file that is not a matplotlib vector file,
    matplotlib required (no fallback), and the two colour-map sets in the simulator (added to part 3). Where it differs
    from the part 3 plan or an earlier answer, it stands. Verified: pyqtgraph's `colors/maps/` has CET-L3, CET-L6,
    CET-R1, CET-D1A and CET-L19 (.csv), the names it uses; `qtplot._builtin_cmap` reads them with `pg.colormap.get`.

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
- [x] 3.2-3.6, 4, 5, 6, 7: implemented, reviewed twice per file group, fixed (open items: `PART1_OPEN.md`)
- [x] Owner's ruling 1 applied to `stm_from_dft/qtgui/__init__.py`
- [x] Copy `stm_from_dft/qtgui/` over `matcher/qtgui/`; byte-identical
- [x] Part 1: py_compile every changed file (30, all pass)
- [x] Owner's answers to `PART1_OPEN.md`: `RESPONSE_PART1_PART3.md` section 1 (ruling 10)
- [ ] Apply them (with part 3: the simulator and mo_ldos_projection items after part 2's workflow); per-file change list (rule 7)
- [x] Part 2: pre-flight (read-only mapping of every bullet onto the code) done; questions in `PART2_QUESTIONS.md`
- [x] Owner's answers to `PART2_QUESTIONS.md`: `PART2_ANSWERS.md` (they override the plan; they also add three
      part-2 items: the synthetic DOSCAR removed, procar_dos.py moved to PROCAR_tools/, Map panels kept in
      increasing energy; and restate ruling 1)
- [x] Part 2: the 'width (V)' spec line removed from the root qtgui/widgets.py and both copies (d5e08a4)
- [x] Part 2 sections 2-11 + X1-X3: implemented, reviewed (completeness, scope, formulas), fixed, integration-reviewed
      (workflow wf_23c47f8c-7d0, base d5e08a4); py_compile 12 files pass; per-file list `PART2_CHANGES.md`
- [ ] Part 2 open items O1-O18 (`PART2_OPEN.md`): owner's rulings, then apply (simulator ones with part 3)
- [x] Part 3 handoff received: `handoff_part3_spectrum_decomp.md` (the decomp, the legend panel, Line/Path marker
      order and colours, sm4_browser's info panel and saves in every qtgui window); owner's ruling 8 on its info panel
- [x] Part 3 pre-flight (read-only mapping, wf_e2476f62-5f5): `PART3_PREFLIGHT.md`; its points settled in `PART3_BRIEF.md`
      (governs the implementation) or put to the owner in `PART3_OPEN.md` (K1-K9; the code they touch stays as today)
- [x] Part 3 questions (`PART3_QUESTIONS.md`, Q1-Q8) answered: `RESPONSE_PART1_PART3.md` section 5
- [ ] Part 3 implementation, with the response document's sections 2-4 (every qtgui copy, the root included) and 6
      (the colour-map sets): simulator.py and mo_ldos_projection edits wait until part 2's workflow has finished
      (one workflow edits a file at a time)
- [ ] The later views-layout handoff: its decisions so far are in `PART3_simulator_views_layout_decisions.md`
      (not part of parts 1 to 3; nothing done for it yet)
- [ ] Delete this folder in the final commit

## Open questions for the owner

- `PART2_OPEN.md` O1-O18 (part 2's open items)
- `PART3_OPEN.md` K1-K9 (part 3's)
- Relief shading (owner, 2026-11-07): asked whether the topo could relief-shade the topo panels and Map energy maps
  (Kovesi's colour x hillshade); then said not to pursue it for now. Research stopped; nothing implemented.
