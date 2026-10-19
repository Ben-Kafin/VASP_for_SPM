# -*- coding: utf-8 -*-
"""
run_mo_band_lists tests: the component-internal classification (ranking,
tie handling, threshold gate, top_n pool depth), the lists text file,
the vaspkitjob.sh generator (heredoc sequence, skip guard, LF endings),
the all-components union mode, and the band_list restriction wired into
run_wfn_mo_sum.
"""
import os
import types

import numpy as np
import pytest

import run_mo_band_lists as rbl
import run_wfn_mo_sum as rws
from mo_ldos_projection._imports import Entity

from mo_ldos_projection.tests.test_wfn_mo_sum import (  # noqa: F401
    write_wfn, grids, wfn_dir)  # fixtures


def _row(comp, offset, weights, label=None):
    return dict(comp=comp, offset=offset,
                weights=np.asarray(weights, dtype=np.float64),
                ent=Entity((1, 2), 0.0, 1),
                label=label or f"{comp} off{offset}")


@pytest.fixture()
def entity_rows():
    #            band:    1     2     3     4      5    6    7    8
    target = _row("M", 0, [0.5, 0.3, 0.2, 1e-6, 0.0, 0.2, 0.4, 0.0],
                  label="M HOMO (1+2)")
    other1 = _row("M", -1, [0.1, 0.4, 0.3, 0.5, 0.6, 0.2, 0.0, 0.0])
    other2 = _row("M", 1, [0.1, 0.0, 0.25, 0.5, 0.0, 0.1, 0.0, 0.0])
    noise = _row("Z", 0, [9.0] * 8)   # different component: never in pool
    return [other1, target, other2, noise]


def test_classify_bands_lists(entity_rows):
    cls = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3, top_n=5)
    # gate removes bands 4 (1e-6), 5 and 8 (0.0)
    assert cls["top5"] == [1, 2, 3, 6, 7]
    # rank 1 = no other entity strictly greater; band 6 is a tie -> rank 1
    assert cls["greatest"] == [1, 6, 7]
    # strict majority over the sum of the others
    assert cls["dominant"] == [1, 7]
    assert cls["n_entities"] == 3
    # dominant ⊆ greatest ⊆ top5
    assert set(cls["dominant"]) <= set(cls["greatest"]) <= set(cls["top5"])
    w, rank, others = cls["detail"][2]
    assert w == pytest.approx(0.3) and rank == 2
    assert others == pytest.approx(0.4)


def test_classify_bands_top_n_pool(entity_rows):
    cls = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3, top_n=2)
    # band 3: target rank 3 (0.3 and 0.25 both beat 0.2) -> out at top_n=2
    assert cls["top5"] == [1, 2, 6, 7]


def test_classify_bands_single_entity():
    rows = [_row("S", 0, [0.2, 0.0, 5e-4])]
    cls = rbl.classify_bands(rows, "S", 0, threshold=1e-3)
    assert cls["top5"] == cls["greatest"] == cls["dominant"] == [1]


def test_classify_bands_errors(entity_rows):
    with pytest.raises(KeyError, match="no entity rows"):
        rbl.classify_bands(entity_rows, "missing", 0)
    with pytest.raises(KeyError, match="offset 7 not available"):
        rbl.classify_bands(entity_rows, "M", 7)


def test_classify_bands_energy_gate(entity_rows):
    # band energies: -2.0 -1.5 -1.0 -0.5  0.0  0.5  1.0  1.5
    E = np.linspace(-2.0, 1.5, 8)
    base = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3)
    cls = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3,
                             energies=E, energy_range=[-2.25, -0.2])
    # window keeps bands 1-4; of the baseline lists only 1,2,3 survive
    assert cls["top5"] == [1, 2, 3]
    assert cls["greatest"] == [1]
    assert cls["dominant"] == [1]
    # the gate drops bands, it never changes a rank
    for b in cls["top5"]:
        assert cls["detail"][b][1] == base["detail"][b][1]
    # reversed bounds are equivalent
    rev = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3,
                             energies=E, energy_range=[-0.2, -2.25])
    assert rev["top5"] == cls["top5"]
    # either argument alone applies no gate
    assert rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3,
                              energies=E)["top5"] == base["top5"]
    with pytest.raises(ValueError, match="does not match"):
        rbl.classify_bands(entity_rows, "M", 0, energies=E[:3],
                           energy_range=[-1.0, 0.0])


def test_write_band_lists_roundtrip(tmp_path, entity_rows):
    cls = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3)
    path = tmp_path / "M_HOMO_band_lists.txt"
    rbl.write_band_lists(str(path), cls, str(tmp_path), "M", 0, 1, 1e-3)
    text = path.read_text()
    assert "[top5]" in text and "[greatest]" in text and "[dominant]" in text
    assert "(5 bands)" in text and "(3 bands)" in text and "(2 bands)" in text
    assert "ranking pool: the 3 Kramers entities of component 'M'" in text
    # a data line: band 2, no energies -> n/a, weight 3e-1, rank 2,
    # others 4e-1
    line = [l for l in text.splitlines() if l.strip().startswith("2 ")][0]
    band, e_col, w, rank, others = line.split()
    assert (int(band), int(rank)) == (2, 2)
    assert e_col == "n/a"
    assert float(w) == pytest.approx(0.3)
    assert float(others) == pytest.approx(0.4)
    assert b"\r" not in path.read_bytes()


def test_write_band_lists_energy_gate(tmp_path, entity_rows):
    E = np.linspace(-2.0, 1.5, 8)
    erange = [-2.25, -0.2]
    cls = rbl.classify_bands(entity_rows, "M", 0, threshold=1e-3,
                             energies=E, energy_range=erange)
    path = tmp_path / "M_HOMO_band_lists.txt"
    rbl.write_band_lists(str(path), cls, str(tmp_path), "M", 0, 1, 1e-3,
                         energies=E, energy_range=erange)
    text = path.read_text()
    assert "# energy gate: -2.25 <= E - E_F(DOSCAR) <= -0.2 eV" in text
    # band 2 carries its energy in the E-Ef column
    line = [l for l in text.splitlines() if l.strip().startswith("2 ")][0]
    band, e_col, w, rank, others = line.split()
    assert float(e_col) == pytest.approx(-1.5)
    # out-of-window bands (6, 7 of the ungated lists) never appear
    assert not [l for l in text.splitlines() if l.strip().startswith("6 ")]
    assert not [l for l in text.splitlines() if l.strip().startswith("7 ")]


def test_write_vaspkit_job(tmp_path):
    path = tmp_path / "vaspkitjob.sh"
    rbl.write_vaspkit_job(str(path), [3, 151, 152], 1)
    raw = path.read_bytes()
    assert b"\r" not in raw                      # LF-only for talapas
    text = raw.decode()
    assert text.startswith("#!/bin/bash\n")
    assert "#SBATCH --account=ACCOUNT" in text
    assert "#SBATCH --ntasks=1" in text
    assert "srun" not in text                    # serial tool, no MPI
    assert "KPOINT=1" in text
    assert "    3 151 152" in text
    # the interactive sequence: 51 -> 515 -> kpoint -> band, run
    # sequentially (no parallel machinery)
    assert "vaspkit <<EOF\n51\n515\n${KPOINT}\n${BAND}\nEOF" in text
    assert "wait -n" not in text and "run_band" not in text
    assert "scratch" not in text
    # idempotence guard on both spinor files
    assert 'WFN_SOC_SQUARED_B${BPAD}_K${KPAD}_UP.vasp' in text
    assert 'WFN_SOC_SQUARED_B${BPAD}_K${KPAD}_DW.vasp' in text


def test_run_all_components_union(tmp_path, monkeypatch, entity_rows):
    rows = entity_rows + [_row("Z", -1, [0.0] * 8)]
    fake = dict(P=types.SimpleNamespace(is_soc=True),
                mol_labels=["M", "Z"], entity_rows=rows,
                entities_by_comp={})
    import mo_ldos_projection.decomp_window as dw
    monkeypatch.setattr(dw, "load_static_mo_data", lambda rd, cfg: fake)
    monkeypatch.setattr(rbl, "matcher_k_index", lambda rd: 1)

    out = rbl.run(str(tmp_path), None, None, 0, 1e-3)
    assert [os.path.basename(p) for p in out["list_paths"]] == \
        ["M_HOMO_band_lists.txt", "Z_HOMO_band_lists.txt"]
    for p in out["list_paths"]:
        assert os.path.isfile(p)
    # union of M top5 {1,2,3,6,7} and Z top5 (all 8 bands at w=9.0)
    assert out["job_bands"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert os.path.isfile(out["job_path"])
    assert out["skipped"] == []


def test_run_union_skips_broken_component(tmp_path, monkeypatch,
                                          entity_rows, capsys):
    # EMPTY has no entity rows (unresolvable HOMO anchor): union mode
    # must skip it, keep the healthy components, and still write the job.
    fake = dict(P=types.SimpleNamespace(is_soc=True),
                mol_labels=["M", "EMPTY", "Z"], entity_rows=entity_rows,
                entities_by_comp={})
    import mo_ldos_projection.decomp_window as dw
    monkeypatch.setattr(dw, "load_static_mo_data", lambda rd, cfg: fake)
    monkeypatch.setattr(rbl, "matcher_k_index", lambda rd: 1)

    out = rbl.run(str(tmp_path), None, None, 0, 1e-3)
    assert [os.path.basename(p) for p in out["list_paths"]] == \
        ["M_HOMO_band_lists.txt", "Z_HOMO_band_lists.txt"]
    assert out["skipped"] == ["EMPTY"]
    assert os.path.isfile(out["job_path"])
    report = capsys.readouterr().out
    assert "SKIPPING component 'EMPTY'" in report
    assert "union job covers only" in report


def test_run_union_all_broken_raises(tmp_path, monkeypatch):
    fake = dict(P=types.SimpleNamespace(is_soc=True),
                mol_labels=["EMPTY"], entity_rows=[],
                entities_by_comp={})
    import mo_ldos_projection.decomp_window as dw
    monkeypatch.setattr(dw, "load_static_mo_data", lambda rd, cfg: fake)
    monkeypatch.setattr(rbl, "matcher_k_index", lambda rd: 1)
    with pytest.raises(RuntimeError, match="no component produced"):
        rbl.run(str(tmp_path), None, None, 0, 1e-3)


def test_run_named_component_still_raises(tmp_path, monkeypatch,
                                          entity_rows):
    # An explicitly named component must NOT be silently skipped.
    fake = dict(P=types.SimpleNamespace(is_soc=True),
                mol_labels=["M"], entity_rows=entity_rows,
                entities_by_comp={})
    import mo_ldos_projection.decomp_window as dw
    monkeypatch.setattr(dw, "load_static_mo_data", lambda rd, cfg: fake)
    monkeypatch.setattr(rbl, "matcher_k_index", lambda rd: 1)
    with pytest.raises(KeyError, match="offset 9 not available"):
        rbl.run(str(tmp_path), None, "M", 9, 1e-3)


def test_wfn_sum_band_list_restriction(wfn_dir, grids, monkeypatch, capsys):
    # Component rows: the target MO loses band 6 to a sibling MO but wins
    # band 5 — 'greatest' must therefore exclude band 6 from the sum.
    target_w = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.0, 0.0])
    other_w = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.0, 0.0])
    comp_rows = [_row("NHC_test", 0, target_w, label="NHC_test HOMO"),
                 _row("NHC_test", 1, other_w)]

    def fake_resolve(run_dir, cfg, component, offset):
        return dict(weights=target_w, ent=Entity((41, 42), -1.0, 41),
                    title="NHC_test_HOMO", label="NHC_test HOMO (41+42)",
                    n_full=8, comp_rows=comp_rows)

    monkeypatch.setattr(rws, "resolve_entity_weights", fake_resolve)
    monkeypatch.setattr(rws, "matcher_k_index", lambda rd: 1)

    out = rws.run(wfn_dir, None, "NHC_test", 0, 1e-3, chunk_values=5,
                  band_list="greatest")
    assert out == os.path.join(
        wfn_dir, "NHC_test_HOMO_greatest_wvfx_squared_probability.vasp")

    expected = 0.4 * (grids["up5"] + grids["dw5"])   # band 6 excluded
    with open(out, "rb") as f:
        rws.read_chgcar_header(f)
        got = np.fromfile(f, dtype=np.float64, count=12, sep=" ")
    np.testing.assert_allclose(got, expected, rtol=1e-8)

    report = capsys.readouterr().out
    assert "band list 'greatest': 1 band(s)" in report
    assert "band 6: weight 2.000e-01 but not on the 'greatest' band list" \
        in report


def test_wfn_sum_band_list_unknown_name(wfn_dir, monkeypatch):
    monkeypatch.setattr(
        rws, "resolve_entity_weights",
        lambda *a: dict(weights=np.zeros(8), ent=Entity((1,), 0.0, 1),
                        title="X_HOMO", label="X", n_full=8, comp_rows=[]))
    monkeypatch.setattr(rws, "matcher_k_index", lambda rd: 1)
    with pytest.raises(KeyError, match="band_list 'bogus' unknown"):
        rws.run(wfn_dir, None, "X", 0, 1e-3, band_list="bogus")


def test_resolve_entity_weights_empty_offmap(monkeypatch):
    fake = dict(P=types.SimpleNamespace(is_soc=True),
                entities_by_comp={"X": {}}, entity_rows=[],
                mol_labels=["X"])
    import mo_ldos_projection.decomp_window as dw
    monkeypatch.setattr(dw, "load_static_mo_data", lambda rd, cfg: fake)
    with pytest.raises(KeyError, match="no resolvable Kramers entities"):
        rws.resolve_entity_weights("rd", None, "X", 0)


def test_resolve_entity_weights_soc_warning(monkeypatch, capsys):
    row = _row("X", 0, [0.1, 0.2])
    fake = dict(P=types.SimpleNamespace(is_soc=False),
                entities_by_comp={"X": {0: row["ent"]}},
                entity_rows=[row], mol_labels=["X"])
    import mo_ldos_projection.decomp_window as dw
    monkeypatch.setattr(dw, "load_static_mo_data", lambda rd, cfg: fake)
    info = rws.resolve_entity_weights("rd", None, "X", 0)
    assert info["title"] == "X_HOMO"
    assert "NOT flagged SOC" in capsys.readouterr().out
