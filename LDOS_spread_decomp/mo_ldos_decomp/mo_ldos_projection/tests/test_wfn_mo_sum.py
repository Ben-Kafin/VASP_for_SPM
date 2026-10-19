# -*- coding: utf-8 -*-
"""
run_wfn_mo_sum tests: filename discovery, CHGCAR header parsing (VASP5 /
VASP4 / vaspkit spacer variants), chunked streaming accumulation, the
worklist split + coverage totals, the CHGCAR writer round-trip, the
monkeypatched end-to-end run, and a smoke test against a real vaspkit
squared-wavefunction file when one is present on this machine.
"""
import io
import os

import numpy as np
import pytest

import run_wfn_mo_sum as rws
from mo_ldos_projection._imports import Entity

# A small real vaspkit output (74^3 grid, ~5.7 MB) for the smoke test.
REAL_WFN = (r"dir/run"
            r"\WFN_SOC_SQUARED_B0009_K0400_UP.vasp")


# ----------------------------------------------------------------------
# Synthetic CHGCAR-format writer (vaspkit layout, incl. the single-space
# spacer line and trailing atom labels on the coordinate rows).
# ----------------------------------------------------------------------
def write_wfn(path, values, dims=(3, 2, 2), vasp4=False, spacer=" \n",
              selective=False):
    values = np.asarray(values, dtype=np.float64)
    assert values.size == int(np.prod(dims))
    lines = ["synthetic Au1   ", " 1.000"]
    lines += ["   -2.08564427   -2.08564427    0.00000000",
              "   -2.08564427    0.00000000   -2.08564427",
              "   -0.00000000   -2.08564427   -2.08564427"]
    if not vasp4:
        lines.append("   Au")
    lines.append("    1")
    if selective:
        lines.append("Selective dynamics")
    lines.append("DIRECT")
    lines.append("    0.0000000000    0.0000000000    0.0000000000     Au1")
    body = [spacer.rstrip("\n")] if spacer else []
    body.append("  %4d %4d %4d" % dims)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines + body) + "\n")
        for start in range(0, values.size, 5):
            row = values[start:start + 5]
            f.write(" ".join("%.10E" % v for v in row) + "\n")


@pytest.fixture()
def grids():
    rng = np.random.default_rng(42)
    return {name: rng.uniform(0.0, 1e-3, size=12)
            for name in ("up5", "dw5", "up6", "up9")}


@pytest.fixture()
def wfn_dir(tmp_path, grids):
    d = tmp_path / "run"
    d.mkdir()
    write_wfn(d / "WFN_SOC_SQUARED_B0005_K0001_UP.vasp", grids["up5"])
    write_wfn(d / "WFN_SOC_SQUARED_B0005_K0001_DW.vasp", grids["dw5"])
    write_wfn(d / "WFN_SOC_SQUARED_B0006_K0001_UP.vasp", grids["up6"])
    write_wfn(d / "WFN_SOC_SQUARED_B0009_K0001_UP.vasp", grids["up9"])
    # Other k-point, below-threshold band, and non-matching names.
    write_wfn(d / "WFN_SOC_SQUARED_B0005_K0002_UP.vasp", grids["up5"])
    write_wfn(d / "WFN_SOC_SQUARED_B0007_K0001_UP.vasp", grids["up6"])
    (d / "WFN_SOC_SQUARED_B0005_K0001_UP.vasp.bak").write_text("x")
    (d / "CHGCAR").write_text("x")
    return str(d)


# ----------------------------------------------------------------------
# Discovery.
# ----------------------------------------------------------------------
def test_discovery_filters_and_groups(wfn_dir):
    by_band, skipped = rws.discover_wfn_files(wfn_dir, 1)
    assert sorted(by_band) == [5, 6, 7, 9]
    assert sorted(by_band[5]) == ["DW", "UP"]
    assert sorted(by_band[6]) == ["UP"]
    assert [os.path.basename(s) for s in skipped] == \
        ["WFN_SOC_SQUARED_B0005_K0002_UP.vasp"]


def test_discovery_accepts_dn_and_plain_wfn(tmp_path):
    write_wfn(tmp_path / "WFN_SQUARED_B0003_K0001_DN.vasp", np.zeros(12))
    by_band, skipped = rws.discover_wfn_files(str(tmp_path), 1)
    assert sorted(by_band) == [3] and sorted(by_band[3]) == ["DW"]
    assert not skipped


# ----------------------------------------------------------------------
# Header parsing.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("vasp4", [False, True])
@pytest.mark.parametrize("spacer", [" \n", "\n", ""])
@pytest.mark.parametrize("selective", [False, True])
def test_header_variants(tmp_path, vasp4, spacer, selective):
    path = tmp_path / "w.vasp"
    vals = np.arange(12, dtype=np.float64)
    write_wfn(path, vals, vasp4=vasp4, spacer=spacer, selective=selective)
    with open(path, "rb") as f:
        header, natoms, dims = rws.read_chgcar_header(f)
        assert natoms == 1 and dims == (3, 2, 2)
        got = np.fromfile(f, dtype=np.float64, count=12, sep=" ")
    np.testing.assert_allclose(got, vals, rtol=1e-9)
    # The header bytes end exactly at the dims line.
    assert header.split(b"\n")[-2].split() == [b"3", b"2", b"2"]


def test_header_truncated_raises(tmp_path):
    path = tmp_path / "w.vasp"
    path.write_text("comment\n 1.0\n")
    with open(path, "rb") as f:
        with pytest.raises(ValueError, match="end of file"):
            rws.read_chgcar_header(f)


def test_header_bad_dims_raises(tmp_path):
    path = tmp_path / "w.vasp"
    vals = np.arange(12, dtype=np.float64)
    write_wfn(path, vals)
    text = path.read_text().replace("     3    2    2", "  3 2")
    path.write_text(text)
    with open(path, "rb") as f:
        with pytest.raises(ValueError):
            rws.read_chgcar_header(f)


# ----------------------------------------------------------------------
# Chunked accumulation.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [1, 5, 7, 12, 1000])
def test_accumulate_chunked(tmp_path, chunk):
    path = tmp_path / "w.vasp"
    vals = np.linspace(0.0, 1.0, 12)
    write_wfn(path, vals)
    acc = np.zeros(12)
    with open(path, "rb") as f:
        rws.read_chgcar_header(f)
        rws.accumulate_grid(f, 0.25, acc, chunk)
    np.testing.assert_allclose(acc, 0.25 * vals, rtol=1e-9)
    # A second pass adds on top (accumulator semantics).
    with open(path, "rb") as f:
        rws.read_chgcar_header(f)
        rws.accumulate_grid(f, 0.5, acc, chunk)
    np.testing.assert_allclose(acc, 0.75 * vals, rtol=1e-9)


def test_accumulate_truncated_raises(tmp_path):
    path = tmp_path / "w.vasp"
    write_wfn(path, np.arange(12, dtype=np.float64))
    # Claim a larger grid than the file holds.
    text = path.read_text().replace("     3    2    2", "     3    2    3")
    path.write_text(text)
    acc = np.zeros(18)
    with open(path, "rb") as f:
        rws.read_chgcar_header(f)
        with pytest.raises(ValueError, match="truncated"):
            rws.accumulate_grid(f, 1.0, acc, 5)


# ----------------------------------------------------------------------
# Worklist split + coverage totals.
# ----------------------------------------------------------------------
def test_build_worklist_split():
    weights = np.array([0.0, 0.0, 0.3, 0.0, 0.4, 0.2, 1e-6, 0.0])
    by_band = {5: {"UP": "a", "DW": "b"}, 6: {"UP": "c"},
               7: {"UP": "d"}, 9: {"UP": "e"}}
    wl = rws.build_worklist(by_band, weights, 1e-3)
    assert [(b, w) for b, w, _ in wl["included"]] == [(5, 0.4), (6, 0.2)]
    assert wl["below"] == [(7, 1e-6)]
    assert wl["above_window"] == [9]
    assert wl["missing"] == [(3, 0.3)]
    assert wl["w_total"] == pytest.approx(weights.sum())
    assert wl["w_passing"] == pytest.approx(0.9)
    assert wl["w_included"] == pytest.approx(0.6)


# ----------------------------------------------------------------------
# Writer round-trip.
# ----------------------------------------------------------------------
def test_write_roundtrip(tmp_path):
    src = tmp_path / "src.vasp"
    vals = np.linspace(-1e-4, 1e-3, 12)
    write_wfn(src, np.zeros(12))
    with open(src, "rb") as f:
        header, _, dims = rws.read_chgcar_header(f)
    out = tmp_path / "out.vasp"
    rws.write_chgcar(str(out), header, vals, chunk_values=7)
    with open(out, "rb") as f:
        header2, natoms2, dims2 = rws.read_chgcar_header(f)
        got = np.fromfile(f, dtype=np.float64, count=12, sep=" ")
        assert f.read().strip() == b""      # nothing after the grid
    assert dims2 == dims and natoms2 == 1
    np.testing.assert_allclose(got, vals, rtol=1e-10)
    # 12 values → two 5-value lines + one 2-value remainder line.
    grid_lines = [l for l in out.read_text().splitlines()][-3:]
    assert [len(l.split()) for l in grid_lines] == [5, 5, 2]


# ----------------------------------------------------------------------
# End-to-end run() with the entity resolution monkeypatched.
# ----------------------------------------------------------------------
def test_run_end_to_end(wfn_dir, grids, monkeypatch, capsys):
    weights = np.array([0.0, 0.0, 0.3, 0.0, 0.4, 0.2, 1e-6, 0.0])

    def fake_resolve(run_dir, cfg, component, offset):
        assert component == "NHC_test" and offset == 0
        return dict(weights=weights, ent=Entity((41, 42), -1.0, 41),
                    title="NHC_test_HOMO", label="NHC_test HOMO (41+42)",
                    n_full=8)

    monkeypatch.setattr(rws, "resolve_entity_weights", fake_resolve)
    monkeypatch.setattr(rws, "matcher_k_index", lambda rd: 1)

    out = rws.run(wfn_dir, None, "NHC_test", 0, 1e-3, chunk_values=5)
    assert out == os.path.join(
        wfn_dir, "NHC_test_HOMO_wvfx_squared_probability.vasp")
    assert os.path.isfile(out)

    expected = (0.4 * (grids["up5"] + grids["dw5"]) + 0.2 * grids["up6"])
    with open(out, "rb") as f:
        _, natoms, dims = rws.read_chgcar_header(f)
        got = np.fromfile(f, dtype=np.float64, count=12, sep=" ")
    assert dims == (3, 2, 2) and natoms == 1
    np.testing.assert_allclose(got, expected, rtol=1e-8)

    report = capsys.readouterr().out
    assert "band 3: weight 3.000e-01 — missing" in report
    assert "band 7: weight 1.000e-06 < threshold" in report
    assert "band 9: file(s) beyond the matcher band window" in report
    assert "spinor partner file missing" in report
    assert "ignoring 1 file(s) at other k-points" in report


def test_run_nothing_passes(wfn_dir, monkeypatch, capsys):
    def fake_resolve(run_dir, cfg, component, offset):
        return dict(weights=np.zeros(8), ent=Entity((1,), -1.0, 1),
                    title="X_HOMO", label="X HOMO (1)", n_full=8)

    monkeypatch.setattr(rws, "resolve_entity_weights", fake_resolve)
    monkeypatch.setattr(rws, "matcher_k_index", lambda rd: 1)
    assert rws.run(wfn_dir, None, "X", 0, 1e-3) is None
    assert "nothing to sum" in capsys.readouterr().out


def test_run_dims_mismatch_raises(tmp_path, monkeypatch):
    d = tmp_path / "run"
    d.mkdir()
    write_wfn(d / "WFN_SOC_SQUARED_B0001_K0001_UP.vasp",
              np.zeros(12), dims=(3, 2, 2))
    write_wfn(d / "WFN_SOC_SQUARED_B0002_K0001_UP.vasp",
              np.zeros(8), dims=(2, 2, 2))
    monkeypatch.setattr(
        rws, "resolve_entity_weights",
        lambda *a: dict(weights=np.array([0.5, 0.5]),
                        ent=Entity((1,), -1.0, 1),
                        title="X_HOMO", label="X HOMO (1)", n_full=2))
    monkeypatch.setattr(rws, "matcher_k_index", lambda rd: 1)
    with pytest.raises(ValueError, match="not summable"):
        rws.run(str(d), None, "X", 0, 1e-3, chunk_values=5)


# ----------------------------------------------------------------------
# Real vaspkit file smoke test (only on this machine).
# ----------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isfile(REAL_WFN),
                    reason="real vaspkit sample not present")
def test_real_vaspkit_file_streams():
    with open(REAL_WFN, "rb") as f:
        header, natoms, dims = rws.read_chgcar_header(f)
        assert natoms == 1 and dims == (74, 74, 74)
        acc = np.zeros(int(np.prod(dims)))
        rws.accumulate_grid(f, 2.0, acc, chunk_values=100_000)
    assert np.isfinite(acc).all() and (acc >= 0.0).all()
    # First grid value of this file is 0.81774E-18 (weighted by 2).
    assert acc[0] == pytest.approx(2 * 0.81774e-18, rel=1e-4)
    assert acc.max() > 0.0
