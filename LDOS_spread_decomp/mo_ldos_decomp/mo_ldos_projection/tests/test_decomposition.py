# -*- coding: utf-8 -*-
"""Section 10.4 — decomposition math: Woodbury vs direct inverse (random
S_MO = I case and general case) to 1e-12; curve identity
molecules + metal⊥ + residual = total at two σ values (rtol 1e-10);
residual clamp count reported; per-state w_MO_sub + w_met⊥ ≤ 1 + 1e-10."""
import numpy as np
import pytest

from mo_ldos_projection.global_decomposition import compute_decomposition
from mo_ldos_projection._imports import _gauss_sum, _curve_grid


@pytest.fixture(scope="module")
def decomp(results):
    if not results["extended"]:
        pytest.skip("extended matcher record required")
    return compute_decomposition(results)


def test_woodbury_identity_random_smo_identity():
    rng = np.random.default_rng(0)
    nm, na = 40, 12
    B = 0.25 * (rng.standard_normal((nm, na))
                + 1j * rng.standard_normal((nm, na))) / np.sqrt(nm)
    direct = np.linalg.inv(np.eye(nm) - B @ B.conj().T)
    wood = np.eye(nm) + B @ np.linalg.solve(
        np.eye(na) - B.conj().T @ B, B.conj().T)
    assert np.max(np.abs(direct - wood)) < 1e-12


def test_woodbury_identity_random_general_smo():
    rng = np.random.default_rng(1)
    nm, na = 40, 12
    B = 0.25 * (rng.standard_normal((nm, na))
                + 1j * rng.standard_normal((nm, na))) / np.sqrt(nm)
    A = 0.1 * (rng.standard_normal((na, na))
               + 1j * rng.standard_normal((na, na)))
    S_MO = np.eye(na) + 0.5 * (A + A.conj().T)
    direct = np.linalg.inv(
        np.eye(nm) - B @ np.linalg.solve(S_MO, B.conj().T))
    wood = np.eye(nm) + B @ np.linalg.solve(
        S_MO - B.conj().T @ B, B.conj().T)
    assert np.max(np.abs(direct - wood)) < 1e-12


def test_exact_smt_solve_vs_direct_inverse(results, decomp):
    """The production Cholesky-solve path against the explicit inverse of
    the exact S_mt = S_mm − B·S_MO⁻¹·B† (approved amendment)."""
    B, S_MO, S_mm = results["B"], results["S_MO"], results["S_mm"]
    S_MO_h = 0.5 * (S_MO + S_MO.conj().T)
    S_mt = 0.5 * (S_mm + S_mm.conj().T) - B @ np.linalg.solve(
        S_MO_h, B.conj().T)
    S_mt = 0.5 * (S_mt + S_mt.conj().T)
    S_mt_inv = np.linalg.inv(S_mt)
    mol_order = []
    for label, _ in results["mo_axis"]:
        if not mol_order or mol_order[-1] != label:
            mol_order.append(label)
    c_mo = np.vstack([results["grid_amp"][l] for l in mol_order])
    chi = results["grid_amp"]["metal"] - B @ np.linalg.solve(S_MO_h, c_mo)
    sub = slice(0, 200)
    ref = np.einsum("mj,mn,nj->j", chi[:, sub].conj(), S_mt_inv,
                    chi[:, sub]).real
    assert np.max(np.abs(ref - decomp["w_met_perp"][sub])) < 1e-10


def test_curve_identity_at_two_sigmas(results, decomp):
    E = results["E_full_raw"] - results["fermi_full"]
    parts_w = (decomp["w_mo_sub"], decomp["w_met_perp"], decomp["residual"])
    for sigma in (0.05, 0.35):
        x = _curve_grid(float(E.min()) - 3.0, float(E.max()) + 3.0, sigma)
        tot = _gauss_sum(x, [(float(e), 1.0) for e in E], sigma)
        acc = np.zeros_like(tot)
        for w in parts_w:
            acc += _gauss_sum(x, list(zip(E.tolist(), w.tolist())), sigma)
        np.testing.assert_allclose(acc, tot, rtol=1e-10,
                                   atol=1e-10 * float(tot.max()))


def test_residual_bounds_and_clamps(decomp):
    r = decomp["residual"]
    assert np.all(r >= 0.0)
    assert isinstance(decomp["clamp_count"], int)
    s = decomp["w_mo_sub"] + decomp["w_met_perp"]
    assert np.all(s <= 1.0 + 1e-10)


def test_mo_sum_gap_reported(decomp):
    assert np.isfinite(decomp["mo_sum_gap_max"])
