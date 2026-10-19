# -*- coding: utf-8 -*-
"""Section 10.2 — synthetic DOSCAR: SpinAwareDosParser auto-detects NCL;
column and ion counts match the real DOSCAR; energies come back
Fermi-shifted correctly; block-0 total integrates to the band count."""
import numpy as np

from mo_ldos_projection._imports import resolve_stm

_TRAPZ = getattr(np, "trapezoid", np.trapz)


def test_parser_autodetect_and_shapes(synth_doscar, procar, doscar_header):
    _, doscar_parser = resolve_stm()
    path, info = synth_doscar
    p = doscar_parser.SpinAwareDosParser(path)
    assert p.spin_mode == doscar_parser.SpinMode.NCL_SOC
    assert p.ef == doscar_header["efermi"]
    # ion count and per-site column count match the real DOSCAR layout
    assert p.spin_up_dos.shape == (procar.nions, info["nedos"], procar.norb)
    assert p.mag_dos is not None
    # energies come back Fermi-shifted
    assert abs(p.energies[0] - (doscar_header["emin"]
                                - doscar_header["efermi"])) < 1e-9
    assert abs(p.energies[-1] - (doscar_header["emax"]
                                 - doscar_header["efermi"])) < 1e-9


def test_block0_integrates_to_band_count(synth_doscar, procar):
    path, info = synth_doscar
    nedos = info["nedos"]
    with open(path) as f:
        for _ in range(6):
            f.readline()
        rows = np.array([f.readline().split() for _ in range(nedos)],
                        dtype=float)
    integral = _TRAPZ(rows[:, 1], rows[:, 0])
    # unit weight per state; all example eigenvalues lie inside the span,
    # so only Gaussian tails leak (tail tolerance)
    assert abs(integral - procar.nbands) < 0.05 * procar.nbands / 100 + 1.0
    # integrated-DOS column reaches the band count too
    assert abs(rows[-1, 2] - procar.nbands) < 1.0


def test_site_block_matches_shared_gauss_sum(synth_doscar, procar):
    """A site value equals the shared _gauss_sum of that (ion, orbital)
    weight over all bands — one Gaussian convention everywhere."""
    from mo_ldos_projection._imports import _gauss_sum
    _, doscar_parser = resolve_stm()
    path, info = synth_doscar
    p = doscar_parser.SpinAwareDosParser(path)
    x = info["energies"]
    ion, orb = procar.nions // 2, 0
    states = [(float(e), float(w)) for e, w in
              zip(procar.energies, procar.total[:, ion, orb])]
    ref = _gauss_sum(x, states, info["sigma"])
    got = p.spin_up_dos[ion, :, orb]
    # file stores %.4E — compare at that precision
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=1e-8)


def test_workdir_provisioning(tmp_path, procar, doscar_header, example_dir):
    """σ-tagged dir contains the synthetic DOSCAR plus every enumerated
    simulator input (hardlink-or-copy)."""
    import os
    import shutil
    from mo_ldos_projection import synthetic_doscar as sd
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Miniature stand-ins for the provisioned inputs
    for name in ("CONTCAR", "POSCAR", "LOCPOT"):
        (run_dir / name).write_text("stub\n")
    np.save(str(run_dir / "LOCPOT.npy"), np.zeros((2, 2, 2)))
    shutil.copyfile(os.path.join(example_dir, "DOSCAR"),
                    str(run_dir / "DOSCAR")) if False else None
    # header passed explicitly so the real DOSCAR need not be copied
    wd = sd.make_sigma_workdir(str(run_dir), 0.3141, procar,
                               header=doscar_header)
    assert wd.endswith(f"sigma_{0.3141:.4f}")
    assert os.path.isfile(os.path.join(wd, "DOSCAR"))
    for name in sd.SIM_INPUT_FILES:
        assert os.path.isfile(os.path.join(wd, name)), name
    # reuse: second call must not rebuild (same mtime)
    m1 = os.path.getmtime(os.path.join(wd, "DOSCAR"))
    sd.make_sigma_workdir(str(run_dir), 0.3141, procar, header=doscar_header)
    assert os.path.getmtime(os.path.join(wd, "DOSCAR")) == m1
