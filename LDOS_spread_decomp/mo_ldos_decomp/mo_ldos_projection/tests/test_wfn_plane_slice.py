# -*- coding: utf-8 -*-
"""
run_wfn_plane_slice tests: CHGCAR structure parsing (Direct/Cartesian,
scale factor), the best-fit vertical plane, periodic plane sampling
against an analytically known field, the atom overlay selection with
periodic images, default filename derivation, and the end-to-end run
producing a PNG.
"""
import os

import numpy as np
import pytest

import run_wfn_plane_slice as rps


LATTICE = np.diag([4.0, 5.0, 6.0])


def write_multiatom_chg(path, values, dims, lattice=LATTICE,
                        species=("Au", "C"), counts=(1, 2),
                        frac_coords=((0.25, 0.5, 0.1),
                                     (0.25, 0.5, 0.5),
                                     (0.75, 0.5, 0.5)),
                        scale=1.0, cartesian=False):
    values = np.asarray(values, dtype=np.float64)
    assert values.size == int(np.prod(dims))
    lines = ["slice test", f" {scale}"]
    for v in np.asarray(lattice) / scale:
        lines.append("  %18.12f %18.12f %18.12f" % tuple(v))
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(c) for c in counts))
    if cartesian:
        lines.append("Cartesian")
        for fc in frac_coords:
            cart = np.asarray(fc) @ np.asarray(lattice) / scale
            lines.append("  %18.12f %18.12f %18.12f" % tuple(cart))
    else:
        lines.append("DIRECT")
        for fc in frac_coords:
            lines.append("  %18.12f %18.12f %18.12f" % tuple(fc))
    lines.append(" ")
    lines.append("  %4d %4d %4d" % tuple(dims))
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
        for start in range(0, values.size, 5):
            row = values[start:start + 5]
            f.write(" ".join("%.10E" % v for v in row) + "\n")


@pytest.fixture()
def ramp_file(tmp_path):
    """Field f(r) = frac_x on an (8,10,12) grid — analytically known."""
    dims = (8, 10, 12)
    A = np.broadcast_to((np.arange(8) / 8.0)[:, None, None], dims)
    path = tmp_path / "ramp.vasp"
    write_multiatom_chg(str(path), A.flatten(order="F"), dims)
    return str(path), dims


def test_parse_structure_direct(ramp_file):
    path, dims = ramp_file
    with open(path, "rb") as f:
        header, natoms, got_dims = rps.read_chgcar_header(f)
    assert natoms == 3 and got_dims == dims
    s = rps.parse_structure(header)
    np.testing.assert_allclose(s["lattice"], LATTICE, rtol=1e-12)
    assert s["elements"] == ["Au", "C", "C"]
    np.testing.assert_allclose(s["cart"][0], [1.0, 2.5, 0.6], atol=1e-9)
    np.testing.assert_allclose(s["cart"][2], [3.0, 2.5, 3.0], atol=1e-9)
    assert s["volume"] == pytest.approx(120.0)


def test_parse_structure_cartesian_and_scale(tmp_path):
    dims = (2, 2, 2)
    path = tmp_path / "cart.vasp"
    write_multiatom_chg(str(path), np.zeros(8), dims, scale=2.0,
                        cartesian=True)
    with open(path, "rb") as f:
        header, _, _ = rps.read_chgcar_header(f)
    s = rps.parse_structure(header)
    np.testing.assert_allclose(s["lattice"], LATTICE, rtol=1e-12)
    np.testing.assert_allclose(s["cart"][0], [1.0, 2.5, 0.6], atol=1e-9)


def test_load_wfx_grid_order_and_volume(ramp_file):
    path, dims = ramp_file
    struct, data = rps.load_wfx_grid(path, divide_by_volume=False)
    assert data.shape == dims
    # x-fastest CHGCAR order: data[i,j,k] = i/8 everywhere
    np.testing.assert_allclose(data[3, :, :], 3 / 8.0, rtol=1e-9)
    _, data_v = rps.load_wfx_grid(path, divide_by_volume=True)
    np.testing.assert_allclose(data_v[3, :, :], 3 / 8.0 / 120.0, rtol=1e-9)


def test_plane_from_atoms_two_and_fit():
    cart = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 5.0],
                     [2.0, 1.1, 2.0]])
    origin, u = rps.plane_from_atoms(cart, [1, 2])
    np.testing.assert_allclose(origin, [2.0, 1.0], atol=1e-12)
    # canonical orientation: s runs from the first atom toward the last
    np.testing.assert_allclose(u, [1.0, 0.0], atol=1e-12)
    _, u_rev = rps.plane_from_atoms(cart, [2, 1])
    np.testing.assert_allclose(u_rev, [-1.0, 0.0], atol=1e-12)
    # three near-collinear atoms: best-fit stays essentially along x
    origin3, u3 = rps.plane_from_atoms(cart, [1, 2, 3])
    assert abs(u3[0]) > 0.99


def test_plane_from_atoms_errors():
    cart = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 3.0]])
    with pytest.raises(ValueError, match="at least two"):
        rps.plane_from_atoms(cart, [1])
    with pytest.raises(IndexError, match="out of range"):
        rps.plane_from_atoms(cart, [1, 5])
    with pytest.raises(ValueError, match="coincide in xy"):
        rps.plane_from_atoms(cart, [1, 2])   # same xy, different z


def test_sample_vertical_plane_matches_field(ramp_file):
    path, dims = ramp_file
    struct, data = rps.load_wfx_grid(path, divide_by_volume=False)
    origin, u = rps.plane_from_atoms(struct["cart"], [1, 3])  # along x
    S, Z, V = rps.sample_vertical_plane(data, struct["lattice"], origin,
                                        u, n_s=41, n_z=7)
    assert S.shape == V.shape == (41, 7)
    # full footprint along x through origin x=2: s in [-2, 2]
    assert S.min() == pytest.approx(-2.0) and S.max() == pytest.approx(2.0)
    assert Z.min() == pytest.approx(0.0) and Z.max() == pytest.approx(6.0)
    # V must reproduce frac_x = ((2 + s)/4) % 1 wherever the sawtooth
    # does not wrap (frac_x <= 7/8), independent of z.
    x = origin[0] + S[:, 0] * u[0]
    frac_x = (x / 4.0) % 1.0
    ok = frac_x <= 7 / 8.0
    np.testing.assert_allclose(V[ok, :], np.broadcast_to(
        frac_x[ok, None], (ok.sum(), 7)), atol=1e-9)
    # wrap region interpolates between 7/8 and the periodic image (0):
    # value stays within [0, 7/8], never extrapolates
    assert V.max() <= 7 / 8.0 + 1e-9 and V.min() >= -1e-9


def test_atoms_near_plane_selection_and_images():
    struct = dict(
        lattice=LATTICE,
        elements=["Au", "C"],
        cart=np.array([[1.0, 2.5, 0.6], [1.0, 4.4, 3.0]]),
    )
    origin, u = np.array([2.0, 2.5]), np.array([1.0, 0.0])
    atoms = rps.atoms_near_plane(struct, origin, u, 1.0, (-2.0, 2.0))
    # Au on the plane; C at y=4.4 is 1.9 away directly, but its -b image
    # sits at y=-0.6, i.e. 3.1 away — no image within 1.0 either.
    assert [a["elem"] for a in atoms] == ["Au"]
    assert atoms[0]["s"] == pytest.approx(-1.0)
    assert atoms[0]["z"] == pytest.approx(0.6)
    # widen the tolerance: the C atom (dist 1.9) now qualifies
    atoms = rps.atoms_near_plane(struct, origin, u, 2.0, (-2.0, 2.0))
    assert sorted(a["elem"] for a in atoms) == ["Au", "C"]
    # periodic images: atom at x=3.9 has candidate s values -2.1 (image
    # -a), 1.9 (original), 5.9 (image +a); only s=1.9 is inside the
    # range, so exactly one marker survives.
    struct2 = dict(lattice=LATTICE, elements=["H"],
                   cart=np.array([[3.9, 2.5, 1.0]]))
    atoms = rps.atoms_near_plane(struct2, origin, u, 0.5, (-2.0, 2.0))
    assert len(atoms) == 1
    assert atoms[0]["s"] == pytest.approx(1.9)


def test_atoms_near_plane_tilted_image_height():
    # a has a z-component: a periodic image's marker height must shift
    # with it, not stay at the original atom's z.
    lat = np.array([[4.0, 0.0, 1.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])
    struct = dict(lattice=lat, elements=["H"],
                  cart=np.array([[3.9, 2.5, 1.0]]))
    origin, u = np.array([0.0, 2.5]), np.array([1.0, 0.0])
    atoms = rps.atoms_near_plane(struct, origin, u, 0.5, (-2.0, 2.0))
    assert len(atoms) == 1                     # only the -a image is in s-range
    assert atoms[0]["s"] == pytest.approx(-0.1)
    assert atoms[0]["z"] == pytest.approx(0.0)  # 1.0 - a_z


def test_default_input_file():
    assert rps.default_input_file("d", "NHC_c1m1", 0, None).endswith(
        "NHC_c1m1_HOMO_wvfx_squared_probability.vasp")
    assert rps.default_input_file("d", "NHC_c1m1", -2, "top5").endswith(
        "NHC_c1m1_HOMO-2_top5_wvfx_squared_probability.vasp")


@pytest.mark.parametrize("norm_mode", ["log", "linear", "power"])
def test_run_end_to_end_saves_png(ramp_file, norm_mode):
    path, dims = ramp_file
    fig, ax, (S, Z, V) = rps.run(path, [1, 3], n_s=21, n_z=5,
                                 norm_mode=norm_mode, save_png=True)
    png = os.path.splitext(path)[0] + "_plane_slice.png"
    assert os.path.isfile(png)
    assert V.shape == (21, 5)
    # the gamma widgets belong to power mode only
    assert hasattr(fig, "_power_gamma_widgets") == (norm_mode == "power")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_unit_normalize():
    out = rps.unit_normalize(np.full((4, 5, 6), 2.0), 120.0)
    assert out.sum() * (120.0 / out.size) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rps.unit_normalize(np.zeros((2, 2, 2)), 8.0)
    # non-finite totals must fail loudly, not divide through silently
    with pytest.raises(ValueError):
        rps.unit_normalize(np.array([[[np.nan, 1.0]]]), 8.0)
    with pytest.raises(ValueError):
        rps.unit_normalize(np.array([[[np.inf, 1.0]]]), 8.0)


def test_run_multi_input_compare(ramp_file, tmp_path):
    # A second file that is the ramp scaled x2: after per-grid unit
    # normalization both panels must be identical, on one shared norm.
    path1, dims = ramp_file
    A = np.broadcast_to((np.arange(8) / 8.0)[:, None, None], dims)
    path2 = str(tmp_path / "ramp2x.vasp")
    write_multiatom_chg(path2, 2.0 * A.flatten(order="F"), dims)
    fig, axes, panels = rps.run([path1, path2], [1, 3], n_s=21, n_z=5,
                                norm_mode="power", save_png=True)
    assert len(axes) == 2 and len(panels) == 2
    for S, Z, V in panels:
        assert V.shape == (21, 5)
    np.testing.assert_allclose(panels[0][2], panels[1][2],
                               rtol=1e-9, atol=1e-12)
    # both meshes share the very same norm object (single colorbar scale)
    assert axes[0].collections[0].norm is axes[1].collections[0].norm
    # STM simulator power convention: PowerNorm(gamma, vmin=0, vmax=plain
    # max over the panels) on the mesh, the data untransformed.
    norm = axes[0].collections[0].norm
    assert norm.vmin == 0.0
    assert norm.vmax == pytest.approx(float(np.nanmax(panels[0][2])))
    M = np.asarray(axes[0].collections[0].get_array())
    np.testing.assert_array_equal(M.reshape(panels[0][2].shape),
                                  panels[0][2])
    png = os.path.splitext(path1)[0] + "_plane_slice_compare.png"
    assert os.path.isfile(png)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_multi_input_shared_scale_log(ramp_file, tmp_path):
    # A genuinely DIFFERENT second field makes the cross-panel scale
    # discriminating: with the constant listed first, the shared vmax
    # must still come from the ramp panel, and the constant panel must
    # be the absolute unit density 1/volume everywhere.
    path1, dims = ramp_file
    path2 = str(tmp_path / "const.vasp")
    write_multiatom_chg(path2, np.full(int(np.prod(dims)), 3.0), dims)
    fig, axes, panels = rps.run([path2, path1], [1, 3], n_s=21, n_z=5,
                                norm_mode="log", save_png=False)
    V_const, V_ramp = panels[0][2], panels[1][2]
    np.testing.assert_allclose(V_const, 1.0 / 120.0, rtol=1e-9)
    norm = axes[0].collections[0].norm
    vmax_all = max(float(np.nanmax(V_const)), float(np.nanmax(V_ramp)))
    assert norm.vmax == pytest.approx(vmax_all)
    assert norm.vmax > float(np.nanmax(V_const))    # from the 2nd panel
    pos = np.concatenate([V[V > 0].ravel() for V in (V_const, V_ramp)])
    assert norm.vmin == pytest.approx(max(float(pos.min()),
                                          vmax_all * 1e-8))
    # log mode masks non-positives on the mesh but returns the raw V
    M = axes[1].collections[0].get_array().reshape(V_ramp.shape)
    assert (V_ramp <= 0).any() and np.ma.is_masked(M)
    np.testing.assert_allclose(np.ma.filled(M, -1.0),
                               np.where(V_ramp > 0, V_ramp, -1.0),
                               rtol=1e-12)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_multi_input_heterogeneous_defaults(ramp_file, tmp_path, capsys):
    # Second file with different grid dims, a different lattice, and
    # different atom positions: the plane, the default sample counts,
    # and the mismatch warning must all come from the FIRST input.
    path1, dims1 = ramp_file                    # (8, 10, 12)
    dims2 = (6, 7, 9)
    lat2 = np.diag([4.0, 5.0, 6.5])
    path2 = str(tmp_path / "other.vasp")
    write_multiatom_chg(path2, np.full(int(np.prod(dims2)), 1.0), dims2,
                        lattice=lat2,
                        frac_coords=((0.5, 0.2, 0.1), (0.1, 0.9, 0.5),
                                     (0.9, 0.1, 0.5)))
    fig, axes, panels = rps.run([path1, path2], [1, 3], save_png=False,
                                norm_mode="linear")
    out = capsys.readouterr().out
    assert "WARNING: lattice of other.vasp differs" in out
    # plane from the FIRST input's atoms 1 and 3: origin (2.0, 2.5)
    assert "origin (2.000, 2.500)" in out
    # n_s/n_z=None resolved ONCE from the first grid: (max(8,10), 12)
    for S, Z, V in panels:
        assert V.shape == (10, 12)
    # each panel overlays its OWN structure's atoms: the first file has
    # all 3 atoms on the plane (mesh + 3 scatters); the second file's
    # atoms all sit farther than tol from it (mesh only)
    assert len(axes[0].collections) == 4
    assert len(axes[1].collections) == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_parse_cli_args():
    paths, idx = rps._parse_cli_args(["a.vasp", "none", "12", "b.vasp", "7"])
    assert paths == ["a.vasp", None, "b.vasp"]
    assert idx == [12, 7]
    assert rps._parse_cli_args([]) == ([], [])


def test_power_gamma_widgets(ramp_file):
    path, dims = ramp_file
    fig, ax, _ = rps.run(path, [1, 3], n_s=21, n_z=5,
                         norm_mode="power", save_png=False)
    sld, tb = fig._power_gamma_widgets
    pcm = ax.collections[0]
    assert pcm.norm.gamma == pytest.approx(rps.power_gamma)
    sld.set_val(1.5)
    assert pcm.norm.gamma == pytest.approx(1.5)
    assert tb.text == "1.50"               # box mirrors the slider
    tb.set_val("2.75")   # above the slider range — allowed via the box
    assert pcm.norm.gamma == pytest.approx(2.75)
    assert sld.val == pytest.approx(2.0)   # slider display pegs at its max
    assert tb.text == "2.75"               # box keeps the true gamma
    for bad in ("not a number", "nan", "inf", "-1", "0"):
        tb.set_val(bad)                    # invalid: silently reverted
        assert pcm.norm.gamma == pytest.approx(2.75)
        assert tb.text == "2.75"
    import matplotlib.pyplot as plt
    plt.close(fig)
