import pathlib

import numpy as np
import pytest

import meshio

from . import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh_2d,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.hex_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.med")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.med")


def test_reference_file_with_mixed_cells(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    mesh = meshio.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 16.53169892762988)

    # CellBlock
    ref_num_cells = {"pyramid": 18, "quad": 18, "line": 17, "tetra": 63, "triangle": 4}
    assert {
        cell_block.type: len(cell_block) for cell_block in mesh.cells
    } == ref_num_cells

    # Point tags
    assert mesh.point_data["point_tags"].sum() == 52
    ref_point_tags_info = {2: ["Side"], 3: ["Side", "Top"], 4: ["Top"]}
    assert mesh.point_tags == ref_point_tags_info

    # Cell tags
    ref_sum_cell_tags = {
        "pyramid": -116,
        "quad": -75,
        "line": -48,
        "tetra": -24,
        "triangle": -30,
    }
    assert {
        c.type: sum(d) for c, d in zip(mesh.cells, mesh.cell_data["cell_tags"])
    } == ref_sum_cell_tags
    ref_cell_tags_info = {
        -6: ["Top circle"],
        -7: ["Top", "Top and down"],
        -8: ["Top and down"],
        -9: ["A", "B"],
        -10: ["B"],
        -11: ["B", "C"],
        -12: ["C"],
    }
    assert mesh.cell_tags == ref_cell_tags_info

    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def test_reference_file_with_point_cell_data(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "box.med"

    mesh = meshio.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 12)

    # CellBlock
    assert {cell_block.type: len(cell_block) for cell_block in mesh.cells} == {
        "hexahedron": 1
    }

    # Point data
    data_u = mesh.point_data["resu____DEPL"]
    assert data_u.shape == (8, 3)
    assert np.isclose(data_u.sum(), 12)

    # Cell data
    # ELNO (1 data point for every node of each element)
    data_eps = mesh.cell_data["resu____EPSI_ELNO"][0]
    assert data_eps.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_eps_mean = np.mean(data_eps, axis=1)[0]
    eps_ref = np.array([1, 0, 0, 0.5, 0.5, 0])
    assert np.allclose(data_eps_mean, eps_ref)

    data_sig = mesh.cell_data["resu____SIEF_ELNO"][0]
    assert data_sig.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_sig_mean = np.mean(data_sig, axis=1)[0]
    sig_ref = np.array(
        [7328.44611253, 2645.87030114, 2034.06063679, 1202.6, 569.752, 0]
    )
    assert np.allclose(data_sig_mean, sig_ref)

    data_psi = mesh.cell_data["resu____ENEL_ELNO"][0]
    assert data_psi.shape == (1, 8, 1)  # (n_cells, n_nodes_per_element, n_components)

    # ELEM (1 data point for each element)
    data_psi_elem = mesh.cell_data["resu____ENEL_ELEM"][0]
    assert np.isclose(np.mean(data_psi, axis=1)[0, 0], data_psi_elem[0])

    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def _elga_mesh():
    """Mesh with a triangle and a tetrahedron block, two Gauss localizations,
    and ELGA fields referring to them."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 1.0],
        ]
    )
    cells = [
        ("triangle", [[0, 1, 2], [0, 2, 3]]),
        ("tetra", [[0, 1, 2, 4], [1, 2, 3, 4]]),
    ]

    rng = np.random.default_rng(0)
    # triangle rule with 3 Gauss points (as many as nodes, must not be
    # mistaken for ELNO data), tetrahedron rule with 4 Gauss points
    tri_data = rng.random((2, 3, 2))
    tet_data = rng.random((2, 4, 2))
    tri_data2 = rng.random((2, 3))
    tet_data2 = rng.random((2, 4))
    cell_data = {
        "sig": [tri_data, tet_data],
        "temp": [tri_data2, tet_data2],
    }

    localizations = {
        "tri_gauss": {
            "cell_type": "triangle",
            "ref_nodes": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "points": np.array([[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]]),
            "weights": np.array([0.1, 0.2, 0.3]),
        },
        "tet_gauss": {
            "cell_type": "tetra",
            "ref_nodes": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "points": np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.7, 0.1, 0.1],
                    [0.1, 0.7, 0.1],
                    [0.1, 0.1, 0.7],
                ]
            ),
            "weights": np.array([0.4, 0.3, 0.2, 0.1]),
        },
    }
    cell_data_localizations = {
        "sig": ["tri_gauss", "tet_gauss"],
        "temp": ["tri_gauss", "tet_gauss"],
    }

    mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    mesh.localizations = localizations
    mesh.cell_data_localizations = cell_data_localizations
    return mesh


def test_elga_localization_roundtrip(tmp_path):
    mesh_in = _elga_mesh()
    filepath = tmp_path / "test.med"
    meshio.med.write(filepath, mesh_in)

    # check the file layout: ELGA supports must point at the localizations
    with h5py.File(filepath, "r") as f:
        assert set(f["GAUSS"].keys()) == {"tri_gauss", "tet_gauss"}
        step = list(f["CHA"]["sig"].keys())[0]
        tri_supp = f["CHA"]["sig"][step]["MAI.TR3"]
        assert tri_supp.attrs["GAU"].decode() == "tri_gauss"
        assert tri_supp["MED_NO_PROFILE_INTERNAL"].attrs["NGA"] == 3
        tet_supp = f["CHA"]["sig"][step]["MAI.TE4"]
        assert tet_supp.attrs["GAU"].decode() == "tet_gauss"

    mesh = meshio.med.read(filepath)

    # localizations: reference nodes, Gauss points (order), and weights
    for name, loc in mesh_in.localizations.items():
        out = mesh.localizations[name]
        assert out["cell_type"] == loc["cell_type"]
        assert np.allclose(out["ref_nodes"], loc["ref_nodes"])
        assert np.allclose(out["points"], loc["points"])
        assert np.allclose(out["weights"], loc["weights"])

    # fields explicitly refer to the localizations (block order may change
    # in the file, so compare per cell type)
    for name in mesh_in.cell_data_localizations:
        ref = dict(zip([c.type for c in mesh_in.cells], mesh_in.cell_data_localizations[name]))
        out = dict(zip([c.type for c in mesh.cells], mesh.cell_data_localizations[name]))
        assert out == ref

    # field values
    for name, d in mesh_in.cell_data.items():
        ref = dict(zip([c.type for c in mesh_in.cells], d))
        out = dict(zip([c.type for c in mesh.cells], mesh.cell_data[name]))
        for cell_type in ref:
            assert np.allclose(out[cell_type], ref[cell_type])


def test_elga_missing_localization(tmp_path):
    mesh = _elga_mesh()
    mesh.cell_data_localizations["sig"][0] = "no_such_rule"
    with pytest.raises(meshio.WriteError, match="sig"):
        meshio.med.write(tmp_path / "test.med", mesh)


def test_elga_incompatible_localization(tmp_path):
    mesh = _elga_mesh()
    # wrong number of Gauss points for the field data
    mesh.localizations["tri_gauss"]["weights"] = np.array([0.5, 0.5])
    with pytest.raises(meshio.WriteError, match="sig"):
        meshio.med.write(tmp_path / "test.med", mesh)

    # reference element does not match the cell type
    mesh = _elga_mesh()
    mesh.cell_data_localizations["sig"][0] = "tet_gauss"
    with pytest.raises(meshio.WriteError, match="sig"):
        meshio.med.write(tmp_path / "test.med", mesh)


def test_elga_data_without_localization(tmp_path):
    mesh = helpers.add_cell_data(helpers.tri_mesh, [("a", (5, 2), np.float64)])
    with pytest.raises(meshio.WriteError, match="a"):
        meshio.med.write(tmp_path / "test.med", mesh)


def test_elno_still_written_without_localization(tmp_path):
    # data at every node of each cell, no localization: written as ELNO
    mesh = helpers.add_cell_data(helpers.tri_mesh, [("a", (3, 2), np.float64)])
    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)
