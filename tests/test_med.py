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


# Reference element nodes and quadrature rules used in the ELGA tests
tri_ref_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
tri_gauss_points = np.array(
    [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]]
)
tri_weights = np.full(3, 1.0 / 6.0)

tet_ref_points = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_a = 0.585410196624969
_b = 0.138196601125011
tet_gauss_points = np.array(
    [[_a, _b, _b], [_b, _a, _b], [_b, _b, _a], [_b, _b, _b]]
)
tet_weights = np.full(4, 1.0 / 24.0)


def _elga_localizations():
    return {
        "tri_rule": meshio.GaussLocalization(
            "triangle", tri_ref_points, tri_gauss_points, tri_weights
        ),
        "tet_rule": meshio.GaussLocalization(
            "tetra", tet_ref_points, tet_gauss_points, tet_weights
        ),
    }


def _elga_mesh():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = [
        ("triangle", np.array([[0, 1, 2], [1, 3, 2]])),
        ("tetra", np.array([[0, 1, 2, 4]])),
    ]
    rng = np.random.default_rng(0)
    # Two fields sharing the same two localizations; each cell block uses its
    # own rule. Note that the triangle rule has as many Gauss points as the
    # cell has nodes, so it must not be mistaken for ELNO data.
    cell_data = {
        "sigma": [rng.random((2, 3, 2)), rng.random((1, 4, 2))],
        "tau": [rng.random((2, 3, 1)), rng.random((1, 4, 1))],
    }
    cell_data_localizations = {
        "sigma": ["tri_rule", "tet_rule"],
        "tau": ["tri_rule", "tet_rule"],
    }
    return meshio.Mesh(
        points,
        cells,
        cell_data=cell_data,
        gauss_localizations=_elga_localizations(),
        cell_data_localizations=cell_data_localizations,
    )


def test_elga_roundtrip(tmp_path):
    mesh = _elga_mesh()
    filepath = tmp_path / "elga.med"
    meshio.med.write(filepath, mesh)

    # The ELGA fields must be stored as MAI.<type> with a reference to the
    # localization, not as ELNO (NOE.<type>) data
    with h5py.File(filepath, "r") as f:
        assert set(f["GAUSS"].keys()) == {"tri_rule", "tet_rule"}
        for field_name in ("sigma", "tau"):
            (step,) = f["CHA"][field_name].keys()
            field = f["CHA"][field_name][step]
            assert "NOE.TR3" not in field
            assert "NOE.TE4" not in field
            assert field["MAI.TR3"].attrs["GAU"].decode() == "tri_rule"
            assert field["MAI.TE4"].attrs["GAU"].decode() == "tet_rule"

    out = meshio.med.read(filepath)

    # Localizations: name, reference nodes, Gauss points (order!), weights
    assert set(out.gauss_localizations) == {"tri_rule", "tet_rule"}
    tri_loc = out.gauss_localizations["tri_rule"]
    assert tri_loc.cell_type == "triangle"
    assert np.allclose(tri_loc.reference_points, tri_ref_points)
    assert np.allclose(tri_loc.gauss_points, tri_gauss_points)
    assert np.allclose(tri_loc.weights, tri_weights)
    tet_loc = out.gauss_localizations["tet_rule"]
    assert tet_loc.cell_type == "tetra"
    assert np.allclose(tet_loc.reference_points, tet_ref_points)
    assert np.allclose(tet_loc.gauss_points, tet_gauss_points)
    assert np.allclose(tet_loc.weights, tet_weights)

    # Fields explicitly refer to their localizations. The cell block order
    # may differ from the input order, so compare per cell block type.
    loc_of_type = {"triangle": "tri_rule", "tetra": "tet_rule"}
    for name in ("sigma", "tau"):
        for cell_block, loc_name in zip(out.cells, out.cell_data_localizations[name]):
            assert loc_name == loc_of_type[cell_block.type]

    # Field values
    for name in ("sigma", "tau"):
        data_in = dict(zip((c.type for c in mesh.cells), mesh.cell_data[name]))
        for cell_block, data_out in zip(out.cells, out.cell_data[name]):
            assert data_in[cell_block.type].shape == data_out.shape
            assert np.allclose(data_in[cell_block.type], data_out)


def test_elga_missing_localization(tmp_path):
    # 2 points per triangle cell is neither ELNO nor ELEM; without an
    # associated localization the field must not be skipped silently
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    cell_data = {"sigma": [np.ones((1, 2, 1))]}
    mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    with pytest.raises(meshio.WriteError, match="sigma"):
        meshio.med.write(tmp_path / "test.med", mesh)


def test_elga_unknown_localization(tmp_path):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    cell_data = {"sigma": [np.ones((1, 3, 1))]}
    with pytest.raises(ValueError, match="no_such_rule"):
        meshio.Mesh(
            points,
            cells,
            cell_data=cell_data,
            cell_data_localizations={"sigma": ["no_such_rule"]},
        )


def test_elga_incompatible_localization(tmp_path):
    # The localization has 4 Gauss points, the data only 3
    mesh = _elga_mesh()
    mesh.gauss_localizations["tri_rule"] = meshio.GaussLocalization(
        "triangle", tri_ref_points, tet_gauss_points[:, :2], tet_weights
    )
    with pytest.raises(meshio.WriteError, match="sigma"):
        meshio.med.write(tmp_path / "test.med", mesh)

    # The localization is defined for the wrong cell type
    mesh = _elga_mesh()
    mesh.cell_data_localizations = {
        "sigma": ["tet_rule", "tet_rule"],
        "tau": ["tri_rule", "tet_rule"],
    }
    with pytest.raises(meshio.WriteError, match="sigma"):
        meshio.med.write(tmp_path / "test.med", mesh)


def test_elno_without_localization(tmp_path):
    # Data with one value per cell node and no localization reference is
    # still written as ELNO
    mesh = _elga_mesh()
    mesh.cell_data = {"sigma": [mesh.cell_data["sigma"][0]]}
    mesh.cells = mesh.cells[:1]
    mesh.gauss_localizations = {}
    mesh.cell_data_localizations = {}
    filepath = tmp_path / "elno.med"
    meshio.med.write(filepath, mesh)
    with h5py.File(filepath, "r") as f:
        assert "GAUSS" not in f
        (step,) = f["CHA"]["sigma"].keys()
        assert "NOE.TR3" in f["CHA"]["sigma"][step]
    out = meshio.med.read(filepath)
    assert out.gauss_localizations == {}
    assert out.cell_data_localizations == {}
    assert np.allclose(out.cell_data["sigma"][0], mesh.cell_data["sigma"][0])
