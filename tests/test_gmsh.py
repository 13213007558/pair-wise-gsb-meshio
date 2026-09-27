import copy
import pathlib
from functools import partial

import numpy as np
import pytest

import meshio

from . import helpers


def gmsh_periodic():
    mesh = copy.deepcopy(helpers.quad_mesh)
    trns = [0] * 16  # just for io testing
    mesh.gmsh_periodic = [
        [0, (3, 1), None, [[2, 0]]],
        [0, (4, 6), None, [[3, 5]]],
        [1, (2, 1), trns, [[5, 0], [4, 1], [4, 2]]],
    ]
    return mesh


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        # helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.tri_mesh, 9),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        gmsh_periodic(),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh22(mesh, binary, tmp_path):
    writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=binary)
    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        # helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.tri_mesh, 9),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh40(mesh, binary, tmp_path):
    writer = partial(meshio.gmsh.write, fmt_version="4.0", binary=binary)

    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        # helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.tri_mesh, 9),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        gmsh_periodic(),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41(mesh, binary, tmp_path):
    writer = partial(meshio.gmsh.write, fmt_version="4.1", binary=binary)
    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.msh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.msh")


PARTITIONED_MSH = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PartitionedEntities
2
1
1 3 2
0 1 1 0
3 2 7 1 1 0 0 0 0 0 0 0 0
7 0 0 2 1 2 0 0 0 0 0 0 0 1 3
$EndPartitionedEntities
$Nodes
1 4 1 4
2 7 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
1 2 5 9
2 7 2 2
5 1 2 3
9 1 3 4
$EndElements
$GhostElements
1
9 2 1 1
$EndGhostElements
"""


def test_gmsh41_read_partitioned(tmp_path):
    p = tmp_path / "partitioned.msh"
    p.write_text(PARTITIONED_MSH)
    mesh = meshio.gmsh.read(p)

    assert mesh.gmsh_partitioned_entities == {
        "num_partitions": 2,
        "ghost_entities": [(1, 3, 2)],
        "entities": {
            (1, 3): {"parent": (2, 7), "partitions": [1]},
            (2, 7): {"parent": (0, 0), "partitions": [1, 2]},
        },
    }
    assert mesh.gmsh_ghost_elements == {9: (2, [1])}
    # the element tags are sparse and must be preserved
    assert len(mesh.cell_data["gmsh:element_tags"]) == 1
    assert list(mesh.cell_data["gmsh:element_tags"][0]) == [5, 9]
    assert [c.type for c in mesh.cells] == ["triangle"]
    assert np.array_equal(mesh.cells[0].data, [[0, 1, 2], [0, 2, 3]])

    # round-trip through both ascii and binary
    for binary in [False, True]:
        out = tmp_path / f"out-{binary}.msh"
        meshio.gmsh.write(out, mesh, fmt_version="4.1", binary=binary)
        mesh2 = meshio.gmsh.read(out)
        assert mesh2.gmsh_partitioned_entities == mesh.gmsh_partitioned_entities
        assert mesh2.gmsh_ghost_elements == mesh.gmsh_ghost_elements
        assert list(mesh2.cell_data["gmsh:element_tags"][0]) == [5, 9]
        assert np.array_equal(mesh2.cells[0].data, mesh.cells[0].data)


def _partitioned_mesh():
    mesh = meshio.Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        cells=[
            ("triangle", [[0, 1, 2], [0, 2, 3]]),
            ("triangle", [[1, 4, 5], [1, 5, 2]]),
        ],
        point_data={"gmsh:dim_tags": [[2, 1]] * 3 + [[2, 2]] * 3},
        cell_data={
            "gmsh:geometrical": [[1, 1], [2, 2]],
            # sparse, non-contiguous and unordered across cell blocks
            "gmsh:element_tags": [[10, 30], [5, 7]],
        },
    )
    mesh.gmsh_partitioned_entities = {
        "num_partitions": 3,
        "ghost_entities": [(2, 2, 3)],
        "entities": {
            # one entity shared by several partitions
            (2, 1): {"parent": (0, 0), "partitions": [1, 2]},
            (2, 2): {"parent": (2, 1), "partitions": [2, 3]},
        },
    }
    mesh.gmsh_ghost_elements = {30: (2, [1]), 5: (3, [1, 2])}
    return mesh


@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41_partitioned_roundtrip(binary, tmp_path):
    mesh = _partitioned_mesh()
    out = tmp_path / "test.msh"
    meshio.gmsh.write(out, mesh, fmt_version="4.1", binary=binary)
    mesh2 = meshio.gmsh.read(out)

    assert mesh2.gmsh_partitioned_entities == mesh.gmsh_partitioned_entities
    assert mesh2.gmsh_ghost_elements == mesh.gmsh_ghost_elements
    for tags0, tags1 in zip(
        mesh.cell_data["gmsh:element_tags"], mesh2.cell_data["gmsh:element_tags"]
    ):
        assert list(tags0) == list(tags1)
    for c0, c1 in zip(mesh.cells, mesh2.cells):
        assert c0.type == c1.type
        assert np.array_equal(c0.data, c1.data)


@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41_partitioned_invalid_parent(binary, tmp_path):
    mesh = _partitioned_mesh()
    # parent entity (2, 99) does not exist
    mesh.gmsh_partitioned_entities["entities"][(2, 2)]["parent"] = (2, 99)
    with pytest.raises(meshio.WriteError):
        meshio.gmsh.write(tmp_path / "bad.msh", mesh, fmt_version="4.1", binary=binary)


@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41_ghost_unknown_element(binary, tmp_path):
    mesh = _partitioned_mesh()
    # element tag 12345 does not exist
    mesh.gmsh_ghost_elements = {12345: (1, [2])}
    with pytest.raises(meshio.WriteError):
        meshio.gmsh.write(tmp_path / "bad.msh", mesh, fmt_version="4.1", binary=binary)


@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41_duplicate_element_tags(binary, tmp_path):
    mesh = _partitioned_mesh()
    mesh.cell_data["gmsh:element_tags"] = [[10, 10], [5, 7]]
    with pytest.raises(meshio.WriteError):
        meshio.gmsh.write(tmp_path / "bad.msh", mesh, fmt_version="4.1", binary=binary)


@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41_unpartitioned_untouched(binary, tmp_path):
    # ordinary meshes must not pick up any partitioning data
    out = tmp_path / "test.msh"
    meshio.gmsh.write(out, helpers.tri_mesh, fmt_version="4.1", binary=binary)
    mesh = meshio.gmsh.read(out)
    assert mesh.gmsh_partitioned_entities is None
    assert mesh.gmsh_ghost_elements is None
    assert "gmsh:element_tags" not in mesh.cell_data


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("insulated-2.2.msh", 2.001762136876221, [21, 111])],
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file(filename, ref_sum, ref_num_cells, binary, tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "msh" / filename
    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum) < tol * ref_sum
    assert [c.type for c in mesh.cells] == ["line", "triangle"]
    assert [len(c.data) for c in mesh.cells] == ref_num_cells
    assert list(map(len, mesh.cell_data["gmsh:geometrical"])) == ref_num_cells
    assert list(map(len, mesh.cell_data["gmsh:physical"])) == ref_num_cells

    writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=binary)
    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells, ref_num_cells_in_cell_sets",
    [
        (
            "insulated-4.1.msh",
            2.001762136876221,
            {"line": 21, "triangle": 111},
            {"line": 27, "triangle": 120},
        )
    ],
    # Note that testing on number of cells in
    # cell_sets_dict will count both cells associated with physical tags, and
    # bounding entities.
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file_with_entities(
    filename, ref_sum, ref_num_cells, ref_num_cells_in_cell_sets, binary, tmp_path
):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "msh" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum) < tol * ref_sum
    assert {k: len(v) for k, v in mesh.cells_dict.items()} == ref_num_cells
    assert {
        k: len(v) for k, v in mesh.cell_data_dict["gmsh:physical"].items()
    } == ref_num_cells

    writer = partial(meshio.gmsh.write, fmt_version="4.1", binary=binary)

    num_cells = {k: 0 for k in ref_num_cells_in_cell_sets}
    for vv in mesh.cell_sets_dict.values():
        for k, v in vv.items():
            num_cells[k] += len(v)
    assert num_cells == ref_num_cells_in_cell_sets

    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)
