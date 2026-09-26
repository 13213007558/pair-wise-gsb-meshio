import pytest

import meshio
import numpy as np

from . import helpers

netCDF4 = pytest.importorskip("netCDF4")

test_set = [
    helpers.empty_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_point_sets(helpers.tri_mesh),
    helpers.add_point_sets(helpers.tet_mesh),
]


@pytest.mark.parametrize("mesh", test_set)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.exodus.write, meshio.exodus.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.e")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.e")


def _two_block_tet10_mesh():
    # Two tetra10 blocks. The first block has one cell, the second has two;
    # the first cell of the second block shares the face (0, 1, 3) with the
    # cell of the first block, with opposite orientation.
    points = [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.5, 0.0, 0.0],  # 4
        [0.5, 0.5, 0.0],  # 5
        [0.0, 0.5, 0.0],  # 6
        [0.0, 0.0, 0.5],  # 7
        [0.5, 0.0, 0.5],  # 8
        [0.0, 0.5, 0.5],  # 9
        [0.3, -1.0, 0.3],  # 10
        [0.15, -0.5, 0.15],  # 11
        [0.65, -0.5, 0.15],  # 12
        [0.15, -0.5, 0.65],  # 13
        # second cell of block 1: shifted copy of the first
        [0.0, -2.0, 0.0],  # 14
        [1.0, -2.0, 0.0],  # 15
        [0.0, -2.0, 1.0],  # 16
        [0.3, -3.0, 0.3],  # 17
        [0.5, -2.0, 0.0],  # 18
        [0.5, -2.0, 0.5],  # 19
        [0.0, -2.0, 0.5],  # 20
        [0.15, -2.5, 0.15],  # 21
        [0.65, -2.5, 0.15],  # 22
        [0.15, -2.5, 0.65],  # 23
    ]
    cells = [
        ("tetra10", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (
            "tetra10",
            [
                [0, 1, 3, 10, 4, 8, 7, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            ],
        ),
    ]
    side_sets = {
        "wall": meshio.SideSet(
            # (local cell, local face) per block; the two faces are the same
            # geometric face, seen from opposite sides
            [np.array([[0, 0]]), np.array([[0, 3]])],
            id=7,
            dist_facts=[
                [np.arange(0.1, 0.7, 0.1)],
                [np.arange(1.1, 1.7, 0.1)],
            ],
        )
    }
    return meshio.Mesh(points, cells, side_sets=side_sets)


def test_side_sets_roundtrip(tmp_path):
    mesh = _two_block_tet10_mesh()
    filepath = tmp_path / "test.e"
    meshio.exodus.write(filepath, mesh)
    mesh2 = meshio.exodus.read(filepath)

    assert set(mesh2.side_sets.keys()) == {"wall"}
    side_set = mesh2.side_sets["wall"]
    assert side_set.id == 7
    assert len(side_set.data) == 2
    assert np.array_equal(side_set.data[0], [[0, 0]])
    assert np.array_equal(side_set.data[1], [[0, 3]])

    faces = mesh2.get_side_set_face_nodes("wall")
    assert len(faces) == 2
    # oriented face nodes, including the higher-order edge nodes
    assert np.array_equal(faces[0], [0, 1, 3, 4, 8, 7])
    assert np.array_equal(faces[1], [0, 3, 1, 7, 8, 4])
    # same geometric face, opposite orientation
    assert set(faces[0]) == set(faces[1])

    assert np.allclose(side_set.dist_facts[0][0], np.arange(0.1, 0.7, 0.1))
    assert np.allclose(side_set.dist_facts[1][0], np.arange(1.1, 1.7, 0.1))


def test_side_sets_block_reorder(tmp_path):
    mesh = _two_block_tet10_mesh()
    faces_before = {tuple(sorted(f)) for f in mesh.get_side_set_face_nodes("wall")}

    # reorder the cell blocks; the side set has to be reordered along
    side_set = mesh.side_sets["wall"]
    reordered = meshio.Mesh(
        mesh.points,
        [mesh.cells[1], mesh.cells[0]],
        side_sets={
            "wall": meshio.SideSet(
                [side_set.data[1], side_set.data[0]],
                id=side_set.id,
                dist_facts=[side_set.dist_facts[1], side_set.dist_facts[0]],
            )
        },
    )
    filepath = tmp_path / "test.e"
    meshio.exodus.write(filepath, reordered)
    mesh2 = meshio.exodus.read(filepath)

    # the blocks have different sizes (2 and 1 cells), so the global element
    # numbers in the file differ; the set must still point to the same faces
    side_set2 = mesh2.side_sets["wall"]
    assert side_set2.id == 7
    assert np.array_equal(side_set2.data[0], [[0, 3]])
    assert np.array_equal(side_set2.data[1], [[0, 0]])
    faces_after = {tuple(sorted(f)) for f in mesh2.get_side_set_face_nodes("wall")}
    assert faces_before == faces_after
    assert np.allclose(side_set2.dist_facts[0][0], np.arange(1.1, 1.7, 0.1))
    assert np.allclose(side_set2.dist_facts[1][0], np.arange(0.1, 0.7, 0.1))


def test_side_sets_mixed_face_sizes(tmp_path):
    # hex20 block and wedge block; the wedge contributes a quadrilateral and
    # a triangular face with different numbers of face nodes
    points = [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
        [0.5, 0.0, 0.0],  # 8
        [1.0, 0.5, 0.0],  # 9
        [0.5, 1.0, 0.0],  # 10
        [0.0, 0.5, 0.0],  # 11
        [0.5, 0.0, 1.0],  # 12
        [1.0, 0.5, 1.0],  # 13
        [0.5, 1.0, 1.0],  # 14
        [0.0, 0.5, 1.0],  # 15
        [0.0, 0.0, 0.5],  # 16
        [1.0, 0.0, 0.5],  # 17
        [1.0, 1.0, 0.5],  # 18
        [0.0, 1.0, 0.5],  # 19
        [2.0, 0.0, 0.0],  # 20
        [3.0, 0.0, 0.0],  # 21
        [2.0, 1.0, 0.0],  # 22
        [2.0, 0.0, 1.0],  # 23
        [3.0, 0.0, 1.0],  # 24
        [2.0, 1.0, 1.0],  # 25
    ]
    cells = [
        ("hexahedron20", [list(range(20))]),
        ("wedge", [[20, 21, 22, 23, 24, 25]]),
    ]
    side_sets = {
        "bc": meshio.SideSet(
            [np.array([[0, 5]]), np.array([[0, 0], [0, 3]])],
            id=3,
            dist_facts=[
                [np.full(8, 0.5)],
                [np.full(4, 1.5), np.full(3, 2.5)],
            ],
        )
    }
    mesh = meshio.Mesh(points, cells, side_sets=side_sets)

    filepath = tmp_path / "test.e"
    meshio.exodus.write(filepath, mesh)
    mesh2 = meshio.exodus.read(filepath)

    side_set = mesh2.side_sets["bc"]
    assert side_set.id == 3
    faces = mesh2.get_side_set_face_nodes("bc")
    assert np.array_equal(faces[0], [4, 5, 6, 7, 12, 13, 14, 15])
    assert np.array_equal(faces[1], [20, 21, 24, 23])
    assert np.array_equal(faces[2], [20, 22, 21])
    assert np.allclose(side_set.dist_facts[0][0], np.full(8, 0.5))
    assert np.allclose(side_set.dist_facts[1][0], np.full(4, 1.5))
    assert np.allclose(side_set.dist_facts[1][1], np.full(3, 2.5))


def test_side_sets_dist_facts_mismatch(tmp_path):
    mesh = _two_block_tet10_mesh()
    side_set = mesh.side_sets["wall"]

    # 5 factors for a 6-node face
    with pytest.raises(ValueError, match="wall"):
        meshio.Mesh(
            mesh.points,
            mesh.cells,
            side_sets={
                "wall": meshio.SideSet(
                    side_set.data,
                    id=7,
                    dist_facts=[[np.zeros(5)], []],
                )
            },
        )

    # mutating a valid mesh must be caught at write time, too
    mesh.side_sets["wall"].dist_facts[1][0] = np.zeros(3)
    with pytest.raises(meshio.WriteError, match="wall"):
        meshio.exodus.write(tmp_path / "test.e", mesh)
