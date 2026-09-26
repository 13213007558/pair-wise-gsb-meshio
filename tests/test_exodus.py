import numpy as np
import pytest

import meshio
from meshio._exceptions import ReadError

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


def _side_set_mesh():
    # Two tetra10 sharing a face (first block) and one hexahedron20 (second
    # block). The shared face appears in the side set from both elements, i.e.,
    # with opposite orientations.
    points = [
        # tet A corners
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        # tet A edge nodes
        [0.5, 0.0, 0.0],  # 4
        [0.5, 0.5, 0.0],  # 5
        [0.0, 0.5, 0.0],  # 6
        [0.0, 0.0, 0.5],  # 7
        [0.5, 0.0, 0.5],  # 8
        [0.0, 0.5, 0.5],  # 9
        # tet B extra corner
        [1.0, 1.0, 1.0],  # 10
        # tet B extra edge nodes
        [0.5, 0.5, 0.5],  # 11
        [1.0, 0.5, 0.5],  # 12
        [0.5, 0.5, 1.0],  # 13
        # hex corners
        [2.0, 0.0, 0.0],  # 14
        [3.0, 0.0, 0.0],  # 15
        [3.0, 1.0, 0.0],  # 16
        [2.0, 1.0, 0.0],  # 17
        [2.0, 0.0, 1.0],  # 18
        [3.0, 0.0, 1.0],  # 19
        [3.0, 1.0, 1.0],  # 20
        [2.0, 1.0, 1.0],  # 21
        # hex edge nodes
        [2.5, 0.0, 0.0],  # 22
        [3.0, 0.5, 0.0],  # 23
        [2.5, 1.0, 0.0],  # 24
        [2.0, 0.5, 0.0],  # 25
        [2.5, 0.0, 1.0],  # 26
        [3.0, 0.5, 1.0],  # 27
        [2.5, 1.0, 1.0],  # 28
        [2.0, 0.5, 1.0],  # 29
        [2.0, 0.0, 0.5],  # 30
        [3.0, 0.0, 0.5],  # 31
        [3.0, 1.0, 0.5],  # 32
        [2.0, 1.0, 0.5],  # 33
    ]
    tet_a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # shares the face (0, 1, 3) with tet A
    tet_b = [0, 1, 3, 10, 4, 8, 7, 11, 12, 13]
    hex20 = list(range(14, 34))
    cells = [
        ("tetra10", [tet_a, tet_b]),
        ("hexahedron20", [hex20]),
    ]
    empty = np.empty((0, 2), dtype=int)
    side_sets = {
        # crosses both blocks; the first two sides are the same geometric face
        # seen from the two adjacent tets (opposite orientation)
        "wall": meshio.SideSet(
            [np.array([[0, 0], [1, 3]]), np.array([[0, 5]])],
            id=7,
            distribution_factors=np.linspace(0.1, 2.0, 20),
        ),
        "top": meshio.SideSet([empty, np.array([[0, 4]])], id=3),
    }
    return meshio.Mesh(points, cells, side_sets=side_sets)


def _assert_side_sets_equal(mesh0, mesh1):
    assert set(mesh0.side_sets.keys()) == set(mesh1.side_sets.keys())
    for name in mesh0.side_sets:
        ss0 = mesh0.side_sets[name]
        ss1 = mesh1.side_sets[name]
        assert ss0.id == ss1.id
        assert len(ss0.data) == len(ss1.data)
        for d0, d1 in zip(ss0.data, ss1.data):
            assert np.array_equal(d0, d1)
        if ss0.distribution_factors is None:
            assert ss1.distribution_factors is None
        else:
            assert np.allclose(
                ss0.distribution_factors, ss1.distribution_factors, rtol=0
            )


def test_side_sets_roundtrip(tmp_path):
    mesh = _side_set_mesh()
    path = tmp_path / "test.e"
    meshio.exodus.write(path, mesh)
    m = meshio.exodus.read(path)

    _assert_side_sets_equal(mesh, m)

    # oriented face nodes, including the high-order edge nodes
    nodes = m.get_side_set_nodes("wall")
    assert len(nodes) == 3
    assert np.array_equal(nodes[0], [0, 1, 3, 4, 8, 7])
    # same geometric face as nodes[0], but opposite orientation
    assert np.array_equal(nodes[1], [0, 3, 1, 7, 8, 4])
    assert np.array_equal(nodes[2], [18, 19, 20, 21, 26, 27, 28, 29])

    nodes = m.get_side_set_nodes("top")
    assert len(nodes) == 1
    assert np.array_equal(nodes[0], [14, 17, 16, 15, 25, 24, 23, 22])


def test_side_sets_block_reorder(tmp_path):
    mesh = _side_set_mesh()

    # same mesh, but with the cell blocks swapped
    side_sets = {}
    for name, ss in mesh.side_sets.items():
        df = ss.distribution_factors
        if df is not None:
            # distribution factors are grouped by block; regroup them
            df = np.concatenate([df[12:], df[:12]])
        side_sets[name] = meshio.SideSet(
            [ss.data[1], ss.data[0]], id=ss.id, distribution_factors=df
        )
    reordered = meshio.Mesh(
        mesh.points,
        [mesh.cells[1], mesh.cells[0]],
        side_sets=side_sets,
    )

    path = tmp_path / "test.e"
    meshio.exodus.write(path, reordered)
    m = meshio.exodus.read(path)

    _assert_side_sets_equal(reordered, m)

    # the sets must still point to the same actual element faces, with the
    # same orientation
    def faces(mesh, name):
        return sorted(
            tuple(tuple(row) for row in mesh.points[n])
            for n in mesh.get_side_set_nodes(name)
        )

    for name in mesh.side_sets:
        assert faces(mesh, name) == faces(m, name)


def test_side_set_distribution_factors_mismatch(tmp_path):
    mesh = _side_set_mesh()
    ss = mesh.side_sets["wall"]
    # 20 face nodes, but only 19 factors
    with pytest.raises(ValueError, match="wall"):
        meshio.Mesh(
            mesh.points,
            mesh.cells,
            side_sets={
                "wall": meshio.SideSet(
                    ss.data, id=ss.id, distribution_factors=np.zeros(19)
                )
            },
        )

    # also caught at write time if the mesh was mutated after construction
    mesh.side_sets["wall"].distribution_factors = np.zeros(19)
    with pytest.raises(ValueError, match="wall"):
        meshio.exodus.write(tmp_path / "mismatch.e", mesh)


def test_side_set_distribution_factors_mismatch_file(tmp_path):
    # a file whose distribution factor count doesn't match the side node count
    path = tmp_path / "bad.e"
    with netCDF4.Dataset(path, "w") as rootgrp:
        rootgrp.createDimension("num_nodes", 4)
        rootgrp.createDimension("num_dim", 3)
        rootgrp.createDimension("num_elem", 1)
        rootgrp.createDimension("num_el_blk", 1)
        rootgrp.createDimension("num_el_in_blk1", 1)
        rootgrp.createDimension("num_nod_per_el1", 4)
        rootgrp.createDimension("num_side_sets", 1)
        rootgrp.createDimension("num_elem_ss1", 1)
        rootgrp.createDimension("num_df_ss1", 2)
        rootgrp.createDimension("len_string", 33)
        coord = rootgrp.createVariable("coord", "f8", ("num_dim", "num_nodes"))
        coord[:] = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        connect = rootgrp.createVariable(
            "connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1")
        )
        connect.elem_type = "TETRA"
        connect[:] = [[1, 2, 3, 4]]
        ss_prop1 = rootgrp.createVariable("ss_prop1", "i4", "num_side_sets")
        ss_prop1[:] = [5]
        ss_names = rootgrp.createVariable(
            "ss_names", "S1", ("num_side_sets", "len_string")
        )
        ss_names.set_auto_mask(False)
        ss_names[0, 0] = b"w"
        ss_names[0, 1] = b"a"
        ss_names[0, 2] = b"l"
        ss_names[0, 3] = b"l"
        elem_ss1 = rootgrp.createVariable("elem_ss1", "i4", ("num_elem_ss1",))
        elem_ss1[:] = [1]
        side_ss1 = rootgrp.createVariable("side_ss1", "i4", ("num_elem_ss1",))
        side_ss1[:] = [1]
        # a tet face has 3 nodes, not 2
        dist_fact_ss1 = rootgrp.createVariable(
            "dist_fact_ss1", "f8", ("num_df_ss1",)
        )
        dist_fact_ss1[:] = [1.0, 2.0]

    with pytest.raises(ReadError, match="wall"):
        meshio.exodus.read(path)
