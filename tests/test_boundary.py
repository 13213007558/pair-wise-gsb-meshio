import copy
from itertools import product

import numpy as np
import pytest

import meshio
from meshio._boundary import boundary_faces


def _face_geometry(points, cell_type, nodes):
    # area and (unit) normal of a planar face, from its corner nodes
    num_corners = {"triangle": 3, "triangle6": 3, "quad": 4, "quad8": 4, "quad9": 4}[
        cell_type
    ]
    p = points[np.asarray(nodes[:num_corners])]
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    area = 0.5 * np.linalg.norm(normal)
    if num_corners == 4:
        n2 = np.cross(p[2] - p[0], p[3] - p[0])
        area += 0.5 * np.linalg.norm(n2)
        normal = normal + n2
    return area, normal / np.linalg.norm(normal)


def _total_area(mesh):
    return sum(
        _face_geometry(mesh.points, block.type, cell)[0]
        for block in mesh.cells
        for cell in block.data
    )


def _check_parents(mesh, boundary):
    # verify the parent mapping of every boundary/interface face
    for block, pb, pc, pf in zip(
        boundary.cells,
        boundary.cell_data["parent_block"],
        boundary.cell_data["parent_cell"],
        boundary.cell_data["parent_face"],
    ):
        for face_nodes, parents_b, parents_c, parents_f in zip(
            block.data, pb, pc, pf
        ):
            for side in range(2):
                if parents_b[side] < 0:
                    continue
                parent = mesh.cells[parents_b[side]]
                _, local_nodes = boundary_faces[parent.type][parents_f[side]]
                parent_face = parent.data[parents_c[side]][local_nodes]
                assert sorted(parent_face.tolist()) == sorted(face_nodes.tolist())


def _mixed_cavity_mesh():
    # 3x3x3 grid of unit cubes in [0, 3]^3 with the center cube removed,
    # leaving an inner cavity. Cells with i+j even are hexahedra (split into
    # two blocks of the same type), cells with i+j odd are pairs of wedges.
    points = np.array(
        [[i, j, k] for k in range(4) for j in range(4) for i in range(4)],
        dtype=float,
    )

    def pid(i, j, k):
        return i + 4 * (j + 4 * k)

    inner = {(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)}

    hexes = [[], []]
    hex_mat = [[], []]
    wedges = []
    wedge_mat = []
    for i, j, k in product(range(3), repeat=3):
        if (i, j, k) == (1, 1, 1):
            continue
        c = [
            pid(i, j, k),
            pid(i + 1, j, k),
            pid(i + 1, j + 1, k),
            pid(i, j + 1, k),
            pid(i, j, k + 1),
            pid(i + 1, j, k + 1),
            pid(i + 1, j + 1, k + 1),
            pid(i, j + 1, k + 1),
        ]
        mat = 1 if (i, j, k) in inner else 2
        if (i + j) % 2 == 0:
            target = len(hexes[0]) <= len(hexes[1])
            hexes[0 if target else 1].append(c)
            hex_mat[0 if target else 1].append(mat)
        else:
            wedges.append([c[0], c[1], c[2], c[4], c[5], c[6]])
            wedges.append([c[0], c[2], c[3], c[4], c[6], c[7]])
            wedge_mat += [mat, mat]

    return meshio.Mesh(
        points,
        [
            ("hexahedron", hexes[0]),
            ("wedge", wedges),
            ("hexahedron", hexes[1]),
        ],
        cell_data={"material": [hex_mat[0], wedge_mat, hex_mat[1]]},
    )


def test_mixed_cavity_boundary():
    mesh = _mixed_cavity_mesh()
    original = copy.deepcopy(mesh)

    boundary = mesh.extract_boundary()

    # 52 quads (outer surface minus wedge top/bottom splits, plus cavity)
    # and 16 triangles (wedge top/bottom faces on z=0 and z=3)
    assert {b.type: len(b) for b in boundary.cells} == {"quad": 52, "triangle": 16}
    # outer surface 6 * 9, cavity surface 6 * 1
    assert _total_area(boundary) == pytest.approx(60.0)

    center = np.array([1.5, 1.5, 1.5])
    for block, pb, pc in zip(
        boundary.cells, boundary.cell_data["parent_block"], boundary.cell_data["parent_cell"]
    ):
        assert np.all(pb[:, 1] == -1)
        for face_nodes, b, c in zip(block.data, pb, pc):
            _, normal = _face_geometry(boundary.points, block.type, face_nodes)
            face_center = np.mean(boundary.points[face_nodes[:3]], axis=0)
            parent_nodes = mesh.cells[b[0]].data[c[0]]
            parent_center = np.mean(mesh.points[parent_nodes], axis=0)
            # normal points out of the parent cell
            assert np.dot(normal, face_center - parent_center) > 0
            # and away from the solid (into the cavity or out of the box)
            if np.all(face_center >= 1.0) and np.all(face_center <= 2.0):
                assert np.dot(normal, center - face_center) > 0
            else:
                assert np.dot(normal, face_center - center) > 0

    _check_parents(mesh, boundary)

    # the input mesh must not be modified
    assert np.array_equal(mesh.points, original.points)
    for b0, b1 in zip(mesh.cells, original.cells):
        assert b0.type == b1.type
        assert np.array_equal(b0.data, b1.data)
    for key in mesh.cell_data:
        for d0, d1 in zip(mesh.cell_data[key], original.cell_data[key]):
            assert np.array_equal(d0, d1)


def test_mixed_cavity_interfaces():
    mesh = _mixed_cavity_mesh()
    boundary = mesh.extract_boundary(material_name="material")

    num_interface = 0
    interface_area = 0.0
    for block, is_interface, mat, pb, pc in zip(
        boundary.cells,
        boundary.cell_data["is_interface"],
        boundary.cell_data["material"],
        boundary.cell_data["parent_block"],
        boundary.cell_data["parent_cell"],
    ):
        for face_nodes, is_int, m, b, c in zip(block.data, is_interface, mat, pb, pc):
            if not is_int:
                assert b[1] == -1
                assert m[1] == -1
                continue
            num_interface += 1
            # both parents are present, with different materials; the first
            # parent carries the smaller material tag
            assert b[1] >= 0
            assert m[0] == 1
            assert m[1] == 2
            area, normal = _face_geometry(boundary.points, block.type, face_nodes)
            interface_area += area
            # normal points out of the first (material 1) parent
            parent_nodes = mesh.cells[b[0]].data[c[0]]
            parent_center = np.mean(mesh.points[parent_nodes], axis=0)
            face_center = np.mean(boundary.points[face_nodes[:3]], axis=0)
            assert np.dot(normal, face_center - parent_center) > 0

    # 6 inner cells with 4 interface faces each (area 1)
    assert num_interface == 32
    assert interface_area == pytest.approx(24.0)

    # boundary faces are unaffected
    is_int = np.concatenate(boundary.cell_data["is_interface"])
    assert np.sum(~is_int) == 68

    _check_parents(mesh, boundary)

    # interfaces can be switched off
    boundary2 = mesh.extract_boundary(material_name="material", include_interfaces=False)
    assert not np.any(np.concatenate(boundary2.cell_data["is_interface"]))
    assert sum(len(b) for b in boundary2.cells) == 68


def test_tetra10_boundary():
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    mesh = meshio.Mesh(points, [("tetra10", [list(range(10))])])
    boundary = mesh.extract_boundary()
    assert {b.type: len(b) for b in boundary.cells} == {"triangle6": 4}
    # mid nodes are kept
    assert set(np.unique(boundary.cells[0].data).tolist()) == set(range(10))
    # surface area of the reference tetrahedron
    assert _total_area(boundary) == pytest.approx(1.5 + 0.5 * np.sqrt(3.0))


def _two_hexahedron27_mesh(mismatched_midnode=False):
    # two hexahedron27 sharing the face x=1 of the first cell
    cell_a = list(range(27))
    cell_b = [
        1, 27, 28, 2, 5, 29, 30, 6,
        31, 32, 33, 9, 34, 35, 36, 13, 17, 37, 38, 18,
        39, 40, 41, 42, 43, 23,
        44,
    ]
    if mismatched_midnode:
        # shared corners, but one mid-edge node differs
        cell_b[11] = 45
    rng = np.random.default_rng(0)
    points = rng.random((46, 3))
    return meshio.Mesh(points, [("hexahedron27", [cell_a, cell_b])])


def test_hexahedron27_shared_face():
    mesh = _two_hexahedron27_mesh()
    boundary = mesh.extract_boundary()
    # 6 + 6 - 2 faces
    assert {b.type: len(b) for b in boundary.cells} == {"quad9": 10}
    # mid nodes are kept
    assert boundary.cells[0].data.shape == (10, 9)


def test_hexahedron27_mismatched_midnode():
    # same corner topology, but the mid nodes do not agree: the faces must
    # not cancel
    mesh = _two_hexahedron27_mesh(mismatched_midnode=True)
    boundary = mesh.extract_boundary()
    assert {b.type: len(b) for b in boundary.cells} == {"quad9": 12}


def test_hexahedron20_shared_face():
    cell_a = list(range(20))
    cell_b = [
        1, 20, 21, 2, 5, 22, 23, 6,
        24, 25, 26, 9, 27, 28, 29, 13, 17, 30, 31, 18,
    ]
    rng = np.random.default_rng(0)
    points = rng.random((32, 3))
    mesh = meshio.Mesh(points, [("hexahedron20", [cell_a, cell_b])])
    boundary = mesh.extract_boundary()
    assert {b.type: len(b) for b in boundary.cells} == {"quad8": 10}


def test_wedge18_boundary():
    mesh = meshio.Mesh(
        np.random.default_rng(0).random((18, 3)),
        [("wedge18", [list(range(18))])],
    )
    boundary = mesh.extract_boundary()
    assert {b.type: len(b) for b in boundary.cells} == {"triangle6": 2, "quad9": 3}


def test_wedge15_shared_face():
    # two wedge15 sharing the quad face (0, 1, 4, 3) of the first cell
    cell_a = list(range(15))
    cell_b = [
        1, 15, 0, 4, 16, 3,
        17, 18, 6, 19, 20, 9, 13, 21, 12,
    ]
    rng = np.random.default_rng(0)
    points = rng.random((22, 3))
    mesh = meshio.Mesh(points, [("wedge15", [cell_a, cell_b])])
    boundary = mesh.extract_boundary()
    # 2 * (2 triangle6 + 3 quad8) - 2 quad8
    assert {b.type: len(b) for b in boundary.cells} == {"triangle6": 4, "quad8": 4}


def test_nonmanifold_face():
    # three tetrahedra sharing the same face (0, 1, 2)
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.5, 0.5, 1.0],
    ]
    mesh = meshio.Mesh(
        points, [("tetra", [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]])]
    )
    boundary = mesh.extract_boundary()
    # 3 * 4 - 2 faces cancel pairwise... no: the shared face is kept once
    assert sum(len(b) for b in boundary.cells) == 10
    # the non-manifold face is kept with two of its parents
    pb = np.concatenate(boundary.cell_data["parent_block"])
    assert np.any(pb[:, 1] >= 0)


def test_unsupported_cell_type():
    mesh = meshio.Mesh(
        np.zeros((5, 3)), [("pyramid", [[0, 1, 2, 3, 4]])]
    )
    with pytest.raises(ValueError, match="pyramid"):
        mesh.extract_boundary()


def test_unknown_material():
    mesh = meshio.Mesh(
        np.zeros((4, 3)), [("tetra", [[0, 1, 2, 3]])]
    )
    with pytest.raises(ValueError, match="cell_data"):
        mesh.extract_boundary(material_name="no-such-data")
