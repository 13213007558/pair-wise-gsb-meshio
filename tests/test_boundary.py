import copy

import numpy as np
import pytest

import meshio
from meshio._boundary import _volume_faces

num_corners = {
    "tetra": 4,
    "tetra10": 4,
    "hexahedron": 8,
    "hexahedron20": 8,
    "hexahedron27": 8,
    "wedge": 6,
    "wedge15": 6,
    "wedge18": 6,
}


def face_num_corners(face_type):
    return 4 if face_type.startswith("quad") else 3


def face_normal(points, nodes):
    # Newell's method
    p = points[np.asarray(nodes)]
    normal = np.zeros(3)
    for k in range(len(p)):
        a, b = p[k], p[(k + 1) % len(p)]
        normal[0] += (a[1] - b[1]) * (a[2] + b[2])
        normal[1] += (a[2] - b[2]) * (a[0] + b[0])
        normal[2] += (a[0] - b[0]) * (a[1] + b[1])
    return normal


def face_area(points, face_type, nodes):
    # area of the linear face through the corner nodes
    n = face_num_corners(face_type)
    p = points[np.asarray(nodes[:n])]
    area = 0.0
    for k in range(1, n - 1):
        area += 0.5 * np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0]))
    return area


def boundary_area(bmesh):
    area = 0.0
    for block in bmesh.cells:
        for face in block.data:
            area += face_area(bmesh.points, block.type, face)
    return area


def num_boundary_faces(bmesh):
    return sum(len(block) for block in bmesh.cells)


def check_mid_nodes(points, face_type, nodes):
    # the mid nodes of a higher-order face must sit on the edges/face of the
    # corner nodes
    p = points[np.asarray(nodes)]
    n = face_num_corners(face_type)
    if face_type in ("triangle6", "quad8", "quad9"):
        for k in range(n):
            assert np.allclose(p[n + k], 0.5 * (p[k] + p[(k + 1) % n]))
    if face_type == "quad9":
        assert np.allclose(p[8], p[:4].mean(axis=0))


def check_parents_and_orientation(mesh, bmesh):
    # every boundary face must come from the local face of its parent cell(s)
    # and its normal must point out of the first (into the second) parent
    for block, pbs, pcs, pfs in zip(
        bmesh.cells,
        bmesh.cell_data["parent_block"],
        bmesh.cell_data["parent_cell"],
        bmesh.cell_data["parent_face"],
    ):
        nc = face_num_corners(block.type)
        for face, pb, pc, pf in zip(block.data, pbs, pcs, pfs):
            center = bmesh.points[face[:nc]].mean(axis=0)
            normal = face_normal(bmesh.points, face[:nc])
            for side, sign in ((0, 1.0), (1, -1.0)):
                if pb[side] < 0:
                    continue
                parent_block = mesh.cells[pb[side]]
                conn = parent_block.data[pc[side]]
                face_type, local_nodes = _volume_faces[parent_block.type][pf[side]]
                assert face_type == block.type
                expected = conn[np.asarray(local_nodes)]
                if side == 0:
                    # the face keeps the parent-local node order
                    assert np.array_equal(expected, face)
                else:
                    assert np.array_equal(np.sort(expected), np.sort(face))
                cell_center = mesh.points[conn[: num_corners[parent_block.type]]].mean(
                    axis=0
                )
                assert sign * np.dot(normal, center - cell_center) > 0.0


def check_mesh_unchanged(mesh, original):
    assert np.array_equal(mesh.points, original.points)
    assert len(mesh.cells) == len(original.cells)
    for block, oblock in zip(mesh.cells, original.cells):
        assert block.type == oblock.type
        assert np.array_equal(block.data, oblock.data)
    assert mesh.cell_data.keys() == original.cell_data.keys()
    for key, data in mesh.cell_data.items():
        for d, o in zip(data, original.cell_data[key]):
            assert np.array_equal(d, o)


def edge_midpoints(corners, edges):
    return [
        (np.asarray(corners[a], dtype=float) + np.asarray(corners[b], dtype=float))
        * 0.5
        for a, b in edges
    ]


tetra_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
hexahedron_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
wedge_edges = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 4),
    (2, 5),
]


def single_cell_mesh(cell_type):
    if cell_type.startswith("tetra"):
        corners = [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        points = list(corners)
        if cell_type == "tetra10":
            points += edge_midpoints(corners, tetra_edges)
    elif cell_type.startswith("hexahedron"):
        corners = [
            [0.0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
        points = list(corners)
        if cell_type in ("hexahedron20", "hexahedron27"):
            points += edge_midpoints(corners, hexahedron_edges)
        if cell_type == "hexahedron27":
            face_corners = [
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ]
            points += [np.mean([corners[k] for k in f], axis=0) for f in face_corners]
            points += [np.mean(corners, axis=0)]
    else:
        assert cell_type.startswith("wedge")
        corners = [
            [0.0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ]
        points = list(corners)
        if cell_type in ("wedge15", "wedge18"):
            points += edge_midpoints(corners, wedge_edges)
        if cell_type == "wedge18":
            quad_faces = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
            points += [np.mean([corners[k] for k in f], axis=0) for f in quad_faces]
    return meshio.Mesh(
        np.asarray(points, dtype=float),
        [(cell_type, [list(range(len(points)))])],
    )


@pytest.mark.parametrize(
    "cell_type,surface_area",
    [
        ("tetra", 1.5 + np.sqrt(3) / 2),
        ("tetra10", 1.5 + np.sqrt(3) / 2),
        ("hexahedron", 6.0),
        ("hexahedron20", 6.0),
        ("hexahedron27", 6.0),
        ("wedge", 3.0 + np.sqrt(2)),
        ("wedge15", 3.0 + np.sqrt(2)),
        ("wedge18", 3.0 + np.sqrt(2)),
    ],
)
def test_single_cell(cell_type, surface_area):
    mesh = single_cell_mesh(cell_type)
    bmesh = mesh.extract_boundary()

    # one boundary face per local face of the cell
    faces = _volume_faces[cell_type]
    assert num_boundary_faces(bmesh) == len(faces)

    # face types and mid nodes
    expected_types = {}
    for face_type, _ in faces:
        expected_types[face_type] = expected_types.get(face_type, 0) + 1
    assert {block.type: len(block) for block in bmesh.cells} == expected_types
    for block in bmesh.cells:
        for face in block.data:
            check_mid_nodes(bmesh.points, block.type, face)

    # surface area and outward orientation
    assert boundary_area(bmesh) == pytest.approx(surface_area)
    check_parents_and_orientation(mesh, bmesh)

    # all faces belong to cell 0 of block 0, every local face exactly once
    parent_face = np.concatenate(bmesh.cell_data["parent_face"])
    assert np.array_equal(np.sort(parent_face[:, 0]), np.arange(len(faces)))
    assert np.all(parent_face[:, 1] == -1)
    assert np.all(np.concatenate(bmesh.cell_data["parent_block"]) == [[0, -1]])
    assert np.all(np.concatenate(bmesh.cell_data["parent_cell"]) == [[0, -1]])


def mixed_cavity_mesh():
    # Hexahedron/wedge mesh of the cube [0, 2]^3 with the cubic cavity
    # [0.5, 1.5]^3. The 26 boxes of the 3x3x3 grid (grid lines at 0, 0.5, 1.5,
    # 2) around the central cavity are filled with hexahedra and wedge pairs;
    # the hexahedra are split into two blocks of the same type. Material 1 is
    # assigned to the top layer (z >= 1.5), material 0 elsewhere.
    g = [0.0, 0.5, 1.5, 2.0]
    points = []
    vertex_id = {}
    for a in range(4):
        for b in range(4):
            for c in range(4):
                vertex_id[(a, b, c)] = len(points)
                points.append([g[a], g[b], g[c]])

    def v(a, b, c):
        return vertex_id[(a, b, c)]

    hexes, wedges = [], []
    hex_mats, wedge_mats = [], []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i, j, k) == (1, 1, 1):
                    continue  # the cavity
                mat = 1 if k == 2 else 0
                if (i + j) % 2 == 0:
                    hexes.append(
                        [
                            v(i, j, k),
                            v(i + 1, j, k),
                            v(i + 1, j + 1, k),
                            v(i, j + 1, k),
                            v(i, j, k + 1),
                            v(i + 1, j, k + 1),
                            v(i + 1, j + 1, k + 1),
                            v(i, j + 1, k + 1),
                        ]
                    )
                    hex_mats.append(mat)
                else:
                    # split the box into two wedges along the anti-diagonal
                    wedges.append(
                        [
                            v(i, j, k),
                            v(i + 1, j, k),
                            v(i, j + 1, k),
                            v(i, j, k + 1),
                            v(i + 1, j, k + 1),
                            v(i, j + 1, k + 1),
                        ]
                    )
                    wedges.append(
                        [
                            v(i + 1, j + 1, k),
                            v(i, j + 1, k),
                            v(i + 1, j, k),
                            v(i + 1, j + 1, k + 1),
                            v(i, j + 1, k + 1),
                            v(i + 1, j, k + 1),
                        ]
                    )
                    wedge_mats += [mat, mat]

    # two hexahedron blocks of the same type
    return meshio.Mesh(
        points,
        [
            ("hexahedron", hexes[:7]),
            ("wedge", wedges),
            ("hexahedron", hexes[7:]),
        ],
        cell_data={"material": [hex_mats[:7], wedge_mats, hex_mats[7:]]},
    )


def test_mixed_cavity_mesh():
    mesh = mixed_cavity_mesh()
    original = copy.deepcopy(mesh)

    bmesh = mesh.extract_boundary()

    # the input mesh must not be modified
    check_mesh_unchanged(mesh, original)

    # 52 quads (46 outer + 6 cavity) and 16 triangles
    assert {block.type: len(block) for block in bmesh.cells} == {
        "triangle": 16,
        "quad": 52,
    }
    # outer surface (6 * 2^2) plus cavity surface (6 * 1^2)
    assert boundary_area(bmesh) == pytest.approx(30.0)

    # all faces are on the outer boundary
    assert np.all(np.concatenate(bmesh.cell_data["parent_block"])[:, 1] == -1)

    check_parents_and_orientation(mesh, bmesh)


def test_mixed_cavity_mesh_material_interface():
    mesh = mixed_cavity_mesh()
    original = copy.deepcopy(mesh)

    bmesh = mesh.extract_boundary(material="material")

    # the input mesh must not be modified
    check_mesh_unchanged(mesh, original)

    # 68 outer boundary faces plus 12 material interface faces
    assert num_boundary_faces(bmesh) == 80

    interface_area = 0.0
    outer_area = 0.0
    num_interface = 0
    for block, pbs, mats in zip(
        bmesh.cells, bmesh.cell_data["parent_block"], bmesh.cell_data["material"]
    ):
        for face, pb, mat in zip(block.data, pbs, mats):
            area = face_area(bmesh.points, block.type, face)
            if pb[1] >= 0:
                # interface between material 0 and material 1
                num_interface += 1
                interface_area += area
                assert set(mat) == {0, 1}
            else:
                outer_area += area
                assert mat[1] == -1
                assert mat[0] in (0, 1)

    # the interface is the plane z = 1.5 minus the cavity cross section
    assert num_interface == 12
    assert interface_area == pytest.approx(3.0)
    assert outer_area == pytest.approx(30.0)

    check_parents_and_orientation(mesh, bmesh)


def tetra_box_mesh():
    # two unit cubes next to each other (x in [0, 2]), each split into 6
    # tetrahedra around the (0,0,0)-(1,1,1) body diagonal; one block per cube
    points = []
    vertex_id = {}
    for a in range(3):
        for b in range(2):
            for c in range(2):
                vertex_id[(a, b, c)] = len(points)
                points.append([float(a), float(b), float(c)])

    def box_tets(x0):
        def v(a, b, c):
            return vertex_id[(x0 + a, b, c)]

        return [
            [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(1, 1, 1)],
            [v(0, 0, 0), v(1, 1, 0), v(0, 1, 0), v(1, 1, 1)],
            [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(1, 1, 1)],
            [v(0, 0, 0), v(0, 1, 1), v(0, 0, 1), v(1, 1, 1)],
            [v(0, 0, 0), v(0, 0, 1), v(1, 0, 1), v(1, 1, 1)],
            [v(0, 0, 0), v(1, 0, 1), v(1, 0, 0), v(1, 1, 1)],
        ]

    return meshio.Mesh(points, [("tetra", box_tets(0)), ("tetra", box_tets(1))])


def test_tetra_box():
    mesh = tetra_box_mesh()
    bmesh = mesh.extract_boundary()

    # 12 surface triangles per cube, 2 shared triangles cancel
    assert {block.type: len(block) for block in bmesh.cells} == {"triangle": 20}
    # surface of [0, 2] x [0, 1] x [0, 1]
    assert boundary_area(bmesh) == pytest.approx(10.0)

    check_parents_and_orientation(mesh, bmesh)


def test_tetra_box_material_interface():
    mesh = tetra_box_mesh()
    mesh.cell_data["material"] = [np.zeros(6, dtype=int), np.ones(6, dtype=int)]

    bmesh = mesh.extract_boundary(material="material")

    # 20 outer boundary faces plus the 2 interface triangles at x = 1
    assert num_boundary_faces(bmesh) == 22

    interface_area = 0.0
    for block, pbs, mats in zip(
        bmesh.cells, bmesh.cell_data["parent_block"], bmesh.cell_data["material"]
    ):
        for face, pb, mat in zip(block.data, pbs, mats):
            if pb[1] >= 0:
                interface_area += face_area(bmesh.points, block.type, face)
                assert set(mat) == {0, 1}
    assert interface_area == pytest.approx(1.0)

    check_parents_and_orientation(mesh, bmesh)


class PointCloud:
    # collects points; identical coordinates share the same node
    def __init__(self):
        self.points = []
        self.point_id = {}

    def node(self, coord):
        coord = tuple(float(x) for x in coord)
        if coord not in self.point_id:
            self.point_id[coord] = len(self.points)
            self.points.append(coord)
        return self.point_id[coord]

    def midpoint(self, coords):
        return self.node(np.mean(np.asarray(coords, dtype=float), axis=0))


def quadratic_mixed_mesh():
    # 2x2x1 boxes of the cube [0, 2]^2 x [0, 1]: two hexahedron27 on the
    # diagonal, the other two boxes split into wedge18 pairs
    cloud = PointCloud()

    def hex27(i, j):
        corners = [
            (i, j, 0),
            (i + 1, j, 0),
            (i + 1, j + 1, 0),
            (i, j + 1, 0),
            (i, j, 1),
            (i + 1, j, 1),
            (i + 1, j + 1, 1),
            (i, j + 1, 1),
        ]
        faces = [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        ids = [cloud.node(p) for p in corners]
        ids += [cloud.midpoint([corners[a], corners[b]]) for a, b in hexahedron_edges]
        ids += [cloud.midpoint([corners[k] for k in f]) for f in faces]
        ids += [cloud.midpoint(corners)]
        return ids

    def wedge18(corners):
        quad_faces = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
        ids = [cloud.node(p) for p in corners]
        ids += [cloud.midpoint([corners[a], corners[b]]) for a, b in wedge_edges]
        ids += [cloud.midpoint([corners[k] for k in f]) for f in quad_faces]
        return ids

    def wedge_pair(i, j):
        return [
            wedge18(
                [
                    (i, j, 0),
                    (i + 1, j, 0),
                    (i, j + 1, 0),
                    (i, j, 1),
                    (i + 1, j, 1),
                    (i, j + 1, 1),
                ]
            ),
            wedge18(
                [
                    (i + 1, j + 1, 0),
                    (i, j + 1, 0),
                    (i + 1, j, 0),
                    (i + 1, j + 1, 1),
                    (i, j + 1, 1),
                    (i + 1, j, 1),
                ]
            ),
        ]

    return meshio.Mesh(
        cloud.points,
        [
            ("hexahedron27", [hex27(0, 0), hex27(1, 1)]),
            ("wedge18", wedge_pair(1, 0) + wedge_pair(0, 1)),
        ],
    )


def test_quadratic_mixed_mesh():
    mesh = quadratic_mixed_mesh()
    original = copy.deepcopy(mesh)

    bmesh = mesh.extract_boundary()
    check_mesh_unchanged(mesh, original)

    # 12 quad9 (8 free hexahedron faces, 4 free wedge sides) and 8 triangle6
    # (wedge tops and bottoms); all interior quad9 faces cancel
    assert {block.type: len(block) for block in bmesh.cells} == {
        "triangle6": 8,
        "quad9": 12,
    }
    # surface of [0, 2]^2 x [0, 1]
    assert boundary_area(bmesh) == pytest.approx(16.0)

    for block in bmesh.cells:
        for face in block.data:
            check_mid_nodes(bmesh.points, block.type, face)

    check_parents_and_orientation(mesh, bmesh)


def tetra10_pair_mesh(shared_mid_nodes):
    # two tetra10 sharing the face through the corners (0,0,0), (1,0,0),
    # (0,1,0); with shared_mid_nodes=False the second cell gets its own mid
    # nodes (at the same positions) on the shared face
    cloud = PointCloud()
    new_points = []

    def node_new(coord):
        new_points.append(tuple(float(x) for x in coord))
        return len(cloud.points) + len(new_points) - 1

    def tetra10(corners, mid_node):
        ids = [cloud.node(p) for p in corners]
        ids += [
            mid_node(
                0.5 * (np.asarray(corners[a], dtype=float) + np.asarray(corners[b]))
            )
            for a, b in tetra_edges
        ]
        return ids

    corners_a = [(0.0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    corners_b = [(0.0, 0, 0), (0, 1, 0), (1, 0, 0), (0.5, 0.5, -1)]
    mid_node = cloud.node if shared_mid_nodes else node_new
    cells = [tetra10(corners_a, cloud.node), tetra10(corners_b, mid_node)]
    return meshio.Mesh(cloud.points + new_points, [("tetra10", cells)])


def test_tetra10_shared_face():
    mesh = tetra10_pair_mesh(shared_mid_nodes=True)
    bmesh = mesh.extract_boundary()

    # the shared face cancels: 2 * 4 - 2 = 6 boundary faces
    assert {block.type: len(block) for block in bmesh.cells} == {"triangle6": 6}
    for block in bmesh.cells:
        for face in block.data:
            check_mid_nodes(bmesh.points, block.type, face)
    check_parents_and_orientation(mesh, bmesh)


def test_tetra10_mismatched_mid_nodes():
    # the two faces share their corner nodes but not their mid nodes; they
    # must not be mistaken for an interior face
    mesh = tetra10_pair_mesh(shared_mid_nodes=False)
    bmesh = mesh.extract_boundary()

    assert {block.type: len(block) for block in bmesh.cells} == {"triangle6": 8}


def test_nonmanifold_face():
    # three tetrahedra sharing the face (0, 1, 2)
    mesh = meshio.Mesh(
        [
            [0.0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, -1],
            [0.2, 0.2, 2],
        ],
        [("tetra", [[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]])],
    )
    with pytest.warns(UserWarning, match="non-manifold face"):
        bmesh = mesh.extract_boundary()

    # all three occurrences of the shared face are kept
    count = 0
    for block in bmesh.cells:
        for face in block.data:
            if set(face) == {0, 1, 2}:
                count += 1
    assert count == 3


def test_unsupported_cell_type():
    mesh = meshio.Mesh(
        [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [("pyramid", [[0, 1, 2, 3, 4]])],
    )
    with pytest.raises(ValueError, match="cannot extract boundary"):
        mesh.extract_boundary()


def test_unknown_material_key():
    mesh = single_cell_mesh("tetra")
    with pytest.raises(ValueError, match="no cell_data"):
        mesh.extract_boundary(material="no-such-key")


def test_no_volume_cells():
    mesh = meshio.Mesh([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], [("triangle", [[0, 1, 2]])])
    bmesh = mesh.extract_boundary()
    assert len(bmesh.cells) == 0
