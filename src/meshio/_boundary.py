from __future__ import annotations

import warnings

import numpy as np

from ._mesh import CellBlock, Mesh

# Local faces of the supported volume cell types, oriented such that the face
# normal points out of the cell. The node ordering follows meshio conventions
# (see <https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells>):
#
# - tetra/tetra10: VTK ordering, edge nodes (0,1), (1,2), (2,0), (0,3), (1,3),
#   (2,3)
# - hexahedron/hexahedron20/hexahedron27: VTK ordering; for hexahedron27 the
#   face-center nodes 20-25 belong to the bottom, top, and the four side faces
#   (in this order), node 26 is the volume center
# - wedge/wedge15/wedge18: gmsh prism corner ordering (the (0,1,2) triangle
#   normal points into the cell); edge nodes (0,1), (1,2), (2,0), (3,4), (4,5),
#   (5,3), (0,3), (1,4), (2,5); for wedge18 the nodes 15-17 are the centers of
#   the three quadrilateral faces
_volume_faces = {
    "tetra": [
        ("triangle", (0, 2, 1)),
        ("triangle", (0, 1, 3)),
        ("triangle", (1, 2, 3)),
        ("triangle", (2, 0, 3)),
    ],
    "tetra10": [
        ("triangle6", (0, 2, 1, 6, 5, 4)),
        ("triangle6", (0, 1, 3, 4, 8, 7)),
        ("triangle6", (1, 2, 3, 5, 9, 8)),
        ("triangle6", (2, 0, 3, 6, 7, 9)),
    ],
    "hexahedron": [
        ("quad", (0, 3, 2, 1)),
        ("quad", (4, 5, 6, 7)),
        ("quad", (0, 1, 5, 4)),
        ("quad", (1, 2, 6, 5)),
        ("quad", (2, 3, 7, 6)),
        ("quad", (3, 0, 4, 7)),
    ],
    "hexahedron20": [
        ("quad8", (0, 3, 2, 1, 11, 10, 9, 8)),
        ("quad8", (4, 5, 6, 7, 12, 13, 14, 15)),
        ("quad8", (0, 1, 5, 4, 8, 17, 12, 16)),
        ("quad8", (1, 2, 6, 5, 9, 18, 13, 17)),
        ("quad8", (2, 3, 7, 6, 10, 19, 14, 18)),
        ("quad8", (3, 0, 4, 7, 11, 16, 15, 19)),
    ],
    "hexahedron27": [
        ("quad9", (0, 3, 2, 1, 11, 10, 9, 8, 20)),
        ("quad9", (4, 5, 6, 7, 12, 13, 14, 15, 21)),
        ("quad9", (0, 1, 5, 4, 8, 17, 12, 16, 22)),
        ("quad9", (1, 2, 6, 5, 9, 18, 13, 17, 23)),
        ("quad9", (2, 3, 7, 6, 10, 19, 14, 18, 24)),
        ("quad9", (3, 0, 4, 7, 11, 16, 15, 19, 25)),
    ],
    "wedge": [
        ("triangle", (0, 2, 1)),
        ("triangle", (3, 4, 5)),
        ("quad", (0, 1, 4, 3)),
        ("quad", (1, 2, 5, 4)),
        ("quad", (2, 0, 3, 5)),
    ],
    "wedge15": [
        ("triangle6", (0, 2, 1, 8, 7, 6)),
        ("triangle6", (3, 4, 5, 9, 10, 11)),
        ("quad8", (0, 1, 4, 3, 6, 13, 9, 12)),
        ("quad8", (1, 2, 5, 4, 7, 14, 10, 13)),
        ("quad8", (2, 0, 3, 5, 8, 12, 11, 14)),
    ],
    "wedge18": [
        ("triangle6", (0, 2, 1, 8, 7, 6)),
        ("triangle6", (3, 4, 5, 9, 10, 11)),
        ("quad9", (0, 1, 4, 3, 6, 13, 9, 12, 15)),
        ("quad9", (1, 2, 5, 4, 7, 14, 10, 13, 16)),
        ("quad9", (2, 0, 3, 5, 8, 12, 11, 14, 17)),
    ],
}

# output cell blocks are created in this order
_face_type_order = ["triangle", "triangle6", "quad", "quad8", "quad9"]


def extract_boundary(mesh: Mesh, material: str | None = None) -> Mesh:
    """Extract the boundary surface of the volume cells of a mesh.

    Faces shared by exactly two volume cells are interior and dropped, unless
    'material' is given and the two adjacent cells carry different material
    labels; in that case the face is a material interface and is kept. Face
    adjacency is detected by the complete (sorted) node topology of the faces,
    so higher-order faces only cancel if all of their nodes, including mid
    nodes, coincide.

    The face orientation follows the local orientation of the parent cell,
    i.e. boundary face normals point out of the volume. For interface faces,
    the orientation is taken from the first of the two parent cells.

    The returned mesh shares the (unmodified) points of the input mesh and
    carries the cell data

    - 'parent_block', 'parent_cell', 'parent_face': for every boundary face
      and each of its two sides, the index of the parent cell block, the
      index of the parent cell within the block, and the local face index of
      the face in the parent cell. The second side is -1 for faces on the
      outer boundary.
    - 'material' (if given): the material labels of the cells on the two
      sides of the face; -1 for the exterior.

    Faces shared by more than two cells are non-manifold; a warning is issued
    and every occurrence is kept as a boundary face.
    """
    if material is not None and material not in mesh.cell_data:
        raise ValueError(f'no cell_data "{material}" in mesh')

    # group all volume faces by their complete node topology
    face_groups = {}
    for block_id, block in enumerate(mesh.cells):
        if block.dim != 3:
            continue
        if block.type not in _volume_faces:
            raise ValueError(
                f"cannot extract boundary from cell type '{block.type}'; "
                "supported types: " + ", ".join(_volume_faces)
            )
        faces = _volume_faces[block.type]
        for cell_id, conn in enumerate(block.data):
            for face_id, (face_type, local_nodes) in enumerate(faces):
                nodes = tuple(int(conn[k]) for k in local_nodes)
                key = tuple(sorted(nodes))
                face_groups.setdefault(key, []).append(
                    (block_id, cell_id, face_id, face_type, nodes)
                )

    if material is not None:
        material_data = mesh.cell_data[material]

    def get_material(block_id, cell_id):
        if material is None:
            return -1
        return int(material_data[block_id][cell_id])

    faces_by_type = {face_type: [] for face_type in _face_type_order}
    parents_by_type = {face_type: [] for face_type in _face_type_order}
    materials_by_type = {face_type: [] for face_type in _face_type_order}

    def emit(face_type, nodes, parent0, parent1):
        faces_by_type[face_type].append(nodes)
        parents_by_type[face_type].append((parent0, parent1))
        materials_by_type[face_type].append(
            (
                get_material(*parent0[:2]),
                get_material(*parent1[:2]) if parent1[0] >= 0 else -1,
            )
        )

    no_parent = (-1, -1, -1)
    for group in face_groups.values():
        if len(group) == 1:
            # outer boundary face
            block_id, cell_id, face_id, face_type, nodes = group[0]
            emit(face_type, nodes, (block_id, cell_id, face_id), no_parent)
        elif len(group) == 2:
            # interior face; keep it only if it separates two materials
            if material is None:
                continue
            (b0, c0, f0, face_type, nodes), (b1, c1, f1, _, _) = group
            if get_material(b0, c0) == get_material(b1, c1):
                continue
            emit(face_type, nodes, (b0, c0, f0), (b1, c1, f1))
        else:
            # non-manifold face shared by more than two cells; pairing them up
            # is ambiguous, so keep every occurrence
            warnings.warn(
                f"non-manifold face shared by {len(group)} cells; "
                "keeping all occurrences as boundary faces"
            )
            for block_id, cell_id, face_id, face_type, nodes in group:
                emit(face_type, nodes, (block_id, cell_id, face_id), no_parent)

    cells = []
    cell_data = {"parent_block": [], "parent_cell": [], "parent_face": []}
    if material is not None:
        cell_data[material] = []

    for face_type in _face_type_order:
        faces = faces_by_type[face_type]
        if not faces:
            continue
        cells.append(CellBlock(face_type, np.asarray(faces, dtype=np.int64)))
        parents = np.asarray(parents_by_type[face_type], dtype=np.int64)
        cell_data["parent_block"].append(parents[:, :, 0])
        cell_data["parent_cell"].append(parents[:, :, 1])
        cell_data["parent_face"].append(parents[:, :, 2])
        if material is not None:
            cell_data[material].append(
                np.asarray(materials_by_type[face_type], dtype=np.int64)
            )

    # share the points with the input mesh; they are not modified
    return Mesh(mesh.points, cells, cell_data=cell_data)
