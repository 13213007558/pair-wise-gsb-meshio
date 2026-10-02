from __future__ import annotations

import numpy as np

from ._common import warn

# Local boundary faces of 3D cell types, given as local node IDs. The node
# order of each face follows the parent cell's local node ordering such that
# the face normal points out of the cell (right-hand rule). Higher-order faces
# include all of their mid nodes.
boundary_faces = {
    "tetra": [
        ("triangle", [0, 2, 1]),
        ("triangle", [0, 1, 3]),
        ("triangle", [0, 3, 2]),
        ("triangle", [1, 2, 3]),
    ],
    "tetra10": [
        ("triangle6", [0, 2, 1, 6, 5, 4]),
        ("triangle6", [0, 1, 3, 4, 8, 7]),
        ("triangle6", [0, 3, 2, 7, 9, 6]),
        ("triangle6", [1, 2, 3, 5, 9, 8]),
    ],
    "hexahedron": [
        ("quad", [0, 3, 2, 1]),
        ("quad", [4, 5, 6, 7]),
        ("quad", [0, 1, 5, 4]),
        ("quad", [1, 2, 6, 5]),
        ("quad", [2, 3, 7, 6]),
        ("quad", [0, 4, 7, 3]),
    ],
    "hexahedron20": [
        ("quad8", [0, 3, 2, 1, 11, 10, 9, 8]),
        ("quad8", [4, 5, 6, 7, 12, 13, 14, 15]),
        ("quad8", [0, 1, 5, 4, 8, 17, 12, 16]),
        ("quad8", [1, 2, 6, 5, 9, 18, 13, 17]),
        ("quad8", [2, 3, 7, 6, 10, 19, 14, 18]),
        ("quad8", [0, 4, 7, 3, 16, 15, 19, 11]),
    ],
    "hexahedron27": [
        ("quad9", [0, 3, 2, 1, 11, 10, 9, 8, 20]),
        ("quad9", [4, 5, 6, 7, 12, 13, 14, 15, 21]),
        ("quad9", [0, 1, 5, 4, 8, 17, 12, 16, 22]),
        ("quad9", [1, 2, 6, 5, 9, 18, 13, 17, 23]),
        ("quad9", [2, 3, 7, 6, 10, 19, 14, 18, 24]),
        ("quad9", [0, 4, 7, 3, 16, 15, 19, 11, 25]),
    ],
    "wedge": [
        ("triangle", [0, 2, 1]),
        ("triangle", [3, 4, 5]),
        ("quad", [0, 1, 4, 3]),
        ("quad", [1, 2, 5, 4]),
        ("quad", [2, 0, 3, 5]),
    ],
    "wedge15": [
        ("triangle6", [0, 2, 1, 8, 7, 6]),
        ("triangle6", [3, 4, 5, 9, 10, 11]),
        ("quad8", [0, 1, 4, 3, 6, 13, 9, 12]),
        ("quad8", [1, 2, 5, 4, 7, 14, 10, 13]),
        ("quad8", [2, 0, 3, 5, 8, 12, 11, 14]),
    ],
    "wedge18": [
        ("triangle6", [0, 2, 1, 8, 7, 6]),
        ("triangle6", [3, 4, 5, 9, 10, 11]),
        ("quad9", [0, 1, 4, 3, 6, 13, 9, 12, 15]),
        ("quad9", [1, 2, 5, 4, 7, 14, 10, 13, 16]),
        ("quad9", [2, 0, 3, 5, 8, 12, 11, 14, 17]),
    ],
}


def extract_boundary(
    mesh, material_name: str | None = None, include_interfaces: bool = True
):
    """Extract the boundary surface mesh of the 3D cells of a mesh.

    Faces are matched by their complete node topology (all point IDs, not
    just the corners), so higher-order faces only cancel when all of their
    nodes coincide. Faces adjacent to a single cell form the outer boundary.
    If material_name (a key of mesh.cell_data) is given, faces between cells
    of different materials are additionally reported as internal interfaces
    (unless include_interfaces=False). Interfaces keep both of their parent
    cells; the face orientation follows the parent with the smaller material
    tag.

    The input mesh is left untouched. The returned mesh holds a copy of the
    input point array and carries the cell data

    - parent_block: (n, 2) index into mesh.cells of the adjacent parent cell
      blocks (-1 if there is no second parent),
    - parent_cell: (n, 2) cell index within the parent block,
    - parent_face: (n, 2) local face index within the parent cell,
    - is_interface: (n,) whether the face separates two materials,
    - material: (n, 2) material tags of the two sides (only if material_name
      is given; -1 on the outside of boundary faces).
    """
    # avoid circular import
    from ._mesh import CellBlock, Mesh

    # indices of the 3D cell blocks
    block_ids = []
    for k, block in enumerate(mesh.cells):
        if block.dim < 3:
            continue
        if block.type not in boundary_faces:
            raise ValueError(
                f"Boundary extraction not implemented for cell type '{block.type}'."
            )
        block_ids.append(k)

    # material tag per 3D cell, in the order of block_ids
    if material_name is not None:
        if material_name not in mesh.cell_data:
            raise ValueError(f"Mesh has no cell_data '{material_name}'.")
        material = np.concatenate(
            [np.asarray(mesh.cell_data[material_name][k]) for k in block_ids]
        )
    else:
        material = None

    # map the running 3D cell index to (position in block_ids, index in block)
    cell_block = []
    cell_index = []
    for bpos, k in enumerate(block_ids):
        num = len(mesh.cells[k])
        cell_block += [bpos] * num
        cell_index += list(range(num))

    # Pass 1: collect all cell faces, keyed by the sorted tuple of their
    # point IDs. Sorting the _full_ node topology (not just the corner nodes)
    # makes sure that faces are only matched when all of their nodes,
    # including higher-order mid nodes, coincide.
    face_owners: dict[tuple, list] = {}
    gcell = 0
    for k in block_ids:
        block = mesh.cells[k]
        table = boundary_faces[block.type]
        for cell in block.data:
            for local_face, (_, local_nodes) in enumerate(table):
                key = tuple(sorted(cell[local_nodes].tolist()))
                face_owners.setdefault(key, []).append((gcell, local_face))
            gcell += 1

    # Decide what to do with each face; the first entry of the info tuple is
    # the "emitter", the parent whose local orientation defines the output
    # face orientation.
    face_info: dict[tuple, tuple] = {}
    for key, owners in face_owners.items():
        if len(owners) == 1:
            # outer boundary
            face_info[key] = (owners[0], None, False)
        elif len(owners) == 2:
            (g0, f0), (g1, f1) = owners
            if g0 == g1:
                # degenerate: the same cell on both sides
                continue
            if (
                material is not None
                and include_interfaces
                and material[g0] != material[g1]
            ):
                # internal interface; orient from the smaller material tag
                if material[g0] <= material[g1]:
                    face_info[key] = ((g0, f0), (g1, f1), True)
                else:
                    face_info[key] = ((g1, f1), (g0, f0), True)
            # else: interior face, cancels out
        else:
            warn(
                f"Face shared by {len(owners)} cells (non-manifold mesh); "
                "treating it as a boundary face."
            )
            face_info[key] = (owners[0], owners[1], False)

    # Pass 2: emit the faces in deterministic (cell, local face) order
    out: dict[str, list] = {}
    gcell = 0
    for k in block_ids:
        block = mesh.cells[k]
        table = boundary_faces[block.type]
        for cell in block.data:
            for local_face, (face_type, local_nodes) in enumerate(table):
                key = tuple(sorted(cell[local_nodes].tolist()))
                info = face_info.get(key)
                if info is None or info[0] != (gcell, local_face):
                    continue
                _, parent1, is_interface = info
                out.setdefault(face_type, []).append(
                    (cell[local_nodes], gcell, local_face, parent1, is_interface)
                )
            gcell += 1

    cells = []
    cell_data = {
        "parent_block": [],
        "parent_cell": [],
        "parent_face": [],
        "is_interface": [],
    }
    if material is not None:
        cell_data["material"] = []

    for face_type, faces in out.items():
        cells.append(CellBlock(face_type, np.array([f[0] for f in faces])))

        num = len(faces)
        parent_block = np.full((num, 2), -1, dtype=int)
        parent_cell = np.full((num, 2), -1, dtype=int)
        parent_face = np.full((num, 2), -1, dtype=int)
        is_interface = np.zeros(num, dtype=bool)
        mat = np.full((num, 2), -1, dtype=int) if material is not None else None

        for row, (_, g0, f0, parent1, is_int) in enumerate(faces):
            parent_block[row, 0] = block_ids[cell_block[g0]]
            parent_cell[row, 0] = cell_index[g0]
            parent_face[row, 0] = f0
            if mat is not None:
                mat[row, 0] = material[g0]
            if parent1 is not None:
                g1, f1 = parent1
                parent_block[row, 1] = block_ids[cell_block[g1]]
                parent_cell[row, 1] = cell_index[g1]
                parent_face[row, 1] = f1
                if mat is not None:
                    mat[row, 1] = material[g1]
            is_interface[row] = is_int

        cell_data["parent_block"].append(parent_block)
        cell_data["parent_cell"].append(parent_cell)
        cell_data["parent_face"].append(parent_face)
        cell_data["is_interface"].append(is_interface)
        if mat is not None:
            cell_data["material"].append(mat)

    return Mesh(
        points=np.array(mesh.points, copy=True), cells=cells, cell_data=cell_data
    )
