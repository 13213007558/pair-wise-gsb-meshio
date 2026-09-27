import pathlib

import numpy as np
import pytest

import meshio

from . import helpers

# A mesh that mixes polyhedral and other cells: Two consecutive blocks of
# standard cells (tetra, hexahedron), two polyhedron blocks (with differing
# numbers of faces per cell), and another standard block at the end.
mixed_polyhedron_mesh = meshio.Mesh(
    [  # Two unit cubes, side by side
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    [
        ("tetra", [[0, 1, 3, 4], [1, 2, 3, 6]]),
        ("hexahedron", [[8, 9, 10, 11, 12, 13, 14, 15]]),
        (
            "polyhedron4",
            [
                [
                    [1, 2, 5],
                    [1, 2, 7],
                    [1, 5, 7],
                    [2, 5, 7],
                ],
                [
                    [2, 5, 6],
                    [2, 6, 7],
                    [2, 5, 7],
                    [5, 6, 7],
                ],
            ],
        ),
        (
            "polyhedron5",
            [
                [
                    [0, 1, 2, 3],
                    [0, 1, 7],
                    [1, 2, 7],
                    [2, 3, 7],
                    [3, 0, 7],
                ],
                [
                    [0, 1, 5],
                    [0, 4, 5],
                    [0, 1, 7],
                    [1, 5, 7],
                    [5, 4, 7],
                    [0, 4, 7],
                ],
            ],
        ),
        ("tetra", [[9, 10, 12, 15]]),
    ],
    cell_data={
        "a": [
            np.array([1.0, 2.0]),
            np.array([3.0]),
            np.array([4.0, 5.0]),
            np.array([6.0, 7.0]),
            np.array([8.0]),
        ]
    },
)

test_set = [
    # helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_one_cell,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.polygon_mesh,
    helpers.polygon_mesh_one_cell,
    helpers.polygon2_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
    helpers.polyhedron_mesh,
    helpers.lagrange_high_order_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_quad_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float32)]),
    helpers.add_cell_data(helpers.tri_mesh, [("b", (3,), np.float64)]),
    helpers.add_cell_data(helpers.polygon_mesh, [("a", (), np.float32)]),
    helpers.add_cell_data(helpers.polyhedron_mesh, [("a", (2,), np.float32)]),
    mixed_polyhedron_mesh,
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize(
    "data_type", [(False, None), (True, None), (True, "lzma"), (True, "zlib")]
)
def test(mesh, data_type, tmp_path):
    binary, compression = data_type

    def writer(*args, **kwargs):
        return meshio.vtu.write(*args, binary=binary, compression=compression, **kwargs)

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/-/issues/17038#note_264052>
    tol = 1.0e-15 if binary else 1.0e-10
    helpers.write_read(tmp_path, writer, meshio.vtu.read, mesh, tol)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.vtu")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.vtu")


@pytest.mark.parametrize(
    "filename, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("00_raw_binary.vtu", "tetra", 162, 64),
        ("01_raw_binary_int64.vtu", "tetra", 162, 64),
        ("02_raw_compressed.vtu", "tetra", 162, 64),
    ],
)
def test_read_from_file(filename, ref_cells, ref_num_cells, ref_num_pnt):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtu" / filename

    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


def _independent_vtu_parse(filename):
    """Parse the cell section of a VTU file straight from the VTK XML file format
    specification (<https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>),
    without any meshio code. This serves as an independent check of what meshio
    writes; meshio's own reader is not involved.
    """
    import base64
    import lzma as lzma_module
    import zlib
    from xml.etree import ElementTree as ET

    vtk_to_numpy = {
        "Float32": np.float32,
        "Float64": np.float64,
        "Int8": np.int8,
        "Int16": np.int16,
        "Int32": np.int32,
        "Int64": np.int64,
        "UInt8": np.uint8,
        "UInt16": np.uint16,
        "UInt32": np.uint32,
        "UInt64": np.uint64,
    }

    root = ET.parse(filename).getroot()
    assert root.tag == "VTKFile"
    header_dtype = np.dtype(vtk_to_numpy[root.get("header_type", "UInt32")])
    compressor = root.get("compressor")

    def read_data_array(da):
        dtype = np.dtype(vtk_to_numpy[da.get("type")])
        fmt = da.get("format", "ascii")
        if fmt == "ascii":
            text = (da.text or "").strip()
            if text == "":
                return np.empty(0, dtype=dtype)
            return np.fromstring(text, sep=" ", dtype=dtype)

        assert fmt == "binary"
        text = da.text.strip()
        itemsize = header_dtype.itemsize
        if compressor is None:
            # The first header item is the number of bytes of the data
            raw = base64.b64decode(text)
            num_bytes = int(np.frombuffer(raw[:itemsize], header_dtype)[0])
            return np.frombuffer(raw[itemsize : itemsize + num_bytes], dtype).copy()

        # Compressed data: the header is num_blocks, the (uncompressed) block
        # size, the size of the last block, and the compressed sizes of all
        # blocks. Header and block data are base64-encoded separately.
        def num_base64_chars(num_bytes):
            return -(-num_bytes // 3) * 4

        first = base64.b64decode(text[: num_base64_chars(itemsize)])
        num_blocks = int(np.frombuffer(first[:itemsize], header_dtype)[0])
        num_header_bytes = itemsize * (3 + num_blocks)
        num_header_chars = num_base64_chars(num_header_bytes)
        header = np.frombuffer(base64.b64decode(text[:num_header_chars]), header_dtype)
        decompress = {
            "vtkZLibDataCompressor": zlib.decompress,
            "vtkLZMADataCompressor": lzma_module.decompress,
        }[compressor]
        body = base64.b64decode(text[num_header_chars:])
        data = b""
        pos = 0
        for compressed_size in header[3:]:
            data += decompress(body[pos : pos + int(compressed_size)])
            pos += int(compressed_size)
        return np.frombuffer(data, dtype).copy()

    piece = root.find("UnstructuredGrid").find("Piece")
    cell_arrays = {da.get("Name"): read_data_array(da) for da in piece.find("Cells")}
    cell_data_tag = piece.find("CellData")
    cell_data = {}
    if cell_data_tag is not None:
        cell_data = {da.get("Name"): read_data_array(da) for da in cell_data_tag}
    return cell_arrays, cell_data, int(piece.get("NumberOfCells"))


def _cells_from_spec(cell_arrays):
    """Reconstruct the cells (type, connectivity, faces) from the VTK arrays,
    following the file format specification. In the faces stream, every cell is
    represented: polyhedra by their face description, all other cells by a
    single 0.
    """
    connectivity = cell_arrays["connectivity"]
    offsets = cell_arrays["offsets"]
    types = cell_arrays["types"]
    faces = cell_arrays.get("faces")
    faceoffsets = cell_arrays.get("faceoffsets")

    cells = []
    conn_start = 0
    face_start = 0
    for cell_id, vtk_type in enumerate(types):
        conn = connectivity[conn_start : offsets[cell_id]]
        conn_start = offsets[cell_id]

        cell_faces = []
        if faces is not None:
            face_end = faceoffsets[cell_id]
            stream = faces[face_start:face_end]
            face_start = face_end
            num_faces = stream[0]
            pos = 1
            for _ in range(num_faces):
                num_nodes = stream[pos]
                cell_faces.append(stream[pos + 1 : pos + 1 + num_nodes].tolist())
                pos += num_nodes + 1
            # The face description of the cell must be consumed exactly
            assert pos == len(stream)

        cells.append((int(vtk_type), conn.tolist(), cell_faces))
    return cells


# VTK cell types, cf. vtkCellType.h
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_POLYHEDRON = 42

# The expected cells of mixed_polyhedron_mesh, in file order
mixed_polyhedron_expected_types = [
    VTK_TETRA,
    VTK_TETRA,
    VTK_HEXAHEDRON,
    VTK_POLYHEDRON,
    VTK_POLYHEDRON,
    VTK_POLYHEDRON,
    VTK_POLYHEDRON,
    VTK_TETRA,
]


@pytest.mark.parametrize(
    "binary,compression",
    [(False, None), (True, None), (True, "zlib"), (True, "lzma")],
)
def test_mixed_polyhedron_independent_read(tmp_path, binary, compression):
    # Write the mixed mesh and check the file with an independent, spec-based
    # parser (not meshio's reader).
    filename = tmp_path / "mixed.vtu"
    meshio.vtu.write(
        filename, mixed_polyhedron_mesh, binary=binary, compression=compression
    )

    cell_arrays, cell_data, num_cells = _independent_vtu_parse(filename)

    # cell types and their number
    assert num_cells == len(mixed_polyhedron_expected_types)
    cells = _cells_from_spec(cell_arrays)
    assert [cell[0] for cell in cells] == mixed_polyhedron_expected_types

    # connectivity of the standard cells; the polyhedra are represented by
    # their (sorted) unique nodes
    expected_connectivity = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 5, 7],
        [2, 5, 6, 7],
        [0, 1, 2, 3, 7],
        [0, 1, 4, 5, 7],
        [9, 10, 12, 15],
    ]
    assert [cell[1] for cell in cells] == expected_connectivity

    # The faces stream: standard cells are represented by a single 0 (no
    # faces), the polyhedra keep their faces exactly as given, including the
    # orientation and order of the face nodes.
    expected_faces = [
        [],
        [],
        [],
        [[1, 2, 5], [1, 2, 7], [1, 5, 7], [2, 5, 7]],
        [[2, 5, 6], [2, 6, 7], [2, 5, 7], [5, 6, 7]],
        [[0, 1, 2, 3], [0, 1, 7], [1, 2, 7], [2, 3, 7], [3, 0, 7]],
        [[0, 1, 5], [0, 4, 5], [0, 1, 7], [1, 5, 7], [5, 4, 7], [0, 4, 7]],
        [],
    ]
    assert [cell[2] for cell in cells] == expected_faces

    # cell data, aligned with the cells
    assert np.allclose(
        cell_data["a"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], rtol=0.0
    )

    # Now read the file back with meshio and make sure the mesh is unchanged
    mesh = meshio.vtu.read(filename)
    assert [block.type for block in mesh.cells] == [
        "tetra",
        "hexahedron",
        "polyhedron4",
        "polyhedron5",
        "tetra",
    ]
    assert np.array_equal(mesh.cells[0].data, [[0, 1, 3, 4], [1, 2, 3, 6]])
    assert np.array_equal(mesh.cells[1].data, [[8, 9, 10, 11, 12, 13, 14, 15]])
    for block, ref in zip(mesh.cells[2:4], [expected_faces[3:5], expected_faces[5:7]]):
        for cell, ref_cell in zip(block.data, ref):
            assert len(cell) == len(ref_cell)
            for face, ref_face in zip(cell, ref_cell):
                assert np.array_equal(face, ref_face)
    assert np.array_equal(mesh.cells[4].data, [[9, 10, 12, 15]])
    for block_data, ref in zip(
        mesh.cell_data["a"], [[1.0, 2.0], [3.0], [4.0, 5.0], [6.0, 7.0], [8.0]]
    ):
        assert np.allclose(block_data, ref, rtol=0.0)


@pytest.mark.parametrize(
    "binary,compression", [(False, None), (True, None), (True, "zlib")]
)
def test_mixed_polyhedron_empty_cell_blocks(tmp_path, binary, compression):
    # Empty cell blocks (standard and polyhedron) between non-empty ones must
    # not disturb the cell stream.
    poly_cell = [[1, 2, 5], [1, 2, 7], [1, 5, 7], [2, 5, 7]]
    mesh = meshio.Mesh(
        mixed_polyhedron_mesh.points[:8],
        [
            ("tetra", np.empty((0, 4), dtype=int)),
            ("polyhedron4", []),
            ("tetra", [[0, 1, 3, 4]]),
            ("polyhedron4", [poly_cell]),
        ],
        cell_data={"a": [np.empty(0), np.empty(0), np.array([1.0]), np.array([2.0])]},
    )

    filename = tmp_path / "empty_blocks.vtu"
    meshio.vtu.write(filename, mesh, binary=binary, compression=compression)

    # independent check of the file content
    cell_arrays, cell_data, num_cells = _independent_vtu_parse(filename)
    assert num_cells == 2
    cells = _cells_from_spec(cell_arrays)
    assert [cell[0] for cell in cells] == [VTK_TETRA, VTK_POLYHEDRON]
    assert cells[0][2] == []
    assert cells[1][2] == poly_cell
    assert np.allclose(cell_data["a"], [1.0, 2.0], rtol=0.0)

    # meshio roundtrip
    out = meshio.vtu.read(filename)
    assert [block.type for block in out.cells] == ["tetra", "polyhedron4"]
    assert np.array_equal(out.cells[0].data, [[0, 1, 3, 4]])
    assert len(out.cells[1].data) == 1
    for face, ref_face in zip(out.cells[1].data[0], poly_cell):
        assert np.array_equal(face, ref_face)
    assert np.allclose(out.cell_data["a"][0], [1.0], rtol=0.0)
    assert np.allclose(out.cell_data["a"][1], [2.0], rtol=0.0)


def test_mixed_polyhedron_vtk_cross_check(tmp_path):
    # Cross-check the written file with VTK itself (if available): cell types
    # and their number must come out as expected.
    vtk = pytest.importorskip("vtk")

    filename = tmp_path / "mixed.vtu"
    meshio.vtu.write(filename, mixed_polyhedron_mesh, binary=True, compression="zlib")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    output = reader.GetOutput()

    assert output.GetNumberOfCells() == len(mixed_polyhedron_expected_types)
    assert [
        output.GetCellType(i) for i in range(output.GetNumberOfCells())
    ] == mixed_polyhedron_expected_types
