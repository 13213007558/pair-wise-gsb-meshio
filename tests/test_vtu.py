import pathlib

import numpy as np
import pytest

import meshio

from . import helpers

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


def _mixed_polyhedron_mesh():
    # A mesh mixing polyhedra with regular cells:
    # - two consecutive tetra blocks (separated by other blocks)
    # - polyhedra with 4, 5, and 6 faces (tetrahedron, pyramid, cube)
    # - a hexahedron block with two cells
    # - an empty cell block
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [3.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
    )

    def cube_faces(offset):
        o = offset
        return [
            [o + 0, o + 1, o + 2, o + 3],
            [o + 4, o + 5, o + 6, o + 7],
            [o + 0, o + 1, o + 5, o + 4],
            [o + 2, o + 3, o + 7, o + 6],
            [o + 1, o + 2, o + 6, o + 5],
            [o + 0, o + 3, o + 7, o + 4],
        ]

    cells = [
        ("tetra", np.array([[0, 1, 2, 4], [1, 2, 3, 6]])),
        ("polyhedron4", [[[0, 1, 2], [0, 1, 4], [1, 2, 4], [0, 2, 4]]]),
        (
            "polyhedron5",
            [[[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]],
        ),
        ("polyhedron8", [cube_faces(8), cube_faces(8)]),
        (
            "hexahedron",
            np.array([[8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7]]),
        ),
        ("tetra", np.array([[8, 9, 10, 12]])),
        ("tetra", np.empty((0, 4), dtype=int)),
    ]
    # one scalar value per cell, equal to the global cell index; the empty
    # block contributes no values
    cell_data = {
        "a": [
            np.array([0, 1]),
            np.array([2]),
            np.array([3]),
            np.array([4, 5]),
            np.array([6, 7]),
            np.array([8]),
            np.empty((0,)),
        ]
    }
    return meshio.Mesh(points, cells, cell_data=cell_data)


# expected block structure after reading back; the empty block vanishes, the
# two tetra blocks stay separate since they are not consecutive in the file
_mixed_polyhedron_ref_blocks = [
    ("tetra", 2),
    ("polyhedron4", 1),
    ("polyhedron5", 1),
    ("polyhedron8", 2),
    ("hexahedron", 2),
    ("tetra", 1),
]


@pytest.mark.parametrize(
    "data_type", [(False, None), (True, None), (True, "zlib"), (True, "lzma")]
)
def test_mixed_polyhedron(data_type, tmp_path):
    binary, compression = data_type
    mesh = _mixed_polyhedron_mesh()

    path = tmp_path / "mixed.vtu"
    meshio.vtu.write(path, mesh, binary=binary, compression=compression)
    out = meshio.vtu.read(path)

    assert [(c.type, len(c.data)) for c in out.cells] == _mixed_polyhedron_ref_blocks

    # regular cell blocks keep their connectivity
    assert np.array_equal(out.cells[0].data, mesh.cells[0].data)
    assert np.array_equal(out.cells[4].data, mesh.cells[4].data)
    assert np.array_equal(out.cells[5].data, mesh.cells[5].data)

    # polyhedron faces keep their orientation and node order
    for out_block, in_block in zip(
        [out.cells[1], out.cells[2], out.cells[3]],
        [mesh.cells[1], mesh.cells[2], mesh.cells[3]],
    ):
        for out_cell, in_cell in zip(out_block.data, in_block.data):
            assert len(out_cell) == len(in_cell)
            for out_face, in_face in zip(out_cell, in_cell):
                assert np.array_equal(out_face, np.asarray(in_face))

    # cell data stays aligned with the cells
    assert [d.tolist() for d in out.cell_data["a"]] == [
        [0, 1],
        [2],
        [3],
        [4, 5],
        [6, 7],
        [8],
    ]


def test_mixed_polyhedron_file_structure(tmp_path):
    # Independently check the structure of the written file (without going
    # through the meshio reader): regular cells must carry a -1 placeholder
    # in the face stream, polyhedra their face description.
    from xml.etree import ElementTree as ET

    mesh = _mixed_polyhedron_mesh()
    path = tmp_path / "mixed.vtu"
    meshio.vtu.write(path, mesh, binary=False)

    root = ET.parse(path).getroot()
    names = {"connectivity", "offsets", "types", "faces", "faceoffsets"}
    arrays = {
        da.attrib["Name"]: np.fromstring(da.text, dtype=int, sep=" ")
        for da in root.iter("DataArray")
        if da.attrib.get("Name") in names
    }

    # vtk types: 10 = tetra, 12 = hexahedron, 42 = polyhedron
    types = arrays["types"]
    assert types.tolist() == [10, 10, 42, 42, 42, 42, 12, 12, 10]
    assert len(arrays["offsets"]) == len(types)

    faces = arrays["faces"]
    faceoffsets = arrays["faceoffsets"]
    assert len(faceoffsets) == len(types)
    # faceoffsets point to the end of each cell's face stream
    assert faceoffsets[-1] == len(faces)

    starts = np.concatenate([[0], faceoffsets[:-1]])
    for k, t in enumerate(types):
        stream = faces[starts[k] : faceoffsets[k]]
        if t == 42:
            # first entry is the number of faces of the polyhedron
            assert stream[0] > 0
        else:
            # regular cells are represented by a single -1 placeholder
            assert stream.tolist() == [-1]

    # the face stream of the first polyhedron (a tetrahedron with 4
    # triangular faces) keeps its node order
    assert faces[starts[2] : faceoffsets[2]].tolist() == [
        4,
        3,
        0,
        1,
        2,
        3,
        0,
        1,
        4,
        3,
        1,
        2,
        4,
        3,
        0,
        2,
        4,
    ]


@pytest.mark.parametrize("data_type", [(False, None), (True, None), (True, "zlib")])
def test_mixed_polyhedron_vtk_verification(data_type, tmp_path):
    # Read the meshio-written file with VTK itself and check cell types and
    # counts independently of meshio's reader.
    vtk = pytest.importorskip("vtk")

    binary, compression = data_type
    mesh = _mixed_polyhedron_mesh()

    path = tmp_path / "mixed.vtu"
    meshio.vtu.write(path, mesh, binary=binary, compression=compression)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    assert grid.GetNumberOfPoints() == 16
    assert grid.GetNumberOfCells() == 9

    ref_types = [
        vtk.VTK_TETRA,
        vtk.VTK_TETRA,
        vtk.VTK_POLYHEDRON,
        vtk.VTK_POLYHEDRON,
        vtk.VTK_POLYHEDRON,
        vtk.VTK_POLYHEDRON,
        vtk.VTK_HEXAHEDRON,
        vtk.VTK_HEXAHEDRON,
        vtk.VTK_TETRA,
    ]
    types = [grid.GetCellType(k) for k in range(grid.GetNumberOfCells())]
    assert types == ref_types
    assert types.count(vtk.VTK_POLYHEDRON) == 4

    # number of faces of the polyhedra: tetrahedron, pyramid, two cubes
    num_faces = []
    for k in range(grid.GetNumberOfCells()):
        if grid.GetCellType(k) == vtk.VTK_POLYHEDRON:
            faces = grid.GetFaces(k)
            num_faces.append(faces.GetValue(0))
    assert num_faces == [4, 5, 6, 6]


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
