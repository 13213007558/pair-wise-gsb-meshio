import numpy as np
import pytest

import meshio

from . import helpers

test_set_full = [
    helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.line_tri_mesh,
    helpers.tri_mesh_2d,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
]

test_set_reduced = [
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.quad_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.hex_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
]


@pytest.mark.parametrize("mesh", test_set_full)
@pytest.mark.parametrize(
    "kwargs0",
    [
        {"data_format": "XML"},
        {"data_format": "Binary"},
        {"data_format": "HDF", "compression": None},
        {"data_format": "HDF", "compression": "gzip"},
    ],
)
def test_xdmf3(mesh, kwargs0, tmp_path):
    def write(*args, **kwargs):
        return meshio.xdmf.write(*args, **{**kwargs0, **kwargs})

    helpers.write_read(tmp_path, write, meshio.xdmf.read, mesh, 1.0e-14)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.xdmf")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.xdmf")


def test_time_series():
    # write the data
    filename = "out.xdmf"

    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(helpers.tri_mesh_2d.points, helpers.tri_mesh_2d.cells)
        n = helpers.tri_mesh_2d.points.shape[0]

        times = np.linspace(0.0, 1.0, 5)
        point_data = [
            {
                "phi": np.full(n, t),
                "u": np.full(helpers.tri_mesh_2d.points.shape, t),
            }
            for t in times
        ]
        for t, pd in zip(times, point_data):
            writer.write_data(
                t, point_data=pd, cell_data={"a": {"triangle": [3.0, 4.2]}}
            )

    # read it back in
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, pd, cd = reader.read_data(k)
            assert np.abs(times[k] - t) < 1.0e-12
            for key, value in pd.items():
                assert np.all(np.abs(value - point_data[k][key]) < 1.0e-12)


# def test_information_xdmf():
#     mesh_out = meshio.Mesh(
#         np.array(
#             [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
#         )
#         / 3,
#         [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
#         field_data={
#             "bottom": np.array([1, 1]),
#             "right": np.array([2, 1]),
#             "top": np.array([3, 1]),
#             "left": np.array([4, 1]),
#         },
#     )
#     # write the data
#     points, cells, field_data = mesh_out.points, mesh_out.cells, mesh_out.field_data
#
#     assert cells[0].type == "triangle"
#     meshio.write(
#         "mesh.xdmf",
#         meshio.Mesh(points=points, cells=[cells[0]], field_data=field_data),
#     )
#
#     # read it back in
#     mesh_in = meshio.read("mesh.xdmf")
#     assert len(mesh_in.field_data) == len(mesh_out.field_data)


if __name__ == "__main__":
    test_time_series()


class _DataItemFactory:
    # Builds XDMF <DataItem> XML, either with inline XML data or with the data
    # written to an HDF5 file.
    def __init__(self, tmp_path, data_format):
        self.tmp_path = tmp_path
        self.data_format = data_format
        self.h5_file = None
        self.counter = 0

    def __call__(self, dims, values, data_type="Float", precision="8"):
        values = np.asarray(values)
        dims_str = " ".join(str(d) for d in dims)
        if self.data_format == "XML":
            text = " ".join(str(v) for v in values.ravel())
            return (
                f'<DataItem DataType="{data_type}" Precision="{precision}" '
                f'Dimensions="{dims_str}" Format="XML">{text}</DataItem>'
            )
        import h5py

        if self.h5_file is None:
            self.h5_file = h5py.File(self.tmp_path / "data.h5", "w")
        name = f"data{self.counter}"
        self.counter += 1
        self.h5_file.create_dataset(name, data=values)
        return (
            f'<DataItem DataType="{data_type}" Precision="{precision}" '
            f'Dimensions="{dims_str}" Format="HDF">data.h5:/{name}</DataItem>'
        )

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()


def _read_structured(tmp_path, topology, geometry, attributes=""):
    (tmp_path / "mesh.xdmf").write_text(
        '<Xdmf Version="3.0"><Domain><Grid Name="Grid">'
        + topology
        + geometry
        + attributes
        + "</Grid></Domain></Xdmf>"
    )
    return meshio.read(tmp_path / "mesh.xdmf")


def _markers(dims):
    # unique marker per grid position, encoding the (x, y, z) index such that
    # any axis swap or reordering shows up in the values
    grids = np.indices(dims)
    marker = np.zeros(dims, dtype=int)
    for k, grid in enumerate(grids[::-1]):
        marker += grid * 100**k
    return marker


def _attributes(data_items, point_marker, cell_marker):
    return (
        '<Attribute Name="pm" AttributeType="Scalar" Center="Node">'
        + data_items(point_marker.shape, point_marker, "Int", "4")
        + "</Attribute>"
        + '<Attribute Name="cm" AttributeType="Scalar" Center="Cell">'
        + data_items(cell_marker.shape, cell_marker, "Int", "4")
        + "</Attribute>"
    )


@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_corect_mesh_3d(tmp_path, data_format):
    nx, ny, nz = 4, 3, 2
    origin = (1.0, 10.0, 100.0)
    spacing = (0.5, 2.0, 3.0)
    data_items = _DataItemFactory(tmp_path, data_format)
    topology = '<Topology TopologyType="3DCoRectMesh" Dimensions="2 3 4"/>'
    geometry = (
        '<Geometry GeometryType="ORIGIN_DXDYDZ">'
        + data_items((3,), origin)
        + data_items((3,), spacing)
        + "</Geometry>"
    )
    point_marker = _markers((nz, ny, nx))
    cell_marker = _markers((nz - 1, ny - 1, nx - 1))
    attributes = _attributes(data_items, point_marker, cell_marker)
    data_items.close()

    mesh = _read_structured(tmp_path, topology, geometry, attributes)

    xs = origin[0] + spacing[0] * np.arange(nx)
    ys = origin[1] + spacing[1] * np.arange(ny)
    zs = origin[2] + spacing[2] * np.arange(nz)
    expected_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    assert mesh.points.shape == (nx * ny * nz, 3)
    assert np.allclose(mesh.points, expected_points)

    assert [cb.type for cb in mesh.cells] == ["hexahedron"]
    index = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    expected_conn = np.array(
        [
            [
                index[iz, iy, ix],
                index[iz, iy, ix + 1],
                index[iz, iy + 1, ix + 1],
                index[iz, iy + 1, ix],
                index[iz + 1, iy, ix],
                index[iz + 1, iy, ix + 1],
                index[iz + 1, iy + 1, ix + 1],
                index[iz + 1, iy + 1, ix],
            ]
            for iz in range(nz - 1)
            for iy in range(ny - 1)
            for ix in range(nx - 1)
        ]
    )
    assert np.array_equal(mesh.cells[0].data, expected_conn)

    assert np.array_equal(mesh.point_data["pm"], point_marker.ravel())
    assert np.array_equal(mesh.cell_data["cm"][0], cell_marker.ravel())


@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_rect_mesh_3d(tmp_path, data_format):
    # non-uniform per-axis coordinates
    xs = [0.0, 1.0, 3.0, 6.0]
    ys = [10.0, 11.0, 14.0]
    zs = [100.0, 103.0]
    nx, ny, nz = len(xs), len(ys), len(zs)
    data_items = _DataItemFactory(tmp_path, data_format)
    topology = '<Topology TopologyType="3DRectMesh" Dimensions="2 3 4"/>'
    geometry = (
        '<Geometry GeometryType="VXVYVZ">'
        + data_items((nx,), xs)
        + data_items((ny,), ys)
        + data_items((nz,), zs)
        + "</Geometry>"
    )
    point_marker = _markers((nz, ny, nx))
    cell_marker = _markers((nz - 1, ny - 1, nx - 1))
    attributes = _attributes(data_items, point_marker, cell_marker)
    data_items.close()

    mesh = _read_structured(tmp_path, topology, geometry, attributes)

    expected_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    assert np.allclose(mesh.points, expected_points)
    assert [cb.type for cb in mesh.cells] == ["hexahedron"]
    assert mesh.cells[0].data.shape == ((nx - 1) * (ny - 1) * (nz - 1), 8)
    # spot-check one interior hex against the marker values
    index = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    assert np.array_equal(
        mesh.cells[0].data[0],
        [
            index[0, 0, 0],
            index[0, 0, 1],
            index[0, 1, 1],
            index[0, 1, 0],
            index[1, 0, 0],
            index[1, 0, 1],
            index[1, 1, 1],
            index[1, 1, 0],
        ],
    )
    assert np.array_equal(mesh.point_data["pm"], point_marker.ravel())
    assert np.array_equal(mesh.cell_data["cm"][0], cell_marker.ravel())


@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_smesh_3d(tmp_path, data_format):
    nx, ny, nz = 4, 3, 2
    # non-separable (curvilinear) coordinates
    coords = np.zeros((nz, ny, nx, 3))
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                coords[iz, iy, ix] = [
                    ix + 0.1 * iy,
                    10.0 + iy + 0.2 * iz,
                    100.0 + iz + 0.3 * ix,
                ]
    data_items = _DataItemFactory(tmp_path, data_format)
    topology = '<Topology TopologyType="3DSMesh" Dimensions="2 3 4"/>'
    geometry = (
        '<Geometry GeometryType="XYZ">'
        + data_items((nz, ny, nx, 3), coords)
        + "</Geometry>"
    )
    point_marker = _markers((nz, ny, nx))
    cell_marker = _markers((nz - 1, ny - 1, nx - 1))
    attributes = _attributes(data_items, point_marker, cell_marker)
    data_items.close()

    mesh = _read_structured(tmp_path, topology, geometry, attributes)

    assert np.allclose(mesh.points, coords.reshape(-1, 3))
    assert [cb.type for cb in mesh.cells] == ["hexahedron"]
    assert mesh.cells[0].data.shape == ((nx - 1) * (ny - 1) * (nz - 1), 8)
    assert np.array_equal(mesh.point_data["pm"], point_marker.ravel())
    assert np.array_equal(mesh.cell_data["cm"][0], cell_marker.ravel())


@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_corect_mesh_2d(tmp_path, data_format):
    nx, ny = 4, 3
    origin = (1.0, 10.0)
    spacing = (0.5, 2.0)
    data_items = _DataItemFactory(tmp_path, data_format)
    topology = '<Topology TopologyType="2DCoRectMesh" Dimensions="3 4"/>'
    geometry = (
        '<Geometry GeometryType="ORIGIN_DXDY">'
        + data_items((2,), origin)
        + data_items((2,), spacing)
        + "</Geometry>"
    )
    point_marker = _markers((ny, nx))
    cell_marker = _markers((ny - 1, nx - 1))
    attributes = _attributes(data_items, point_marker, cell_marker)
    data_items.close()

    mesh = _read_structured(tmp_path, topology, geometry, attributes)

    xs = origin[0] + spacing[0] * np.arange(nx)
    ys = origin[1] + spacing[1] * np.arange(ny)
    expected_points = np.array([[x, y] for y in ys for x in xs])
    assert mesh.points.shape == (nx * ny, 2)
    assert np.allclose(mesh.points, expected_points)

    assert [cb.type for cb in mesh.cells] == ["quad"]
    index = np.arange(nx * ny).reshape(ny, nx)
    expected_conn = np.array(
        [
            [
                index[iy, ix],
                index[iy, ix + 1],
                index[iy + 1, ix + 1],
                index[iy + 1, ix],
            ]
            for iy in range(ny - 1)
            for ix in range(nx - 1)
        ]
    )
    assert np.array_equal(mesh.cells[0].data, expected_conn)
    assert np.array_equal(mesh.point_data["pm"], point_marker.ravel())
    assert np.array_equal(mesh.cell_data["cm"][0], cell_marker.ravel())


@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_rect_mesh_2d(tmp_path, data_format):
    xs = [0.0, 1.0, 3.0, 6.0]
    ys = [10.0, 11.0, 14.0]
    nx, ny = len(xs), len(ys)
    data_items = _DataItemFactory(tmp_path, data_format)
    topology = '<Topology TopologyType="2DRectMesh" Dimensions="3 4"/>'
    geometry = (
        '<Geometry GeometryType="VXVY">'
        + data_items((nx,), xs)
        + data_items((ny,), ys)
        + "</Geometry>"
    )
    point_marker = _markers((ny, nx))
    cell_marker = _markers((ny - 1, nx - 1))
    attributes = _attributes(data_items, point_marker, cell_marker)
    data_items.close()

    mesh = _read_structured(tmp_path, topology, geometry, attributes)

    expected_points = np.array([[x, y] for y in ys for x in xs])
    assert np.allclose(mesh.points, expected_points)
    assert [cb.type for cb in mesh.cells] == ["quad"]
    assert mesh.cells[0].data.shape == ((nx - 1) * (ny - 1), 4)
    assert np.array_equal(mesh.point_data["pm"], point_marker.ravel())
    assert np.array_equal(mesh.cell_data["cm"][0], cell_marker.ravel())


@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_smesh_2d(tmp_path, data_format):
    nx, ny = 4, 3
    coords = np.zeros((ny, nx, 2))
    for iy in range(ny):
        for ix in range(nx):
            coords[iy, ix] = [ix + 0.25 * iy, 10.0 + iy + 0.5 * ix]
    data_items = _DataItemFactory(tmp_path, data_format)
    topology = '<Topology TopologyType="2DSMesh" Dimensions="3 4"/>'
    geometry = (
        '<Geometry GeometryType="XY">' + data_items((ny, nx, 2), coords) + "</Geometry>"
    )
    point_marker = _markers((ny, nx))
    cell_marker = _markers((ny - 1, nx - 1))
    attributes = _attributes(data_items, point_marker, cell_marker)
    data_items.close()

    mesh = _read_structured(tmp_path, topology, geometry, attributes)

    assert np.allclose(mesh.points, coords.reshape(-1, 2))
    assert [cb.type for cb in mesh.cells] == ["quad"]
    assert mesh.cells[0].data.shape == ((nx - 1) * (ny - 1), 4)
    assert np.array_equal(mesh.point_data["pm"], point_marker.ravel())
    assert np.array_equal(mesh.cell_data["cm"][0], cell_marker.ravel())


def test_xdmf_structured_degenerate_axis(tmp_path):
    # one axis with a single node: drop down to lower-dimensional cells
    data_items = _DataItemFactory(tmp_path, "XML")
    topology = '<Topology TopologyType="3DCoRectMesh" Dimensions="1 3 4"/>'
    geometry = (
        '<Geometry GeometryType="ORIGIN_DXDYDZ">'
        + data_items((3,), (1.0, 10.0, 100.0))
        + data_items((3,), (0.5, 2.0, 3.0))
        + "</Geometry>"
    )
    data_items.close()
    mesh = _read_structured(tmp_path, topology, geometry)

    assert mesh.points.shape == (12, 3)
    assert np.allclose(mesh.points[:, 2], 100.0)
    assert [cb.type for cb in mesh.cells] == ["quad"]
    index = np.arange(12).reshape(3, 4)
    expected_conn = np.array(
        [
            [
                index[iy, ix],
                index[iy, ix + 1],
                index[iy + 1, ix + 1],
                index[iy + 1, ix],
            ]
            for iy in range(2)
            for ix in range(3)
        ]
    )
    assert np.array_equal(mesh.cells[0].data, expected_conn)

    # two degenerate axes: line cells
    topology = '<Topology TopologyType="3DCoRectMesh" Dimensions="1 1 4"/>'
    mesh = _read_structured(tmp_path, topology, geometry)
    assert mesh.points.shape == (4, 3)
    assert [cb.type for cb in mesh.cells] == ["line"]
    assert np.array_equal(mesh.cells[0].data, [[0, 1], [1, 2], [2, 3]])
