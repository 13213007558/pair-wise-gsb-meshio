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


# --- Structured grids (CoRectMesh, RectMesh, SMesh) ------------------------


def _expected_corner_offsets(cell_type, active_axes, ndim):
    zero = [0] * ndim
    if cell_type == "line":
        offsets = []
        for step in (0, 1):
            offset = list(zero)
            offset[active_axes[0]] = step
            offsets.append(tuple(offset))
        return offsets
    if cell_type == "quad":
        offsets = []
        for first, second in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            offset = list(zero)
            offset[active_axes[0]] = first
            offset[active_axes[1]] = second
            offsets.append(tuple(offset))
        return offsets
    if cell_type == "hexahedron":
        return [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
    return [tuple(zero)]


def _check_structured_mesh(mesh, axes, cell_type, weights=(1, 10, 100)):
    dims = [len(axis) for axis in axes]
    ndim = len(axes)

    assert isinstance(mesh, meshio.Mesh)
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == cell_type
    cells = mesh.cells[0].data

    # Every point maps to a unique (i, j[, k]) position.
    axis_lookup = [{value: i for i, value in enumerate(axis)} for axis in axes]
    positions = [
        tuple(axis_lookup[a][point[a]] for a in range(ndim)) for point in mesh.points
    ]
    assert len(set(positions)) == np.prod(dims)

    # Node data follows the same axis order as the coordinates.
    marker = mesh.point_data["marker"]
    for node, position in enumerate(positions):
        expected = sum(weights[a] * position[a] for a in range(ndim))
        assert marker[node] == expected

    active_axes = [a for a in range(ndim) if dims[a] > 1]
    offsets = _expected_corner_offsets(cell_type, active_axes, ndim)
    cell_marker = mesh.cell_data["cell_marker"][0]
    assert cells.shape[0] == np.prod([dims[a] - 1 for a in active_axes])

    for cell_id, node_ids in enumerate(cells):
        corners = [positions[node] for node in node_ids]
        assert len(set(corners)) == len(corners)
        minimum = tuple(min(corner[a] for corner in corners) for a in range(ndim))

        actual_offsets = [
            tuple(corner[a] - minimum[a] for a in range(ndim)) for corner in corners
        ]
        assert actual_offsets == offsets

        expected = sum(weights[a] * minimum[a] for a in active_axes)
        assert cell_marker[cell_id] == expected


def _data_item(name, data, data_format, h5_file, dtype="Float"):
    data = np.asarray(data)
    dims = " ".join(str(d) for d in data.shape)
    if data_format == "XML":
        if dtype == "Int":
            body = " ".join(str(v) for v in data.ravel())
        else:
            body = " ".join(repr(float(v)) for v in data.ravel())
        return (
            f'<DataItem DataType="{dtype}" Precision="8" '
            f'Dimensions="{dims}" Format="XML">{body}</DataItem>'
        )
    h5_file.create_dataset(name, data=data)
    filename = h5_file.filename.split("/")[-1]
    return (
        f'<DataItem DataType="{dtype}" Precision="8" '
        f'Dimensions="{dims}" Format="HDF">{filename}:/{name}</DataItem>'
    )


def _write_structured_xdmf(
    path, topology_type, axes, geometry, data_format="XML", version="3.0"
):
    import h5py

    dims = [len(axis) for axis in axes]
    ndim = len(axes)
    weights = [1, 10, 100][:ndim]

    h5_file = None
    if data_format == "HDF":
        h5_file = h5py.File(path.with_name("structured.h5"), "w")

    if geometry == "uniform":
        origin = np.array([axis[0] for axis in axes])
        displacement = np.array(
            [axis[1] - axis[0] if len(axis) > 1 else 0.0 for axis in axes]
        )
        geometry_type = "ORIGIN_DXDY" if ndim == 2 else "ORIGIN_DXDYDZ"
        items = [
            _data_item("origin", origin, data_format, h5_file),
            _data_item("displacement", displacement, data_format, h5_file),
        ]
    elif geometry == "per_axis":
        geometry_type = "VXVY" if ndim == 2 else "VXVYVZ"
        items = [
            _data_item(f"axis_{axis}", axes[axis], data_format, h5_file)
            for axis in range(ndim)
        ]
    else:
        assert geometry == "explicit"
        geometry_type = "XY" if ndim == 2 else "XYZ"
        grids = np.meshgrid(*axes, indexing="ij")
        # Storage is (n_k, n_j, n_i, 3), i.e. reversed topology axes.
        coordinates = np.column_stack(
            [grid.ravel(order="F") for grid in grids]
        ).reshape(tuple(reversed(dims)) + (ndim,))
        items = [_data_item("coordinates", coordinates, data_format, h5_file)]

    geometry_xml = (
        f'<Geometry GeometryType="{geometry_type}">' + "".join(items) + "</Geometry>"
    )

    # Unique marker per node/cell so axis swaps or ordering errors show up.
    node_index = np.indices(dims)
    node_marker = sum(weights[axis] * node_index[axis] for axis in range(ndim)).astype(
        np.int64
    )
    stored_node_marker = np.transpose(node_marker)

    cell_shape = tuple(max(d - 1, 1) for d in dims)
    active_axes = [a for a in range(ndim) if dims[a] > 1]
    cell_index = np.indices(cell_shape)
    cell_marker = np.zeros(cell_shape, dtype=np.int64)
    for axis in active_axes:
        cell_marker += weights[axis] * cell_index[axis]
    stored_cell_marker = np.transpose(cell_marker)

    attributes_xml = (
        '<Attribute Name="marker" Center="Node" AttributeType="Scalar">'
        + _data_item("node_marker", stored_node_marker, data_format, h5_file, "Int")
        + "</Attribute>"
        + '<Attribute Name="cell_marker" Center="Cell" AttributeType="Scalar">'
        + _data_item("cell_marker", stored_cell_marker, data_format, h5_file, "Int")
        + "</Attribute>"
    )

    if h5_file is not None:
        h5_file.close()

    topology_key = "TopologyType" if version == "2" else "Type"
    dims_string = " ".join(str(d) for d in dims)
    xml = (
        f'<?xml version="1.0"?>\n<Xdmf Version="{version}">\n<Domain>\n'
        '<Grid Name="Grid">\n'
        f'<Topology {topology_key}="{topology_type}" '
        f'Dimensions="{dims_string}"/>\n'
        f"{geometry_xml}\n{attributes_xml}\n"
        "</Grid>\n</Domain>\n</Xdmf>\n"
    )
    path.write_text(xml)


@pytest.mark.parametrize(
    ("topology_type", "geometry", "axes", "cell_type"),
    [
        (
            "3DCoRectMesh",
            "uniform",
            [
                np.array([0.0, 1.0]),
                np.array([10.0, 11.0, 12.0]),
                np.array([-2.0, 0.0, 2.0, 4.0]),
            ],
            "hexahedron",
        ),
        (
            "3DRectMesh",
            "per_axis",
            [
                np.array([0.0, 0.5]),
                np.array([1.0, 2.0, 5.0]),
                np.array([-1.0, 0.0, 0.3, 7.0]),
            ],
            "hexahedron",
        ),
        (
            "3DSMesh",
            "explicit",
            [
                np.array([0.0, 0.5]),
                np.array([1.0, 2.0, 5.0]),
                np.array([-1.0, 0.0, 0.3, 7.0]),
            ],
            "hexahedron",
        ),
        (
            "2DCoRectMesh",
            "uniform",
            [np.array([2.0, 2.5, 3.0]), np.array([-1.0, 1.0])],
            "quad",
        ),
        (
            "2DRectMesh",
            "per_axis",
            [np.array([0.0, 0.5, 2.5]), np.array([-1.0, 1.0])],
            "quad",
        ),
        (
            "2DSMesh",
            "explicit",
            [np.array([0.0, 0.5, 2.5]), np.array([-1.0, 1.0])],
            "quad",
        ),
    ],
)
@pytest.mark.parametrize("data_format", ["XML", "HDF"])
def test_xdmf_structured(
    tmp_path, topology_type, geometry, axes, cell_type, data_format
):
    path = tmp_path / "structured.xdmf"
    _write_structured_xdmf(path, topology_type, axes, geometry, data_format=data_format)
    mesh = meshio.xdmf.read(path)
    _check_structured_mesh(mesh, axes, cell_type)


def test_xdmf_structured_xdmf2(tmp_path):
    axes = [np.array([0.0, 0.5, 2.5]), np.array([-1.0, 1.0])]
    path = tmp_path / "structured.xdmf"
    _write_structured_xdmf(path, "2DRectMesh", axes, "per_axis", version="2")
    mesh = meshio.xdmf.read(path)
    _check_structured_mesh(mesh, axes, "quad")


def test_xdmf_structured_hdf_xml_equivalence(tmp_path):
    axes = [
        np.array([0.0, 0.5]),
        np.array([1.0, 2.0, 5.0]),
        np.array([-1.0, 0.0, 0.3, 7.0]),
    ]
    _write_structured_xdmf(
        tmp_path / "rect_xml.xdmf", "3DRectMesh", axes, "per_axis", data_format="XML"
    )
    _write_structured_xdmf(
        tmp_path / "rect_hdf.xdmf", "3DRectMesh", axes, "per_axis", data_format="HDF"
    )
    mesh_xml = meshio.xdmf.read(tmp_path / "rect_xml.xdmf")
    mesh_hdf = meshio.xdmf.read(tmp_path / "rect_hdf.xdmf")

    np.testing.assert_array_equal(mesh_xml.points, mesh_hdf.points)
    assert mesh_xml.cells[0].type == mesh_hdf.cells[0].type
    np.testing.assert_array_equal(mesh_xml.cells[0].data, mesh_hdf.cells[0].data)
    np.testing.assert_array_equal(
        mesh_xml.point_data["marker"], mesh_hdf.point_data["marker"]
    )
    np.testing.assert_array_equal(
        mesh_xml.cell_data["cell_marker"][0], mesh_hdf.cell_data["cell_marker"][0]
    )


@pytest.mark.parametrize(
    ("topology_type", "geometry", "axes", "cell_type"),
    [
        (
            "3DRectMesh",
            "per_axis",
            [
                np.array([5.0]),
                np.array([1.0, 2.0, 5.0]),
                np.array([-1.0, 7.0]),
            ],
            "quad",
        ),
        (
            "3DCoRectMesh",
            "uniform",
            [
                np.array([0.0, 1.0]),
                np.array([3.0]),
                np.array([-1.0, 0.0, 1.0]),
            ],
            "quad",
        ),
        (
            "3DRectMesh",
            "per_axis",
            [np.array([5.0]), np.array([2.0]), np.array([-1.0, 0.0, 0.3, 7.0])],
            "line",
        ),
        (
            "2DRectMesh",
            "per_axis",
            [np.array([1.0]), np.array([0.0, 0.5, 2.5])],
            "line",
        ),
        (
            "3DCoRectMesh",
            "uniform",
            [np.array([1.0]), np.array([2.0]), np.array([3.0])],
            "vertex",
        ),
    ],
)
def test_xdmf_structured_degenerate(tmp_path, topology_type, geometry, axes, cell_type):
    path = tmp_path / "degenerate.xdmf"
    _write_structured_xdmf(path, topology_type, axes, geometry)
    mesh = meshio.xdmf.read(path)
    _check_structured_mesh(mesh, axes, cell_type)


def test_xdmf_structured_unsupported_dimension(tmp_path):
    path = tmp_path / "unsupported.xdmf"
    path.write_text(
        '<?xml version="1.0"?>\n<Xdmf Version="3.0">\n<Domain>\n'
        '<Grid Name="Grid">\n'
        '<Topology Type="3DRectMesh" Dimensions="2 2 2 2"/>\n'
        "</Grid>\n</Domain>\n</Xdmf>\n"
    )
    with pytest.raises(meshio.ReadError):
        meshio.xdmf.read(path)


def test_xdmf_structured_axis_length_mismatch(tmp_path):
    path = tmp_path / "mismatch.xdmf"
    path.write_text(
        '<?xml version="1.0"?>\n<Xdmf Version="3.0">\n<Domain>\n'
        '<Grid Name="Grid">\n'
        '<Topology Type="2DRectMesh" Dimensions="3 2"/>\n'
        '<Geometry GeometryType="VXVY">\n'
        '<DataItem Dimensions="2" Format="XML">0.0 1.0</DataItem>\n'
        '<DataItem Dimensions="2" Format="XML">0.0 1.0</DataItem>\n'
        "</Geometry>\n"
        "</Grid>\n</Domain>\n</Xdmf>\n"
    )
    with pytest.raises(meshio.ReadError):
        meshio.xdmf.read(path)


if __name__ == "__main__":
    test_time_series()
