from xml.etree import ElementTree as ET

import numpy as np
import pytest

import meshio
from meshio._exceptions import ReadError

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


def test_time_series_moving_and_remeshing_mesh(tmp_path):
    filename = tmp_path / "moving-mesh.xdmf"

    # step 0: initial mesh
    points0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells0 = [("triangle", np.array([[0, 1, 2]]))]
    # step 1: same topology, moved nodes
    points1 = points0 + 0.5
    # step 2: remeshed, different number of points and cells
    points2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells2 = [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]

    points_per_step = [points0, points1, points2]
    cells_per_step = [cells0, cells0, cells2]
    times = [0.0, 0.5, 1.0]
    point_data = [
        {"phi": np.full(len(points), t)}
        for points, t in zip(points_per_step, times)
    ]
    cell_data = [
        {"a": [np.full(len(cells[0][1]), t)]}
        for cells, t in zip(cells_per_step, times)
    ]

    with meshio.xdmf.TimeSeriesWriter(filename, data_format="XML") as writer:
        for k in range(3):
            writer.write_data(
                times[k],
                point_data=point_data[k],
                cell_data=cell_data[k],
                points=points_per_step[k],
                cells=cells_per_step[k],
            )

    # check the XML structure: step 1 must reference the topology of step 0,
    # steps 0 and 2 define their mesh inline
    tree = ET.parse(filename)
    domain = tree.getroot().find("Domain")
    collection = [g for g in domain if g.get("GridType") == "Collection"][0]
    steps = list(collection)
    assert len(steps) == 3
    assert steps[0].find("Topology") is not None
    assert steps[0].find("Geometry") is not None
    # step 1 only redefines the geometry; the topology is referenced
    assert steps[1].find("Topology") is None
    assert steps[1].find("Geometry") is not None
    includes = [c for c in steps[1] if c.tag.endswith("include")]
    assert len(includes) == 1
    ptr = includes[0].get("xpointer")
    assert 'Grid[@Name="mesh_1"]' in ptr
    assert ptr.endswith("/Topology)")
    # step 2 is remeshed: full inline mesh
    assert steps[2].find("Topology") is not None
    assert steps[2].find("Geometry") is not None

    # read everything back; read all steps before checking anything to make
    # sure that reading a step doesn't modify the objects returned for the
    # previous steps
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        assert reader.num_steps == 3
        results = []
        for k in range(reader.num_steps):
            points, cells = reader.read_points_cells(k)
            t, pd, cd = reader.read_data(k)
            results.append((points, cells, t, pd, cd))

    for k, (points, cells, t, pd, cd) in enumerate(results):
        assert np.allclose(points, points_per_step[k])
        assert len(cells) == 1
        assert cells[0].type == "triangle"
        assert np.array_equal(cells[0].data, cells_per_step[k][0][1])
        assert np.abs(times[k] - t) < 1.0e-12
        assert np.allclose(pd["phi"], point_data[k]["phi"])
        assert np.allclose(cd["a"][0], cell_data[k]["a"][0])


def test_time_series_moving_mesh_static_geometry(tmp_path):
    # same topology throughout, only the coordinates change; write_data
    # without points/cells must reference the previous step's mesh
    filename = tmp_path / "moving-mesh2.xdmf"

    points0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]

    with meshio.xdmf.TimeSeriesWriter(filename, data_format="XML") as writer:
        writer.write_data(0.0, points=points0, cells=cells)
        writer.write_data(0.5, points=points0 + 1.0)
        writer.write_data(1.0)

    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        assert reader.num_steps == 3
        for k, offset in enumerate([0.0, 1.0, 1.0]):
            points, cells_read = reader.read_points_cells(k)
            assert np.allclose(points, points0 + offset)
            assert np.array_equal(cells_read[0].data, cells[0][1])


def test_time_series_data_mesh_mismatch(tmp_path):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]

    # point_data not matching the number of points of the step
    filename = tmp_path / "mismatch-point-data.xdmf"
    with meshio.xdmf.TimeSeriesWriter(filename, data_format="XML") as writer:
        writer.write_data(
            0.0, point_data={"phi": np.zeros(4)}, points=points, cells=cells
        )
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        with pytest.raises(ReadError):
            reader.read_data(0)

    # cell_data not matching the number of cells of the step
    filename = tmp_path / "mismatch-cell-data.xdmf"
    with meshio.xdmf.TimeSeriesWriter(filename, data_format="XML") as writer:
        writer.write_data(
            0.0, cell_data={"a": [np.zeros(2)]}, points=points, cells=cells
        )
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        with pytest.raises(ReadError):
            reader.read_data(0)


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
