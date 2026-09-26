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


def test_time_series_per_step_mesh(tmp_path):
    # Three steps: the second only moves the points, the third remeshes
    # (different number of points and cells).
    filename = tmp_path / "moving.xdmf"

    points0 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells0 = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
    points1 = points0 + np.array([0.5, 0.0])
    points2 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells2 = [("triangle", np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))]

    times = [0.0, 0.5, 1.0]
    point_data = [
        {"phi": np.full(4, 0.0)},
        {"phi": np.full(4, 0.5)},
        {"phi": np.full(5, 1.0)},
    ]
    cell_data = [
        {"a": {"triangle": [1.0, 2.0]}},
        {"a": {"triangle": [3.0, 4.0]}},
        {"a": {"triangle": [5.0, 6.0, 7.0, 8.0]}},
    ]
    # the writer restructures the cell_data dicts in place, so keep the
    # expected values separately
    expected_cell_data = [np.array(cd["a"]["triangle"]) for cd in cell_data]

    with meshio.xdmf.TimeSeriesWriter(filename, data_format="XML") as writer:
        writer.write_points_cells(points0, cells0)
        writer.write_data(times[0], point_data=point_data[0], cell_data=cell_data[0])
        writer.write_data(
            times[1], point_data=point_data[1], cell_data=cell_data[1], points=points1
        )
        writer.write_data(
            times[2],
            point_data=point_data[2],
            cell_data=cell_data[2],
            points=points2,
            cells=cells2,
        )

    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        assert reader.num_steps == 3

        # the static convenience call still gives the first mesh
        points, cells = reader.read_points_cells()
        assert np.allclose(points, points0)
        assert len(cells) == 1
        assert np.array_equal(cells[0].data, cells0[0][1])

        # per-step meshes
        p0, c0 = reader.read_mesh(0)
        p1, c1 = reader.read_mesh(1)
        p2, c2 = reader.read_mesh(2)
        assert np.allclose(p0, points0)
        assert np.allclose(p1, points1)
        assert np.allclose(p2, points2)
        # step 1 references the topology of the first grid; make sure the
        # reference is not resolved to one of the other grids' Topology nodes
        assert np.array_equal(c1[0].data, cells0[0][1])
        assert np.array_equal(c2[0].data, cells2[0][1])

        # snapshot step 0; reading further steps must not modify the
        # previously returned objects in place
        p0_snapshot = p0.copy()
        c0_snapshot = c0[0].data.copy()

        for k in range(reader.num_steps):
            t, pd, cd = reader.read_data(k)
            assert np.abs(times[k] - t) < 1.0e-12
            assert np.allclose(pd["phi"], point_data[k]["phi"])
            assert np.allclose(cd["a"][0], expected_cell_data[k])

        assert np.array_equal(p0, p0_snapshot)
        assert np.array_equal(c0[0].data, c0_snapshot)


def test_time_series_data_mesh_mismatch(tmp_path):
    filename = tmp_path / "mismatch.xdmf"
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]

    with meshio.xdmf.TimeSeriesWriter(filename, data_format="XML") as writer:
        writer.write_points_cells(points, cells)
        writer.write_data(0.0, point_data={"phi": np.full(7, 0.0)})
        writer.write_data(1.0, cell_data={"a": {"triangle": [1.0, 2.0, 3.0]}})

    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        with pytest.raises(meshio.ReadError, match="Point data 'phi'"):
            reader.read_data(0)
        with pytest.raises(meshio.ReadError, match="Cell data 'a'"):
            reader.read_data(1)


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
