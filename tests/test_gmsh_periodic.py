import numpy as np
import pytest

import meshio
from meshio.gmsh import resolve_periodic


def _translation(dx, dy, dz):
    transform = np.eye(4)
    transform[:3, 3] = [dx, dy, dz]
    return transform


def _rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return transform


def _flat(transform):
    # gmsh stores the 16 entries of the 4x4 matrix in row-major order
    return np.asarray(transform).reshape(16)


def _make_mesh(periodic):
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]]))],
    )
    mesh.gmsh_periodic = periodic
    return mesh


def test_translation_chain():
    # node 1 -> node 0 and node 2 -> node 1, each shifted by (1, 0, 0)
    periodic = [[1, (2, 1), _flat(_translation(1.0, 0.0, 0.0)), [[1, 0], [2, 1]]]]
    result = resolve_periodic(_make_mesh(periodic))

    assert result.conflicts == []
    assert result.representative == {0: 0, 1: 0, 2: 0}
    # transform from the representative (node 0) to the node
    assert np.allclose(result.transform[0], np.eye(4))
    assert np.allclose(result.transform[1], _translation(-1.0, 0.0, 0.0))
    assert np.allclose(result.transform[2], _translation(-2.0, 0.0, 0.0))
    assert result.entities[1] == [(1, 2, 1)]
    assert result.num_independent == 1


def test_rotation():
    # node 1 is node 0 rotated by 90 degrees about z
    rotation = _rotation_z(np.pi / 2)
    periodic = [[0, (3, 1), _flat(rotation), [[1, 0]]]]
    result = resolve_periodic(_make_mesh(periodic))

    assert result.conflicts == []
    assert result.representative[1] == 0
    # representative -> node is the inverse rotation
    assert np.allclose(result.transform[1], _rotation_z(-np.pi / 2))


def test_multilevel_mixed():
    # node 1 -> node 0 via rotation, node 2 -> node 1 via translation;
    # the composed transform must follow the chain
    rotation = _rotation_z(np.pi / 2)
    translation = _translation(0.0, 1.0, 0.0)
    periodic = [
        [0, (3, 1), _flat(rotation), [[1, 0]]],
        [0, (4, 3), _flat(translation), [[2, 1]]],
    ]
    result = resolve_periodic(_make_mesh(periodic))

    assert result.conflicts == []
    assert result.representative == {0: 0, 1: 0, 2: 0}
    # x_0 = R @ x_1 = R @ T @ x_2  =>  x_2 = (R @ T)^-1 @ x_0
    assert np.allclose(result.transform[2], np.linalg.inv(rotation @ translation))
    # both entities are reported for the shared middle node
    assert result.entities[1] == [(0, 3, 1), (0, 4, 3)]


def test_consistent_cycle():
    # four nodes linked by 90-degree rotations; the loop closes exactly
    rotation = _rotation_z(np.pi / 2)
    periodic = [
        [0, (5, 1), _flat(rotation), [[1, 0], [2, 1], [3, 2], [0, 3]]],
    ]
    result = resolve_periodic(_make_mesh(periodic))

    assert result.conflicts == []
    assert set(result.representative.values()) == {0}
    for node in range(4):
        assert np.allclose(
            result.transform[node], _rotation_z(-node * np.pi / 2), atol=1.0e-12
        )


def test_conflicting_cycle():
    # node numbers could be merged by a plain union-find, but the geometric
    # transforms contradict each other: 1 -> 0 shifts by (1,0,0) while the
    # direct link 0 -> 1 is the identity
    periodic = [
        [1, (2, 1), _flat(_translation(1.0, 0.0, 0.0)), [[1, 0]]],
        [1, (1, 2), _flat(np.eye(4)), [[0, 1]]],
    ]
    result = resolve_periodic(_make_mesh(periodic))

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert {conflict.slave, conflict.master} == {0, 1}
    # the loop transform is the inconsistent translation
    assert not np.allclose(conflict.loop_transform, np.eye(4))
    # both gmsh entities are located in the report
    assert conflict.entities == [(1, 1, 2), (1, 2, 1)]
    assert "(1, 2, 1)" in conflict.message
    # the nodes are still identified through the consistent first link;
    # the conflict concerns the geometry, not the numbering
    assert result.representative[0] == result.representative[1]


def test_same_node_two_entities_compatible():
    # node 2 reaches node 0 directly (entity A) and via node 1 (entity B);
    # both routes agree geometrically
    shift = _flat(_translation(1.0, 0.0, 0.0))
    periodic = [
        [1, (10, 1), _flat(_translation(2.0, 0.0, 0.0)), [[2, 0]]],
        [1, (11, 1), shift, [[1, 0], [2, 1]]],
    ]
    result = resolve_periodic(_make_mesh(periodic))

    assert result.conflicts == []
    assert result.representative == {0: 0, 1: 0, 2: 0}
    assert np.allclose(result.transform[2], _translation(-2.0, 0.0, 0.0))
    assert result.entities[2] == [(1, 10, 1), (1, 11, 1)]


def test_same_node_two_entities_incompatible():
    # same topology as the compatible case, but the direct link shifts by
    # (2.5, 0, 0) while the chain gives (2, 0, 0)
    periodic = [
        [1, (10, 1), _flat(_translation(2.5, 0.0, 0.0)), [[2, 0]]],
        [1, (11, 1), _flat(_translation(1.0, 0.0, 0.0)), [[1, 0], [2, 1]]],
    ]
    result = resolve_periodic(_make_mesh(periodic))

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.entities == [(1, 10, 1), (1, 11, 1)]
    assert np.max(np.abs(conflict.loop_transform - np.eye(4))) == pytest.approx(
        0.5
    )


def test_tolerance():
    # a mismatch below tol is accepted, above tol is a conflict
    eps = 1.0e-6
    periodic = [
        [1, (10, 1), _flat(_translation(2.0 + eps, 0.0, 0.0)), [[2, 0]]],
        [1, (11, 1), _flat(_translation(1.0, 0.0, 0.0)), [[1, 0], [2, 1]]],
    ]
    assert resolve_periodic(_make_mesh(periodic), tol=1.0e-5).conflicts == []
    assert len(resolve_periodic(_make_mesh(periodic), tol=1.0e-7).conflicts) == 1


def test_no_affine_means_identity():
    # gmsh omits the affine block for plain identification
    periodic = [[0, (3, 1), None, [[2, 0]]]]
    result = resolve_periodic(_make_mesh(periodic))

    assert result.conflicts == []
    assert result.representative[2] == 0
    assert np.allclose(result.transform[2], np.eye(4))


def test_empty():
    mesh = _make_mesh(None)
    result = resolve_periodic(mesh)
    assert result.representative == {}
    assert result.transform == {}
    assert result.entities == {}
    assert result.conflicts == []


def test_mesh_not_modified():
    periodic = [[1, (2, 1), _flat(_translation(1.0, 0.0, 0.0)), [[1, 0]]]]
    mesh = _make_mesh(periodic)
    points_before = mesh.points.copy()
    cells_before = mesh.cells[0].data.copy()
    resolve_periodic(mesh)
    assert np.array_equal(mesh.points, points_before)
    assert np.array_equal(mesh.cells[0].data, cells_before)
    assert len(mesh.cells) == 1


def test_invalid_affine_size():
    mesh = _make_mesh([[1, (2, 1), np.zeros(9), [[1, 0]]]])
    with pytest.raises(ValueError, match="16 entries"):
        resolve_periodic(mesh)
