import numpy as np
import pytest

import meshio
from meshio.gmsh import (
    PeriodicConflictError,
    resolve_periodic_constraints,
)


def translation(dx, dy, dz):
    transform = np.eye(4)
    transform[:3, 3] = [dx, dy, dz]
    return transform


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:2, :2] = [[c, -s], [s, c]]
    return transform


def make_mesh(points, gmsh_periodic):
    points = np.asarray(points, dtype=float)
    cells = [("vertex", np.arange(len(points)).reshape(-1, 1))]
    return meshio.Mesh(points, cells, gmsh_periodic=gmsh_periodic)


def homogeneous(point):
    return np.append(point, 1.0)


def test_translation_periodic():
    shift = translation(1.0, 0.0, 0.0)
    mesh = make_mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[1, (2, 1), shift.flatten(), [[1, 0]]]],
    )
    result = resolve_periodic_constraints(mesh)

    assert result.representatives[1] == 0
    assert np.allclose(result.transforms[1], shift)
    # node 2 is unconstrained
    assert result.representatives[2] == 2
    assert np.allclose(result.transforms[2], np.eye(4))
    # the transform maps the representative's coordinates onto the node's
    assert np.allclose(
        result.transforms[1] @ homogeneous(mesh.points[0]),
        homogeneous(mesh.points[1]),
    )
    # no nodes are merged and no coordinates are moved
    assert len(mesh.points) == 3


def test_rotation_periodic():
    rot = rotation_z(np.pi / 2)
    mesh = make_mesh(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1, (4, 3), rot.flatten(), [[1, 0]]]],
    )
    result = resolve_periodic_constraints(mesh)

    assert result.representatives[1] == 0
    assert np.allclose(result.transforms[1], rot)
    assert np.allclose(
        result.transforms[1] @ homogeneous(mesh.points[0]),
        homogeneous(mesh.points[1]),
    )


def test_multilevel_mixed_transformations():
    # chain: 3 <- 2 (rotation), 2 <- 1 (translation), 1 <- 0 (rotation),
    # deliberately listed out of order
    rot90 = rotation_z(np.pi / 2)
    shift = translation(0.0, 0.0, 2.0)
    rot30 = rotation_z(np.pi / 6)
    mesh = make_mesh(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
            [-0.5, np.sqrt(3) / 2, 2.0],
        ],
        [
            [1, (12, 11), shift.flatten(), [[2, 1]]],
            [1, (13, 12), rot30.flatten(), [[3, 2]]],
            [1, (11, 10), rot90.flatten(), [[1, 0]]],
        ],
    )
    result = resolve_periodic_constraints(mesh)

    # every slave ends up at the final representative 0
    assert result.representatives[0] == 0
    assert result.representatives[1] == 0
    assert result.representatives[2] == 0
    assert result.representatives[3] == 0
    # transforms compose along the chain, in the right direction
    assert np.allclose(result.transforms[1], rot90)
    assert np.allclose(result.transforms[2], shift @ rot90)
    assert np.allclose(result.transforms[3], rot30 @ shift @ rot90)
    # geometric consistency with the actual coordinates
    for node in range(4):
        assert np.allclose(
            result.transforms[node] @ homogeneous(mesh.points[0]),
            homogeneous(mesh.points[node]),
        )
    assert list(result.masters) == [0]
    assert list(result.constrained_nodes) == [1, 2, 3]


def test_consistent_closed_loop():
    # three 120-degree rotations compose to the identity: the loop closes
    # consistently and all nodes share one representative
    rot = rotation_z(2 * np.pi / 3)
    mesh = make_mesh(
        [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [-0.5, -np.sqrt(3) / 2, 0.0]],
        [
            [0, (2, 1), rot.flatten(), [[1, 0]]],
            [0, (3, 2), rot.flatten(), [[2, 1]]],
            [0, (1, 3), rot.flatten(), [[0, 2]]],
        ],
    )
    result = resolve_periodic_constraints(mesh)

    assert len(set(result.representatives)) == 1
    root = result.representatives[0]
    for node in range(3):
        assert np.allclose(
            result.transforms[node] @ homogeneous(mesh.points[root]),
            homogeneous(mesh.points[node]),
        )


def test_conflicting_closed_loop():
    # the node correspondences would merge fine in a plain union-find, but
    # three 90-degree rotations compose to 270 degrees, not the identity
    rot = rotation_z(np.pi / 2)
    mesh = make_mesh(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [
            [0, (2, 1), rot.flatten(), [[0, 1]]],
            [0, (3, 2), rot.flatten(), [[1, 2]]],
            [0, (1, 3), rot.flatten(), [[2, 0]]],
        ],
    )
    with pytest.raises(PeriodicConflictError) as excinfo:
        resolve_periodic_constraints(mesh)
    # the conflicting Gmsh periodic entity can be located from the message
    assert "slave tag=1" in str(excinfo.value)
    assert "master tag=3" in str(excinfo.value)


def test_same_node_two_entities_compatible():
    # node 2 is constrained by two different periodic entities; the two
    # routes agree because shift_a = shift_b @ shift_c
    shift_a = translation(2.0, 0.0, 0.0)
    shift_b = translation(1.0, 0.0, 0.0)
    shift_c = translation(1.0, 0.0, 0.0)
    mesh = make_mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [
            [1, (5, 4), shift_a.flatten(), [[2, 0]]],
            [1, (6, 4), shift_b.flatten(), [[2, 1]]],
            [1, (7, 4), shift_c.flatten(), [[1, 0]]],
        ],
    )
    result = resolve_periodic_constraints(mesh, tol=1.0e-12)

    assert result.representatives[2] == 0
    assert np.allclose(result.transforms[2], shift_a)


def test_same_node_two_entities_conflicting():
    # same node-number structure as above, but the two routes to node 2
    # contradict each other geometrically
    shift_a = translation(2.0, 0.0, 0.0)
    shift_b = translation(1.0, 0.0, 0.0)
    shift_c = translation(0.5, 0.0, 0.0)
    mesh = make_mesh(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [
            [1, (5, 4), shift_a.flatten(), [[2, 0]]],
            [1, (6, 4), shift_b.flatten(), [[2, 1]]],
            [1, (7, 4), shift_c.flatten(), [[1, 0]]],
        ],
    )
    with pytest.raises(PeriodicConflictError) as excinfo:
        resolve_periodic_constraints(mesh)
    assert "slave tag=7" in str(excinfo.value)


def test_entity_provenance():
    shift = translation(1.0, 0.0, 0.0)
    mesh = make_mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [
            [1, (11, 10), shift.flatten(), [[1, 0]]],
            [2, (21, 20), shift.flatten(), [[2, 1]]],
        ],
    )
    result = resolve_periodic_constraints(mesh)

    # every constraint can be traced back to its Gmsh periodic entity
    assert result.links == [(1, 11, 10), (2, 21, 20)]
    assert result.sources[1] == [0, 1]
    assert result.sources[2] == [1]
    assert result.sources[0] == [0]
    edim, stag, mtag = result.links[result.sources[2][0]]
    assert (edim, stag, mtag) == (2, 21, 20)


def test_no_periodic_data():
    mesh = meshio.Mesh(np.zeros((1, 3)), [("vertex", [[0]])])
    with pytest.raises(ValueError, match="no gmsh_periodic"):
        resolve_periodic_constraints(mesh)


def test_node_index_out_of_range():
    mesh = make_mesh(
        [[0.0, 0.0, 0.0]],
        [[0, (2, 1), None, [[0, 5]]]],
    )
    with pytest.raises(ValueError, match="outside"):
        resolve_periodic_constraints(mesh)


def test_identity_affine_is_default():
    # affine=None (and empty affine) means the identity transformation
    mesh = make_mesh(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [
            [0, (2, 1), None, [[1, 0]]],
            [0, (3, 2), [], [[2, 1]]],
        ],
    )
    result = resolve_periodic_constraints(mesh)
    assert result.representatives[2] == 0
    assert np.allclose(result.transforms[2], np.eye(4))


def test_roundtrip_through_gmsh_file(tmp_path):
    shift = translation(0.0, 1.0, 0.0)
    mesh = make_mesh(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1, (2, 1), shift.flatten(), [[1, 0]]]],
    )
    for fmt_version, binary in [("2.2", False), ("4.1", False), ("4.1", True)]:
        path = tmp_path / ("test" + fmt_version + str(binary) + ".msh")
        meshio.gmsh.write(path, mesh, fmt_version=fmt_version, binary=binary)
        mesh2 = meshio.gmsh.read(path)
        result = resolve_periodic_constraints(mesh2)
        assert result.representatives[1] == 0
        assert np.allclose(result.transforms[1], shift)
        assert result.links == [(1, 2, 1)]
