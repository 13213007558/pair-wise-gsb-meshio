from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Tolerance is compared against the maximum absolute deviation of a composed
# loop transform from the identity.
_DEFAULT_TOL = 1.0e-9


@dataclass
class PeriodicConflict:
    """A cycle of periodic constraints whose composed transform is not the
    identity (within tolerance), i.e. the constraints contradict each other.
    """

    slave: int
    master: int
    # Composed transform around the cycle; should be the identity for a
    # consistent set of constraints.
    loop_transform: np.ndarray
    # Gmsh periodic entities (edim, slave_tag, master_tag) constraining the
    # two nodes of the conflicting link.
    entities: list[tuple[int, int, int]] = field(default_factory=list)
    message: str = ""


@dataclass
class PeriodicConstraints:
    """Resolved periodic node constraints.

    Only parses constraints: no nodes are merged and no coordinates moved.
    """

    # node -> final representative node (the end of the slave->master chain)
    representative: dict[int, int]
    # node -> 4x4 homogeneous affine transform mapping the coordinates of the
    # representative to the coordinates of the node
    transform: dict[int, np.ndarray]
    # node -> list of Gmsh periodic entities (edim, slave_tag, master_tag)
    # that constrain the node
    entities: dict[int, list[tuple[int, int, int]]]
    conflicts: list[PeriodicConflict]

    @property
    def num_independent(self) -> int:
        """Number of distinct representatives among the constrained nodes."""
        return len(set(self.representative.values()))


class _WeightedUnionFind:
    """Union-find carrying a 4x4 affine transform per link.

    For every node, to_root satisfies x_root = to_root @ x_node where x_root
    are the homogeneous coordinates of the current set representative.
    """

    def __init__(self):
        self.parent: dict[int, int] = {}
        self.to_root: dict[int, np.ndarray] = {}

    def find(self, node: int) -> tuple[int, np.ndarray]:
        if node not in self.parent:
            self.parent[node] = node
            self.to_root[node] = np.eye(4)
            return node, self.to_root[node]
        parent = self.parent[node]
        if parent != node:
            root, parent_to_root = self.find(parent)
            # path compression with transform accumulation
            self.to_root[node] = parent_to_root @ self.to_root[node]
            self.parent[node] = root
        return self.parent[node], self.to_root[node]

    def union(self, slave: int, master: int, affine: np.ndarray, tol: float):
        """Add the constraint x_master = affine @ x_slave.

        Returns the composed loop transform if the link closes an
        inconsistent cycle, else None.
        """
        root_slave, slave_to_root = self.find(slave)
        root_master, master_to_root = self.find(master)
        # loop maps coordinates of root_slave to root_master via the new link
        loop = master_to_root @ affine @ np.linalg.inv(slave_to_root)
        if root_slave == root_master:
            if np.max(np.abs(loop - np.eye(4))) > tol:
                return loop
            return None
        # attach the root of the slave under the root of the master
        self.parent[root_slave] = root_master
        self.to_root[root_slave] = loop
        return None


def _as_transform(affine) -> np.ndarray:
    if affine is None or len(affine) == 0:
        return np.eye(4)
    affine = np.asarray(affine, dtype=float)
    if affine.size != 16:
        raise ValueError(
            f"Gmsh periodic affine transform must have 16 entries, got {affine.size}."
        )
    return affine.reshape(4, 4)


def resolve_periodic(mesh_or_periodic, tol: float = _DEFAULT_TOL) -> PeriodicConstraints:
    """Resolve Gmsh periodic node constraints.

    Follows the slave->master chains in gmsh_periodic and composes the affine
    transforms (rotations, translations, multi-level links) along each chain.
    For every constrained node, returns its final representative and the
    affine transform from the representative to the node. Cycles whose
    composed transform deviates from the identity by more than tol are
    reported as conflicts, as are incompatible constraints imposed on the
    same node by different periodic entities.

    This function only parses constraints; it neither merges nodes nor moves
    coordinates.
    """
    periodic = getattr(mesh_or_periodic, "gmsh_periodic", mesh_or_periodic)

    uf = _WeightedUnionFind()
    entities: dict[int, list[tuple[int, int, int]]] = {}
    conflicts: list[PeriodicConflict] = []

    for entry in periodic or []:
        edim, (stag, mtag), affine, slave_master = entry
        transform = _as_transform(affine)
        entity = (int(edim), int(stag), int(mtag))
        for slave, master in np.asarray(slave_master).reshape(-1, 2):
            slave, master = int(slave), int(master)
            for node in (slave, master):
                node_entities = entities.setdefault(node, [])
                if entity not in node_entities:
                    node_entities.append(entity)
            loop = uf.union(slave, master, transform, tol)
            if loop is not None:
                involved = sorted(set(entities[slave]) | set(entities[master]))
                conflicts.append(
                    PeriodicConflict(
                        slave=slave,
                        master=master,
                        loop_transform=loop,
                        entities=involved,
                        message=(
                            f"Inconsistent periodic constraints: linking node "
                            f"{slave} to node {master} closes a cycle whose "
                            f"composed transform deviates from the identity by "
                            f"{np.max(np.abs(loop - np.eye(4))):.3e} "
                            f"(tolerance {tol:.3e}). Entities involved: "
                            f"{involved}."
                        ),
                    )
                )

    representative = {}
    transform = {}
    for node in entities:
        root, node_to_root = uf.find(node)
        representative[node] = root
        # transform from the representative to the node
        transform[node] = np.linalg.inv(node_to_root)

    return PeriodicConstraints(
        representative=representative,
        transform=transform,
        entities=entities,
        conflicts=conflicts,
    )
