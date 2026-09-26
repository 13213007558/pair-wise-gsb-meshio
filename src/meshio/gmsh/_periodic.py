"""
Resolution of Gmsh periodic node constraints.

Gmsh stores periodicity as a list of links, each connecting a slave entity to a
master entity through node correspondences and an optional affine
transformation (cf. the $Periodic section of the msh format).  This module
turns that raw data into something a solver can consume directly: for every
constrained node, the final representative node and the affine transformation
mapping the representative's coordinates onto the node's coordinates.

Only the constraints are resolved here; nodes are never merged and point
coordinates are never touched.
"""

from __future__ import annotations

import numpy as np


class PeriodicConflictError(Exception):
    """Raised when periodic constraints are geometrically inconsistent.

    This happens when the node correspondences close a cycle (or constrain the
    same node through different periodic entities) and the composed affine
    transformations do not agree within the requested tolerance.
    """


class PeriodicConstraints:
    """Resolved periodic constraints.

    Attributes:
        representatives: int array of shape (num_points,) mapping every node
            to its final representative.  Unconstrained nodes map to
            themselves.
        transforms: float array of shape (num_points, 4, 4) with the
            homogeneous affine transformation from the representative to the
            node, i.e. x_node = transforms[i] @ x_representative (in
            homogeneous coordinates).  Identity for unconstrained nodes.
        sources: list of length num_points; sources[i] holds the indices of
            the periodic links (into links / the original gmsh_periodic list)
            that directly constrain node i.
        links: list of (entity_dim, slave_tag, master_tag) tuples, one per
            entry of the original gmsh_periodic list, allowing every
            constraint to be traced back to its Gmsh periodic entity.
    """

    def __init__(self, representatives, transforms, sources, links):
        self.representatives = representatives
        self.transforms = transforms
        self.sources = sources
        self.links = links

    @property
    def constrained_nodes(self):
        """Indices of all nodes constrained to a different representative."""
        return np.flatnonzero(
            self.representatives != np.arange(len(self.representatives))
        )

    @property
    def masters(self):
        """Sorted unique representatives of all constrained nodes."""
        return np.unique(self.representatives[self.constrained_nodes])


def _find(parent, potential, node):
    """Find the root of 'node' and the transform from the root to the node.

    potential[i] maps the parent of i to i such that
    x_i = potential[i] @ x_parent[i].  Applies path compression.
    """
    root = node
    while parent[root] != root:
        root = parent[root]
    if root == node:
        return root, np.eye(4)
    # compose the transforms along the path from node up to root
    chain = []
    n = node
    while parent[n] != n:
        chain.append(n)
        n = parent[n]
    transform = np.eye(4)
    for p in reversed(chain):
        transform = potential[p] @ transform
        potential[p] = transform.copy()
        parent[p] = root
    return root, transform


def resolve_periodic_constraints(mesh, tol: float = 1.0e-9) -> PeriodicConstraints:
    """Resolve the periodic constraints stored in mesh.gmsh_periodic.

    For every node constrained by a Gmsh periodic link, compute the final
    representative node and the affine transformation from the representative
    to the node.  Transformations of chained links (a master that is itself
    the slave of another entity) are composed along the chain, so rotations,
    translations and mixed multi-level transformations accumulate correctly.

    Args:
        mesh: A meshio.Mesh with gmsh_periodic data.
        tol: Absolute tolerance used when checking that redundant constraints
            (closed loops or nodes constrained by several periodic entities)
            are geometrically compatible.

    Returns:
        A PeriodicConstraints instance.

    Raises:
        ValueError: If the mesh carries no gmsh_periodic data or a link
            references a node outside the mesh.
        PeriodicConflictError: If redundant constraints contradict each other
            within tol.
    """
    if mesh.gmsh_periodic is None:
        raise ValueError("mesh has no gmsh_periodic data")

    num_points = len(mesh.points)
    parent = list(range(num_points))
    potential = np.tile(np.eye(4), (num_points, 1, 1))
    sources = [[] for _ in range(num_points)]
    links = []
    slave_nodes = set()

    for link_index, (edim, (stag, mtag), affine, slave_master) in enumerate(
        mesh.gmsh_periodic
    ):
        links.append((edim, stag, mtag))
        if affine is None or len(affine) == 0:
            link_transform = np.eye(4)
        else:
            link_transform = np.asarray(affine, dtype=float).reshape(4, 4)

        for slave, master in np.asarray(slave_master, dtype=np.int64):
            if not (0 <= slave < num_points and 0 <= master < num_points):
                raise ValueError(
                    f"periodic link {link_index} (entity dim {edim}, slave tag "
                    f"{stag}, master tag {mtag}) references node pair "
                    f"({slave}, {master}) outside [0, {num_points})"
                )
            sources[slave].append(link_index)
            sources[master].append(link_index)
            slave_nodes.add(slave)

            root_slave, trans_slave = _find(parent, potential, slave)
            root_master, trans_master = _find(parent, potential, master)

            if root_slave == root_master:
                # The new constraint closes a cycle (or duplicates a route
                # through another periodic entity).  It is only valid if the
                # composed transformations agree, i.e. the loop closes with
                # the identity within tolerance.
                residual = trans_slave - link_transform @ trans_master
                if np.max(np.abs(residual)) > tol:
                    raise PeriodicConflictError(
                        f"inconsistent periodic constraints: nodes {slave} and "
                        f"{master} already share representative {root_slave}, "
                        f"but the constraint from periodic entity "
                        f"(dim={edim}, slave tag={stag}, master tag={mtag}) "
                        f"disagrees with the existing route "
                        f"(max deviation {np.max(np.abs(residual)):.3e} > "
                        f"tol={tol:.1e})\n"
                        f"existing transform to representative:\n{trans_slave}\n"
                        f"transform implied by entity ({stag} <- {mtag}):\n"
                        f"{link_transform @ trans_master}"
                    )
            else:
                # Attach the slave's tree under the master's root so that
                # x_slave = link_transform @ x_master holds exactly:
                # x_root_slave = inv(T_s) @ T_link @ T_m @ x_root_master.
                parent[root_slave] = root_master
                potential[root_slave] = (
                    np.linalg.inv(trans_slave) @ link_transform @ trans_master
                )

    roots = np.empty(num_points, dtype=np.int64)
    root_transforms = np.empty((num_points, 4, 4))
    for node in range(num_points):
        root, transform = _find(parent, potential, node)
        roots[node] = root
        root_transforms[node] = transform

    # Pick the representative of every component: a node that is not the
    # slave of any link (a true master) if there is one, else the lowest
    # node index.  Transforms are then re-based from the component root to
    # the representative.
    representatives = np.empty(num_points, dtype=np.int64)
    transforms = np.empty((num_points, 4, 4))
    component_rep = {}
    for node in range(num_points):
        root = roots[node]
        if root not in component_rep:
            component_rep[root] = node
        elif component_rep[root] in slave_nodes and node not in slave_nodes:
            component_rep[root] = node
    for node in range(num_points):
        rep = component_rep[roots[node]]
        representatives[node] = rep
        transforms[node] = root_transforms[node] @ np.linalg.inv(
            root_transforms[rep]
        )

    return PeriodicConstraints(representatives, transforms, sources, links)
