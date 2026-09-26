"""
I/O for Exodus II.

See
<https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf>,
in particular Appendix A (page 171, Implementation of EXODUS II with netCDF).
"""

import datetime
import re

import numpy as np

from ..__about__ import __version__
from .._common import local_faces, warn
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from .._mesh import Mesh, SideSet

exodus_to_meshio_type = {
    "SPHERE": "vertex",
    # curves
    "BEAM": "line",
    "BEAM2": "line",
    "BEAM3": "line3",
    "BAR2": "line",
    # surfaces
    "SHELL": "quad",
    "SHELL4": "quad",
    "SHELL8": "quad8",
    "SHELL9": "quad9",
    "QUAD": "quad",
    "QUAD4": "quad",
    "QUAD5": "quad5",
    "QUAD8": "quad8",
    "QUAD9": "quad9",
    #
    "TRI": "triangle",
    "TRIANGLE": "triangle",
    "TRI3": "triangle",
    "TRI6": "triangle6",
    "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL6': 'triangle6',
    # 'TRISHELL7': 'triangle',
    #
    # volumes
    "HEX": "hexahedron",
    "HEXAHEDRON": "hexahedron",
    "HEX8": "hexahedron",
    "HEX9": "hexahedron9",
    "HEX20": "hexahedron20",
    "HEX27": "hexahedron27",
    #
    "TETRA": "tetra",
    "TETRA4": "tetra4",
    "TET4": "tetra4",
    "TETRA8": "tetra8",
    "TETRA10": "tetra10",
    "TETRA14": "tetra14",
    #
    "PYRAMID": "pyramid",
    "WEDGE": "wedge",
}
meshio_to_exodus_type = {v: k for k, v in exodus_to_meshio_type.items()}


def read(filename):  # noqa: C901
    import netCDF4

    with netCDF4.Dataset(filename) as nc:
        # assert nc.version == np.float32(5.1)
        # assert nc.api_version == np.float32(5.1)
        # assert nc.floating_point_word_size == 8

        # assert b''.join(nc.variables['coor_names'][0]) == b'X'
        # assert b''.join(nc.variables['coor_names'][1]) == b'Y'
        # assert b''.join(nc.variables['coor_names'][2]) == b'Z'

        points = np.zeros((len(nc.dimensions["num_nodes"]), 3))
        point_data_names = []
        cell_data_names = []
        pd = {}
        cd = {}
        cells = []
        ns_names = []
        # eb_names = []
        ns = []
        point_sets = {}
        info = []
        ss_names = []
        ss_ids = []
        ss_elem = {}
        ss_side = {}
        ss_df = {}

        for key, value in nc.variables.items():
            if key == "info_records":
                value.set_auto_mask(False)
                for c in value[:]:
                    try:
                        info += [b"".join(c).decode("UTF-8")]
                    except UnicodeDecodeError:
                        # https://github.com/nschloe/meshio/issues/983
                        pass
            elif key == "qa_records":
                value.set_auto_mask(False)
                for val in value:
                    info += [b"".join(c).decode("UTF-8") for c in val[:]]
            elif key[:7] == "connect":
                meshio_type = exodus_to_meshio_type[value.elem_type.upper()]
                cells.append((int(key[7:]) - 1, meshio_type, value[:] - 1))
            elif key == "coord":
                points = nc.variables["coord"][:].T
            elif key == "coordx":
                points[:, 0] = value[:]
            elif key == "coordy":
                points[:, 1] = value[:]
            elif key == "coordz":
                points[:, 2] = value[:]
            elif key == "name_nod_var":
                value.set_auto_mask(False)
                point_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            elif key[:12] == "vals_nod_var":
                idx = 0 if len(key) == 12 else int(key[12:]) - 1
                value.set_auto_mask(False)
                # For now only take the first value
                pd[idx] = value[0]
                if len(value) > 1:
                    warn("Skipping some time data")
            elif key == "name_elem_var":
                value.set_auto_mask(False)
                cell_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            elif key[:13] == "vals_elem_var":
                # eb: element block
                m = re.match("vals_elem_var(\\d+)?(?:eb(\\d+))?", key)
                idx = 0 if m.group(1) is None else int(m.group(1)) - 1
                block = 0 if m.group(2) is None else int(m.group(2)) - 1

                value.set_auto_mask(False)
                # For now only take the first value
                if idx not in cd:
                    cd[idx] = {}
                cd[idx][block] = value[0]

                if len(value) > 1:
                    warn("Skipping some time data")
            elif key == "ns_names":
                value.set_auto_mask(False)
                ns_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            # elif key == "eb_names":
            #     value.set_auto_mask(False)
            #     eb_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            elif key.startswith("node_ns"):  # Expected keys: node_ns1, node_ns2
                ns.append(value[:] - 1)  # Exodus is 1-based
            elif key == "ss_names":
                value.set_auto_mask(False)
                ss_names = [
                    b"".join(c).decode("UTF-8").rstrip("\x00") for c in value[:]
                ]
            elif key == "ss_prop1":
                ss_ids = [int(v) for v in value[:]]
            elif key.startswith("elem_ss"):  # Expected keys: elem_ss1, elem_ss2
                ss_elem[int(key[7:]) - 1] = value[:] - 1  # Exodus is 1-based
            elif key.startswith("side_ss"):  # Expected keys: side_ss1, side_ss2
                ss_side[int(key[7:]) - 1] = value[:] - 1  # Exodus is 1-based
            elif key.startswith("dist_fact_ss"):
                ss_df[int(key[12:]) - 1] = value[:]

        # sort cells by block index; the global element numbering used in side
        # sets runs across the blocks in this order
        cells = [(meshio_type, data) for _, meshio_type, data in sorted(cells)]

        # merge element block data; can't handle blocks yet
        for k, value in cd.items():
            cd[k] = np.concatenate([value[b] for b in sorted(value)])

        # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
        # triplets in the point data. If yes, they belong together.
        single, double, triple = categorize(point_data_names)

        point_data = {}
        for name, idx in single:
            point_data[name] = pd[idx]
        for name, idx0, idx1 in double:
            point_data[name] = np.column_stack([pd[idx0], pd[idx1]])
        for name, idx0, idx1, idx2 in triple:
            point_data[name] = np.column_stack([pd[idx0], pd[idx1], pd[idx2]])

        cell_data = {}
        k = 0
        for _, cell in cells:
            n = len(cell)
            for name, data in zip(cell_data_names, cd.values()):
                if name not in cell_data:
                    cell_data[name] = []
                cell_data[name].append(data[k : k + n])
            k += n

        point_sets = {name: dat for name, dat in zip(ns_names, ns)}

        side_sets = _read_side_sets(cells, ss_names, ss_ids, ss_elem, ss_side, ss_df)

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        point_sets=point_sets,
        side_sets=side_sets,
        info=info,
    )


def _read_side_sets(cells, ss_names, ss_ids, ss_elem, ss_side, ss_df):
    # map the global (file) element numbers to (block, local cell) indices
    block_sizes = [len(data) for _, data in cells]
    offsets = np.concatenate([[0], np.cumsum(block_sizes)])

    side_sets = {}
    for k in sorted(ss_elem):
        if k < len(ss_names):
            name = ss_names[k]
        elif k < len(ss_ids):
            name = str(ss_ids[k])
        else:
            name = f"ss{k + 1}"

        elems = ss_elem[k]
        sides = ss_side[k]
        if len(elems) > 0 and (np.any(elems < 0) or np.any(elems >= offsets[-1])):
            raise ReadError(f"Side set '{name}': element number out of range.")

        block_idx = np.searchsorted(offsets, elems, side="right") - 1
        local_cells = elems - offsets[block_idx]

        data = []
        for b, (cell_type, _) in enumerate(cells):
            mask = block_idx == b
            block_sides = sides[mask]
            if cell_type not in local_faces:
                if np.any(mask):
                    raise ReadError(
                        f"Side set '{name}': no local faces defined for cell "
                        f"type '{cell_type}'."
                    )
            elif len(block_sides) > 0 and (
                np.any(block_sides < 0)
                or np.any(block_sides >= len(local_faces[cell_type]))
            ):
                raise ReadError(
                    f"Side set '{name}': side number out of range for cell "
                    f"type '{cell_type}'."
                )
            data.append(np.column_stack([local_cells[mask], block_sides]))

        dist_facts = None
        if k in ss_df:
            # split the flat distribution factor array into one array per
            # face; each face contributes as many factors as it has nodes
            df = ss_df[k]
            counts = [
                len(local_faces[cells[b][0]][s]) for b, s in zip(block_idx, sides)
            ]
            if len(df) != sum(counts):
                raise ReadError(
                    f"Side set '{name}': {len(df)} distribution factors, but "
                    f"the {len(sides)} faces have {sum(counts)} nodes in "
                    f"total."
                )
            split = np.split(df, np.cumsum(counts)[:-1])
            dist_facts = [[] for _ in cells]
            for b, df_face in zip(block_idx, split):
                dist_facts[b].append(df_face)

        side_sets[name] = SideSet(
            data,
            id=ss_ids[k] if k < len(ss_ids) else None,
            dist_facts=dist_facts,
        )
    return side_sets


def categorize(names):
    # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
    # triplets in the point data. If yes, they belong together.
    single = []
    double = []
    triple = []
    is_accounted_for = [False] * len(names)
    k = 0
    while True:
        if k == len(names):
            break
        if is_accounted_for[k]:
            k += 1
            continue
        name = names[k]
        if name[-1] == "X":
            ix = k
            try:
                iy = names.index(name[:-1] + "Y")
            except ValueError:
                iy = None
            try:
                iz = names.index(name[:-1] + "Z")
            except ValueError:
                iz = None
            if iy and iz:
                triple.append((name[:-1], ix, iy, iz))
                is_accounted_for[ix] = True
                is_accounted_for[iy] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ix))
                is_accounted_for[ix] = True
        elif name[-2:] == "_R":
            ir = k
            try:
                iz = names.index(name[:-2] + "_Z")
            except ValueError:
                iz = None
            if iz:
                double.append((name[:-2], ir, iz))
                is_accounted_for[ir] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ir))
                is_accounted_for[ir] = True
        else:
            single.append((name, k))
            is_accounted_for[k] = True

        k += 1

    if not all(is_accounted_for):
        raise ReadError()
    return single, double, triple


numpy_to_exodus_dtype = {
    "float32": "f4",
    "float64": "f8",
    "int8": "i1",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
    "uint8": "u1",
    "uint16": "u2",
    "uint32": "u4",
    "uint64": "u8",
}


def write(filename, mesh):
    import netCDF4

    with netCDF4.Dataset(filename, "w") as rootgrp:
        # set global data
        now = datetime.datetime.now().isoformat()
        rootgrp.title = f"Created by meshio v{__version__}, {now}"
        rootgrp.version = np.float32(5.1)
        rootgrp.api_version = np.float32(5.1)
        rootgrp.floating_point_word_size = 8

        # set dimensions
        total_num_elems = sum(c.data.shape[0] for c in mesh.cells)
        rootgrp.createDimension("num_nodes", len(mesh.points))
        rootgrp.createDimension("num_dim", mesh.points.shape[1])
        rootgrp.createDimension("num_elem", total_num_elems)
        rootgrp.createDimension("num_el_blk", len(mesh.cells))
        rootgrp.createDimension("num_node_sets", len(mesh.point_sets))
        rootgrp.createDimension("num_side_sets", len(mesh.side_sets))
        rootgrp.createDimension("len_string", 33)
        rootgrp.createDimension("len_line", 81)
        rootgrp.createDimension("four", 4)
        rootgrp.createDimension("time_step", None)

        # dummy time step
        data = rootgrp.createVariable("time_whole", "f4", ("time_step",))
        data[:] = 0.0

        # points
        coor_names = rootgrp.createVariable(
            "coor_names", "S1", ("num_dim", "len_string")
        )
        coor_names.set_auto_mask(False)
        coor_names[0, 0] = b"X"
        coor_names[1, 0] = b"Y"
        if mesh.points.shape[1] == 3:
            coor_names[2, 0] = b"Z"
        data = rootgrp.createVariable(
            "coord",
            numpy_to_exodus_dtype[mesh.points.dtype.name],
            ("num_dim", "num_nodes"),
        )
        data[:] = mesh.points.T

        # cells
        # ParaView needs eb_prop1 -- some ID. The values don't seem to matter as
        # long as they are different for the for different blocks.
        data = rootgrp.createVariable("eb_prop1", "i4", "num_el_blk")
        for k in range(len(mesh.cells)):
            data[k] = k
        for k, cell_block in enumerate(mesh.cells):
            dim1 = f"num_el_in_blk{k + 1}"
            dim2 = f"num_nod_per_el{k + 1}"
            rootgrp.createDimension(dim1, cell_block.data.shape[0])
            rootgrp.createDimension(dim2, cell_block.data.shape[1])
            dtype = numpy_to_exodus_dtype[cell_block.data.dtype.name]
            data = rootgrp.createVariable(f"connect{k + 1}", dtype, (dim1, dim2))
            data.elem_type = meshio_to_exodus_type[cell_block.type]
            # Exodus is 1-based
            data[:] = cell_block.data + 1

        # point data
        # The variable `name_nod_var` holds the names and indices of the node variables, the
        # variables `vals_nod_var{1,2,...}` hold the actual data.
        num_nod_var = len(mesh.point_data)
        if num_nod_var > 0:
            rootgrp.createDimension("num_nod_var", num_nod_var)
            # set names
            point_data_names = rootgrp.createVariable(
                "name_nod_var", "S1", ("num_nod_var", "len_string")
            )
            point_data_names.set_auto_mask(False)
            for k, name in enumerate(mesh.point_data.keys()):
                for i, letter in enumerate(name):
                    point_data_names[k, i] = letter.encode()

            # Set data. ParaView might have some problems here, see
            # <https://gitlab.kitware.com/paraview/paraview/-/issues/18403>.
            for k, (name, data) in enumerate(mesh.point_data.items()):
                for i, s in enumerate(data.shape):
                    rootgrp.createDimension(f"dim_nod_var{k}{i}", s)
                dims = ["time_step"] + [
                    f"dim_nod_var{k}{i}" for i in range(len(data.shape))
                ]
                node_data = rootgrp.createVariable(
                    f"vals_nod_var{k + 1}",
                    numpy_to_exodus_dtype[data.dtype.name],
                    tuple(dims),
                    fill_value=False,
                )
                node_data[0] = data

        # node sets
        num_point_sets = len(mesh.point_sets)
        if num_point_sets > 0:
            data = rootgrp.createVariable("ns_prop1", "i4", "num_node_sets")
            data_names = rootgrp.createVariable(
                "ns_names", "S1", ("num_node_sets", "len_string")
            )
            for k, name in enumerate(mesh.point_sets.keys()):
                data[k] = k
                for i, letter in enumerate(name):
                    data_names[k, i] = letter.encode()
            for k, (key, values) in enumerate(mesh.point_sets.items()):
                dim1 = f"num_nod_ns{k + 1}"
                rootgrp.createDimension(dim1, values.shape[0])
                dtype = numpy_to_exodus_dtype[values.dtype.name]
                data = rootgrp.createVariable(f"node_ns{k + 1}", dtype, (dim1,))
                # Exodus is 1-based
                data[:] = values + 1

        # side sets
        num_side_sets = len(mesh.side_sets)
        if num_side_sets > 0:
            try:
                mesh._validate_side_sets()
            except ValueError as e:
                raise WriteError(str(e)) from e

            data = rootgrp.createVariable("ss_prop1", "i4", "num_side_sets")
            data_names = rootgrp.createVariable(
                "ss_names", "S1", ("num_side_sets", "len_string")
            )
            data_names.set_auto_mask(False)
            # global element numbers run across the blocks in order
            offsets = np.cumsum([0] + [len(c.data) for c in mesh.cells])
            for k, (name, side_set) in enumerate(mesh.side_sets.items()):
                data[k] = side_set.id if side_set.id is not None else k + 1
                for i, letter in enumerate(name):
                    data_names[k, i] = letter.encode()

                elems = []
                sides = []
                dfs = []
                for b, block_data in enumerate(side_set.data):
                    elems.append(offsets[b] + block_data[:, 0])
                    sides.append(block_data[:, 1])
                    if side_set.dist_facts is not None:
                        dfs += side_set.dist_facts[b]
                elems = np.concatenate(elems) if elems else np.array([], int)
                sides = np.concatenate(sides) if sides else np.array([], int)

                dim1 = f"num_side_ss{k + 1}"
                rootgrp.createDimension(dim1, len(elems))
                dtype = numpy_to_exodus_dtype[elems.dtype.name]
                # Exodus is 1-based
                data = rootgrp.createVariable(f"elem_ss{k + 1}", dtype, (dim1,))
                data[:] = elems + 1
                data = rootgrp.createVariable(f"side_ss{k + 1}", dtype, (dim1,))
                data[:] = sides + 1

                if side_set.dist_facts is not None:
                    dim2 = f"num_df_ss{k + 1}"
                    rootgrp.createDimension(dim2, sum(len(df) for df in dfs))
                    data = rootgrp.createVariable(f"dist_fact_ss{k + 1}", "f8", (dim2,))
                    data[:] = np.concatenate(dfs) if dfs else np.array([], float)


register_format("exodus", [".e", ".exo", ".ex2"], read, {"exodus": write})
