"""
I/O for XDMF.
https://xdmf.org/index.php/XDMF_Model_and_Format
"""

import os
import pathlib
from io import BytesIO
from xml.etree import ElementTree as ET

import numpy as np

from .._common import cell_data_from_raw, raw_from_cell_data, write_xml
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from .._mesh import CellBlock, Mesh
from .common import (
    attribute_type,
    dtype_to_format_string,
    meshio_to_xdmf_type,
    meshio_type_to_xdmf_index,
    numpy_to_xdmf_dtype,
    translate_mixed_cells,
    xdmf_to_meshio_type,
    xdmf_to_numpy_type,
)

# Structured topologies. Unlike the unstructured types they don't list node
# indices explicitly; the connectivity is implied by the Dimensions.
structured_topology_types = {
    "2DCoRectMesh",
    "2DRectMesh",
    "2DSMesh",
    "3DCoRectMesh",
    "3DRectMesh",
    "3DSMesh",
}


def _reshape_structured_data(data, dims, center):
    # Data on a structured grid is stored with the topology axes reversed
    # (last axis varies fastest), mirroring the C order of the coordinate
    # array. Flatten to (num_items, ...).
    if center == "Node":
        shape = tuple(reversed(dims))
    else:
        shape = tuple(max(d - 1, 1) for d in reversed(dims))

    num_items = int(np.prod(shape))
    if data.shape[: len(shape)] != shape and data.shape[0] != num_items:
        raise ReadError(
            "Structured grid attribute dimensions "
            f"{' '.join(str(s) for s in data.shape)} don't match the "
            f"grid dimensions {' '.join(str(s) for s in shape)}."
        )
    return data.reshape((num_items,) + data.shape[len(shape) :])


def _structured_cells(dims):
    ndim = len(dims)
    active_axes = [a for a in range(ndim) if dims[a] > 1]
    zero = [0] * ndim

    # Corner offsets of one element in VTK/meshio node order.
    if len(active_axes) == 0:
        return [CellBlock("vertex", np.array([[0]]))]
    elif len(active_axes) == 1:
        cell_type = "line"
        corner_offsets = []
        for step in (0, 1):
            offset = list(zero)
            offset[active_axes[0]] = step
            corner_offsets.append(tuple(offset))
    elif len(active_axes) == 2:
        cell_type = "quad"
        corner_offsets = []
        for first, second in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            offset = list(zero)
            offset[active_axes[0]] = first
            offset[active_axes[1]] = second
            corner_offsets.append(tuple(offset))
    else:
        cell_type = "hexahedron"
        corner_offsets = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]

    # Node ids follow node(i, j, k) = i + n_i * (j + n_j * k). Build the
    # origin-corner id of every cell and add the corner offsets.
    strides = np.cumprod([1, *dims[:-1]]).astype(np.int64)
    cell_shape = tuple(max(d - 1, 1) for d in dims)
    indices = np.indices(cell_shape)
    base = np.zeros(cell_shape, dtype=np.int64)
    for axis in range(ndim):
        base += indices[axis] * strides[axis]

    corner_ids = [
        sum(offset[axis] * strides[axis] for axis in range(ndim))
        for offset in corner_offsets
    ]
    # Fortran order: axis 0 varies fastest, matching both the node numbering
    # and cell-centered attributes whose dimensions are reversed.
    connectivity = np.column_stack(
        [(base + corner_id).ravel(order="F") for corner_id in corner_ids]
    )
    return [CellBlock(cell_type, connectivity)]


def read(filename):
    return XdmfReader(filename).read()


class XdmfReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        parser = ET.XMLParser()
        tree = ET.parse(self.filename, parser)
        root = tree.getroot()

        if root.tag != "Xdmf":
            raise ReadError()

        version = root.attrib["Version"]

        if version.split(".")[0] == "2":
            return self.read_xdmf2(root)

        if version.split(".")[0] != "3":
            raise ReadError(f"Unknown XDMF version {version}.")

        return self.read_xdmf3(root)

    def _read_data_item(self, data_item, root=None):
        import h5py

        if "Reference" in data_item.attrib:
            reference = data_item.attrib["Reference"]
            xpath = (data_item.text if reference == "XML" else reference).strip()
            if xpath.startswith("/"):
                return self._read_data_item(
                    root.find(".//" + "/".join(xpath.split("/")[2:])), root
                )
            raise ValueError(f"Can't read XPath {xpath}.")

        dims = [int(d) for d in data_item.get("Dimensions").split()]

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files
        # out there use both keys interchangeably.
        if "DataType" in data_item.attrib and "NumberType" in data_item.attrib:
            raise ReadError()
        elif "DataType" in data_item.attrib:
            data_type = data_item.attrib["DataType"]
        elif "NumberType" in data_item.attrib:
            data_type = data_item.attrib["NumberType"]
        else:
            # Default, see
            # <https://xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
            data_type = "Float"

        precision = data_item.get("Precision", default="4")
        fmt = data_item.attrib["Format"]

        if fmt == "XML":
            dtype = xdmf_to_numpy_type[(data_type, precision)]
            if data_item.text.strip() == "":
                # https://github.com/numpy/numpy/issues/18435
                data = np.empty((0,), dtype=dtype)
            else:
                data = np.fromstring(data_item.text, dtype=dtype, sep=" ")
            return data.reshape(dims)

        elif fmt == "Binary":
            return np.fromfile(
                data_item.text.strip(), dtype=xdmf_to_numpy_type[(data_type, precision)]
            ).reshape(dims)

        if fmt != "HDF":
            raise ReadError(f"Unknown XDMF Format '{fmt}'.")

        info = data_item.text.strip()
        filename, h5path = info.split(":")

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        dirname = pathlib.Path(self.filename).resolve().parent
        full_hdf5_path = dirname / filename

        f = h5py.File(full_hdf5_path, "r")

        # Some files don't contain the leading slash /.
        if h5path[0] == "/":
            h5path = h5path[1:]

        for key in h5path.split("/"):
            f = f[key]
        # `[()]` gives a np.ndarray
        return f[()]

    def read_information(self, c_data):
        field_data = {}
        root = ET.fromstring(c_data)
        for child in root:
            str_tag = child.attrib["key"]
            dim = int(child.attrib["dim"])
            if child.text is None:
                raise ReadError()
            num_tag = int(child.text)
            field_data[str_tag] = np.array([num_tag, dim])
        return field_data

    def _read_structured_grid(self, grid, topology, is_xdmf2):
        # Structured grids (CoRectMesh, RectMesh, SMesh) define the cells
        # implicitly through their Dimensions. Expand them into an explicit
        # unstructured `Mesh`.
        #
        # The Topology Dimensions are (n_i, n_j[, n_k]) with i the fastest
        # varying axis. Coordinate and attribute arrays are stored with the
        # axes reversed, i.e. the last dimension varies fastest, which is the
        # usual C order. The implicit node numbering is
        #   node(i, j[, k]) = i + n_i * (j + n_j * k)
        dimensions = topology.attrib.get("Dimensions")
        if dimensions is None:
            raise ReadError("Structured grid <Topology> needs Dimensions.")
        dims = [int(d) for d in dimensions.split()]
        if len(dims) not in (2, 3):
            raise ReadError(
                "XDMF reader: Only 2D and 3D structured meshes are supported."
            )
        if any(d < 1 for d in dims):
            raise ReadError("Structured grid Dimensions must be positive.")

        geometry = None
        attributes = []
        field_data = {}
        for c in grid:
            if c.tag == "Topology":
                continue
            elif c.tag == "Geometry":
                if geometry is not None:
                    raise ReadError()
                geometry = c
            elif c.tag == "Attribute":
                attributes.append(c)
            elif c.tag == "Information":
                c_data = c.text
                if not c_data:
                    raise ReadError()
                field_data = self.read_information(c_data)
            else:
                raise ReadError(f"Unknown section '{c.tag}'.")

        if geometry is None:
            raise ReadError("Structured grid needs a <Geometry> section.")

        points = self._read_structured_geometry(geometry, dims)
        cells = _structured_cells(dims)

        point_data = {}
        cell_data_raw = {}
        for attribute in attributes:
            data_items = list(attribute)
            if len(data_items) != 1:
                raise ReadError()
            data = self._read_data_item(data_items[0])

            name = attribute.attrib["Name"]
            center = attribute.attrib["Center"]
            if center == "Node":
                point_data[name] = _reshape_structured_data(data, dims, center)
            elif center == "Cell":
                cell_data_raw[name] = _reshape_structured_data(data, dims, center)
            elif is_xdmf2 and center == "Grid":
                # TODO field data?
                continue
            else:
                raise ReadError(f"Unknown center '{center}'.")

        cell_data = cell_data_from_raw(cells, cell_data_raw)

        return Mesh(
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    def _read_structured_geometry(self, geometry, dims):
        ndim = len(dims)

        if "Type" in geometry.attrib and "GeometryType" in geometry.attrib:
            raise ReadError()
        geo_type = geometry.attrib.get("Type", geometry.attrib.get("GeometryType"))
        if geo_type is None:
            raise ReadError("Structured grid <Geometry> needs a type.")
        geo_type = geo_type.upper()

        data_items = list(geometry)

        if geo_type in ("ORIGIN_DXDYDZ", "ORIGIN_DXDY"):
            # Uniform spacing: origin and per-axis displacement vectors.
            if len(data_items) != 2:
                raise ReadError(
                    f"{geo_type} geometry needs exactly 2 DataItems "
                    "(origin and displacement)."
                )
            origin = np.asarray(
                self._read_data_item(data_items[0]), dtype=float
            ).ravel()
            displacement = np.asarray(
                self._read_data_item(data_items[1]), dtype=float
            ).ravel()
            if origin.size != ndim or displacement.size != ndim:
                raise ReadError(
                    f"{geo_type} origin/displacement vectors must have {ndim} "
                    "entries."
                )
            axes = [
                origin[axis] + displacement[axis] * np.arange(dims[axis])
                for axis in range(ndim)
            ]
        elif geo_type in ("VXVYVZ", "VXVY"):
            # One coordinate vector per axis: VX (i), VY (j), VZ (k).
            if len(data_items) != ndim:
                raise ReadError(
                    f"{geo_type} geometry needs {ndim} DataItems, "
                    f"found {len(data_items)}."
                )
            axes = []
            for axis, data_item in enumerate(data_items):
                coordinates = np.asarray(
                    self._read_data_item(data_item), dtype=float
                ).ravel()
                if coordinates.size != dims[axis]:
                    raise ReadError(
                        f"{geo_type} axis {axis} has {coordinates.size} entries, "
                        f"expected {dims[axis]}."
                    )
                axes.append(coordinates)
        elif geo_type in ("XYZ", "XY"):
            # Explicit coordinates. The array has shape
            # (n_k, n_j, n_i[, 3]) with i varying fastest.
            if len(data_items) != 1:
                raise ReadError()
            data = np.asarray(self._read_data_item(data_items[0]), dtype=float)
            num_points = int(np.prod(dims))
            expected_shape = tuple(reversed(dims)) + (ndim,)
            if data.shape != expected_shape and data.size != num_points * ndim:
                raise ReadError(
                    f"{geo_type} geometry dimensions don't match the "
                    f"structured grid dimensions {dims}."
                )
            return data.reshape(num_points, ndim)
        else:
            raise ReadError(f'Illegal geometry type "{geo_type}".')

        # Expand the per-axis coordinates. Fortran order keeps axis 0 (i) the
        # fastest varying axis, matching the implicit node numbering.
        grids = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([grid.ravel(order="F") for grid in grids])

    def read_xdmf2(self, root):  # noqa: C901
        domains = list(root)
        if len(domains) != 1:
            raise ReadError()
        domain = domains[0]
        if domain.tag != "Domain":
            raise ReadError()

        grids = list(domain)
        if len(grids) != 1:
            raise ReadError("XDMF reader: Only supports one grid right now.")
        grid = grids[0]
        if grid.tag != "Grid":
            raise ReadError()

        if "GridType" in grid.attrib and grid.attrib["GridType"] != "Uniform":
            raise ReadError()

        # Structured grids don't list cell connectivity explicitly.
        for c in grid:
            if c.tag != "Topology":
                continue
            topology_type = c.attrib.get("TopologyType", c.attrib.get("Type"))
            if topology_type in structured_topology_types:
                return self._read_structured_grid(grid, c, is_xdmf2=True)
            break

        points = None
        cells = []
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
            if c.tag == "Topology":
                data_items = list(c)
                if len(data_items) != 1:
                    message = (
                        "Need exactly 1 data item in <Topology>, "
                        f"found {len(data_items)}."
                    )
                    raise ReadError(message)

                topology_type = c.attrib["TopologyType"]
                if topology_type == "Mixed":
                    cells = translate_mixed_cells(
                        np.fromstring(
                            data_items[0].text,
                            int,
                            int(data_items[0].get("Dimensions")),
                            " ",
                        )
                    )
                else:
                    data = self._read_data_item(data_items[0])
                    cells.append(CellBlock(xdmf_to_meshio_type[topology_type], data))

            elif c.tag == "Geometry":
                if "GeometryType" in c.attrib:
                    geo_type = c.attrib["GeometryType"]
                    if geo_type != "XYZ":
                        raise ReadError(f'Expected GeometryType "XYZ", not {geo_type}.')
                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                points = self._read_data_item(data_items[0])

            elif c.tag == "Information":
                c_data = c.text
                if not c_data:
                    raise ReadError()
                field_data = self.read_information(c_data)

            elif c.tag == "Attribute":
                # assert c.attrib['Active'] == '1'
                # assert c.attrib['AttributeType'] == 'None'

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()

                data = self._read_data_item(data_items[0])

                name = c.attrib["Name"]
                if c.attrib["Center"] == "Node":
                    point_data[name] = data
                elif c.attrib["Center"] == "Cell":
                    cell_data_raw[name] = data
                else:
                    # TODO field data?
                    if c.attrib["Center"] != "Grid":
                        raise ReadError()
            else:
                raise ReadError(f"Unknown section '{c.tag}'.")

        cell_data = cell_data_from_raw(cells, cell_data_raw)

        return Mesh(
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    def read_xdmf3(self, root):  # noqa: C901
        domains = list(root)
        if len(domains) != 1:
            raise ReadError()
        domain = domains[0]
        if domain.tag != "Domain":
            raise ReadError()

        grids = list(domain)
        if len(grids) != 1:
            raise ReadError("XDMF reader: Only supports one grid right now.")
        grid = grids[0]
        if grid.tag != "Grid":
            raise ReadError()

        # Structured grids don't list cell connectivity explicitly.
        for c in grid:
            if c.tag != "Topology":
                continue
            topology_type = c.attrib.get("Type", c.attrib.get("TopologyType"))
            if topology_type in structured_topology_types:
                return self._read_structured_grid(grid, c, is_xdmf2=False)
            break

        points = None
        cells = []
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
            if c.tag == "Topology":
                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]

                data = self._read_data_item(data_item)

                # The XDMF2 key is `TopologyType`, just `Type` for XDMF3. Allow either
                # one.
                if "Type" in c.attrib and "TopologyType" in c.attrib:
                    raise ReadError()
                elif "Type" in c.attrib:
                    cell_type = c.attrib["Type"]
                else:
                    cell_type = c.attrib["TopologyType"]

                if cell_type == "Mixed":
                    cells = translate_mixed_cells(data)
                else:
                    cells.append(CellBlock(xdmf_to_meshio_type[cell_type], data))

            elif c.tag == "Geometry":
                if "Type" in c.attrib and "GeometryType" in c.attrib:
                    raise ReadError()
                elif "Type" in c.attrib:
                    geometry_type = c.attrib["Type"]
                else:
                    geometry_type = c.attrib["GeometryType"]

                if geometry_type not in ["XY", "XYZ"]:
                    raise ReadError(f'Illegal geometry type "{geometry_type}".')

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]
                points = self._read_data_item(data_item)

            elif c.tag == "Information":
                c_data = c.text
                if not c_data:
                    raise ReadError()
                field_data = self.read_information(c_data)

            elif c.tag == "Attribute":
                # Don't be too strict here: FEniCS, for example, calls this
                # 'AttributeType'.
                # assert c.attrib['Type'] == 'None'

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]

                data = self._read_data_item(data_item)

                if c.attrib["Center"] not in ["Node", "Cell"]:
                    raise ReadError(f"Unknown center '{c.attrib['Center']}'.")

                name = c.attrib["Name"]
                if c.attrib["Center"] == "Node":
                    point_data[name] = data
                else:
                    cell_data_raw[name] = data

            else:
                raise ReadError(f"Unknown section '{c.tag}'.")

        cell_data = cell_data_from_raw(cells, cell_data_raw)

        return Mesh(
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )


class XdmfWriter:
    def __init__(
        self, filename, mesh, data_format="HDF", compression="gzip", compression_opts=4
    ):
        import h5py

        if data_format not in ["XML", "Binary", "HDF"]:
            raise WriteError(
                "Unknown XDMF data format "
                f"'{data_format}' (use 'XML', 'Binary', or 'HDF'.)"
            )

        self.filename = pathlib.Path(filename)
        self.data_format = data_format
        self.data_counter = 0
        self.compression = compression
        self.compression_opts = None if compression is None else compression_opts

        if data_format == "HDF":
            self.h5_filename = self.filename.with_suffix(".h5")
            self.h5_file = h5py.File(self.h5_filename, "w")

        xdmf_file = ET.Element("Xdmf", Version="3.0")

        domain = ET.SubElement(xdmf_file, "Domain")
        grid = ET.SubElement(domain, "Grid", Name="Grid")
        # information = ET.SubElement(
        #     grid, "Information", Name="Information", Value=str(len(mesh.field_data))
        # )

        self.write_points(grid, mesh.points)
        # self.field_data(mesh.field_data, information)
        self.write_cells(mesh.cells, grid)
        self.write_point_data(mesh.point_data, grid)
        self.write_cell_data(mesh.cell_data, grid)

        ET.register_namespace("xi", "https://www.w3.org/2001/XInclude/")

        write_xml(filename, xdmf_file)

    def numpy_to_xml_string(self, data):
        if self.data_format == "XML":
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype.name]
            np.savetxt(s, data, fmt)
            return "\n" + s.getvalue().decode()
        elif self.data_format == "Binary":
            base = os.path.splitext(self.filename)[0]
            bin_filename = f"{base}{self.data_counter}.bin"
            self.data_counter += 1
            # write binary data to file
            with open(bin_filename, "wb") as f:
                data.tofile(f)
            return bin_filename

        if self.data_format != "HDF":
            raise WriteError(f'Unknown data format "{self.data_format}"')
        name = f"data{self.data_counter}"
        self.data_counter += 1
        self.h5_file.create_dataset(
            name,
            data=data,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        return os.path.basename(self.h5_filename) + ":/" + name

    def write_points(self, grid, points):
        if points.shape[1] > 3:
            raise WriteError("Can only write points up to dimension 3.")

        geometry_type = "XYZ"[: points.shape[1]]

        geo = ET.SubElement(grid, "Geometry", GeometryType=geometry_type)
        dt, prec = numpy_to_xdmf_dtype[points.dtype.name]
        dim = "{} {}".format(*points.shape)
        data_item = ET.SubElement(
            geo,
            "DataItem",
            DataType=dt,
            Dimensions=dim,
            Format=self.data_format,
            Precision=prec,
        )
        data_item.text = self.numpy_to_xml_string(points)

    def write_cells(self, cells, grid):
        if len(cells) == 0:
            return
        if len(cells) == 1:
            meshio_type = cells[0].type
            num_cells = len(cells[0].data)
            xdmf_type = meshio_to_xdmf_type[meshio_type][0]
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType=xdmf_type,
                NumberOfElements=str(num_cells),
                NodesPerElement=str(cells[0].data.shape[1]),
            )
            dt, prec = numpy_to_xdmf_dtype[cells[0].data.dtype.name]
            dim = "{} {}".format(*cells[0].data.shape)
            data_item = ET.SubElement(
                topo,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(cells[0].data)

        else:
            assert len(cells) > 1
            total_num_cells = sum(c.data.shape[0] for c in cells)
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType="Mixed",
                NumberOfElements=str(total_num_cells),
            )
            total_num_cell_items = sum(np.prod(c.data.shape) for c in cells)
            num_vertices_and_lines = sum(
                c.data.shape[0] for c in cells if c.type in {"vertex", "line"}
            )
            dim = str(total_num_cell_items + total_num_cells + num_vertices_and_lines)
            cd = np.concatenate(
                [
                    np.hstack(
                        [
                            np.full(
                                (
                                    cell_block.data.shape[0],
                                    2 if cell_block.type in {"vertex", "line"} else 1,
                                ),
                                meshio_type_to_xdmf_index[cell_block.type],
                            ),
                            cell_block.data,
                        ]
                    ).flatten()
                    for cell_block in cells
                ]
            )
            dt, prec = numpy_to_xdmf_dtype[cd.dtype.name]
            data_item = ET.SubElement(
                topo,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(cd)

    def write_point_data(self, point_data, grid):
        for name, data in point_data.items():
            att = ET.SubElement(
                grid,
                "Attribute",
                Name=name,
                AttributeType=attribute_type(data),
                Center="Node",
            )
            dt, prec = numpy_to_xdmf_dtype[data.dtype.name]
            dim = " ".join([str(s) for s in data.shape])
            data_item = ET.SubElement(
                att,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(data)

    def write_cell_data(self, cell_data, grid):
        raw = raw_from_cell_data(cell_data)
        for name, data in raw.items():
            att = ET.SubElement(
                grid,
                "Attribute",
                Name=name,
                AttributeType=attribute_type(data),
                Center="Cell",
            )
            dt, prec = numpy_to_xdmf_dtype[data.dtype.name]
            dim = " ".join([str(s) for s in data.shape])
            data_item = ET.SubElement(
                att,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(data)

    # The original idea was to implement field data as XML CDATA. Unfortunately, in
    # Python's XML, CDATA handled poorly. There are workarounds, e.g.,
    # <https://stackoverflow.com/a/30019607/353337>, but those mess around with
    # ET._serialize_xml and lead to bugs elsewhere
    # <https://github.com/nschloe/meshio/issues/806>.
    # def field_data(self, field_data, information):
    #     info = ET.Element("main")
    #     for name, data in field_data.items():
    #         data_item = ET.SubElement(info, "map", key=name, dim=str(data[1]))
    #         data_item.text = str(data[0])
    #     information.text = ET.CDATA(ET.tostring(info))
    #     information.append(CDATA(ET.tostring(info).decode()))


def write(*args, **kwargs):
    XdmfWriter(*args, **kwargs)


# TODO register all xdmf except hdf outside this try block
register_format(
    "xdmf",
    [".xdmf", ".xmf"],
    read,
    {"xdmf": write},
)
