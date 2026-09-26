from __future__ import annotations

import os
import pathlib
import re
from io import BytesIO
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import ArrayLike

from .._common import cell_data_from_raw, raw_from_cell_data, write_xml
from .._exceptions import ReadError, WriteError
from .._mesh import CellBlock
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

# xpointer(//Grid[@Name="mesh"]/Topology),
# xpointer(//Grid[@Name="mesh"]/*[self::Topology or self::Geometry]), ...
_XPOINTER_RE = re.compile(
    r"xpointer\(//Grid\[@Name=[\"'](?P<name>[^\"']+)[\"']\]/(?P<selector>[^)]+)\)"
)


class TimeSeriesReader:
    def __init__(self, filename):  # noqa: C901
        self.filename = pathlib.Path(filename)

        parser = ET.XMLParser()
        tree = ET.parse(self.filename, parser)
        root = tree.getroot()

        if root.tag != "Xdmf":
            raise ReadError()

        version = root.attrib["Version"]
        if version.split(".")[0] != "3":
            raise ReadError(f"Unknown XDMF version {version}.")

        domains = list(root)
        if len(domains) != 1:
            raise ReadError()
        self.domain = domains[0]
        if self.domain.tag != "Domain":
            raise ReadError()

        grids = list(self.domain)

        # find the collection grid
        collection_grid = None
        for g in grids:
            if g.get("GridType") == "Collection":
                collection_grid = g
        if collection_grid is None:
            raise ReadError("Couldn't find the mesh grid")
        if collection_grid.tag != "Grid":
            raise ReadError()
        if collection_grid.get("CollectionType") != "Temporal":
            raise ReadError()

        # get the collection at once
        self.collection = list(collection_grid)
        self.num_steps = len(self.collection)
        self.cells = None
        self.hdf5_files = {}
        self._step_meshes = {}

        # find the uniform grid
        self.mesh_grid = None
        for g in grids:
            if g.get("GridType") == "Uniform":
                self.mesh_grid = g
        # if not found, take the first uniform grid in the collection grid
        if self.mesh_grid is None:
            for g in self.collection:
                if g.get("GridType") == "Uniform":
                    self.mesh_grid = g
                    break
        if self.mesh_grid is not None and self.mesh_grid.tag != "Grid":
            raise ReadError()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Those files are opened in _read_data_item()
        for f in self.hdf5_files.values():
            f.close()

    def read_points_cells(self, k: int | None = None):
        """Read the points and cells of the mesh.

        Without argument, read the static mesh of the time series (the
        historical behavior). With a time step index `k`, read the mesh of
        that particular step; every step may define its own Geometry and
        Topology or reference parts of previously defined meshes.
        """
        if k is None:
            if self.mesh_grid is None:
                raise ReadError("Couldn't find the mesh grid")
            points, cells = self._read_mesh_elements(
                self._mesh_elements(self.mesh_grid)
            )
            self.cells = cells
            return points, cells

        mesh = self._read_step_mesh(k)
        if mesh is None:
            raise ReadError(f"Couldn't find mesh information for time step {k}")
        return mesh

    def _step_grid(self, k: int):
        if not 0 <= k < self.num_steps:
            raise ReadError(
                f"Time step index {k} out of range ({self.num_steps} steps)"
            )
        return self.collection[k]

    def _read_step_mesh(self, k: int):
        """Return the (points, cells) of time step k, or None if the step
        carries no mesh information and there is no static mesh."""
        if k not in self._step_meshes:
            elements = self._mesh_elements(self._step_grid(k))
            if elements:
                mesh = self._read_mesh_elements(elements)
            elif self.mesh_grid is not None:
                # fall back to the static mesh
                mesh = self._read_mesh_elements(self._mesh_elements(self.mesh_grid))
            else:
                mesh = None
            self._step_meshes[k] = mesh
        return self._step_meshes[k]

    def _mesh_elements(self, grid):
        """Collect the Topology/Geometry elements of a grid, resolving
        xi:include references to previously defined grids."""
        elements = []
        for child in grid:
            # strip the namespace (relevant for xi:include)
            tag = child.tag.split("}")[-1]
            if tag in ["Topology", "Geometry"]:
                elements.append(child)
            elif tag == "include":
                elements.extend(self._resolve_xpointer(child.attrib["xpointer"]))
        return elements

    def _resolve_xpointer(self, xpointer: str):
        match = _XPOINTER_RE.fullmatch(xpointer.strip())
        if match is None:
            raise ReadError(f"Cannot parse xpointer '{xpointer}'")
        name = match.group("name")
        selector = match.group("selector")

        # Find the referenced grid by its Name attribute. Many grids contain
        # elements with identical tag names (Topology, Geometry), so the Name
        # is the only reliable handle.
        target = None
        for g in list(self.domain) + self.collection:
            if g.tag == "Grid" and g.get("Name") == name:
                target = g
                break
        if target is None:
            raise ReadError(f"Cannot resolve xpointer '{xpointer}'")

        if selector == "*[self::Topology or self::Geometry]":
            tags = ("Topology", "Geometry")
        elif selector in ("Topology", "Geometry"):
            tags = (selector,)
        else:
            raise ReadError(f"Unsupported xpointer selector '{selector}'")

        elements = [c for c in target if c.tag in tags]
        if not elements:
            raise ReadError(f"Cannot resolve xpointer '{xpointer}'")
        return elements

    def _read_mesh_elements(self, elements):
        points = None
        cells = []

        for c in elements:
            if c.tag == "Topology":
                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]

                data = self._read_data_item(data_item)

                # The XDMF2 key is `TopologyType`, just `Type` for XDMF3.
                # Allow both.
                if "Type" in c.attrib:
                    if "TopologyType" in c.attrib:
                        raise ReadError("Both 'Type' and 'TopologyType' keys present")
                    cell_type = c.attrib["Type"]
                else:
                    cell_type = c.attrib["TopologyType"]

                if cell_type == "Mixed":
                    cells = translate_mixed_cells(data)
                else:
                    cells.append(CellBlock(xdmf_to_meshio_type[cell_type], data))

            elif c.tag == "Geometry":
                if "GeometryType" in c.attrib:
                    geo_type = c.attrib["GeometryType"]
                    if geo_type not in ["XY", "XYZ"]:
                        raise ReadError(f"Unknown geometry_type '{geo_type}'")

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]
                points = self._read_data_item(data_item)

        return points, cells

    def read_data(self, k: int):
        point_data = {}
        cell_data_raw = {}

        t = None

        for c in list(self._step_grid(k)):
            if c.tag == "Time":
                t = float(c.attrib["Value"])
            elif c.tag == "Attribute":
                name = c.get("Name")

                if len(list(c)) != 1:
                    raise ReadError()
                data_item = list(c)[0]
                data = self._read_data_item(data_item)

                if c.get("Center") == "Node":
                    point_data[name] = data
                else:
                    if c.get("Center") != "Cell":
                        raise ReadError()
                    cell_data_raw[name] = data
            else:
                # skip the mesh (Topology/Geometry, possibly xi:included);
                # it is handled by _read_step_mesh()
                continue

        mesh = self._read_step_mesh(k)
        if mesh is None:
            raise ReadError(f"Couldn't find mesh information for time step {k}")
        points, cells = mesh

        # check the data against the mesh of this time step
        if points is not None:
            num_points = points.shape[0]
            for name, data in point_data.items():
                if data.shape[0] != num_points:
                    raise ReadError(
                        f"Time step {k}: point_data '{name}' has "
                        f"{data.shape[0]} entries, but the mesh has "
                        f"{num_points} points"
                    )
        num_cells = sum(len(cell_block) for cell_block in cells)
        for name, data in cell_data_raw.items():
            if data.shape[0] != num_cells:
                raise ReadError(
                    f"Time step {k}: cell_data '{name}' has {data.shape[0]} "
                    f"entries, but the mesh has {num_cells} cells"
                )

        cell_data = cell_data_from_raw(cells, cell_data_raw)
        if t is None:
            raise ReadError()

        return t, point_data, cell_data

    def _read_data_item(self, data_item):
        dims = [int(d) for d in data_item.get("Dimensions").split()]

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files out there
        # use both keys interchangeably.
        if data_item.get("DataType"):
            if data_item.get("NumberType"):
                raise ReadError()
            data_type = data_item.get("DataType")
        elif data_item.get("NumberType"):
            if data_item.get("DataType"):
                raise ReadError()
            data_type = data_item.get("NumberType")
        else:
            # Default, see
            # <https://xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
            data_type = "Float"

        try:
            precision = data_item.attrib["Precision"]
        except KeyError:
            precision = "4"

        data_format = data_item.attrib["Format"]

        if data_format == "XML":
            return np.fromstring(
                data_item.text,
                dtype=xdmf_to_numpy_type[(data_type, precision)],
                sep=" ",
            ).reshape(dims)
        elif data_format == "Binary":
            return np.fromfile(
                data_item.text.strip(), dtype=xdmf_to_numpy_type[(data_type, precision)]
            ).reshape(dims)

        if data_format != "HDF":
            raise ReadError(f"Unknown XDMF Format '{data_format}'.")

        info = data_item.text.strip()
        filename, h5path = info.split(":")

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        dirpath = self.filename.resolve().parent
        full_hdf5_path = dirpath / filename

        if full_hdf5_path in self.hdf5_files:
            f = self.hdf5_files[full_hdf5_path]
        else:
            import h5py

            f = h5py.File(full_hdf5_path, "r")
            self.hdf5_files[full_hdf5_path] = f

        if h5path[0] != "/":
            raise ReadError()

        for key in h5path[1:].split("/"):
            f = f[key]
        # `[()]` gives a np.ndarray
        return f[()]


class TimeSeriesWriter:
    def __init__(self, filename, data_format: str = "HDF") -> None:
        if data_format not in ["XML", "Binary", "HDF"]:
            raise WriteError(
                "Unknown XDMF data format "
                f"'{data_format}' (use 'XML', 'Binary', or 'HDF'.)"
            )

        self.filename = pathlib.Path(filename)
        self.data_format = data_format
        self.data_counter = 0

        self.xdmf_file = ET.Element("Xdmf", Version="3.0")

        self.domain = ET.SubElement(self.xdmf_file, "Domain")
        self.collection = ET.SubElement(
            self.domain,
            "Grid",
            Name="TimeSeries_meshio",
            GridType="Collection",
            CollectionType="Temporal",
        )

        ET.register_namespace("xi", "https://www.w3.org/2001/XInclude/")

        self.has_mesh = False
        self.mesh_name = "mesh"
        self._mesh_counter = 0
        # names of the grids holding the most recently defined
        # Topology/Geometry; new time steps may reference them
        self._topology_ref = None
        self._geometry_ref = None
        # (grid name, cell blocks) of all grids with an inline Topology,
        # used to deduplicate identical topologies
        self._topologies = []

    def __enter__(self):
        if self.data_format == "HDF":
            import h5py

            self.h5_filename = self.filename.stem + ".h5"
            self.h5_file = h5py.File(self.h5_filename, "w")
        return self

    def __exit__(self, *_):
        write_xml(self.filename, self.xdmf_file)
        if self.data_format == "HDF":
            self.h5_file.close()

    def write_points_cells(
        self,
        points: ArrayLike,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
    ) -> None:
        # <Grid Name="mesh" GridType="Uniform">
        #   <Topology NumberOfElements="16757" TopologyType="Triangle" NodesPerElement="3">
        #     <DataItem Dimensions="16757 3" NumberType="UInt" Format="HDF">maxwell.h5:/Mesh/0/mesh/topology</DataItem>
        #   </Topology>
        #   <Geometry GeometryType="XY">
        #     <DataItem Dimensions="8457 2" Format="HDF">maxwell.h5:/Mesh/0/mesh/geometry</DataItem>
        #   </Geometry>
        # </Grid>
        grid = ET.SubElement(
            self.domain, "Grid", Name=self.mesh_name, GridType="Uniform"
        )
        cell_blocks = self._to_cell_blocks(cells)
        self.points(grid, np.asarray(points))
        self.cells(cell_blocks, grid)
        self.has_mesh = True
        self._topology_ref = self.mesh_name
        self._geometry_ref = self.mesh_name
        self._topologies.append((self.mesh_name, cell_blocks))

    def write_data(self, t, point_data=None, cell_data=None, points=None, cells=None):
        """Write the data of one time step.

        Without `points`/`cells`, the step references the most recently
        defined mesh (the static mesh written with `write_points_cells` for
        static time series). Pass `points` to move the nodes of the current
        mesh; pass `points` and `cells` for a completely new mesh. If the
        given `cells` are identical to a previously written topology, only
        a reference to that topology is written.
        """
        cell_data = {} if cell_data is None else cell_data
        # <Grid>
        #   <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_phi&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
        #   <Time Value="3.3333333333333335e-05" />
        #   <Attribute Name="phi" AttributeType="Scalar" Center="Node">
        #     <DataItem Dimensions="8457 1" Format="HDF">maxwell.h5:/VisualisationVector/1</DataItem>
        #   </Attribute>
        # </Grid>
        grid = ET.SubElement(self.collection, "Grid", GridType="Uniform")

        if points is None and cells is None:
            # reference the current mesh
            if self._topology_ref is None or self._geometry_ref is None:
                raise WriteError(
                    "No mesh defined yet; call write_points_cells() first or "
                    "pass points and cells"
                )
            if self._topology_ref == self._geometry_ref:
                ptr = f'xpointer(//Grid[@Name="{self._topology_ref}"]/*[self::Topology or self::Geometry])'
                self._include(grid, ptr)
            else:
                self._include(
                    grid,
                    f'xpointer(//Grid[@Name="{self._topology_ref}"]/Topology)',
                )
                self._include(
                    grid,
                    f'xpointer(//Grid[@Name="{self._geometry_ref}"]/Geometry)',
                )
        else:
            # this step defines its own mesh, at least in part
            self._mesh_counter += 1
            grid_name = f"mesh_{self._mesh_counter}"
            grid.set("Name", grid_name)
            if points is not None:
                self.points(grid, np.asarray(points))
                self._geometry_ref = grid_name
            else:
                if self._geometry_ref is None:
                    raise WriteError("No geometry defined yet; pass points")
                self._include(
                    grid,
                    f'xpointer(//Grid[@Name="{self._geometry_ref}"]/Geometry)',
                )
            if cells is not None:
                cell_blocks = self._to_cell_blocks(cells)
                ref = self._find_topology(cell_blocks)
                if ref is None:
                    self.cells(cell_blocks, grid)
                    self._topologies.append((grid_name, cell_blocks))
                    self._topology_ref = grid_name
                else:
                    # same topology as before, only reference it
                    self._include(
                        grid, f'xpointer(//Grid[@Name="{ref}"]/Topology)'
                    )
            else:
                if self._topology_ref is None:
                    raise WriteError("No topology defined yet; pass cells")
                self._include(
                    grid,
                    f'xpointer(//Grid[@Name="{self._topology_ref}"]/Topology)',
                )

        ET.SubElement(grid, "Time", Value=str(t))

        if point_data:
            self.point_data(point_data, grid)

        # permit old dict structure, convert it to list of tuples
        for name, entry in cell_data.items():
            if isinstance(entry, dict):
                cell_data[name] = np.array(list(entry.values()))
        if cell_data:
            self.cell_data(cell_data, grid)

    @staticmethod
    def _include(grid, ptr: str) -> None:
        ET.SubElement(grid, "{http://www.w3.org/2003/XInclude}include", xpointer=ptr)

    @staticmethod
    def _to_cell_blocks(cells) -> list:
        if isinstance(cells, dict):
            # convert dict to list of tuples
            cells = list(cells.items())

        # convert to cell_blocks
        cell_blocks = []
        for cell_block in cells:
            if isinstance(cell_block, tuple):
                cell_type, data = cell_block
                cell_block = CellBlock(cell_type, np.asarray(data))
            cell_blocks.append(cell_block)
        return cell_blocks

    def _find_topology(self, cell_blocks) -> str | None:
        """Return the name of a grid whose Topology matches `cell_blocks`,
        or None if there is none."""
        for name, blocks in self._topologies:
            if len(blocks) != len(cell_blocks):
                continue
            if all(
                block.type == new_block.type
                and block.data.shape == new_block.data.shape
                and block.data.dtype == new_block.data.dtype
                and np.array_equal(block.data, new_block.data)
                for block, new_block in zip(blocks, cell_blocks)
            ):
                return name
        return None

    def numpy_to_xml_string(self, data):
        if self.data_format == "XML":
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype.name]
            np.savetxt(s, data.flatten(), fmt)
            return s.getvalue().decode()
        elif self.data_format == "Binary":
            bin_filename = f"{self.filename.stem}{self.data_counter}.bin"
            self.data_counter += 1
            # write binary data to file
            with open(bin_filename, "wb") as f:
                data.tofile(f)
            return bin_filename

        if self.data_format != "HDF":
            raise WriteError()
        name = f"data{self.data_counter}"
        self.data_counter += 1
        self.h5_file.create_dataset(name, data=data)
        return os.path.basename(self.h5_filename) + ":/" + name

    def points(self, grid, points):
        if points.shape[1] == 2:
            geometry_type = "XY"
        else:
            if points.shape[1] != 3:
                raise WriteError("Need 3D points.")
            geometry_type = "XYZ"

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

    def cells(
        self,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
        grid: ET.Element,
    ) -> None:
        cell_blocks = self._to_cell_blocks(cells)

        if len(cell_blocks) == 1:
            meshio_type = cell_blocks[0].type
            num_cells = len(cell_blocks[0].data)
            xdmf_type = meshio_to_xdmf_type[meshio_type][0]
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType=xdmf_type,
                NumberOfElements=str(num_cells),
            )
            dt, prec = numpy_to_xdmf_dtype[cell_blocks[0].data.dtype.name]
            data_item = ET.SubElement(
                topo,
                "DataItem",
                DataType=dt,
                Dimensions="{} {}".format(*cell_blocks[0].data.shape),
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(cell_blocks[0].data)
        elif len(cell_blocks) > 1:
            total_num_cells = sum(len(c.data) for c in cell_blocks)
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType="Mixed",
                NumberOfElements=str(total_num_cells),
            )
            total_num_cell_items = sum(np.prod(c.data.shape) for c in cell_blocks)
            dim = total_num_cell_items + total_num_cells
            # Lines translate to Polylines, and one needs to specify the exact
            # number of nodes. Hence, prepend 2.
            for c in cell_blocks:
                if c.type == "line":
                    c.data[:] = np.insert(c.data, 0, 2, axis=1)
                    dim += len(c.data)
            dim = str(dim)
            cd = np.concatenate(
                [
                    # prepend column with xdmf type index
                    np.insert(
                        c.data, 0, meshio_type_to_xdmf_index[c.type], axis=1
                    ).flatten()
                    for c in cell_blocks
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

    def point_data(self, point_data: dict[str, np.ndarray], grid: ET.Element):
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

    def cell_data(
        self, cell_data: dict[str, list[np.ndarray]], grid: ET.Element
    ) -> None:
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
