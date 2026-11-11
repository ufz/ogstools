# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Write MeshSeries to XDMF + single HDF5 file (OGS output format)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pyvista as pv

from ogstools.meshseries.xdmf_reader import _split_hdf5_reference

if TYPE_CHECKING:
    from ogstools.meshseries._meshseries import MeshSeries
    from ogstools.meshseries.xdmf_reader import XDMFReader

# A VTK line is written as an XDMF polyline: [_XDMF_POLYLINE, _LINE_NODES, p0, p1]
_XDMF_POLYLINE = 2
_LINE_NODES = 2

# VTK cell type → XDMF Mixed topology type code
_VTK_TO_XDMF: dict[int, int] = {
    pv.CellType.VERTEX: 1,
    pv.CellType.LINE: _XDMF_POLYLINE,
    pv.CellType.TRIANGLE: 4,
    pv.CellType.QUAD: 5,
    pv.CellType.TETRA: 6,
    pv.CellType.PYRAMID: 7,
    pv.CellType.WEDGE: 8,
    pv.CellType.HEXAHEDRON: 9,
    pv.CellType.QUADRATIC_TRIANGLE: 36,
    pv.CellType.QUADRATIC_QUAD: 37,
    pv.CellType.QUADRATIC_TETRA: 38,
    pv.CellType.QUADRATIC_PYRAMID: 39,
    pv.CellType.QUADRATIC_WEDGE: 40,
    pv.CellType.QUADRATIC_HEXAHEDRON: 48,
}

_XI = "http://www.w3.org/2001/XInclude"
ET.register_namespace("xi", _XI)

# XPointers from a later timestep to the Geometry and Topology of the first:
# Xdmf / Domain / temporal Grid / first Grid / child 2 and 3 (child 1 is Time)
_FIRST_GEOMETRY = "element(/1/1/1/1/2)"
_FIRST_TOPOLOGY = "element(/1/1/1/1/3)"

# numpy dtype.type → (XDMF DataType string, Precision string)
_DTYPE_TO_XDMF: dict[type, tuple[str, str]] = {
    np.float32: ("Float", "4"),
    np.float64: ("Float", "8"),
    np.int8: ("Int", "1"),
    np.int16: ("Int", "2"),
    np.int32: ("Int", "4"),
    np.int64: ("Int", "8"),
    np.uint8: ("UInt", "1"),
    np.uint16: ("UInt", "2"),
    np.uint32: ("UInt", "4"),
    np.uint64: ("UInt", "8"),
    np.bool_: ("UInt", "1"),
}


def _encode_topology(mesh: pv.DataSet) -> np.ndarray:
    """Convert pyvista topology to XDMF Mixed flat int32 array.

    VTK LINE (type 3) encodes as [2, 2, p0, p1] (polyline with 2 nodes).
    All other types encode as [xdmf_type, p0, p1, ..., pk].

    :param mesh: The mesh whose cells are encoded.
    :returns:    The XDMF Mixed topology.
    """
    if isinstance(mesh, pv.PolyData):
        mesh = mesh.cast_to_unstructured_grid()
    connectivity = mesh.cell_connectivity  # node indices only, no size prefixes
    offsets = mesh.offset  # (n_cells + 1,) cumulative into connectivity
    celltypes = mesh.celltypes  # (n_cells,)
    n_cells = mesh.n_cells
    n_pts = np.diff(offsets)  # nodes per cell
    is_line = celltypes == pv.CellType.LINE

    # Output entries per cell: LINE → 4; others → 1 + n_pts
    out_sizes = np.where(is_line, 2 + _LINE_NODES, 1 + n_pts)
    out_offsets = np.concatenate([[0], np.cumsum(out_sizes)])
    xdmf_topo = np.empty(int(out_offsets[-1]), dtype=np.int32)

    # Fast path: uniform non-LINE mesh (covers most OGS outputs)
    if not is_line.any() and n_cells > 0 and np.all(n_pts == n_pts[0]):
        nodes_per_cell = int(n_pts[0])
        view = xdmf_topo.reshape(n_cells, 1 + nodes_per_cell)
        view[:, 0] = np.fromiter(
            (_VTK_TO_XDMF[int(t)] for t in celltypes),
            dtype=np.int32,
            count=n_cells,
        )
        view[:, 1:] = connectivity.reshape(n_cells, nodes_per_cell)
        return xdmf_topo

    # General path: mixed types or LINE cells
    for i in range(n_cells):
        out = int(out_offsets[i])
        conn = int(offsets[i])
        if is_line[i]:
            xdmf_topo[out] = _XDMF_POLYLINE
            xdmf_topo[out + 1] = _LINE_NODES
            xdmf_topo[out + 2] = connectivity[conn]
            xdmf_topo[out + 3] = connectivity[conn + 1]
        else:
            xdmf_topo[out] = _VTK_TO_XDMF[int(celltypes[i])]
            xdmf_topo[out + 1 : int(out_offsets[i + 1])] = connectivity[
                conn : int(offsets[i + 1])
            ]
    return xdmf_topo


def _hyperslab_ref(
    h5_path_str: str, ts: int, n_ts: int, shape: tuple[int, ...]
) -> str:
    """Build OGS-style HDF5 reference with hyperslab notation for one timestep.

    Format: `h5file.h5:/path/to/dataset|start:step:count:total`
    where each column is space-separated per dimension.
    The leading dimension is the timestep axis; shape gives the remaining dims.

    :param h5_path_str: `file.h5:/path/to/dataset`.
    :param ts:          The index of the time step.
    :param n_ts:        The number of time steps in the dataset.
    :param shape:       The shape of one time step.
    :returns:           The reference with its hyperslab selection.
    """
    ndim = 1 + len(shape)
    start = " ".join([str(ts)] + ["0"] * len(shape))
    step = " ".join(["1"] * ndim)
    count = " ".join(["1"] + [str(d) for d in shape])
    total = " ".join([str(n_ts)] + [str(d) for d in shape])
    return f"{h5_path_str}|{start}:{step}:{count}:{total}"


def _add_attribute(
    parent: ET.Element,
    name: str,
    center: str,
    dims_str: str,
    data_type: str,
    precision: str,
    h5_ref: str,
) -> None:
    """Append an Attribute with one HDF DataItem to `parent`.

    :param parent:    The Grid element.
    :param name:      The name of the data array.
    :param center:    `Node`, `Cell` or `Other`.
    :param dims_str:  The dimensions of one time step, space separated.
    :param data_type: The XDMF DataType, e.g. `Float`.
    :param precision: The XDMF Precision in bytes, e.g. `8`.
    :param h5_ref:    The HDF5 reference, from `_hyperslab_ref`.
    """
    attr = ET.SubElement(parent, "Attribute")
    attr.attrib.update(
        {
            "Center": center,
            "ElementCell": "",
            "ElementDegree": "0",
            "ElementFamily": "",
            "ItemType": "",
            "Name": name,
            "Type": "None",
        }
    )
    data_item = ET.SubElement(attr, "DataItem")
    data_item.attrib.update(
        {
            "DataType": data_type,
            "Dimensions": dims_str,
            "Format": "HDF",
            "Precision": precision,
        }
    )
    data_item.text = h5_ref


def _new_collection() -> tuple[ET.Element, ET.Element]:
    """Return a new Xdmf root and its temporal grid collection.

    :returns: The Xdmf element and its temporal collection Grid.
    """
    xdmf = ET.Element("Xdmf", {"Version": "3.0"})
    domain = ET.SubElement(xdmf, "Domain")
    collection = ET.SubElement(
        domain,
        "Grid",
        {
            "CollectionType": "Temporal",
            "GridType": "Collection",
            "Name": "Collection",
        },
    )
    return xdmf, collection


def _write_xml(xdmf: ET.Element, xdmf_path: Path) -> None:
    """Write `xdmf` indented to `xdmf_path`.

    :param xdmf:      The Xdmf root element.
    :param xdmf_path: The file to write.
    """
    ET.indent(xdmf)
    ET.ElementTree(xdmf).write(
        xdmf_path, encoding="utf-8", xml_declaration=True
    )


def _write_xdmf_xml(
    xdmf_path: Path,
    h5_name: str,
    h5_group: str,
    timevalues: np.ndarray,
    n_nodes: int,
    topo_size: int,
    var_meta: list[dict],
) -> None:
    """Write the XDMF XML file.

    :param xdmf_path:  The XDMF file to write.
    :param h5_name:    HDF5 file name (or relative/absolute path) used in
                       DataItem text references, e.g. `"out.h5"`.
    :param h5_group:   HDF5 group path, e.g. `"/meshes/out"`.
    :param timevalues: The time values, one grid each.
    :param n_nodes:    The number of points of the mesh.
    :param topo_size:  The length of the encoded topology.
    :param var_meta:   The metadata of the data arrays, from
                       `_collect_var_meta`.
    """
    n_ts = len(timevalues)
    xdmf, collection = _new_collection()

    geom_dims = f"{n_nodes} 3"
    geom_h5 = f"{h5_name}:{h5_group}/geometry"
    topo_h5 = f"{h5_name}:{h5_group}/topology"

    for ts, tv in enumerate(timevalues):
        grid = ET.SubElement(
            collection, "Grid", {"Name": "Grid", "GridType": "Uniform"}
        )
        ET.SubElement(grid, "Time", {"Value": repr(float(tv))})

        if ts == 0:
            geom = ET.SubElement(
                grid, "Geometry", {"Origin": "", "Type": "XYZ"}
            )
            data_item = ET.SubElement(
                geom,
                "DataItem",
                {
                    "DataType": "Float",
                    "Dimensions": geom_dims,
                    "Format": "HDF",
                    "Precision": "8",
                },
            )
            data_item.text = _hyperslab_ref(geom_h5, 0, 1, (n_nodes, 3))

            topo_el = ET.SubElement(
                grid,
                "Topology",
                {"Dimensions": str(topo_size), "Type": "Mixed"},
            )
            data_item = ET.SubElement(
                topo_el,
                "DataItem",
                {
                    "DataType": "Int",
                    "Dimensions": str(topo_size),
                    "Format": "HDF",
                    "Precision": "4",
                },
            )
            data_item.text = _hyperslab_ref(topo_h5, 0, 1, (topo_size,))
        else:
            ET.SubElement(
                grid, f"{{{_XI}}}include", {"xpointer": _FIRST_GEOMETRY}
            )
            ET.SubElement(
                grid, f"{{{_XI}}}include", {"xpointer": _FIRST_TOPOLOGY}
            )

        for var in var_meta:
            h5_ref = _hyperslab_ref(
                f"{h5_name}:{h5_group}/{var['name']}", ts, n_ts, var["shape"]
            )
            dims_str = " ".join(str(d) for d in var["shape"])
            _add_attribute(
                grid,
                var["name"],
                var["center"],
                dims_str,
                var["xtype"],
                var["prec"],
                h5_ref,
            )

    _write_xml(xdmf, xdmf_path)


def _collect_var_meta(mesh0: pv.UnstructuredGrid) -> list[dict]:
    """Build the metadata of the point, cell and field data of the first mesh.

    :param mesh0: The mesh of the first time step.
    :returns:     One dict per data array with its name, center, shape,
                  dtype and XDMF type and precision.
    """
    var_meta: list[dict] = []
    for center, data in [
        ("Node", mesh0.point_data),
        ("Cell", mesh0.cell_data),
        ("Other", mesh0.field_data),
    ]:
        for key in data:
            arr = np.asarray(data[key])
            if arr.dtype.type not in _DTYPE_TO_XDMF:
                msg = (
                    f"XDMF cannot store the data array '{key}' of type "
                    f"{arr.dtype}. Convert it to numbers or delete it, or "
                    "save as .pvd."
                )
                raise ValueError(msg)
            xtype, prec = _DTYPE_TO_XDMF[arr.dtype.type]
            var_meta.append(
                {
                    "name": key,
                    "center": center,
                    "shape": arr.shape,
                    "dtype": arr.dtype,
                    "xtype": xtype,
                    "prec": prec,
                }
            )
    return var_meta


def save_xdmf(
    ms: MeshSeries,
    path: Path | str,
) -> list[Path]:
    """Save MeshSeries to XDMF + single HDF5 file matching OGS output format.

    Produces `{stem}.xdmf` and `{stem}.h5` in the same directory.
    As in OGS output, the points, cells and array names of the first time
    step hold for all time steps.

    :param ms:        MeshSeries to save.
    :param path:      Output path. Its suffix is replaced by `.xdmf`.
    :returns:         `[xdmf_path, h5_path]`
    """
    path = Path(path)
    xdmf_path = path.with_suffix(".xdmf")
    h5_path = path.with_suffix(".h5")
    mesh_name = xdmf_path.stem

    xdmf_path.parent.mkdir(parents=True, exist_ok=True)

    mesh0 = ms.mesh(0)
    n_nodes = mesh0.n_points
    xdmf_topo = _encode_topology(mesh0)
    topo_size = len(xdmf_topo)
    # Always XYZ, as OGS writes it: a 2D mesh may lie in the X-Z plane
    geometry = mesh0.points.astype(np.float64)  # (n_nodes, 3)
    n_ts = len(ms.timevalues)
    timevalues = ms.timevalues
    h5_group = f"/meshes/{mesh_name}"
    var_meta = _collect_var_meta(mesh0)

    with h5py.File(h5_path, "w") as h5f:
        grp = h5f.require_group(h5_group)
        # Static datasets stored with a leading size-1 timestep dimension
        # shape: 1 x n_nodes x 3
        grp.create_dataset("geometry", data=geometry[np.newaxis])
        # shape: 1 x topo_size
        grp.create_dataset("topology", data=xdmf_topo[np.newaxis])
        for var in var_meta:
            grp.create_dataset(
                var["name"],
                shape=(n_ts,) + var["shape"],
                dtype=var["dtype"],
                chunks=True,
            )
        for ts in range(n_ts):
            mesh = ms.mesh(ts)
            data = {
                "Node": mesh.point_data,
                "Cell": mesh.cell_data,
                "Other": mesh.field_data,
            }
            for var in var_meta:
                grp[var["name"]][ts] = data[var["center"]][var["name"]]

    _write_xdmf_xml(
        xdmf_path,
        h5_name=h5_path.name,
        h5_group=h5_group,
        timevalues=timevalues,
        n_nodes=n_nodes,
        topo_size=topo_size,
        var_meta=var_meta,
    )
    return [xdmf_path, h5_path]


def _hdf5_items(element: ET.Element) -> Iterator[ET.Element]:
    """Yield the DataItems below `element` that reference an HDF5 file.

    :param element: The element to search.
    :returns:       The HDF DataItems that have a text.
    """
    for data_item in element.iter("DataItem"):
        if data_item.get("Format") == "HDF" and data_item.text is not None:
            yield data_item


def source_h5_files(reader: XDMFReader) -> set[Path]:
    """Return the HDF5 files that the XDMF file of `reader` references.

    :param reader: Reader of the source XDMF file.
    :returns:      The resolved paths of the HDF5 files.
    """
    return {
        (
            reader.filename.parent / _split_hdf5_reference(item.text or "")[0]
        ).resolve()
        for item in _hdf5_items(reader.domain)
    }


def _relink_hdf5(
    element: ET.Element, src_dir: Path, xdmf_dir: Path
) -> set[Path]:
    """Point the HDF5 references below `element` from `src_dir` to `xdmf_dir`.

    :param element:  Copy of an element of the source XDMF file.
    :param src_dir:  Folder of the source XDMF file.
    :param xdmf_dir: Folder of the XDMF file that gets `element`.
    :returns:        The resolved HDF5 files that `element` references.
    """
    h5_files = set()
    for data_item in _hdf5_items(element):
        file, path, selection = _split_hdf5_reference(
            (data_item.text or "").strip()
        )
        h5 = (src_dir / file).resolve()
        h5_files.add(h5)
        try:
            # relative, so the XDMF and HDF5 files can be moved together
            ref = Path(os.path.relpath(h5, xdmf_dir.resolve()))
        except ValueError:  # on another Windows drive
            ref = h5
        data_item.text = f"{ref}:{path}" + (
            f"|{selection}" if selection else ""
        )
    return h5_files


def save_xdmf_shallow(ms: MeshSeries, path: Path | str) -> list[Path]:
    """Write XDMF XML only, referencing the HDF5 data of the source XDMF file.

    Used for `deep=False` saves of a series read from XDMF.
    The grids of the time steps in the series are copied from the source XDMF,
    so any HDF5 layout that it reads is kept.
    Changes to the data in memory are not saved.

    :param ms:   MeshSeries read from an XDMF file with fast access.
    :param path: Output path for the `.xdmf` file.
    :returns:    The `.xdmf` file and the HDF5 files that it references.
    """
    xdmf_path = Path(path).with_suffix(".xdmf")
    xdmf_path.parent.mkdir(parents=True, exist_ok=True)
    reader = ms._xdmf_reader
    src_dir = reader.filename.parent
    h5_files: set[Path] = set()

    xdmf, collection = _new_collection()
    for ts, tv in zip(ms._xdmf_time_indices(), ms.timevalues, strict=True):
        source_grid = reader.collection[ts]
        # meshio leaves GridType out on the time steps, the reader needs it
        grid = ET.SubElement(
            collection, "Grid", {"GridType": "Uniform", **source_grid.attrib}
        )
        for child in source_grid:
            # inline xi:include, its target may be a grid that is left out
            sources = (
                reader._resolve_xinclude(child)
                if "XInclude" in child.tag
                else [child]
            )
            for source in sources:
                item = deepcopy(source)
                if item.tag == "Time":
                    item.set("Value", repr(float(tv)))
                h5_files |= _relink_hdf5(item, src_dir, xdmf_path.parent)
                grid.append(item)

    _write_xml(xdmf, xdmf_path)
    return [xdmf_path, *sorted(h5_files)]
