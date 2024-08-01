# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""
This file provides an override to meshios XDMF Reader since it misses a feature
to handle hyperslabs (there are two ways to handle hyperslab:
the common documented `here <https://www.xdmf.org/index.php/XDMF_Model_and_Format#HyperSlab>`_
and the way paraview supports it (documentation missing).

Example::

    2D_single_fracture_HT.h5:/meshes/2D_single_fracture/temperature|0 0:1 1:1 190:97 190


to be read like::

    | start : stride : count : end

"""

from xml.etree.ElementTree import Element

import meshio
import numpy as np
from meshio._mesh import CellBlock
from meshio.xdmf import common, time_series
from meshio.xdmf.time_series import (
    ReadError,
    cell_data_from_raw,
    xdmf_to_numpy_type,
)


def _translate_mixed_cells_patched(data: list) -> list:
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (cell_type1, p0, p1, ... ,pk, cell_type2, p10, p11, ..., p1k, ...

    # https://xdmf.org/index.php/XDMF_Model_and_Format#Arbitrary
    # https://gitlab.kitware.com/xdmf/xdmf/blob/master/XdmfTopologyType.hpp#L394
    xdmf_idx_to_num_nodes = {
        1: 1,  # POLYVERTEX
        2: 2,  # POLYLINE
        4: 3,  # TRIANGLE
        5: 4,  # QUADRILATERAL
        6: 4,  # TETRAHEDRON
        7: 5,  # PYRAMID
        8: 6,  # WEDGE
        9: 8,  # HEXAHEDRON
        35: 9,  # QUADRILATERAL_9
        36: 6,  # TRIANGLE_6
        37: 8,  # QUADRILATERAL_8
        38: 10,  # TETRAHEDRON_10
        39: 13,  # PYRAMID_13
        40: 15,  # WEDGE_15
        41: 18,  # WEDGE_18
        48: 20,  # HEXAHEDRON_20
        49: 24,  # HEXAHEDRON_24
        50: 27,  # HEXAHEDRON_27
    }

    # collect types and offsets
    types = []
    _offsets = []
    r = 0
    while r < len(data):
        xdmf_type = data[r]
        types.append(xdmf_type)
        _offsets.append(r)
        if xdmf_type == 2:  # line
            if data[r + 1] != 2:  # polyline
                msg = "XDMF reader: Only supports 2-point lines for now"
                raise ReadError(msg)
            r += 1
        r += 1
        r += xdmf_idx_to_num_nodes[xdmf_type]

    types = np.asarray(types)
    offsets = np.asarray(_offsets)

    b = np.concatenate(
        [[0], np.where(types[:-1] != types[1:])[0] + 1, [len(types)]]
    )
    cells = []
    for start, end in zip(b[:-1], b[1:], strict=False):  # noqa: RUF007
        meshio_type = common.xdmf_idx_to_meshio_type[types[start]]
        n = xdmf_idx_to_num_nodes[types[start]]
        point_offsets = offsets[start:end] + (2 if types[start] == 2 else 1)
        indices = np.asarray([np.arange(n) + o for o in point_offsets])
        cells.append(CellBlock(meshio_type, data[indices]))

    return cells


time_series.translate_mixed_cells = _translate_mixed_cells_patched


class XDMFReader(meshio.xdmf.TimeSeriesReader):
    def __init__(self, filename: str):
        super().__init__(filename)

    def read_data(self, k: int) -> tuple[float, dict, dict, dict]:
        point_data = {}
        cell_data_raw = {}
        other_data = {}

        t = None

        for c in list(self.collection[k]):
            if c.tag == "Time":
                t = float(c.attrib["Value"])
            elif c.tag == "Attribute":
                name = c.get("Name")

                if len(list(c)) != 1:
                    raise ReadError()
                data_item = next(iter(c))
                data = self._read_data_item(data_item)

                if c.get("Center") == "Node":
                    point_data[name] = data
                elif c.get("Center") == "Cell":
                    cell_data_raw[name] = data
                elif c.get("Center") == "Other":
                    other_data[name] = data
                else:
                    raise ReadError()

            else:
                # skip the xi:included mesh
                continue

        if self.cells is None:
            raise ReadError()
        cell_data = cell_data_from_raw(self.cells, cell_data_raw)
        if t is None:
            raise ReadError()

        return t, point_data, cell_data, other_data

    def _read_data_item(self, data_item: Element) -> np.ndarray:
        dims = [int(d) for d in data_item.get("Dimensions", "").split()]

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

        assert isinstance(data_item.text, str)
        if data_format == "XML":
            return np.fromstring(
                data_item.text,
                dtype=xdmf_to_numpy_type[(data_type, precision)],
                sep=" ",
            ).reshape(dims)
        if data_format == "Binary":
            return np.fromfile(
                data_item.text.strip(),
                dtype=xdmf_to_numpy_type[(data_type, precision)],
            ).reshape(dims)

        if data_format != "HDF":
            msg = f"Unknown XDMF Format '{data_format}'."
            raise ReadError(msg)

        file_info = data_item.text.strip()
        file_h5path__selections = file_info.split("|")
        file_h5path = file_h5path__selections[0]
        selections = (
            file_h5path__selections[1]
            if len(file_h5path__selections) > 1
            else None
        )
        filename, h5path = file_h5path.split(":")
        if selections:
            # offsets, slices, current_data_extends, global_data_extends by dimension
            m = [
                list(map(int, att.split(" "))) for att in selections.split(":")
            ]
            t = np.transpose(m)
            selection = tuple(
                slice(start, start + extend, step)
                for start, step, extend, _ in t
            )
        else:
            selection = ()

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
        return f[selection].squeeze()
