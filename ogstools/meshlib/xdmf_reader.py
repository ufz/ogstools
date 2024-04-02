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
from meshio.xdmf.time_series import (
    ReadError,
    cell_data_from_raw,
    xdmf_to_numpy_type,
)


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
                data_item = list(c)[0]
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
