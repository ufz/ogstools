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


from abc import ABC, abstractmethod
from pathlib import Path
from xml.etree.ElementTree import Element

import h5py
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


class DataItem(ABC):
    """
    Abstract base class for all classes that end with DataItem.
    """

    rawdata_path: Path

    @abstractmethod
    def __getitem__(self, args: tuple | int | slice | np.ndarray) -> np.ndarray:
        pass


class H5DataItem(DataItem):
    """
    A class to handle the data item in the xdmf file that references to a h5 file.
    With init only the xdmf meta data is read. (light computation)
    With selected_values the requested values are read from h5 file (heavy computation)
    """

    def __init__(self, file_info: str, xdmf_path: Path):
        """
        :param file_info: The file_info string from the XDMF file
            example: 2D_single_fracture_HT.h5:/meshes/2D_single_fracture/geometry|0 0 0:1 1 1:1 190 3:97 190 3
        :param xdmf_path: Path to the xdmf file that references to the h5 file
        """

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

        self.selection = selection
        self.h5path = h5path

        self.key = h5path[1:].split("/")[-1]

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        dirpath = xdmf_path.resolve().parent
        self.rawdata_path = dirpath / filename

    def __getitem__(self, args: tuple | int | slice | np.ndarray) -> np.ndarray:
        """
        Reads value from HDF5 file based on given selection.

        param args: See numpy array indexing https://numpy.org/doc/stable/user/basics.indexing.html#
        :returns:   A numpy array (sliced) of the requested data for all timesteps

        """
        with h5py.File(self.rawdata_path, "r") as file:
            f = file
            if self.h5path[0] != "/":
                raise ReadError()

            for key in self.h5path[1:].split("/"):
                f = f[key]
            return f[args]

    def selected_values(self) -> np.ndarray:
        """
        Returns all values of the DataItem that are selected in the xdmf file.
        An empty selection means all values are read.

        In the xdmf file with <DataItem> tag you optionally find a string containing
        the 1.h5filename, 2. name of h5 group, 3. selection
        e.g. 2D_single_fracture_HT.h5:/meshes/2D_single_fracture/geometry|0 0 0:1 1 1:1 190 3:97 190 3
        The selection here is:
        0 0 0:1 1 1:1 190 3:97 190 3
        The meaning is: [(offset(0,0,0): step(1,1,1) : end(1,190,3) : of_data_with_size(97,190,30))]


        :returns:   A numpy array (sliced) of the requested data for all timesteps
        """
        # should always be just one timestep
        return self[self.selection].squeeze()


class XMLDataItem(DataItem):
    def __init__(
        self, name: str, dims: list[int], data_type: str | None, precision: str
    ):
        self.key = name
        self._data_type = data_type
        self._precision = precision
        self._dims = dims

    def __getitem__(self, args: tuple | int | slice | np.ndarray) -> np.ndarray:
        return np.fromstring(
            self.key,
            dtype=xdmf_to_numpy_type[(self._data_type, self._precision)],
            sep=" ",
        ).reshape(self._dims)


class BinaryDataItem(DataItem):
    # BinaryDataItem(                data_item.text.strip(), dims, data_type, precision            )
    def __init__(
        self, name: str, dims: list[int], data_type: str | None, precision: str
    ):
        self.key = name
        self._data_type = data_type
        self._precision = precision
        self._dims = dims

    def __getitem__(self, args: tuple | int | slice | np.ndarray) -> np.ndarray:
        return np.fromfile(
            self.key,
            dtype=xdmf_to_numpy_type[(self._data_type, self._precision)],
        ).reshape(self._dims)


class DataItems:
    def __init__(self, items: list[DataItem], center: str):
        self.items = items
        self.center = center

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files out there
        # use both keys interchangeably.

        all_in_h5 = np.all(
            [isinstance(item, H5DataItem) for item in self.items]
        )
        if not all_in_h5:
            self.fast_access = False
            return

        all_in_same_file = (
            # can ignore lint because isinstance is checked
            len(np.unique([item.h5path for item in self.items]))  # type: ignore[attr-defined]
            == 1
        )
        if not all_in_same_file:
            self.fast_access = False
            return

        self.fast_access = all_in_h5 and all_in_same_file

    def __getitem__(self, args: tuple | slice | int) -> np.ndarray:
        key = args if isinstance(args, tuple) else (args,)

        if not self.fast_access:
            all_time_steps = self.items[key[0]]
            if not isinstance(all_time_steps, list):
                return self.items[key[0]][key[1:]]
            arrays = [item[key[1:]] for item in all_time_steps]
            return np.stack(arrays)
        # If all items are stored within same h5 file, take info from 1st time step
        return self.items[0][key]


class XDMFReader(meshio.xdmf.TimeSeriesReader):
    def __init__(self, filename: str):
        super().__init__(filename)

        ### extension for indexing
        self.filename: Path = Path(self.filename)
        data_items: dict[str, list[DataItem]] = {}

        self.data_items: dict[str, DataItems] = {}
        data_attribute: dict[str, str] = {}

        self.t = None
        for grid in self.collection:
            for item in grid:
                if item.tag == "Time":
                    self.t = float(item.attrib["Value"])
                elif item.tag == "Attribute":
                    name = item.get("Name")
                    if len(list(item)) != 1:
                        raise ReadError()
                    data_item = next(iter(item))
                    if item.get("Center") not in [
                        "Node",
                        "Cell",
                        "Other",
                    ]:
                        raise ReadError()
                    center = item.get("Center")
                    data = self.select_item(data_item)
                    if name in data_items:
                        data_items[name].append(data)
                    else:
                        data_items[name] = [data]
                        data_attribute[name] = center

        for key, value in data_items.items():
            self.data_items[key] = DataItems(value, data_attribute[key])

    def has_fast_access(self, key: str | None = None) -> bool:
        if len(self.data_items) == 0:
            return False  # if there is no data, there is no fast access

        if key is None:
            all_fast = {
                key: item.fast_access for key, item in self.data_items.items()
            }
            return all(all_fast.values())

        key = next(iter(self.data_items))  # checked for len > 0
        return self.data_items[key].fast_access

    def rawdata_path(self, key: str | None = None) -> Path:
        # This function should usually work for OGS Simulation result in XDMF [single hdf5 file]
        # To be used in combination with h5py to read/save/manipulate of the hdf5file

        if self.has_fast_access(key):
            if key is None:
                key = next(
                    iter(self.data_items)
                )  # checked for len > 0 in has_fast_access
            return self.data_items[key].items[0].rawdata_path
        return self.filename

    def read_data(self, k: int) -> tuple[float, dict, dict, dict]:
        point_data = {}
        cell_data_raw: dict = {}
        other_data = {}
        t = None
        cell_data = cell_data_from_raw(self.cells, cell_data_raw)

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

    def select_item(self, data_item: Element) -> np.ndarray:
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
            return XMLDataItem(data_item.text, dims, data_type, precision)
        if data_format == "Binary":
            return BinaryDataItem(
                data_item.text.strip(), dims, data_type, precision
            )
        if data_format == "HDF":
            return H5DataItem(
                file_info=data_item.text.strip(), xdmf_path=self.filename
            )

        msg = f"Unknown XDMF Format '{data_format}'."
        raise ReadError(msg)

    # Copy of _read_data_item of meshio with fix for slices
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

        with h5py.File(full_hdf5_path, "r") as file:
            if h5path[0] != "/":
                raise ReadError()

            f = file
            for key in h5path[1:].split("/"):
                f = f[key]
            # `[()]` gives a np.ndarray
            selected_shape = f[selection]

            try:
                data = np.reshape(selected_shape, dims)  # may be copy or view
            except ValueError as e:
                msg = f"Error reshaping data: {e}. Shape: {selected_shape.shape}, dims: {dims}, DataItem: {data_item.text}."
                raise ValueError(msg) from e
            assert data.base is not None  # the xdmf dim should be correct
            return np.copy(data)
