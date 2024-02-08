"""A class to handle Meshseries data."""

from pathlib import Path
from typing import Literal, Optional, Union

import meshio
import numpy as np
import pyvista as pv
import vtuIO
from meshio.xdmf.time_series import (
    ReadError,
    cell_data_from_raw,
    xdmf_to_numpy_type,
)
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from tqdm.auto import tqdm


class TimeSeriesReader(meshio.xdmf.TimeSeriesReader):
    def __init__(self, filename):
        super().__init__(filename)

    def read_data(self, k: int):
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


class MeshSeries:
    """
    A wrapper around pyvista and meshio for reading of pvd and xdmf timeseries.

    Will be replaced by own module in ogstools with similar interface.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        time_unit: Optional[Optional[str]] = None,
    ) -> None:
        """
        Initialize a MeshSeries object

            :param filepath:    Path to the PVD or XDMF file.
            :param time_unit:   Data unit of the timevalues.

            :returns:           A MeshSeries object
        """
        if isinstance(filepath, Path):
            filepath = str(filepath)
        self.filepath = filepath
        self.time_unit = time_unit
        self._data: dict[int, pv.UnstructuredGrid] = {}
        self._data_type = filepath.split(".")[-1]
        if self._data_type == "pvd":
            self._pvd_reader = pv.PVDReader(filepath)
        elif self._data_type == "xdmf":
            self._xdmf_reader = TimeSeriesReader(filepath)
            self._read_xdmf(0)  # necessary to initialize hdf5_files
            meshes = self.hdf5["meshes"]
            self.hdf5_bulk_name = list(meshes.keys())[
                np.argmax([meshes[m]["geometry"].shape[1] for m in meshes])
            ]
        elif self._data_type == "vtu":
            self._vtu_reader = pv.XMLUnstructuredGridReader(filepath)
        else:
            msg = "Can only read 'pvd', 'xdmf' or 'vtu' files."
            raise TypeError(msg)

    @property
    def hdf5(self):
        # We assume there is only one h5 file
        return next(iter(self._xdmf_reader.hdf5_files.values()))

    def _read_pvd(self, timestep: int) -> pv.UnstructuredGrid:
        self._pvd_reader.set_active_time_point(timestep)
        return self._pvd_reader.read()[0]

    def _read_xdmf(self, timestep: int) -> pv.UnstructuredGrid:
        points, cells = self._xdmf_reader.read_points_cells()
        _, point_data, cell_data, field_data = self._xdmf_reader.read_data(
            timestep
        )
        meshio_mesh = meshio.Mesh(points, cells, point_data, cell_data)
        return pv.from_meshio(meshio_mesh)

    def read(
        self, timestep: int, lazy_eval: bool = True
    ) -> pv.UnstructuredGrid:
        """Lazy read function."""
        if timestep in self._data:
            return self._data[timestep]
        if self._data_type == "pvd":
            mesh = self._read_pvd(timestep)
        elif self._data_type == "xdmf":
            mesh = self._read_xdmf(timestep)
        elif self._data_type == "vtu":
            mesh = self._vtu_reader.read()
        if lazy_eval:
            self._data[timestep] = mesh
        return mesh

    def clear(self) -> None:
        self._data.clear()

    @property
    def timesteps(self) -> range:
        """Return the timesteps of the timeseries data."""
        if self._data_type == "vtu":
            return range(0)
        if self._data_type == "pvd":
            return range(self._pvd_reader.number_time_points)
        # elif self._data_type == "xdmf":
        return range(len(self.timevalues))

    @property
    def timevalues(self) -> np.ndarray:
        """Return the timevalues of the timeseries data."""
        if self._data_type == "vtu":
            return np.zeros(1)
        if self._data_type == "pvd":
            return np.asarray(self._pvd_reader.time_values)
        # elif self._data_type == "xdmf":
        time_values = []
        for collection_i in self._xdmf_reader.collection:
            for element in collection_i:
                if element.tag == "Time":
                    time_values += [float(element.attrib["Value"])]
        return np.asarray(time_values)

    def closest_timestep(self, timevalue: float) -> int:
        """Return the corresponding timestep from a timevalue."""
        return int(np.argmin(np.abs(self.timevalues - timevalue)))

    def closest_timevalue(self, timevalue: float) -> float:
        """Return the closest timevalue to a timevalue."""
        return self.timevalues[self.closest_timestep(timevalue)]

    def read_closest(self, timevalue: float) -> pv.UnstructuredGrid:
        """Return the closest timestep in the data for a given timevalue."""
        return self.read(self.closest_timestep(timevalue))

    def read_interp(
        self, timevalue: float, lazy_eval: bool = True
    ) -> pv.UnstructuredGrid:
        """Return the temporal interpolated mesh for a given timevalue."""
        t_vals = self.timevalues
        ts1 = int(t_vals.searchsorted(timevalue, "right") - 1)
        ts2 = min(ts1 + 1, len(t_vals) - 1)
        if np.isclose(timevalue, t_vals[ts1]):
            return self.read(ts1, lazy_eval)
        mesh1 = self.read(ts1, lazy_eval)
        mesh2 = self.read(ts2, lazy_eval)
        mesh = mesh1.copy(deep=True)
        for key in mesh1.point_data:
            if np.all(mesh1.point_data[key] == mesh2.point_data[key]):
                continue
            dt = t_vals[ts2] - t_vals[ts1]
            slope = (mesh2.point_data[key] - mesh1.point_data[key]) / dt
            mesh.point_data[key] = mesh1.point_data[key] + slope * (
                timevalue - t_vals[ts1]
            )
        return mesh

    def values(self, data_name: str) -> np.ndarray:
        """
        Get the data in the MeshSeries for all timesteps.

        :param data_name:   Name of the data in the MeshSeries.

        :returns:   A numpy array of the requested data for all timesteps
        """
        mesh = self.read(0).copy()
        if self._data_type == "xdmf":
            return self.hdf5["meshes"][self.hdf5_bulk_name][data_name]
        if self._data_type == "pvd":
            return [self.read(t)[data_name] for t in tqdm(self.timesteps)]
        return mesh[data_name]

    def _probe_pvd(
        self,
        points: np.ndarray,
        data_name: str,
        interp_method: Optional[Literal["nearest", "probefilter"]] = None,
        interp_backend: Optional[Literal["vtk", "scipy"]] = None,
    ):
        obs_pts_dict = {f"pt{j}": point for j, point in enumerate(points)}
        dim = self.read(0).get_cell(0).dimension
        pvd_path = self.filepath
        pvdio = vtuIO.PVDIO(
            pvd_path, dim=dim, interpolation_backend=interp_backend
        )
        values_dict = pvdio.read_time_series(
            data_name, obs_pts_dict, interpolation_method=interp_method
        )
        return np.asarray(list(values_dict.values()))

    def _probe_xdmf(
        self,
        points: np.ndarray,
        data_name: str,
        interp_method: Optional[Literal["nearest", "linear"]] = None,
    ):
        values = self.hdf5["meshes"][self.hdf5_bulk_name][data_name][:]
        geom = self.hdf5["meshes"][self.hdf5_bulk_name]["geometry"][0]

        # remove flat dimensions for interpolation
        for index, axis in enumerate(geom.T):
            if np.all(np.isclose(axis, axis[0])):
                geom = np.delete(geom, index, 1)
                points = np.delete(points, index, 1)

        if interp_method is None:
            interp_method = "linear"
        interp = {
            "nearest": NearestNDInterpolator(geom, values.T),
            "linear": LinearNDInterpolator(geom, values.T, np.nan),
        }[interp_method]

        return interp(points).T

    def probe(
        self,
        points: np.ndarray,
        data_name: str,
        interp_method: Optional[
            Literal["nearest", "linear", "probefilter"]
        ] = None,
        interp_backend_pvd: Optional[Literal["vtk", "scipy"]] = None,
    ) -> np.ndarray:
        """
        Probe the MeshSeries at observation points.

        :param points:          The points to sample at.
        :param data_name:       Name of the data to sample.
        :param interp_method:   Choose the interpolation method, defaults to
                                `linear` for xdmf MeshSeries and `probefilter`
                                for pvd MeshSeries.
        :param interp_backend:  Interpolation backend for PVD MeshSeries.

        :returns:   `numpy` array of interpolated data at observation points.
        """
        if self._data_type == "xdmf":
            assert interp_method != "probefilter"
            return self._probe_xdmf(points, data_name, interp_method)
        assert self._data_type == "pvd"
        assert interp_method != "linear"
        return self._probe_pvd(
            points, data_name, interp_method, interp_backend_pvd
        )
