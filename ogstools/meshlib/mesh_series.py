# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""A class to handle Meshseries data."""

from pathlib import Path
from typing import Literal, Optional, Union

import meshio
import numpy as np
import pyvista as pv
import vtuIO
from h5py import File
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from tqdm.auto import tqdm

from ogstools.propertylib import Property

from .xdmf_reader import XDMFReader


class MeshSeries:
    """
    A wrapper around pyvista and meshio for reading of pvd and xdmf timeseries.

    Will be replaced by own module in ogstools with similar interface.
    """

    def __init__(
        self, filepath: Union[str, Path], time_unit: Optional[str] = "s"
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
            self._xdmf_reader = XDMFReader(filepath)
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
    def hdf5(self) -> File:
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
            return np.asarray(
                [self.read(t)[data_name] for t in tqdm(self.timesteps)]
            )
        return mesh[data_name]

    def aggregate(
        self,
        mesh_property: Union[Property, str],
        func: Literal["min", "max", "mean", "median", "sum", "std", "var"],
    ) -> pv.UnstructuredGrid:
        """Aggregate data over all timesteps using a specified function.

        :param mesh_property:
            The mesh property to be aggregated. If given as type `Property`, the
            :meth:`~ogstools.propertylib.property.Property.transform` function
            will be applied on each timestep and aggregation afterwards.
        :param func:
            The aggregation function to apply. It must be one of "min", "max",
            "mean", "median", "sum", "std", "var", where the equally named numpy
            function will be used to aggregate over all timesteps or "min_time"
            or "max_time", which return the timevalue when the limit occurs.
        :returns:   A mesh with aggregated data according to the given function.

        """
        np_func = {
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
            "median": np.median,
            "sum": np.sum,
            "std": np.std,
            "var": np.var,
            "min_time": np.argmin,
            "max_time": np.argmax,
        }[func]
        mesh = self.read(0).copy(deep=True)
        mesh.clear_data()
        if isinstance(mesh_property, Property):
            if mesh_property.mesh_dependent:
                vals = np.asarray(
                    [
                        mesh_property.transform(self.read(t))
                        for t in tqdm(self.timesteps)
                    ]
                )
            else:
                vals = mesh_property.transform(
                    self.values(mesh_property.data_name)
                )
        else:
            vals = self.values(mesh_property)
        output_name = (
            f"{mesh_property.output_name}_{func}"
            if isinstance(mesh_property, Property)
            else f"{mesh_property}_{func}"
        )
        if func in ["min_time", "max_time"]:
            mesh[output_name] = self.timevalues[np_func(vals, axis=0)]
        else:
            mesh[output_name] = np.empty(vals.shape[1:])
            assert isinstance(np_func, type(np.max))
            np_func(vals, out=mesh[output_name], axis=0)
        return mesh

    def _probe_pvd(
        self,
        points: np.ndarray,
        data_name: str,
        interp_method: Optional[Literal["nearest", "probefilter"]] = None,
        interp_backend: Optional[Literal["vtk", "scipy"]] = None,
    ) -> np.ndarray:
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
    ) -> np.ndarray:
        values = self.hdf5["meshes"][self.hdf5_bulk_name][data_name][:]
        geom = self.hdf5["meshes"][self.hdf5_bulk_name]["geometry"][0]
        values = np.swapaxes(values, 0, 1)

        # remove flat dimensions for interpolation
        for index, axis in enumerate(geom.T):
            if np.all(np.isclose(axis, axis[0])):
                geom = np.delete(geom, index, 1)
                points = np.delete(points, index, 1)

        if interp_method is None:
            interp_method = "linear"
        interp = {
            "nearest": NearestNDInterpolator(geom, values),
            "linear": LinearNDInterpolator(geom, values, np.nan),
        }[interp_method]

        return np.swapaxes(interp(points), 0, 1)

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
