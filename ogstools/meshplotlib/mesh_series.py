"""A class to handle Meshseries data."""

# flake8: noqa
import h5py  # for requirements (needed for xdmf reading in meshio )
import meshio
import numpy as np
import pyvista as pv

from .mesh import Mesh
from typing import Callable, Optional as Opt


class MeshSeries:
    """
    A wrapper around pyvista and meshio for reading of pvd and xdmf timeseries.

    Will be replaced by own module in ogstools with similar interface.
    """

    def __init__(
        self, filepath: str, func: Opt[Callable] = None, **func_args
    ) -> None:
        self._data: dict[int, Mesh] = {}
        self._data_type = filepath.split(".")[-1]
        if self._data_type == "pvd":
            self._pvd_reader = pv.PVDReader(filepath)
        elif self._data_type == "xdmf":
            self._xdmf_reader = meshio.xdmf.TimeSeriesReader(filepath)
        else:
            msg = "Can only read 'pvd' or 'xdmf' files."
            raise TypeError(msg)
        self.func = func
        self.func_args = func_args

    def _read_pvd(self, timestep: int) -> Mesh:
        self._pvd_reader.set_active_time_point(timestep)
        return Mesh(self._pvd_reader.read()[0])

    def _read_xdmf(self, timestep: int) -> Mesh:
        points, cells = self._xdmf_reader.read_points_cells()
        _, point_data, cell_data, field_data = self._xdmf_reader.read_data(
            timestep
        )
        meshio_mesh = meshio.Mesh(points, cells, point_data, cell_data)
        return Mesh(pv.from_meshio(meshio_mesh))

    def read(self, timestep: int, lazy_eval: bool = True) -> Mesh:
        """Lazy read function."""
        if timestep in self._data:
            return self._data[timestep]
        if self._data_type == "pvd":
            mesh = self._read_pvd(timestep)
        elif self._data_type == "xdmf":
            mesh = self._read_xdmf(timestep)
        if lazy_eval:
            self._data[timestep] = (
                self.func(mesh, **self.func_args)
                if self.func is not None
                else mesh
            )
        return (
            self.func(mesh, **self.func_args) if self.func is not None else mesh
        )

    def clear(self) -> None:
        self._data.clear()

    @property
    def timesteps(self) -> range:
        """Return the timesteps of the timeseries data."""
        if self._data_type == "pvd":
            return range(self._pvd_reader.number_time_points)
        # elif self._data_type == "xdmf":
        return range(len(self.timevalues))

    @property
    def timevalues(self) -> list[float]:
        """Return the timevalues of the timeseries data."""
        if self._data_type == "pvd":
            return self._pvd_reader.time_values
        # elif self._data_type == "xdmf":
        time_values = []
        for collection_i in self._xdmf_reader.collection:
            for element in collection_i:
                if element.tag == "Time":
                    time_values += [float(element.attrib["Value"])]
        return time_values

    def closest_timestep(self, timevalue: float) -> int:
        """Return the corresponding timestep from a timevalue."""
        return int(np.argmin(np.abs(np.array(self.timevalues) - timevalue)))

    def closest_timevalue(self, timevalue: float) -> float:
        """Return the closest timevalue to a timevalue."""
        return self.timevalues[self.closest_timestep(timevalue)]

    def read_closest(self, timevalue: float) -> Mesh:
        """Return the closest timestep in the data for a given timevalue."""
        return self.read(self.closest_timestep(timevalue))

    def read_interp(self, timevalue: float, lazy_eval: bool = True) -> Mesh:
        """Return the temporal interpolated mesh for a given timevalue."""
        t_vals = np.array(self.timevalues)
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
