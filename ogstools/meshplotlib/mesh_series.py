"""A class to handle Meshseries data."""

# flake8: noqa
import h5py  # for requirements (needed for xdmf reading in meshio )
import meshio
import numpy as np
import pyvista as pv

from .mesh import Mesh


class MeshSeries:
    """
    A wrapper around pyvista and meshio for reading of pvd and xdmf timeseries.

    Will be replaced by own module in ogstools with similar interface.
    """

    def __init__(self, filepath: str) -> None:
        self._data: dict[int, Mesh] = {}
        self._data_type = filepath.split(".")[-1]
        if self._data_type == "pvd":
            self._pvd_reader = pv.PVDReader(filepath)
        elif self._data_type == "xdmf":
            self._xdmf_reader = meshio.xdmf.TimeSeriesReader(filepath)
        else:
            msg = "Can only read 'pvd' or 'xdmf' files."
            raise TypeError(msg)

    def _read_pvd(self, timestep: int) -> None:
        self._pvd_reader.set_active_time_point(timestep)
        self._data[timestep] = Mesh(self._pvd_reader.read()[0])

    def _read_xdmf(self, timestep: int) -> None:
        points, cells = self._xdmf_reader.read_points_cells()
        _, point_data, cell_data = self._xdmf_reader.read_data(timestep)
        meshio_mesh = meshio.Mesh(points, cells, point_data, cell_data)
        self._data[timestep] = Mesh(pv.from_meshio(meshio_mesh))

    def read(self, timestep: int) -> Mesh:
        """Lazy read function."""
        if timestep in self._data:
            return self._data[timestep]
        if self._data_type == "pvd":
            self._read_pvd(timestep)
        elif self._data_type == "xdmf":
            self._read_xdmf(timestep)
        return self._data[timestep]

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

    def timestep_from_value(self, timevalue: float) -> int:
        """Return the corresponding timestep from a timevalue."""
        return np.abs(timevalue - np.array(self.timevalues)).argmin()

    def read_closest_timestep(self, timevalue: float) -> Mesh:
        """Return the closest timestep in the data for a given timevalue."""
        return self.read(self.timestep_from_value(timevalue))
