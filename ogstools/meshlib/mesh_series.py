# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal

import meshio
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from tqdm.auto import tqdm

from ogstools import plot
from ogstools.variables import Variable, get_preset, u_reg

from .mesh import Mesh
from .xdmf_reader import DataItems, XDMFReader


class PVDDataItems:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __getitem__(
        self, index: tuple | int | slice | np.ndarray
    ) -> np.ndarray:
        return self.values[index]


class MeshSeries:
    """
    A wrapper around pyvista and meshio for reading of pvd and xdmf timeseries.
    """

    def __init__(
        self,
        filepath: str | Path,
        time_unit: str = "s",
        spatial_unit: str = "m",
        spatial_output_unit: str = "m",
    ) -> None:
        """
        Initialize a MeshSeries object

            :param filepath:    Path to the PVD or XDMF file.
            :param time_unit:   Data unit of the timevalues.
            :param data_length_unit:    Length unit of the mesh data.
            :param output_length_unit:  Length unit in plots.

            :returns:           A MeshSeries object
        """
        if isinstance(filepath, Path):
            filepath = str(filepath)
        self.filepath = filepath
        self.time_unit = time_unit
        self.spatial_unit = spatial_unit
        self.spatial_output_unit = spatial_output_unit
        self._data: dict[int, Mesh] = {}
        self._data_type = filepath.split(".")[-1]
        self._dataitems: dict[str, Any] = {}
        if self._data_type == "xmf":
            self._data_type = "xdmf"

        if self._data_type == "pvd":
            self._pvd_reader = pv.PVDReader(filepath)
            self.timestep_files = [
                str(Path(filepath).parent / dataset.path)
                for dataset in self._pvd_reader.datasets
            ]
            self._timevalues = np.asarray(self._pvd_reader.time_values)
        elif self._data_type == "xdmf":
            self._xdmf_reader = XDMFReader(filepath)

            self._timevalues = np.asarray(
                [
                    float(element.attrib["Value"])
                    for collection_i in self._xdmf_reader.collection
                    for element in collection_i
                    if element.tag == "Time"
                ]
            )
        elif self._data_type == "vtu":
            self._vtu_reader = pv.XMLUnstructuredGridReader(filepath)
            self._timevalues = np.zeros(1)
        elif self._data_type == "synthetic":
            return
        else:
            msg = "Can only read 'pvd', 'xdmf', 'xmf'(from Paraview) or 'vtu' files."
            raise TypeError(msg)

    def __getitem__(self, index: int) -> list[pv.UnstructuredGrid]:
        # ToDo performance optimization for XDMF / HDF5 files
        return self.mesh(index)

    def __len__(self) -> int:
        return len(self.timesteps)

    def data(self, variable_name: str) -> DataItems:
        """
        Returns an DataItems object, that allows array indexing.
        To get "geometry"/"points" or "topology"/"cells" read the first time step and use
        pyvista functionality
        Selection example:
        ms = MeshSeries()
        temp = ms.data("temperature")
        time_step1_temps = temp[1,:]
        temps_at_some_points = temp[:,1:3]
        :param variable_name: Name the variable (e.g."temperature")
        :returns:   Returns an objects that allows array indexing.
        """

        if self._data_type == "xdmf":
            return self._xdmf_reader.data_items[variable_name]
        # for pvd and vtu check if data is already read or construct it
        if self._dataitems and self._dataitems[variable_name]:
            return self._dataitems[variable_name]

        all_meshes = [self.mesh(i) for i in self.timesteps]

        dataitems = self._structure_dataitems(all_meshes)
        # Lazy dataitems
        self._dataitems = {
            key: PVDDataItems(np.asarray(value))
            for key, value in dataitems.items()
        }
        return self._dataitems[variable_name]

    def _structure_dataitems(
        self, all_meshes: list[pv.UnstructuredGrid]
    ) -> dict[str, list]:
        # Reads all meshes and returns a dict with variables as key
        # (e.g. "temperature")
        dataitems: dict[str, list] = {}
        for mesh in all_meshes:
            for name in mesh.cell_data:
                if name in dataitems:
                    dataitems[name].append(mesh.cell_data[name])
                else:
                    dataitems[name] = [mesh.cell_data[name]]
            for name in mesh.point_data:
                if name in dataitems:
                    dataitems[name].append(mesh.point_data[name])
                else:
                    dataitems[name] = [mesh.point_data[name]]
        return dataitems

    def __repr__(self) -> str:
        if self._data_type == "vtu":
            reader = self._vtu_reader
        elif self._data_type == "pvd":
            reader = self._pvd_reader
        else:
            reader = self._xdmf_reader
        return (
            f"MeshSeries:\n"
            f"filepath:       {self.filepath}\n"
            f"spatial_unit:   {self.spatial_unit}\n"
            f"data_type:      {self._data_type}\n"
            f"timevalues:     {self._timevalues[0]}{self.time_unit} to {self._timevalues[0]}{self.time_unit} in {len(self._timevalues)} steps\n"
            f"reader:         {reader}\n"
            f"rawdata_file:   {self.rawdata_file()}\n"
        )

    def aggregate(
        self,
        variable: Variable | str,
        np_func: Callable,
        axis: int,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Aggregate data of all timesteps using a specified function.

        :param variable:   The mesh variable to be aggregated.
        :param func:            The aggregation function to apply.
        :returns:   A `numpy.ndarray` of the same length as the timesteps if
                    axis=0 or of the same length as the data if axis=1.
        """
        if isinstance(variable, Variable):
            if variable.mesh_dependent:
                vals = np.asarray(
                    [
                        variable.transform(self.mesh(t))
                        for t in tqdm(self.timesteps)
                    ]
                )
            else:
                vals = variable.transform(self.values(variable.data_name))
        else:
            vals = self.values(variable)
        return (
            np_func(vals, axis=axis)
            if mask is None
            else np_func(vals[:, mask], axis=axis)
        )

    _np_str: ClassVar = {
        "min": np.min, "max": np.max, "mean": np.mean, "median": np.median,
        "sum": np.sum, "std": np.std, "var": np.var,
    }  # fmt: skip

    def aggregate_over_time(
        self,
        variable: Variable | str,
        func: Literal["min", "max", "mean", "median", "sum", "std", "var"],
    ) -> Mesh:
        """Aggregate data over all timesteps using a specified function.

        :param variable:   The mesh variable to be aggregated.
        :param func:            The aggregation function to apply.
        :returns:   A mesh with aggregated data according to the given function.
        """
        np_func = self._np_str[func]
        # TODO: add function to create an empty mesh from a given on
        # custom field_data may need to be preserved
        mesh = self.mesh(0).copy(deep=True)
        mesh.clear_point_data()
        mesh.clear_cell_data()
        if isinstance(variable, Variable):
            output_name = f"{variable.output_name}_{func}"
        else:
            output_name = f"{variable}_{func}"
        mesh[output_name] = self.aggregate(variable, np_func, axis=0)
        return mesh

    def clear(self) -> None:
        self._data.clear()

    def closest_timestep(self, timevalue: float) -> int:
        """Return the corresponding timestep from a timevalue."""
        return int(np.argmin(np.abs(self._timevalues - timevalue)))

    def closest_timevalue(self, timevalue: float) -> float:
        """Return the closest timevalue to a timevalue."""
        return self._timevalues[self.closest_timestep(timevalue)]

    def ip_tesselated(self) -> "MeshSeries":
        "Create a new MeshSeries from integration point tessellation."
        ip_ms = MeshSeries(
            Path(self.filepath).parent / "ip_meshseries.synthetic",
            self.time_unit,
            self.spatial_unit,
            self.spatial_output_unit,
        )
        ip_mesh = self.mesh(0).to_ip_mesh()
        ip_pt_cloud = self.mesh(0).to_ip_point_cloud()
        ordering = ip_mesh.find_containing_cell(ip_pt_cloud.points)
        for ts in self.timesteps:
            ip_data = {
                key: self.mesh(ts).field_data[key][np.argsort(ordering)]
                for key in ip_mesh.cell_data
            }
            ip_mesh.cell_data.update(ip_data)
            ip_ms._data[ts] = ip_mesh.copy()  # pylint: disable=protected-access
        ip_ms._timevalues = self._timevalues  # pylint: disable=protected-access
        return ip_ms

    def mesh(self, timestep: int, lazy_eval: bool = True) -> Mesh:
        """Selects mesh at given timestep all data function."""
        timestep = self.timesteps[timestep]
        if timestep in self._data:
            pv_mesh = self._data[timestep]
        else:
            if self._data_type == "pvd":
                pv_mesh = self._read_pvd(timestep)
            elif self._data_type == "xdmf":
                pv_mesh = self._read_xdmf(timestep)
            elif self._data_type == "vtu":
                pv_mesh = self._vtu_reader.read()
            if lazy_eval:
                self._data[timestep] = pv_mesh
        mesh = Mesh(pv_mesh, self.spatial_unit, self.spatial_output_unit)
        if self._data_type == "pvd":
            mesh.filepath = Path(self.timestep_files[timestep])
        else:
            mesh.filepath = Path(self.filepath)
        return mesh

    def rawdata_file(self) -> Path | None:
        """
        Checks, if working with the raw data is possible. For example,
        OGS Simulation results with XDMF support efficient raw data access via
        `h5py <https://docs.h5py.org/en/stable/quick.html#quick>`_

        :return: The location of the file containing the raw data. If it does not
                 support efficient read (e.g., no efficient slicing), it returns None.
        """
        if self._data_type == "xdmf" and self._xdmf_reader.has_fast_access():
            return self._xdmf_reader.rawdata_path()  # single h5 file
        return None

    def read_interp(self, timevalue: float, lazy_eval: bool = True) -> Mesh:
        """Return the temporal interpolated mesh for a given timevalue."""
        t_vals = self._timevalues
        ts1 = int(t_vals.searchsorted(timevalue, "right") - 1)
        ts2 = min(ts1 + 1, len(t_vals) - 1)
        if np.isclose(timevalue, t_vals[ts1]):
            return self.mesh(ts1, lazy_eval)
        mesh1 = self.mesh(ts1, lazy_eval)
        mesh2 = self.mesh(ts2, lazy_eval)
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

    @property
    def timesteps(self) -> range:
        """Return the timesteps of the timeseries data."""
        if self._data_type == "vtu":
            return range(1)
        if self._data_type == "pvd":
            return range(self._pvd_reader.number_time_points)
        # elif self._data_type == "xdmf":
        return range(len(self._timevalues))

    def timevalues(self, time_unit: str | None = None) -> np.ndarray:
        "Return the timevalues, optionally converted to another time unit."
        return (
            u_reg.Quantity(self._timevalues, self.time_unit)
            .to(self.time_unit if time_unit is None else time_unit)
            .magnitude
        )

    def values(
        self, data_name: str, selection: slice | np.ndarray | None = None
    ) -> np.ndarray:
        """
        Get the data in the MeshSeries for all timesteps.

        :param data_name: Name of the data in the MeshSeries.
        :param selection: Can limit the data to be read.
            - **Time** is always the first dimension.
            - If `None`, it takes the selection that is defined in the xdmf file.
            - If a tuple or `np.ndarray`: see how `h5py` uses Numpy array indexing.
            - If a slice: see Python slice reference.
            - If a string: see example:

            Example: ``"|0 0 0:1 1 1:1 190 3:97 190 3"``

            This represents the selection
            ``[(offset(0,0,0): step(1,1,1) : end(1,190,3) : of_data_with_size(97,190,30))]``.

        :returns: A numpy array of the requested data for all timesteps.
        """

        if isinstance(selection, np.ndarray | tuple):
            time_selection = selection[0]
        else:
            time_selection = slice(None)

        if self._data_type == "xdmf":
            return self._xdmf_reader.data_items[data_name][time_selection]
        if self._data_type == "pvd":
            return np.asarray(
                [
                    self.mesh(t)[data_name]
                    for t in tqdm(self.timesteps[time_selection])
                ]
            )
        if self._data_type == "synthetic":
            return np.asarray(
                [
                    self._data[t][data_name]
                    for t in self.timesteps[time_selection]
                ]
            )
        # vtu
        mesh = self.mesh(0)
        return mesh[data_name]

    def _read_pvd(self, timestep: int) -> pv.UnstructuredGrid:
        self._pvd_reader.set_active_time_point(timestep)
        return self._pvd_reader.read()[0]

    def _read_xdmf(self, timestep: int | tuple | slice) -> pv.UnstructuredGrid:
        if isinstance(timestep, int):
            points, cells = self._xdmf_reader.read_points_cells()
            _, point_data, cell_data, field_data = self._xdmf_reader.read_data(
                timestep
            )
            meshio_mesh = meshio.Mesh(
                points, cells, point_data, cell_data, field_data
            )
            # pv.from_meshio does not copy field_data (fix in pyvista?)
            pv_mesh = pv.from_meshio(meshio_mesh)
            pv_mesh.field_data.update(field_data)
            return pv_mesh

        # ToDo support effective read of multiple meshes
        msg = "Only single timestep reading is supported for XDMF files."
        raise NotImplementedError(msg)

    def _time_of_extremum(
        self,
        variable: Variable | str,
        np_func: Callable,
        prefix: Literal["min", "max"],
    ) -> Mesh:
        """Returns a Mesh with the time of a given variable extremum as data.

        The data is named as `f'{prefix}_{variable.output_name}_time'`."""
        mesh = self.mesh(0).copy(deep=True)
        variable = get_preset(variable, mesh)
        mesh.clear_point_data()
        mesh.clear_cell_data()
        output_name = f"{prefix}_{variable.output_name}_time"
        mesh[output_name] = self._timevalues[
            self.aggregate(variable, np_func, axis=0)
        ]
        return mesh

    def time_of_min(self, variable: Variable | str) -> Mesh:
        "Returns a Mesh with the time of the variable minimum as data."
        return self._time_of_extremum(variable, np.argmin, "min")

    def time_of_max(self, variable: Variable | str) -> Mesh:
        "Returns a Mesh with the time of the variable maximum as data."
        return self._time_of_extremum(variable, np.argmax, "max")

    def aggregate_over_domain(
        self,
        variable: Variable | str,
        func: Literal["min", "max", "mean", "median", "sum", "std", "var"],
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Aggregate data over domain per timestep using a specified function.

        :param variable:   The mesh variable to be aggregated.
        :param func:            The aggregation function to apply.
        :param mask:            A numpy array as a mask for the domain.

        :returns:   A numpy array with aggregated data.
        """
        np_func = self._np_str[func]
        return self.aggregate(variable, np_func, axis=1, mask=mask)

    def plot_domain_aggregate(
        self,
        variable: Variable | str,
        func: Literal["min", "max", "mean", "median", "sum", "std", "var"],
        timesteps: slice | None = None,
        time_unit: str | None = "s",
        mask: np.ndarray | None = None,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> plt.Figure | None:
        """
        Plot the transient aggregated data over the domain per timestep.

        :param variable:   The mesh variable to be aggregated.
        :param func:            The aggregation function to apply.
        :param timesteps:       A slice to select the timesteps. Default: all.
        :param time_unit:       Output unit of the timevalues.
        :param mask:            A numpy array as a mask for the domain.
        :param ax:              matplotlib axis to use for plotting
        :param kwargs:      Keyword args passed to matplotlib's plot function.

        :returns:   A matplotlib Figure or None if plotting on existing axis.
        """
        variable = get_preset(variable, self.mesh(0))
        timeslice = slice(None, None) if timesteps is None else timesteps
        values = self.aggregate_over_domain(variable.magnitude, func, mask)[
            timeslice
        ]
        time_unit = time_unit if time_unit is not None else self.time_unit
        x_values = self.timevalues(time_unit)[timeslice]
        x_label = f"time t / {time_unit}"
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        if "label" in kwargs:
            label = kwargs.pop("label")
            ylabel = variable.get_label() + " " + func
        else:
            label = func
            ylabel = variable.get_label()
        ax.plot(x_values, values, label=label, **kwargs)
        ax.set_axisbelow(True)
        ax.grid(which="major", color="lightgrey", linestyle="-")
        ax.grid(which="minor", color="0.95", linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.label_outer()
        ax.minorticks_on()
        return fig

    def probe(
        self,
        points: np.ndarray,
        data_name: str,
        interp_method: Literal["nearest", "linear"] = "linear",
    ) -> np.ndarray:
        """
        Probe the MeshSeries at observation points.

        :param points:          The observation points to sample at.
        :param data_name:       Name of the data to sample.
        :param interp_method:   Interpolation method, defaults to `linear`

        :returns:   `numpy` array of interpolated data at observation points.
        """
        points = np.asarray(points).reshape((-1, 3))
        values = np.swapaxes(self.values(data_name), 0, 1)
        geom = self.mesh(0).points

        if values.shape[0] != geom.shape[0]:
            # assume cell_data
            geom = self.mesh(0).cell_centers().points

        # remove flat dimensions for interpolation
        flat_axis = np.argwhere(np.all(np.isclose(geom, geom[0]), axis=0))
        geom = np.delete(geom, flat_axis, 1)
        points = np.delete(points, flat_axis, 1)

        interp = {
            "nearest": NearestNDInterpolator(geom, values),
            "linear": LinearNDInterpolator(geom, values, np.nan),
        }[interp_method]

        return np.swapaxes(interp(points), 0, 1)

    def plot_probe(
        self,
        points: np.ndarray,
        variable: Variable | str,
        variable_abscissa: Variable | str | None = None,
        labels: list[str] | None = None,
        time_unit: str | None = "s",
        interp_method: Literal["nearest", "linear"] = "linear",
        colors: list | None = None,
        linestyles: list | None = None,
        ax: plt.Axes | None = None,
        fill_between: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | None:
        """
        Plot the transient variable on the observation points in the MeshSeries.

            :param points:          The points to sample at.
            :param variable:   The variable to be sampled.
            :param labels:          The labels for each observation point.
            :param time_unit:       Output unit of the timevalues.
            :param interp_method:   Choose the interpolation method, defaults to
                                    `linear` for xdmf MeshSeries and
                                    `probefilter` for pvd MeshSeries.
            :param interp_backend:  Interpolation backend for PVD MeshSeries.
            :param kwargs:          Keyword arguments passed to matplotlib's
                                    plot function.

            :returns:   A matplotlib Figure
        """
        points = np.asarray(points).reshape((-1, 3))
        variable = get_preset(variable, self.mesh(0))
        values = variable.magnitude.transform(
            self.probe(points, variable.data_name, interp_method)
        )
        if values.shape[0] == 1:
            values = values.flatten()
        Q_ = u_reg.Quantity
        time_unit_conversion = Q_(Q_(self.time_unit), time_unit).magnitude
        if variable_abscissa is None:
            x_values = time_unit_conversion * self._timevalues
            x_label = f"time / {time_unit}" if time_unit else "time"
        else:
            variable_abscissa = get_preset(variable_abscissa, self.mesh(0))
            x_values = variable_abscissa.magnitude.transform(
                self.probe(points, variable_abscissa.data_name, interp_method)
            )
            x_unit_str = (
                f" / {variable_abscissa.get_output_unit()}"
                if variable_abscissa.get_output_unit()
                else ""
            )
            x_label = (
                variable_abscissa.output_name.replace("_", " ") + x_unit_str
            )
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        if points.shape[0] > 1:
            ax.set_prop_cycle(
                plot.utils.get_style_cycler(len(points), colors, linestyles)
            )
        if fill_between:
            ax.fill_between(
                x_values,
                np.min(values, axis=-1),
                np.max(values, axis=-1),
                label=labels,
                **kwargs,
            )
        else:
            ax.plot(x_values, values, label=labels, **kwargs)
        if labels is not None:
            ax.legend(
                facecolor="white", framealpha=1, prop={"family": "monospace"}
            )
        ax.set_axisbelow(True)
        ax.grid(which="major", color="lightgrey", linestyle="-")
        ax.grid(which="minor", color="0.95", linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(variable.get_label(plot.setup.label_split))
        ax.label_outer()
        ax.minorticks_on()
        return fig

    def animate(
        self,
        variable: Variable,
        timesteps: Sequence | None = None,
        mesh_func: Callable[[Mesh], Mesh] = lambda mesh: mesh,
        plot_func: Callable[[plt.Axes, float], None] = lambda *_: None,
        **kwargs: Any,
    ) -> FuncAnimation:
        """
        Create an animation for a variable with given timesteps.

        :param variable: the field to be visualized on all timesteps
        :param timesteps: if sequence of int: the timesteps to animate
                        if sequence of float: the timevalues to animate
        :param mesh_func:   A function which expects to read a mesh and return a
                            mesh. Useful, for slicing / clipping / thresholding
                            the meshseries for animation.
        :param plot_func:   A function which expects to read a matplotlib Axes
                            and the time value of the current frame. Useful to
                            customize the plot in the animation.

        :Keyword Arguments: See :py:mod:`ogstools.plot.contourf`
        """
        plot.setup.layout = "tight"
        plot.setup.combined_colorbar = True

        ts = self.timesteps if timesteps is None else timesteps

        fig = plot.contourf(mesh_func(self.mesh(0, False)), variable)
        assert isinstance(fig, plt.Figure)
        plot_func(fig.axes[0], 0.0)

        def init() -> None:
            pass

        def animate_func(i: int | float, fig: plt.Figure) -> None:
            fig.axes[-1].remove()  # remove colorbar
            for ax in np.ravel(np.asarray(fig.axes)):
                ax.clear()
            if isinstance(i, int):
                mesh = mesh_func(self.mesh(i))
            else:
                mesh = mesh_func(self.read_interp(i, True))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_func(fig.axes[0], i)
                plot.contourplots.draw_plot(
                    mesh, variable, fig=fig, axes=fig.axes[0], **kwargs
                )  # type: ignore[assignment]
                plot.utils.update_font_sizes(fig.axes)

        _func = partial(animate_func, fig=fig)

        return FuncAnimation(
            fig,  # type: ignore[arg-type]
            _func,  # type: ignore[arg-type]
            frames=tqdm(ts),
            blit=False,
            interval=50,
            repeat=False,
            init_func=init,  # type: ignore[arg-type]
        )

    def plot_time_slice(
        self,
        variable: Variable | str,
        points: np.ndarray,
        y_axis: Literal["x", "y", "z", "dist", "auto"] = "auto",
        interpolate: bool = True,
        time_unit: str = "s",
        time_logscale: bool = False,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        cbar: bool = True,
        **kwargs: Any,
    ) -> plt.Figure:
        """
        :param variable:    The variable to be visualized.
        :param points:  The points along which the data is sampled over time.
        :param y_axis:  The component of the sampling points which labels the
                        y-axis. By default, if only one coordinate of the points
                        is changing, this axis is taken, otherwise the distance
                        along the line is taken.
        :param interpolate:     Smoothen the result be interpolation.
        :param time_unit:       Time unit displayed on the x-axis.
        :param time_logscale:   Should log-scaling be applied to the time-axis?
        :param fig:             matplotlib figure to use for plotting.
        :param ax:              matplotlib axis to use for plotting.
        :param cb_loc:          Colorbar location. If None, omit colorbar.

        :Keyword Arguments:
            - cb_labelsize:       colorbar labelsize
            - cb_loc:             colorbar location ('left' or 'right')
            - cb_pad:             colorbar padding
            - cmap:               colormap
            - vmin:               minimum value for colorbar
            - vmax:               maximum value for colorbar
            - num_levels:         number of levels for colorbar
            - figsize:            figure size
            - dpi:                resolution
        """
        if ax is None and fig is None:
            fig, ax = plt.subplots(
                figsize=kwargs.get("figsize", [18, 14]),
                dpi=kwargs.get("dpi", 100),
            )
        elif ax is None or fig is None:
            msg = "Please provide fig and ax together or not at all."
            raise ValueError(msg)

        time = Variable("", self.time_unit, time_unit).transform(
            self._timevalues
        )
        if time_logscale:
            time = np.log10(time, where=time != 0)
            time[0] = time[1] - (time[2] - time[1])

        variable = get_preset(variable, self.mesh(0))
        values = variable.transform(self.probe(points, variable.data_name))
        if "levels" in kwargs:
            levels = np.asarray(kwargs.pop("levels"))
        else:
            levels = plot.levels.compute_levels(
                kwargs.get("vmin", plot.setup.vmin) or np.nanmin(values),
                kwargs.get("vmax", plot.setup.vmax) or np.nanmax(values),
                kwargs.get("num_levels", plot.setup.num_levels),
            )
        cmap, norm = plot.utils.get_cmap_norm(levels, variable)
        cmap = kwargs.get("cmap", cmap)

        non_flat_axis = np.argwhere(
            np.invert(np.all(np.isclose(points, points[0]), axis=0))
        )
        if y_axis == "auto" and non_flat_axis.shape[0] == 1:
            y = points[:, non_flat_axis[0, 0]]
            ylabel = "xyz"[non_flat_axis[0, 0]]
        elif y_axis in "xyz":
            y = points[:, "xyz".index(y_axis)]
            ylabel = y_axis
        else:
            y = np.linalg.norm(points - points[0], axis=1)
            ylabel = "distance"

        if interpolate:
            grid_interp = RegularGridInterpolator(
                (time, y), values, method="cubic"
            )
            tmin, tmax = (np.min(time), np.max(time))
            ymin, ymax = (np.min(y), np.max(y))
            t_linspace = np.linspace(tmin, tmax, num=100)
            y_linspace = np.linspace(ymin, ymax, num=100)
            z_grid = grid_interp(tuple(np.meshgrid(t_linspace, y_linspace)))
            ax.imshow(
                z_grid[::-1],
                cmap=cmap,
                norm=norm,
                extent=(tmin, tmax, ymin, ymax),
                aspect=(tmax - tmin) / (ymax - ymin),
                interpolation="bicubic",
            )
            if variable.bilinear_cmap and levels[0] < 0.0 < levels[-1]:
                ax.contour(time, y, values, [0], colors="white")
        else:
            ax.pcolormesh(time, y, values.T, cmap=cmap, norm=norm)

        spatial = plot.shared.spatial_quantity(self.mesh(0))
        fontsize = kwargs.get("fontsize", plot.setup.fontsize)
        ax.set_ylabel(ylabel + " / " + spatial.output_unit, fontsize=fontsize)
        xlabel = "time / " + time_unit
        if time_logscale:
            xlabel = "log10( " + xlabel + " )"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize, length=fontsize * 0.5)
        if cbar:
            plot.contourplots.add_colorbars(fig, ax, variable, levels, **kwargs)
        plot.utils.update_font_sizes(fig.axes, fontsize)
        return fig

    # TODO: add member function to MeshSeries to get a difference for to timesteps
