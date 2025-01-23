# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast, overload

import meshio
import numpy as np
import pyvista as pv
from lxml import etree as ET
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from tqdm.auto import tqdm
from typeguard import typechecked

from ogstools import plot
from ogstools.variables import Variable, get_preset, u_reg

from .mesh import Mesh
from .xdmf_reader import XDMFReader


class MeshSeries:
    """
    A wrapper around pyvista and meshio for reading of pvd and xdmf timeseries.
    """

    def __init__(self, filepath: str | Path) -> None:
        """
        Initialize a MeshSeries object

            :param filepath:    Path to the PVD or XDMF file.

            :returns:           A MeshSeries object
        """
        if isinstance(filepath, Path):
            filepath = str(filepath)
        self.filepath = filepath
        self._spatial_factor = 1.0
        self._time_factor = 1.0
        self._mesh_cache: dict[float, Mesh] = {}
        self._mesh_func_opt: Callable[[Mesh], Mesh] | None = None
        self._data_type = filepath.split(".")[-1]
        # list of slices to be able to have nested slices with xdmf
        # (but only the first slice will be efficient)
        self._time_indices: list[slice | Any] = [slice(None)]
        self._timevalues: np.ndarray
        "original data timevalues - do not change except in synthetic data."
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

    def __deepcopy__(self, memo: dict) -> MeshSeries:
        # Deep copy is the default when using self.copy()
        # For shallow copy: self.copy(deep=False)
        cls = self.__class__
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for key, value in self.__dict__.items():
            if key != "_pvd_reader" and key != "_xdmf_reader":
                if isinstance(value, pv.UnstructuredGrid):
                    # For PyVista objects use their own copy method
                    setattr(self_copy, key, value.copy(deep=True))
                else:
                    # For everything that is neither reader nor PyVista object
                    # use the deepcopy
                    setattr(self_copy, key, deepcopy(value, memo))
            else:
                # Shallow copy of reader is needed, because timesteps are
                # stored in reader, deep copy doesn't work for _pvd_reader
                # and _xdmf_reader
                setattr(self_copy, key, copy(value))
        return self_copy

    def copy(self, deep: bool = True) -> MeshSeries:
        """
        Create a copy of MeshSeries object.
        Deep copy is the default.

        :param deep: switch to choose between deep (default) and shallow
                     (self.copy(deep=False)) copy.

        :returns: Copy of self.
        """
        return deepcopy(self) if deep else self

    @overload
    def __getitem__(self, index: int) -> Mesh:
        ...

    @overload
    def __getitem__(self, index: slice | Sequence) -> MeshSeries:
        ...

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int):
            return self.mesh(index)
        if isinstance(index, slice | Sequence):
            ms_copy = self.copy(deep=True)
            if ms_copy._time_indices == [slice(None)]:
                ms_copy._time_indices = [index]
            else:
                ms_copy._time_indices += [index]
            return ms_copy
        raise ValueError

    def __len__(self) -> int:
        return len(self.timesteps)

    def __iter__(self) -> Iterator[Mesh]:
        for i in np.arange(len(self.timevalues), dtype=int):
            yield self.mesh(i)

    def __str__(self) -> str:
        if self._data_type == "vtu":
            reader = self._vtu_reader
        elif self._data_type == "pvd":
            reader = self._pvd_reader
        else:
            reader = self._xdmf_reader
        return (
            f"MeshSeries:\n"
            f"filepath:         {self.filepath}\n"
            f"data_type:        {self._data_type}\n"
            f"timevalues:       {self.timevalues[0]} to {self.timevalues[-1]} in {len(self.timevalues)} steps\n"
            f"reader:           {reader}\n"
            f"rawdata_file:     {self.rawdata_file()}\n"
        )

    def aggregate_over_time(
        self, variable: Variable | str, func: Callable
    ) -> Mesh:
        """Aggregate data over all timesteps using a specified function.

        :param variable:    The mesh variable to be aggregated.
        :param func:        The aggregation function to apply. E.g. np.min,
                            np.max, np.mean, np.median, np.sum, np.std, np.var
        :returns:   A mesh with aggregated data according to the given function.
        """
        # TODO: add function to create an empty mesh from a given on
        # custom field_data may need to be preserved
        mesh = self.mesh(0).copy(deep=True)
        mesh.clear_point_data()
        mesh.clear_cell_data()
        if isinstance(variable, Variable):
            output_name = f"{variable.output_name}_{func.__name__}"
        else:
            output_name = f"{variable}_{func.__name__}"
        mesh[output_name] = func(self.values(variable), axis=0)
        return mesh

    def clear_cache(self) -> None:
        self._mesh_cache.clear()

    def closest_timestep(self, timevalue: float) -> int:
        """Return the corresponding timestep from a timevalue."""
        return int(np.argmin(np.abs(self.timevalues - timevalue)))

    def closest_timevalue(self, timevalue: float) -> float:
        """Return the closest timevalue to a timevalue."""
        return self.timevalues[self.closest_timestep(timevalue)]

    def ip_tesselated(self) -> MeshSeries:
        "Create a new MeshSeries from integration point tessellation."
        ip_ms = MeshSeries(
            Path(self.filepath).parent / "ip_meshseries.synthetic"
        )
        ip_mesh = self.mesh(0).to_ip_mesh()
        ip_pt_cloud = self.mesh(0).to_ip_point_cloud()
        ordering = ip_mesh.find_containing_cell(ip_pt_cloud.points)
        for i in np.arange(len(self.timevalues), dtype=int):
            ip_data = {
                key: self.mesh(i).field_data[key][np.argsort(ordering)]
                for key in ip_mesh.cell_data
            }
            ip_mesh.cell_data.update(ip_data)
            ip_ms._mesh_cache[
                self.timevalues[i]
            ] = ip_mesh.copy()  # pylint: disable=protected-access
        ip_ms._timevalues = self._timevalues  # pylint: disable=protected-access
        return ip_ms

    def mesh(self, timestep: int, lazy_eval: bool = True) -> Mesh:
        """Returns the mesh at the given timestep."""
        timevalue = self.timevalues[timestep]
        if not np.any(self.timevalues == timevalue):
            msg = f"Value {timevalue} not found in the array."
            raise ValueError(msg)
        data_timestep = np.argmax(
            self._timevalues * self._time_factor == timevalue
        )
        if timevalue in self._mesh_cache:
            mesh = self._mesh_cache[timevalue]
        else:
            match self._data_type:
                case "pvd":
                    pv_mesh = self._read_pvd(data_timestep)
                case "xdmf":
                    pv_mesh = self._read_xdmf(data_timestep)
                case "vtu":
                    pv_mesh = self._vtu_reader.read()
                case _:
                    msg = f"Unexpected datatype {self._data_type}."
                    raise TypeError(msg)
            mesh = Mesh(self.mesh_func(pv_mesh))
            if lazy_eval:
                self._mesh_cache[timevalue] = mesh
        if self._data_type == "pvd":
            mesh.filepath = Path(self.timestep_files[data_timestep])
        else:
            mesh.filepath = Path(self.filepath)
        return mesh

    def rawdata_file(self) -> Path | None:
        """
        Checks, if working with the raw data is possible. For example,
        OGS Simulation results with XDMF support efficient raw data access via
        `h5py <https://docs.h5py.org/en/stable/quick.html#quick>`_

        :returns: The location of the file containing the raw data. If it does not
                 support efficient read (e.g., no efficient slicing), it returns None.
        """
        if self._data_type == "xdmf" and self._xdmf_reader.has_fast_access():
            return self._xdmf_reader.rawdata_path()  # single h5 file
        return None

    def read_interp(self, timevalue: float, lazy_eval: bool = True) -> Mesh:
        """Return the temporal interpolated mesh for a given timevalue."""
        t_vals = self.timevalues
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
    def timevalues(self) -> np.ndarray:
        "Return the timevalues."
        vals = self._timevalues
        for index in self._time_indices:
            vals = vals[index]
        return vals * self._time_factor

    @property
    def timesteps(self) -> list:
        """
        Return the OGS simulation timesteps of the timeseries data.
        Not to be confused with timevalues which returns a list of
        times usually given in time units.
        """

        # TODO: read time steps from fn string if available
        return np.arange(len(self.timevalues), dtype=int)

    def _xdmf_values(self, variable_name: str) -> np.ndarray:
        dataitems = self._xdmf_reader.data_items[variable_name]
        # pv filters produces these arrays, which we can use for slicing
        # to also reflect the previous use of self.transform here
        mask_map = {
            "vtkOriginalPointIds": self.mesh(0).point_data,
            "vtkOriginalCellIds": self.mesh(0).cell_data,
        }
        for mask, data in mask_map.items():
            if variable_name in data and mask in data:
                result = dataitems[self._time_indices[0], self.mesh(0)[mask]]
                break
        else:
            result = dataitems[self._time_indices[0]]
        for index in self._time_indices[1:]:
            result = result[index]
        if self._mesh_func_opt is not None and not any(
            mask in data for mask, data in mask_map.items()
        ):
            # if transform function doesn't produce the mask arrays we have to
            # map data xdmf data to the entire list of meshes and apply the
            # function on each mesh individually.
            ms_copy = self.copy(deep=True)
            ms_copy._mesh_func_opt = None  # pylint: disable=protected-access
            ms_copy.clear_cache()
            raw_meshes = list(ms_copy)
            for mesh, data in zip(raw_meshes, result, strict=True):
                mesh[variable_name] = data
            meshes = list(map(self.mesh_func, raw_meshes))
            result = np.asarray([mesh[variable_name] for mesh in meshes])
        return result

    def values(self, variable: str | Variable) -> np.ndarray:
        """
        Get the data in the MeshSeries for all timesteps.

        Adheres to time slicing via `__get_item__` and an applied pyvista filter
        via `transform` if the applied filter produced 'vtkOriginalPointIds' or
        'vtkOriginalCellIds' (e.g. `clip(..., crinkle=True)`,
        `extract_cells(...)`, `threshold(...)`.)

        :param variable: Variable to read/process from the MeshSeries.

        :returns:   A numpy array of shape (n_timesteps, n_points/c_cells).
                    If given an argument of type Variable is given, its
                    transform function is applied on the data.
        """
        if isinstance(variable, Variable):
            if variable.mesh_dependent:
                return np.asarray([variable.transform(mesh) for mesh in self])
            variable_name = variable.data_name
        else:
            variable_name = variable

        if (
            self._data_type == "xdmf"
            and variable_name in self._xdmf_reader.data_items
            and not all(tv in self._mesh_cache for tv in self.timevalues)
        ):
            result = self._xdmf_values(variable_name)
        else:
            result = np.asarray([mesh[variable_name] for mesh in self])
        if isinstance(variable, Variable):
            return variable.transform(result)
        return result

    def _read_pvd(self, timestep: int) -> pv.UnstructuredGrid:
        self._pvd_reader.set_active_time_point(timestep)
        return self._pvd_reader.read()[0]

    def _read_xdmf(self, timestep: int) -> pv.UnstructuredGrid:
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
        mesh[output_name] = self.timevalues[
            np_func(self.values(variable), axis=0)
        ]
        return mesh

    def time_of_min(self, variable: Variable | str) -> Mesh:
        "Returns a Mesh with the time of the variable minimum as data."
        return self._time_of_extremum(variable, np.argmin, "min")

    def time_of_max(self, variable: Variable | str) -> Mesh:
        "Returns a Mesh with the time of the variable maximum as data."
        return self._time_of_extremum(variable, np.argmax, "max")

    def aggregate_over_domain(
        self, variable: Variable | str, func: Callable
    ) -> np.ndarray:
        """Aggregate data over domain per timestep using a specified function.

        :param variable:    The mesh variable to be aggregated.
        :param func:        The aggregation function to apply. E.g. np.min,
                            np.max, np.mean, np.median, np.sum, np.std, np.var

        :returns:   A numpy array with aggregated data.
        """
        return func(self.values(variable), axis=1)

    def plot_domain_aggregate(
        self,
        variable: Variable | str,
        func: Callable,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> plt.Figure | None:
        """
        Plot the transient aggregated data over the domain per timestep.

        :param variable:    The mesh variable to be aggregated.
        :param func:        The aggregation function to apply. E.g. np.min,
                            np.max, np.mean, np.median, np.sum, np.std, np.var
        :param ax:          matplotlib axis to use for plotting
        :param kwargs:      Keyword args passed to matplotlib's plot function.

        :returns:   A matplotlib Figure or None if plotting on existing axis.
        """
        variable = get_preset(variable, self.mesh(0))
        values = self.aggregate_over_domain(variable.magnitude, func)
        x_values = self.timevalues
        x_label = f"time t / {plot.setup.time_unit}"
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        if "label" in kwargs:
            label = kwargs.pop("label")
            ylabel = variable.get_label() + " " + func.__name__
        else:
            label = func.__name__
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
            values = values.ravel()
        if variable_abscissa is None:
            x_values = self.timevalues
            x_label = f"time / {plot.setup.time_unit}"
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
        ax.set_ylabel(variable.magnitude.get_label(plot.setup.label_split))
        ax.label_outer()
        ax.minorticks_on()
        return fig

    def animate(
        self,
        variable: Variable,
        timevalues: Sequence | None = None,
        plot_func: Callable[[plt.Axes, float], None] = lambda *_: None,
        **kwargs: Any,
    ) -> FuncAnimation:
        """
        Create an animation for a variable with given timevalues.

        :param variable: the field to be visualized on all timevalues
        :param timevalues: the timevalues to animate
        :param plot_func:   A function which expects to read a matplotlib Axes
                            and the time value of the current frame. Useful to
                            customize the plot in the animation.

        :Keyword Arguments: See :py:mod:`ogstools.plot.contourf`
        """
        plot.setup.layout = "tight"
        plot.setup.combined_colorbar = True

        ts = self.timevalues if timevalues is None else timevalues

        fig = plot.contourf(self.mesh(0, lazy_eval=False), variable)
        assert isinstance(fig, plt.Figure)
        plot_func(fig.axes[0], 0.0)
        fontsize = kwargs.get("fontsize", plot.setup.fontsize)
        plot.utils.update_font_sizes(fig.axes, fontsize)

        def init() -> None:
            pass

        def animate_func(tv: float, fig: plt.Figure) -> None:
            fig.axes[-1].remove()  # remove colorbar
            for ax in np.ravel(np.asarray(fig.axes)):
                ax.clear()
            mesh = self.read_interp(tv, True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_func(fig.axes[0], tv)
                plot.contourplots.draw_plot(
                    mesh, variable, fig=fig, axes=fig.axes[0], **kwargs
                )  # type: ignore[assignment]
                plot.utils.update_font_sizes(fig.axes, fontsize)

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

        time = self.timevalues
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
        cmap, norm = plot.utils.get_cmap_norm(levels, variable, **kwargs)

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
                aspect=(tmax - tmin) / (ymax - ymin) / kwargs.get("aspect", 1),
                interpolation="bicubic",
            )
            if variable.bilinear_cmap and levels[0] < 0.0 < levels[-1]:
                ax.contour(time, y, values.T, [0], colors="white")
        else:
            ax.pcolormesh(time, y, values.T, cmap=cmap, norm=norm)

        fontsize = kwargs.get("fontsize", plot.setup.fontsize)
        ax.set_ylabel(
            ylabel + " / " + plot.setup.spatial_unit, fontsize=fontsize
        )
        xlabel = "time / " + plot.setup.time_unit
        if time_logscale:
            xlabel = "log10( " + xlabel + " )"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize, length=fontsize * 0.5)
        if cbar:
            plot.contourplots.add_colorbars(fig, ax, variable, levels, **kwargs)
        plot.utils.update_font_sizes(fig.axes, fontsize)
        return fig

    @property
    def mesh_func(self) -> Callable[[Mesh], Mesh]:
        """Returns stored transformation function or identity if not given."""
        if self._mesh_func_opt is None:
            return lambda mesh: mesh.scale(self._spatial_factor)
        return lambda mesh: Mesh(
            self._mesh_func_opt(mesh).scale(  # type: ignore[misc]
                self._spatial_factor
            )
        )

    def transform(
        self, mesh_func: Callable[[Mesh], Mesh] = lambda mesh: mesh
    ) -> MeshSeries:
        """
        Apply a transformation function to the underlying mesh.

        :param mesh_func: A function which expects to read a mesh and return a
                          mesh. Useful for slicing / clipping / thresholding.

        :returns: A deep copy of this MeshSeries with transformed meshes.
        """
        ms_copy = self.copy(deep=True)
        # pylint: disable=protected-access
        for cache_timevalue, cache_mesh in self._mesh_cache.items():
            ms_copy._mesh_cache[cache_timevalue] = Mesh(mesh_func(cache_mesh))
        ms_copy._mesh_func_opt = lambda mesh: mesh_func(self.mesh_func(mesh))
        return ms_copy

    def scale(
        self,
        spatial: float | tuple[str, str] = 1.0,
        time: float | tuple[str, str] = 1.0,
    ) -> MeshSeries:
        """Scale the spatial coordinates and timevalues.

        Useful to convert to other units, e.g. "m" to "km" or "s" to "a".
        If given as tuple of strings, the latter units will also be set in
        ot.plot.setup.spatial_unt and ot.plot.setup.time_unit for plotting.

        :param spatial: Float factor or a tuple of str (from_unit, to_unit).
        :param time:    Float factor or a tuple of str (from_unit, to_unit).
        """
        Qty = u_reg.Quantity
        if isinstance(spatial, float):
            spatial_factor = spatial
        else:
            spatial_factor = Qty(Qty(spatial[0]), spatial[1]).magnitude
            plot.setup.spatial_unit = spatial[1]
        if isinstance(time, float):
            time_factor = time
        else:
            time_factor = Qty(Qty(time[0]), time[1]).magnitude
            plot.setup.time_unit = time[1]
        self._spatial_factor *= spatial_factor
        self._time_factor *= time_factor

        scaled_cache = {
            timevalue * time_factor: Mesh(mesh.scale(spatial_factor))
            for timevalue, mesh in self._mesh_cache.items()
        }
        self.clear_cache()
        self._mesh_cache = scaled_cache

        return self

    @typechecked
    def extract(
        self,
        index: slice | int | np.ndarray | list,
        preference: Literal["points", "cells"] = "points",
    ) -> MeshSeries:
        """
        Extract a subset of the domain by point or cell indices.

        :param index:       Indices of points or cells to extract.
        :param preference:  Selected entities.

        :returns: A MeshSeries with the selected domain subset.
        """
        func: dict[str, Callable[[Mesh], Mesh]] = {
            "points": lambda mesh: mesh.extract_points(
                np.arange(mesh.n_points)[index], include_cells=False
            ),
            "cells": lambda mesh: mesh.extract_cells(
                np.arange(mesh.n_points)[index]
            ),
        }
        return self.transform(func[preference])

    def _rename_vtufiles(self, new_pvd_fn: Path, fns: list[Path]) -> list:
        fns_new: list[Path] = []
        for filename in fns:
            filepathparts_at_timestep = list(filename.parts)
            filepathparts_at_timestep[-1] = filepathparts_at_timestep[
                -1
            ].replace(
                Path(self.filepath).name.split(".")[0],
                new_pvd_fn.name.split(".")[0],
            )
            fns_new.append(Path(*filepathparts_at_timestep))
        return fns_new

    def _save_vtu(self, new_pvd_fn: Path, fns: list[Path]) -> None:
        for i, timestep in enumerate(self.timesteps):
            if ".vtu" in fns[i].name:
                pv.save_meshio(
                    Path(new_pvd_fn.parent, fns[i].name), self.mesh(i)
                )
            elif ".xdmf" in fns[i].name:
                newname = fns[i].name.replace(
                    ".xdmf", f"_ts_{timestep}_t_{self.timevalues[i]}.vtu"
                )
                pv.save_meshio(Path(new_pvd_fn.parent, newname), self.mesh(i))
            else:
                s = "File type not supported."
                raise RuntimeError(s)

    def _save_pvd(self, new_pvd_fn: Path, fns: list[Path]) -> None:
        root = ET.Element("VTKFile")
        root.attrib["type"] = "Collection"
        root.attrib["version"] = "0.1"
        root.attrib["byte_order"] = "LittleEndian"
        root.attrib["compressor"] = "vtkZLibDataCompressor"
        collection = ET.SubElement(root, "Collection")
        for i, timestep in enumerate(self.timevalues):
            timestepselement = ET.SubElement(collection, "DataSet")
            timestepselement.attrib["timestep"] = str(timestep)
            timestepselement.attrib["group"] = ""
            timestepselement.attrib["part"] = "0"
            if ".xdmf" in fns[i].name:
                newname = fns[i].name.replace(
                    ".xdmf", f"_ts_{self.timesteps[i]}_t_{timestep}.vtu"
                )
                timestepselement.attrib["file"] = newname
            elif ".vtu" in fns[i].name:
                timestepselement.attrib["file"] = fns[i].name
            else:
                s = "File type not supported."
                raise RuntimeError(s)
        tree = ET.ElementTree(root)
        tree.write(
            new_pvd_fn,
            encoding="ISO-8859-1",
            xml_declaration=True,
            pretty_print=True,
        )

    def _check_path(self, filename: Path | None) -> Path:
        if not isinstance(filename, Path):
            s = "filename is empty"
            raise RuntimeError(s)
        assert isinstance(filename, Path)
        return cast(Path, filename)

    def save(self, filename: str, deep: bool = True) -> None:
        """
        Save mesh series to disk.

        :param filename:   Filename to save the series to. Extension specifies
                           the file type. Currently only PVD is supported.
        :param deep:  Specifies whether VTU/H5 files should be written.
        """
        fn = Path(filename)
        fns = [
            self._check_path(self.mesh(t).filepath)
            for t in np.arange(len(self.timevalues), dtype=int)
        ]
        if ".pvd" in fn.name:
            if deep is True:
                fns = self._rename_vtufiles(fn, fns)
                self._save_vtu(fn, fns)
            self._save_pvd(fn, fns)
        else:
            s = "Currently the save method is implemented for PVD/VTU only."
            raise RuntimeError(s)
