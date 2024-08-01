# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import meshio
import numpy as np
import pyvista as pv
from h5py import File
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
)
from tqdm.auto import tqdm

from ogstools import plot
from ogstools.propertylib.properties import Property, get_preset
from ogstools.propertylib.unit_registry import u_reg

from .mesh import Mesh
from .xdmf_reader import XDMFReader


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
        if self._data_type == "pvd":
            self._pvd_reader = pv.PVDReader(filepath)
            self.timestep_files = [
                str(Path(filepath).parent / dataset.path)
                for dataset in self._pvd_reader.datasets
            ]
            self.timevalues = np.asarray(self._pvd_reader.time_values)
        elif self._data_type == "xdmf":
            self._xdmf_reader = XDMFReader(filepath)
            self._read_xdmf(0)  # necessary to initialize hdf5_files
            meshes = self.hdf5["meshes"]
            self.hdf5_bulk_name = list(meshes.keys())[
                np.argmax([meshes[m]["geometry"].shape[1] for m in meshes])
            ]
            _time_values = []
            for collection_i in self._xdmf_reader.collection:
                for element in collection_i:
                    if element.tag == "Time":
                        _time_values += [float(element.attrib["Value"])]
            self.timevalues = np.asarray(_time_values)
        elif self._data_type == "vtu":
            self._vtu_reader = pv.XMLUnstructuredGridReader(filepath)
            self.timevalues = np.zeros(1)
        elif self._data_type == "synthetic":
            return
        else:
            msg = "Can only read 'pvd', 'xdmf' or 'vtu' files."
            raise TypeError(msg)

    def ip_tesselated(self) -> "MeshSeries":
        "Create a new MeshSeries from integration point tessellation."
        ip_ms = MeshSeries(
            Path(self.filepath).parent / "ip_meshseries.synthetic",
            self.time_unit,
            self.spatial_unit,
            self.spatial_output_unit,
        )
        ip_mesh = self.read(0).to_ip_mesh()
        ip_pt_cloud = self.read(0).to_ip_point_cloud()
        ordering = ip_mesh.find_containing_cell(ip_pt_cloud.points)
        for ts in self.timesteps:
            ip_data = {
                key: self.read(ts).field_data[key][np.argsort(ordering)]
                for key in ip_mesh.cell_data
            }
            ip_mesh.cell_data.update(ip_data)
            ip_ms._data[ts] = ip_mesh.copy()
        ip_ms.timevalues = self.timevalues
        return ip_ms

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
        meshio_mesh = meshio.Mesh(
            points, cells, point_data, cell_data, field_data
        )
        # pv.from_meshio does not copy field_data (fix in pyvista?)
        pv_mesh = pv.from_meshio(meshio_mesh)
        pv_mesh.field_data.update(field_data)
        return pv_mesh

    def read(self, timestep: int, lazy_eval: bool = True) -> Mesh:
        """Lazy read function."""
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

    def clear(self) -> None:
        self._data.clear()

    @property
    def timesteps(self) -> range:
        """Return the timesteps of the timeseries data."""
        if self._data_type == "vtu":
            return range(1)
        if self._data_type == "pvd":
            return range(self._pvd_reader.number_time_points)
        # elif self._data_type == "xdmf":
        return range(len(self.timevalues))

    def closest_timestep(self, timevalue: float) -> int:
        """Return the corresponding timestep from a timevalue."""
        return int(np.argmin(np.abs(self.timevalues - timevalue)))

    def closest_timevalue(self, timevalue: float) -> float:
        """Return the closest timevalue to a timevalue."""
        return self.timevalues[self.closest_timestep(timevalue)]

    def read_closest(self, timevalue: float) -> Mesh:
        """Return the closest timestep in the data for a given timevalue."""
        return self.read(self.closest_timestep(timevalue))

    def read_interp(self, timevalue: float, lazy_eval: bool = True) -> Mesh:
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
        if self._data_type == "synthetic":
            return np.asarray(
                [self._data[t][data_name] for t in self.timesteps]
            )
        return mesh[data_name]

    def aggregate(
        self,
        mesh_property: Property | str,
        func: Literal["min", "max", "mean", "median", "sum", "std", "var"],
    ) -> Mesh:
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
        mesh.clear_point_data()
        mesh.clear_cell_data()
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
        # TODO: put in separate function
        if func in ["min_time", "max_time"]:
            assert isinstance(np_func, type(np.argmax))
            mesh[output_name] = self.timevalues[np_func(vals, axis=0)]
        else:
            mesh[output_name] = np.empty(vals.shape[1:])
            assert isinstance(np_func, type(np.max))
            np_func(vals, out=mesh[output_name], axis=0)
        return mesh

    def probe(
        self,
        points: np.ndarray,
        data_name: str,
        interp_method: Literal["nearest", "linear"] = "linear",
    ) -> np.ndarray:
        """
        Probe the MeshSeries at observation points.

        :param points:          The points to sample at.
        :param data_name:       Name of the data to sample.
        :param interp_method:   Interpolation method, defaults to `linear`

        :returns:   `numpy` array of interpolated data at observation points.
        """
        points = np.asarray(points).reshape((-1, 3))
        values = np.swapaxes(self.values(data_name), 0, 1)
        geom = self.read(0).points

        if values.shape[0] != geom.shape[0]:
            # assume cell_data
            geom = self.read(0).cell_centers().points

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
        mesh_property: Property | str,
        mesh_property_abscissa: Property | str | None = None,
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
        Plot the transient property on the observation points in the MeshSeries.

            :param points:          The points to sample at.
            :param mesh_property:   The property to be sampled.
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
        mesh_property = get_preset(mesh_property, self.read(0))
        values = mesh_property.magnitude.transform(
            self.probe(points, mesh_property.data_name, interp_method)
        )
        if values.shape[0] == 1:
            values = values.flatten()
        Q_ = u_reg.Quantity
        time_unit_conversion = Q_(Q_(self.time_unit), time_unit).magnitude
        if mesh_property_abscissa is None:
            x_values = time_unit_conversion * self.timevalues
            x_label = f"time / {time_unit}" if time_unit else "time"
        else:
            mesh_property_abscissa = get_preset(
                mesh_property_abscissa, self.read(0)
            )
            x_values = mesh_property_abscissa.magnitude.transform(
                self.probe(
                    points, mesh_property_abscissa.data_name, interp_method
                )
            )
            x_unit_str = (
                f" / {mesh_property_abscissa.get_output_unit()}"
                if mesh_property_abscissa.get_output_unit()
                else ""
            )
            x_label = (
                mesh_property_abscissa.output_name.replace("_", " ")
                + x_unit_str
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
        ax.set_ylabel(mesh_property.get_label(plot.setup.label_split))
        ax.label_outer()
        ax.minorticks_on()
        return fig

    def animate(
        self,
        mesh_property: Property,
        timesteps: Sequence | None = None,
        titles: list[str] | None = None,
    ) -> FuncAnimation:
        """
        Create an animation for a property with given timesteps.

        :param property: the property field to be visualized on all timesteps
        :param timesteps: if sequence of int: the timesteps to animate
                        if sequence of float: the timevalues to animate
        :param titles: the title on top of the animation for each frame
        """
        plot.setup.layout = "tight"
        plot.setup.combined_colorbar = True

        ts = self.timesteps if timesteps is None else timesteps

        fig = self.read(0, False).plot_contourf(mesh_property)

        def init() -> None:
            pass

        def animate_func(i: int | float, fig: plt.Figure) -> None:
            index = np.argmin(np.abs(np.asarray(ts) - i))

            fig.axes[-1].remove()  # remove colorbar
            for ax in np.ravel(np.asarray(fig.axes)):
                ax.clear()
            if titles is not None:
                plot.setup.title_center = titles[index]
            if isinstance(i, int):
                mesh = self.read(i)
            else:
                mesh = self.read_interp(i, True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = plot.contourplots.draw_plot(
                    mesh, mesh_property, fig=fig, axes=fig.axes[0]
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
        mesh_property: Property | str,
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
        :param mesh_property:   The property to be visualized.
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

        time = Property("", self.time_unit, time_unit).transform(
            self.timevalues
        )
        if time_logscale:
            time = np.log10(time, where=time != 0)
            time[0] = time[1] - (time[2] - time[1])

        mesh_property = get_preset(mesh_property, self.read(0))
        values = mesh_property.transform(
            self.probe(points, mesh_property.data_name)
        )
        if "levels" in kwargs:
            levels = np.asarray(kwargs.pop("levels"))
        else:
            levels = plot.levels.compute_levels(
                kwargs.get("vmin", plot.setup.vmin) or np.nanmin(values),
                kwargs.get("vmax", plot.setup.vmax) or np.nanmax(values),
                kwargs.get("num_levels", plot.setup.num_levels),
            )
        cmap, norm = plot.utils.get_cmap_norm(levels, mesh_property)
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
            if mesh_property.bilinear_cmap and levels[0] < 0.0 < levels[-1]:
                ax.contour(time, y, values, [0], colors="white")
        else:
            ax.pcolormesh(time, y, values.T, cmap=cmap, norm=norm)

        spatial = plot.shared.spatial_quantity(self.read(0))
        fontsize = kwargs.get("fontsize", plot.setup.fontsize)
        ax.set_ylabel(ylabel + " / " + spatial.output_unit, fontsize=fontsize)
        xlabel = "time / " + time_unit
        if time_logscale:
            xlabel = "log10( " + xlabel + " )"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize, length=fontsize * 0.5)
        if cbar:
            plot.contourplots.add_colorbars(
                fig, ax, mesh_property, levels, **kwargs
            )
        plot.utils.update_font_sizes(fig.axes, fontsize)
        return fig

    # TODO: add member function to MeshSeries to get a difference for to timesteps
