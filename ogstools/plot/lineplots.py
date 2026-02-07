# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.figure import Figure

from ogstools.plot import setup, utils
from ogstools.variables import Variable, _normalize_vars


def _format_ax(
    ax: plt.Axes,
    x_var: Variable,
    y_var: Variable,
    show_grid: bool,
) -> None:
    if ax.get_xlabel() == "":
        ax.set_xlabel(x_var.get_label(setup.label_split))
    if ax.get_ylabel() == "":
        ax.set_ylabel(y_var.get_label(setup.label_split))

    if show_grid:
        ax.grid(which="major", color="lightgrey", linestyle="-")
        ax.grid(which="minor", color="0.95", linestyle="--")
        ax.minorticks_on()


def _separate_by_empty_cells(
    mesh: pv.DataSet, *arrays: list[np.ndarray]
) -> None:
    if "vtkGhostType" not in mesh.cell_data:
        return
    mask = (
        mesh.ctp().point_data.get("vtkGhostType", np.zeros(mesh.n_points)) != 0
    )
    if not all(len(mask) == len(arr) for arr in arrays):
        return
    for array in arrays:
        array[mask] = np.nan


def line(
    dataset: pv.DataSet | Sequence[pv.DataSet],
    var1: str | Variable | None = None,
    var2: str | Variable | None = None,
    ax: plt.Axes | None = None,
    sort: bool = True,
    outer_legend: bool | tuple[float, float] = False,
    **kwargs: Any,
) -> Figure | None:
    """Plot some data of a (1D) dataset.

    You can pass "x", "y" or "z" to either of x_var or y_var to specify which
    spatial dimension should be used for the corresponding axis. By passing
    "time" the timevalues will be use for this axis. You can also pass two data
    variables for a phase plot. if no value is given, automatic
    detection of spatial axis is tried.

    >>> line(ms, ot.variables.temperature)          # temperature over time
    >>> line(ms, ot.variables.temperature, "time")  # time over temperature
    >>> line(ms, "pressure", "temperature")     # temperature over pressure
    >>> line(mesh, ot.variables.temperature)    # temperature over x, y or z
    >>> line(mesh, "y", "temperature")          # temperature over y
    >>> line(mesh, ot.variables.pressure, "y")  # y over pressure
    >>> line(mesh)  # z=const: y over x, y=const: z over x, x=const: z over y

    :param dataset:    The mesh or meshseries which contains the data to plot.
    :param var1:    Variable for the x-axis if var2 is given else for y-axis.
    :param var2:    Variable for the y-axis if var1 is given.
    :param ax:      The matplotlib axis to use for plotting, if None a new
                    figure will be created.
    :param sort:    Automatically sort the values along the dimension of the
                    mesh with the largest extent (only for pointclouds).
    :outer_legend:  Draw legend to the right next to the plot area.
                    By default False (legend stays inside).
                    User can pass a tuple of two floats (x, y), which will be
                    passed to bbox_to_anchor parameter in matplotlib legend call.
                    True will pass the default values (1.05, 1.0).

    Keyword Arguments:
        - figsize:      figure size (default=[16, 10])
        - dpi:          resolution of the figure
        - color:        color of the line
        - linewidth:    width of the line
        - linestyle:    style of the line
        - label:        label in the legend
        - grid:         if True, show grid
        - monospace:    if True, the legend uses a monospace font
        - loc:          location of the legend (default="upper right")
        - clip_on:      If True, clip the output to stay within the Axes.
                        (default=False)
        - all other kwargs get passed to matplotlib's plot function

    Note:
        Using loc="best" will take a long time, if you plot lines on top of a
        contourplot, as matplotlib is calculating the best position against all
        the underlying cells.
    """

    ##### prepare figure/ax ##################################################
    if isinstance(var1, plt.Axes) or isinstance(var2, plt.Axes):
        msg = "Please provide ax as keyword argument only!"
        raise TypeError(msg)
    figsize = kwargs.pop("figsize", [16, 10])
    dpi = kwargs.pop("dpi", None)
    ax_: plt.Axes
    ax_ = plt.subplots(figsize=figsize, dpi=dpi)[1] if ax is None else ax

    ##### process variables ##################################################
    is_meshseries = isinstance(dataset, Sequence)
    mesh: pv.DataSet = dataset[0] if is_meshseries else dataset
    default = ["time", "time"] if is_meshseries else ["x", "y", "z"]
    x_var, y_var = _normalize_vars(var1, var2, mesh, default)
    pure_spatial = y_var.data_name in "xyz" and x_var.data_name in "xyz"

    # prefer point data over cell data
    x_cell_data = (x_var.data_name in mesh.cell_data) and (
        x_var.data_name not in mesh.point_data
    )
    y_cell_data = (y_var.data_name in mesh.cell_data) and (
        y_var.data_name not in mesh.point_data
    )

    ##### kwargs processing ##################################################
    if is_meshseries and "color" not in kwargs:
        color = kwargs.pop("colors", "tab10")
        colorlist = utils.colors_from_cmap(color, len(dataset))
        ax_.set_prop_cycle(color=colorlist)
    else:
        kwargs.setdefault("color", y_var.color)
        ax_.set_prop_cycle(linestyle=["-", "--", ":", "-."])
    lw_scale = 4 if pure_spatial else 2.5
    kwargs.setdefault("linewidth", kwargs.pop("lw", None) or setup.linewidth)
    kwargs.setdefault("clip_on", True)
    kwargs["linewidth"] *= lw_scale
    labels = kwargs.pop("labels", kwargs.pop("label", None))
    if labels:
        kwargs["label"] = labels
    outer_bool = outer_legend is True or isinstance(outer_legend, tuple)
    if outer_bool:
        loc = "upper left"
    else:
        loc = kwargs.pop("loc", "upper right" if pure_spatial else "best")
    fontsize = kwargs.pop("fontsize", setup.fontsize)
    prop = {"size": fontsize}
    if kwargs.pop("monospace", False):
        prop["family"] = "monospace"
    show_grid = kwargs.pop("grid", True) and not pure_spatial
    _format_ax(ax_, x_var, y_var, show_grid)

    ##### prepare data for plotting ##########################################
    x = x_var.transform(dataset)
    y = y_var.transform(dataset)
    if "vtkGhostType" in mesh.cell_data:
        x = x.astype(float)
        y = y.astype(float)
        _separate_by_empty_cells(mesh, x, y)
    # transposing to get individual lines int the plot in the case of plotting
    # linesamples for multiple timesteps or timeseries of multiple points
    if len(x.shape) < len(y.shape) and x.shape[0] != y.shape[0]:
        y = y.T
    if len(x.shape) > len(y.shape) and x.shape[0] != y.shape[0]:
        x = x.T

    def sorted_ids(
        mesh: pv.DataSet, use_cells: bool = False
    ) -> slice | np.ndarray:
        if is_meshseries or not sort:
            return slice(None)
        sort_idx = np.argmax(np.abs(np.diff(np.reshape(mesh.bounds, (3, 2)))))
        mesh_ = mesh.cell_centers() if use_cells else mesh
        return np.argsort(mesh_.points[:, sort_idx])

    ##### plotting ###########################################################
    cell_types = np.unique(
        getattr(mesh, "celltypes", {cell.type for cell in mesh.cell})
    )
    only_points = cell_types in [{0}, {1}]
    surf: pv.PolyData = mesh.extract_surface()
    strip: pv.PolyData = surf.strip()

    if is_meshseries or only_points or strip.n_cells <= 1:
        x_sort_ids = sorted_ids(mesh=mesh, use_cells=x_cell_data)
        if x_cell_data == y_cell_data:
            # pure cell or point data
            ax_.plot(x[x_sort_ids], y[x_sort_ids], **kwargs)
        elif x_cell_data or y_cell_data:
            if mesh.n_cells != mesh.n_points - 1:
                msg = "Line Plot of CellData vs. PointData for cells with inner points currently not supported!"
                raise ValueError(msg)
            # mixed point data and cell data - special case
            y_sort_ids = sorted_ids(mesh=mesh, use_cells=y_cell_data)

            def prepare_data(data: np.ndarray, use_cells: bool) -> np.ndarray:
                if use_cells:
                    # repeat the cell data to map it to the start and end point of the cell
                    return np.repeat(data, 2)
                # only repeat inner points
                return np.concatenate(
                    [[data[0]], np.repeat(data[1:-1], 2), [data[-1]]]
                )

            x_plot_vals = prepare_data(x[x_sort_ids], x_cell_data)
            y_plot_vals = prepare_data(y[y_sort_ids], y_cell_data)

            ax_.plot(x_plot_vals, y_plot_vals, **kwargs)
    else:
        kwargs.setdefault("linestyle", kwargs.pop("ls", "-"))
        orig_ids = np.arange(mesh.n_points, dtype=np.int32)
        if x_cell_data or y_cell_data:
            msg = "Plotting CellData for interrupted lines currently not supported! Convert CellData to PointData to use this function."
            raise ValueError(msg)
        for cell_id, linestrip in enumerate(strip.cell):
            sort_ids = strip.cell_data.get("vtkOriginalPointIds", orig_ids)[
                linestrip.point_ids
            ]
            label = kwargs.get("label") if cell_id == 0 else None
            ax_.plot(x[sort_ids], y[sort_ids], **{**kwargs, "label": label})

    ##### leged and final formatting #########################################
    if labels is not None:
        leg_prop: dict[str, Any] = {"loc": loc}
        if outer_legend is True:
            outer_legend = (1.05, 1.0)
        if isinstance(outer_legend, tuple):
            leg_prop["bbox_to_anchor"] = outer_legend
            leg_prop["borderaxespad"] = 0.0
        ax_.legend(prop=prop, **leg_prop)

    utils.update_font_sizes(axes=ax_, fontsize=fontsize)
    if not pure_spatial:
        ax_.figure.tight_layout()
    return ax_.figure if ax is None else None
