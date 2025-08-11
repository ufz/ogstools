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

from ogstools.plot import setup, utils
from ogstools.variables import Variable, normalize_vars


def _format_ax(
    ax: plt.Axes,
    x_var: Variable,
    y_var: Variable,
    pure_spatial: bool,
    kwargs: dict,
) -> None:
    show_grid = kwargs.pop("grid", True) and not pure_spatial

    if ax.get_xlabel() == "":
        ax.set_xlabel(x_var.get_label(setup.label_split))
    if ax.get_ylabel() == "":
        ax.set_ylabel(y_var.get_label(setup.label_split))

    if show_grid:
        ax.grid(which="major", color="lightgrey", linestyle="-")
        ax.grid(which="minor", color="0.95", linestyle="--")
        ax.minorticks_on()

    if not pure_spatial:
        ax.figure.tight_layout()


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
    # it seems that in doing ctp on PolyLines the direction of points
    # is inverted or something similar, thus the reverse slicing
    for array in arrays:
        array[mask[::-1]] = np.nan


def line(
    dataset: pv.DataSet | Sequence[pv.DataSet],
    var1: str | Variable | None = None,
    var2: str | Variable | None = None,
    ax: plt.Axes | None = None,
    sort: bool = True,
    **kwargs: Any,
) -> plt.Figure | None:
    """Plot some data of a (1D) mesh.

    You can pass "x", "y" or "z" to either of x_var or y_var to specify which
    spatial dimension should be used for the corresponding axis. You can also
    pass two data variables for a phase plot. if no value is given, automatic
    detection of spatial axis is tried.

    >>> line(mesh)  # z=const: y over x, y=const: z over x, x=const: z over y
    >>> line(mesh, ot.variables.temperature)    # temperature over x, y or z
    >>> line(mesh, "y", "temperature")          # temperature over y
    >>> line(mesh, ot.variables.pressure, "y")  # y over pressure
    >>> line(mesh, "pressure", "temperature")   # temperature over pressure

    :param dataset:    The mesh which contains the data to plot
    :param var1:    Variable for the x-axis if var2 is given else for y-axis.
    :param var2:    Variable for the y-axis if var1 is given.
    :param ax:      The matplotlib axis to use for plotting, if None a new
                    figure will be created.
    :param sort:    Automatically sort the values along the dimension of the
                    mesh with the largest extent

    Keyword Arguments:
        - figsize:      figure size (default=[16, 10])
        - color:        color of the line
        - linewidth:    width of the line
        - linestyle:    style of the line
        - label:        label in the legend
        - grid:         if True, show grid
        - monospace:    if True, the legend uses a monospace font
        - loc:          location of the legend (default="upper right")
        - annotate:     string to be annotate at the center of the mesh
        - all other kwargs get passed to matplotlib's plot function

    Note:
        Using loc="best" will take a long time, if you plot lines on top of a
        contourplot, as matplotlib is calculating the best position against all
        the underlying cells.
    """
    if isinstance(var1, plt.Axes) or isinstance(var2, plt.Axes):
        msg = "Please provide ax as keyword argument only!"
        raise TypeError(msg)
    figsize = kwargs.pop("figsize", [16, 10])
    ax_: plt.Axes
    ax_ = plt.subplots(figsize=figsize)[1] if ax is None else ax

    mesh = dataset[0] if isinstance(dataset, Sequence) else dataset
    region_mesh = mesh.connectivity("all")
    x_var, y_var = normalize_vars(var1, var2, mesh)

    if isinstance(dataset, Sequence) and "color" not in kwargs:
        color = kwargs.pop("colors", "tab10")
        colorlist = utils.colors_from_cmap(color, len(dataset))
        ax_.set_prop_cycle(color=colorlist)
    else:
        kwargs.setdefault("color", y_var.color)
        ax_.set_prop_cycle(linestyle=["-", "--", ":", "-."])
    pure_spatial = y_var.data_name in "xyz" and x_var.data_name in "xyz"
    lw_scale = 4 if pure_spatial else 2.5
    kwargs.setdefault("linewidth", kwargs.pop("lw", None) or setup.linewidth)
    kwargs["linewidth"] *= lw_scale
    labels = kwargs.pop("labels", kwargs.pop("label", None))
    annotation = kwargs.pop("annotate", None)
    loc = kwargs.pop("loc", "upper right")

    if sort and "time" not in [var1, var2]:
        sort_idx = np.argmax(np.abs(np.diff(np.reshape(mesh.bounds, (3, 2)))))
        sort_ids = np.argsort(mesh.points[:, sort_idx])
    else:
        sort_ids = slice(None)
    x = x_var.transform(dataset)[..., sort_ids]
    y = y_var.transform(dataset)[..., sort_ids]
    if "vtkGhostType" in mesh.cell_data:
        _separate_by_empty_cells(mesh, x, y)
    if len(x.shape) < len(y.shape) and x.shape[0] != y.shape[0]:
        y = y.T
    if len(x.shape) > len(y.shape) and x.shape[0] != y.shape[0]:
        x = x.T
    if labels:
        kwargs["label"] = labels
    _format_ax(ax_, x_var, y_var, pure_spatial, kwargs)
    fontsize = kwargs.pop("fontsize", setup.fontsize)
    monospace = kwargs.pop("monospace", False)
    cell_types = np.unique(
        getattr(mesh, "celltypes", {cell.type for cell in mesh.cell})
    )
    only_points = (cell_types == [0]) or (cell_types == [1])
    reg_ids = np.unique(region_mesh.cell_data.get("RegionId", []))
    if isinstance(dataset, Sequence) or only_points or len(reg_ids) <= 1:
        ax_.plot(x, y, **kwargs)
    else:
        kwargs.setdefault("linestyle", kwargs.pop("ls", "-"))
        pt_regions = region_mesh.ctp().point_data["RegionId"]
        for reg_id in reg_ids:
            idx = pt_regions == reg_id
            label = kwargs.get("label") if reg_id == reg_ids[0] else None
            ax_.plot(x[idx], y[idx], **{**kwargs, "label": label})
    if annotation is not None:
        style = {"size": fontsize, "backgroundcolor": "0.8", "ha": "center"}
        label_xy = utils.padded(ax_, mesh.center[0], mesh.center[1])
        ax_.annotate(annotation, label_xy, **style)
    if labels:
        prop = {"size": fontsize}
        if monospace:
            prop["family"] = "monospace"
        ax_.legend(prop=prop, loc=loc)
    utils.update_font_sizes(axes=ax_, fontsize=fontsize)
    return ax_.figure if ax is None else None
