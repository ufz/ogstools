from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.plot import setup, utils
from ogstools.variables import Variable, get_preset


def line(
    mesh: pv.DataSet,
    var1: str | Variable | None = None,
    var2: str | Variable | None = None,
    ax: plt.Axes | None = None,
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

    :param mesh:    The mesh which contains the data to plot
    :param var1:    Variable for the x-axis if var2 is given else for y-axis.
    :param var2:    Variable for the y-axis if var1 is given.
    :param ax:      The matplotlib axis to use for plotting, if None a new
                    figure will be created.
    :Keyword Arguments:
        - figsize:      figure size (default=[16, 10])
        - color:        color of the line
        - linewidth:    width of the line
        - linestyle:    style of the line
        - label:        label in the legend
        - grid:         if True, show grid
        - all other kwargs get passed to matplotlib's plot function
    """
    if isinstance(var1, plt.Axes) or isinstance(var2, plt.Axes):
        msg = "Please provide ax as keyword argument only!"
        raise TypeError(msg)

    figsize = kwargs.pop("figsize", [16, 10])
    ax_ = plt.subplots(figsize=figsize)[1] if ax is None else ax

    axes_idx = np.argwhere(
        np.invert(np.all(np.isclose(mesh.points, mesh.points[0]), axis=0))
    ).ravel()
    if len(axes_idx) == 0:
        axes_idx = [0, 1]

    match var1, var2:
        case None, None:
            if len(axes_idx) == 1:
                axes_idx = [0, axes_idx[0] if axes_idx[0] != 0 else 1]
            x_var = get_preset("xyz"[axes_idx[0]], mesh).magnitude
            y_var = get_preset("xyz"[axes_idx[1]], mesh).magnitude
        case var1, None:
            x_var = get_preset("xyz"[axes_idx[0]], mesh).magnitude
            y_var = get_preset(var1, mesh).magnitude  # type: ignore[arg-type]
        case None, var2:
            x_var = get_preset("xyz"[axes_idx[0]], mesh).magnitude
            y_var = get_preset(var2, mesh).magnitude  # type: ignore[arg-type]
        case var1, var2:
            x_var = get_preset(var1, mesh).magnitude  # type: ignore[arg-type]
            y_var = get_preset(var2, mesh).magnitude  # type: ignore[arg-type]

    kwargs.setdefault("color", y_var.color)
    pure_spatial = y_var.data_name in "xyz" and x_var.data_name in "xyz"
    lw_scale = 4 if pure_spatial else 2.5
    kwargs.setdefault("linewidth", setup.linewidth * lw_scale)
    fontsize = kwargs.pop("fontsize", setup.fontsize)
    show_grid = kwargs.pop("grid", True) and not pure_spatial

    ax_.plot(x_var.transform(mesh), y_var.transform(mesh), **kwargs)

    if "label" in kwargs:
        ax_.legend(fontsize=fontsize)

    if ax_.get_xlabel() == "":
        ax_.set_xlabel(x_var.get_label(setup.label_split))
    if ax_.get_ylabel() == "":
        ax_.set_ylabel(y_var.get_label(setup.label_split))

    utils.update_font_sizes(axes=ax_, fontsize=fontsize)

    if show_grid:
        ax_.grid(which="major", color="lightgrey", linestyle="-")
        ax_.grid(which="minor", color="0.95", linestyle="--")
        ax_.minorticks_on()

    if ax is not None:
        return ax.figure

    return ax_.figure
