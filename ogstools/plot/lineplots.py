from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.plot import setup, utils
from ogstools.variables import Variable, get_preset


def line(
    mesh: pv.UnstructuredGrid,
    y_var: str | Variable,
    x_var: str | Variable | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Figure | None:
    """Plot some data of a (1D) mesh.

    You can pass "x", "y" or "z" to either of x_var or y_var to specify which
    spatial dimension should be used for the corresponding axis. You can also
    pass two data variables for a phase plot.

    :param mesh:    The mesh which contains the data to plot
    :param y_var:   The variable to use for the y-axis
    :param x_var:   The variable to use for the x-axis, if None automatic
                    detection of spatial axis is tried.
    :param ax:      The matplotlib axis to use for plotting, if None creates a
                    new figure.
    :Keyword Arguments:
        - figsize:      figure size
        - color:        color of the line
        - linewidth:    width of the line
        - linestyle:    style of the line
        - label:        label in the legend
        - grid:         if True, show grid
        - all other kwargs get passed to matplotlib's plot function
    """

    figsize = kwargs.pop("figsize", [16, 10])
    ax_ = plt.subplots(figsize=figsize)[1] if ax is None else ax

    if x_var is None:
        non_flat_axis = np.argwhere(
            np.invert(np.all(np.isclose(mesh.points, mesh.points[0]), axis=0))
        ).ravel()
        x_var = "xyz"[non_flat_axis[0]]

    x_var = get_preset(x_var, mesh).magnitude
    y_var = get_preset(y_var, mesh).magnitude

    kwargs.setdefault("color", y_var.color)
    pure_spatial = y_var.data_name in "xyz" and x_var.data_name in "xyz"
    lw_scale = 5 if pure_spatial else 3
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
