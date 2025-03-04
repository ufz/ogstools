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
        - all other kwargs get passed to matplotlib's plot function
    """
    if isinstance(var1, plt.Axes) or isinstance(var2, plt.Axes):
        msg = "Please provide ax as keyword argument only!"
        raise TypeError(msg)
    figsize = kwargs.pop("figsize", [16, 10])
    ax_: plt.Axes
    ax_ = plt.subplots(figsize=figsize)[1] if ax is None else ax

    mesh = dataset[0] if isinstance(dataset, Sequence) else dataset
    x_var, y_var = normalize_vars(var1, var2, mesh)

    if isinstance(dataset, Sequence):
        color = kwargs.pop("colors", kwargs.pop("color", "tab10"))
        colorlist = utils.colors_from_cmap(color, len(dataset))
        ax_.set_prop_cycle(color=colorlist)
    else:
        kwargs.setdefault("color", y_var.color)

    pure_spatial = y_var.data_name in "xyz" and x_var.data_name in "xyz"
    lw_scale = 4 if pure_spatial else 2.5
    kwargs.setdefault("linewidth", kwargs.pop("lw", None) or setup.linewidth)
    kwargs["linewidth"] *= lw_scale
    labels = kwargs.pop("labels", kwargs.pop("label", None))

    if sort and "time" not in [var1, var2]:
        sort_idx = np.argmax(np.abs(np.diff(np.reshape(mesh.bounds, (3, 2)))))
        sort_ids = np.argsort(mesh.points[:, sort_idx])
    else:
        sort_ids = slice(None)
    x = x_var.transform(dataset)[..., sort_ids]
    y = y_var.transform(dataset)[..., sort_ids]
    if len(x.shape) < len(y.shape) and x.shape[0] != y.shape[0]:
        y = y.T
    if len(x.shape) > len(y.shape) and x.shape[0] != y.shape[0]:
        x = x.T
    if labels:
        kwargs["label"] = labels
    _format_ax(ax_, x_var, y_var, pure_spatial, kwargs)
    fontsize = kwargs.pop("fontsize", setup.fontsize)
    ax_.plot(x, y, **kwargs)
    if labels:
        ax_.legend(fontsize=fontsize)
    utils.update_font_sizes(axes=ax_, fontsize=fontsize)
    return ax_.figure if ax is None else None
