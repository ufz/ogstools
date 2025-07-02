# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""heatmap functions."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ogstools.variables import Variable

from .contourplots import add_colorbars
from .levels import compute_levels
from .utils import get_cmap_norm, update_font_sizes


def heatmap(
    data: np.ndarray,
    variable: Variable,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    x_vals: np.ndarray | None = None,
    y_vals: np.ndarray | None = None,
    **kwargs: Any,
) -> plt.Figure | None:
    """
    Create a heatmap plot of given data.

    :param data: The two-dimensional data of interest.
    :param variable: Provides the label and colormap for the colorbar.
    :param fig: Optionally plot into this figure.
    :param ax: Optionally plot into this Axes.
    :param x_vals: one-dimensional x_values of the data.
    :param y_vals: one-dimensional y_values of the data.

    Keyword Arguments:
        - figsize:      figure size
        - dpi:          resolution
        - vmin:         minimum value of the colorbar
        - vmax:         maximum value of the colorbar
        - num_levels:   Number of levels (approximation)
        - log_scale:    If True, use logarithmic sclaing
        - aspect:       Aspect ratio of the plt.Axes (y/x)
        - fontsize:     fontsize

    :returns: A figure with a heatmap
    """
    log_scale = kwargs.get("log_scale", False)
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=kwargs.get("figsize", (30, 10)), dpi=kwargs.get("dpi", 120)
        )
        optional_return_figure = fig
    elif fig is not None and ax is not None:
        optional_return_figure = None
    else:
        msg = "Please provide fig and ax or none of both."
        raise KeyError(msg)
    ax.grid(which="major", color="lightgrey", linestyle="-")
    ax.grid(which="minor", color="0.95", linestyle="--")
    ax.minorticks_on()
    vals = variable.magnitude.transform(data)
    if log_scale:
        vals = data
        vals[vals > 0.0] = np.log10(vals[vals > 0.0])
    else:
        vals = data
    vmin = kwargs.get("vmin", np.nanmin(vals))
    vmax = kwargs.get("vmax", np.nanmax(vals))
    levels = compute_levels(vmin, vmax, kwargs.get("num_levels", 11))
    cmap, norm = get_cmap_norm(levels, variable)

    x_vals = np.arange(vals.shape[1] + 1) if x_vals is None else x_vals
    y_vals = np.arange(vals.shape[0] + 1) if y_vals is None else y_vals
    ax.pcolormesh(x_vals, y_vals, vals, cmap=cmap, norm=norm, zorder=100)
    add_colorbars(fig, ax, variable, levels, cb_pad=0.02)
    if log_scale:
        log_y_labels = [
            rf"$10^{{{t.get_text()}}}$" for t in fig.axes[-1].get_yticklabels()
        ]
        fig.axes[-1].set_yticklabels(log_y_labels)
    update_font_sizes(fig.axes, kwargs.get("fontsize", 32))
    aspect_factor = np.ptp(x_vals) / np.ptp(y_vals)
    ax.set_aspect(kwargs.get("aspect", 0.5) * aspect_factor)
    return optional_return_figure
