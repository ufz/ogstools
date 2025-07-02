# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ogstools.plot import heatmap
from ogstools.variables import Scalar

from .common_ogs_analyses import (
    convergence_order_per_ts_iteration,
    errors_per_ts_iteration,
    model_and_clock_time,
)


def _format_fig(
    fig: plt.Figure, x_ticks: np.ndarray, y_ticks: np.ndarray, x_label: str
) -> plt.Figure:
    fig.axes[0].set_xticks(x_ticks, minor=True)
    fig.axes[0].set_yticks(y_ticks, minor=True)
    fig.axes[0].set_xlabel(x_label.replace("_", " "))
    fig.axes[0].set_ylabel("iteration")
    fig.tight_layout()
    return fig


def _x_data(
    df: pd.DataFrame,
    x_metric: Literal["time_step", "model_time"],
    default_len_x: int,
) -> tuple[np.ndarray | None, range]:
    if x_metric == "time_step":
        return None, range(default_len_x)
    df_time = model_and_clock_time(df).reset_index()
    df_time = df_time.pivot_table(
        "model_time", "time_step", aggfunc="last"
    ).reset_index()
    x_vals = df_time[x_metric].to_numpy()
    x_ticks = 0.5 * (x_vals[1:] + x_vals[:-1])
    if len(x_vals) == default_len_x:
        # TODO: Not sure why, but sometimes the required length of the x_vals
        # is off by 1 compared to the required length to plot a heatmap with
        # errors or orders. The following fixes this but the underlying problem
        # is not yet understood.
        x_vals = np.append(0.0, x_vals)
    return x_vals, x_ticks


def plot_convergence_order(
    df: pd.DataFrame,
    n: Literal[3, 4] = 3,
    x_metric: Literal["time_step", "model_time"] = "time_step",
    **kwargs: Any,
) -> plt.Figure:
    """
    Create a heatmap of the nonlinear solver convergence order.

    see: :py:func:`~ogstools.logparser.common_ogs_analyses.convergence_order_per_ts_iteration`
    In order to estimate the convergence order we need to take into account
    multiple values and thus cannot assign each iteration a convergence order.
    Only for iterations `i` of `i  >= n` an order is calculated and plotted.
    Per default the scale is limited to a range of 0 to 2 to limit the view to
    meaningful data. Set the keyword arguments `vmin` and `vmax`to `None` to
    see the entire scale.

    :param df:          Dataframe of a simulation log.
    :param n:           Number of error values to use to estimate the
                        convergence order.
    :param x_metric:    x_axis can represent either "time_step" or "model_time"

    Keyword Arguments:
        - see: :py:func:`~ogstools.plot.heatmaps.heatmap`)

    :returns: A figure with a heatmap of the nonlinear solver convergence order.
    """
    orders = convergence_order_per_ts_iteration(df, n=n)
    x_vals, x_ticks = _x_data(df, x_metric, orders.shape[1])
    y_ticks = range(orders.shape[0])
    order_var = Scalar("convergence_order", cmap="RdBu", symbol="q")
    kwargs.setdefault("vmin", 0)
    kwargs.setdefault("vmax", 2)
    res = heatmap(orders, order_var, x_vals=x_vals, **kwargs)
    fig = kwargs.get("fig", res)
    return _format_fig(fig, x_ticks, y_ticks, x_label=x_metric)


def plot_convergence(
    df: pd.DataFrame,
    metric: Literal["dx", "dx_x", "x"] = "dx",
    x_metric: Literal["time_step", "model_time"] = "time_step",
    **kwargs: Any,
) -> plt.Figure:
    """
    Create a heatmap of the nonlinear solver convergence data.

    The individual values in the heatmap correspond to the top right indices on
    the x- and y-axis. E.g. the very first entry which fills the space between
    timesteps 0-1 and iteration 0-1 belongs to the first iteration of the first
    timestep. Thus we immediately read on which iteration a timestep converged
    and on which timestep the simulation ended. Per default logarithmic scaling
    is used. Set `log_scale` to `False` to use linear scaling.

    :param df:          Dataframe of a simulation log.
    :param metric:      Which metric / column of the Dataframe to plot.
                        dx (absolute error), dx_x (relative error), x (residual)
    :param x_metric:    x_axis can represent either "time_step" or "model_time"

    Keyword Arguments:
        - see: :py:func:`~ogstools.plot.heatmaps.heatmap`)

    :returns: A figure with a heatmap of the nonlinear solver convergence data.
    """
    errors = errors_per_ts_iteration(df, metric)
    x_vals, x_ticks = _x_data(df, x_metric, errors.shape[1])
    y_ticks = range(errors.shape[0])
    names = {"dx": "absolute error", "dx_x": "relative error", "x": "residual"}
    symbol = str(metric).replace("_", " / ")
    err_var = Scalar(names[metric], cmap="viridis", symbol=symbol)
    kwargs.setdefault("log_scale", True)
    res = heatmap(errors, err_var, x_vals=x_vals, **kwargs)
    fig = kwargs.get("fig", res)
    return _format_fig(fig, x_ticks, y_ticks, x_label=x_metric)
