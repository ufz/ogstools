# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd

from .common_ogs_analyses import (
    analysis_convergence_coupling_iteration,
    analysis_convergence_newton_iteration,
    analysis_simulation,
    analysis_simulation_termination,
    analysis_time_step,
    fill_ogs_context,
)
from .log_parser import parse_file


class Log:
    """
    Parser and analyzer for OGS simulation log files.

    Parses OGS log files and provides methods to extract and analyze
    simulation data including convergence behavior, time step information,
    and simulation status.
    """

    def __init__(self, file: str | Path):
        """
        Initialize a Log parser.

        :param file: Path to the OGS log file to parse.
        """
        self.file = Path(file)
        records = parse_file(self.file)
        self.df_records = pd.DataFrame(records)
        self.df_log = fill_ogs_context(self.df_records)

    def convergence_coupling_iteration(self) -> pd.DataFrame:
        """
        Extract coupling iteration convergence data.

        :returns: DataFrame with coupling iteration convergence information.
        """
        return analysis_convergence_coupling_iteration(self.df_log)

    def convergence_newton_iteration(self) -> pd.DataFrame:
        """
        Extract Newton iteration convergence data.

        :returns: DataFrame with Newton iteration convergence metrics
                  (errors, residuals, convergence order).
        """
        return analysis_convergence_newton_iteration(self.df_log)

    def simulation(self) -> pd.DataFrame:
        """
        Extract overall simulation information.

        :returns: DataFrame with simulation-level data.
        """
        return analysis_simulation(self.df_log)

    def simulation_termination(self) -> pd.DataFrame:
        """
        Extract simulation termination information.

        :returns: DataFrame with termination status and final state.
        """
        return analysis_simulation_termination(self.df_log)

    def time_step(self) -> pd.DataFrame:
        """
        Extract time step information.

        :returns: DataFrame with time step data (model time, clock time, etc.).
        """
        return analysis_time_step(self.df_log)

    def plot_convergence(
        self,
        metric: Literal["dx", "dx_x", "x"] = "dx",
        x_metric: Literal["time_step", "model_time"] = "time_step",
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Create a heatmap of the nonlinear solver convergence data.

        Visualizes convergence behavior across time steps and iterations.
        Per default uses logarithmic scaling.

        :param metric:      Which metric to plot. Options:
                            - "dx": absolute error
                            - "dx_x": relative error
                            - "x": residual
        :param x_metric:    x_axis representation:
                            - "time_step": timestep number
                            - "model_time": simulation time
        :param kwargs:      Additional arguments passed to heatmap function
                            (see :func:`~ogstools.plot.heatmaps.heatmap`).

        :returns: A matplotlib Figure with the convergence heatmap.
        """
        from .plots import plot_convergence

        return plot_convergence(
            self.df_log, metric=metric, x_metric=x_metric, **kwargs
        )

    def plot_convergence_order(
        self,
        n: Literal[3, 4] = 3,
        x_metric: Literal["time_step", "model_time"] = "time_step",
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Create a heatmap of the nonlinear solver convergence order.

        Estimates and visualizes the convergence order across iterations.
        Only iterations i >= n are assigned a convergence order.

        :param n:           Number of error values to use for estimating
                            convergence order (3 or 4).
        :param x_metric:    x_axis representation:
                            - "time_step": timestep number
                            - "model_time": simulation time
        :param kwargs:      Additional arguments passed to heatmap function.
                            Default scale is limited to 0-2 for meaningful data.

        :returns: A matplotlib Figure with the convergence order heatmap.
        """
        from .plots import plot_convergence_order

        return plot_convergence_order(
            self.df_log, n=n, x_metric=x_metric, **kwargs
        )
