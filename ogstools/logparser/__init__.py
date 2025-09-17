# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""functions used by logparser."""

from .common_ogs_analyses import (
    analysis_convergence_coupling_iteration,
    analysis_convergence_newton_iteration,
    analysis_simulation,
    analysis_simulation_termination,
    analysis_time_step,
    convergence_order_per_ts_iteration,
    errors_per_ts_iteration,
    fill_ogs_context,
    model_and_clock_time,
    time_step_vs_iterations,
)
from .log_parser import parse_file, read_version
from .plots import plot_convergence, plot_convergence_order
from .regexes import ogs_regexes

__all__ = [
    "analysis_convergence_coupling_iteration",
    "analysis_convergence_newton_iteration",
    "analysis_simulation",
    "analysis_simulation_termination",
    "analysis_time_step",
    "convergence_order_per_ts_iteration",
    "errors_per_ts_iteration",
    "fill_ogs_context",
    "model_and_clock_time",
    "ogs_regexes",
    "parse_file",
    "plot_convergence",
    "plot_convergence_order",
    "read_version",
    "time_step_vs_iterations",
]
