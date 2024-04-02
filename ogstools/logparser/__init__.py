# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""functions used by logparser."""

from .common_ogs_analyses import (
    analysis_convergence_coupling_iteration,
    analysis_convergence_newton_iteration,
    analysis_simulation,
    analysis_simulation_termination,
    analysis_time_step,
    fill_ogs_context,
    time_step_vs_iterations,
)
from .log_parser import parse_file
from .regexes import ogs_regexes

__all__ = [
    "parse_file",
    "analysis_convergence_coupling_iteration",
    "analysis_simulation",
    "analysis_convergence_newton_iteration",
    "analysis_simulation_termination",
    "analysis_time_step",
    "fill_ogs_context",
    "time_step_vs_iterations",
    "ogs_regexes",
]
