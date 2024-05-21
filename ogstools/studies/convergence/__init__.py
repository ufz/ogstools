# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""functions to generate a convergence study."""

from .convergence import (
    add_grid_spacing,
    convergence_metrics,
    convergence_metrics_evolution,
    grid_convergence,
    log_fit,
    plot_convergence,
    plot_convergence_error_evolution,
    plot_convergence_errors,
    plot_convergence_order_evolution,
    richardson_extrapolation,
)
from .study import run_convergence_study

__all__ = [
    "add_grid_spacing",
    "convergence_metrics",
    "convergence_metrics_evolution",
    "grid_convergence",
    "log_fit",
    "plot_convergence",
    "plot_convergence_errors",
    "plot_convergence_error_evolution",
    "plot_convergence_order_evolution",
    "richardson_extrapolation",
    "run_convergence_study",
]
