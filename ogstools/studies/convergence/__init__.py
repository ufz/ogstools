# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""functions to generate a convergence study."""

from . import examples
from .convergence import (
    add_grid_spacing,
    convergence_metrics,
    grid_convergence,
    log_fit,
    plot_convergence,
    plot_convergence_errors,
    richardson_extrapolation,
)
from .study import run_convergence_study

__all__ = [
    "add_grid_spacing",
    "convergence_metrics",
    "examples",
    "grid_convergence",
    "log_fit",
    "plot_convergence",
    "plot_convergence_errors",
    "richardson_extrapolation",
    "run_convergence_study",
]
