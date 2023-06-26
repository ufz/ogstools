# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""meshplotlib utilities for simple access."""

from ._plot_setup import PlotSetup
from ._plot_setup import _plot_setup as setup
from .core import plot, plot_isometric, subplot
from .mesh import Mesh
from .mesh_series import MeshSeries

__all__ = [
    "PlotSetup",
    "setup",
    "plot",
    "plot_isometric",
    "subplot",
    "Mesh",
    "MeshSeries",
]
