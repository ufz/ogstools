# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""meshplotlib utilities for simple access."""

from .plot_setup import _setup as setup  # noqa: I001: noqa

from .core import plot, plot_isometric, subplot
from .plot_features import plot_on_top

__all__ = [
    "setup",
    "plot",
    "plot_isometric",
    "plot_on_top",
    "subplot",
]
