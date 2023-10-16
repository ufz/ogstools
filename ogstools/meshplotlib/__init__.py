# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""meshplotlib utilities for simple access."""

from .plot_setup import _setup as setup  # noqa: I001: noqa

from .core import plot, subplot
from .plot_features import plot_contour, plot_on_top

__all__ = [
    "setup",
    "plot",
    "plot_on_top",
    "plot_contour",
    "subplot",
]
