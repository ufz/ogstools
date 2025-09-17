# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Plotting utilities for simple access."""

from .shared import setup  # noqa: I001
from . import utils
from .animation import animate
from .contourplots import contourf, subplot
from .features import shape_on_top
from .heatmaps import heatmap
from .levels import compute_levels
from .lineplots import line
from .contourplots_pv import contourf_pv
from .vectorplots import quiver, streamlines

__all__ = [
    "animate",
    "compute_levels",
    "contourf",
    "contourf_pv",
    "heatmap",
    "line",
    "quiver",
    "setup",
    "shape_on_top",
    "streamlines",
    "subplot",
    "utils",
]
