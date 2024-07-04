# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Plotting utilities for simple access."""

from . import utils
from .contourplots import contourf, subplot
from .features import outline, shape_on_top
from .levels import compute_levels
from .shared import setup
from .vectorplots import quiver, streamlines

__all__ = [
    "compute_levels",
    "contourf",
    "outline",
    "plot_time_slice",
    "quiver",
    "setup",
    "shape_on_top",
    "streamlines",
    "subplot",
    "utils",
]
