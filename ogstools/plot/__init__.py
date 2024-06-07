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
from .utils import clear_labels, label_spatial_axes, update_font_sizes
from .vectorplots import quiver, streamlines

__all__ = [
    "clear_labels",
    "compute_levels",
    "contourf",
    "label_spatial_axes",
    "outline",
    "quiver",
    "setup",
    "shape_on_top",
    "streamlines",
    "subplot",
    "update_font_sizes",
    "utils",
]
