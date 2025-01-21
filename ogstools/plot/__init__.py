# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Plotting utilities for simple access."""

from .shared import setup  # noqa: I001
from . import utils
from .contourplots import contourf, subplot
from .features import shape_on_top
from .levels import compute_levels
from .lineplots import line
from .vectorplots import quiver, streamlines

__all__ = [
    "compute_levels",
    "contourf",
    "line",
    "quiver",
    "setup",
    "shape_on_top",
    "streamlines",
    "subplot",
    "utils",
]
