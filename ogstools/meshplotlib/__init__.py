# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""meshplotlib utilities for simple access."""

from .plot_setup import _setup as setup  # noqa: I001: noqa

from .core import (
    plot_probe,
    plot,
    subplot,
    update_font_sizes,
    label_spatial_axes,
    clear_labels,
    color_twin_axes,
)
from .plot_features import plot_contour, plot_on_top, plot_profile, lineplot

__all__ = [
    "setup",
    "plot",
    "plot_contour",
    "plot_on_top",
    "plot_probe",
    "subplot",
    "update_font_sizes",
    "label_spatial_axes",
    "clear_labels",
    "plot_profile",
    "lineplot",
    "color_twin_axes",
]
