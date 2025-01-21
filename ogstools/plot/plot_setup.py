# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Plot configuration setup."""

from dataclasses import dataclass
from typing import Any

from .plot_setup_defaults import setup_dict


@dataclass
class PlotSetup:
    """
    Configuration class for easy plot adjustments.

    Each entry has a default value as listed in
    :obj:`ogstools.plot.plot_setup_defaults`.
    """

    combined_colorbar: bool
    "True if all subplots share on colorbar, else each has its own colorbar."
    custom_cmap: Any | None
    "Custom colormap to use if given"
    dpi: int
    "The resolution (dots per inch) for the figure."
    min_ax_aspect: float | None
    "Minimum aspect ratio of subplots."
    max_ax_aspect: float | None
    "Maximum aspect ratio of subplots."
    invert_colorbar: bool
    "A boolean indicating whether to invert the colorbar."
    layout: str
    "Layout of the figure"
    material_names: dict
    "A dictionary that maps material names to regions (MaterialIDs)."
    num_levels: int
    """The aimed number of levels / bins of the colorbar. See
    :obj:`ogstools.plot.levels`"""
    num_streamline_interp_pts: int | None
    "The number of interpolation points for streamlines."
    vmax: float | None
    "The fixed upper limit for the current scale."
    vmin: float | None
    "The fixed lower limit for the current scale."
    show_element_edges: bool | str
    """Controls the display of element edges, can be a boolean or 'str'. In the
    latter case element edges are always shown for if the name matches the
    variable data name."""
    log_scaled: bool
    "A boolean indicating whether the scaling should be logarithmic."
    show_region_bounds: bool
    "Controls the display of region (MaterialIDs) edges."
    tick_pad: int
    "Padding of tick labels"
    tick_length: int
    "Size of ticks"
    fontsize: float
    "Size for all texts."
    linewidth: float
    "Thickness of lines."
    label_split: int | None
    "Split Variable labels if they exceed this value."
    spatial_unit: str
    "Unit of the spatial dimension."
    time_unit: str
    "Unit of the time dimension."

    @classmethod
    def from_dict(cls: type["PlotSetup"], obj: dict) -> "PlotSetup":
        """Create a PlotSetup instance from a dictionary."""
        return cls(
            min_ax_aspect=obj["min_ax_aspect"],
            max_ax_aspect=obj["max_ax_aspect"],
            invert_colorbar=obj["invert_colorbar"],
            dpi=obj["dpi"],
            num_levels=obj["num_levels"],
            num_streamline_interp_pts=obj["num_streamline_interp_pts"],
            vmin=obj["vmin"],
            vmax=obj["vmax"],
            show_region_bounds=obj["show_region_bounds"],
            show_element_edges=obj["show_element_edges"],
            log_scaled=obj["log_scaled"],
            layout=obj["layout"],
            material_names=obj["material_names"],
            combined_colorbar=obj["combined_colorbar"],
            custom_cmap=obj["custom_cmap"],
            tick_pad=obj["tick_pad"],
            tick_length=obj["tick_length"],
            fontsize=obj["fontsize"],
            linewidth=obj["linewidth"],
            label_split=obj["label_split"],
            spatial_unit=obj["spatial_unit"],
            time_unit=obj["time_unit"],
        )

    def reset(self) -> None:
        """Reset the plot setup to default values."""
        for k, v in self.from_dict(setup_dict).__dict__.items():
            self.__dict__[k] = v

    def set_units(
        self, spatial: str | None = None, time: str | None = None
    ) -> None:
        "Convenience function to update spatial and time unit at once."
        if spatial is not None:
            self.spatial_unit = spatial
        if time is not None:
            self.time_unit = time
