"""Plot configuration setup."""


from dataclasses import dataclass
from typing import Union

from matplotlib.colors import Colormap

from ogstools.propertylib.property import Property, Scalar

from .plot_setup_defaults import setup_dict


@dataclass
class PlotSetup:
    """
    Configuration class for easy plot adjustments.

    Each entry has a default value as listed in
    :obj:`ogstools.meshplotlib.plot_setup_defaults`.
    """

    combined_colorbar: bool
    "True if all subplots share on colorbar, else each has its own colorbar."
    custom_cmap: Colormap
    "If provided, this colormap will be used for any plot."
    cmap_dict_if_bilinear: dict
    "A dictionary that maps bilinear colormaps to properties."
    cmap_dict: dict
    "A dictionary that maps colormaps to properties."
    cmap_if_mask: list
    "A list of colors corresponding to [True, False] values of masks."
    default_cmap: str
    "The default colormap to use."
    dpi: int
    "The resolution (dots per inch) for the figure."
    fig_scale: float
    "A scaling factor for the figure."
    ax_aspect_limits: tuple[float, float]
    "Lower and upper limit of the ax aspect ratio. For meshes with data ratios"
    "outside these bounds the aspect ratio gets clamped. Inside these bounds,"
    "the axes keep an aspect ratio of 1 (units are equally long on both axes)."
    fig_aspect_limits: tuple[float, float]
    "Lower and upper limit of the figure aspect ratio. If a figure would exceed"
    "them, they get clipped to prevent overly long or wide figures."
    invert_colorbar: bool
    "A boolean indicating whether to invert the colorbar."
    layout: str
    "Layout of the figure"
    length: Scalar
    "A property to set data and output unit of a models spatial extension."
    material_names: dict
    "A dictionary that maps material names to regions (MaterialIDs)."
    num_levels: int
    """The aimed number of levels / bins of the colorbar. See
    :obj:`ogstools.meshplotlib.levels`"""
    num_streamline_interp_pts: int
    "The number of interpolation points for streamlines."
    p_max: float
    "The fixed upper limit for the current scale."
    p_min: float
    "The fixed lower limit for the current scale."
    rcParams: dict
    """Matplotlib runtime configuration. See
    :obj:`ogstools.meshplotlib.plot_setup_defaults`"""
    show_element_edges: Union[bool, str]
    """Controls the display of element edges, can be a boolean or 'str'. In the
    latter case element edges are always shown for if the name matches the
    property data name."""
    title_center: str
    "The center part of the plot's title."
    title_left: str
    "The left part of the plot's title."
    title_right: str
    "The right part of the plot's title."
    x_label: str
    "The label for the x-axis."
    y_label: str
    "The label for the y-axis."
    log_scaled: bool
    "A boolean indicating whether the scaling should be logarithmic."
    show_region_bounds: bool
    "Controls the display of region (MaterialIDs) edges."
    embedded_region_names_color: str
    "Color of the embedded region names inside the plot."

    def cmap_str(self, property: Property) -> Union[str, list]:
        """Get the colormap string for a given property."""
        if property.is_mask():
            return self.cmap_if_mask
        if property.bilinear_cmap:
            if property.data_name in self.cmap_dict_if_bilinear:
                return self.cmap_dict_if_bilinear[property.data_name]
        elif property.data_name in self.cmap_dict:
            return self.cmap_dict[property.data_name]
        return self.default_cmap

    @property
    def rcParams_scaled(self) -> dict:
        """Get the scaled rcParams values."""
        params = self.rcParams
        for k, v in self.rcParams.items():
            if isinstance(v, int):
                params[k] = v * self.fig_scale
        return params

    @classmethod
    def from_dict(cls: type["PlotSetup"], obj: dict):
        """Create a PlotSetup instance from a dictionary."""
        return cls(
            fig_scale=obj["fig_scale"],
            ax_aspect_limits=obj["ax_aspect_limits"],
            fig_aspect_limits=obj["fig_aspect_limits"],
            invert_colorbar=obj["invert_colorbar"],
            dpi=obj["dpi"],
            num_levels=obj["num_levels"],
            num_streamline_interp_pts=obj["num_streamline_interp_pts"],
            p_min=obj["p_min"],
            p_max=obj["p_max"],
            show_region_bounds=obj["show_region_bounds"],
            show_element_edges=obj["show_element_edges"],
            embedded_region_names_color=obj["embedded_region_names_color"],
            title_center=obj["title_center"],
            title_left=obj["title_left"],
            title_right=obj["title_right"],
            x_label=obj["x_label"],
            y_label=obj["y_label"],
            log_scaled=obj["log_scaled"],
            layout=obj["layout"],
            length=Scalar("", obj["length"][0], obj["length"][1], ""),
            material_names=obj["material_names"],
            combined_colorbar=obj["combined_colorbar"],
            custom_cmap=obj["custom_cmap"],
            cmap_dict=obj["cmap_dict"],
            cmap_dict_if_bilinear=obj["cmap_dict_if_bilinear"],
            cmap_if_mask=obj["cmap_if_mask"],
            default_cmap=obj["default_cmap"],
            rcParams=obj["rcParams"],
        )

    def reset(self) -> None:
        """Reset the plot setup to default values."""
        for k, v in self.from_dict(setup_dict).__dict__.items():
            self.__dict__[k] = v


_setup = PlotSetup.from_dict(setup_dict)
