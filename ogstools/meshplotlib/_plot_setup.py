"""Plot configuration setup."""


from dataclasses import dataclass
from typing import Literal, Union

from ogstools.propertylib.property import Property, ScalarProperty

from .default_setup import default_setup


@dataclass
class PlotSetup:
    """Configuration setup for the plot."""

    cmap_dict_if_component: dict
    cmap_dict: dict
    cmap_if_mask: list
    default_cmap: str
    dpi: int
    fig_scale: float
    figsize: list[int]
    invert_colorbar: bool
    length: ScalarProperty
    material_names: dict
    num_levels: int
    num_streamline_interp_pts: int
    p_max: float
    p_min: float
    rcParams: dict
    scale_type: Literal["equal", "scaled", "tight", "auto", "image", "square"]
    show_aspect_ratio: bool
    show_element_edges: Union[bool, str]
    title_center: str
    title_left: str
    title_right: str
    x_label: str
    y_label: str
    log_scaled: bool
    """ if a string, element edges are shown if it equals \n
    the current property.data_name, otherwise True or False. """
    show_layer_bounds: bool

    def cmap_str(self, property: Property) -> Union[str, list]:
        """Get the colormap string for a given property."""
        if property.is_mask():
            return self.cmap_if_mask
        if property.is_component():
            if property.data_name in self.cmap_dict_if_component:
                return self.cmap_dict_if_component[property.data_name]
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
            figsize=obj["figsize"],
            invert_colorbar=obj["invert_colorbar"],
            dpi=obj["dpi"],
            num_levels=obj["num_levels"],
            num_streamline_interp_pts=obj["num_streamline_interp_pts"],
            p_min=obj["p_min"],
            p_max=obj["p_max"],
            scale_type=obj["scale_type"],
            show_layer_bounds=obj["show_layer_bounds"],
            show_element_edges=obj["show_element_edges"],
            show_aspect_ratio=obj["show_aspect_ratio"],
            title_center=obj["title_center"],
            title_left=obj["title_left"],
            title_right=obj["title_right"],
            x_label=obj["x_label"],
            y_label=obj["y_label"],
            log_scaled=obj["log_scaled"],
            length=ScalarProperty("", obj["length"][0], obj["length"][1], ""),
            material_names=obj["material_names"],
            cmap_dict=obj["cmap_dict"],
            cmap_dict_if_component=obj["cmap_dict_if_component"],
            cmap_if_mask=obj["cmap_if_mask"],
            default_cmap=obj["default_cmap"],
            rcParams=obj["rcParams"],
        )

    def reset(self) -> None:
        """Reset the plot setup to default values."""
        for k, v in self.from_dict(default_setup).__dict__.items():
            self.__dict__[k] = v


_plot_setup = PlotSetup.from_dict(default_setup)
