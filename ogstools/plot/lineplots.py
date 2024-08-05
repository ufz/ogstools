from string import ascii_uppercase
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.meshlib.data_processing import sample_polyline
from ogstools.plot import contourf, setup, utils
from ogstools.plot.shared import spatial_quantity
from ogstools.propertylib.properties import Property, get_preset


# TODO: ability to swap x and y?
def linesample(
    mesh: pv.UnstructuredGrid,
    x: str,
    y_property: str | Property,
    profile_points: np.ndarray,
    ax: plt.Axes,
    resolution: int | None = 100,
    grid: Literal["major", "both", None] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plot selected properties obtained from sample_over_polyline function,
    this function calls to it internally. Values provided in param x and y
    refer to columns of the DataFrame returned by it.

    :param mesh: mesh to sample from.
    :param x: Value to be used on x-axis of the plot
    :param y_property: Values to be used on y-axis of the plot
    :param profile_points: Points defining the profile (and its segments)
    :param ax: User-created array of Matplotlib axis object
    :param resolution: Resolution of the sampled profile. Total number of
        points within all profile segments.
    :param resolution: Resolution of the sampled profile. Total number of
        points within all profile segments.
    :param grid: Which gridlines should be drawn?
    :param kwargs: Optional keyword arguments passed to matplotlib.pyplot.plot
        to customize plot options like a line label (for auto legends), linewidth,
        antialiasing, marker face color.

    :return: Matplotlib Axes object
    """

    mesh_property = get_preset(y_property, mesh)
    mesh_sp, _ = sample_polyline(
        mesh, mesh_property, profile_points, resolution
    )

    assert isinstance(ax, plt.Axes)

    spatial_qty = spatial_quantity(mesh)
    kwargs.setdefault("label", mesh_property.data_name)
    kwargs.setdefault("color", mesh_property.color)
    kwargs.setdefault("linestyle", mesh_property.linestyle)
    if "ls" in kwargs:
        kwargs.pop("linestyle")

    utils.update_font_sizes(axes=ax, fontsize=kwargs.pop("fontsize", 20))
    ax.plot(
        spatial_qty.transform(mesh_sp[x]),
        mesh_sp[mesh_property.data_name],
        **kwargs,
    )
    ax.set_xlabel("Profile distance / " + spatial_qty.output_unit)
    ax.set_ylabel(mesh_property.get_label(setup.label_split))

    if grid in ["both", "major"]:
        ax.grid(which="major", color="lightgrey", linestyle="-")
    if grid == "major":
        ax.minorticks_off()
    if grid == "both":
        ax.grid(which="minor", color="0.95", linestyle="--")
        ax.minorticks_on()

    return ax


def linesample_contourf(
    mesh: pv.UnstructuredGrid,
    properties: str | list | Property,
    profile_points: np.ndarray,
    resolution: int | None = None,
    plot_nodal_pts: bool | None = True,
    nodal_pts_labels: str | list | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Default plot for the data obtained from sampling along a profile on a mesh.

    :param mesh: mesh to plot and sample from.
    :param properties: Properties to be read from the mesh
    :param profile_points: Points defining the profile (and its segments)
    :param resolution: Resolution of the sampled profile. Total number of \
        points within all profile segments.
    :param plot_nodal_pts: Plot and annotate all nodal points in profile
    :param nodal_pts_labels: Labels for nodal points (only use if \
        plot_nodal_points is set to True)

    :return: Tuple containing Matplotlib Figure and Axis objects
    """
    # TODO: Add support for plotting only geometry at top subplot and
    # lineplot with twinx in the bottom one
    if not isinstance(properties, list):
        properties = [properties]

    _, dist_at_knot = sample_polyline(
        mesh, properties, profile_points, resolution=resolution
    )

    fig, ax = plt.subplots(
        2, len(properties), figsize=(len(properties) * 13, 12), squeeze=False
    )
    spatial_qty = spatial_quantity(mesh)
    x_id, y_id, _, _ = utils.get_projection(mesh)
    for index, mesh_property in enumerate(properties):
        contourf(mesh, mesh_property, fig=fig, ax=ax[0, index])
        linesample(
            mesh,
            x="dist",
            y_property=mesh_property,
            profile_points=profile_points,
            ax=ax[1, index],
            resolution=resolution,
            grid="both",
        )

        if plot_nodal_pts:
            if nodal_pts_labels is None:
                nodal_pts_labels = list(
                    ascii_uppercase[0 : len(profile_points)]
                )
            ax[0][index].plot(
                spatial_qty.transform(profile_points[:, x_id]),
                spatial_qty.transform(profile_points[:, y_id]),
                "-*",
                linewidth=2,
                markersize=7,
                color="orange",
            )
            for nodal_pt_id, nodal_pt in enumerate(dist_at_knot):
                xy = profile_points[nodal_pt_id, [x_id, y_id]]
                text_xy = utils.padded(ax[0][index], *spatial_qty.transform(xy))
                ax[0][index].text(
                    *text_xy,
                    nodal_pts_labels[nodal_pt_id],
                    color="orange",
                    fontsize=setup.fontsize,
                    ha="center",
                    va="center",
                )
                ax[1][index].axvline(
                    spatial_qty.transform(nodal_pt),
                    linestyle="--",
                    color="orange",
                    linewidth=2,
                )
            ax_twiny = ax[1][index].twiny()
            ax_twiny.set_xlim(ax[1][index].get_xlim())
            ax_twiny.set_xticks(
                spatial_qty.transform(dist_at_knot),
                nodal_pts_labels,
                color="orange",
            )
    utils.update_font_sizes(fig.axes)
    fig.tight_layout()

    return fig, ax
