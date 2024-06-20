# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Plotting core utilitites."""

import warnings

import numpy as np
import pyvista as pv
from matplotlib import cm as mcm
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle as Rect

from ogstools.plot import utils
from ogstools.plot.levels import combined_levels, level_boundaries
from ogstools.plot.utils import get_cmap_norm
from ogstools.propertylib.properties import Property, Vector, get_preset

from . import features
from .levels import compute_levels, median_exponent
from .shared import setup, spatial_quantity
from .vectorplots import streamlines

# TODO: define default data_name for regions in setup


def _q_zero_line(mesh_property: Property, levels: np.ndarray) -> bool:
    return mesh_property.bilinear_cmap or (
        mesh_property.data_name == "temperature" and levels[0] < 0 < levels[-1]
    )


def get_ticklabels(ticks: np.ndarray) -> tuple[list[str], str | None]:
    """Get formatted tick labels and optional offset str.

    If all values in ticks are too close together offset notation is used.
    """
    if median_exponent(ticks) >= 2 + median_exponent(ticks[-1] - ticks[0]):
        # use offset notation
        label_lens = np.asarray([len(str(tick)) for tick in ticks])
        offset = ticks[np.argmin(label_lens)]
    else:
        offset = 0
    if np.issubdtype(ticks.dtype, np.integer):
        return [str(tick) for tick in ticks], (
            None if offset == 0 else f"{offset:g}"
        )

    for precision in [1, 2, 3, 4]:
        fmt = "f" if abs(median_exponent(ticks - offset)) <= 2 else "e"
        tick_labels: list[str] = [
            f"{0.0 + tick:.{precision}{fmt}}" for tick in ticks - offset
        ]
        if len(tick_labels) == len(set(tick_labels)):
            break

    # pretty hacky but seems to do the job
    for idx, adj in [(0, 1), (-1, -2)]:
        if float(tick_labels[idx]) != float(tick_labels[adj]):
            continue
        for precision in range(12):
            new_ticklabel = f"{0.0 + ticks[idx] - offset:.{precision}{fmt}}"
            adj_ticklabel = f"{0.0 + ticks[adj] - offset:.{precision}{fmt}}"
            if float(new_ticklabel) != float(adj_ticklabel):
                tick_labels[idx] = new_ticklabel
                break
    if fmt != "e":
        tick_labels = [label.rstrip("0").rstrip(".") for label in tick_labels]
    return tick_labels, None if offset == 0 else f"{offset:g}"


def add_colorbars(
    fig: plt.Figure,
    ax: plt.Axes | list[plt.Axes],
    mesh_property: Property,
    levels: np.ndarray,
    pad: float = 0.05,
    labelsize: float | None = None,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    ticks = levels
    if mesh_property.categoric or (len(levels) == 2):
        bounds = level_boundaries(levels)
        ticks = bounds[:-1] + 0.5 * np.diff(bounds)

    cmap, norm = get_cmap_norm(levels, mesh_property)
    cm = mcm.ScalarMappable(norm=norm, cmap=cmap)

    cb = fig.colorbar(
        cm,
        norm=norm,
        ax=ax,
        ticks=ticks,
        drawedges=True,
        location="right",
        spacing="uniform",
        pad=pad,  # fmt: skip
    )
    # Formatting the colorbar label and ticks

    tick_labels, offset = get_ticklabels(ticks)
    cb_label = mesh_property.get_label()
    if offset is not None:
        if offset[0] == "-":
            cb_label += " + " + offset[1:]
        else:
            cb_label += " - " + offset
    if setup.log_scaled:
        cb_label = f"log$_{{10}}$( {cb_label} )"
    labelsize = (
        setup.rcParams_scaled["font.size"] if labelsize is None else labelsize
    )
    cb.set_label(cb_label, size=labelsize)

    # special formatting for MaterialIDs
    if (
        mesh_property.data_name == "MaterialIDs"
        and setup.material_names is not None
    ):
        tick_labels = [
            setup.material_names.get(mat_id, mat_id) for mat_id in levels
        ]
        cb.ax.set_ylabel("")
    elif mesh_property.categoric:
        tick_labels = [str(level) for level in levels.astype(int)]
    cb.ax.tick_params(labelsize=labelsize, direction="out")
    cb.ax.set_yticklabels(tick_labels)

    # miscellaneous

    if mesh_property.is_mask():
        cb.ax.add_patch(Rect((0, 0.5), 1, -1, lw=0, fc="none", hatch="/"))
    if setup.invert_colorbar:
        cb.ax.invert_yaxis()
    if _q_zero_line(mesh_property, ticks):
        cb.ax.axhline(
            y=0, color="w", lw=2 * setup.rcParams_scaled["lines.linewidth"]
        )


def subplot(
    mesh: pv.UnstructuredGrid,
    mesh_property: Property | str,
    ax: plt.Axes,
    levels: np.ndarray | None = None,
) -> None:
    "Plot the property field of a mesh on a matplotlib.axis."

    mesh_property = get_preset(mesh_property, mesh)
    if mesh.get_cell(0).dimension == 3:
        msg = "This method is for 2D meshes only, but found 3D elements."
        raise ValueError(msg)

    ax.axis("auto")

    if mesh_property.mask_used(mesh):
        subplot(mesh, mesh_property.get_mask(), ax)
        mesh = mesh.ctp(True).threshold(
            value=[1, 1], scalars=mesh_property.mask
        )

    surf_tri = mesh.triangulate().extract_surface()

    # get projection
    x_id, y_id = utils.get_projection(mesh)
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    spatial = spatial_quantity(surf_tri)
    x, y = spatial.transform(surf_tri.points.T[[x_id, y_id]])
    tri = surf_tri.faces.reshape((-1, 4))[:, 1:]
    values = mesh_property.magnitude.transform(surf_tri)
    if setup.log_scaled:
        values_temp = np.where(values > 1e-14, values, 1e-14)
        values = np.log10(values_temp)
    p_min, p_max = np.nanmin(values), np.nanmax(values)

    if levels is None:
        num_levels = min(setup.num_levels, len(np.unique(values)))
        levels = compute_levels(p_min, p_max, num_levels)
    cmap, norm = get_cmap_norm(levels, mesh_property)

    if mesh_property.data_name in mesh.point_data:
        ax.tricontourf(  # type: ignore[call-overload]
            x,
            y,
            tri,
            values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        if _q_zero_line(mesh_property, levels):
            ax.tricontour(  # type: ignore[call-overload]
                x, y, tri, values, levels=[0], colors="w"
            )
    else:
        ax.tripcolor(x, y, tri, facecolors=values, cmap=cmap, norm=norm)
        if mesh_property.is_mask():
            ax.tripcolor(x, y, tri, facecolors=values, mask=(values == 1),
                         cmap=cmap, norm=norm, hatch="/")  # fmt: skip

    surf = mesh.extract_surface()

    show_edges = setup.show_element_edges
    if isinstance(setup.show_element_edges, str):
        show_edges = setup.show_element_edges == mesh_property.data_name
    if show_edges:
        features.element_edges(ax, surf, projection)

    if setup.show_region_bounds and "MaterialIDs" in mesh.cell_data:
        features.layer_boundaries(ax, surf, projection)

    if isinstance(mesh_property, Vector):
        streamlines(surf_tri, ax, mesh_property, projection)

    ax.margins(0, 0)  # otherwise it shrinks the plot content

    if abs(max(mean_normal) - 1) > 1e-6:
        sec_id = np.argmax(np.delete(mean_normal, projection))
        sec_labels = []
        for tick in ax.get_xticks():
            origin = np.array(mesh.center)
            origin[sec_id] = min(
                max(tick, mesh.bounds[2 * sec_id] + 1e-6),
                mesh.bounds[2 * sec_id + 1] - 1e-6,
            )
            sec_mesh = mesh.slice("xyz"[sec_id], origin)
            if sec_mesh.n_cells:
                sec_labels += [f"{sec_mesh.bounds[2 * projection]:.1f}"]
            else:
                sec_labels += [""]
        # TODO: use a function to make this short
        secax = ax.secondary_xaxis("top")
        secax.xaxis.set_major_locator(
            mticker.FixedLocator(list(ax.get_xticks()))
        )
        secax.set_xticklabels(sec_labels)
        secax.set_xlabel(f'{"xyz"[projection]} / {spatial.output_unit}')


# TODO: fixed_figure_size -> ax aspect automatic


def _fig_init(
    rows: int, cols: int, aspect: float = 1.0
) -> tuple[plt.Figure, plt.Axes]:
    nx_cb = 1 if setup.combined_colorbar else cols
    default_size = 8
    cb_width = 3
    y_label_width = 2
    x_label_height = 1
    figsize = setup.fig_scale * np.asarray(
        [
            default_size * cols * aspect + cb_width * nx_cb + y_label_width,
            default_size * rows + x_label_height,
        ]
    )
    fig, ax = plt.subplots(
        rows,
        cols,
        dpi=setup.dpi * setup.fig_scale,
        figsize=figsize,
        layout=setup.layout,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(1)
    return fig, ax


# TODO: Have a look at fig and ax logic and make it more readable


def draw_plot(
    meshes: list[pv.UnstructuredGrid] | np.ndarray | pv.UnstructuredGrid,
    mesh_property: Property,
    fig: plt.Figure | None = None,
    axes: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Plot the property field of meshes on existing figure.

    :param meshes: Singular mesh of 2D numpy array of meshes
    :param property: the property field to be visualized on all meshes
    :param fig: Matplotlib figure to use for plotting (optional)
    :param axes: Matplotlib Axes to use for plotting (optional)
    """
    shape = utils.get_rows_cols(meshes)
    np_meshes = np.reshape(meshes, shape)
    if fig is not None and axes is not None:
        np_axs = np.reshape(np.array(axes), shape)
    elif fig is not None and axes is None:
        # Only Fig is given
        # Multiple meshes should be accepted
        warnings.warn(
            "This is not a good practice. Consider providing both fig and ax instead. This option may lead to unexpected behaviour and may be removed without warning in the future.",
            Warning,
            stacklevel=4,
        )
        np_axs = np.reshape(np.asarray(fig.axes), shape)
    elif fig is None and axes is not None:
        # Only ax is given
        # Only one mesh should be accepted
        if shape != (1, 1):
            msg = "You have provided only one Axis object but multiple meshes. Provide only one mesh per Axis object, or provide Figure object instead."
            raise ValueError(msg)
        np_axs = np.reshape(np.array(axes), (1, 1))
    else:
        msg = "Neither Figure nor Axis object was provided."
        raise TypeError(msg)
    if setup.combined_colorbar:
        levels_combined = combined_levels(np_meshes, mesh_property)
    for i in range(shape[0]):
        for j in range(shape[1]):
            _levels = (
                levels_combined
                if setup.combined_colorbar
                else combined_levels(np_meshes[i, j, None], mesh_property)
            )
            subplot(np_meshes[i, j], mesh_property, np_axs[i, j], _levels)

    # One mesh is sufficient, it should be the same for all of them
    x_id, y_id = utils.get_projection(np_meshes[0, 0])
    utils.label_spatial_axes(np_axs, "xyz"[x_id], "xyz"[y_id])
    np_axs[0, 0].set_title(setup.title_center, loc="center", y=1.02)
    np_axs[0, 0].set_title(setup.title_left, loc="left", y=1.02)
    np_axs[0, 0].set_title(setup.title_right, loc="right", y=1.02)
    # make extra space for the upper limit of the colorbar
    if setup.layout == "tight":
        plt.tight_layout(pad=1.4)
    if setup.combined_colorbar:
        if fig is None:
            warnings.warn(
                "Cannot plot combined colorbar if Figure object is not provided!",
                Warning,
                stacklevel=4,
            )
        else:
            cb_axs = np.ravel(np.asarray(fig.axes)).tolist()
            add_colorbars(
                fig, cb_axs, mesh_property, levels_combined, pad=0.05 / shape[1]
            )
    else:
        # TODO: restructure this logic
        if fig is None:
            warnings.warn(
                "Figure object is required to plot individual colorbars for Axes objects.",
                Warning,
                stacklevel=4,
            )
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    _levels = combined_levels(
                        np_meshes[i, j, None], mesh_property
                    )
                    add_colorbars(
                        fig,
                        np_axs[i, j],
                        mesh_property,
                        _levels,
                        pad=0.05 / shape[1],
                    )
    return fig


# TODO: add as arguments: cmap, limits
# TODO: num_levels should be min_levels
# TODO: split for single mesh plot and multi mesh plot
def contourf(
    meshes: list[pv.UnstructuredGrid] | np.ndarray | pv.UnstructuredGrid,
    mesh_property: Property | str,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Plot the property field of meshes with default settings.

    The resulting figure adheres to the configurations in plot.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes:      Singular mesh of 2D numpy array of meshes
    :param property:    The property field to be visualized on all meshes
    :param fig: Matplotlib Figure object to use for plotting (optional)
    :param ax: Matplotlib Axis object to use for plotting (optional)
    """
    rcParams.update(setup.rcParams_scaled)
    shape = utils.get_rows_cols(meshes)
    _meshes = np.reshape(meshes, shape).ravel()
    mesh_property = get_preset(mesh_property, _meshes[0])
    data_aspects = np.asarray([utils.get_data_aspect(mesh) for mesh in _meshes])
    if setup.min_ax_aspect is None and setup.max_ax_aspect is None:
        fig_aspect = np.mean(data_aspects)
    else:
        fig_aspect = np.mean(
            np.clip(data_aspects, setup.min_ax_aspect, setup.max_ax_aspect)
        )
    ax_aspects = fig_aspect / data_aspects
    n_axs = shape[0] * shape[1]
    if ax is None and fig is None:
        _fig, _ax = _fig_init(rows=shape[0], cols=shape[1], aspect=fig_aspect)
        fig = draw_plot(meshes, mesh_property, fig=_fig, axes=_ax)
        assert isinstance(fig, plt.Figure)
        for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects, strict=False):
            ax.set_aspect(1.0 / aspect)
    elif ax is not None and fig is None:
        draw_plot(meshes, mesh_property, axes=ax)
        ax.set_aspect(1.0 / ax_aspects[0])
    elif ax is None and fig is not None:
        fig = draw_plot(meshes, mesh_property, fig=fig)
        assert isinstance(fig, plt.Figure)
        for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects, strict=False):
            ax.set_aspect(1.0 / aspect)
    elif ax is not None and fig is not None:
        draw_plot(meshes, mesh_property, fig=fig, axes=ax)
        for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects, strict=False):
            ax.set_aspect(1.0 / aspect)
    return fig
