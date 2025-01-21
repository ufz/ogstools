# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Plotting core utilitites."""

import warnings
from math import nextafter
from typing import Any

import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle as Rect

from ogstools.plot import utils
from ogstools.plot.levels import combined_levels, level_boundaries
from ogstools.variables import Variable, Vector, get_preset

from . import features
from .levels import compute_levels, median_exponent
from .shared import setup
from .vectorplots import streamlines

# TODO: define default data_name for regions in setup


def get_ticklabels(ticks: np.ndarray) -> tuple[list[str], str | None]:
    """Get formatted tick labels and optional offset str.

    If all values in ticks are too close together offset notation is used.
    """
    if (
        median_exponent(ticks) >= 3 + median_exponent(ticks[-1] - ticks[0])
        and nextafter(ticks[0], np.inf) != ticks[-1]
    ):
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
        fmt = (
            "f"
            if abs(median_exponent(ticks - offset)) <= 2
            or nextafter(ticks[0], np.inf) == ticks[-1]
            else "e"
        )
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
    variable: Variable,
    levels: np.ndarray,
    **kwargs: Any,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    ticks = levels
    if variable.categoric or (len(levels) == 2):
        bounds = level_boundaries(levels)
        ticks = bounds[:-1] + 0.5 * np.diff(bounds)

    with np.errstate(over="ignore"):
        cmap, norm = utils.get_cmap_norm(levels, variable, **kwargs)
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(
            scalar_mappable,
            norm=norm,
            ax=ax,
            ticks=ticks,
            drawedges=True,
            location=kwargs.get("cb_loc", "right"),
            spacing="uniform",
            pad=kwargs.get("cb_pad", 0.05),
            extendrect=True,
        )
    # Formatting the colorbar label and ticks

    tick_labels, offset = get_ticklabels(ticks)
    cb_label = variable.magnitude.get_label(setup.label_split)
    if offset is not None:
        if offset[0] == "-":
            cb_label += " + " + offset[1:]
        else:
            cb_label += " - " + offset
    if kwargs.get("log_scaled", setup.log_scaled):
        cb_label = f"log$_{{10}}$( {cb_label} )"
    labelsize = kwargs.get(
        "cb_labelsize", kwargs.get("fontsize", setup.fontsize)
    )
    cb.set_label(cb_label, size=labelsize)

    # special formatting for MaterialIDs
    if variable.data_name == "MaterialIDs" and setup.material_names is not None:
        tick_labels = [
            setup.material_names.get(mat_id, mat_id) for mat_id in levels
        ]
        cb.ax.set_ylabel("")
    elif variable.categoric:
        tick_labels = [str(level) for level in levels.astype(int)]
    cb.ax.tick_params(labelsize=labelsize, direction="out")
    cb.ax.set_yticklabels(tick_labels)

    # miscellaneous

    if variable.is_mask():
        cb.ax.add_patch(Rect((0, 0.5), 1, -1, lw=0, fc="none", hatch="/"))
    if setup.invert_colorbar:
        cb.ax.invert_yaxis()
    if variable.bilinear_cmap:
        cb.ax.axhline(y=0, color="w", lw=2 * setup.linewidth)


def subplot(
    mesh: pv.UnstructuredGrid,
    variable: Variable | str,
    ax: plt.Axes,
    levels: np.ndarray | None = None,
    **kwargs: Any,
) -> None:
    "Plot the variable field of a mesh on a matplotlib.axis."

    variable = get_preset(variable, mesh)
    if mesh.get_cell(0).dimension == 3:
        msg = "This method is for 2D meshes only, but found 3D elements."
        raise ValueError(msg)

    ax.axis("auto")

    if variable.mask_used(mesh):
        subplot(mesh, variable.get_mask(), ax, **kwargs)
        mesh = mesh.ctp(True).threshold(value=[1, 1], scalars=variable.mask)

    surf_tri = mesh.triangulate().extract_surface()
    # Get rid of 1D elements in the mesh
    surf_tri = surf_tri.extract_cells_by_type(pv.CellType.TRIANGLE)
    x_id, y_id, projection, mean_normal = utils.get_projection(mesh)

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    x, y = surf_tri.points.T[[x_id, y_id]]
    tri = surf_tri.faces.reshape((-1, 4))[:, 1:]
    values = variable.magnitude.transform(surf_tri)
    # Passing the data and not the mesh here purposely to ensure correct shape
    # of mask. Otherwise transform() might remove additional triangulated cells
    # due to those being part of the usual mask e.g. "pressure_active".
    nan_mask = np.isnan(surf_tri.ptc()[variable.data_name])
    # Getting rid of extra dimension for vectors and matrices
    if len(nan_mask.shape) == 2:
        nan_mask = np.sum(nan_mask, axis=-1)
    if kwargs.get("log_scaled", setup.log_scaled) and not variable.is_mask():
        values_temp = np.where(values > 1e-14, values, 1e-14)
        values = np.log10(values_temp)
    vmin, vmax = np.nanmin(values), np.nanmax(values)

    if levels is None:
        num_levels = min(
            kwargs.get("num_levels", setup.num_levels), len(np.unique(values))
        )
        levels = compute_levels(vmin, vmax, num_levels)
    cmap, norm = utils.get_cmap_norm(levels, variable, **kwargs)

    # norm.__call__ overflows if vals are all equal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if variable.data_name in mesh.point_data:
            ax.tricontourf(
                x, y, tri, values, levels=kwargs.get("levels", levels),
                cmap=cmap, norm=norm, extend="both", mask=nan_mask
            )  # fmt: skip
            if variable.bilinear_cmap:
                ax.tricontour(
                    x, y, tri, values, levels=[0], mask=nan_mask, colors="w"
                )
        else:
            ax.tripcolor(
                x, y, tri, facecolors=values, mask=nan_mask,
                cmap=cmap, norm=norm  # fmt: skip
            )
            if variable.is_mask():
                ax.tripcolor(
                    x, y, tri, facecolors=values, mask=(values == 1),
                    cmap=cmap, norm=norm, hatch="/"
                )  # fmt: skip

    show_edges = setup.show_element_edges
    if isinstance(setup.show_element_edges, str):
        show_edges = setup.show_element_edges == variable.data_name
    show_edges = kwargs.get("show_edges", show_edges)
    if show_edges:
        features.element_edges(ax, mesh, projection)
    show_region_bounds = kwargs.get(
        "show_region_bounds", setup.show_region_bounds
    )
    if show_region_bounds and "MaterialIDs" in mesh.cell_data:
        features.layer_boundaries(ax, mesh, projection)

    if isinstance(variable, Vector):
        streamlines(surf_tri, ax, variable, projection)

    ax.margins(0, 0)  # otherwise it shrinks the plot content
    show_max = kwargs.get("show_max", False) and not variable.is_mask()
    show_min = kwargs.get("show_min", False) and not variable.is_mask()
    fontsize = kwargs.get("fontsize", setup.fontsize)

    for show, func, level_index in zip(
        [show_min, show_max], [np.argmin, np.argmax], [0, -1], strict=True
    ):
        if not show:
            continue
        index = np.unravel_index(func(values), values.shape)[0]
        x_pos, y_pos = mesh.points[index, [x_id, y_id]]
        value = values[mesh.find_closest_point(mesh.points[index])]
        color = utils.contrast_color(cmap(norm(value)))
        ax.plot(
            x_pos, y_pos, color=color, marker="x", clip_on=False,
            markersize=fontsize*0.625, markeredgewidth=3
        )  # fmt: skip
        pad_x = not (show_min and show_max)  # prevent overlap
        text_xy = utils.padded(ax, x_pos, y_pos, pad_x)
        text = f"{levels[level_index]:.3g}"
        ha, va = ("center", "center")
        ax.text(*text_xy, s=text, ha=ha, va=va, fontsize=fontsize, color=color)

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
        if not all(label == sec_labels[0] for label in sec_labels):
            # TODO: use a function to make this short
            secax = ax.secondary_xaxis("top")
            secax.xaxis.set_major_locator(
                mticker.FixedLocator(list(ax.get_xticks()))
            )
            secax.set_xticklabels(sec_labels)
            secax.set_xlabel(f'{"xyz"[projection]} / {setup.spatial_unit}')
            utils.update_font_sizes(secax, fontsize)


# TODO: fixed_figure_size -> ax aspect automatic


def _fig_init(
    rows: int, cols: int, aspect: float = 1.0, **kwargs: Any
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    nx_cb = 1 if setup.combined_colorbar else cols
    default_size = 8
    cb_width = 3
    y_label_width = 2
    x_label_height = 1
    figsize = np.asarray(
        [
            default_size * cols * aspect + cb_width * nx_cb + y_label_width,
            default_size * rows + x_label_height,
        ]
    )
    # TODO: maybe use height and width_ratios argument?
    fig, ax = plt.subplots(
        rows,
        cols,
        dpi=kwargs.get("dpi", setup.dpi),
        figsize=kwargs.get("figsize", figsize),
        layout=kwargs.get("layout", setup.layout),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(1)
    return fig, ax


# TODO: Have a look at fig and ax logic and make it more readable


def _take_axes(
    fig: plt.Figure | None, axes: plt.Axes | None, shape: tuple[int, ...]
) -> np.ndarray:
    match fig, axes:
        case None, None:
            msg = "Neither figure nor axes was provided."
            raise TypeError(msg)
        case fig, None:
            msg = (
                "This is not a good practice. Consider providing both fig "
                "and axes instead. This option may lead to unexpected "
                "behaviour and may be removed in the future."
            )
            warnings.warn(msg, Warning, stacklevel=4)
            assert isinstance(fig, plt.Figure)
            return np.reshape(np.asarray(fig.axes), shape)
        case None, axes:
            if shape != (1, 1):
                msg = (
                    "You provided only one axes but multiple meshes."
                    "Provide one mesh per axes or provide a figure."
                )
                raise ValueError(msg)
            return np.reshape(np.array(axes), (1, 1))
    return np.reshape(np.asarray(axes), shape)


def draw_plot(
    meshes: list[pv.UnstructuredGrid] | np.ndarray | pv.UnstructuredGrid,
    variable: Variable,
    fig: plt.Figure | None = None,
    axes: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Figure | None:
    """
    Plot the variable field of meshes on an existing figure or axis.

    :param meshes: singular mesh or 2D numpy array of meshes
    :param variable: the field to be visualized on all meshes
    :param fig: matplotlib figure to use for plotting
    :param axes: matplotlib axes to use for plotting
    """
    shape = utils.get_rows_cols(meshes)
    np_meshes = np.reshape(meshes, shape)
    np_axs = _take_axes(fig, axes, shape)

    if setup.combined_colorbar:
        _levels = combined_levels(np_meshes, variable, **kwargs)
    for i, j in [(r0, r1) for r0 in range(shape[0]) for r1 in range(shape[1])]:
        if "levels" in kwargs:
            _levels = np.asarray(kwargs.pop("levels"))
        elif not setup.combined_colorbar:
            _levels = combined_levels(np_meshes[i, j, None], variable, **kwargs)
        subplot(np_meshes[i, j], variable, np_axs[i, j], _levels, **kwargs)
        if fig is None or setup.combined_colorbar:
            continue
        add_colorbars(fig, np_axs[i, j], variable, _levels, **kwargs)

    # One mesh is sufficient, it should be the same for all of them
    x_id, y_id, _, _ = utils.get_projection(np_meshes[0, 0])
    utils.label_spatial_axes(
        np_axs, "xyz"[x_id], "xyz"[y_id], spatial_unit=setup.spatial_unit
    )
    # make extra space for the upper limit of the colorbar
    if setup.layout == "tight" and fig is not None:
        fig.tight_layout(pad=1.4)
    if fig is not None and setup.combined_colorbar:
        cb_axs = np.ravel(np.asarray(fig.axes)).tolist()
        add_colorbars(fig, cb_axs, variable, _levels, **kwargs)
    return fig


def contourf(
    meshes: list[pv.UnstructuredGrid] | np.ndarray | pv.UnstructuredGrid,
    variable: Variable | str,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Figure | None:
    """
    Plot the variable field of meshes with default settings.

    The resulting figure adheres to the configurations in plot.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes:      Singular mesh of 2D numpy array of meshes
    :param variable:   The field to be visualized on all meshes
    :param fig:         matplotlib figure to use for plotting
    :param ax:          matplotlib axis to use for plotting
    :Keyword Arguments:
        - cb_labelsize:       colorbar labelsize
        - cb_loc:             colorbar location ('left' or 'right')
        - cb_pad:             colorbar padding
        - cmap:               colormap
        - dpi:                resolution
        - figsize:            figure size
        - fontsize            size for labels and captions
        - levels:             user defined levels
        - log_scaled:         logarithmic scaling
        - show_edges:         show element edges
        - show_max:           mark the location of the maximum value
        - show_min:           mark the location of the minimum value
        - show_region_bounds: show the edges of the different regions
        - vmin:               minimum value
        - vmax:               maximum value
    """
    shape = utils.get_rows_cols(meshes)
    _meshes = np.reshape(meshes, shape).ravel()
    variable = get_preset(variable, _meshes[0])
    data_aspects = np.asarray([utils.get_data_aspect(mesh) for mesh in _meshes])
    if setup.min_ax_aspect is None and setup.max_ax_aspect is None:
        fig_aspect = np.mean(data_aspects)
    else:
        fig_aspect = np.mean(
            np.clip(data_aspects, setup.min_ax_aspect, setup.max_ax_aspect)
        )
    ax_aspects = fig_aspect / data_aspects
    n_axs = shape[0] * shape[1]
    if fig is None and ax is None:
        fig, ax = _fig_init(shape[0], shape[1], fig_aspect, **kwargs)
    fig = draw_plot(meshes, variable, fig=fig, axes=ax, **kwargs)
    if ax is not None and isinstance(ax, plt.Axes):
        ax.set_aspect(1.0 / ax_aspects[0])
    elif fig is not None:
        for _ax, aspect in zip(fig.axes[:n_axs], ax_aspects, strict=True):
            _ax.set_aspect(1.0 / aspect)
    if fig is None:
        return None
    utils.update_font_sizes(
        fig.axes, fontsize=kwargs.get("fontsize", setup.fontsize)
    )

    return fig
