# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Meshplotlib core utilitites."""

import warnings
from math import nextafter
from typing import Any, Literal, Optional, Union

import numpy as np
import pyvista as pv
from matplotlib import cm as mcm
from matplotlib import colormaps, rcParams
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle as Rect
from typeguard import typechecked

from ogstools.meshlib import MeshSeries
from ogstools.propertylib import Property, Vector
from ogstools.propertylib.properties import get_preset
from ogstools.propertylib.unit_registry import u_reg

from . import plot_features as pf
from . import setup
from .levels import compute_levels, median_exponent
from .utils import get_style_cycler

# TODO: define default data_name for regions in setup


def _q_zero_line(mesh_property: Property, levels: np.ndarray) -> bool:
    return mesh_property.bilinear_cmap or (
        mesh_property.data_name == "temperature" and levels[0] < 0 < levels[-1]
    )


def get_level_boundaries(levels: np.ndarray) -> np.ndarray:
    return np.array(
        [
            levels[0] - 0.5 * (levels[1] - levels[0]),
            *0.5 * (levels[:-1] + levels[1:]),
            levels[-1] + 0.5 * (levels[-1] - levels[-2]),
        ]
    )


def get_cmap_norm(
    levels: np.ndarray, mesh_property: Property
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """Construct a discrete colormap and norm for the property field."""
    vmin, vmax = (levels[0], levels[-1])
    if mesh_property.categoric:
        vmin += 0.5
        vmax += 0.5

    if isinstance(mesh_property.cmap, str):
        continuous_cmap = colormaps[mesh_property.cmap]
    else:
        continuous_cmap = mesh_property.cmap
    conti_norm: Union[mcolors.TwoSlopeNorm, mcolors.Normalize]
    if mesh_property.bilinear_cmap:
        if vmin <= 0.0 <= vmax:
            vcenter = 0.0
            vmin, vmax = np.max(np.abs([vmin, vmax])) * np.array([-1.0, 1.0])
            conti_norm = mcolors.TwoSlopeNorm(vcenter, vmin, vmax)
        else:
            # only use one half of the diverging colormap
            col_range = np.linspace(0.0, nextafter(0.5, -np.inf), 128)
            if vmax > 0.0:
                col_range += 0.5
            continuous_cmap = mcolors.LinearSegmentedColormap.from_list(
                "half_cmap", continuous_cmap(col_range)
            )
            conti_norm = mcolors.Normalize(vmin, vmax)
    else:
        conti_norm = mcolors.Normalize(vmin, vmax)
    mid_levels = np.append((levels[:-1] + levels[1:]) * 0.5, levels[-1])
    colors = [continuous_cmap(conti_norm(m_l)) for m_l in mid_levels]
    cmap = mcolors.ListedColormap(colors, name="custom")
    boundaries = (
        get_level_boundaries(levels) if mesh_property.categoric else levels
    )
    norm = mcolors.BoundaryNorm(
        boundaries=boundaries, ncolors=len(boundaries), clip=False
    )
    return cmap, norm


def get_ticklabels(ticks: np.ndarray) -> tuple[list[str], Optional[str]]:
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
    fig: mfigure.Figure,
    ax: Union[plt.Axes, list[plt.Axes]],
    mesh_property: Property,
    levels: np.ndarray,
    pad: float = 0.05,
    labelsize: Optional[float] = None,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    ticks = levels
    if mesh_property.categoric or (len(levels) == 2):
        bounds = get_level_boundaries(levels)
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


def get_projection(
    mesh: pv.UnstructuredGrid,
) -> tuple[int, int]:
    """
    Identify which projection is used: XY, XZ or YZ.

    :param mesh: singular mesh

    """
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = np.delete([0, 1, 2], projection)
    return x_id, y_id


def subplot(
    mesh: pv.UnstructuredGrid,
    mesh_property: Union[Property, str],
    ax: plt.Axes,
    levels: Optional[np.ndarray] = None,
) -> None:
    """
    Plot the property field of a mesh on a matplotlib.axis.

    In 3D the mesh gets sliced according to slice_type
    and the origin in the PlotSetup in meshplotlib.setup.
    Custom levels and a colormap string can be provided.
    """

    mesh_property = get_preset(mesh_property, mesh)
    if mesh.get_cell(0).dimension == 3:
        msg = "meshplotlib is for 2D meshes only, but found 3D elements."
        raise ValueError(msg)

    ax.axis("auto")

    if mesh_property.mask_used(mesh):
        subplot(mesh, mesh_property.get_mask(), ax)
        mesh = mesh.ctp(True).threshold(
            value=[1, 1], scalars=mesh_property.mask
        )

    surf_tri = mesh.triangulate().extract_surface()

    # get projection
    x_id, y_id = get_projection(mesh)
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    x, y = setup.length.transform(surf_tri.points.T[[x_id, y_id]])
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
        pf.plot_element_edges(ax, surf, projection)

    if setup.show_region_bounds and "MaterialIDs" in mesh.cell_data:
        pf.plot_layer_boundaries(ax, surf, projection)

    if isinstance(mesh_property, Vector):
        pf.plot_streamlines(ax, surf_tri, mesh_property, projection)

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
        secax.set_xlabel(f'{"xyz"[projection]} / {setup.length.output_unit}')


def clear_labels(axes: Union[plt.Axes, np.ndarray]) -> None:
    ax: plt.Axes
    for ax in np.ravel(np.array(axes)):
        ax.set_xlabel("")
        ax.set_ylabel("")


@typechecked
def label_spatial_axes(
    axes: Union[plt.Axes, np.ndarray],
    x_label: str = "x",
    y_label: str = "y",
    label_axes: str = "both",
    fontsize: int = 20,
) -> None:
    """
    Add labels to x and y axis.

    If given an array of axes, only the outer axes will be labeled.
    """
    if isinstance(axes, np.ndarray):
        ax: plt.Axes
        for ax in axes[-1, :]:
            if label_axes in ["x", "both"]:
                ax.set_xlabel(
                    f"{x_label} / {setup.length.output_unit}", fontsize=fontsize
                )
        for ax in axes[:, 0]:
            if label_axes in ["y", "both"]:
                ax.set_ylabel(
                    f"{y_label} / {setup.length.output_unit}", fontsize=fontsize
                )
    else:
        if label_axes in ["x", "both"]:
            axes.set_xlabel(
                f"{x_label} / {setup.length.output_unit}", fontsize=fontsize
            )
        if label_axes in ["y", "both"]:
            axes.set_ylabel(
                f"{y_label} / {setup.length.output_unit}", fontsize=fontsize
            )


def _get_rows_cols(
    meshes: Union[
        list[pv.UnstructuredGrid],
        np.ndarray,
        pv.UnstructuredGrid,
        pv.MultiBlock,
    ],
) -> tuple[int, ...]:
    if isinstance(meshes, np.ndarray):
        if meshes.ndim in [1, 2]:
            return meshes.shape
        msg = "Input numpy array must be 1D or 2D."
        raise ValueError(msg)
    if isinstance(meshes, list):
        return (1, len(meshes))
    if isinstance(meshes, pv.MultiBlock):
        return (1, meshes.n_blocks)
    return (1, 1)


# TODO: fixed_figure_size -> ax aspect automatic


def _fig_init(
    rows: int, cols: int, aspect: float = 1.0
) -> tuple[mfigure.Figure, plt.Axes]:
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


def get_combined_levels(
    meshes: np.ndarray, mesh_property: Union[Property, str]
) -> np.ndarray:
    """
    Calculate well spaced levels for the encompassing property range in meshes.
    """
    mesh_property = get_preset(mesh_property, meshes.ravel()[0])
    p_min, p_max = np.inf, -np.inf
    unique_vals = np.array([])
    for mesh in np.ravel(meshes):
        values = mesh_property.magnitude.transform(mesh)
        if setup.log_scaled:  # TODO: can be improved
            values = np.log10(np.where(values > 1e-14, values, 1e-14))
        p_min = min(p_min, np.nanmin(values)) if setup.p_min is None else p_min
        p_max = max(p_max, np.nanmax(values)) if setup.p_max is None else p_max
        unique_vals = np.unique(
            np.concatenate((unique_vals, np.unique(values)))
        )
    p_min = setup.p_min if setup.p_min is not None else p_min
    p_max = setup.p_max if setup.p_max is not None else p_max
    if p_min == p_max:
        return np.array([p_min, p_max + 1e-12])
    if (
        all(val.is_integer() for val in unique_vals)
        and setup.p_min is None
        and setup.p_max is None
    ):
        return unique_vals[(p_min <= unique_vals) & (unique_vals <= p_max)]
    return compute_levels(p_min, p_max, setup.num_levels)


# TODO: Have a look at fig and ax logic and make it more readable


def _draw_plot(
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid],
    mesh_property: Property,
    fig: Optional[mfigure.Figure] = None,
    axes: Optional[plt.Axes] = None,
) -> Optional[mfigure.Figure]:
    """
    Plot the property field of meshes on existing figure.

    :param meshes: Singular mesh of 2D numpy array of meshes
    :param property: the property field to be visualized on all meshes
    :param fig: Matplotlib figure to use for plotting (optional)
    :param axes: Matplotlib Axes to use for plotting (optional)
    """
    shape = _get_rows_cols(meshes)
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
        combined_levels = get_combined_levels(np_meshes, mesh_property)
    for i in range(shape[0]):
        for j in range(shape[1]):
            _levels = (
                combined_levels
                if setup.combined_colorbar
                else get_combined_levels(np_meshes[i, j, None], mesh_property)
            )
            subplot(np_meshes[i, j], mesh_property, np_axs[i, j], _levels)

    x_id, y_id = get_projection(
        np_meshes[0, 0]
    )  # One mesh is sufficient, it should be the same for all of them
    label_spatial_axes(np_axs, "xyz"[x_id], "xyz"[y_id])
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
                fig, cb_axs, mesh_property, combined_levels, pad=0.05 / shape[1]
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
                    _levels = get_combined_levels(
                        np_meshes[i, j, None], mesh_property
                    )
                    add_colorbars(fig, np_axs[i, j], mesh_property, _levels)
    return fig


def get_data_aspect(mesh: pv.UnstructuredGrid) -> float:
    """
    Calculate the data aspect ratio of a 2D mesh.
    """
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = 2 * np.delete([0, 1, 2], projection)
    lims = mesh.bounds
    return abs(lims[x_id + 1] - lims[x_id]) / abs(lims[y_id + 1] - lims[y_id])


def update_font_sizes(
    fontsize: int = 20,
    label_axes: str = "both",
    fig: Optional[mfigure.Figure] = None,
    ax: Optional[plt.axes] = None,
) -> mfigure.Figure:
    """
    Update font sizes of labels and ticks in all subplots

    :param fig: Matplotlib Figure object to use for plotting
    :param fontsize: New font size for the labels and ticks (optional)
    :param label_axes: Apply labels to axis: "x", "y", "both", "none"
    """
    # TODO: Remove labeling axes from this function
    if fig is None and ax is None:
        err_msg = "Neither Figure nor Axes was provided"
        raise ValueError(err_msg)
    if isinstance(ax, np.ndarray):
        err_msg = "If you want apply this function to multiple subplots,\
            please provide Figure."
        raise ValueError(err_msg)
    if fig is not None and ax is None:
        axes = fig.get_axes()
    elif fig is None and ax is not None:
        axes = np.array([ax])
    else:
        err_msg = "Invalid combination of Axis and Figure!"
        raise ValueError(err_msg)

    for subax in axes:
        if label_axes != "none":
            label_spatial_axes(
                subax,
                x_label="X",
                y_label="Y",
                label_axes=label_axes,
                fontsize=fontsize,
            )
        subax_xlim = subax.get_xlim()
        subax_ylim = subax.get_ylim()
        subax.set_xticks(
            subax.get_xticks(),
            [label.get_text() for label in subax.get_xticklabels()],
            fontsize=fontsize,
        )
        subax.set_yticks(
            subax.get_yticks(),
            [label.get_text() for label in subax.get_yticklabels()],
            fontsize=fontsize,
        )
        subax.set_xlim(subax_xlim)
        subax.set_ylim(subax_ylim)
        subax.xaxis.label.set_fontsize(fontsize)
        subax.yaxis.label.set_fontsize(fontsize)
    return fig, ax


# TODO: add as arguments: cmap, limits
# TODO: num_levels should be min_levels
def plot(
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid],
    mesh_property: Union[Property, str],
    fig: Optional[mfigure.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[mfigure.Figure]:
    """
    Plot the property field of meshes with default settings.

    The resulting figure adheres to the configurations in meshplotlib.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes:      Singular mesh of 2D numpy array of meshes
    :param property:    The property field to be visualized on all meshes
    :param fig: Matplotlib Figure object to use for plotting (optional)
    :param ax: Matplotlib Axis object to use for plotting (optional)
    """
    rcParams.update(setup.rcParams_scaled)
    shape = _get_rows_cols(meshes)
    _meshes = np.reshape(meshes, shape).ravel()
    mesh_property = get_preset(mesh_property, _meshes[0])
    data_aspects = np.asarray([get_data_aspect(mesh) for mesh in _meshes])
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
        fig = _draw_plot(meshes, mesh_property, fig=_fig, axes=_ax)
        assert isinstance(fig, plt.Figure)
        for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects):
            ax.set_aspect(1.0 / aspect)
    elif ax is not None and fig is None:
        _draw_plot(meshes, mesh_property, axes=ax)
        ax.set_aspect(1.0 / ax_aspects[0])
    elif ax is None and fig is not None:
        fig = _draw_plot(meshes, mesh_property, fig=fig)
        assert isinstance(fig, plt.Figure)
        for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects):
            ax.set_aspect(1.0 / aspect)
    elif ax is not None and fig is not None:
        _draw_plot(meshes, mesh_property, fig=fig, axes=ax)
        for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects):
            ax.set_aspect(1.0 / aspect)
    return fig


def plot_probe(
    mesh_series: MeshSeries,
    points: np.ndarray,
    mesh_property: Union[Property, str],
    mesh_property_abscissa: Optional[Union[Property, str]] = None,
    labels: Optional[list[str]] = None,
    time_unit: Optional[str] = "s",
    interp_method: Optional[Literal["nearest", "linear", "probefilter"]] = None,
    interp_backend_pvd: Optional[Literal["vtk", "scipy"]] = None,
    colors: Optional[list] = None,
    linestyles: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    fill_between: bool = False,
    **kwargs: Any,
) -> Optional[mfigure.Figure]:
    """
    Plot the transient property on the observation points in the MeshSeries.

        :param mesh_series: MeshSeries object containing the data to be plotted.
        :param points:          The points to sample at.
        :param mesh_property:   The property to be sampled.
        :param labels:          The labels for each observation point.
        :param time_unit:       Output unit of the timevalues.
        :param interp_method:   Choose the interpolation method, defaults to
                                `linear` for xdmf MeshSeries and `probefilter`
                                for pvd MeshSeries.
        :param interp_backend:  Interpolation backend for PVD MeshSeries.
        :param kwargs:          Keyword arguments passed to matplotlib's plot
                                function.

        :returns:   A matplotlib Figure
    """
    points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[np.newaxis]
    mesh_property = get_preset(mesh_property, mesh_series.read(0))
    values = mesh_property.magnitude.transform(
        mesh_series.probe(
            points, mesh_property.data_name, interp_method, interp_backend_pvd
        )
    )
    if values.shape[0] == 1:
        values = values.flatten()
    Q_ = u_reg.Quantity
    time_unit_conversion = Q_(Q_(mesh_series.time_unit), time_unit).magnitude
    if mesh_property_abscissa is None:
        x_values = time_unit_conversion * mesh_series.timevalues
        x_label = f"time / {time_unit}" if time_unit else "time"
    else:
        mesh_property_abscissa = get_preset(
            mesh_property_abscissa, mesh_series.read(0)
        )
        x_values = mesh_property_abscissa.magnitude.transform(
            mesh_series.probe(
                points,
                mesh_property_abscissa.data_name,
                interp_method,
                interp_backend_pvd,
            )
        )
        x_unit_str = (
            f" / {mesh_property_abscissa.get_output_unit()}"
            if mesh_property_abscissa.get_output_unit()
            else ""
        )
        x_label = (
            mesh_property_abscissa.output_name.replace("_", " ") + x_unit_str
        )
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.set_prop_cycle(get_style_cycler(len(points), colors, linestyles))
    if fill_between:
        ax.fill_between(
            x_values,
            np.min(values, axis=-1),
            np.max(values, axis=-1),
            label=labels,
            **kwargs,
        )
    else:
        ax.plot(x_values, values, label=labels, **kwargs)
    if labels is not None:
        ax.legend(facecolor="white", framealpha=1, prop={"family": "monospace"})
    ax.set_axisbelow(True)
    # TODO: wrap this in apply_mpl_style()
    ax.grid(which="major", color="lightgrey", linestyle="-")
    ax.grid(which="minor", color="0.95", linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(mesh_property.get_label())
    ax.label_outer()
    ax.minorticks_on()
    return fig


def color_twin_axes(axes: list, colors: list) -> None:
    for ax_temp, color_temp in zip(axes, colors):
        ax_temp.tick_params(axis="y", which="both", colors=color_temp)
        ax_temp.yaxis.label.set_color(color_temp)
    # Axis spine color has to be applied on twin axis for both sides
    axes[1].spines["left"].set_color(colors[0])
    axes[1].spines["right"].set_color(colors[1])
