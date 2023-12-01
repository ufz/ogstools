"""Meshplotlib core utilitites."""
import types
from typing import Optional as Opt
from typing import Union

import numpy as np
import pyvista as pv
from matplotlib import cm as mcm
from matplotlib import colormaps, rcParams
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib import transforms as mtransforms
from matplotlib.patches import Rectangle as Rect

from ogstools.propertylib import Property, Vector
from ogstools.propertylib.presets import _resolve_property

from . import plot_features as pf
from . import setup
from .levels import get_levels

# TODO: define default data_name for regions in setup


def _q_zero_line(property: Property, levels: np.ndarray):
    return property.bilinear_cmap or (
        property.data_name == "temperature" and levels[0] < 0 < levels[-1]
    )


def get_data(mesh: pv.UnstructuredGrid, property: Property) -> np.ndarray:
    """Get the data associated with a scalar or vector property from a mesh."""
    if property.data_name in mesh.point_data:
        return mesh.point_data[property.data_name]
    if property.data_name in mesh.cell_data:
        return mesh.cell_data[property.data_name]
    msg = f"Property not found in mesh {mesh}."
    raise IndexError(msg)


def get_level_boundaries(levels: np.ndarray):
    return np.array(
        [
            levels[0] - 0.5 * (levels[1] - levels[0]),
            *0.5 * (levels[:-1] + levels[1:]),
            levels[-1] + 0.5 * (levels[-1] - levels[-2]),
        ]
    )


def get_cmap_norm(
    levels: np.ndarray, property: Property
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """Construct a discrete colormap and norm for the property field."""
    vmin, vmax = (levels[0], levels[-1])
    bilinear = property.bilinear_cmap and vmin <= 0.0 <= vmax
    cmap_str = setup.cmap_str(property)
    if property.is_mask():
        conti_cmap = mcolors.ListedColormap(cmap_str)
    elif isinstance(cmap_str, list):
        conti_cmap = [colormaps[c] for c in cmap_str]
    else:
        conti_cmap = colormaps[cmap_str]
    if property.data_name == "temperature":
        cool_colors = conti_cmap[0](np.linspace(0, 0.75, 128 * (vmin < 0)))
        warm_colors = conti_cmap[1](np.linspace(0, 1, 128 * (vmax >= 0)))
        conti_cmap = mcolors.LinearSegmentedColormap.from_list(
            "temperature_cmap", np.vstack((cool_colors, warm_colors))
        )
        bilinear = vmin < 0 < vmax
    if bilinear:
        vmin, vmax = np.max(np.abs([vmin, vmax])) * np.array([-1.0, 1.0])
    if property.categoric:
        vmin += 0.5
        vmax += 0.5
    conti_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mid_levels = np.append((levels[:-1] + levels[1:]) * 0.5, levels[-1])
    colors = [conti_cmap(conti_norm(m_l)) for m_l in mid_levels]
    cmap = mcolors.ListedColormap(colors, name="custom")
    if setup.custom_cmap is not None:
        cmap = setup.custom_cmap
    boundaries = get_level_boundaries(levels) if property.categoric else levels
    norm = mcolors.BoundaryNorm(
        boundaries=boundaries, ncolors=len(boundaries), clip=True
    )
    return cmap, norm


# to fix scientific offset position
# https://github.com/matplotlib/matplotlib/issues/4476#issuecomment-105627334
def fix_scientific_offset_position(axis, func):
    axis._update_offset_text_position = types.MethodType(func, axis)


def y_update_offset_text_position(self, bboxes, bboxes2):  # noqa: ARG001
    x, y = self.offsetText.get_position()
    # y in axes coords, x in display coords
    self.offsetText.set_transform(
        mtransforms.blended_transform_factory(
            self.axes.transAxes, mtransforms.IdentityTransform()
        )
    )
    top = self.axes.bbox.ymax
    y = top + 2 * self.OFFSETTEXTPAD * self.figure.dpi / 72.0
    self.offsetText.set_position((x, y))


def add_colorbars(
    fig: mfigure.Figure,
    ax: Union[plt.Axes, list[plt.Axes]],
    property: Property,
    levels: np.ndarray,
    pad: float = 0.05,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    cmap, norm = get_cmap_norm(levels, property)
    cm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    categoric = property.categoric or (len(levels) == 2)
    if categoric:
        bounds = get_level_boundaries(levels)
        ticks = bounds[:-1] + 0.5 * np.diff(bounds)
    else:
        ticks = levels
    cb = fig.colorbar(
        cm, norm=norm, ax=ax, ticks=ticks, drawedges=True, location="right",
        spacing="uniform", pad=pad, format="%.3g"  # fmt: skip
    )
    if setup.invert_colorbar:
        cb.ax.invert_yaxis()
    if property.is_mask():
        cb.ax.add_patch(Rect((0, 0.5), 1, -1, lw=0, fc="none", hatch="/"))
    if not categoric and setup.log_scaled:
        levels = 10**levels

    unit_str = (
        f" / {property.get_output_unit()}" if property.get_output_unit() else ""
    )
    cb.set_label(
        property.output_name.replace("_", " ") + unit_str,
        size=setup.rcParams_scaled["font.size"],
    )
    cb.ax.tick_params(
        labelsize=setup.rcParams_scaled["font.size"], direction="out"
    )
    mf = mticker.ScalarFormatter(useMathText=True, useOffset=True)
    mf.set_scientific(True)
    mf.set_powerlimits([-3, 3])
    fix_scientific_offset_position(cb.ax.yaxis, y_update_offset_text_position)
    cb.ax.yaxis.set_offset_position("left")
    cb.ax.yaxis.set_major_formatter(mf)

    if _q_zero_line(property, levels):
        cb.ax.axhline(
            y=0, color="w", lw=2 * setup.rcParams_scaled["lines.linewidth"]
        )
    if setup.log_scaled:
        cb.ax.set_yticklabels(10**ticks)

    if property.data_name == "MaterialIDs" and setup.material_names is not None:
        region_names = []
        for mat_id in levels:
            if mat_id in setup.material_names:
                region_names += [setup.material_names[mat_id]]
            else:
                region_names += [mat_id]
        cb.ax.set_yticklabels(region_names)
        cb.ax.set_ylabel("")
    elif property.categoric:
        cb.ax.set_yticklabels(levels.astype(int))


def subplot(
    mesh: pv.UnstructuredGrid,
    property: Union[Property, str],
    ax: plt.Axes,
    levels: Opt[np.ndarray] = None,
) -> None:
    """
    Plot the property field of a mesh on a matplotlib.axis.

    In 3D the mesh gets sliced according to slice_type
    and the origin in the PlotSetup in meshplotlib.setup.
    Custom levels and a colormap string can be provided.
    """

    if isinstance(property, str):
        data_shape = mesh[property].shape
        property = _resolve_property(property, data_shape)
    if mesh.get_cell(0).dimension == 3:
        msg = "meshplotlib is for 2D meshes only, but found 3D elements."
        raise ValueError(msg)

    ax.axis("auto")

    if (
        not property.is_mask()
        and property.mask in mesh.cell_data
        and len(mesh.cell_data[property.mask])
    ):
        subplot(mesh, property.get_mask(), ax)
        mesh = mesh.ctp(True).threshold(value=[1, 1], scalars=property.mask)

    surf_tri = mesh.triangulate().extract_surface()

    # get projection
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = np.delete([0, 1, 2], projection)

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    x, y = setup.length.strip_units(surf_tri.points.T[[x_id, y_id]])
    tri = surf_tri.faces.reshape((-1, 4))[:, 1:]
    values = property.magnitude.strip_units(get_data(surf_tri, property))
    if setup.log_scaled:
        values_temp = np.where(values > 1e-14, values, 1e-14)
        values = np.log10(values_temp)
    p_min, p_max = np.nanmin(values), np.nanmax(values)

    if levels is None:
        num_levels = min(setup.num_levels, len(np.unique(values)))
        levels = get_levels(p_min, p_max, num_levels)
    cmap, norm = get_cmap_norm(levels, property)

    if (
        property.data_name in mesh.cell_data
        and property.data_name not in mesh.point_data
    ):
        ax.tripcolor(x, y, tri, facecolors=values, cmap=cmap, norm=norm)
        if property.is_mask():
            ax.tripcolor(x, y, tri, facecolors=values, mask=(values == 1),
                         cmap=cmap, norm=norm, hatch="/")  # fmt: skip
    else:
        ax.tricontourf(x, y, tri, values, levels=levels, cmap=cmap, norm=norm)
        if _q_zero_line(property, levels):
            ax.tricontour(x, y, tri, values, levels=[0], colors="w")

    surf = mesh.extract_surface()

    if setup.show_region_bounds and "MaterialIDs" in mesh.cell_data:
        pf.plot_layer_boundaries(ax, surf, projection)

    show_edges = setup.show_element_edges
    if isinstance(setup.show_element_edges, str):
        show_edges = setup.show_element_edges == property.data_name
    if show_edges:
        pf.plot_element_edges(ax, surf, projection)

    if isinstance(property, Vector):
        pf.plot_streamlines(ax, surf_tri, property, projection)

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
        secax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
        secax.set_xticklabels(sec_labels)
        secax.set_xlabel(f'{"xyz"[projection]} / {setup.length.output_unit}')

    x_label = setup.x_label or f'{"xyz"[x_id]} / {setup.length.output_unit}'
    y_label = setup.y_label or f'{"xyz"[y_id]} / {setup.length.output_unit}'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _get_rows_cols(
    meshes: Union[
        list[pv.UnstructuredGrid],
        np.ndarray,
        pv.UnstructuredGrid,
        pv.MultiBlock,
    ]
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


def _fig_init(rows: int, cols: int, ax_aspect: float = 1.0) -> mfigure.Figure:
    nx_cb = 1 if setup.combined_colorbar else cols
    default_size = 8
    cb_width = 4
    y_label_width = 2
    x_label_height = 1
    figsize = setup.fig_scale * np.asarray(
        [
            default_size * cols * ax_aspect + cb_width * nx_cb + y_label_width,
            default_size * rows + x_label_height,
        ]
    )
    if figsize[0] / figsize[1] > setup.fig_aspect_limits[1]:
        figsize[0] = figsize[1] * setup.fig_aspect_limits[1]
    elif figsize[0] / figsize[1] < setup.fig_aspect_limits[0]:
        figsize[0] = figsize[1] * setup.fig_aspect_limits[0]
    fig, _ = plt.subplots(
        rows,
        cols,
        dpi=setup.dpi * setup.fig_scale,
        figsize=figsize,
        layout=setup.layout,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_alpha(1)
    return fig


def get_combined_levels(
    meshes: np.ndarray, property: Union[Property, str]
) -> np.ndarray:
    """
    Calculate well spaced levels for the encompassing property range in meshes.
    """
    if isinstance(property, str):
        data_shape = meshes[0][property].shape
        property = _resolve_property(property, data_shape)
    p_min, p_max = np.inf, -np.inf
    unique_vals = np.array([])
    for mesh in np.ravel(meshes):
        values = property.magnitude.strip_units(get_data(mesh, property))
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
    return get_levels(p_min, p_max, setup.num_levels)


def _plot_on_figure(
    fig: mfigure.Figure,
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid],
    property: Property,
) -> mfigure.Figure:
    """
    Plot the property field of meshes on existing figure.

    :param meshes: Singular mesh of 2D numpy array of meshes
    :param property: the property field to be visualized on all meshes
    """
    shape = _get_rows_cols(meshes)
    np_meshes = np.reshape(meshes, shape)
    np_axs = np.reshape(fig.axes, shape)
    if setup.combined_colorbar:
        combined_levels = get_combined_levels(np_meshes, property)

    for i in range(shape[0]):
        for j in range(shape[1]):
            _levels = (
                combined_levels
                if setup.combined_colorbar
                else get_combined_levels(np_meshes[i, j], property)
            )
            subplot(np_meshes[i, j], property, np_axs[i, j], _levels)

    np_axs[0, 0].set_title(setup.title_center, loc="center", y=1.02)
    np_axs[0, 0].set_title(setup.title_left, loc="left", y=1.02)
    np_axs[0, 0].set_title(setup.title_right, loc="right", y=1.02)
    # make extra space for the upper limit of the colorbar
    if setup.layout == "tight":
        plt.tight_layout(pad=1.4)

    if setup.combined_colorbar:
        cb_axs = np.ravel(fig.axes).tolist()
        add_colorbars(
            fig, cb_axs, property, combined_levels, pad=0.05 / shape[1]
        )
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                _levels = get_combined_levels(np_meshes[i, j], property)
                add_colorbars(fig, np_axs[i, j], property, _levels)

    return fig


def get_data_aspect(mesh: pv.DataSet) -> float:
    """
    Calculate the data aspect ratio of a 2D mesh.
    """
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = 2 * np.delete([0, 1, 2], projection)
    lims = mesh.bounds
    return abs(lims[x_id + 1] - lims[x_id]) / abs(lims[y_id + 1] - lims[y_id])


# TODO: add as arguments: cmap, limits
# TODO: num_levels should be min_levels
def plot(
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid],
    property: Union[Property, str],
) -> mfigure.Figure:
    """
    Plot the property field of meshes with default settings.

    The resulting figure adheres to the configurations in meshplotlib.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes:      Singular mesh of 2D numpy array of meshes
    :param property:    The property field to be visualized on all meshes
    """

    rcParams.update(setup.rcParams_scaled)
    shape = _get_rows_cols(meshes)
    _meshes = np.reshape(meshes, shape).flatten()
    if isinstance(property, str):
        data_shape = _meshes[0][property].shape
        property = _resolve_property(property, data_shape)
    data_aspects = np.asarray([get_data_aspect(mesh) for mesh in _meshes])
    data_aspects[
        (data_aspects > setup.ax_aspect_limits[0])
        & (data_aspects < setup.ax_aspect_limits[1])
    ] = 1.0
    clamped_aspects = np.clip(data_aspects, *setup.ax_aspect_limits)
    ax_aspects = data_aspects / clamped_aspects
    _fig = _fig_init(
        rows=shape[0], cols=shape[1], ax_aspect=np.mean(ax_aspects)
    )
    n_axs = shape[0] * shape[1]
    # setting the aspect twice is intended
    # the first time results in properly spaced ticks
    # the second time fixes any deviations from the set aspect due to
    # additional colorbar(s) or secondary axes.
    for ax, aspect in zip(_fig.axes[: n_axs + 1], clamped_aspects):
        ax.set_aspect(aspect)
    fig = _plot_on_figure(_fig, meshes, property)
    for ax, aspect in zip(fig.axes[: n_axs + 1], clamped_aspects):
        ax.set_aspect(aspect)
    return fig
