"""Meshplotlib core utilitites."""
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
from matplotlib.patches import Rectangle as Rect

from ogstools.propertylib import THM, Property
from ogstools.propertylib import VectorProperty as Vector

from . import plot_features as pf
from . import setup
from .levels import get_levels

# TODO: toggle colorbar per ax
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


def add_colorbar(
    fig: mfigure.Figure,
    property: Property,
    levels: np.ndarray,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    cmap, norm = get_cmap_norm(levels, property)
    cm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    scale_mag = np.median(np.abs(np.diff(levels)))
    scale_exp = np.ceil(np.log10(scale_mag)) if scale_mag > 1e-12 else 0
    if scale_exp >= 3:
        levels *= 10 ** (-scale_exp)
        norm.vmin *= 10 ** (-scale_exp)
        norm.vmax *= 10 ** (-scale_exp)
    categoric = property.categoric or (len(levels) == 2)
    if categoric:
        bounds = get_level_boundaries(levels)
        ticks = bounds[:-1] + 0.5 * np.diff(bounds)
    else:
        bin_sizes = np.diff(levels)
        bot_extra = int(bin_sizes[0] != bin_sizes[1]) or None
        top_extra = -int(bin_sizes[-1] != bin_sizes[-2]) or None
        ticks = levels[bot_extra:top_extra]
    _axs = np.ravel(fig.axes).tolist()
    spacing = "uniform" if categoric else "proportional"
    kwargs = {"location": "right", "spacing": spacing, "pad": 0.02}
    cb = fig.colorbar(
        cm, norm=norm, ax=_axs, ticks=ticks, drawedges=True, **kwargs
    )
    if setup.invert_colorbar:
        cb.ax.invert_yaxis()
    if property.is_mask():
        cb.ax.add_patch(Rect((0, 0.5), 1, -1, lw=0, fc="none", hatch="/"))
    if not categoric:
        kwargs = {"transform": cb.ax.transAxes, "ha": "left"}
        if setup.log_scaled:
            levels = 10**levels
        ids = [-1, 0] if setup.invert_colorbar else [0, -1]
        if [bot_extra, top_extra][ids[0]]:
            cb.ax.text(1.8, -0.02, f"{levels[ids[0]]:.5g}", **kwargs, va="top")
        if [bot_extra, top_extra][ids[1]]:
            cb.ax.text(
                1.8, 1.02, f"{levels[ids[1]]:.5g}", **kwargs, va="bottom"
            )

    factor_str = rf"$10^{{{int(scale_exp)}}}$" if scale_exp >= 3 else ""
    if property.get_output_unit():
        unit_str = f" / {factor_str} {property.get_output_unit()}"
    else:
        unit_str = factor_str
    cb.set_label(
        property.output_name.replace("_", " ") + unit_str,
        size=setup.rcParams_scaled["font.size"],
    )
    cb.ax.tick_params(
        labelsize=setup.rcParams_scaled["font.size"], direction="out"
    )
    cb.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cb.ax.ticklabel_format(useOffset=False, style="plain")

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

    property = resolve_property(property)
    if mesh.get_cell(0).dimension == 3:
        msg = "meshplotlib is for 2D meshes only, but found 3D elements."
        raise TypeError(msg)

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


def _fig_init(rows: int, cols: int) -> mfigure.Figure:
    figsize = np.array(setup.figsize) * setup.fig_scale
    fig, _ = plt.subplots(
        rows, cols, dpi=setup.dpi * setup.fig_scale, figsize=figsize
    )
    fig.patch.set_alpha(1)
    return fig


def get_combined_levels(
    meshes: np.ndarray, property: Union[Property, str]
) -> np.ndarray:
    property = resolve_property(property)
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
    if all(val.is_integer() for val in unique_vals):
        return unique_vals[(p_min <= unique_vals) & (unique_vals <= p_max)]
    return get_levels(p_min, p_max, setup.num_levels)


def resolve_property(property: Union[Property, str]) -> Property:
    if isinstance(property, Property):
        return property
    return THM.find_property(property)


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
    levels = get_combined_levels(np_meshes, property)

    for i in range(shape[0]):
        for j in range(shape[1]):
            subplot(np_meshes[i, j], property, np_axs[i, j], levels)

    np_axs[0, 0].set_title(setup.title_center, loc="center", y=1.02)
    np_axs[0, 0].set_title(setup.title_left, loc="left", y=1.02)
    np_axs[0, 0].set_title(setup.title_right, loc="right", y=1.02)
    # make extra space for the upper limit of the colorbar
    plt.tight_layout(pad=1.4)
    add_colorbar(fig, property, levels)

    return fig


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

    property = resolve_property(property)
    rcParams.update(setup.rcParams_scaled)
    shape = _get_rows_cols(meshes)
    _fig = _fig_init(*shape)
    fig = _plot_on_figure(_fig, meshes, property)
    for ax in fig.axes[:-1]:
        aspect = setup.ax_aspect_ratio
        if aspect is None:
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            aspect = abs(xlims[1] - xlims[0]) / abs(ylims[1] - ylims[0])
        ax.set_aspect(aspect)
    return fig
