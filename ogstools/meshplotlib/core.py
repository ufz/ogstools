"""Meshplotlib core utilitites."""
from typing import Optional as Opt
from typing import Union

import numpy as np
import PIL.Image as Image
import pyvista as pv
from matplotlib import cm as mcm
from matplotlib import colormaps, rcParams
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle as Rect

from ogstools.propertylib import MatrixProperty as Matrix
from ogstools.propertylib import Property
from ogstools.propertylib import VectorProperty as Vector

from . import plot_features as pf
from . import setup
from .image_tools import trim
from .levels import get_levels


def xin_cell_data(mesh: pv.UnstructuredGrid, property: Property) -> bool:
    """Determine if the property is exclusive in cell data."""
    return (
        property.data_name in mesh.cell_data
        and property.data_name not in mesh.point_data
    )


def _q_zero_line(property: Property, levels: np.ndarray):
    return property.is_component() or (
        property.data_name == "temperature" and levels[0] < 0 < levels[-1]
    )


def get_data(
    mesh: pv.UnstructuredGrid, property: Property
) -> pv.DataSetAttributes:
    """Get the data associated with a scalar or vector property from a mesh."""
    if property.data_name in mesh.point_data:
        return mesh.point_data
    if property.data_name in mesh.cell_data:
        return mesh.cell_data
    msg = "Property not found in mesh."
    raise IndexError(msg)


def get_cmap_norm(
    levels: np.ndarray, property: Property, cell_data: bool
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """Construct a discrete colormap and norm for the property field."""
    vmin, vmax = (levels[0], levels[-1])
    bilinear = property.is_component() and vmin <= 0.0 <= vmax
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
    if cell_data:
        vmin += 0.5
        vmax += 0.5
    conti_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mid_levels = np.append((levels[:-1] + levels[1:]) * 0.5, levels[-1])
    colors = [conti_cmap(conti_norm(m_l)) for m_l in mid_levels]
    cmap = mcolors.ListedColormap(colors, name="custom")
    boundaries = levels
    if cell_data:
        boundaries = np.array(
            [
                levels[0] - 0.5 * (levels[1] - levels[0]),
                *0.5 * (levels[:-1] + levels[1:]),
                levels[-1] + 0.5 * (levels[-1] - levels[-2]),
            ]
        )
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(boundaries))
    return cmap, norm


def plot_isometric(
    mesh: pv.UnstructuredGrid,
    property: Property,
    levels: Opt[np.ndarray] = None,
) -> Image.Image:
    """Plot an isometric view of the property field on the mesh."""
    mesh = mesh.copy()
    if property.mask in mesh.cell_data and len(mesh.cell_data[property.mask]):
        mesh = mesh.ctp(True).threshold(value=[1, 1], scalars=property.mask)

    get_data(mesh, property).active_scalars_name = property.data_name
    # data = get_data(mesh, property)[property.data_name]
    _p_val = (
        property.magnitude
        if isinstance(property, (Vector, Matrix))
        else property
    )

    data = get_data(mesh, property)[property.data_name]
    get_data(mesh, property)[property.data_name] = _p_val.values(data)

    if levels is None:
        num_levels = min(setup.num_levels, len(np.unique(data)))
        levels = get_levels(np.nanmin(data), np.nanmax(data), num_levels)
    cmap = get_cmap_norm(levels, property, xin_cell_data(mesh, property))[0]

    # add arg show_edges=True if you want to see the cell edges
    # mesh = mesh.scale([1.0, 1.0, 15.0], inplace=False)
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh, cmap=cmap, clim=[levels[0], levels[-1]], lighting=False)
    p.add_mesh(mesh.extract_feature_edges(), color="black")
    mesh_surf = mesh.extract_surface()
    if setup.show_region_bounds and "MaterialIDs" in mesh.cell_data:
        for mat_id in np.unique(mesh.cell_data["MaterialIDs"]):
            mesh_id = mesh_surf.threshold(mat_id, "MaterialIDs")
            p.add_mesh(mesh_id.extract_feature_edges(), color="k")
    p.camera.azimuth += 270
    p.remove_scalar_bar()
    return trim(Image.fromarray(p.screenshot(filename=None)), 50)


def add_colorbar(
    fig: mfigure.Figure,
    property: Property,
    cell_data: bool,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    levels: np.ndarray,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    cm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    scale_mag = np.median(np.abs(np.diff(levels)))
    scale_exp = np.ceil(np.log10(scale_mag)) if scale_mag > 1e-12 else 0
    if abs(scale_exp) >= 3:
        levels *= 10 ** (-scale_exp)
        norm.vmin *= 10 ** (-scale_exp)
        norm.vmax *= 10 ** (-scale_exp)
    ticks = levels if cell_data else levels[1:-1]

    _axs = np.ravel(fig.axes).tolist()
    kwargs = {"location": "right", "spacing": "proportional", "pad": 0.02}
    cb = fig.colorbar(
        cm, norm=norm, ax=_axs, ticks=ticks, drawedges=True, **kwargs
    )
    if setup.invert_colorbar:
        cb.ax.invert_yaxis()
    if property.is_mask():
        cb.ax.add_patch(Rect((0, 0.5), 1, -1, lw=0, fc="none", hatch="/"))
    if not cell_data:
        kwargs = {"transform": cb.ax.transAxes, "ha": "left"}
        if setup.log_scaled:
            levels = 10**levels
        cb.ax.text(0, -0.02, f"{levels[0]:.3g}", **kwargs, va="top")
        cb.ax.text(0, 1.02, f"{levels[-1]:.3g}", **kwargs, va="bottom")

    unit_str = ""
    factor_str = rf"$10^{{{int(scale_exp)}}}$" if abs(scale_exp) >= 3 else ""
    if factor_str:
        unit_str += " " + factor_str
    if property.get_output_unit():
        unit_str += " / " + property.get_output_unit()
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


def subplot(
    mesh: pv.UnstructuredGrid,
    property: Property,
    ax: plt.Axes,
    levels: Opt[np.ndarray] = None,
) -> None:
    """
    Plot the property field of a mesh on a matplotlib.axis.

    In 3D the mesh gets sliced according to slice_type
    and the origin in the PlotSetup in meshplotlib.setup.
    Custom levels and a colormap string can be provided.
    """

    if mesh.get_cell(0).dimension == 3:
        ax.imshow(plot_isometric(mesh, property, levels))
        ax.axis("off")
        return

    ax.axis(setup.scale_type)

    if property.mask in mesh.cell_data and len(mesh.cell_data[property.mask]):
        subplot(mesh, property.get_mask(), ax)
        mesh = mesh.ctp(True).threshold(value=[1, 1], scalars=property.mask)

    surf_tri = mesh.triangulate().extract_surface()

    # get projection
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = np.delete([0, 1, 2], projection)

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    x, y = setup.length.values(surf_tri.points.T[[x_id, y_id]])
    tri = surf_tri.faces.reshape((-1, 4))[:, 1:]
    _property = (
        property.magnitude
        if isinstance(property, (Vector, Matrix))
        else property
    )
    values = _property.values(get_data(surf_tri, property)[property.data_name])
    if setup.log_scaled:
        values_temp = np.where(values > 1e-14, values, 1e-14)
        values = np.log10(values_temp)
    p_min, p_max = np.nanmin(values), np.nanmax(values)

    if levels is None:
        num_levels = min(setup.num_levels, len(np.unique(values)))
        levels = get_levels(p_min, p_max, num_levels)
    cmap, norm = get_cmap_norm(levels, property, xin_cell_data(mesh, property))

    if xin_cell_data(mesh, property):
        ax.tripcolor(x, y, tri, facecolors=values, cmap=cmap, norm=norm)
        if property.is_mask():
            ax.tripcolor(
                x,
                y,
                tri,
                facecolors=values,
                mask=(values == 1),
                cmap=cmap,
                norm=norm,
                hatch="/",
            )
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


def _get_shape(
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid]
) -> tuple[int, ...]:
    if isinstance(meshes, np.ndarray):
        if meshes.ndim in [1, 2]:
            return meshes.shape
        msg = "Input numpy array must be 1D or 2D."
        raise ValueError(msg)
    if isinstance(meshes, list):
        return (1, len(meshes))
    return (1, 1)


def _fig_init(shape: tuple[int, ...]) -> tuple[mfigure.Figure, list[plt.Axes]]:
    figsize = np.array(setup.figsize) * setup.fig_scale
    fig, _ = plt.subplots(
        shape[0], shape[1], dpi=setup.dpi * setup.fig_scale, figsize=figsize
    )
    fig.patch.set_alpha(1)
    return fig


def _plot(
    fig: mfigure.Figure,
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid],
    property: Property,
) -> mfigure.Figure:
    """
    Plot the property field of meshes with default settings.

    The resulting figure adheres to the configurations in meshplotlib.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes: Singular mesh of 2D numpy array of meshes
    :param property: the property field to be visualized on all meshes
    """

    shape = _get_shape(meshes)

    _p_val = (
        property.magnitude
        if isinstance(property, (Vector, Matrix))
        else property
    )
    p_min, p_max, n_values = np.inf, -np.inf, 0
    _meshes = np.reshape(meshes, shape)
    _axs = np.reshape(fig.axes, shape)
    for mesh in np.ravel(_meshes):
        if get_data(mesh, property) is None:
            print("a mesh doesn't contain the requested property.")
            return None
        values = _p_val.values(get_data(mesh, property)[property.data_name])
        if setup.log_scaled:
            values_temp = np.where(values > 1e-14, values, 1e-14)
            values = np.log10(values_temp)
        p_min = min(p_min, np.nanmin(values)) if setup.p_min is None else p_min
        p_max = max(p_max, np.nanmax(values)) if setup.p_max is None else p_max
        n_values = max(n_values, len(np.unique(values)))
    num_levels = min(setup.num_levels, n_values)
    p_min = setup.p_min if setup.p_min is not None else p_min
    p_max = setup.p_max if setup.p_max is not None else p_max
    levels = get_levels(p_min, p_max, num_levels)

    for i in range(shape[0]):
        for j in range(shape[1]):
            subplot(_meshes[i, j], property, _axs[i, j], levels)
    # for ax in fig.axes[:-1]:
    #     ax.set_xlabel("")

    cell_data = xin_cell_data(_meshes[0, 0], property)
    cmap, norm = get_cmap_norm(levels, property, cell_data)

    _axs[0, 0].set_title(setup.title_center, loc="center", y=1.02)
    _axs[0, 0].set_title(setup.title_left, loc="left", y=1.02)
    _axs[0, 0].set_title(setup.title_right, loc="right", y=1.02)
    plt.tight_layout()
    add_colorbar(fig, property, cell_data, cmap, norm, levels)

    return fig


def plot(
    meshes: Union[list[pv.UnstructuredGrid], np.ndarray, pv.UnstructuredGrid],
    property: Property,
) -> mfigure.Figure:
    """
    Plot the property field of meshes with default settings.

    The resulting figure adheres to the configurations in meshplotlib.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes: Singular mesh of 2D numpy array of meshes
    :param property: the property field to be visualized on all meshes
    """

    rcParams.update(setup.rcParams_scaled)
    shape = _get_shape(meshes)
    fig = _fig_init(shape)
    return _plot(fig, meshes, property)
