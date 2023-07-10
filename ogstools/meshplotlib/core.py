"""Meshplotlib core utilitites."""

from typing import Optional as Opt
from typing import Union

import numpy as np
import pyvista as pv
from matplotlib import cm as mcm
from matplotlib import colormaps, rcParams
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

from ogstools.propertylib import ScalarProperty as Scalar
from ogstools.propertylib import VectorProperty as Vector

from . import image_tools, setup
from . import plot_features as pf
from .levels import get_levels
from .mesh import Mesh

Property = Union[Scalar, Vector]


def xin_cell_data(mesh: pv.UnstructuredGrid, property: Property) -> bool:
    """Determine if the property is exclusive in cell data."""
    return (
        property.data_name in mesh.cell_data
        and property.data_name not in mesh.point_data.keys()
    )


def get_data(mesh: pv.UnstructuredGrid, property: Property) -> pv.DataSet:
    """Get the data associated with a scalar or vector property from a mesh."""
    if property.data_name in mesh.point_data:
        return mesh.point_data
    if property.data_name in mesh.cell_data:
        return mesh.cell_data
    msg = "Property not found in mesh."
    raise IndexError(msg)


def get_cmap_norm(
    levels: np.ndarray, property: Scalar, cell_data: bool
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """Construct a discrete colormap and norm for the property field."""
    vmin, vmax = (levels[0], levels[-1])
    bilinear = property.is_component() and vmin <= 0.0 <= vmax
    cmap_str = setup.cmap_str(property)
    if isinstance(cmap_str, list):
        continuous_cmap = mcolors.ListedColormap(cmap_str)
    else:
        continuous_cmap = colormaps[cmap_str]
    if property.data_name == "temperature" and vmin <= 0.0 <= vmax:
        ice_cmap = mcolors.LinearSegmentedColormap.from_list(
            "ice_cmap", ["blue", "lightblue"], N=128
        )
        temp_colors = np.vstack(
            (
                ice_cmap(np.linspace(0, 1, 128)),
                continuous_cmap(np.linspace(0, 1, 128)),
            )
        )
        continuous_cmap = mcolors.LinearSegmentedColormap.from_list(
            "temperature_cmap", temp_colors
        )
        bilinear = True
    if bilinear:
        vmin, vmax = np.max(np.abs([vmin, vmax])) * np.array([-1.0, 1.0])
    if cell_data:
        vmin += 0.5
        vmax += 0.5
    continuous_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mid_levels = np.append((levels[:-1] + levels[1:]) * 0.5, levels[-1])
    colors = [continuous_cmap(continuous_norm(m_l)) for m_l in mid_levels]
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
    mesh: Mesh, property: Property, levels: Opt[np.ndarray] = None
) -> image_tools.Image:
    """Plot an isometric view of the property field on the mesh."""
    mesh = mesh.copy()
    if property.mask in mesh.cell_data and len(mesh.cell_data[property.mask]):
        mesh = mesh.ctp(True).threshold(value=[1, 1], scalars=property.mask)

    get_data(mesh, property).active_scalars_name = property.data_name
    # data = get_data(mesh, property)[property.data_name]
    _p_val = property.magnitude if isinstance(property, Vector) else property

    data = get_data(mesh, property)[property.data_name]
    get_data(mesh, property)[property.data_name] = _p_val.values(data)

    if levels is None:
        num_levels = min(setup.num_levels, len(np.unique(data)))
        levels = get_levels(np.nanmin(data), np.nanmax(data), num_levels)
    cmap = get_cmap_norm(levels, property, xin_cell_data(mesh, property))[0]

    # add arg show_edges=True if you want to see the cell edges
    # mesh = mesh.scale([1.0, 1.0, 15.0], inplace=False)
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, notebook=False)
    p.add_mesh(mesh, cmap=cmap, clim=[levels[0], levels[-1]], lighting=False)
    p.add_mesh(mesh.extract_feature_edges(), color="black")
    mesh_surf = mesh.extract_surface()
    if setup.show_layer_bounds and "MaterialIDs" in mesh.cell_data:
        for mat_id in np.unique(mesh.cell_data["MaterialIDs"]):
            mesh_id = mesh_surf.threshold(mat_id, "MaterialIDs")
            p.add_mesh(mesh_id.extract_feature_edges(), color="k")
    p.camera.azimuth += 270
    p.remove_scalar_bar()
    p.show()
    return image_tools.trim(image_tools.Image.fromarray(p.image), 50)


def add_colorbar(
    fig: mfigure.Figure,
    property: Scalar,
    cell_data: bool,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    levels: np.ndarray,
) -> None:
    """Add a colorbar to the matplotlib figure."""
    cm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    scale_mag = np.median(np.abs(np.diff(levels)))
    scale_exp = np.log10(scale_mag) if scale_mag > 1e-12 else 0
    if abs(scale_exp) >= 3:
        levels *= 10 ** (-scale_exp)
        norm.vmin *= 10 ** (-scale_exp)
        norm.vmax *= 10 ** (-scale_exp)
    ticks = levels if cell_data else levels[1:-1]

    cb = fig.colorbar(
        cm,
        norm=norm,
        ax=np.ravel(fig.axes).tolist(),
        ticks=ticks,
        drawedges=True,
        location="right",
        spacing="proportional",
        pad=0.02,
    )
    if property.is_mask():
        cb.ax.add_patch(
            mpatches.Rectangle(
                (0, 0.5), 1, -1, linewidth=0, facecolor="none", hatch="/"
            )
        )
    if not cell_data:
        cb.ax.text(
            0.5,
            -0.01,
            f"{levels[0]:.3g}",
            transform=cb.ax.transAxes,
            va="top",
            ha="center",
        )
        cb.ax.text(
            0.5,
            1.005,
            f"{levels[-1]:.3g}",
            transform=cb.ax.transAxes,
            va="bottom",
            ha="center",
        )

    unit_str = ""
    factor_str = rf"$10^{{{int(scale_exp)}}}$" if abs(scale_exp) >= 3 else ""
    if factor_str:
        unit_str += " " + factor_str
    if property.get_output_unit():
        unit_str += " / " + property.get_output_unit()
    cb.set_label(
        property.output_name + unit_str, size=setup.rcParams_scaled["font.size"]
    )
    cb.ax.tick_params(labelsize=setup.rcParams_scaled["font.size"])
    cb.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    cb.ax.ticklabel_format(useOffset=False, style="plain")

    if property.is_component():
        cb.ax.axhline(
            y=0, color="w", lw=2 * setup.rcParams_scaled["lines.linewidth"]
        )


def subplot(
    mesh: Mesh,
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
    projection = np.argmax(mean_normal)
    x_id, y_id = np.delete([0, 1, 2], projection)

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    x, y = setup.length.values(surf_tri.points.T[[x_id, y_id]])
    tri = surf_tri.faces.reshape((-1, 4))[:, 1:]
    _property = property.magnitude if isinstance(property, Vector) else property
    values = _property.values(get_data(surf_tri, property)[property.data_name])
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
        if property.is_component():
            ax.tricontour(x, y, tri, values, levels=[0], colors="w")

    surf = mesh.extract_surface()

    if setup.show_layer_bounds and "MaterialIDs" in mesh.cell_data:
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
            origin = mesh.center
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

    ax.set_xlabel(f'{"xyz"[x_id]} / {setup.length.output_unit}')
    ax.set_ylabel(f'{"xyz"[y_id]} / {setup.length.output_unit}')


def plot(
    meshes: Union[np.ndarray[Mesh], Mesh], property: Property
) -> mfigure.Figure:
    """
    Plot the property field of meshes with default settings.

    The resulting figure adheres to the configurations in meshplotlib.setup.
    For 2D, the whole domain, for 3D a set of slices is displayed.

    :param meshes: Singular mesh of 2D numpy array of meshes
    :param property: the property field to be visualized on all meshes
    """

    rcParams.update(setup.rcParams_scaled)

    if len(np.shape(meshes)) == 0:
        meshes = np.array([[meshes]])
    if len(np.shape(meshes)) != 2:
        msg = "Meshes must be a single mesh or a 2D np.array of meshes."
        raise TypeError(msg)
    if isinstance(meshes, list):
        meshes = np.array(meshes)
    figsize = np.array(setup.figsize) * setup.fig_scale
    fig, _axs = plt.subplots(
        len(meshes),
        len(meshes[0]),
        dpi=setup.dpi * setup.fig_scale,
        figsize=figsize,
    )
    fig.patch.set_alpha(1)
    axs: np.ndarray[plt.Axes] = np.reshape(_axs, [len(meshes), len(meshes[0])])

    _p_val = property.magnitude if isinstance(property, Vector) else property
    p_min, p_max, n_values = np.inf, -np.inf, 0
    for mesh in np.ravel(meshes):
        if get_data(mesh, property) is None:
            print("a mesh doesn't contain the requested property.")
            return None
        values = _p_val.values(get_data(mesh, property)[property.data_name])
        if setup.p_min is None:
            p_min = min(p_min, np.nanmin(values))
        if setup.p_max is None:
            p_max = max(p_max, np.nanmax(values))
        n_values = max(n_values, len(np.unique(values)))
    num_levels = min(setup.num_levels, n_values)
    levels = get_levels(p_min, p_max, num_levels)

    for i in range(len(meshes)):
        for j in range(len(meshes[0])):
            subplot(meshes[i, j], property, axs[i, j], levels)
    # for ax in fig.axes[:-1]:
    #     ax.set_xlabel("")

    cmap, norm = get_cmap_norm(
        levels, property, xin_cell_data(meshes[0, 0], property)
    )

    plt.tight_layout()

    add_colorbar(
        fig, property, xin_cell_data(meshes[0, 0], property), cmap, norm, levels
    )

    return fig
