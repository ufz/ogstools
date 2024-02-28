"""Meshplotlib core utilitites."""

from copy import deepcopy
from math import nextafter
from typing import Literal, Optional, Union

import numpy as np
import pyvista as pv
from matplotlib import cm as mcm
from matplotlib import colormaps, rcParams
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle as Rect

from ogstools.meshlib import MeshSeries
from ogstools.propertylib import Property, Vector
from ogstools.propertylib.presets import get_preset
from ogstools.propertylib.unit_registry import u_reg

from . import plot_features as pf
from . import setup
from .levels import get_levels, get_median_exponent
from .utils import get_style_cycler

# TODO: define default data_name for regions in setup


def _q_zero_line(mesh_property: Property, levels: np.ndarray):
    return mesh_property.bilinear_cmap or (
        mesh_property.data_name == "temperature" and levels[0] < 0 < levels[-1]
    )


def get_level_boundaries(levels: np.ndarray):
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
        boundaries=boundaries, ncolors=len(boundaries), clip=True
    )
    return cmap, norm


def get_ticklabels(ticks: np.ndarray) -> tuple[list[str], Optional[str]]:
    """Get formatted tick labels and optional offset str.

    If all values in ticks are too close together offset notation is used.
    """
    fmt = ".3g"
    # "+ 0" prevents output of negative zero, i.e. "-0"
    tick_labels = [f"{0.0 + round(tick, 12):{fmt}}" for tick in ticks]
    if len(tick_labels[1:-1]) != len(set(tick_labels[1:-1])) and len(ticks) > 2:
        label_lens = np.asarray([len(f"{tick:{fmt}}") for tick in ticks])
        offset = ticks[np.argmin(label_lens)]
        new_fmt = (
            "g" if abs(get_median_exponent(ticks - offset)) <= 2 else ".1e"
        )
        tick_labels = [
            f"{0.0 +  round(tick, 14):{new_fmt}}" for tick in ticks - offset
        ]
        return (tick_labels, f"{offset:{fmt}}")

    # pretty hacky but seems to do the job
    def _get_label(x, precision):
        return f"{0.0 + round(x, precision)}"

    for idx, adj in [(0, 1), (-1, -2)]:
        if tick_labels[idx] != tick_labels[adj]:
            continue
        for precision in range(12):
            new_ticklabel = _get_label(ticks[idx], precision)
            adj_ticklabel = _get_label(ticks[adj], precision)
            if float(new_ticklabel) != float(adj_ticklabel):
                tick_labels[idx] = new_ticklabel
                break
    return tick_labels, None


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
        cm, norm=norm, ax=ax, ticks=ticks, drawedges=True, location="right",
        spacing="uniform", pad=pad  # fmt: skip
    )
    # Formatting the colorbar label and ticks

    tick_labels, offset = get_ticklabels(ticks)
    cb_label = mesh_property.output_name.replace("_", " ")
    if (unit := mesh_property.get_output_unit()) != "":
        cb_label += " / " + unit
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

    if isinstance(mesh_property, str):
        data_shape = mesh[mesh_property].shape
        mesh_property = get_preset(mesh_property, data_shape)
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
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = np.delete([0, 1, 2], projection)

    # faces contains a padding indicating number of points per face which gets
    # removed with this reshaping and slicing to get the array of tri's
    x, y = setup.length(surf_tri.points.T[[x_id, y_id]])
    tri = surf_tri.faces.reshape((-1, 4))[:, 1:]
    values = mesh_property.magnitude(surf_tri)
    if setup.log_scaled:
        values_temp = np.where(values > 1e-14, values, 1e-14)
        values = np.log10(values_temp)
    p_min, p_max = np.nanmin(values), np.nanmax(values)

    if levels is None:
        num_levels = min(setup.num_levels, len(np.unique(values)))
        levels = get_levels(p_min, p_max, num_levels)
    cmap, norm = get_cmap_norm(levels, mesh_property)

    if (
        mesh_property.data_name in mesh.cell_data
        and mesh_property.data_name not in mesh.point_data
    ):
        ax.tripcolor(x, y, tri, facecolors=values, cmap=cmap, norm=norm)
        if mesh_property.is_mask():
            ax.tripcolor(x, y, tri, facecolors=values, mask=(values == 1),
                         cmap=cmap, norm=norm, hatch="/")  # fmt: skip
    else:
        ax.tricontourf(x, y, tri, values, levels=levels, cmap=cmap, norm=norm)
        if _q_zero_line(mesh_property, levels):
            ax.tricontour(x, y, tri, values, levels=[0], colors="w")

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


def _fig_init(rows: int, cols: int, aspect: float = 1.0) -> mfigure.Figure:
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
    meshes: np.ndarray, mesh_property: Union[Property, str]
) -> np.ndarray:
    """
    Calculate well spaced levels for the encompassing property range in meshes.
    """
    if isinstance(mesh_property, str):
        data_shape = meshes[0][mesh_property].shape
        mesh_property = get_preset(mesh_property, data_shape)
    p_min, p_max = np.inf, -np.inf
    unique_vals = np.array([])
    for mesh in np.ravel(meshes):
        values = mesh_property.magnitude(mesh)
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
    mesh_property: Property,
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
        combined_levels = get_combined_levels(np_meshes, mesh_property)

    for i in range(shape[0]):
        for j in range(shape[1]):
            _levels = (
                combined_levels
                if setup.combined_colorbar
                else get_combined_levels(np_meshes[i, j], mesh_property)
            )
            subplot(np_meshes[i, j], mesh_property, np_axs[i, j], _levels)

    np_axs[0, 0].set_title(setup.title_center, loc="center", y=1.02)
    np_axs[0, 0].set_title(setup.title_left, loc="left", y=1.02)
    np_axs[0, 0].set_title(setup.title_right, loc="right", y=1.02)
    # make extra space for the upper limit of the colorbar
    if setup.layout == "tight":
        plt.tight_layout(pad=1.4)

    if setup.combined_colorbar:
        cb_axs = np.ravel(fig.axes).tolist()
        add_colorbars(
            fig, cb_axs, mesh_property, combined_levels, pad=0.05 / shape[1]
        )
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                _levels = get_combined_levels(np_meshes[i, j], mesh_property)
                add_colorbars(fig, np_axs[i, j], mesh_property, _levels)

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
    mesh_property: Union[Property, str],
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
    if isinstance(mesh_property, str):
        data_shape = _meshes[0][mesh_property].shape
        mesh_property = get_preset(mesh_property, data_shape)
    data_aspects = np.asarray([get_data_aspect(mesh) for mesh in _meshes])
    if setup.min_ax_aspect is None and setup.max_ax_aspect is None:
        fig_aspect = np.mean(data_aspects)
    else:
        fig_aspect = np.mean(
            np.clip(data_aspects, setup.min_ax_aspect, setup.max_ax_aspect)
        )
    ax_aspects = fig_aspect / data_aspects
    _fig = _fig_init(rows=shape[0], cols=shape[1], aspect=fig_aspect)
    n_axs = shape[0] * shape[1]
    fig = _plot_on_figure(_fig, meshes, mesh_property)
    for ax, aspect in zip(fig.axes[: n_axs + 1], ax_aspects):
        ax.set_aspect(1.0 / aspect)
    return fig


def plot_diff(
    mesh1: pv.UnstructuredGrid,
    mesh2: pv.UnstructuredGrid,
    mesh_property: Union[Property, str],
) -> mfigure.Figure:
    if isinstance(mesh_property, str):
        data_shape = mesh1[mesh_property].shape
        mesh_property = get_preset(mesh_property, data_shape)
    diff_mesh = deepcopy(mesh1)
    diff_mesh[mesh_property.data_name] -= mesh2[mesh_property.data_name]
    data_property = mesh_property.replace(output_unit=mesh_property.data_unit)
    diff_unit = str(
        (
            data_property(1, strip_unit=False)
            - data_property(1, strip_unit=False)
        ).units
    )
    diff_property = mesh_property.replace(
        data_unit=diff_unit,
        output_unit=diff_unit,
        output_name=mesh_property.output_name + " difference",
        bilinear_cmap=True,
        cmap=mesh_property.cmap if mesh_property.bilinear_cmap else "coolwarm",
    )
    return plot(diff_mesh, diff_property)


def plot_limit(
    mesh_series: MeshSeries,
    mesh_property: Union[Property, str],
    limit: Literal["min", "max"],
) -> mfigure.Figure:
    """
    Plot the property limits through all timesteps of a MeshSeries.

    :param mesh_series: MeshSeries object containing the data to be plotted
    :param property:    The property field to be evaluated
    :param limit:       Type of limit to be computed

    :returns:   A matplotlib Figure
    """
    mesh = mesh_series.read(0)
    if isinstance(mesh_property, str):
        data_shape = mesh[mesh_property].shape
        mesh_property = get_preset(mesh_property, data_shape)
    func = {"min": np.min, "max": np.max}[limit]
    vals = mesh_series.values(mesh_property.data_name)
    func(vals, out=mesh[mesh_property.data_name], axis=0)
    limit_property = mesh_property.replace(
        output_name=limit + " " + mesh_property.output_name
    )
    return plot(mesh, limit_property)


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
    **kwargs,
) -> mfigure.Figure:
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
    if isinstance(mesh_property, str):
        data_shape = mesh_series.read(0)[mesh_property].shape
        mesh_property = get_preset(mesh_property, data_shape)
    values = mesh_property.magnitude(
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
        if isinstance(mesh_property_abscissa, str):
            data_shape = mesh_series.read(0)[mesh_property_abscissa].shape
            mesh_property_abscissa = get_preset(
                mesh_property_abscissa, data_shape
            )
        x_values = mesh_property_abscissa.magnitude(
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
    ax.grid(which="major", color="lightgrey", linestyle="-")
    ax.grid(which="minor", color="0.95", linestyle="--")
    unit_str = (
        f" / {mesh_property.get_output_unit()}"
        if mesh_property.get_output_unit()
        else ""
    )
    y_label = mesh_property.output_name.replace("_", " ") + unit_str
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.label_outer()
    ax.minorticks_on()
    return fig
