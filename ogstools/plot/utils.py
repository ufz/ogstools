# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from math import nextafter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from cycler import Cycler
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from typeguard import typechecked

from ogstools.plot.levels import level_boundaries
from ogstools.propertylib.properties import Property


def get_style_cycler(
    min_number_of_styles: int,
    colors: list | None | None = None,
    linestyles: list | None = None,
) -> Cycler:
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if linestyles is None:
        linestyles = ["-", "--", ":", "-."]
    styles_len = min(len(colors), len(linestyles))
    c_cycler = plt.cycler(color=colors)
    ls_cycler = plt.cycler(linestyle=linestyles)
    if min_number_of_styles <= styles_len:
        style_cycler = c_cycler[:styles_len] + ls_cycler[:styles_len]
    else:
        style_cycler = ls_cycler * c_cycler
    return style_cycler


def justified_labels(points: np.ndarray) -> list[str]:
    "Formats an array of points to a list of aligned str."

    def fmt(val: float) -> str:
        return f"{val:.2f}".rstrip("0").rstrip(".")

    col_lens = np.max(
        [[len(fmt(coord)) for coord in point] for point in points], axis=0
    )
    dim = points.shape[1]
    return [
        ",".join(fmt(point[i]).rjust(col_lens[i]) for i in range(dim))
        for point in points
    ]


@typechecked
def label_spatial_axes(
    axes: plt.Axes | np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    spatial_unit: str = "m",
) -> None:
    """
    Add labels to x and y axis.

    If given an array of axes, only the outer axes will be labeled.
    """
    if isinstance(axes, np.ndarray):
        ax: plt.Axes
        for ax in axes[-1, :]:
            ax.set_xlabel(f"{x_label} / {spatial_unit}")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{y_label} / {spatial_unit}")
    else:
        axes.set_xlabel(f"{x_label} / {spatial_unit}")
        axes.set_ylabel(f"{y_label} / {spatial_unit}")


def update_font_sizes(
    fontsize: int = 20,
    fig: plt.Figure | None = None,
    ax: plt.Axes | np.ndarray | None = None,
) -> None:
    """
    Update font sizes of labels and ticks in all subplots

    :param fontsize: font size for the labels and ticks
    :param fig: matplotlib figure which should be updated
    :param ax: matplotlib axes which should be updated
    """
    match fig, ax:
        case None, None:
            err_msg = "Neither Figure nor Axes was provided!"
            raise ValueError(err_msg)
        case fig, None:
            assert isinstance(fig, plt.Figure)
            axes = fig.get_axes()
        case None, ax:
            axes = [ax] if isinstance(ax, plt.Axes) else np.ravel(ax)
        case fig, ax:
            err_msg = "Please only provide a figure OR axes!"
            raise ValueError(err_msg)

    for subax in axes:
        subax.tick_params(axis="both", which="major", labelsize=fontsize)
        subax.tick_params(axis="both", which="minor", labelsize=fontsize)
        subax.xaxis.label.set_fontsize(fontsize)
        subax.yaxis.label.set_fontsize(fontsize)
        subax.yaxis.get_offset_text().set_fontsize(fontsize)
    return


def get_data_aspect(mesh: pv.UnstructuredGrid) -> float:
    """
    Calculate the data aspect ratio of a 2D mesh.
    """
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(mean_normal))
    x_id, y_id = 2 * np.delete([0, 1, 2], projection)
    lims = mesh.bounds
    return abs(lims[x_id + 1] - lims[x_id]) / abs(lims[y_id + 1] - lims[y_id])


def get_rows_cols(
    meshes: list[pv.UnstructuredGrid]
    | np.ndarray
    | pv.UnstructuredGrid
    | pv.MultiBlock,
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


def clear_labels(axes: plt.Axes | np.ndarray) -> None:
    ax: plt.Axes
    for ax in np.ravel(np.array(axes)):
        ax.set_xlabel("")
        ax.set_ylabel("")


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


def save_animation(anim: FuncAnimation, filename: str, fps: int) -> bool:
    """
    Save a FuncAnimation with some codec presets.

    :param anim:        the FuncAnimation to be saved
    :param filename:    the name of the resulting file
    :param fps:         the number of frames per second
    """
    print("Start saving animation...")
    codec_args = (
        "-crf 28 -preset ultrafast -pix_fmt yuv420p "
        "-vf pad=ceil(iw/2)*2:ceil(ih/2)*2"
    ).split(" ")

    writer: FFMpegWriter | ImageMagickWriter | None = None
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, codec="libx265", extra_args=codec_args)
        filename += ".mp4"
    else:
        print("\nffmpeg not available. It is recommended for saving animation.")
        filename += ".gif"
        if ImageMagickWriter.isAvailable():
            writer = ImageMagickWriter()
        else:
            print(
                "ImageMagick also not available. Falling back to"
                f" {mpl.rcParams['animation.writer']}."
            )
    try:
        anim.save(filename, writer=writer)
        print("Successful!")
        return True
    except Exception as err:
        print("\nSaving Animation failed with the following error:")
        print(err)
        return False


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
    conti_norm: mcolors.TwoSlopeNorm | mcolors.Normalize
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
    boundaries = level_boundaries(levels) if mesh_property.categoric else levels
    norm = mcolors.BoundaryNorm(
        boundaries=boundaries, ncolors=len(boundaries), clip=False
    )
    return cmap, norm