# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from cycler import Cycler
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from typeguard import typechecked


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
    label_axes: str = "both",
    fig: plt.Figure | None = None,
    ax: plt.Axes | np.ndarray | None = None,
) -> None:
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
        axes = [ax]
    else:
        err_msg = "Invalid combination of Axis and Figure!"
        raise ValueError(err_msg)

    for subax in axes:
        if label_axes != "none":
            label_spatial_axes(subax)
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


def _get_rows_cols(
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
