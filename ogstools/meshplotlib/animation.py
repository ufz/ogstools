import warnings
from collections.abc import Sequence
from functools import partial
from typing import Optional, Union

import matplotlib as mpl
import numpy as np
from matplotlib import figure as mfigure
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter
from tqdm.auto import tqdm

from ogstools.meshlib import MeshSeries
from ogstools.propertylib import Property

from . import setup
from .core import _plot_on_figure, plot


def animate(
    mesh_series: MeshSeries,
    property: Property,
    timesteps: Optional[Sequence] = None,
    titles: Optional[list[str]] = None,
) -> FuncAnimation:
    """
    Create an animation for a property of a mesh series with given timesteps.

    :param mesh_series: the mesh series containing the data to visualize
    :param property: the property field to be visualized on all timesteps
    :param timesteps: if sequence of int: the timesteps to animate
                      if sequence of float: the timevalues to animate
    :param titles: the title on top of the animation for each frame
    """
    setup.layout = "tight"

    ts = mesh_series.timesteps if timesteps is None else timesteps

    fig = plot(mesh_series.read(0, False), property)

    def init() -> None:
        pass

    def animate_func(i: Union[int, float], fig: mfigure.Figure) -> None:
        index = np.argmin(np.abs(np.asarray(ts) - i))

        fig.axes[-1].remove()  # remove colorbar
        for ax in np.ravel(fig.axes):
            ax.clear()
        if titles is not None:
            setup.title_center = titles[index]
        if isinstance(i, int):
            mesh = mesh_series.read(i)
        else:
            mesh = mesh_series.read_interp(i, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = _plot_on_figure(fig, mesh, property)

    _func = partial(animate_func, fig=fig)

    return FuncAnimation(
        fig,
        _func,
        frames=tqdm(ts),
        blit=False,
        interval=50,
        repeat=False,
        init_func=init,
    )


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

    writer = None
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, codec="libx265", extra_args=codec_args)
        filename += ".mp4"
    else:
        print("\nffmpeg not available. It is recommended for saving animation.")
        filename += ".gif"
        if ImageMagickWriter.isAvailable():
            writer = "imagemagick"
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
