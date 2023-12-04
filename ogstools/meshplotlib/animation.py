import time
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Optional as Opt
from typing import Union

import matplotlib as mpl
import numpy as np
from matplotlib import figure as mfigure
from matplotlib.animation import FFMpegWriter, FuncAnimation, ImageMagickWriter

from ogstools.meshlib import MeshSeries
from ogstools.propertylib import Property

from . import setup
from .core import _plot_on_figure, plot, plt


def timeline(ax: plt.Axes, x: float, xticks: list[float]) -> None:
    y = 1.1
    ap = {"arrowstyle": "-", "color": "k"}
    axfr = "axes fraction"
    ax.annotate("", (1, y), (0, y), axfr, arrowprops=ap)
    align = dict(ha="center", va="center")  # noqa: C408: noqa
    for xt in xticks:
        ax.annotate("|", (xt, y), (xt, y), axfr, **align, size=28)
    align = dict(ha="center", va="center")  # noqa: C408: noqa
    style = dict(color="g", size=36, weight="bold")  # noqa: C408: noqa
    ax.annotate("|", (x, y), (x, y), axfr, **align, **style)


def animate(
    mesh_series: MeshSeries,
    property: Property,
    timesteps: Opt[Sequence] = None,
    titles: Opt[list[str]] = None,
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
    tv = mesh_series.timevalues

    def t_frac(t):
        return (t - tv[0]) / (tv[-1] - tv[0])

    xticks = [t_frac(t) for t in tv]
    fig = plot(mesh_series.read(0, False), property)

    def init() -> None:
        pass

    def animate_func(i: Union[int, float], fig: mfigure.Figure) -> None:
        index = np.argmin(np.abs(np.asarray(ts) - i))
        progress = 100 * index / (len(ts) - 1)
        bar = "█" * int(progress) + "-" * (100 - int(progress))
        end = "\n" if index == (len(ts) - 1) else "\r"
        print(f"progress: |{bar}| {progress:.2f}% complete", end=end)

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
            x = i / len(ts) if isinstance(i, int) else t_frac(i)
            timeline(fig.axes[0], x, xticks)

    _func = partial(animate_func, fig=fig)
    return FuncAnimation(
        fig, _func, ts, blit=False, interval=50, repeat=False, init_func=init
    )


def save_animation(anim: FuncAnimation, filename: str, fps: int) -> bool:
    """
    Save a FuncAnimation with some codec presets.

    :param anim:        the FuncAnimation to be saved
    :param filename:    the name of the resulting file
    :param fps:         the number of frames per second
    """
    start_time = time.time()
    print("Start saving animation...")
    codec_args = (
        "-crf 28 -preset ultrafast -pix_fmt yuv420p "
        "-vf crop='iw-mod(iw,2)':'ih-mod(ih,2)'"
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
        print("\ndone!")
        print(f"Elapsed time: {(time.time() - start_time):.2f}")
        return True
    except Exception as err:
        print("\nSaving Animation failed with the following error:")
        print(err)
        return False
