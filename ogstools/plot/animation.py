# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def animate(
    fig: plt.Figure, plot_func: Callable, *args: tuple[Sequence], **kwargs: Any
) -> FuncAnimation:
    """
    Create an animation by applying a plot function on a sequence of meshes.

    :param fig:         The figure on which the animation is rendered
    :param plot_func:   The function which is applied for all timevalues.
                        Expects to read a time value and a mesh.

    Positional Arguments:
        Sequences where each element corresponds to a frame. `plot_func` has to
        accept the individual elements of these sequences as arguments.
        Most common choice here would be a MeshSeries as an iterator for the
        meshes and possibly its timevalues for labeling.

    Keyword Arguments:
        - interval: Delay between frames in milliseconds (default=50).
        - repeat:   Whether the animation repeats at the end.
    """

    lengths = [len(list(arg)) for arg in args]
    assert all(
        length == lengths[0] for length in lengths
    ), "Additionally passed arguments have to have the same length!"

    def init() -> None:
        pass

    def animate_func(i: int) -> None:
        plot_func(*[arg[i] for arg in args])

    return FuncAnimation(
        fig,  # type: ignore[arg-type]
        animate_func,  # type: ignore[arg-type]
        frames=tqdm(range(lengths[0]), total=lengths[0] - 1),
        interval=kwargs.get("interval", 50),
        repeat=kwargs.get("repeat", False),
        init_func=init,  # type: ignore[arg-type]
        blit=False,
    )
