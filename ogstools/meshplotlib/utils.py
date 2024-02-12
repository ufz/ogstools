from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def justified_labels(points: np.ndarray) -> list[str]:
    "Formats an array of points to a list of aligned str."

    def fmt(val: float):
        return f"{val:.2f}".rstrip("0").rstrip(".")

    col_lens = np.max(
        [[len(fmt(coord)) for coord in point] for point in points], axis=0
    )
    dim = points.shape[1]
    return [
        ",".join(fmt(point[i]).rjust(col_lens[i]) for i in range(dim))
        for point in points
    ]


def get_style_cycler(
    min_number_of_styles: int,
    colors: Optional[Optional[list]] = None,
    linestyles: Optional[list] = None,
) -> plt.cycler:
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
