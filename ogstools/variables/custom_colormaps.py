# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"Definitions of custom colormaps."

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.colors import ListedColormap

none_cmap = ListedColormap(name="white__cmap", colors=["lightgrey"])
mask_cmap = ListedColormap(name="mask__cmap", colors=["lightgrey", "g"])
grey_cmap = ListedColormap(name="grey__cmap", colors=["lightgrey"] * 2)
temperature_cmap = LSC.from_list(
    "temperature_cmap",
    np.r_[
        colormaps["Blues"](np.linspace(0, 0.75, 128)),
        colormaps["plasma"](np.linspace(0, 1, 128)),
    ],
)
integrity_cmap = LSC.from_list(
    "criterion_cmap",
    np.r_[
        colormaps["viridis"](np.linspace(0, 0.75, 128)),
        colormaps["autumn_r"](np.linspace(0.0, 1.0, 128)),
    ],
)
