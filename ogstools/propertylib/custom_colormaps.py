"Definitions of custom colormaps."

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap as LSC

mask_cmap = LSC.from_list("mask_cmap", ["lightgrey", "g"])
temperature_cmap = LSC.from_list(
    "temperature_cmap",
    np.r_[
        colormaps["Blues"](np.linspace(0, 0.75, 128, endpoint=True)),
        colormaps["plasma"](np.linspace(0, 1, 128, endpoint=True)),
    ],
)
