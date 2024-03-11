"Definitions of custom colormaps."

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap as LSC

mask_cmap = LSC.from_list("mask_cmap", ["lightgrey", "g"])
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
