# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import numpy as np
from scipy.special import erfc


def heat_conduction_temperature(
    x: np.ndarray, t: np.ndarray, Tb=373.15, Ta=293.15, alpha=1e-6
) -> np.ndarray:
    "Returns the temperature for 1D heatconduction"
    x = x[:, None]
    t = np.atleast_1d(t)[None, :]
    divisor = np.divide(
        1.0, (alpha * t) ** 0.5, where=t != 0, out=np.ones_like(t) * 1e6
    )
    return np.squeeze((Tb - Ta) * erfc(0.5 * x @ divisor) + Ta).T
