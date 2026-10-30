# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"Functions related to stress based integrity analysis."

import numpy as np

from .tensor_math import mean, octahedral_shear


def dilatancy_critescu(
    stress: np.ndarray,
    a: float = -0.01697,
    b: float = 0.8996,
    sigma_0: float = 1e6,
) -> np.ndarray:
    """Compute the dilatancy criterion. Assumes effective stress.

    .. math::

        F_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma_m}{\\sigma_0}

    <https://www.sciencedirect.com/science/article/pii/S0360544222000512?via%3Dihub>
    """
    sigma_m = mean(-stress)
    tau_oct = octahedral_shear(-stress)
    return (
        tau_oct / sigma_0 - a * (sigma_m / sigma_0) ** 2 - b * sigma_m / sigma_0
    )


def dilatancy_alkan(
    stress: np.ndarray,
    b: float = 0.04,
    tau_max: float = 33e6,
    sigma_0: float = 1e6,
) -> np.ndarray:
    """Compute the dilatancy criterion. Assumes effective stress.

    .. math::

        F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma_m}{\\sigma_0 + b \\cdot \\sigma_m}

    <https://www.sciencedirect.com/science/article/pii/S1365160906000979>
    """
    sigma_m = mean(-stress)
    tau = octahedral_shear(-stress)
    return tau - tau_max * (b * sigma_m / (sigma_0 + b * sigma_m))
