# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"Functions related to stress based integrity analysis."

import numpy as np

from .tensor_math import eigenvalues, mean, octahedral_shear


def fluid_pressure_criterion(
    stress: np.ndarray,
    pressure: np.ndarray,
    biot: float = 1.0,
) -> np.ndarray:
    """Calculates the maximum effective principal stress.

    The fluid pressure criterion is fulfilled when the third principal effective
    stress (minimum compressive stress / maximum tensile stress) is larger then
    the fluid pressure:

    .. math::

        \\sigma_\\mathrm{III}' =  \\sigma_\\mathrm{III}^\\mathrm{tot} + \\alpha_B \\cdot p_\\mathrm{fl} < 0
    """
    min_compressive_stress = eigenvalues(stress)[..., 2]
    return min_compressive_stress + biot * pressure


def dilatancy_critescu(
    stress: np.ndarray,
    pressure: np.ndarray | None = None,
    a: float = -0.01697,
    b: float = 0.8996,
) -> np.ndarray:
    """Compute the dilatancy criterion.

    Requires "sigma" and "pressure" to be in the mesh (in Pa).

    For total stresses it is defined as:

    .. math::

        F_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma_m}{\\sigma_0}

    For effective stresses it is defined as:

    .. math::

        F'_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma'_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma'_m}{\\sigma_0}

    <https://www.sciencedirect.com/science/article/pii/S0360544222000512?via%3Dihub>
    """
    sigma_0 = 1e6
    sigma_m = mean(-stress)
    if pressure is not None:
        sigma_m -= pressure
    tau_oct = octahedral_shear(-stress)
    return (
        tau_oct / sigma_0 - a * (sigma_m / sigma_0) ** 2 - b * sigma_m / sigma_0
    )


def dilatancy_alkan(
    stress: np.ndarray,
    pressure: np.ndarray | None = None,
    b: float = 0.04,
    tau_max: float = 33e6,
) -> np.ndarray:
    """Compute the dilatancy criterion.

    Requires "sigma" and "pressure" to be in the mesh (in Pa).

    For total stresses it is defined as:

    .. math::

        F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma'_m}{\\sigma_0 + b \\cdot \\sigma'_m}

    For effective stresses it is defined as:

    .. math::

        F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma'_m}{\\sigma_0 + b \\cdot \\sigma'_m}

    <https://www.sciencedirect.com/science/article/pii/S1365160906000979>
    """
    sigma_m = mean(-stress)
    if pressure is not None:
        sigma_m -= pressure
    tau = octahedral_shear(-stress)
    return tau - tau_max * (b * sigma_m / (1e6 + b * sigma_m))
