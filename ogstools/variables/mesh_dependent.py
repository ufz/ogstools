# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"Functions related to stress analysis which can be only applied to a mesh."

from collections.abc import Sequence

import numpy as np
import pyvista as pv

from .tensor_math import eigenvalues, mean, octahedral_shear


def fluid_pressure_criterion(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Compute the fluid pressure criterion.

    Requires "sigma" and "pressure" to be in the mesh and having the same units.
    The criterion is defined as:

        fluid pressure - minimal principal stress (compression positive).

    .. math::

        F_{p} = p_{fl} - \\sigma_{min}
    """
    sig_min = eigenvalues(-mesh["sigma"])[..., 0]
    return mesh["pressure"] - sig_min


def dilatancy_critescu(
    mesh: pv.UnstructuredGrid,
    a: float = -0.01697,
    b: float = 0.8996,
    effective: bool = False,
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

    sigma = -mesh["sigma"]
    sigma_0 = 1e6
    sigma_m = mean(sigma)
    if effective:
        sigma_m -= mesh["pressure"]
    tau_oct = octahedral_shear(sigma)
    return (
        tau_oct / sigma_0 - a * (sigma_m / sigma_0) ** 2 - b * sigma_m / sigma_0
    )


def dilatancy_alkan(
    mesh: pv.UnstructuredGrid,
    b: float = 0.04,
    tau_max: float = 33e6,
    effective: bool = False,
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

    sigma = -mesh["sigma"]
    sigma_m = mean(sigma)
    if effective:
        sigma_m -= mesh["pressure"]
    tau = octahedral_shear(sigma)
    return tau - tau_max * (b * sigma_m / (1e6 + b * sigma_m))
