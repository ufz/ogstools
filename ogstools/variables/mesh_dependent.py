# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"Functions related to stress analysis which can be only applied to a mesh."


from collections.abc import Callable

import numpy as np
import pyvista as pv
from pint.facets.plain import PlainQuantity

from .tensor_math import _split_quantity, eigenvalues, mean, octahedral_shear
from .unit_registry import u_reg
from .variable import Variable


def get_pts(
    index: int,
) -> Callable[[pv.UnstructuredGrid, Variable], np.ndarray]:
    "Returns the coordinates of all points with the given index"

    def get_pts_coordinate(
        mesh: pv.UnstructuredGrid, _: Variable
    ) -> np.ndarray:
        return mesh.points[:, index]

    return get_pts_coordinate


def fluid_pressure_criterion(
    mesh: pv.UnstructuredGrid, variable: Variable
) -> PlainQuantity:
    """Return the fluid pressure criterion.

    Defined as the difference between fluid pressure and minimal principal
    stress (compression positive). Requires "pressure" to be available in the
    mesh's point_data.

    .. math::

        F_{p} = p_{fl} - \\sigma_{min}
    """

    Qty = u_reg.Quantity
    sigma = mesh[variable.data_name]
    pressure = mesh["pressure"]
    sig_min = _split_quantity(eigenvalues(-sigma))[0][..., 0]
    return Qty(pressure, "Pa") - Qty(sig_min, variable.data_unit)


def dilatancy_critescu(
    mesh: pv.UnstructuredGrid,
    variable: Variable,
    a: float = -0.01697,
    b: float = 0.8996,
    effective: bool = False,
) -> PlainQuantity:
    """Return the dilatancy criterion defined as:

    .. math::

        F_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma_m}{\\sigma_0}

    for total stresses and defined as:

    .. math::

        F'_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma'_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma'_m}{\\sigma_0}

    for effective stresses. Requires "pressure" to be available in the
    mesh's point_data.

    <https://www.sciencedirect.com/science/article/pii/S0360544222000512?via%3Dihub>
    """

    Qty = u_reg.Quantity
    sigma = -Qty(mesh[variable.data_name], variable.data_unit)
    sigma_0 = Qty(1, "MPa")
    sigma_m = mean(sigma)
    pressure = mesh["pressure"]
    if effective:
        sigma_m -= Qty(pressure, "Pa")
    tau_oct = octahedral_shear(sigma)
    return (
        tau_oct / sigma_0 - a * (sigma_m / sigma_0) ** 2 - b * sigma_m / sigma_0
    )


def dilatancy_alkan(
    mesh: pv.UnstructuredGrid,
    variable: Variable,
    b: float = 0.04,
    effective: bool = False,
) -> PlainQuantity | np.ndarray:
    """Return the dilatancy criterion defined as:

    .. math::

        F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma'_m}{\\sigma_0 + b \\cdot \\sigma'_m}

    for total stresses and defined as:

    .. math::

        F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma'_m}{\\sigma_0 + b \\cdot \\sigma'_m}

    for effective stresses. Requires "pressure" to be available in the
    mesh's point_data.

    <https://www.sciencedirect.com/science/article/pii/S1365160906000979>
    """

    Qty = u_reg.Quantity
    sigma = -Qty(mesh[variable.data_name], variable.data_unit)
    tau_max = Qty(33, "MPa")
    sigma_m = mean(sigma)
    pressure = mesh["pressure"]
    if effective:
        sigma_m -= Qty(pressure, "Pa")
    tau = octahedral_shear(sigma)
    return tau - tau_max * (b * sigma_m / (Qty(1, "MPa") + b * sigma_m))
