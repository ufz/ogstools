"Functions related to stress analysis which can be only applied to a mesh."

from functools import partial
from typing import Union

import numpy as np
import pyvista as pv
from pint.facets.plain import PlainQuantity

from .property import Property
from .tensor_math import eigenvalues, mean, octahedral_shear
from .unit_registry import u_reg

ValType = Union[PlainQuantity, np.ndarray]


# TODO: find smart way to choose between y values and depth
def depth(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Return the depth values of the mesh.

    This assumes the y-axes is the vertical axis. For each point, the shortest
    distance to the top boundary is returned.
    """
    # return -mesh.points[:, 1]
    edges = mesh.extract_feature_edges()
    cell_bounds = np.asarray([cell.bounds for cell in edges.cell])
    cell_vec = np.abs(cell_bounds[..., 1:2] - cell_bounds[..., :2])
    ids = np.argmax(cell_vec, axis=-1) == 0
    ids2 = edges.cell_centers().points[:, 1] > edges.center[1]
    top = edges.extract_cells(ids & ids2)

    vert_diff = mesh.points[:, 1, None] - top.points[:, 1]
    return np.abs(np.min(vert_diff, axis=-1))


def p_fluid(mesh: pv.UnstructuredGrid) -> PlainQuantity:
    """Return the fluid pressure in the mesh.

    If "pressure" is not given in the mesh, it is calculated by a hypothetical
    water column defined as:

    .. math::

        p_{fl} = 1000 \\frac{kg}{m^3} 9.81 \\frac{m}{s^2} h

    where `h` is the depth below surface. If "depth" is not given in the mesh,
    it is calculated via :py:func:`ogstools.propertylib.mesh_dependent.depth`.
    """
    qty = u_reg.Quantity
    if "pressure" in mesh.point_data:
        return qty(mesh["pressure"], "Pa")
    _depth = mesh["depth"] if "depth" in mesh.point_data else depth(mesh)
    return qty(1000, "kg/m^3") * qty(9.81, "m/s^2") * qty(_depth, "m")


def fluid_pressure_criterion(
    mesh: pv.UnstructuredGrid, mesh_property: Property
) -> PlainQuantity:
    """Return the fluid pressure criterion.

    Defined as the difference between fluid pressure and minimal principal
    stress (compression positive).

    .. math::

        F_{p} = p_{fl} - \\sigma_{min}

    Fluid pressure is evaluated via
    :py:func:`ogstools.propertylib.mesh_dependent.p_fluid`.
    """

    qty = u_reg.Quantity
    sigma = -qty(mesh[mesh_property.data_name], mesh_property.data_unit)
    return p_fluid(mesh) - eigenvalues(sigma)[..., 0]


def dilatancy_critescu(
    mesh: pv.UnstructuredGrid,
    mesh_property: Property,
    a: float = -0.01697,
    b: float = 0.8996,
    effective: bool = False,
) -> PlainQuantity:
    """Return the dilatancy criterion defined as:

    .. math::

        F_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma_m}{\\sigma_0}

    <https://www.sciencedirect.com/science/article/pii/S0360544222000512?via%3Dihub>
    """

    qty = u_reg.Quantity
    sigma = -qty(mesh[mesh_property.data_name], mesh_property.data_unit)
    sigma_0 = qty(1, "MPa")
    sigma_m = mean(sigma)
    if effective:
        sigma_m -= p_fluid(mesh)
    tau_oct = octahedral_shear(sigma)
    return (
        tau_oct / sigma_0 - a * (sigma_m / sigma_0) ** 2 - b * sigma_m / sigma_0
    )


dilatancy_critescu_eff = partial(dilatancy_critescu, effective=True)
"""Return the dilatancy criterion defined as:

.. math::

    F'_{dil} = \\frac{\\tau_{oct}}{\\sigma_0} - a \\left( \\frac{\\sigma'_m}{\\sigma_0} \\right)^2 - b \\frac{\\sigma'_m}{\\sigma_0}

<https://www.sciencedirect.com/science/article/pii/S0360544222000512?via%3Dihub>
"""


def dilatancy_alkan(
    mesh: pv.UnstructuredGrid,
    mesh_property: Property,
    b: float = 0.04,
    effective: bool = False,
) -> ValType:
    """Return the dilatancy criterion defined as:

    .. math::

        F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma'_m}{\\sigma_0 + b \\cdot \\sigma'_m}

    <https://www.sciencedirect.com/science/article/pii/S1365160906000979>
    """

    qty = u_reg.Quantity
    sigma = -qty(mesh[mesh_property.data_name], mesh_property.data_unit)
    tau_max = qty(33, "MPa")
    sigma_m = mean(sigma)
    if effective:
        sigma_m -= p_fluid(mesh)
    tau = octahedral_shear(sigma)
    return tau - tau_max * (b * sigma_m / (qty(1, "MPa") + b * sigma_m))


dilatancy_alkan_eff = partial(dilatancy_alkan, effective=True)
"""Return the dilatancy criterion defined as:

.. math::

    F_{dil} = \\tau_{oct} - \\tau_{max} \\cdot b \\frac{\\sigma'_m}{\\sigma_0 + b \\cdot \\sigma'_m}

<https://www.sciencedirect.com/science/article/pii/S1365160906000979>
"""
