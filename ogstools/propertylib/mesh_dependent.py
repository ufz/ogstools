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


def depth(mesh: pv.UnstructuredGrid, use_coords: bool = False) -> np.ndarray:
    """Return the depth values of the mesh.

    For 2D, the last non-flat dimension is used as the vertical axis, for 3D,
    the z-axes is used.
    If `use_coords` is True, returns the negative coordinate value of the
    vertical axis. Otherwise, the vertical distance to the top facing edges /
    surfaces are returned.
    """
    if mesh.volume > 0:
        # prevents inner edge (from holes) to be detected as a top boundary
        edges = mesh.extract_surface().connectivity("point_seed", point_ids=[0])
        top_id = 2
        if use_coords:
            return -mesh.points[:, top_id]
        point_upwards = edges.cell_normals[..., top_id] > 0
        top_cells = point_upwards
    else:
        mean_normal = np.abs(
            np.mean(mesh.extract_surface().cell_normals, axis=0)
        )
        top_id = np.delete([0, 1, 2], int(np.argmax(mean_normal)))[-1]
        if use_coords:
            return -mesh.points[:, top_id]
        # prevents inner edge (from holes) to be detected as a top boundary
        edges = mesh.extract_feature_edges().connectivity(
            "point_seed", point_ids=[0]
        )
        edge_horizontal_extent = [
            np.diff(np.reshape(cell.bounds, (-1, 2))).ravel()[0]
            for cell in edges.cell
        ]
        edge_centers = edges.cell_centers().points
        adj_cells = [mesh.find_closest_cell(point) for point in edge_centers]
        adj_cell_centers = mesh.extract_cells(adj_cells).cell_centers().points
        are_above = edge_centers[..., top_id] > adj_cell_centers[..., top_id]
        are_non_vertical = np.asarray(edge_horizontal_extent) > 1e-12
        top_cells = are_above & are_non_vertical
    top = edges.extract_cells(top_cells)
    eucl_distance_projected_top_points = np.sum(
        np.abs(np.delete(mesh.points[:, None] - top.points, top_id, axis=-1)),
        axis=-1,
    )
    matching_top = np.argmin(eucl_distance_projected_top_points, axis=-1)
    return np.abs(mesh.points[..., top_id] - top.points[matching_top, top_id])


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
