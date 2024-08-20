import numpy as np
import pyvista as pv
from pint.facets.plain import PlainQuantity

from ogstools.variables import u_reg


def depth(mesh: pv.UnstructuredGrid, use_coords: bool = False) -> np.ndarray:
    """Returns the depth values of the mesh.

    For 2D, the last axis of the plane wherein the mesh is lying is used as
    the vertical axis (i.e. y if the mesh is in the xy-plane, z if it is in
    the xz-plane), for 3D, the z-axes is used.
    If `use_coords` is True, returns the negative coordinate value of the
    vertical axis. Otherwise, the vertical distance to the top facing edges
    surfaces are returned.
    """
    if mesh.volume > 0:
        # prevents inner edge (from holes) to be detected as a top boundary
        edges = mesh.extract_surface().connectivity("point_seed", point_ids=[0])
        vertical_dim = 2
        if use_coords:
            return -mesh.points[:, vertical_dim]
        point_upwards = edges.cell_normals[..., vertical_dim] > 0
        top_cells = point_upwards
    else:
        mean_normal = np.abs(
            np.mean(mesh.extract_surface().cell_normals, axis=0)
        )
        vertical_dim = np.delete([0, 1, 2], int(np.argmax(mean_normal)))[-1]
        if use_coords:
            return -mesh.points[:, vertical_dim]
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
        adj_centers = np.vstack(
            [
                mesh.extract_cells(adj_cell).cell_centers().points
                for adj_cell in adj_cells
            ]
        )
        are_above = [
            edge_center[vertical_dim] > adj_center[vertical_dim]
            for edge_center, adj_center in zip(
                edge_centers, adj_centers, strict=False
            )
        ]
        are_non_vertical = np.asarray(edge_horizontal_extent) > 1e-12
        top_cells = are_above & are_non_vertical
    top = edges.extract_cells(top_cells)
    eucl_distance_projected_top_points = np.sum(
        np.abs(
            np.delete(mesh.points[:, None] - top.points, vertical_dim, axis=-1)
        ),
        axis=-1,
    )
    matching_top = np.argmin(eucl_distance_projected_top_points, axis=-1)
    return np.abs(
        mesh.points[..., vertical_dim] - top.points[matching_top, vertical_dim]
    )


def p_fluid(mesh: pv.UnstructuredGrid) -> PlainQuantity:
    """Return the fluid pressure in the mesh.

    If "depth" is given in the mesh's point _data, it is used return a
    hypothetical water column defined as:

    .. math::

        p_{fl} = 1000 \\frac{kg}{m^3} 9.81 \\frac{m}{s^2} h

    where `h` is the depth below surface. Otherwise, If "pressure" is given
    in the mesh, return the "pressure" data of the mesh. If that is also not
    the case, the hypothetical water column from above is returned with the
    depth being calculated via
    :py:func:`ogstools.meshlib.geo.depth`.
    """

    Qty = u_reg.Quantity
    if "depth" in mesh.point_data:
        return (
            Qty(1000, "kg/m^3") * Qty(9.81, "m/s^2") * Qty(mesh["depth"], "m")
        )
    if "pressure" in mesh.point_data:
        return Qty(mesh["pressure"], "Pa")
    return Qty(1000, "kg/m^3") * Qty(9.81, "m/s^2") * Qty(depth(mesh), "m")
