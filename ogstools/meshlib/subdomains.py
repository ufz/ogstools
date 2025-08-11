# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree as scikdtree


def named_boundaries(
    subdomains: list[pv.UnstructuredGrid], x_id: int = 0, y_id: int = 1
) -> dict[str, pv.UnstructuredGrid]:
    """Name 1D meshes according to their position (top, bottom, left, right)

    :param subdomains: List of meshes to name
    :param x_id:       id of the horizontal axis (0: x, 1: y, 2: z).
    :param y_id:       id of the vertical axis   (0: x, 1: y, 2: z).
    :returns:          A dict mapping the meshes to top, bottom, left and right.
    """
    centers = np.array([mesh.center for mesh in subdomains])
    return {
        "top": subdomains[np.argmax(centers[:, y_id])],
        "bottom": subdomains[np.argmin(centers[:, y_id])],
        "left": subdomains[np.argmin(centers[:, x_id])],
        "right": subdomains[np.argmax(centers[:, x_id])],
    }


def split_by_threshold_angle(
    mesh: pv.UnstructuredGrid, threshold_angle: float
) -> list[pv.UnstructuredGrid]:
    """Split a continuous 1D boundary by a threshold angle

    :param mesh:            1D mesh to be split apart.
    :param threshold_angle: Represents the angle (in degrees) between
                            neighbouring elements which - if exceeded -
                            determines the corners of the mesh.
    :returns:   A list of meshes, as the result of splitting the mesh at its
                corners.
    """
    dim = 3 if mesh.volume else 2 if mesh.area else 1 if mesh.length else 0
    assert dim == 1, f"Expected a mesh of dim 1, but given mesh has {dim=}"
    cell_pts = np.asarray([cell.points for cell in mesh.cell])
    ordered_cell_ids = [0]
    current_id = 0
    for _ in range(mesh.number_of_cells):
        a_equals_b = np.equal(cell_pts[current_id, 1], cell_pts[:, 0])
        next_cell_id = np.argmax(np.all(a_equals_b, axis=1))
        ordered_cell_ids += [next_cell_id]
        current_id = next_cell_id

    vectors = np.diff(cell_pts[ordered_cell_ids], axis=1)[:, 0]
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles_diff_pos = np.abs(np.diff(angles))
    angles_diff = [360.0 - ang if ang > 180 else ang for ang in angles_diff_pos]
    corners = np.where(np.abs(angles_diff) > threshold_angle)[0]
    corners = np.append(corners, corners[0] + mesh.number_of_cells)

    subdomains = []
    for i in range(len(corners) - 1):
        cell_ids = np.array(
            [
                ordered_cell_ids[(k + 1) % mesh.number_of_cells]
                for k in range(corners[i], corners[i + 1])
            ],
            dtype=int,
        )
        subdomain = mesh.extract_cells(cell_ids)
        subdomain.clear_data()
        subdomains.append(subdomain)
    return subdomains


def split_by_vertical_lateral_edges(
    mesh: pv.UnstructuredGrid, x_id: int = 0
) -> list[pv.UnstructuredGrid]:
    """Split a continuous 1D boundary by assumption of vertical lateral edges

    Only works properly if you have 2 perfectly vertical boundaries: one at the
    very left and one at the very right of the model.

    :param mesh:            1D mesh to be split apart.
    :param x_id:            id of the horizontal axis (0: x, 1: y, 2: z).
    :returns:   A list of meshes, as the result of splitting the mesh at its
                corners.
    """
    dim = 3 if mesh.volume else 2 if mesh.area else 1 if mesh.length else 0
    assert dim == 1, f"Expected a mesh of dim 1, but given mesh has {dim=}"
    subdomains = []
    centers = mesh.cell_centers().points
    is_left = centers[:, x_id] == mesh.bounds[x_id * 2]
    is_right = centers[:, x_id] == mesh.bounds[x_id * 2 + 1]
    subdomains.append(mesh.extract_cells(is_left))
    subdomains.append(mesh.extract_cells(is_right))
    top_bottom = mesh.extract_cells(
        np.invert(is_left) & np.invert(is_right)
    ).connectivity(largest=False)
    for reg_id in np.unique(top_bottom.cell_data.get("RegionId", [])):
        subdomain = top_bottom.threshold([reg_id, reg_id], "RegionId")
        subdomain.clear_data()
        subdomains.append(subdomain)
    return subdomains


def extract_boundaries(
    mesh: pv.UnstructuredGrid, threshold_angle: float | None = 15.0
) -> dict[str, pv.UnstructuredGrid]:
    """Extract 1D boundaries of a 2D mesh.

    :param mesh:            The 2D domain
    :param threshold_angle: If None, the boundary will be split by the
                            assumption of vertical lateral boundaries. Otherwise
                            it represents the angle (in degrees) between
                            neighbouring elements which - if exceeded -
                            determines the corners of the boundary mesh.
    :returns:               A dictionary of top, bottom, left and right sections
                            of the boundary mesh.
    """

    dim = 3 if mesh.volume else 2 if mesh.area else 1 if mesh.length else 0
    assert dim == 2, f"Expected a mesh of dim 2, but given mesh has {dim=}"
    boundary = mesh.extract_feature_edges()
    flat_axis = np.argwhere(
        np.all(np.isclose(mesh.points, mesh.points[0]), axis=0)
    ).flatten()[0]
    x_id, y_id = np.delete([0, 1, 2], flat_axis)
    if threshold_angle is None:
        subdomains = split_by_vertical_lateral_edges(boundary, x_id)
    else:
        subdomains = split_by_threshold_angle(boundary, threshold_angle)
    identify_subdomains(mesh, subdomains)
    return named_boundaries(subdomains, x_id, y_id)


def identify_subdomains(
    mesh: pv.UnstructuredGrid, subdomains: list[pv.UnstructuredGrid]
) -> None:
    """Add bulk_node_ids and bulk_element_ids mapping to the subdomains.

    :param mesh:        The domain mesh
    :param subdomains:  List of subdomain meshes.
    """
    for subdomain in subdomains:
        bulk_cell_ids = mesh.find_containing_cell(
            subdomain.cell_centers().points
        )
        tree = scikdtree(mesh.points)
        bulk_node_ids = tree.query(subdomain.points)[1]
        subdomain.cell_data["bulk_elem_ids"] = np.uint64(bulk_cell_ids)
        subdomain.point_data["bulk_node_ids"] = np.uint64(bulk_node_ids)
