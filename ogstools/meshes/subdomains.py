# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from functools import reduce
from itertools import chain
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv

from ogstools.mesh.utils import axis_ids_2D, remove_data


def named_boundaries(
    subdomains: list[pv.UnstructuredGrid],
) -> dict[str, pv.UnstructuredGrid]:
    """Name 1D meshes according to their position (top, bottom, left, right)

    :param subdomains: List of meshes to name
    :returns:          A dict mapping the meshes to top, bottom, left and right.
    """
    horizontal_id, vertical_id = axis_ids_2D(pv.merge(subdomains))
    centers = np.array([mesh.center for mesh in subdomains])
    return {
        "top": subdomains[np.argmax(centers[:, vertical_id])],
        "bottom": subdomains[np.argmin(centers[:, vertical_id])],
        "left": subdomains[np.argmin(centers[:, horizontal_id])],
        "right": subdomains[np.argmax(centers[:, horizontal_id])],
    }


def extract_surfaces(
    mesh: pv.UnstructuredGrid, angle: float
) -> dict[str, pv.UnstructuredGrid]:
    """Extract the 2D surfaces of a 3D mesh.

    :param mesh:    3D mesh to be split apart.
    :param angle:   Tolerated angle (in degrees) between given normal and
                    element normal.

    :returns:   A list of meshes, as the result of splitting the mesh at its
                edges.
    """
    from ogstools._find_ogs import cli
    from ogstools.mesh.file_io import read, save

    tmp_dir = Path(mkdtemp("extract_surfaces"))

    directions = {
        "bottom": ("0", "0", "1"),
        "top": ("0", "0", "-1"),
        "front": ("0", "1", "0"),
        "back": ("0", "-1", "0"),
        "left": ("1", "0", "0"),
        "right": ("-1", "0", "0"),
    }
    save(mesh, tmp_dir / "domain.vtu")

    for name, (x, y, z) in directions.items():
        cli().ExtractSurface(
            i=tmp_dir / "domain.vtu",
            o=tmp_dir / f"{name}.vtu",
            x=x,
            y=y,
            z=z,
            a=angle,
        )
    return {name: read(tmp_dir / f"{name}.vtu") for name in directions}


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
    dim = mesh.GetMaxSpatialDimension()
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
    horizontal_id, vertical_id = axis_ids_2D(mesh)
    angles = np.degrees(
        np.arctan2(vectors[:, vertical_id], vectors[:, horizontal_id])
    )
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
    mesh: pv.UnstructuredGrid,
) -> list[pv.UnstructuredGrid]:
    """Split a continuous 1D boundary by assumption of vertical lateral edges

    Only works properly if you have 2 perfectly vertical boundaries: one at the
    very left and one at the very right of the model.

    :param mesh:            1D mesh to be split apart.
    :returns:   A list of meshes, as the result of splitting the mesh at its
                corners.
    """
    dim = mesh.GetMaxSpatialDimension()
    assert dim == 1, f"Expected a mesh of dim 1, but given mesh has {dim=}"
    subdomains = []
    centers = mesh.cell_centers().points
    axis_1, axis_2 = axis_ids_2D(mesh)
    is_left = centers[:, axis_1] == mesh.bounds[axis_1 * 2]
    is_right = centers[:, axis_1] == mesh.bounds[axis_1 * 2 + 1]
    subdomains.append(mesh.extract_cells(is_left))
    subdomains.append(mesh.extract_cells(is_right))
    top_bottom = mesh.extract_cells(
        np.invert(is_left) & np.invert(is_right)
    ).connectivity(largest=False)
    for reg_id in np.unique(top_bottom.cell_data.get("RegionId", [])):
        subdomain = top_bottom.threshold([reg_id, reg_id], scalars="RegionId")
        subdomain.clear_data()
        subdomains.append(subdomain)
    return subdomains


def extract_boundaries(
    mesh: pv.UnstructuredGrid, threshold_angle: float | None = 15.0
) -> dict[str, pv.UnstructuredGrid]:
    """Extract boundaries of a 2D or 3D mesh.

    :param mesh:            The domain mesh
    :param threshold_angle: If None, the boundary will be split by the
                            assumption of vertical lateral boundaries. Otherwise
                            it represents the angle (in degrees) between
                            neighbouring elements which - if exceeded -
                            determines the corners of the boundary mesh.
    :returns:               A dictionary of top, bottom, left and right sections
                            of the boundary mesh.
    """

    dim = mesh.GetMaxSpatialDimension()
    if dim == 3:
        return extract_surfaces(
            mesh, 0.0 if threshold_angle is None else threshold_angle
        )
    if dim == 2:
        boundary = mesh.extract_feature_edges()
        if threshold_angle is None:
            subdomains = split_by_vertical_lateral_edges(boundary)
        else:
            subdomains = split_by_threshold_angle(boundary, threshold_angle)
        return named_boundaries(subdomains)
    msg = f"mesh dim has to be 2 or 3, but is {dim}."
    raise TypeError(msg)


def identify_subdomains(
    mesh: pv.UnstructuredGrid, subdomains: list[pv.UnstructuredGrid]
) -> None:
    """Add bulk_node_ids and bulk_element_ids mapping to the subdomains.

    :param mesh:        The domain mesh
    :param subdomains:  List of subdomain meshes.
    """
    from scipy.spatial import KDTree as scikdtree

    datanames = ["bulk_element_ids", "bulk_node_ids", "number_bulk_elements"]
    remove_data(mesh, datanames)
    dim = mesh.GetMaxSpatialDimension()
    for subdomain in subdomains:
        remove_data(subdomain, datanames)
        tree = scikdtree(mesh.points)
        bulk_nodes = tree.query(subdomain.points)[1]

        sub_dim = subdomain.GetMaxSpatialDimension()
        if sub_dim == dim:
            bulk_elements = mesh.find_containing_cell(
                subdomain.cell_centers().points
            )
            subdomain.cell_data["bulk_element_ids"] = np.uint64(bulk_elements)
        else:
            nodes_per_sub_cell = [cell.point_ids for cell in subdomain.cell]
            cells_per_sub_node = {
                sub_node_id: mesh.point_cell_ids(bulk_node_id)
                for sub_node_id, bulk_node_id in enumerate(bulk_nodes)
            }

            cells_per_sub_cell = []
            for nodes in nodes_per_sub_cell:
                cells_per_node = [set(cells_per_sub_node[n]) for n in nodes]
                cells_per_sub_cell.append(
                    reduce(set.intersection, cells_per_node)
                )

            ncells_per_sub_cell = [len(cells) for cells in cells_per_sub_cell]
            bulk_elements = list(chain.from_iterable(cells_per_sub_cell))
            if set(ncells_per_sub_cell) != {1}:
                # special case: subdomain cells touch multiple bulk cells
                subdomain.field_data["bulk_element_ids"] = np.uint64(
                    bulk_elements
                )
                subdomain.cell_data["number_bulk_elements"] = np.uint64(
                    ncells_per_sub_cell
                )
            else:
                subdomain.cell_data["bulk_element_ids"] = np.uint64(
                    bulk_elements
                )
        subdomain.point_data["bulk_node_ids"] = np.uint64(bulk_nodes)
