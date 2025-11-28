# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Sequence
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import pyvista as pv


def _tessellation_map(cell_type: pv.CellType, integration_order: int) -> list:
    """Return the list of point ids, which form tessellated cells.

    For now only 2D elements are covered.
    """
    if integration_order == 2:
        if cell_type == pv.CellType.TRIANGLE:
            return [[0, 3, 6, 5], [1, 4, 6, 3], [2, 5, 6, 4]]
        if cell_type == pv.CellType.QUAD:
            return [[0, 4, 8, 7], [3, 7, 8, 6], [1, 5, 8, 4], [2, 6, 8, 5]]
    if integration_order == 3:
        if cell_type == pv.CellType.TRIANGLE:
            return [[3, 4, 5], [2, 5, 4], [0, 3, 5], [1, 4, 3]]
        if cell_type in [pv.CellType.QUAD, pv.CellType.QUADRATIC_QUAD]:
            return [
                [0, 4, 12, 11],
                [7, 11, 12, 15],
                [3, 7, 15, 10],
                [4, 8, 13, 12],
                [12, 13, 14, 15],
                [6, 10, 15, 14],
                [1, 5, 13, 8],
                [5, 9, 14, 13],
                [2, 6, 14, 9],
            ]
    if integration_order == 4:
        if cell_type in [pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE]:
            return [
                [9, 10, 11],
                [5, 8, 9, 11],
                [3, 6, 10, 9],
                [4, 7, 11, 10],
                [1, 4, 10, 6],
                [2, 5, 11, 7],
                [0, 3, 9, 8],
            ]
        if cell_type in [pv.CellType.QUAD, pv.CellType.QUADRATIC_QUAD]:
            return [
                [2, 6, 18, 13],
                [9, 13, 18, 21],
                [5, 9, 21, 17],
                [1, 5, 17, 12],
                [6, 10, 22, 18],
                [18, 22, 24, 21],
                [17, 21, 24, 20],
                [8, 12, 17, 20],
                [10, 14, 19, 22],
                [19, 23, 24, 22],
                [16, 20, 24, 23],
                [4, 8, 20, 16],
                [3, 7, 19, 14],
                [7, 11, 23, 19],
                [11, 15, 16, 23],
                [0, 4, 16, 15],
            ]
    msg = f"Tessellation not implemented ({cell_type=}, {integration_order=})."
    raise TypeError(msg)


# The following functions create the points which form the cells of the
# tessellated mesh.


def _tri_quad_2(mesh: pv.DataSet) -> np.ndarray:
    corners = np.asarray([cell.points for cell in mesh.cell])
    edge_centers = 0.5 * (corners + np.roll(corners, shift=-1, axis=1))
    cell_centers = np.reshape(mesh.cell_centers().points, (-1, 1, 3))
    return np.concatenate([corners, edge_centers, cell_centers], axis=1)


def _tri_3(mesh: pv.DataSet) -> np.ndarray:
    corners = np.asarray([cell.points for cell in mesh.cell])
    edge_centers = 0.5 * (corners + np.roll(corners, shift=-1, axis=1))
    return np.concatenate([corners, edge_centers], axis=1)


def _tri_4(mesh: pv.DataSet) -> np.ndarray:
    points = np.asarray([cell.points[:3] for cell in mesh.cell])
    edge_vecs = np.roll(points, shift=-1, axis=1) - points
    edge_mids = points + edge_vecs * 1.0 / 2.0
    edge_thirds1 = points + edge_vecs * 1.0 / 3.0
    edge_thirds2 = points + edge_vecs * 2.0 / 3.0
    mid_vecs = edge_mids[:, [1, 2, 0]] - points
    mid_points = points + mid_vecs * 2.0 / 5.0
    return np.concatenate(
        [points, edge_thirds1, edge_thirds2, mid_points], axis=1
    )


def _quad_3(mesh: pv.DataSet) -> np.ndarray:
    corners = np.asarray([cell.points[:4] for cell in mesh.cell])
    edge_vecs = np.roll(corners, shift=-1, axis=1) - corners
    edge_thirds1 = corners + edge_vecs * 1.0 / 3.0
    edge_thirds2 = corners + edge_vecs * 2.0 / 3.0
    mid_vecs = np.roll(edge_thirds2, shift=-2, axis=1) - edge_thirds1
    mid_thirds = edge_thirds1 + mid_vecs * 1.0 / 3.0
    return np.concatenate(
        [corners, edge_thirds1, edge_thirds2, mid_thirds], axis=1
    )


def _quad_4(mesh: pv.DataSet) -> np.ndarray:
    corners = np.asarray([cell.points[:4] for cell in mesh.cell])
    edge_vecs = np.roll(corners, shift=-1, axis=1) - corners
    edge_mids1 = corners + edge_vecs * 1.0 / 4.0
    edge_mids2 = corners + edge_vecs * 2.0 / 4.0
    edge_mids3 = corners + edge_vecs * 3.0 / 4.0
    mids1_vecs = np.roll(edge_mids3, shift=-2, axis=1) - edge_mids1
    mid_corners = edge_mids1 + mids1_vecs * 1.0 / 4.0
    mids2_vecs = np.roll(edge_mids2, shift=-2, axis=1) - edge_mids2
    mid_mids = edge_mids2 + mids2_vecs * 1.0 / 4.0
    mid_center = (edge_mids2 + mids2_vecs * 1.0 / 2.0)[:, :1]
    return np.concatenate(
        [
            corners,
            edge_mids1,
            edge_mids2,
            edge_mids3,
            mid_corners,
            mid_mids,
            mid_center,
        ],
        axis=1,
    )


def _compute_points(
    mesh: pv.DataSet, cell_type: pv.CellType, integration_order: int
) -> np.ndarray:
    "Create the points for the cells of the tessellated mesh."
    if integration_order == 2 and cell_type in [
        pv.CellType.TRIANGLE,
        pv.CellType.QUAD,
    ]:
        return _tri_quad_2(mesh)
    if integration_order == 3:
        if cell_type == pv.CellType.TRIANGLE:
            return _tri_3(mesh)
        if cell_type == pv.CellType.QUAD:
            return _quad_3(mesh)
    if integration_order == 4:
        if cell_type in [pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE]:
            return _tri_4(mesh)
        if cell_type in [pv.CellType.QUAD, pv.CellType.QUADRATIC_QUAD]:
            return _quad_4(mesh)
    msg = f"Tessellation not implemented ({cell_type=}, {integration_order=})."
    raise TypeError(msg)


def _connectivity(
    mesh: pv.DataSet, points: np.ndarray, subcell_ids: list
) -> list:
    connectivity = []
    for index in range(mesh.number_of_cells):
        offset = index * points.shape[1]
        for ids in subcell_ids:
            connectivity += [len(ids)] + (np.asarray(ids) + offset).astype(
                int
            ).tolist()
    return connectivity


def tessellate(
    mesh: pv.DataSet, cell_type: pv.CellType, integration_order: int
) -> pv.PolyData:
    "Create a tessellated mesh with one subcell per integration point."
    subcell_ids = _tessellation_map(cell_type, integration_order)
    points = _compute_points(mesh, cell_type, integration_order)
    connectivity = _connectivity(mesh, points, subcell_ids)

    return pv.PolyData(np.reshape(points, (-1, 3)), faces=connectivity)


def ip_metadata(mesh: pv.UnstructuredGrid) -> list[dict[str, Any]]:
    "return the IntegrationPointMetaData in the mesh's field_data as a dict."

    import json

    if "IntegrationPointMetaData" not in mesh.field_data:
        msg = "Required IntegrationPointMetaData not in mesh."
        raise KeyError(msg)
    data = bytes(mesh.field_data["IntegrationPointMetaData"]).decode("utf-8")
    return json.loads(data)["integration_point_arrays"]


def to_ip_point_cloud(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    "Convert integration point data to a pyvista point cloud."
    ip_keys = [arr["name"] for arr in ip_metadata(mesh)]
    ip_len = max(len(mesh.field_data[key]) for key in ip_keys)
    _mesh = mesh.copy()
    # Filter data out, which is not on the entire mesh, i.e. material model
    # dependent data when different material models are used within one mesh.
    for key in ip_keys:
        if len(_mesh.field_data[key]) != ip_len:
            _mesh.field_data.remove(key)
    tmp_dir = Path(mkdtemp())
    input_file = tmp_dir / "ipDataToPointCloud_input.vtu"
    output_file = tmp_dir / "ip_mesh.vtu"
    _mesh.save(input_file)

    from ogstools._find_ogs import cli

    cli().ipDataToPointCloud(i=str(input_file), o=str(output_file))  # type: ignore[union-attr]
    return pv.XMLUnstructuredGridReader(output_file).read()


def to_ip_mesh(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    "Create a mesh with cells centered around integration points."
    ip_mesh = to_ip_point_cloud(mesh)
    integration_order = int(ip_metadata(mesh)[0]["integration_order"])
    new_meshes: list[pv.PolyData] = []
    cell_types = np.unique(
        getattr(mesh, "celltypes", {cell.type for cell in mesh.cell})
    )
    for cell_type in cell_types:
        _mesh = mesh.extract_cells_by_type(cell_type)
        new_meshes += [tessellate(_mesh, cell_type, integration_order)]
    new_mesh = new_meshes[0]
    for _mesh in new_meshes[1:]:
        new_mesh = new_mesh.merge(_mesh)
    new_mesh = new_mesh.clean()

    # if we add new cell_type / integration_order combination, the following
    # helps, to bring the new_mesh's cells in the correct order:

    # ordering = new_mesh.find_containing_cell(ip_mesh.points)
    # order = np.argsort(ordering)
    # if not np.all(np.diff(order) == 1):
    #     print(np.argsort(ordering))

    new_mesh.cell_data.update(ip_mesh.point_data)

    return new_mesh


def ip_data_threshold(
    mesh: pv.UnstructuredGrid,
    value: int | Sequence[int],
    scalars: str = "MaterialIDs",
    invert: bool = False,
) -> dict[str, np.ndarray]:
    """Filters integration point data to match the threshold criterion.

    Similar to ``pyvista``'s threshold filter, but only acting on the field data
    and returning the modified field data dict.

    :param mesh:    original mesh, needs to contain MaterialIDs and
                    IntegratioPointMetaData.
    :param value:   Single value or (min, max) to be used for the threshold.
                    If a sequence, then length must be 2. If single value, it
                    is used as the lower bound and selecting everything above.
    :param scalars: Name of data to threshold on.
    :param invert:  Invert the threshold results
    """
    #
    value_bounds = (value, np.inf) if isinstance(value, int) else value
    if len(value_bounds) != 2:
        msg = "If given as a Sequence, length of value must be 2."
        raise ValueError(msg)

    result = mesh.copy()
    # remove all nan data as ogs cli tools will throw errors if they exist
    for data in [result.point_data, result.cell_data, result.field_data]:
        nan_keys = [k for k, v in data.items() if np.all(np.isnan(v))]
        for key in nan_keys:
            del data[key]

    mesh_ip = to_ip_point_cloud(result)
    # in 2D there can be a floating point offset in the flat dimension resulting
    # in the sampling not finding all points, thus we have to align the ip_mesh
    # perfectly.
    if mesh.GetMaxSpatialDimension() == 2:
        pts = mesh.points
        idx = np.squeeze(np.argwhere(np.all(np.isclose(pts, pts[0]), axis=0)))
        mesh_ip.points[:, idx] = np.mean(result.points[:, idx])

    data = mesh_ip.sample(result)[scalars]
    condition = (data >= value_bounds[0]) & (data <= value_bounds[1])
    if invert:
        condition = np.invert(condition)
    cells_to_keep = np.squeeze(np.argwhere(condition))
    if len(cells_to_keep) == 0:
        msg = "Threshold resulted in empty mesh."
        raise ValueError(msg)
    for arr in ip_metadata(result):
        key = arr["name"]
        result.field_data[key] = result.field_data[key][cells_to_keep]
    return result.field_data
