# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import numpy as np
import pyvista as pv


def reindex_material_ids(mesh: pv.UnstructuredGrid) -> None:
    unique_mat_ids = np.unique(mesh["MaterialIDs"])
    id_map = dict(
        zip(*np.unique(unique_mat_ids, return_inverse=True), strict=True)
    )
    mesh["MaterialIDs"] = np.int32(list(map(id_map.get, mesh["MaterialIDs"])))
    return


def remove_data(mesh: pv.UnstructuredGrid, datanames: list[str]) -> None:
    for dataname in datanames:
        mesh.point_data.pop(dataname, None)
        mesh.cell_data.pop(dataname, None)
        mesh.field_data.pop(dataname, None)


def axis_ids_2D(mesh: pv.DataSet) -> tuple[int, int]:
    "Return the two axes, in which the mesh (predominantly) lives in."
    from ogstools.plot.utils import get_projection

    tri = pv.Triangle(
        [mesh.points[0], mesh.points[mesh.n_points // 2], mesh.points[-1]]
    )
    axis_1, axis_2, _, _ = get_projection(tri)
    len1, len2 = (len(np.unique(mesh.points[:, ax])) for ax in [axis_1, axis_2])
    if len1 == len2:
        if axis_2 > axis_1:
            return axis_1, axis_2
        return axis_2, axis_1
    if len1 <= len2:
        return axis_1, axis_2
    return axis_2, axis_1


def reshape_obs_points(
    points: np.ndarray | list, mesh: pv.UnstructuredGrid | None = None
) -> np.ndarray:
    points = np.asarray(points)

    pts = points.reshape((-1, points.shape[-1]))

    # Add missing columns to comply with pyvista expectations
    if pts.shape[1] == 3:
        pts_pyvista = pts
    elif mesh is None:
        pts_pyvista = np.hstack(
            (pts, np.zeros((pts.shape[0], 3 - pts.shape[1])))
        )
    else:
        # Detect and handle flat dimensions
        geom = mesh.points
        flat_axis = np.argwhere(np.all(np.isclose(geom, geom[0]), axis=0))
        flat_axis = flat_axis.flatten()
        if pts.shape[1] + len(flat_axis) < 3:
            err_msg = (
                "Number of flat axis and number of coordinates"
                " in provided points doesn't add up to 3."
                " Please ensure that the provided points match"
                " the plane of the mesh."
            )
            raise RuntimeError(err_msg)
        pts_pyvista = np.empty((pts.shape[0], 3))
        pts_id = 0
        for col_id in range(3):
            if col_id in flat_axis:
                pts_pyvista[:, col_id] = (
                    np.ones((pts.shape[0],)) * geom[0, col_id]
                )
            else:
                pts_pyvista[:, col_id] = pts[:, pts_id]
                pts_id = pts_id + 1
    return pts_pyvista
