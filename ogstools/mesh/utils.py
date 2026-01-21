# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv

from ogstools._find_ogs import cli

from .file_io import save


def node_reordering(
    mesh: pv.UnstructuredGrid, method: int = 1
) -> pv.UnstructuredGrid:
    """Reorders mesh nodes to make a mesh compatible with OGS6.

    :param mesh: mesh whose nodes are to be reordered.
    :param method:
        0: Reversing order of nodes for all elements.\n
        1: Reversing order of nodes unless it's perceived correct by OGS6
           standards. This is the default selection.\n
        2: Fixing node ordering issues between VTK and OGS6 (only applies
           to prism-elements).\n
        3: Re-ordering of mesh node vector such that all base nodes are
           sorted before all nonlinear nodes.
    """
    tmp_file = Path(mkdtemp(prefix="node_reordering")) / "mesh.vtu"
    save(tmp_file, mesh)
    cli().NodeReordering(i=str(tmp_file), o=str(tmp_file), m=method)
    return pv.XMLUnstructuredGridReader(tmp_file).read()


def check_datatypes(
    mesh: pv.UnstructuredGrid, strict: bool = False, name: str = ""
) -> bool:
    mat_ids = mesh.cell_data.get("MaterialIDs", np.int32(0))
    elem_ids = mesh.cell_data.get("bulk_element_ids", np.uint64(0))
    node_ids = mesh.point_data.get("bulk_node_ids", np.uint64(0))
    type_map = {
        mesh.points.dtype: ("Point coordinates", np.double),
        mat_ids.dtype: ("'MaterialIDs'", np.int32),
        elem_ids.dtype: ("'bulk_element_ids'", np.uint64),
        node_ids.dtype: ("'bulk_node_ids'", np.uint64),
    }
    for datatype, (name, ref_type) in type_map.items():
        if datatype != ref_type:
            msg = (
                f"{name} datatype needs to be {ref_type} for OGS, "
                f"but instead it is {datatype}. "
            )
            if name != "":
                msg += f"Error raised by mesh with {name=}"
            if strict:
                raise TypeError(msg)
            return False
    return True


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
