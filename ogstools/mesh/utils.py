# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from ogstools._find_ogs import cli
from ogstools.definitions import temp_file

from .file_io import save


def node_reordering(
    mesh: pv.UnstructuredGrid, method: int = 1, log: bool = True
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
    :param log: If False, silence the NodeReordering tool's own log output.
    """
    tmp_file = temp_file(".vtu", "node_reordering")
    save(mesh, tmp_file)
    cli().NodeReordering(
        i=str(tmp_file), o=str(tmp_file), m=method, l="info" if log else "none"
    )
    return pv.XMLUnstructuredGridReader(tmp_file).read()


def to_linear(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    "Convert to a linear mesh."
    tmp_file = temp_file(".vtu", "mesh", "to_linear")
    save(mesh, tmp_file)
    cli().convertToLinearMesh(i=str(tmp_file), o=str(tmp_file))
    return pv.XMLUnstructuredGridReader(tmp_file).read()


def to_quadratic(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    "Convert to a quadratic mesh."
    tmp_file = temp_file(".vtu", "mesh", "to_quadratic")
    save(mesh, tmp_file)
    cli().createQuadraticMesh(i=str(tmp_file), o=str(tmp_file))
    return pv.XMLUnstructuredGridReader(tmp_file).read()


def validate(
    mesh: pv.UnstructuredGrid | Path | str, strict: bool = False
) -> bool:
    """Check conformity of mesh with OGS.

    :param mesh:    pyvista mesh or path to the mesh file.
    :param strict:  If True, raise a UserWarning if checkMesh returns an error.
    """

    if isinstance(mesh, pv.DataSet):
        mesh_file = str(temp_file(".vtu", "validate"))
        save(mesh, mesh_file)
    else:
        mesh_file = str(mesh)

    # ToDo Either checkMesh must return status of mesh (not of itself) OR
    #      cli() can handle stdout
    if shutil.which("checkMesh") is None:
        return True
    ret = subprocess.run(
        ["checkMesh", mesh_file, "-v"], stdout=subprocess.PIPE, check=False
    )
    msg = ret.stdout.decode("utf-8")
    is_valid = "No errors found." in msg
    if not is_valid:
        print(msg)
    if strict and not is_valid:
        msg = "Provided mesh is not compliant with OGS."
        raise UserWarning(msg)
    return is_valid


def check_datatypes(
    mesh: pv.UnstructuredGrid, strict: bool = False, meshname: str = ""
) -> bool:
    mat_ids = mesh.cell_data.get("MaterialIDs", np.int32(0))
    elem_ids = mesh.cell_data.get("bulk_element_ids", np.uint64(0))
    node_ids = mesh.point_data.get("bulk_node_ids", np.uint64(0))
    type_map = {
        # Point coordinates is chosen as alternative/easier to read name for mesh.points
        "Point coordinates": (
            mesh.points.dtype,
            {np.dtype("float32"), np.dtype("float64")},
        ),
        "'MaterialIDs'": (
            mat_ids.dtype,
            {np.dtype("int32"), np.dtype("uint32")},
        ),
        "'bulk_element_ids'": (elem_ids.dtype, {np.dtype("uint64")}),
        "'bulk_node_ids'": (node_ids.dtype, {np.dtype("uint64")}),
    }
    for name, (datatype, ref_type) in type_map.items():
        if datatype not in ref_type:
            msg = (
                f"{name} datatype needs to be {ref_type} for OGS, "
                f"but instead it is {datatype}. "
            )
            if meshname != "":
                msg += f"Error raised by mesh with {meshname=}"
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


def ordered_cell_ids(edges: pv.PolyData) -> list[int]:
    n_cells = edges.n_cells
    # shape=(n_cells, 2, 3), the 2 is for pointA and pointB
    cell_pts = np.asarray([cell.points for cell in edges.cell])

    ordered_cell_ids = [0]
    cell_id = 0
    compare_idx = 1

    def next_unused(length: int, used: list[int]) -> int:
        return next(idx for idx in range(length) if idx not in sorted(used))

    for _ in range(n_cells - 1):
        matching = np.equal(
            cell_pts[cell_id, compare_idx], cell_pts[:, 1 - compare_idx]
        ).all(axis=1)
        if not any(matching):
            ordered_cell_ids = ordered_cell_ids[::-1]
            compare_idx = 1 - compare_idx
            matching = np.equal(
                cell_pts[ordered_cell_ids[-1], compare_idx],
                cell_pts[:, 1 - compare_idx],
            ).all(axis=1)
            if not any(matching):
                next_id = next_unused(n_cells, ordered_cell_ids)
            else:
                next_id = np.argmax(matching)
        else:
            next_id = np.argmax(matching)

        if next_id in ordered_cell_ids:
            next_id = next_unused(n_cells, ordered_cell_ids)
        ordered_cell_ids += [int(next_id)]
        cell_id = int(next_id)
    return ordered_cell_ids


def unique_cell_types(mesh: pv.DataSet) -> list[pv.CellType]:
    "Returns the unique cell types of the mesh"
    if hasattr(mesh, "celltypes"):
        return np.unique(mesh.celltypes).tolist()
    return list({cell.type for cell in mesh.cell})


def pv_set_attr(mesh: pv.DataSet, attr: str, value: Any) -> None:
    """
    Set a PyVista mesh attribute.

    Updates the attribute if it already exists; otherwise creates it
    using :func:`pyvista.set_new_attribute`.
    """
    if hasattr(mesh, attr):
        setattr(mesh, attr, value)
    else:
        pv.set_new_attribute(mesh, attr, value)


def angles(
    dataset: pv.DataSet | Sequence[pv.DataSet],
    center: Sequence = (0.0, 0.0, 0.0),
    normal: Sequence = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Compute the angles of the mesh's points around a normal and center.

    :param mesh:    For the points of this mesh the angles are computed.
    :param center:  Center of rotation.
    :param normal:  Normal axis of rotation.
    """
    mesh = dataset if isinstance(dataset, pv.DataSet) else dataset[0]

    n = np.asarray(normal, dtype=float)
    assert n.shape == (3,), "normal must be length-3"
    assert not np.allclose(n, [0, 0, 0]), "normal must have a length"
    n_unit = n / np.linalg.norm(n)

    vecs = mesh.points - np.asarray(center, dtype=float)
    # project each vector into the plane
    v_dot_n = np.dot(vecs, n_unit)  # (N,)
    v_proj = vecs - np.outer(v_dot_n, n_unit)  # (N,3)

    trial_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial_axis, n_unit)) > 0.9:
        trial_axis = np.array([0.0, 1.0, 0.0])

    u_axis = trial_axis - np.dot(trial_axis, n_unit) * n_unit
    u_unit = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(n_unit, u_unit)
    v_unit = v_axis / np.linalg.norm(v_axis)

    # coordinates in plane
    x = np.dot(v_proj, u_unit)
    y = np.dot(v_proj, v_unit)

    result = np.arctan2(y, x)  # in radians, range (-pi, pi]
    return np.where(np.hypot(x, y) > 1e-12, result, 0.0)


def azimuth(dataset: pv.DataSet | pv.DataSet) -> np.ndarray | None:
    "Calculate the azimuth angle with regards to the z-axis"
    mesh = dataset if isinstance(dataset, pv.DataSet) else dataset[0]

    if mesh.GetMaxSpatialDimension() == 2:
        return None
    pts, z = (mesh.points, mesh.points[:, 2])
    r = np.hypot(*pts[:, [0, 1]].T)
    return np.arctan(
        np.divide(r, z, out=np.ones_like(z) * 1e12, where=z != 0.0)
    )
