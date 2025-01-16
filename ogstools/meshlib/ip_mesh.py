from pathlib import Path
from typing import TypeVar

import numpy as np
import ogs
import pyvista as pv

Mesh = TypeVar("Mesh", bound=pv.UnstructuredGrid)


def _tessellation_map(cell_type: pv.CellType, integration_order: int) -> list:
    """Return the list of point ids, which form tessellated cells.

    For now only 2D elements are covered.
    """
    if integration_order == 2:
        if cell_type == pv.CellType.TRIANGLE:
            return [[0, 3, 6, 5], [1, 4, 6, 3], [2, 5, 6, 4]]
        if cell_type == pv.CellType.QUAD:
            return [[0, 4, 8, 7], [1, 5, 8, 4], [2, 6, 8, 5], [3, 7, 8, 6]]
    if integration_order == 3:
        if cell_type == pv.CellType.TRIANGLE:
            return [[0, 3, 5], [1, 4, 3], [2, 5, 4], [3, 4, 5]]
        if cell_type in [pv.CellType.QUAD, pv.CellType.QUADRATIC_QUAD]:
            return [
                [0, 4, 12, 11],
                [1, 5, 13, 8],
                [2, 6, 14, 9],
                [3, 7, 15, 10],
                [4, 8, 13, 12],
                [5, 9, 14, 13],
                [6, 10, 15, 14],
                [7, 11, 12, 15],
                [12, 13, 14, 15],  # fmt:skip
            ]
    if integration_order == 4:
        if cell_type in [pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE]:
            return [
                [0, 3, 9, 8],
                [1, 4, 10, 6],
                [2, 5, 11, 7],
                [3, 6, 10, 9],
                [4, 7, 11, 10],
                [5, 8, 9, 11],
                [9, 10, 11],  # fmt:skip
            ]
        if cell_type in [pv.CellType.QUAD, pv.CellType.QUADRATIC_QUAD]:
            return [
                [0, 4, 16, 15],
                [1, 5, 17, 12],
                [2, 6, 18, 13],
                [3, 7, 19, 14],
                [4, 8, 20, 16],
                [5, 9, 21, 17],
                [6, 10, 22, 18],
                [7, 11, 23, 19],
                [8, 12, 17, 20],
                [9, 13, 18, 21],
                [10, 14, 19, 22],
                [11, 15, 16, 23],
                [16, 20, 24, 23],
                [17, 21, 24, 20],
                [18, 22, 24, 21],
                [19, 23, 24, 22],  # fmt:skip
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

    return pv.PolyData(
        np.reshape(points, (-1, 3)),
        faces=connectivity,
        n_faces=len(subcell_ids) * points.shape[0],
    )


def to_ip_point_cloud(mesh: Mesh) -> pv.UnstructuredGrid:
    "Convert integration point data to a pyvista point cloud."
    # ipDataToPointCloud can't handle this
    bad_keys = [
        "material_state_variable_ElasticStrain_ip",
        "free_energy_density_ip",
    ]
    _mesh = mesh.copy()
    for key in bad_keys:
        if key in _mesh.field_data:
            _mesh.field_data.remove(key)
    parentpath = Path() if mesh.filepath is None else mesh.filepath.parent
    input_file = parentpath / "ipDataToPointCloud_input.vtu"
    _mesh.save(input_file)
    output_file = parentpath / "ip_mesh.vtu"
    ogs.cli.ipDataToPointCloud(i=str(input_file), o=str(output_file))
    return pv.XMLUnstructuredGridReader(output_file).read()


def to_ip_mesh(mesh: Mesh) -> pv.UnstructuredGrid:
    "Create a mesh with cells centered around integration points."
    meta = mesh.field_data["IntegrationPointMetaData"]
    meta_str = "".join([chr(val) for val in meta])
    integration_order = int(
        meta_str.split('"integration_order":')[1].split(",")[0]
    )
    ip_mesh = to_ip_point_cloud(mesh)

    cell_types = list({cell.type for cell in mesh.cell})
    new_meshes: list[pv.PolyData] = []
    for cell_type in cell_types:
        _mesh = mesh.extract_cells_by_type(cell_type)
        new_meshes += [tessellate(_mesh, cell_type, integration_order)]
    new_mesh = new_meshes[0]
    for _mesh in new_meshes[1:]:
        new_mesh = new_mesh.merge(_mesh)
    new_mesh = new_mesh.clean()

    ordering = new_mesh.find_containing_cell(ip_mesh.points)
    ip_data = {
        k: v[np.argsort(ordering)] for k, v in ip_mesh.point_data.items()
    }
    new_mesh.cell_data.update(ip_data)

    return new_mesh
