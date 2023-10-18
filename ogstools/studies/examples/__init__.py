import importlib.resources as pkg_resources
from copy import deepcopy

import numpy as np
import pyvista as pv

meshes = [
    pv.read(str(pkg_resources.files(__name__) / f"square_1e0_neumann_{i}.vtu"))
    for i in range(6)
]


def _c_k(k):
    return 0.5 * (2 * k - 1) * np.pi


def _a_k(k):
    return 2 / (_c_k(k) ** 2 * np.cosh(_c_k(k)))


def _h(points):
    result = np.ones(len(points))
    for k in np.arange(1, 100):
        c_k_val = _c_k(k)
        sin_c_k = np.sin(c_k_val * points[:, 1])
        sinh_c_k = np.sinh(c_k_val * points[:, 0])
        result += _a_k(k) * sin_c_k * sinh_c_k
    return result


def analytical_solution(target_mesh: pv.DataSet) -> pv.DataSet:
    new_mesh = deepcopy(target_mesh)
    new_mesh.point_data.clear()
    points = new_mesh.points
    new_mesh.point_data["pressure"] = _h(points)
    return new_mesh
