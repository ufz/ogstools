from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import pyvista as pv


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


def analytical_solution(topology: Union[Path, pv.DataSet]) -> pv.DataSet:
    mesh = topology if isinstance(topology, pv.DataSet) else pv.read(topology)
    new_mesh = deepcopy(mesh)
    new_mesh.clear_point_data()
    points = new_mesh.points
    new_mesh.point_data["pressure"] = _h(points)
    return new_mesh
