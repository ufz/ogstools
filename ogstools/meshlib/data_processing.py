# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from itertools import product
from typing import TypeVar

import numpy as np
import pyvista as pv
from typeguard import typechecked

from ogstools.variables import Variable

Mesh = TypeVar("Mesh", bound=pv.UnstructuredGrid)


def _raw_differences_all_data(base_mesh: Mesh, subtract_mesh: Mesh) -> Mesh:
    diff = base_mesh.copy(deep=True)
    sampled_subtract_mesh = base_mesh.sample(subtract_mesh)
    for point_data_key in base_mesh.point_data:
        diff.point_data[point_data_key] -= sampled_subtract_mesh.point_data[
            point_data_key
        ]
    for cell_data_key in base_mesh.cell_data:
        if cell_data_key == "MaterialIDs":
            continue
        diff.cell_data[cell_data_key] -= sampled_subtract_mesh.cell_data[
            cell_data_key
        ]
    return diff


def difference(
    base_mesh: Mesh,
    subtract_mesh: Mesh,
    variable: Variable | str | None = None,
) -> Mesh:
    """
    Compute the difference of variables between two meshes.

    :param base_mesh:       The mesh to subtract from.
    :param subtract_mesh:   The mesh whose data is to be subtracted.
    :param variable:        The variable of interest. If not given, all
                            point and cell_data will be processed raw.
    :returns:   A new mesh containing the difference of `variable` or
                of all datasets between both meshes.
    """
    if variable is None:
        return _raw_differences_all_data(base_mesh, subtract_mesh)
    if isinstance(variable, str):
        variable = Variable(data_name=variable, output_name=variable)

    var_key = variable.data_name if isinstance(variable, Variable) else variable
    is_same_topology = (
        base_mesh.points.shape == subtract_mesh.points.shape
        and base_mesh.cells.shape == subtract_mesh.cells.shape
        and np.all(np.equal(base_mesh.points, subtract_mesh.points))
        and np.all(np.equal(base_mesh.cells, subtract_mesh.cells))
    )
    if is_same_topology:
        sub_mesh = subtract_mesh
        mask = None
    else:
        sub_mesh = base_mesh.sample(subtract_mesh)
        mask = sub_mesh[var_key] == 0.0
    diff_mesh = base_mesh.copy(deep=True)
    diff_mesh.clear_point_data()
    diff_mesh.clear_cell_data()
    outname = variable.difference.output_name
    vals = np.asarray(
        [variable.transform(mesh) for mesh in [base_mesh, sub_mesh]]
    )
    diff_mesh[outname] = np.empty(vals.shape[1:])
    diff_mesh[outname] = vals[0] - vals[1]
    if mask is not None:
        diff_mesh[outname][mask] = np.nan
    return diff_mesh


def difference_pairwise(
    meshes_1: list | np.ndarray,
    meshes_2: list | np.ndarray,
    variable: Variable | str | None = None,
) -> np.ndarray:
    """
    Compute pairwise difference between meshes from two lists/arrays
    (they have to be of the same length).

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes whose data is subtracted
                     from the first list/array of meshes - meshes_1.
    :param variable:   The variable of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `variable`
                or all datasets between meshes_1 and meshes_2.
    """
    meshes_1 = np.asarray(meshes_1).ravel()
    meshes_2 = np.asarray(meshes_2).ravel()
    if len(meshes_1) != len(meshes_2):
        msg = "Mismatch in length of provided lists/arrays. \
              Their length has to be identical to calculate pairwise \
              difference. Did you intend to use difference_matrix()?"
        raise RuntimeError(msg)
    return np.asarray(
        [
            difference(m1, m2, variable)
            for m1, m2 in zip(meshes_1, meshes_2, strict=False)
        ]
    )


@typechecked
def difference_matrix(
    meshes_1: list | np.ndarray,
    meshes_2: list | np.ndarray | None = None,
    variable: Variable | str | None = None,
) -> np.ndarray:
    """
    Compute the difference between all combinations of two meshes
    from one or two arrays based on a specified variable.

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes, it is subtracted from
                     the first list/array of meshes - meshes_1 (optional).
    :param variable:   The variable of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `variable`
                or all datasets between meshes_1 and meshes_2 for all possible
                combinations.
    """
    meshes_1 = np.asarray(meshes_1).ravel()
    if meshes_2 is None:
        meshes_2 = meshes_1.copy()
    meshes_2 = np.asarray(meshes_2).ravel()
    diff_meshes = [
        difference(m1, m2, variable) for m1, m2 in product(meshes_1, meshes_2)
    ]
    return np.asarray(diff_meshes).reshape((len(meshes_1), len(meshes_2)))
