# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from itertools import product
from typing import Optional, Union

import numpy as np
import pyvista as pv
from typeguard import typechecked

from ogstools.propertylib import Property


def _raw_differences_all_data(
    mesh1: pv.UnstructuredGrid, mesh2: pv.UnstructuredGrid
) -> pv.UnstructuredGrid:
    diff_mesh = mesh1.copy(deep=True)
    for point_data_key in mesh1.point_data:
        diff_mesh.point_data[point_data_key] -= mesh2.point_data[point_data_key]
    for cell_data_key in mesh1.cell_data:
        if cell_data_key == "MaterialIDs":
            continue
        diff_mesh.cell_data[cell_data_key] -= mesh2.cell_data[cell_data_key]
    return diff_mesh


def difference(
    mesh1: pv.UnstructuredGrid,
    mesh2: pv.UnstructuredGrid,
    mesh_property: Optional[Union[Property, str]] = None,
) -> pv.UnstructuredGrid:
    """
    Compute the difference of properties between two meshes.

    :param mesh1: The first mesh to be subtracted from.
    :param mesh2: The second mesh whose data is subtracted from the first mesh.
    :param mesh_property:   The property of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   A new mesh containing the difference of `mesh_property` or all
                datasets between mesh1 and mesh2.
    """
    if mesh_property is None:
        return _raw_differences_all_data(mesh1, mesh2)
    if isinstance(mesh_property, Property):
        vals = np.asarray(
            [mesh_property.transform(mesh) for mesh in [mesh1, mesh2]]
        )
        outname = mesh_property.output_name + "_difference"
    else:
        vals = np.asarray([mesh[mesh_property] for mesh in [mesh1, mesh2]])
        outname = mesh_property + "_difference"

    diff_mesh = mesh1.copy(deep=True)
    diff_mesh.clear_data()
    diff_mesh[outname] = np.empty(vals.shape[1:])
    diff_mesh[outname] = vals[0] - vals[1]
    return diff_mesh


def difference_pairwise(
    meshes_1: Union[list, np.ndarray],
    meshes_2: Union[list, np.ndarray],
    mesh_property: Optional[Union[Property, str]] = None,
) -> np.ndarray:
    """
    Compute pairwise difference between meshes from two lists/arrays
    (they have to be of the same length).

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes whose data is subtracted
                     from the first list/array of meshes - meshes_1.
    :param mesh_property:   The property of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `mesh_property`
                or all datasets between meshes_1 and meshes_2.
    """
    meshes_1 = np.asarray(meshes_1).flatten()
    meshes_2 = np.asarray(meshes_2).flatten()
    if len(meshes_1) != len(meshes_2):
        msg = "Mismatch in length of provided lists/arrays. \
              Their length has to be identical to calculate pairwise \
              difference. Did you intend to use difference_matrix()?"
        raise RuntimeError(msg)
    return np.asarray(
        [
            difference(m1, m2, mesh_property)
            for m1, m2 in zip(meshes_1, meshes_2)
        ]
    )


@typechecked
def difference_matrix(
    meshes_1: Union[list, np.ndarray],
    meshes_2: Optional[Union[list, np.ndarray]] = None,
    mesh_property: Optional[Union[Property, str]] = None,
) -> np.ndarray:
    """
    Compute the difference between all combinations of two meshes
    from one or two arrays based on a specified property.

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes, it is subtracted from
                     the first list/array of meshes - meshes_1 (optional).
    :param mesh_property:   The property of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `mesh_property`
                or all datasets between meshes_1 and meshes_2 for all possible
                combinations.
    """
    meshes_1 = np.asarray(meshes_1).flatten()
    if meshes_2 is None:
        meshes_2 = meshes_1.copy()
    meshes_2 = np.asarray(meshes_2).flatten()
    diff_meshes = [
        difference(m1, m2, mesh_property)
        for m1, m2 in product(meshes_1, meshes_2)
    ]
    return np.asarray(diff_meshes).reshape((len(meshes_1), len(meshes_2)))
