from itertools import product
from typing import Optional, Union

import numpy as np
import pyvista as pv

from ogstools.propertylib import Property, presets


def difference(
    mesh_property: Union[Property, str],
    mesh1: pv.UnstructuredGrid,
    mesh2: pv.UnstructuredGrid,
) -> pv.UnstructuredGrid:
    """
    Compute the difference between two meshes based on a specified property.

    :param mesh_property: The property to of interest.
    :param mesh1: The first mesh to be subtracted from.
    :param mesh2: The second mesh whose data is subtracted from the first mesh.
    :returns: A new mesh representing the difference between mesh1 and mesh2.
    """
    mesh_property = presets.get_preset(mesh_property, mesh1)
    diff_mesh = mesh1.copy(deep=True)
    diff_mesh[mesh_property.data_name] -= mesh2[mesh_property.data_name]
    return diff_mesh


def difference_pairwise(
    mesh_property: Union[Property, str],
    meshes_1: Union[list, np.ndarray],
    meshes_2: Union[list, np.ndarray],
) -> np.ndarray:
    """
    Compute pairwise difference between meshes from two lists/arrays
    (they have to be of the same length).

    :param mesh_property: The property to of interest.
    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes whose data is subtracted
                     from the first list/array of meshes - meshes_1.
    :returns: An array of differences between meshes from meshes_1 and meshes_2.
    """
    meshes_1 = np.asarray(meshes_1).flatten()
    meshes_2 = np.asarray(meshes_2).flatten()
    if len(meshes_1) != len(meshes_2):
        msg = "Mismatch in length of provided lists/arrays. \
              Their length has to be identical to calculate pairwise \
              difference. Did you intend to use difference_matrix()?"
        raise RuntimeError(msg)
    diff_mesh = [
        difference(mesh_property, m1, m2) for m1, m2 in zip(meshes_1, meshes_2)
    ]
    return np.array(diff_mesh)


def difference_matrix(
    mesh_property: Union[Property, str],
    meshes_1: Union[list, np.ndarray],
    meshes_2: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Compute the difference between all combinations of two meshes
    from one or two arrays based on a specified property.

    :param mesh_property: The property to of interest.
    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes, it is subtracted from
                     the first list/array of meshes - meshes_1 (optional).
    :returns: An array of differences between meshes from meshes_1 and meshes_2.
    """
    if not isinstance(meshes_1, (list, np.ndarray)):
        msg = "mesh1 is neither of type list nor np.ndarray"  # type: ignore[unreachable]
        raise TypeError(msg)
    meshes_1 = np.asarray(meshes_1).flatten()
    if not isinstance(meshes_2, (list, np.ndarray)) and meshes_2 is not None:
        msg = "mesh2 is neither of type list nor np.ndarray."  # type: ignore[unreachable]
        raise TypeError(msg)
    if meshes_2 is None:
        meshes_2 = meshes_1.copy()
    meshes_2 = np.asarray(meshes_2).flatten()
    diff_mesh = [
        difference(mesh_property, m1, m2)
        for m1, m2 in product(meshes_1, meshes_2)
    ]
    return np.array(diff_mesh).reshape((len(meshes_1), len(meshes_2)))
