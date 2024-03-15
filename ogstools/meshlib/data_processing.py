from typing import Union

import pyvista as pv

from ogstools.propertylib import Property, presets


def difference(
    mesh1: pv.UnstructuredGrid,
    mesh2: pv.UnstructuredGrid,
    mesh_property: Union[Property, str],
) -> pv.UnstructuredGrid:
    """
    Compute the difference between two meshes based on a specified property.

    :param mesh1: The first mesh to be subtracted from.
    :param mesh2: The second mesh whose data is subtracted from the first mesh.
    :param mesh_property: The property to of interest.
    :returns: A new mesh representing the difference between mesh1 and mesh2.
    """
    if isinstance(mesh_property, str):
        data_shape = mesh1[mesh_property].shape
        mesh_property = presets.get_preset(mesh_property, data_shape)
    diff_mesh = mesh1.copy(deep=True)
    diff_mesh[mesh_property.data_name] -= mesh2[mesh_property.data_name]
    return diff_mesh
