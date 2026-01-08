# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Literal

import numpy as np
import pyvista as pv


def depth(
    mesh: pv.UnstructuredGrid,
    top_mesh: pv.UnstructuredGrid,
    vertical_axis: int | Literal["x", "y", "z"] | None = None,
) -> np.ndarray:
    """Returns the depth values of the mesh.

    Computes the distance between each point of `mesh` and `top_mesh` along the
    `vertical_axis`. Uses linear interpolation to interpolate in between points
    of `top_mesh`.

    :param mesh:            Mesh at the points of which the depth is computed.
    :param top_mesh:        The mesh which defines the vertical boundary.
    :param vertical_axis:   If not given: For 3D, the z-axes is used.
        For 2D, the last axis of the plane wherein the mesh is lying is used
        (i.e. y if the mesh is in the xy-plane; z if it is in the xz-plane).
    """
    from scipy import interpolate

    if vertical_axis is None:
        vertical_axis = 2 if mesh.volume > 0 else 1

    top_dim = top_mesh.GetMaxSpatialDimension()
    values = top_mesh.points[:, vertical_axis]
    if top_dim == 2:
        geom = np.delete(top_mesh.points, vertical_axis, axis=-1)
        pts = np.delete(mesh.points, vertical_axis, axis=-1)
        v_max = interpolate.LinearNDInterpolator(geom, values, np.nan)(pts)
    elif top_dim == 1:
        axis = np.argmax(np.abs(np.diff(np.reshape(top_mesh.bounds, (3, 2)))))
        v_max = interpolate.interp1d(
            top_mesh.points[:, axis], values.T, kind="linear"
        )(mesh.points[:, axis])
    else:
        msg = f"Not implemented for top mesh with dim {top_dim}."
        raise TypeError(msg)

    return np.abs(v_max - mesh.points[..., vertical_axis])
