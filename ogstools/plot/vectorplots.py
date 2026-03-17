# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.variables import Vector

from .shared import setup


def _vectorfield(
    mesh: pv.DataSet,
    variable: Vector,
    projection: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot the vector streamlines or arrows on a matplotlib axis.

    :param mesh:        Mesh containing the vector variable
    :param variable:    Vector variable to visualize
    :param projection:  Index of flat dimension (e.g. 2 for z axis),
                        gets automatically determined if not given
    """
    if (n_pts := setup.num_streamline_interp_pts) is None:
        return (np.zeros(1),) * 5
    mean_normal = np.abs(
        np.mean(
            mesh.extract_surface(algorithm="dataset_surface").cell_normals,
            axis=0,
        )
    )
    if projection is None:
        projection = int(np.argmax(mean_normal))
    i_id, j_id = np.delete([0, 1, 2], projection)
    _mesh = mesh.copy()
    _mesh.points[:, projection] = 0.0
    for key in _mesh.point_data:
        if key not in [variable.data_name, variable.mask]:
            del _mesh.point_data[key]

    i_pts = np.linspace(mesh.bounds[2 * i_id], mesh.bounds[2 * i_id + 1], n_pts)
    j_pts = np.linspace(mesh.bounds[2 * j_id], mesh.bounds[2 * j_id + 1], n_pts)
    i_grid, j_grid = np.meshgrid(i_pts, j_pts, indexing="ij")
    grid_input = [i_grid, j_grid]
    grid_input.insert(projection, i_grid * 0)
    grid = pv.StructuredGrid(*grid_input).sample(_mesh, pass_cell_data=False)
    values = variable.transform(grid.point_data[variable.data_name])

    values[np.argwhere(grid["vtkValidPointMask"] == 0), :] = np.nan
    if np.shape(values)[-1] == 3:
        values = np.delete(values, projection, 1)
    val = np.reshape(values, (n_pts, n_pts, 2))

    if variable.mask in grid.point_data:
        mask = np.reshape(grid.point_data[variable.mask], (n_pts, n_pts))
        val[mask == 0, :] = np.nan
    val_norm = np.linalg.norm(np.nan_to_num(val), axis=-1)
    i_grid, j_grid = np.meshgrid(i_pts, j_pts)

    return (i_grid, j_grid, val[..., 0], val[..., 1], val_norm)


def streamlines(
    mesh: pv.DataSet,
    ax: plt.Axes,
    variable: Vector,
    projection: int | None = None,
    arrowsize: float | None = None,
    density: float = 1.5,
    streamlinewidth: float = 2.5,
) -> None:
    """
    Plot the vector streamlines on a matplotlib axis.

    :param mesh:        Mesh containing the vector variable
    :param ax:          Matplotlib axis to plot onto
    :param variable:    Vector variable to visualize
    :param projection:  Index of flat dimension (e.g. 2 for z axis),
                        gets automatically determined if not given
    :param arrowsize:   Sets size of arrows in the plot.
    :param density:     density of streamlines
    :param streamlinewidth:   base width of the streamlines
    """
    if (setup.num_streamline_interp_pts) is None:
        return
    if arrowsize is None:
        arrowsize = setup.arrowsize
    x_g, y_g, u, v, val_norm = _vectorfield(mesh, variable, projection)
    val_norm_norm = val_norm / max(1e-16, np.max(val_norm))
    lw = streamlinewidth * setup.linewidth * val_norm_norm
    ax.streamplot(
        x_g,
        y_g,
        u,
        v,
        color="k",
        linewidth=lw,
        density=density,
        arrowsize=arrowsize,
    )


def quiver(
    mesh: pv.DataSet,
    ax: plt.Axes,
    variable: Vector,
    projection: int | None = None,
    glyph_type: Literal["arrow", "line"] = "arrow",
    arrowsize: float | None = None,
    scale: float = 0.03,
) -> None:
    """
    Plot arrows or lines corresponding to vectors on a matplotlib axis.

    :param mesh:        Mesh containing the vector variable
    :param ax:          Matplotlib axis to plot onto
    :param variable:    Vector variable to visualize
    :param projection:  Index of flat dimension (e.g. 2 for z axis),
                        gets automatically determined if not given
    :param glyph_type:  Whether to plot arrows or lines.
    :param arrowsize:   Sets size of arrows in the plot.
    :param scale:       scaling factor for lines / arrows
    """

    if arrowsize is None:
        arrowsize = setup.arrowsize
    x_g, y_g, u, v, val_norm = _vectorfield(mesh, variable, projection)
    line_args = (
        {"headlength": 0, "headaxislength": 0, "headwidth": 1, "pivot": "mid"}
        if glyph_type == "line"
        else {}
    )
    ax.quiver(x_g, y_g, u, v, **line_args, scale=val_norm.max() / scale)
