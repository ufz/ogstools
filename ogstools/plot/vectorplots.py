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
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    if projection is None:
        projection = int(np.argmax(mean_normal))
    i_id, j_id = np.delete([0, 1, 2], projection)
    _mesh = mesh.copy()
    for key in _mesh.point_data:
        if key not in [variable.data_name, variable.mask]:
            del _mesh.point_data[key]
    i_pts = np.linspace(mesh.bounds[2 * i_id], mesh.bounds[2 * i_id + 1], n_pts)
    j_pts = np.linspace(mesh.bounds[2 * j_id], mesh.bounds[2 * j_id + 1], n_pts)
    i_size = i_pts[-1] - i_pts[0]
    j_size = j_pts[-1] - j_pts[0]
    grid = pv.Plane(_mesh.center, mean_normal, i_size, j_size, *[n_pts - 1] * 2)
    # fixes orientation of plane
    if projection != 2:
        grid = grid.rotate_vector(mean_normal, 90)

    # fix cases where the plane misaligns with the mesh by floating point error
    grid.points = grid.points.astype("float64")
    offset = np.asarray(grid.center) - np.asarray(_mesh.center)
    grid = grid.translate(-offset)

    grid = grid.sample(_mesh, pass_cell_data=False)
    values = variable.transform(grid.point_data[variable.data_name])

    values[np.argwhere(grid["vtkValidPointMask"] == 0), :] = np.nan
    if np.shape(values)[-1] == 3:
        values = np.delete(values, projection, 1)
    val = np.reshape(values, (n_pts, n_pts, 2))

    if variable.mask in grid.point_data:
        mask = np.reshape(grid.point_data[variable.mask], (n_pts, n_pts))
        val[mask == 0, :] = 0
    val_norm = np.linalg.norm(np.nan_to_num(val), axis=-1)
    lw = 2.5 * val_norm / max(1e-16, np.max(val_norm)) * setup.linewidth
    i_grid, j_grid = np.meshgrid(i_pts, j_pts)

    return (i_grid, j_grid, val[..., 0], val[..., 1], lw)


def streamlines(
    mesh: pv.DataSet,
    ax: plt.Axes,
    variable: Vector,
    projection: int | None = None,
) -> None:
    """
    Plot the vector streamlines on a matplotlib axis.

    :param mesh:        Mesh containing the vector variable
    :param ax:          Matplotlib axis to plot onto
    :param variable:    Vector variable to visualize
    :param projection:  Index of flat dimension (e.g. 2 for z axis),
                        gets automatically determined if not given
    """
    if (setup.num_streamline_interp_pts) is None:
        return
    x_g, y_g, u, v, lw = _vectorfield(mesh, variable, projection)
    ax.streamplot(x_g, y_g, u, v, color="k", linewidth=lw, density=1.5)


def quiver(
    mesh: pv.DataSet,
    ax: plt.Axes,
    variable: Vector,
    projection: int | None = None,
    glyph_type: Literal["arrow", "line"] = "arrow",
) -> None:
    """
    Plot arrows or lines corresponding to vectors on a matplotlib axis.

    :param mesh:        Mesh containing the vector variable
    :param ax:          Matplotlib axis to plot onto
    :param variable:    Vector variable to visualize
    :param projection:  Index of flat dimension (e.g. 2 for z axis),
                        gets automatically determined if not given
    :param glyph_type:  Whether to plot arrows or lines.
    """

    x_g, y_g, u, v, _ = _vectorfield(mesh, variable, projection)
    line_args = (
        {"headlength": 0, "headaxislength": 0, "headwidth": 1, "pivot": "mid"}
        if glyph_type == "line"
        else {}
    )
    ax.quiver(x_g, y_g, u, v, **line_args, scale=1.0 / 0.03)
