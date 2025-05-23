# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .mesh import Mesh


def layer_names(folder: str) -> Callable[[dict], Path]:
    def layer_names_folder(row: dict) -> Path:
        txt = "{layer_id:02d}_{model_unit}.vtu"

        layer_name = txt.format(
            layer_id=row["layer_id"],
            model_unit=row["model_unit"],
            folder=folder,
        )

        return Path(folder) / Path(layer_name)

    return layer_names_folder


def dataframe_from_csv(
    layer_set_id: int,
    layer_sets_csvfile: Path,
    parameters_csvfile: Path,
    surfaces_folder: Path,
) -> pd.DataFrame:
    dfs = pd.read_csv(layer_sets_csvfile)
    dfs = dfs[dfs["set_id"] == layer_set_id]
    if len(dfs) == 0:
        msg = f"no model defined with {layer_set_id}"
        raise ValueError(msg)
    dfm = pd.read_csv(parameters_csvfile, delimiter=",")
    model_df = dfs.merge(dfm)
    vtu_names = model_df.apply(layer_names(str(surfaces_folder)), axis=1)
    model_df["filename"] = vtu_names
    model_df.sort_values(by=["layer_id"])
    interest = ["material_id", "filename", "resolution"]
    return model_df[interest]


def centered_range(min_val: float, max_val: float, step: float) -> np.ndarray:
    center = np.mean([min_val, max_val])
    # To ensure equal model sizes for a convergence study, we find the largest
    # 2^n * step which is below an upper resolution limit (assumed to be the
    # max. resolution in the study)
    max_res = 500.0
    max_step = step * (2 ** int(np.log2(max_res / step)))
    # With that max step size we set new bounds
    l_bound = center - max_step * ((center - min_val) // max_step)
    r_bound = center + max_step * ((max_val - center) // max_step)

    left = np.arange(center, l_bound - step, -step)
    right = np.arange(center, r_bound + step, step)
    return np.unique(np.concatenate((left, np.asarray([center]), right)))


def reshape_obs_points(
    points: np.ndarray | list, mesh: Mesh | None = None
) -> np.ndarray:
    points = np.asarray(points)

    pts = points.reshape((-1, points.shape[-1]))

    # Add missing columns to comply with pyvista expectations
    if pts.shape[1] == 3:
        pts_pyvista = pts
    elif mesh is None:
        pts_pyvista = np.hstack(
            (pts, np.zeros((pts.shape[0], 3 - pts.shape[1])))
        )
    elif isinstance(mesh, Mesh):
        # Detect and handle flat dimensions
        geom = mesh.points
        flat_axis = np.argwhere(np.all(np.isclose(geom, geom[0]), axis=0))
        flat_axis = flat_axis.flatten()
        if pts.shape[1] + len(flat_axis) < 3:
            err_msg = (
                "Number of flat axis and number of coordinates"
                " in provided points doesn't add up to 3."
                " Please ensure that the provided points match"
                " the plane of the mesh."
            )
            raise RuntimeError(err_msg)
        pts_pyvista = np.empty((pts.shape[0], 3))
        pts_id = 0
        for col_id in range(3):
            if col_id in flat_axis:
                pts_pyvista[:, col_id] = (
                    np.ones((pts.shape[0],)) * geom[0, col_id]
                )
            else:
                pts_pyvista[:, col_id] = pts[:, pts_id]
                pts_id = pts_id + 1
    return pts_pyvista
