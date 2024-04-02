# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


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
        raise Exception(msg)
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
    return np.unique(np.concatenate((left, [center], right)))
