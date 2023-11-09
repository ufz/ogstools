import pathlib

import numpy as np
import pandas as pd


def layer_names(folder: str):
    def layer_names_folder(row):
        txt = "{layer_id:02d}_{model_unit}.vtu"

        layer_name = txt.format(
            layer_id=row["layer_id"],
            model_unit=row["model_unit"],
            folder=folder,
        )

        return pathlib.Path(folder) / pathlib.Path(layer_name)

    return layer_names_folder


def dataframe_from_csv(
    layer_set_id: int,
    layer_sets_csvfile: pathlib.Path,
    parameters_csvfile: pathlib.Path,
    surfaces_folder: pathlib.Path,
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


def centered_range(min, max, step):
    center = np.mean([min, max])
    # To ensure equal model sizes for a convergence study, we find the largest
    # 2^n * step which is below an upper resolution limit (assumed to be the
    # max. resolution in the study)
    max_res = 500.0
    max_step = step * (2 ** int(np.log2(max_res / step)))
    # With that max step size we set new bounds
    l_bound = center - max_step * ((center - min) // max_step)
    r_bound = center + max_step * ((max - center) // max_step)

    left = np.arange(center, l_bound - step, -step)
    right = np.arange(center, r_bound + step, step)
    return np.unique(np.concatenate((left, [center], right)))
