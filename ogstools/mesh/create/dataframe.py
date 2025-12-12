# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path

import pandas as pd


def dataframe_from_csv(
    layer_set_id: int,
    layer_sets_csvfile: Path | str,
    surfaces: dict[int, Path] | Path | str,
) -> pd.DataFrame:
    """Create a DataFrame from CSV data for a specific layer set.

    This function reads a CSV file containing layer set information and filters
    it for a specific layer set ID. It then maps layer IDs to surface files and
    returns a DataFrame with material IDs, surface filenames, and resolutions.
    The surfaces corresponding to the `layer_id` must be sorted from top to
    bottom (high z values to low z values).

    :param layer_set_id:        The layer set ID to filter the CSV data by.
    :param layer_sets_csvfile:  CSV file containing layer set information.
    :param surfaces:    Either a dictionary mapping layer IDs to surface files,
                        or a path to a directory containing surface files. If a
                        directory path is provided, all .vtu files in the
                        directory will be used. In that case, the `layer_id` in
                        the layer_sets_csvfile has to correspond to the
                        sorted list of .vtu files in the directory.

    :returns:   A DataFrame containing columns 'material_id', 'filename', and
                'resolution' for the specified layer set.

    :raises: ValueError, If no model is defined with the given layer_set_id.
    """
    dfs = pd.read_csv(layer_sets_csvfile)
    dfs = dfs[dfs["set_id"] == layer_set_id]
    if len(dfs) == 0:
        msg = f"no model defined with {layer_set_id}"
        raise ValueError(msg)
    if isinstance(surfaces, Path | str):
        layer_surf_map = dict(enumerate(sorted(Path(surfaces).glob("*.vtu"))))
    else:
        layer_surf_map = surfaces
    dfs["filename"] = [layer_surf_map[l_id] for l_id in dfs["layer_id"]]
    dfs.sort_values(by=["layer_id"])
    interest = ["material_id", "filename", "resolution"]
    return dfs[interest]
