# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from itertools import product
from typing import TypeVar

import numpy as np
import pandas as pd
import pyvista as pv
from typeguard import typechecked

from ogstools.variables import Variable, get_preset

Mesh = TypeVar("Mesh", bound=pv.UnstructuredGrid)


def _raw_differences_all_data(base_mesh: Mesh, subtract_mesh: Mesh) -> Mesh:
    diff = base_mesh.copy(deep=True)
    for point_data_key in base_mesh.point_data:
        diff.point_data[point_data_key] -= subtract_mesh.point_data[
            point_data_key
        ]
    for cell_data_key in base_mesh.cell_data:
        if cell_data_key == "MaterialIDs":
            continue
        diff.cell_data[cell_data_key] -= subtract_mesh.cell_data[cell_data_key]
    return diff


def difference(
    base_mesh: Mesh,
    subtract_mesh: Mesh,
    variable: Variable | str | None = None,
) -> Mesh:
    """
    Compute the difference of variables between two meshes.

    :param base_mesh:       The mesh to subtract from.
    :param subtract_mesh:   The mesh whose data is to be subtracted.
    :param variable:   The variable of interest. If not given, all
                            point and cell_data will be processed raw.
    :returns:   A new mesh containing the difference of `variable` or
                of all datasets between both meshes.
    """
    if variable is None:
        return _raw_differences_all_data(base_mesh, subtract_mesh)
    if isinstance(variable, Variable):
        vals = np.asarray(
            [variable.transform(mesh) for mesh in [base_mesh, subtract_mesh]]
        )
        outname = variable.output_name + "_difference"
    else:
        vals = np.asarray(
            [mesh[variable] for mesh in [base_mesh, subtract_mesh]]
        )
        outname = variable + "_difference"

    diff_mesh = base_mesh.copy(deep=True)
    diff_mesh.clear_point_data()
    diff_mesh.clear_cell_data()
    diff_mesh[outname] = np.empty(vals.shape[1:])
    diff_mesh[outname] = vals[0] - vals[1]
    return diff_mesh


def difference_pairwise(
    meshes_1: list | np.ndarray,
    meshes_2: list | np.ndarray,
    variable: Variable | str | None = None,
) -> np.ndarray:
    """
    Compute pairwise difference between meshes from two lists/arrays
    (they have to be of the same length).

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes whose data is subtracted
                     from the first list/array of meshes - meshes_1.
    :param variable:   The variable of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `variable`
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
            difference(m1, m2, variable)
            for m1, m2 in zip(meshes_1, meshes_2, strict=False)
        ]
    )


@typechecked
def difference_matrix(
    meshes_1: list | np.ndarray,
    meshes_2: list | np.ndarray | None = None,
    variable: Variable | str | None = None,
) -> np.ndarray:
    """
    Compute the difference between all combinations of two meshes
    from one or two arrays based on a specified variable.

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes, it is subtracted from
                     the first list/array of meshes - meshes_1 (optional).
    :param variable:   The variable of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `variable`
                or all datasets between meshes_1 and meshes_2 for all possible
                combinations.
    """
    meshes_1 = np.asarray(meshes_1).flatten()
    if meshes_2 is None:
        meshes_2 = meshes_1.copy()
    meshes_2 = np.asarray(meshes_2).flatten()
    diff_meshes = [
        difference(m1, m2, variable) for m1, m2 in product(meshes_1, meshes_2)
    ]
    return np.asarray(diff_meshes).reshape((len(meshes_1), len(meshes_2)))


@typechecked
def interp_points(points: np.ndarray, resolution: int = 100) -> np.ndarray:
    """
    Provides lists of points on every segment at a line profile between \
          arbitrary number of points pairs.

    :param points: Numpy array of N points to sample between.\
             Has to be of shape (N, 3).

    :param resolution: Resolution of the sampled profile. Total number of \
          points within all profile segments.

    :returns: Numpy array of shape (N, 3), without duplicated nodal points.
    """
    profile = np.zeros([0, 3])
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

    npoints_per_segment = np.ceil(
        distances / np.sum(distances) * resolution
    ).astype(int)

    for index in range(len(points) - 1):
        new_seg_points = np.linspace(
            points[index], points[index + 1], npoints_per_segment[index], False
        )
        profile = np.vstack([profile, new_seg_points])

    return np.vstack([profile, points[-1]])


@typechecked
def distance_in_segments(
    profile_nodes: np.ndarray, profile: np.ndarray
) -> np.ndarray:
    """
    Calculate the distance within segments of a polyline profile.

    :param profile_nodes: 2D array of N points (profile nodes) of shape (N, 3)
    :param profile: output from interp_points function. 2D array of N points \
         (profile nodes) of shape (N, 3)

    :return: 1D array of distances in each segment to its starting point \
        of shape (N, 3), where N is the number of points in profile
    """
    point_index = [
        np.argmin(np.sum(np.abs(profile - pt), axis=1)) for pt in profile_nodes
    ]
    if not (point_index[0] == 0 and point_index[-1] == profile.shape[0] - 1):
        err_msg = "Something went wrong with generating profile_points!"
        raise ValueError(err_msg)
    dist_in_segment = np.zeros([profile.shape[0]])
    for pt_id in range(len(point_index) - 1):
        dist_current_segment = (
            profile[point_index[pt_id] : point_index[pt_id + 1]]
            - profile[point_index[pt_id]]
        )
        dist_current_segment = np.linalg.norm(dist_current_segment, axis=1)
        dist_in_segment[
            point_index[pt_id] : point_index[pt_id + 1],
        ] = dist_current_segment

    # Handle last point
    dist_in_segment[-1] = np.linalg.norm(profile[-1] - profile_nodes[-2])

    return dist_in_segment


@typechecked
def distance_in_profile(points: np.ndarray) -> np.ndarray:
    """
    :param points: 2D array of N points (profile nodes) of shape (N, 3)

    :return: 1D array of distances of each point to the beginning of the \
          profile (first row in points), shape of (N,)
    """
    return np.concatenate(
        ([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


def sample_polyline(
    mesh: pv.UnstructuredGrid,
    variables: str | Variable | list[str] | list[Variable],
    profile_nodes: np.ndarray,
    resolution: int | None = 100,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Sample one or more variables along a polyline.
    Profiles created by user can be passed as profile_nodes parameter. In this
    case user should also set resolution to None in order to avoid further
    interpolation between the points.

    :param mesh: Mesh from which variables will be sampled.
    :param variables: Name or list of names of variables to sample.
    :param profile_nodes: 2D array of N points (profile nodes) of shape (N, 3)
    :param resolution: Total number of sampling points.

    :returns:   tuple containing DataFrame with results of the profile sampling
                and Numpy array of distances from the beginning of the profile
                at points defined in profile_points.
    """
    _variables = [variables] if not isinstance(variables, list) else variables
    variables = [get_preset(var, mesh) for var in _variables]

    if resolution is None:
        # Only cumulative distance alongside the profile will be returned
        profile_points = profile_nodes
    assert isinstance(resolution, int)
    profile_points = interp_points(profile_nodes, resolution=resolution)
    sampled_data_dist_in_segment = distance_in_segments(
        profile_nodes, profile_points
    )
    dist_at_nodes = distance_in_profile(profile_nodes)

    sampled_data_distance = distance_in_profile(profile_points)

    line = pv.PolyData(profile_points)
    sampled_data = line.sample(mesh)

    output_data = {["x", "y", "z"][i]: profile_points[:, i] for i in [0, 1, 2]}

    # TODO: data should be written in output_name otherwise different
    # variables with the same data_name will override each other
    for variable_current in variables:
        # TODO: workaround for Issue 59
        if variable_current.data_name in sampled_data.point_data:
            variable_name = variable_current.data_name
        elif variable_current.output_name in sampled_data.point_data:
            variable_name = variable_current.output_name
        else:
            err_msg = "Cannot match variable name to variables available\
                in mesh!"
            raise KeyError(err_msg)
        sampled_data_variable = sampled_data[variable_name]
        if isinstance(variable_current, Variable):
            sampled_data_variable = variable_current.transform(
                data=sampled_data_variable
            )
        if variable_name not in output_data:
            if len(sampled_data_variable.shape) > 1:
                # Vector variables
                for variable_id in range(sampled_data_variable.shape[1]):
                    variable_key = f"{variable_name}_{variable_id}"
                    output_data[variable_key] = sampled_data_variable[
                        :, variable_id
                    ]
            else:
                # Scalar variables
                output_data[variable_name] = sampled_data_variable

    output_data["dist"] = sampled_data_distance
    if isinstance(resolution, int):
        output_data["dist_in_segment"] = sampled_data_dist_in_segment

    return pd.DataFrame.from_dict(output_data), dist_at_nodes
