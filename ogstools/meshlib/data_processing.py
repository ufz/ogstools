# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from itertools import product
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyvista as pv
from typeguard import typechecked

from ogstools.propertylib import Property
from ogstools.propertylib.properties import get_preset


def _raw_differences_all_data(
    mesh1: pv.UnstructuredGrid, mesh2: pv.UnstructuredGrid
) -> pv.UnstructuredGrid:
    diff_mesh = mesh1.copy(deep=True)
    for point_data_key in mesh1.point_data:
        diff_mesh.point_data[point_data_key] -= mesh2.point_data[point_data_key]
    for cell_data_key in mesh1.cell_data:
        if cell_data_key == "MaterialIDs":
            continue
        diff_mesh.cell_data[cell_data_key] -= mesh2.cell_data[cell_data_key]
    return diff_mesh


def difference(
    mesh1: pv.UnstructuredGrid,
    mesh2: pv.UnstructuredGrid,
    mesh_property: Optional[Union[Property, str]] = None,
) -> pv.UnstructuredGrid:
    """
    Compute the difference of properties between two meshes.

    :param mesh1: The first mesh to be subtracted from.
    :param mesh2: The second mesh whose data is subtracted from the first mesh.
    :param mesh_property:   The property of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   A new mesh containing the difference of `mesh_property` or all
                datasets between mesh1 and mesh2.
    """
    if mesh_property is None:
        return _raw_differences_all_data(mesh1, mesh2)
    if isinstance(mesh_property, Property):
        vals = np.asarray(
            [mesh_property.transform(mesh) for mesh in [mesh1, mesh2]]
        )
        outname = mesh_property.output_name + "_difference"
    else:
        vals = np.asarray([mesh[mesh_property] for mesh in [mesh1, mesh2]])
        outname = mesh_property + "_difference"

    diff_mesh = mesh1.copy(deep=True)
    diff_mesh.clear_data()
    diff_mesh[outname] = np.empty(vals.shape[1:])
    diff_mesh[outname] = vals[0] - vals[1]
    return diff_mesh


def difference_pairwise(
    meshes_1: Union[list, np.ndarray],
    meshes_2: Union[list, np.ndarray],
    mesh_property: Optional[Union[Property, str]] = None,
) -> np.ndarray:
    """
    Compute pairwise difference between meshes from two lists/arrays
    (they have to be of the same length).

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes whose data is subtracted
                     from the first list/array of meshes - meshes_1.
    :param mesh_property:   The property of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `mesh_property`
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
            difference(m1, m2, mesh_property)
            for m1, m2 in zip(meshes_1, meshes_2)
        ]
    )


@typechecked
def difference_matrix(
    meshes_1: Union[list, np.ndarray],
    meshes_2: Optional[Union[list, np.ndarray]] = None,
    mesh_property: Optional[Union[Property, str]] = None,
) -> np.ndarray:
    """
    Compute the difference between all combinations of two meshes
    from one or two arrays based on a specified property.

    :param meshes_1: The first list/array of meshes to be subtracted from.
    :param meshes_2: The second list/array of meshes, it is subtracted from
                     the first list/array of meshes - meshes_1 (optional).
    :param mesh_property:   The property of interest. If not given, all point
                            and cell_data will be processed raw.
    :returns:   An array of meshes containing the differences of `mesh_property`
                or all datasets between meshes_1 and meshes_2 for all possible
                combinations.
    """
    meshes_1 = np.asarray(meshes_1).flatten()
    if meshes_2 is None:
        meshes_2 = meshes_1.copy()
    meshes_2 = np.asarray(meshes_2).flatten()
    diff_meshes = [
        difference(m1, m2, mesh_property)
        for m1, m2 in product(meshes_1, meshes_2)
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

    # Get total length of the profile:
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    npoints_per_segment = np.ceil(
        (distances / np.sum(distances)) * resolution
    ).astype(int)

    for index in range(len(points) - 1):
        vector = points[index + 1] - points[index]
        interp = np.linspace(0, 1, npoints_per_segment[index], endpoint=False)
        new_seg_points = (
            points[index][:, None] + vector[:, None] @ interp[None, :]
        )
        profile = np.vstack([profile, new_seg_points.T])

    return np.vstack([profile, points[-1, :]])


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
    # Get distances within the segment
    point_index = []
    for point in profile_nodes:
        point_index.append(np.sum(np.abs(profile - point), axis=1).argmin())
    if not (point_index[0] == 0 and point_index[-1] == profile.shape[0] - 1):
        err_msg = "Something went wrong with generating profile_points!"
        raise ValueError(err_msg)
    sampled_data_dist_in_segment = np.zeros(
        [
            profile.shape[0],
        ]
    )
    for pt_id in range(len(point_index) - 1):
        dist_current_segment = (
            profile[point_index[pt_id] : point_index[pt_id + 1], :]
            - profile[point_index[pt_id]]
        )
        dist_current_segment = np.linalg.norm(dist_current_segment, axis=1)
        sampled_data_dist_in_segment[
            point_index[pt_id] : point_index[pt_id + 1],
        ] = dist_current_segment

    # Handle last point
    dist_last_point = profile[-1, :] - profile_nodes[-2, :]
    dist_last_point = np.linalg.norm(dist_last_point)
    sampled_data_dist_in_segment[-1] = dist_last_point

    return sampled_data_dist_in_segment


@typechecked
def distance_in_profile(points: np.ndarray) -> np.ndarray:
    """
    :param points: 2D array of N points (profile nodes) of shape (N, 3)

    :return: 1D array of distances of each point to the beginning of the \
          profile (first row in points), shape of (N,)
    """
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0], np.cumsum(distances)))


def sample_polyline(
    mesh: pv.UnstructuredGrid,
    properties: Union[str, Property, list],
    profile_nodes: np.ndarray,
    resolution: Optional[int] = 100,
) -> tuple[pd.DataFrame, np.array]:
    """
    Sample one or more properties along a polyline.
    Profiles created by user can be passed as profile_nodes parameter. In this \
    case user should also set resolution to None in order to avoid further \
    interpolation between the points.

    :param mesh: Mesh from which properties will be sampled.
    :param properties: Name or list of names of properties to sample. \
    :param profile_nodes: 2D array of N points (profile nodes) of shape (N, 3)
    :param resolution: Total number of sampling points.

    :returns: tuple containing DataFrame with results of the profile sampling \
        and Numpy array of distances from the beginning of the profile at \
            points defined in profile_points.
    """
    properties = (
        [properties] if not isinstance(properties, list) else properties
    )
    properties = [get_preset(prop, mesh) for prop in properties]

    if resolution is None:
        # Only cumulative distance alongside the profile will be returned
        profile_points = profile_nodes
        dist_at_nodes = np.empty()
    assert isinstance(resolution, int)
    profile_points = interp_points(profile_nodes, resolution=resolution)
    sampled_data_dist_in_segment = distance_in_segments(
        profile_nodes, profile_points
    )
    dist_at_nodes = distance_in_profile(profile_nodes)

    sampled_data_distance = distance_in_profile(profile_points)

    line = pv.PolyData(profile_points)
    sampled_data = line.sample(mesh)

    # Structure the output data
    output_data = {["x", "y", "z"][i]: profile_points[:, i] for i in [0, 1, 2]}

    for property_current in properties:
        # TODO: workaround for Issue 59
        if property_current.data_name in sampled_data.point_data:
            property_name = property_current.data_name
        elif property_current.output_name in sampled_data.point_data:
            property_name = property_current.output_name
        else:
            err_msg = "Cannot match property name to properties available\
                in mesh!"
            raise KeyError(err_msg)
        sampled_data_property = sampled_data[property_name]
        if isinstance(property_current, Property):
            sampled_data_property = property_current.transform(
                data=sampled_data_property
            )
        if property_name not in output_data:
            if len(sampled_data_property.shape) > 1:
                # Vector properties
                for property_id in range(sampled_data_property.shape[1]):
                    property_key = f"{property_name}_{property_id}"
                    output_data[property_key] = sampled_data_property[
                        :, property_id
                    ]
            else:
                # Scalar properties
                output_data[property_name] = sampled_data_property

    output_data["dist"] = sampled_data_distance
    if isinstance(resolution, int):
        output_data["dist_in_segment"] = sampled_data_dist_in_segment

    return pd.DataFrame.from_dict(output_data), dist_at_nodes
