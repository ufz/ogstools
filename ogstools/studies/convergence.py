from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.propertylib import (
    MatrixProperty,
    Property,
    VectorProperty,
)

_element_length_key = "element_length_mean"

# TODO: add grid convergence Index GCI
# TODO: different properties based on the same data_name -> need other key


def sampled_meshes(target_topology: pv.DataSet, meshes: list[pv.DataSet]):
    meshes_resampled = []
    for mesh in meshes:
        mesh_temp = deepcopy(target_topology)
        mesh_temp = mesh_temp.sample(mesh, pass_cell_data=False)
        meshes_resampled += [mesh_temp]

    return meshes_resampled


def element_length_mean(mesh: pv.DataSet):
    if mesh.get_cell(0).dimension == 1:
        return np.mean(mesh.compute_cell_sizes().cell_data["Length"])
    if mesh.get_cell(0).dimension == 2:
        return np.mean(np.sqrt(mesh.compute_cell_sizes().cell_data["Area"]))
    # if mesh.get_cell(0).dimension == 3:
    #     return np.mean(mesh.compute_cell_sizes().cell_data["Volume"]**(1/3))
    msg = f"Dimension {mesh.get_cell(0).dimension} not yet supported"
    raise ValueError(msg)


def error(mesh: pv.DataSet, mesh_reference: pv.DataSet, property: Property):
    _p_val = (
        property.magnitude
        if isinstance(property, (VectorProperty, MatrixProperty))
        else property
    )

    def _data(m: pv.DataSet):
        return _p_val.values(m.point_data[property.data_name])

    return np.linalg.norm(
        _data(mesh_reference) - _data(mesh), axis=0, ord=2
    ) / np.linalg.norm(_data(mesh_reference), axis=0, ord=2)


def plot_property(property: Property, convergence_data: dict, ax: plt.Axes):
    errors_per_mesh = convergence_data[property.data_name]
    e_lengths = convergence_data[_element_length_key]
    linear_refs_per_mesh = [
        (errors_per_mesh[0] / e_lengths[0]) * el**1 for el in e_lengths
    ]
    quadratic_refs_per_mesh = [
        (errors_per_mesh[0] / e_lengths[0] ** 2) * el**2 for el in e_lengths
    ]

    ax.loglog(e_lengths, errors_per_mesh, "-o")
    ax.loglog(e_lengths, linear_refs_per_mesh, "--", c="k")
    ax.loglog(e_lengths, quadratic_refs_per_mesh, ":", c="k")
    ax.set_xlabel("mean element length / m")
    ax.set_ylabel(f"{property.output_name} error / -")
    ax.legend(["L2 Norm", "linear convergence", "quadratic convergence"])
    ax.grid(True, "major", "both", alpha=0.5)
    ax.grid(True, "minor", "both", alpha=0.1)
    return ax


def plot_convergence(
    target_topology: pv.DataSet,
    meshes: list[pv.DataSet],
    reference_mesh: pv.DataSet,
    properties: Union[Property, list[Property]],
    axs: Union[plt.Axes, list[plt.Axes]],
):
    """
    Plot convergence data for multiple properties on different meshes.

    This function plots convergence data for multiple properties of interest
    calculated across different meshes in comparison to a reference mesh.
    It provides visual representations of how the properties converge with
    increasing mesh refinement.

    For more information see :func:`convergence`.
    """
    if isinstance(axs, plt.Axes):
        axs = [axs]
    if not isinstance(properties, list):
        properties = [properties]
    metrics = convergence(target_topology, meshes, reference_mesh, properties)
    for index, property in enumerate(properties):
        plot_property(property, metrics, axs[index])
    return axs


def richardson_extrapolation(
    mesh_coarse: pv.DataSet,
    mesh_fine: pv.DataSet,
    properties: Union[Property, list[Property]],
):
    """
    Estimate a more accurate approximation of properties on a mesh.

    This function calculates the Richardson Extrapolation based on the change
    in the discretization and the results of the two given meshes.
    The result is given on the topology of mesh_fine.
    See <https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html> for
    more information on this topic.

    :param mesh_coarse:       The mesh with the coarser discretization.
    :param mesh_fine:       The mesh with the finer discretization.
    :param properties:  The properties to be extrapolated.

    :returns:           The topology of mesh_coarse with the properties values
                        based on Richardson Extrapolation.
    """
    rich_ex = deepcopy(mesh_fine)
    r = element_length_mean(mesh_coarse) / element_length_mean(mesh_fine)
    if not isinstance(properties, list):
        properties = [properties]
    for property in properties:
        f1 = mesh_fine.sample(mesh_coarse).point_data[property.data_name]
        f2 = mesh_fine.point_data[property.data_name]
        rich_ex.point_data[property.data_name] = f2 + (f1 - f2) / (r * r - 1)
    return rich_ex


def convergence(
    target_topology: pv.DataSet,
    meshes: list[pv.DataSet],
    reference_mesh: pv.DataSet,
    properties: Union[Property, list[Property]],
):
    """
    Compute the convergence of numerical simulations on multiple meshes.

    The convergence is assessed for multiple properties of interest by comparing
    the results on various meshes to a common reference mesh.

    :param target_topology: The mesh providing the topology for result sampling.
    :param meshes:          A list of meshes which get evaluated.
    :param reference_mesh:  The reference mesh to compare against.
    :param properties:      A list of properties to be compared for convergence.

    :returns: A dictionary containing convergence information for each property.
              The keys are the property names, and the values are lists of
              convergence values for each mesh.

              Convergence is calculated based on the error between the results
              obtained on each mesh and the reference mesh, normalized by the
              mean element length of the target topology mesh.
    """
    meshes_sampled = sampled_meshes(target_topology, meshes + [reference_mesh])
    mean_lengts = [element_length_mean(mesh) for mesh in meshes]
    convergence_dict = {_element_length_key: mean_lengts}
    if not isinstance(properties, list):
        properties = [properties]
    for property in properties:
        convergence_dict[property.data_name] = [
            error(mesh_sampled, meshes_sampled[-1], property)
            for mesh_sampled in meshes_sampled[:-1]
        ]
    return convergence_dict
