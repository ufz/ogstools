from copy import deepcopy

import numpy as np
import pyvista as pv

from ogstools.meshlib import MeshSeries
from ogstools.propertylib import (
    MatrixProperty,
    Property,
    VectorProperty,
)

_element_length_key = "element_length_mean"


def sampled_meshes(reference_mesh: pv.DataSet, meshes: list[pv.DataSet]):
    sim_results_sampled = []
    for mesh in meshes:
        mesh_temp = deepcopy(reference_mesh)
        mesh_temp = mesh_temp.sample(mesh, pass_cell_data=False)
        sim_results_sampled += [mesh_temp]

    return sim_results_sampled


def element_length_mean(mesh: pv.DataSet):
    return np.mean(np.sqrt(mesh.compute_cell_sizes().cell_data["Area"]))


def error(mesh: pv.DataSet, mesh_reference: pv.DataSet, property: Property):
    _p_val = (
        property.magnitude
        if isinstance(property, (VectorProperty, MatrixProperty))
        else property
    )
    return np.linalg.norm(
        _p_val.values(mesh_reference.point_data[property.data_name])
        - _p_val.values(mesh.point_data[property.data_name]),
        axis=0,
        ord=2,
    ) / np.linalg.norm(
        _p_val.values(mesh_reference.point_data[property.data_name]),
        axis=0,
        ord=2,
    )


# TODO: different properties based on the same data_name -> need other key
def plot_property(property: Property, convergence_data, ax=0):
    errors_per_mesh = convergence_data[property.data_name]
    e_lengths = convergence_data[_element_length_key]
    linear_refs_per_mesh = [
        (errors_per_mesh[0] / e_lengths[0]) * el**1 for el in e_lengths
    ]  # to plot
    quadratic_refs_per_mesh = [
        (errors_per_mesh[0] / e_lengths[0] ** 2) * el**2 for el in e_lengths
    ]  # to plot

    ax.loglog(e_lengths, errors_per_mesh, "-o")
    ax.loglog(e_lengths, linear_refs_per_mesh, "--", c="k")
    ax.loglog(e_lengths, quadratic_refs_per_mesh, ":", c="k")
    # ax.set_title(title, loc="center", y=1.02)
    ax.set_xlabel("mean element length / m")
    ax.set_ylabel(f"{property.output_name} error / -")
    ax.legend(["L2 Norm", "p=1", "p=2"])
    ax.grid(True, "major", "both", alpha=0.5)
    ax.grid(True, "minor", "both", alpha=0.1)
    return ax


def plot_convergence(
    refence_sim_result,
    sim_result_files,
    ts,
    properties: list[Property],
    axs,
):
    if not isinstance(axs, list):
        axs = [axs]
    metrics = convergence(refence_sim_result, sim_result_files, ts, properties)
    for pos, property in enumerate(properties):
        plot_property(property, metrics, axs[pos])
    return axs


# TODO: add grid convergence Index GCI


def richardson_extrapolation(
    mesh1: pv.DataSet, mesh2: pv.DataSet, property: Property, r: float
):
    """
    Estimate a more accurate approximation of a property on a mesh.

    This function assumes that both mesh1 and mesh2 are compatible, meaning they
    share the same structure and topology, but their data should correspond to
    different discretizations, suggestively sampled on the base topology.
    See <https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html> for
    more info on the topic.

    :param mesh1:       The mesh with data from coarser discretization.
    :param mesh2:       The mesh with data from finer discretization.
    :param property:    The property to be extrapolated.
    :param r:           The refinement ratio of the underlying discretizations.

    :returns:           A new DataSet with the extrapolated property values
                        based on Richardson Extrapolation.
    """
    rich_ex = deepcopy(mesh1)
    f1 = mesh1.point_data[property.data_name]
    f2 = mesh2.point_data[property.data_name]
    rich_ex.point_data[property.data_name] = f2 + (f1 - f2) / (r * r - 1)
    return rich_ex


def convergence(
    sim_result_reference, sim_result_files, ts, properties: list[Property]
):
    reference_mesh = MeshSeries(sim_result_reference).read(ts)
    meshes = [
        MeshSeries(sim_result).read(ts) for sim_result in sim_result_files
    ]
    meshes_sampled = sampled_meshes(reference_mesh, meshes)

    # dictionary suitable for direct converting to pandas
    # pandas.DataFrame(d)

    mean_lengts = [element_length_mean(mesh) for mesh in meshes]
    convergence_dict = {_element_length_key: mean_lengts}
    ref_ratio = mean_lengts[-2] / mean_lengts[-1]
    for property in properties:
        rich_ex = richardson_extrapolation(
            meshes_sampled[-2], meshes_sampled[-1], property, ref_ratio
        )
        convergence_dict[property.data_name] = [
            error(mesh_sampled, rich_ex, property)
            for mesh_sampled in meshes_sampled
        ]
    return convergence_dict
