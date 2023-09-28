from copy import deepcopy

import numpy as np

from ogstools.meshlib import MeshSeries
from ogstools.propertylib import (
    MatrixProperty,
    Property,
    ScalarProperty,
    VectorProperty,
)

_element_length_key = "element_length_mean"


def sampled_meshes(meshes):
    reference_mesh = meshes[0]
    sim_results_sampled = []
    for mesh in meshes:
        mesh_temp = deepcopy(reference_mesh)
        mesh_temp = mesh_temp.sample(mesh, pass_cell_data=False)
        sim_results_sampled += [mesh_temp]

    return sim_results_sampled


def element_length_mean(mesh):
    return np.mean(np.sqrt(mesh.compute_cell_sizes().cell_data["Area"]))


def error(mesh, mesh_reference, property):
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


# ToDo max_value = (max(_p_val.values(mesh_reference.point_data[property.data_name])))


def plot_property(property, convergence_data, ax=0):
    errors_per_mesh = convergence_data[property.data_name]
    e_lengths = convergence_data[_element_length_key]
    lin_refs_per_mesh = [
        (errors_per_mesh[0] / e_lengths[0]) * el**1 for el in e_lengths
    ]  # to plot
    quadratic_refs_per_mesh = [
        (errors_per_mesh[0] / e_lengths[0] ** 2) * el**2 for el in e_lengths
    ]  # to plot

    ax.loglog(e_lengths, errors_per_mesh, "-o")
    ax.loglog(e_lengths, lin_refs_per_mesh, "--", c="k")
    ax.loglog(e_lengths, quadratic_refs_per_mesh, ":", c="k")
    # ax.set_title(title, loc="center", y=1.02)
    ax.set_xlabel("mean element length / m")
    ax.set_ylabel(f"{property.output_name} error / -")
    ax.legend(["L2 Norm", "p=1", "p=2"])
    ax.grid(True, "major", "both", alpha=0.5)
    ax.grid(True, "minor", "both", alpha=0.1)
    return ax


def plot_convergence(
    sim_result_files, ts, properties: list[ScalarProperty], axs
):
    d = convergence(sim_result_files, ts, properties)
    for pos, property in enumerate(properties):
        plot_property(property, d, axs[pos])
    return axs


# ToDo refinement_ratio to be removed
def richardson_extrapolation(meshes, property: Property, r=2):
    # https://www.sd.rub.de/downloads/Convergence_FEM.pdf

    # check that there are at least 3 meshes
    rich_ex = deepcopy(meshes[0])
    f1 = meshes[-1].point_data[property.data_name]
    f2 = meshes[-2].point_data[property.data_name]
    rich_ex.point_data[property.data_name] = f1 + (f1 - f2) / (r * r - 1)
    return rich_ex


def convergence(sim_result_files, ts, properties):
    meshes = [
        MeshSeries(sim_result).read(ts) for sim_result in sim_result_files
    ]
    meshes_sampled = sampled_meshes(meshes)

    # dictionary suitable for direct converting to pandas
    # pandas.DataFrame(d)

    convergence_dict = {
        _element_length_key: [element_length_mean(mesh) for mesh in meshes]
    }
    for property in properties:
        # ToDo refinement_ratio with elength[-1]/elength[-2] as input to richardson_extrapolation
        rich_ex = richardson_extrapolation(meshes_sampled, property)
        convergence_dict[property.data_name] = [
            error(mesh_sampled, rich_ex, property)
            for mesh_sampled in meshes_sampled
        ]
    return convergence_dict
