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


def sampled_meshes(meshes: list[pv.DataSet]):
    reference_mesh = meshes[0]
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


# ToDo max_value = (max(_p_val.values(mesh_reference.point_data[property.data_name])))


def plot_property(property: Property, convergence_data: dict, ax=0):
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


def plot_convergence(sim_result_files, ts, properties: list[Property], axs):
    d = convergence(sim_result_files, ts, properties)
    for pos, property in enumerate(properties):
        plot_property(property, d, axs[pos])
    return axs


# ToDo refinement_ratio to be removed
def richardson_extrapolation(
    mesh1: pv.DataSet, mesh2: pv.DataSet, property: Property
):
    # https://www.sd.rub.de/downloads/Convergence_FEM.pdf

    # check that there are at least 3 meshes
    rich_ex = deepcopy(mesh1)
    f1 = mesh1.point_data[property.data_name]
    f2 = mesh2.point_data[property.data_name]
    r = element_length_mean(mesh2) / element_length_mean(mesh1)
    rich_ex.point_data[property.data_name] = f1 + (f1 - f2) / (r * r - 1)
    return rich_ex


def convergence(sim_result_files, ts, properties: list[Property]):
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
        rich_ex = richardson_extrapolation(
            meshes_sampled[-2], meshes_sampled[-1], property
        )
        convergence_dict[property.data_name] = [
            error(mesh_sampled, rich_ex, property)
            for mesh_sampled in meshes_sampled
        ]
    return convergence_dict
