# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#
from pathlib import Path
from typing import Optional

from ogs6py import ogs


def steady_state_diffusion(saving_path: Path, model: ogs.OGS = None) -> ogs.OGS:
    """
    A template for a steady state diffusion process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param model: ogs model, which shall be used with the template
    """
    model.processes.set_process(
        name="SteadyStateDiffusion",
        type="STEADY_STATE_DIFFUSION",
        integration_order=2,
    )
    model.processes.add_process_variable(
        secondary_variable="darcy_velocity", output_name="v"
    )
    model.timeloop.add_process(
        process="SteadyStateDiffusion",
        nonlinear_solver_name="basic_picard",
        convergence_type="DeltaX",
        norm_type="NORM2",
        abstol="1e-15",
        time_discretization="BackwardEuler",
    )
    model.timeloop.set_stepping(
        process="SteadyStateDiffusion",
        type="SingleStep",
        t_initial="0",
        t_end="1",
        repeat="1",
        delta_t="0.25",
    )
    model.timeloop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[],
    )
    model.nonlinsolvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="10",
        linear_solver="general_linear_solver",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="cg",
        precon_type="jacobi",
        max_iteration_step="100000",
        error_tolerance="1e-6",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="CG",
        precon_type="DIAGONAL",
        max_iteration_step="100000",
        error_tolerance="1e-6",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="petsc",
        prefix="sd",
        solver_type="cg",
        precon_type="bjacobi",
        max_iteration_step="10000",
        error_tolerance="1e-16",
    )
    return model


def liquid_flow(
    saving_path: Path, model: ogs.OGS = None, dimension: int = 3
) -> ogs.OGS:
    """
    A template for a steady liquid flow process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param model: ogs model, which shall be used with the template
    :param dimension: True, if the model is 2 dimensional.
    """
    # FEFLOW uses hydraulic HEAD instead of pressure as primary variable,
    # which is why the gravity is calculated in the hydraulic conductivity.
    # That is why, in this case we can set the gravity to 0 in all needed spatial
    # directions.
    if dimension == 1:
        gravity = "0"
    elif dimension == 2:
        gravity = "0 0"
    elif dimension == 3:
        gravity = "0 0 0"
    else:
        ValueError("Dimension can be either 1,2 or 3.")
    model.processes.set_process(
        name="LiquidFlow",
        type="LIQUID_FLOW",
        integration_order=2,
        specific_body_force=gravity,
        linear="true",
    )
    model.processes.add_process_variable(
        secondary_variable="darcy_velocity", output_name="v"
    )
    model.timeloop.add_process(
        process="LiquidFlow",
        nonlinear_solver_name="basic_picard",
        convergence_type="DeltaX",
        norm_type="NORM2",
        abstol="1e-10",
        time_discretization="BackwardEuler",
    )
    model.timeloop.set_stepping(
        process="LiquidFlow",
        type="FixedTimeStepping",
        t_initial="0",
        t_end="1",
        repeat="1",
        delta_t="1",
    )
    model.timeloop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[],
    )
    model.nonlinsolvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="10",
        linear_solver="general_linear_solver",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="cg",
        precon_type="jacobi",
        max_iteration_step="10000",
        error_tolerance="1e-20",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="CG",
        precon_type="DIAGONAL",
        max_iteration_step="100000",
        error_tolerance="1e-20",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="petsc",
        prefix="lf",
        solver_type="cg",
        precon_type="bjacobi",
        max_iteration_step="10000",
        error_tolerance="1e-16",
    )
    return model


def component_transport(
    saving_path: Path,
    species: list,
    model: ogs.OGS = None,
    dimension: int = 3,
    fixed_out_times: Optional[list] = None,
) -> ogs.OGS:
    """
    A template for component transport process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param species:
    :param model: ogs model, which shall be used with the template
    :param dimension: True, if the model is 2 dimensional.
    """
    if dimension == 1:
        gravity = "0"
    elif dimension == 2:
        gravity = "0 0"
    elif dimension == 3:
        gravity = "0 0 0"
    else:
        ValueError("Dimension can be either 1,2 or 3.")
    model.processes.set_process(
        name="CT",
        type="ComponentTransport",
        coupling_scheme="staggered",
        integration_order=2,
        specific_body_force=gravity,
    )
    output_variables = species + ["HEAD_OGS"]
    fixed_out_times = [1] if fixed_out_times is None else fixed_out_times
    model.timeloop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat=1,
        each_steps=48384000,
        variables=output_variables,
        fixed_output_times=fixed_out_times,
    )
    model.nonlinsolvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="10",
        linear_solver="general_linear_solver",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="bicgstab",
        precon_type="ilut",
        max_iteration_step="10000",
        error_tolerance="1e-10",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="CG",
        precon_type="DIAGONAL",
        max_iteration_step="100000",
        error_tolerance="1e-20",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="petsc",
        prefix="ct",
        solver_type="bcgs",
        precon_type="bjacobi",
        max_iteration_step="10000",
        error_tolerance="1e-16",
    )
    return model


def hydro_thermal(
    saving_path: Path, model: ogs.OGS = None, dimension: int = 3
) -> ogs.OGS:
    """
    A template for a hydro-thermal process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param model: ogs model, which shall be used with the template
    :param dimension: to known if the model is 2D or not
    """
    if dimension == 1:
        gravity = "0"
    elif dimension == 2:
        gravity = "0 0"
    elif dimension == 3:
        gravity = "0 0 0"
    else:
        ValueError("Dimension can be either 1,2 or 3.")
    model.processes.set_process(
        name="HydroThermal",
        type="HT",
        integration_order=3,
        specific_body_force=gravity,
    )
    model.processes.add_process_variable(
        secondary_variable="darcy_velocity", output_name="v"
    )
    model.timeloop.add_process(
        process="HydroThermal",
        nonlinear_solver_name="basic_picard",
        convergence_type="DeltaX",
        norm_type="NORM2",
        abstol="1e-16",
        time_discretization="BackwardEuler",
    )
    model.timeloop.set_stepping(
        process="HydroThermal",
        type="FixedTimeStepping",
        t_initial="0",
        t_end="1e11",
        repeat="1",
        delta_t="1e10",
    )
    model.timeloop.add_time_stepping_pair(
        process="HydroThermal",
        repeat="1",
        delta_t="1e10",
    )
    model.timeloop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[],
    )
    model.nonlinsolvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="100",
        linear_solver="general_linear_solver",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="bicgstab",
        precon_type="jacobi",
        max_iteration_step="10000",
        error_tolerance="1e-20",
    )
    model.linsolvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="SparseLU",
        scaling="true",
    )
    return model
