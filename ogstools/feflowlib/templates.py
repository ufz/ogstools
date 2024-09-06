# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#
from pathlib import Path

from ogstools.ogs6py import Project


def steady_state_diffusion(saving_path: Path, prj: Project) -> Project:
    """
    A template for a steady state diffusion process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param prj: ogs project, which shall be used with the template
    """
    prj.processes.set_process(
        name="SteadyStateDiffusion",
        type="STEADY_STATE_DIFFUSION",
        integration_order=2,
    )
    prj.processes.add_secondary_variable("darcy_velocity", "v")
    prj.time_loop.add_process(
        process="SteadyStateDiffusion",
        nonlinear_solver_name="basic_picard",
        convergence_type="DeltaX",
        norm_type="NORM2",
        abstol="1e-15",
        time_discretization="BackwardEuler",
    )
    prj.time_loop.set_stepping(
        process="SteadyStateDiffusion",
        type="SingleStep",
        t_initial="0",
        t_end="1",
        repeat="1",
        delta_t="0.25",
    )
    prj.time_loop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[],
    )
    prj.nonlinear_solvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="10",
        linear_solver="general_linear_solver",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="cg",
        precon_type="jacobi",
        max_iteration_step="100000",
        error_tolerance="1e-6",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="CG",
        precon_type="DIAGONAL",
        max_iteration_step="100000",
        error_tolerance="1e-6",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="petsc",
        prefix="sd",
        solver_type="cg",
        precon_type="bjacobi",
        max_iteration_step="10000",
        error_tolerance="1e-16",
    )
    return prj


def liquid_flow(saving_path: Path, prj: Project, dimension: int = 3) -> Project:
    """
    A template for a steady liquid flow process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param prj: ogs project, which shall be used with the template
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
    prj.processes.set_process(
        name="LiquidFlow",
        type="LIQUID_FLOW",
        integration_order=2,
        specific_body_force=gravity,
        linear="true",
    )
    prj.processes.add_secondary_variable("darcy_velocity", "v")
    prj.time_loop.add_process(
        process="LiquidFlow",
        nonlinear_solver_name="basic_picard",
        convergence_type="DeltaX",
        norm_type="NORM2",
        abstol="1e-10",
        time_discretization="BackwardEuler",
    )
    prj.time_loop.set_stepping(
        process="LiquidFlow",
        type="FixedTimeStepping",
        t_initial="0",
        t_end="1",
        repeat="1",
        delta_t="1",
    )
    prj.time_loop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[],
    )
    prj.nonlinear_solvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="10",
        linear_solver="general_linear_solver",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="cg",
        precon_type="jacobi",
        max_iteration_step="10000",
        error_tolerance="1e-20",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="CG",
        precon_type="DIAGONAL",
        max_iteration_step="100000",
        error_tolerance="1e-20",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="petsc",
        prefix="lf",
        solver_type="cg",
        precon_type="bjacobi",
        max_iteration_step="10000",
        error_tolerance="1e-16",
    )
    return prj


def component_transport(
    saving_path: Path,
    species: list,
    prj: Project,
    dimension: int = 3,
    fixed_out_times: list | None = None,
    time_stepping: list | None = None,
    initial_time: int = 1,
    end_time: int = 1,
) -> Project:
    """
    A template for component transport process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param species:
    :param prj: ogs project, which shall be used with the template
    :param dimension: True, if the model is 2 dimensional.
    :param time_stepping: List of how often a time step should be repeated and its time.
    :param initial_time: Beginning of the simulation time.
    :param end_time: End of the simulation time.
    """
    if dimension == 1:
        gravity = "0"
    elif dimension == 2:
        gravity = "0 0"
    elif dimension == 3:
        gravity = "0 0 0"
    else:
        ValueError("Dimension can be either 1,2 or 3.")
    prj.processes.set_process(
        name="CT",
        type="ComponentTransport",
        coupling_scheme="staggered",
        integration_order=2,
        specific_body_force=gravity,
    )
    output_variables = species + ["HEAD_OGS"]
    fixed_out_times = [end_time] if fixed_out_times is None else fixed_out_times
    prj.time_loop.add_output(
        type="VTK",
        prefix=str(saving_path),
        variables=output_variables,
        repeat=1,
        each_steps=end_time,
        fixed_output_times=fixed_out_times,
    )
    prj.nonlinear_solvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="10",
        linear_solver="general_linear_solver",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="bicgstab",
        precon_type="ilut",
        max_iteration_step="10000",
        error_tolerance="1e-10",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="CG",
        precon_type="DIAGONAL",
        max_iteration_step="100000",
        error_tolerance="1e-20",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="petsc",
        prefix="ct",
        solver_type="bcgs",
        precon_type="bjacobi",
        max_iteration_step="10000",
        error_tolerance="1e-16",
    )

    for _i in range(len(species) + 1):
        prj.time_loop.add_process(
            process="CT",
            nonlinear_solver_name="basic_picard",
            convergence_type="DeltaX",
            norm_type="NORM2",
            reltol="1e-6",
            time_discretization="BackwardEuler",
        )
        if time_stepping is None:
            time_stepping = [(1, 1)]

        prj.time_loop.set_stepping(
            process="CT",
            type="FixedTimeStepping",
            t_initial=initial_time,
            t_end=end_time,
            repeat=time_stepping[0][0],
            delta_t=time_stepping[0][1],
        )

        for time_step in time_stepping[1:]:
            prj.time_loop.add_time_stepping_pair(
                repeat=time_step[0], delta_t=time_step[1], process="CT"
            )
    return prj


def hydro_thermal(
    saving_path: Path, prj: Project, dimension: int = 3
) -> Project:
    """
    A template for a hydro-thermal process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param prj: ogs project, which shall be used with the template
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
    prj.processes.set_process(
        name="HydroThermal",
        type="HT",
        integration_order=3,
        specific_body_force=gravity,
    )
    prj.processes.add_secondary_variable("darcy_velocity", "v")
    prj.time_loop.add_process(
        process="HydroThermal",
        nonlinear_solver_name="basic_picard",
        convergence_type="DeltaX",
        norm_type="NORM2",
        abstol="1e-16",
        time_discretization="BackwardEuler",
    )
    prj.time_loop.set_stepping(
        process="HydroThermal",
        type="FixedTimeStepping",
        t_initial="0",
        t_end="1e11",
        repeat="1",
        delta_t="1e10",
    )
    prj.time_loop.add_time_stepping_pair(
        process="HydroThermal",
        repeat="1",
        delta_t="1e10",
    )
    prj.time_loop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[],
    )
    prj.nonlinear_solvers.add_non_lin_solver(
        name="basic_picard",
        type="Picard",
        max_iter="100",
        linear_solver="general_linear_solver",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="lis",
        solver_type="bicgstab",
        precon_type="jacobi",
        max_iteration_step="10000",
        error_tolerance="1e-20",
    )
    prj.linear_solvers.add_lin_solver(
        name="general_linear_solver",
        kind="eigen",
        solver_type="SparseLU",
        scaling="true",
    )
    return prj
