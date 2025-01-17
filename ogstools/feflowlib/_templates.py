# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#
from pathlib import Path

from ogstools.ogs6py import Project


def create_prj_template(
    saving_path: Path,
    prj: Project,
    process_name: str = "undefined",
    dimension: int = 3,
    fixed_out_times: list | None = None,
    time_stepping: list | None = None,
    initial_time: int = 0,
    end_time: int | float = 1,
    error_tolerance: float = 1e-10,
    output_variables: list | None = None,
    eigen_sparseLu: bool = False,
    petsc_CG: bool = False,
    eigen_CG: bool = True,
) -> Project:
    """
    A template for generic process to be simulated in ogs. It shall be used as a
    template for conversion of FELOW to OGS models and the .

    Note:
    FEFLOW uses hydraulic HEAD instead of pressure as primary variable.
    This is why, no viscosity and density are given in the FELOW model.
    To convert the model and simulate it in OGS, OGS needs to be configured to
    simulate the hydraulic head instead of the pressure. To achieve this, the
    gravity is set to 0, viscosity to 1 and density to an arbitrary value (1).

    :param saving_path: path of ogs simulation results
    :param prj: ogs project, which shall be used with the template
    :param dimension: True, if the model is 2 dimensional.
    :param fixed_out_times: Time steps output will be generated for.
    :param time_stepping: List of how often a time step should be repeated and its time.
    :param initial_time: Beginning of the simulation time.
    :param end_time: End of the simulation time.
    """

    if dimension not in [1, 2, 3]:
        error_msg = "Dimension can be either 1, 2, or 3."
        raise ValueError(error_msg)

    gravity = " ".join(["0"] * dimension)

    if process_name == "undefined":
        prj.processes.set_process(
            name=process_name,
            type=process_name,
            integration_order=2,
            specific_body_force=gravity,
        )
        prj.media.add_property(
            name="to be defined manually", type="to be defined manually"
        )

    prj.processes.add_secondary_variable("darcy_velocity", "v")

    if time_stepping is None:
        time_stepping = [(1, end_time)]

    if process_name != "CT" and "mass" not in process_name:
        prj.time_loop.add_process(
            process=process_name,
            nonlinear_solver_name="basic_picard",
            convergence_type="DeltaX",
            norm_type="NORM2",
            abstol="1e-10",
            time_discretization="BackwardEuler",
        )
        if process_name != "SteadyStateDiffusion":
            prj.time_loop.set_stepping(
                process=process_name,
                type="FixedTimeStepping",
                t_initial=initial_time,
                t_end=end_time,
                repeat=time_stepping[0][0],
                delta_t=time_stepping[0][1],
            )

        for time_step in time_stepping[1:]:
            prj.time_loop.add_time_stepping_pair(
                repeat=time_step[0], delta_t=time_step[1], process=process_name
            )
    fixed_out_times = [end_time] if fixed_out_times is None else fixed_out_times

    prj.time_loop.add_output(
        type="VTK",
        prefix=str(saving_path),
        repeat="1",
        each_steps="1",
        variables=[] if output_variables is None else output_variables,
        fixed_output_times=fixed_out_times,
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
        solver_type="cg",
        precon_type="jacobi",
        max_iteration_step="100000",
        error_tolerance=str(error_tolerance),
    )

    if petsc_CG:
        prj.linear_solvers.add_lin_solver(
            name="general_linear_solver",
            kind="petsc",
            prefix="lf",
            solver_type="cg",
            precon_type="bjacobi",
            max_iteration_step="1000",
            error_tolerance=str(error_tolerance),
        )
    if eigen_CG:
        prj.linear_solvers.add_lin_solver(
            name="general_linear_solver",
            kind="eigen",
            solver_type="CG",
            precon_type="DIAGONAL",
            max_iteration_step="100000",
            error_tolerance=str(error_tolerance),
        )
    if eigen_sparseLu:
        prj.linear_solvers.add_lin_solver(
            name="general_linear_solver",
            kind="eigen",
            solver_type="SparseLU",
            scaling="true",
        )

    return prj


def steady_state_diffusion(
    saving_path: Path,
    prj: Project,
    dimension: int = 3,
    error_tolerance: float = 1e-10,
) -> Project:
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
    create_prj_template(
        saving_path,
        prj,
        process_name="SteadyStateDiffusion",
        dimension=dimension,
        error_tolerance=error_tolerance,
    )
    prj.time_loop.set_stepping(
        process="SteadyStateDiffusion",
        type="SingleStep",
        t_initial="0",
        t_end="1",
        repeat="1",
        delta_t="0.25",
    )

    return prj


def liquid_flow(
    saving_path: Path,
    prj: Project,
    dimension: int = 3,
    fixed_out_times: list | None = None,
    time_stepping: list | None = None,
    initial_time: int = 0,
    end_time: int | float = 1,
    error_tolerance: float = 1e-10,
) -> Project:
    """
    A template for a steady liquid flow process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param prj: ogs project, which shall be used with the template
    :param dimension: True, if the model is 2 dimensional.
    :param fixed_out_times: Time steps output will be generated for.
    :param time_stepping: List of how often a time step should be repeated and its time.
    :param initial_time: Beginning of the simulation time.
    :param end_time: End of the simulation time.
    """

    gravity = " ".join(["0"] * dimension)
    prj.processes.set_process(
        name="LiquidFlow",
        type="LIQUID_FLOW",
        integration_order=2,
        specific_body_force=gravity,
        linear="true",
    )
    return create_prj_template(
        saving_path,
        prj,
        process_name="LiquidFlow",
        dimension=dimension,
        fixed_out_times=fixed_out_times,
        time_stepping=time_stepping,
        initial_time=initial_time,
        end_time=end_time,
        error_tolerance=error_tolerance,
    )


def component_transport(
    saving_path: Path,
    species: list,
    prj: Project,
    dimension: int = 3,
    process_name: str = "CT",
    fixed_out_times: list | None = None,
    time_stepping: list | None = None,
    initial_time: int = 0,
    end_time: int | float = 1,
    error_tolerance: float = 1e-10,
) -> Project:
    """
    A template for component transport process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :param species:
    :param prj: ogs project, which shall be used with the template
    :param dimension: True, if the model is 2 dimensional.
    :param fixed_out_times: Time steps output will be generated for.
    :param time_stepping: List of how often a time step should be repeated and its time.
    :param initial_time: Beginning of the simulation time.
    :param end_time: End of the simulation time.
    """
    gravity = " ".join(["0"] * dimension)
    prj.processes.set_process(
        name=process_name,
        type="ComponentTransport",
        coupling_scheme="staggered",
        integration_order=2,
        specific_body_force=gravity,
    )

    for _i in range(1 + len(species)):  # pressure + n species
        prj.time_loop.add_process(
            process=process_name,
            nonlinear_solver_name="basic_picard",
            convergence_type="DeltaX",
            norm_type="NORM2",
            reltol="1e-6",
            time_discretization="BackwardEuler",
        )
        if time_stepping is None:
            time_stepping = [(1, end_time)]

        prj.time_loop.set_stepping(
            process=process_name,
            type="FixedTimeStepping",
            t_initial=initial_time,
            t_end=end_time,
            repeat=time_stepping[0][0],
            delta_t=time_stepping[0][1],
        )

        for time_step in time_stepping[1:]:
            prj.time_loop.add_time_stepping_pair(
                repeat=time_step[0], delta_t=time_step[1], process=process_name
            )

    return create_prj_template(
        saving_path,
        prj,
        process_name=process_name,
        dimension=dimension,
        initial_time=initial_time,
        fixed_out_times=fixed_out_times,
        end_time=end_time,
        time_stepping=time_stepping,
        error_tolerance=error_tolerance,
        output_variables=species + ["HEAD_OGS"],
    )


def hydro_thermal(
    saving_path: Path,
    prj: Project,
    dimension: int = 3,
    fixed_out_times: list | None = None,
    time_stepping: list | None = None,
    initial_time: int = 0,
    end_time: int | float = 1,
    error_tolerance: float = 1e-10,
) -> Project:
    gravity = " ".join(["0"] * dimension)
    prj.processes.set_process(
        name="HydroThermal",
        type="HT",
        integration_order=3,
        specific_body_force=gravity,
    )
    return create_prj_template(
        saving_path,
        prj,
        process_name="HydroThermal",
        dimension=dimension,
        fixed_out_times=fixed_out_times,
        time_stepping=time_stepping,
        initial_time=initial_time,
        end_time=end_time,
        error_tolerance=error_tolerance,
        eigen_sparseLu=True,
    )
