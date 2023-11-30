def steady_state_diffusion(saving_path, model=None):
    """
    A template for steady state diffusion process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :type saving_path: str
    :param model: ogs model, which shall be used with the template
    :type model: ogs6py.ogs.OGS
    """
    model.processes.set_process(
        name="SteadyStateDiffusion",
        type="STEADY_STATE_DIFFUSION",
        integration_order="2",
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


def liquid_flow(saving_path, model=None):
    """
    A template for steady state diffusion process to be simulated in ogs.

    :param saving_path: path of ogs simulation results
    :type saving_path: str
    :param model: ogs model, which shall be used with the template
    :type model: ogs6py.ogs.OGS
    """
    model.processes.set_process(
        name="LiquidFlow",
        type="LIQUID_FLOW",
        integration_order="2",
        specific_body_force="0 0 0",
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
