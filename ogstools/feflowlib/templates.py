def steady_state_diffusion(saving_path, model=None):
    """
    A template for steady state diffusion process to be simulated in ogs.
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
        type="VTK", prefix=saving_path, repeat="1", each_steps="1", variables=[]
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
    return model
