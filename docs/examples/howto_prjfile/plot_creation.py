"""
How to Create Simple Mechanics Problem
======================================

.. sectionauthor:: Jörg Buchwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

The following example consists of a simple mechanics problem.
The source file can be found in the [OGS benchmark folder](https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/Mechanics/Linear/square_1e2.prj?ref_type=heads).
The names of the method calls are based on the corresponding XML tags.

"""

# %%
# 1. Initialize the ogs6py object:

from ogstools.definitions import EXAMPLES_DIR
from ogstools.ogs6py import ogs

model = ogs.OGS(PROJECT_FILE=EXAMPLES_DIR / "prj/simple_mechanics.prj")

# %%
# 2. Define geometry and/or meshes:
model.geometry.add_geometry(filename="square_1x1.gml")
model.mesh.add_mesh(filename="square_1x1_quad_1e2.vtu")

# %%
# 3. Set process and provide process related data:
model.processes.set_process(
    name="SD",
    type="SMALL_DEFORMATION",
    integration_order="2",
    specific_body_force="0 0",
)
model.processes.set_constitutive_relation(
    type="LinearElasticIsotropic", youngs_modulus="E", poissons_ratio="nu"
)
model.processes.add_process_variable(
    process_variable="process_variable", process_variable_name="displacement"
)
model.processes.add_secondary_variable(
    internal_name="sigma", output_name="sigma"
)

# %%
# 4. Define time stepping and output cycles:
model.time_loop.add_process(
    process="SD",
    nonlinear_solver_name="basic_newton",
    convergence_type="DeltaX",
    norm_type="NORM2",
    abstol="1e-15",
    time_discretization="BackwardEuler",
)
model.time_loop.set_stepping(
    process="SD",
    type="FixedTimeStepping",
    t_initial="0",
    t_end="1",
    repeat="4",
    delta_t="0.25",
)
model.time_loop.add_output(
    type="VTK",
    prefix="blubb",
    repeat="1",
    each_steps="10",
    variables=["displacement", "sigma"],
)

model.media.add_property(
    medium_id="0",
    phase_type="Solid",
    name="density",
    type="Constant",
    value="1",
)

# %%
# 5. Define parameters needed for material properties and BC:
model.parameters.add_parameter(name="E", type="Constant", value="1")
model.parameters.add_parameter(name="nu", type="Constant", value="0.3")
model.parameters.add_parameter(name="rho_sr", type="Constant", value="1")
model.parameters.add_parameter(
    name="displacement0", type="Constant", values="0 0"
)
model.parameters.add_parameter(name="dirichlet0", type="Constant", value="0")
model.parameters.add_parameter(name="dirichlet1", type="Constant", value="0.05")

# %%
# 6. Set initial and boundary conditions:
model.process_variables.set_ic(
    process_variable_name="displacement",
    components="2",
    order="1",
    initial_condition="displacement0",
)
model.process_variables.add_bc(
    process_variable_name="displacement",
    geometrical_set="square_1x1_geometry",
    geometry="left",
    type="Dirichlet",
    component="0",
    parameter="dirichlet0",
)
model.process_variables.add_bc(
    process_variable_name="displacement",
    geometrical_set="square_1x1_geometry",
    geometry="bottom",
    type="Dirichlet",
    component="1",
    parameter="dirichlet0",
)
model.process_variables.add_bc(
    process_variable_name="displacement",
    geometrical_set="square_1x1_geometry",
    geometry="top",
    type="Dirichlet",
    component="1",
    parameter="dirichlet1",
)

# %%
# 7. Set linear and nonlinear solver(s):
model.nonlinear_solvers.add_non_lin_solver(
    name="basic_newton",
    type="Newton",
    max_iter="4",
    linear_solver="general_linear_solver",
)
model.linear_solvers.add_lin_solver(
    name="general_linear_solver",
    kind="lis",
    solver_type="cg",
    precon_type="jacobi",
    max_iteration_step="10000",
    error_tolerance="1e-16",
)
model.linear_solvers.add_lin_solver(
    name="general_linear_solver",
    kind="eigen",
    solver_type="CG",
    precon_type="DIAGONAL",
    max_iteration_step="10000",
    error_tolerance="1e-16",
)
model.linear_solvers.add_lin_solver(
    name="general_linear_solver",
    kind="petsc",
    prefix="sd",
    solver_type="cg",
    precon_type="bjacobi",
    max_iteration_step="10000",
    error_tolerance="1e-16",
)

# %%
# 7. Write project file to disc:
model.write_input()

# %%
# 8. Execute file and pipe output to logfile out.log:
model.run_model(logfile="out.log")

# %% [markdown]
# 9. If the desired OGS version is not in PATH, a separate path containing the OGS binary can be specified.
# `model.run_model(path="~/github/ogs/build_mkl/bin")`
