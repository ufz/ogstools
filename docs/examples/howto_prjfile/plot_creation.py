"""
How to Create Simple Mechanics Problem
======================================

.. sectionauthor:: Jörg Buchwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

The following example consists of a simple mechanics problem.
The source file can be found in the [OGS benchmark folder](https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/Mechanics/Linear/square_1e2.prj?ref_type=heads).
The names of the method calls are based on the corresponding XML tags.

"""

# %%
# Initialize the ogs6py object:
from pathlib import Path
from tempfile import mkdtemp

import ogstools as ot
from ogstools.definitions import EXAMPLES_DIR

output_dir = Path(mkdtemp())
prj = ot.Project(output_file=output_dir / "mechanics_new.prj")

# %%
# Define geometry and/or meshes:
prj.geometry.add_geometry(filename="square_1x1.gml")
prj.mesh.add_mesh(filename="square_1x1_quad_1e2.vtu")

# %%
# Set process and provide process related data:
prj.processes.set_process(
    name="SD",
    type="SMALL_DEFORMATION",
    integration_order="2",
    specific_body_force="0 0",
)
prj.processes.set_constitutive_relation(
    type="LinearElasticIsotropic", youngs_modulus="E", poissons_ratio="nu"
)
prj.processes.add_process_variable(
    process_variable="process_variable", process_variable_name="displacement"
)
prj.processes.add_secondary_variable(internal_name="sigma", output_name="sigma")

# %%
# Define time stepping and output cycles:
prj.time_loop.add_process(
    process="SD",
    nonlinear_solver_name="basic_newton",
    convergence_type="DeltaX",
    norm_type="NORM2",
    abstol="1e-15",
    time_discretization="BackwardEuler",
)
prj.time_loop.set_stepping(
    process="SD",
    type="FixedTimeStepping",
    t_initial="0",
    t_end="1",
    repeat="4",
    delta_t="0.25",
)
prj.time_loop.add_output(
    type="VTK",
    prefix="blubb",
    repeat="1",
    each_steps="10",
    variables=["displacement", "sigma"],
)

prj.media.add_property(
    medium_id="0",
    phase_type="Solid",
    name="density",
    type="Constant",
    value="1",
)

# %%
# Define parameters needed for material properties and BC:
prj.parameters.add_parameter(name="E", type="Constant", value="1")
prj.parameters.add_parameter(name="nu", type="Constant", value="0.3")
prj.parameters.add_parameter(name="rho_sr", type="Constant", value="1")
prj.parameters.add_parameter(
    name="displacement0", type="Constant", values="0 0"
)
prj.parameters.add_parameter(name="dirichlet0", type="Constant", value="0")
prj.parameters.add_parameter(name="dirichlet1", type="Constant", value="0.05")

# %%
# Set initial and boundary conditions:
prj.process_variables.set_ic(
    process_variable_name="displacement",
    components="2",
    order="1",
    initial_condition="displacement0",
)
prj.process_variables.add_bc(
    process_variable_name="displacement",
    geometrical_set="square_1x1_geometry",
    geometry="left",
    type="Dirichlet",
    component="0",
    parameter="dirichlet0",
)
prj.process_variables.add_bc(
    process_variable_name="displacement",
    geometrical_set="square_1x1_geometry",
    geometry="bottom",
    type="Dirichlet",
    component="1",
    parameter="dirichlet0",
)
prj.process_variables.add_bc(
    process_variable_name="displacement",
    geometrical_set="square_1x1_geometry",
    geometry="top",
    type="Dirichlet",
    component="1",
    parameter="dirichlet1",
)

# %%
# Set linear and nonlinear solver(s):
prj.nonlinear_solvers.add_non_lin_solver(
    name="basic_newton",
    type="Newton",
    max_iter="4",
    linear_solver="general_linear_solver",
)
prj.linear_solvers.add_lin_solver(
    name="general_linear_solver",
    kind="lis",
    solver_type="cg",
    precon_type="jacobi",
    max_iteration_step="10000",
    error_tolerance="1e-16",
)
prj.linear_solvers.add_lin_solver(
    name="general_linear_solver",
    kind="eigen",
    solver_type="CG",
    precon_type="DIAGONAL",
    max_iteration_step="10000",
    error_tolerance="1e-16",
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

# %%
# Write project file to disc:
prj.write_input()

# %%
# Execute file and pipe output to logfile out.log:
prj.run_model(
    logfile="out.log",
    args="-m " + str(EXAMPLES_DIR / "prj") + " -o " + str(output_dir),
)

# %% [markdown]
# If the desired OGS version is not in PATH, a separate path containing the OGS binary can be specified.
# `model.run_model(path="~/github/ogs/build_mkl/bin")`
