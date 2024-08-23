# ogs6py

## Overview

ogs6py is a python-API for the OpenGeoSys finite element software.
Its main functionalities include creating and altering OGS6 input files as well as executing OGS.
The package allows to streamline OGS-workflows with python or Julia entirely in jupyter or pluto notebooks as demonstrated in the following video:

[![video presentation of ogs6py and vtu interface](https://img.youtube.com/vi/eihNKjK-I-s/0.jpg)](https://www.youtube.com/watch?v=eihNKjK-I-s)

# Features

- alternate existing files
- create new input files from scratch
- execute project files
- tailored alteration of input files e.g. for mesh replacements or restarts
- display and export parameter settings

## Creating a new input file

The following example consists of a simple mechanics problem. The source file can be found in the examples directory The names of the method calls are based on the corresponding XML tags.

```python
from ogs6py import ogs

model = ogs.OGS(PROJECT_FILE="simple_mechanics.prj")
model.geometry.add_geometry(filename="square_1x1.gml")
model.mesh.add_mesh(filename="square_1x1_quad_1e2.vtu")
model.processes.set_process(
    name="SD",
    type="SMALL_DEFORMATION",
    integration_order="2",
    solid_density="rho_sr",
    specific_body_force="0 0",
)
model.processes.set_constitutive_relation(
    type="LinearElasticIsotropic", youngs_modulus="E", poissons_ratio="nu"
)
model.processes.add_process_variable(
    process_variable="process_variable", process_variable_name="displacement"
)
model.processes.add_secondary_variable(internal_name="sigma", output_name="sigma")
model.timeloop.add_process(
    process="SD",
    nonlinear_solver_name="basic_newton",
    convergence_type="DeltaX",
    norm_type="NORM2",
    abstol="1e-15",
    time_discretization="BackwardEuler",
)
model.timeloop.set_stepping(
    process="SD",
    type="FixedTimeStepping",
    t_initial="0",
    t_end="1",
    repeat="4",
    delta_t="0.25",
)
model.timeloop.add_output(
    type="VTK",
    prefix="blubb",
    repeat="1",
    each_steps="10",
    variables=["displacement", "sigma"],
)
model.parameters.add_parameter(name="E", type="Constant", value="1")
model.parameters.add_parameter(name="nu", type="Constant", value="0.3")
model.parameters.add_parameter(name="rho_sr", type="Constant", value="1")
model.parameters.add_parameter(name="displacement0", type="Constant", values="0 0")
model.parameters.add_parameter(name="dirichlet0", type="Constant", value="0")
model.parameters.add_parameter(name="dirichlet1", type="Constant", value="0.05")
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
model.write_input()
model.run_model(logfile="out.log")
```

```python
model.runModel(path="~/github/ogs/build_mkl/bin")
```

An example using the MPL can be find in example_THM.py.

## Alternatively it is possible to alter existing files using the available replace methods:

E.g., to iterate over three Young's moduli one can use the replace parameter method:

```python
Es = [1, 2, 3]
filename = "simple_mechanics.prj"
for E in Es:
    model = OGS(INPUT_FILE=filename, PROJECT_FILE=filename)
    model.replace_parameter(name="E", value=E)
    model.replace_text("out_E=" + str(E), xpath="./time_loop/output/prefix")
    model.write_input()
    model.run_model(path="~/github/ogs/build_mkl/bin")
```

Instead of the `replace_parameter` method, the more general `replace_text` method can also be used to replace the young modulus in this example:

```python
model.replace_text(E, xpath="./parameters/parameter[name='E']/value")
```

The Young's modulus in this file can also be accessed through 0'th occurrence of the place addressed by the xpath `./parameters/parameter/value`

```python
model.replace_text(E, xpath="./parameters/parameter/value", occurrence=0)
```

For MPL based processes, there exist specific functions to set phase and medium properties: e.g.,

```python
model.replace_phase_property(
    mediumid=0, phase="Solid", name="thermal_expansivity", value="42"
)
```

for a phse property and

```python
model.replace_medium_property(mediumid=0, name="porosity", value="0.24")
```

for a property that lives on the medium level.

## FAQ/Troubleshooting

- _OGS execution fails and nothing is written to the logfile._ Please check whether OGS is executed correctly from the terminal. A common issue is related to the fact that the interactive python environment differs from the environment in the terminal. Usually, this can be fixed by setting the required environment variables via a wrapper command. E.g., `model.run_model(wrapper="source /opt/intel/oneapi/setvars.sh intel64 &&")`.
