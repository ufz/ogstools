"""
Spatial refinement - steady state diffusion
===========================================

This example shows one possible implementation of how to do a convergence study.
It uses the project file from the following benchmark with multiple
discretizations to evaluate the accuracy of the numerical solutions.
`ogs: elliptic neumann benchmark
<https://www.opengeosys.org/docs/benchmarks/elliptic/elliptic-neumann/>`_

Here is some theoretical background for the topic of grid convergence:

`Nasa convergence reference
<https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html>`_

`More comprehensive reference
<https://curiosityfluids.com/2016/09/09/establishing-grid-convergence/>`_

At least three meshes of increasing refinement are required for the convergence
study. The three finest meshes are used to calculated the Richardson
extrapolation. The third coarsest mesh will be used for the topology
to evaluate the results. Its nodes should be shared by the finer meshes,
otherwise interpolation will influence the results. With
unstructured grids this can be achieved as well with refinement by splitting.

The results to analyze are generated on the fly with the following code. If you
are only interested in the convergence study, please skip to
`Hydraulic pressure convergence`_.

First, the required packages are imported and an output directory is created:
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv
from IPython.display import HTML

import ogstools as ot
from ogstools import examples, workflow
from ogstools.studies import convergence

temp_dir = Path(mkdtemp(suffix="steady_state_diffusion"))
report_name = str(temp_dir / "report.ipynb")
result_paths = []

# %% [markdown]
# The meshes and their boundaries are generated easily via gmsh and
# :py:mod:`ogstools.meshlib.gmsh_converter.meshes_from_gmsh`.
# Then we run the different simulations with increasingly fine spatial
# discretization via ogs6py and store the results for the convergence study.

# %%
refinements = 6
edge_cells = [2**i for i in range(refinements)]
for n_edge_cells in edge_cells:
    msh_path = temp_dir / "square.msh"
    ot.meshlib.rect(
        n_edge_cells=n_edge_cells, structured_grid=True, out_name=msh_path
    )
    meshes = ot.meshes_from_gmsh(filename=msh_path, log=False)
    for name, mesh in meshes.items():
        pv.save_meshio(Path(temp_dir, name + ".vtu"), mesh)

    model = ot.Project(
        output_file=temp_dir / "default.prj",
        input_file=examples.prj_steady_state_diffusion,
    )
    prefix = "steady_state_diffusion_" + str(n_edge_cells)
    model.replace_text(prefix, ".//prefix")
    model.write_input()
    ogs_args = f"-m {temp_dir} -o {temp_dir}"
    model.run_model(write_logs=False, args=ogs_args)
    result_paths += [str(temp_dir / (prefix + ".pvd"))]

# %% [markdown]
# Here we calculate the analytical solution on one of the meshes:

# %%
analytical_solution_path = temp_dir / "analytical_solution.vtu"
solution = examples.analytical_diffusion(
    ot.MeshSeries(result_paths[-1]).mesh(0)
)
ot.plot.setup.show_element_edges = True
fig = ot.plot.contourf(solution, ot.variables.hydraulic_head)
solution.save(analytical_solution_path)

# %% [markdown]
# Hydraulic pressure convergence
# ------------------------------
#
# The pressure field of this model is converging well. The convergence ratio
# is approximately 1 on the whole mesh and looking at the relative errors we
# see a quadratic convergence behavior.

# %%
convergence.run_convergence_study(
    output_name=report_name,
    mesh_paths=result_paths,
    variable_name="hydraulic_head",
    timevalue=1,
    refinement_ratio=2.0,
    reference_solution_path=str(analytical_solution_path),
)
HTML(workflow.jupyter_to_html(report_name, show_input=False))

# %% [markdown]
# Darcy velocity convergence
# --------------------------
#
# For the velocity we some discrepancy of the convergence ratio in the bottom
# right corner. Thus we know, at these points the mesh isn't properly
# converging (at least for the velocity field).
# We see, that in the bottom right corner, the velocity magnitude seems to be
# steadily increasing, which is also reflected in the Richardson extrapolation,
# which shows an anomalous high value in this spot, hinting at a singularity
# there. This is explained by the notion in the benchmark's documentation of
# "incompatible boundary conditions imposed on the bottom right corner of the
# domain." Regardless of this, the benchmark gives a convergent solution for
# the pressure field.
# The code cells from the templated notebook are show here for transparency.

# %%

convergence.run_convergence_study(
    output_name=report_name,
    mesh_paths=result_paths,
    variable_name="velocity",
    timevalue=1,
    refinement_ratio=2.0,
)
HTML(workflow.jupyter_to_html(report_name, show_input=True))
