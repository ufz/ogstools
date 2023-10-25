"""
Convergence study
=================

This script performs a convergence study and generates plots to analyze the
convergence of numerical simulations. It uses data from the following benchmark
with multiple discretizations to evaluate the accuracy of the numerical
solutions.
https://www.opengeosys.org/docs/benchmarks/elliptic/elliptic-neumann/

Here is some theoretical background for the Richardson extrapolation:
https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html
"""

# %%
# First import the dependencies and adjust some plot settings.

import numpy as np

import ogstools.meshplotlib as mpl
from ogstools.propertylib import ScalarProperty, VectorProperty
from ogstools.studies.convergence import (
    convergence_metrics,
    examples,
    grid_convergence,
    plot_convergence,
    plot_convergence_errors,
    richardson_extrapolation,
)

mpl.setup.reset()
mpl.setup.show_element_edges = True
mpl.setup.ax_aspect_ratio = 1

# %%
# Let's have a look at the different discretizations. The 3 finest will be used
# for the Richadson extrapolation. The coarsest of those will be used for the
# topology to evaluate the results. We inspect the primary variable,
# in this case the hydraulic head.

mesh_property = ScalarProperty(
    data_name="pressure",
    data_unit="m",
    output_unit="m",
    output_name="hydraulic head",
)
fig = mpl.plot(np.reshape(examples.meshes, (2, 3)), mesh_property)

# %%
# Now we calculate the convergence ratio on the whole mesh. Values close to
# 1 indicate that we are in asymptotic range of convergence. We see, that in the
# bottom right corner, there is some small discrepancy, which is explained in
# the benchmark documentation by "incompatible boundary conditions imposed on
# the bottom right corner of the domain." In the end we will see the
# implications for the secondary variable, the velocity.

topology = examples.meshes[-3]
conv_results = grid_convergence(examples.meshes, mesh_property, topology)
fig = mpl.plot(conv_results, "grid_convergence")

# %%
# The Richardson extrapolation can be easily calculated. Again, it uses the 3
# finest meshes from the given list of meshes.

richardson = richardson_extrapolation(examples.meshes, mesh_property, topology)
analytical = examples.analytical_solution(topology)
fig = mpl.plot([richardson, analytical], mesh_property)
fig.axes[0].set_title("Richardson extrapolation")
fig.axes[1].set_title("Analytical Solution")
fig.show()

# %%
# Now we can compute some convergence metrics and display them in a table, ...

metrics = convergence_metrics(examples.meshes, richardson, mesh_property)
metrics.style.format("{:,.5g}").hide()

# %%
# ... plot the converging values in absolute scale ...

mpl.core.plt.rcdefaults()
fig = plot_convergence(metrics, mesh_property)

# %%
# ... and the relative errors in loglog-scale. Note: since the Minimum doesn't
# change in the different discretizations, the error is zero, thus there is no
# curve for it in this plot.

fig = plot_convergence_errors(metrics)


# %%
# Now let's inspect the velocity field. We see, that in the bottom right corner,
# the velocity magnitude seems to be steadily increasing.

mesh_property = VectorProperty("v", "m/s", "m/s", "velocity")
mpl.setup.num_streamline_interp_pts = None
fig = mpl.plot(np.reshape(examples.meshes, (2, 3)), mesh_property)

# %%
# Looking at the grid convergence, we see the bottom left and right corners
# deviating quite a bit from the desired value of 1. Thus we know, at these
# points the mesh isn't properly converging (at least for the velocity field).

conv_results = grid_convergence(examples.meshes, mesh_property, topology)
fig = mpl.plot(conv_results, "grid_convergence")

# %%
# The Richardson extrapolation shows an anomalous value for the velocity in
# the bottom right corner, hinting at a singularity there which is caused by
# the incompatibility of the boundary conditions on the bottom and right in
# this singular point. Regardsless of this, the benchmark gives a convergent
# solution for the pressure field, but not for the velocity field.
richardson = richardson_extrapolation(examples.meshes, mesh_property, topology)
fig = mpl.plot(richardson, mesh_property)
