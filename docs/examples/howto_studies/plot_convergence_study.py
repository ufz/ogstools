"""
Convergence study
=================

This script performs a convergence study and generates plots to analyze the
convergence of numerical simulations. It uses data from the following benchmark
with multiple discretizations to evaluate the accuracy of the numerical
solutions.
https://www.opengeosys.org/docs/benchmarks/elliptic/elliptic-neumann/
"""

# %%
# First import the dependencies.

import matplotlib.pyplot as plt
import pandas as pd

from ogstools.propertylib.defaults import pressure, velocity
from ogstools.studies.convergence import (
    convergence,
    plot_convergence,
    richardson_extrapolation,
)
from ogstools.studies.examples import analytical_solution, meshes

# %%
# Now we define the topology on which the comparison is made (target_mesh).
# In this benchmark we have the advantage of an existing analytical solution
# but even without we can calculate the Richardson extrapolation from the two
# finest meshes to create a reference results to compare against.

target_mesh = meshes[0]
velocity.data_name = "v"
data = [pressure, velocity]
richardson = richardson_extrapolation(meshes[-2], meshes[-1], data)
analytical = analytical_solution(target_mesh)

# %%
# We can run the convergence study and visualize it in a tabular format by
# converting the resulting dictionary to a pandas dataframe.
conv = convergence(target_mesh, meshes, analytical, data)
pd.DataFrame(conv)

# %%
# Finally, let's visualize the convergence by plotting the error (L2-Norm)
# of the pressure field between the different discretizations against the
# analytical solution...
fig, axs = plt.subplots(dpi=200, figsize=[5, 3], facecolor="white", nrows=1)
plot_convergence(target_mesh, meshes, analytical, pressure, axs)
_ = axs.set_title("convergence against the analytical solution")

# %%
# and against the Richardson extrapolation.
fig, axs = plt.subplots(dpi=200, figsize=[5, 3], facecolor="white", nrows=1)
plot_convergence(target_mesh, meshes, richardson, pressure, axs)
_ = axs.set_title("convergence against the Richardson extrapolation")

# %%
# Out of curiosity let's plot this for the velocity field as well.
# Apparently derived properties don't show the same convergence behaviour as
# the main process variables.
fig, axs = plt.subplots(dpi=200, figsize=[5, 3], facecolor="white", nrows=1)
plot_convergence(target_mesh, meshes, richardson, velocity, axs)
_ = axs.set_title("convergence against the Richardson extrapolation")
