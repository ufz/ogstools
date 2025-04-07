"""
Logparser: Predefined Analyses
==============================

In this section, we showcase various predefined analyses available in the log parser.
We utilize project files from the following benchmarks:
`ogs: Constant viscosity (Hydro-Thermal)
<https://www.opengeosys.org/docs/benchmarks/hydro-thermal/constant-viscosity>`_
and for the **staggered scheme** we use a prj from
`ogs tests: HT StaggeredCoupling HeatTransportInStationaryFlow
<https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/Parabolic/HT/StaggeredCoupling/HeatTransportInStationaryFlow/HeatTransportInStationaryFlow.prj>`_

"""

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = -3

# sphinx_gallery_end_ignore

# %%
import pandas as pd

import ogstools as ot
from ogstools.examples import (
    log_adaptive_timestepping,
    log_const_viscosity_thermal_convection,
    log_staggered,
)

pd.set_option("display.max_rows", 8)  # for visualization only

# %%
# The preprocessing of logs remains consistent across all examples and
# thoroughly explained in
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_102_logparser_advanced.py`.

records = ot.logparser.parse_file(log_const_viscosity_thermal_convection)
df_records = pd.DataFrame(records)
df_log = ot.logparser.fill_ogs_context(df_records)

# %%
# Analysis of iterations per time step
# ------------------------------------
# For detailed explanation, refer to:
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_100_logparser_intro.py`.
# (Section: Use predefined analyses)
#
# :py:mod:`ogstools.logparser.analysis_time_step`
df_ts_it = ot.logparser.time_step_vs_iterations(df_log)
df_ts_it


# %%
# Analysis of computational efficiency by time step
# -------------------------------------------------
# The resulting table represents performance metrics for different parts of the simulation,
# organized by time step. It utilizes :py:mod:`ogstools.logparser.analysis_time_step`.
# displaying metrics such
# as output time [s], step size [s], time step solution time [s], assembly time [s],
# Dirichlet time [s], and linear solver time [s].

df_ts = ot.logparser.analysis_time_step(df_log)
df_ts = df_ts.loc[0]
# Removing MPI_process (index=0) from result (all are 0) for serial log.
df_ts

# %%
# Selecting specific metrics (3) and plotting using pandas plot function.
df_ts[["assembly_time", "dirichlet_time", "linear_solver_time"]].plot(
    logy=True, grid=True
)

# %% [markdown]
# Analysis of convergence criteria - Newton iterations
# ----------------------------------------------------
# The :py:mod:`ogstools.logparser.analysis_convergence_newton_iteration`
# function allows for the examination of convergence criteria based on
# Newton iterations. The resulting table provides convergence metrics for monolithic processes.
# For details, refer to the documentation on
# <`convergence_criterion
# <https://doxygen.opengeosys.org/d4/d58/classnumlib_1_1convergencecriterion>`_ > defined in in the prj file.
#
# * **|x|** is a norm of a vector of the global component (e.g. pressure, temperature, displacement).
# * **|dx|** is the change of a norm of the global component between 2 iteration of non linear solver.
# * **|dx|/|x|** is the relative change of a norm of the global component
#
# For this example we had defined in the prj-file:
#
# .. code-block:: python
#
#    <convergence_criterion>
#      <type>DeltaX</type>
#      <norm_type>NORM2</norm_type>
#      <abstol>1.e-3</abstol>
#    </convergence_criterion>
#
# The resulting table contains `|x|`, `|dx|` and `|dx|/|x|` at different time steps, processes and non linear solver iterations.


# %%

ot.logparser.analysis_convergence_newton_iteration(df_log)


# %%
# Staggered
# ~~~~~~~~~
# The resulting table provides convergence criteria for staggered coupled processes,
# utilizing :py:mod:`ogstools.logparser.analysis_convergence_coupling_iteration`
# Logs are generated from running
# `ogs benchmark: HeatTransportInStationaryFlow
# <https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/Parabolic/HT/HeatTransportInStationaryFlow/HeatTransportInStationaryFlow.prj>`_
#
records = ot.logparser.parse_file(log_staggered)
df_records = pd.DataFrame(records)
df_log = ot.logparser.fill_ogs_context(df_records)

# Only for staggered coupled processes !
ot.logparser.analysis_convergence_coupling_iteration(df_log)

# %% [markdown]
# Analysis of model time and clock time
# -------------------------------------
# The :py:mod:`ogstools.logparser.model_and_clock_time` function allows to
# examine needed iterations, clock time, and step size over model time per
# attempted time step. This is especially useful to analyze the runtime
# behaviour of a simulation which employs adaptive time stepping. The following
# example failed as the simulation reached the minimal allowed time step size.

# %%
records = ot.logparser.parse_file(log_adaptive_timestepping)
df_records = pd.DataFrame(records)
df_log = ot.logparser.fill_ogs_context(df_records)
df_t = ot.logparser.model_and_clock_time(df_log)
df_t[["step_size", "clock_time", "iterations"]].plot(grid=True, subplots=True)

# %% [markdown]
# To get an overview of the convergence behavior of the nonlinear solver over
# the entire simulation we can plot the relative error as a heatmap.

# %%
fig = ot.logparser.plot_convergence(df_log, "dx_x")

# %% [markdown]
# We can also plot this data over the model time. Note, that the last timesteps
# are so small, that they are not visible anymore here:

# %%
fig = ot.logparser.plot_convergence(df_log, "dx_x", x_metric="model_time")

# %% [markdown]
# Further we can calculate the convergence order and plot it in the same manner.
# In order to estimate the convergence order we need to take into account
# multiple values and thus cannot assign each iteration a convergence order.
# Only for iterations `i` of `i  >= n` an order is calculated and plotted.
# See: :py:func:`~ogstools.logparser.common_ogs_analyses.convergence_order_per_ts_iteration`
# for more info.

# %%
fig = ot.logparser.plot_convergence_order(df_log)
