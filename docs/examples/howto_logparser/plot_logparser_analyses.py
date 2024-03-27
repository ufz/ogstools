"""
Predefined Analyses
===================

Here we show the different predefined analysis available in the logparser.
We use the project file from the following benchmark:
ogs: Constant viscosity (Hydro-Thermal)
<https://www.opengeosys.org/docs/benchmarks/hydro-thermal/constant-viscosity/>

with `<t_end> 1e8 </t_end>`

and for the **staggered scheme** the variant taken from
Tests/Data/Parabolic/HT/StaggeredCoupling/HeatTransportInStationaryFlow/HeatTransportInStationaryFlow.prj

"""

# %%
import pandas as pd

from ogstools.logparser import (
    analysis_convergence_coupling_iteration,
    analysis_convergence_newton_iteration,
    analysis_time_step,
    fill_ogs_context,
    parse_file,
    time_step_vs_iterations,
)
from ogstools.logparser.examples import (
    const_viscosity_thermal_convection_log,
    staggered_log,
)

# %%
# The log preprocessing is same for all examples and explained in
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_logparser_advanced.py`.

log = const_viscosity_thermal_convection_log
records = parse_file(log)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)

# %%
# Analysis of iterations per time step
# ------------------------------------
# Please see explanation in logparser
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_logparser_intro.py`.
# (Section: Use predefined analyses)
#
# :py:mod:`ogstools.logparser.analysis_time_step`
df_ts_it = time_step_vs_iterations(df_log)
df_ts_it  # noqa: B018


# %%
# Analysis of computational efficiency by time step
# -------------------------------------------------
# The resulting table presents the performance metrics for separate parts of the simulation,
# organized by time step. Is uses :py:mod:`ogstools.logparser.analysis_time_step`.
# Each row corresponds to a specific time step, displaying metrics such
# as output time [s], step size [s], time step solution time [s], assembly time [s],
# Dirichlet time [s], and linear solver time [s].

df_ts = analysis_time_step(df_log)
df_ts = df_ts.loc[0]
# log of serial so we can remove MPI_process (index=0) from result (all are 0)
# - see advanced
df_ts  # noqa: B018
# %%
# We select only some metrics (3) and use pandas plot function.
df_ts[["assembly_time", "dirichlet_time", "linear_solver_time"]].plot(
    logy=True, grid=True
)

# %%
# Analysis of convergence criteria - Newton iterations
# ----------------------------------------------------
# The :py:mod:`ogstools.logparser.analysis_convergence_newton_iteration`
# function allows for the analysis of convergence criteria based on
# Newton iterations. The resulting table provides convergence criteria for monolithic processes.
# Each row represents convergence metrics such as `global component norm |x|`, `change of global component norm |dx|` (change between 2 iteration of non linear solver)
# and `relative change of global component |dx|/|x|` at different time steps, processes and non linear solver iterations.
analysis_convergence_newton_iteration(df_log)


# %%
# Staggered - Analysis of convergence criteria - Newton iterations
# ----------------------------------------------------------------
# The resulting table provides convergence criteria for staggered coupled processes,
# Each row represents convergence metrics such as `global component norm |x|`, `change of global component norm |dx|` (change between 2 iteration of non linear solver)
# and `relative change of global component |dx|/|x|` at different time steps and coupling
# iterations

# :py:mod:`ogstools.logparser.analysis_convergence_coupling_iteration`
# We use the logs generated when running
# https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/Parabolic/HT/HeatTransportInStationaryFlow/HeatTransportInStationaryFlow.prj
#
log = staggered_log
records = parse_file(log)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)

# Only for staggered coupled processes !
analysis_convergence_coupling_iteration(df_log)
