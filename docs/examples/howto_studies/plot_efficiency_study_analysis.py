"""
Efficiency study - Predefined Analyses
=======================================

Here we shows the different predefined analysis available in the log parser.
We uses the project file from the following benchmark:
`ogs: Constant viscosity (Hydro-Thermal)
<https://www.opengeosys.org/docs/benchmarks/hydro-thermal/constant-viscosity/>` with
`<t_end> 1e8 </t_end>`
and for the staggered scheme the variant taken from
`Tests/Data/Parabolic/HT/StaggeredCoupling/HeatTransportInStationaryFlow/HeatTransportInStationaryFlow.prj`
"""

# %%
import pandas as pd

from ogstools.studies.efficiency import (
    analysis_convergence_coupling_iteration,
    analysis_convergence_newton_iteration,
    analysis_time_step,
    fill_ogs_context,
    parse_file,
    time_step_vs_iterations,
)
from ogstools.studies.efficiency.examples import (
    const_viscosity_thermal_convection_log,
)

# %%
log = const_viscosity_thermal_convection_log
records = parse_file(log)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)

# %%
# Every time step of the simulation and how many iterations have been needed.
df_ts_it = time_step_vs_iterations(df_log)
df_ts_it  # noqa: B018


# %%
# Performance of in separate parts by time step
df_ts = analysis_time_step(df_log)
df_ts = df_ts.loc[0]
# log of serial so we can remove MPI_process (index=0) from result (all are 0) - see advanced
df_ts  # noqa: B018
# %%
# Performance of in separate parts by time step - plot
df_ts[
    ["output_time", "assembly_time", "dirichlet_time", "linear_solver_time"]
].plot(logy=True, grid=True)

# %%
# Analysis of convergence criteria - Newton iterations
analysis_convergence_newton_iteration(df_log)


# %%
# Staggered
# Tests/Data/Parabolic/HT/StaggeredCoupling/HeatTransportInStationaryFlow/HeatTransportInStationaryFlow.prj#
#
records = parse_file(
    "/home/meisel/gitlabrepos/ogstools/staggered_heat_transport_in_stationary_flow.log"
)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)

# Only for staggered coupled processes
analysis_convergence_coupling_iteration(df_log)
