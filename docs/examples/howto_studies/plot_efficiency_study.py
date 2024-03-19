"""
Efficiency study
=================

This example shows how to analyse the  OGS log output to get information
about performance of different parts of ogs.
It uses the project file from the following benchmark:
`ogs: LiquidFlow (Primary variable constraint Dirichlet-type boundary condition)
<https://www.opengeosys.org/docs/benchmarks/liquid-flow/primary-variable-constrain-dirichlet-boundary-condition/>`_
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
df_ts_it = time_step_vs_iterations(df_log)
df_ts_it  # noqa: B018

# %%
df_ts_it.plot(grid=True)


# %%
# Performance of in separate parts by time step
df_ts = analysis_time_step(df_log)
df_ts = df_ts.loc[
    0
]  # log of serial so we can remove MPI_process (index=0) from result (all are 0)
df_ts  # noqa: B018
# %%
# Data manipulation with pandas
df_ts[
    ["output_time", "assembly_time", "dirichlet_time", "linear_solver_time"]
].plot(logy=True, grid=True)

# %%

analysis_convergence_newton_iteration(df_log)


# %%
# Staggered

records = parse_file(
    "/home/meisel/gitlabrepos/ogstools/staggered_heat_transport_in_stationary_flow.log"
)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)
analysis_convergence_coupling_iteration(df_log)


# %%
# Parallel
df_ts.loc[0]
# %%
# Advanced

# %%
## Custom
df_records  # noqa: B018
# %%
x = df_records.pivot_table(["step_size", "iteration_number"], ["time_step"])
x.plot(subplots=True, sharex=True, grid=True)

# Computing logs takes to much time
# custom regexes, log level

# force_parallel
