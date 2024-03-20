"""
Log parser - Introduction
================================

This basic example shows a how to analyse the OGS log output to get information
about performance of different parts of ogs.
It uses the project file from the following benchmark:
`ogs: Constant viscosity (Hydro-Thermal)
<https://www.opengeosys.org/docs/benchmarks/hydro-thermal/constant-viscosity/>` with
`<t_end> 1e8 </t_end>`
"""

# %%
import pandas as pd

from ogstools.logparser import (
    fill_ogs_context,
    parse_file,
    time_step_vs_iterations,
)
from ogstools.logparser.examples import (
    const_viscosity_thermal_convection_log,
)

# %%
log = const_viscosity_thermal_convection_log
# Purpose of records and fill_ogs_context is explained in advanced section
records = parse_file(log)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)
# This is one of many predined analyses. All possibilities are show here:
# Here we are interested in every time step of the simulation and how many iterations have been needed.
# For analysis runs only with log of log-level `ogs -l info` or `ogs - l debug` according to
# `OpenGeoSys Docs: Log and Debug Output <https://www.opengeosys.org/docs/devguide/advanced/log-and-debug-output/>`

df_ts_it = time_step_vs_iterations(df_log)
# The result is a pandas.DataFrame. You may manipulate the dataframe to your needs with pandas functionality.
df_ts_it  # noqa: B018

# %%
# Or directly use pandas functionality to plot.
df_ts_it.plot(grid=True)
