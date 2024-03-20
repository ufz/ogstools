"""
Introduction
================================

This basic example shows a how to analyse the OGS log output to get information
about performance of different parts of ogs.
It uses the project file from the following benchmark:
ogs: Constant viscosity (Hydro-Thermal) https://www.opengeosys.org/docs/benchmarks/hydro-thermal/constant-viscosity with
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
# The log file
# -------------
# `log` is a str representing the location of the ogs log file.
# Make sure the log file does not contain ANSI escape (e.g.color) code. https://en.wikipedia.org/wiki/ANSI_escape_code
# Only if: You can remove it: ``cat ogs.log | sed 's/\x1b\[[0-9;]*m//g' > ogs.log```
log = const_viscosity_thermal_convection_log

# %%
# Parsing steps
# ----------------------------
# The functions :py:mod:`ogstools.logparser.parse_file`  and :py:mod:`ogstools.logparser.fill_ogs_context` are explained in :ref:`sphx_glr_auto_examples_howto_logparser_plot_logparser_advanced.py`.
# All predefined analyses need the result of fill_ogs_context.
records = parse_file(log)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)

# %%
# Use predefined analyses
# ----------------------------------------------------------------
# :py:mod:`ogstools.logparser.time_step_vs_iterations` is one of many predined analyses. All possibilities are shown here:
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_logparser_analyses.py`.
#
# Here we are interested in every time step of the simulation and how many iterations have been needed.
# For analysis runs only with log of log-level `ogs -l info` or `ogs - l debug` according to
# (see: https://www.opengeosys.org/docs/devguide/advanced/log-and-debug-output)

df_ts_it = time_step_vs_iterations(df_log)
# The result is a pandas.DataFrame. You may manipulate the dataframe to your needs with pandas functionality.
df_ts_it  # noqa: B018

# %%
# Pandas to plot
# -------------------
# You can directly use pandas plot https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html function from the resulting DataFrame.
df_ts_it.plot(grid=True)
