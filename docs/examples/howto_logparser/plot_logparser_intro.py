"""
Introduction
============

This simple example demonstrates how to analyze the OGS (OpenGeoSys) log output to
extract performance information regarding various computational parts of OGS.
Here we utilize the project file from the benchmark titled:
`OGS: Constant viscosity (Hydro-Thermal)
<https://www.opengeosys.org/docs/benchmarks/hydro-thermal/constant-viscosity>`_



"""

# %%
# Complete example
# ================
# For detailed explanation see all sections below.
import pandas as pd

from ogstools.examples import (
    log_const_viscosity_thermal_convection,
)
from ogstools.logparser import (
    fill_ogs_context,
    parse_file,
    time_step_vs_iterations,
)

records = parse_file(log_const_viscosity_thermal_convection)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)
df_ts_it = time_step_vs_iterations(df_log)
df_ts_it

# %%


# %% [markdown]
# The log file
# -------------
# Running `ogs` in the command line outputs the logs into the console output. With
# `tee
# <https://en.wikipedia.org/wiki/Tee_(command)>`_ in Linux and Mac
# and
# `Tee-Object
# <https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/tee-object?view=powershell-7.4>`_
# in Windows Powershell the log gets directed into a file and into the console.
#
# * **Linux/Mac**: ``ogs <your_prj_file> | tee <your_log_file>``
# * **Windows**: ``ogs <your_prj_file> | Tee-Object -FilePath <your_log_file>``
#
# For first step we recommend to use either **`ogs -l info`** or `ogs - l debug`.
# Make sure the log file does not contain
# `ANSI escape (e.g.color) code
# <https://en.wikipedia.org/wiki/ANSI_escape_code>`_.
# You can remove it with: ``cat ogs.log | sed 's/\x1b\[[0-9;]*m//g' > ogs.log``


# %%
# Parsing steps
# -------------
# The functions :py:mod:`ogstools.logparser.parse_file` and
# :py:mod:`ogstools.logparser.fill_ogs_context` are explained in
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_logparser_advanced.py`.
# All predefined analyses need the result of fill_ogs_context.
# Here `const_viscosity_thermal_convection_log` is string representing the
# location of the ogs log file.
print(log_const_viscosity_thermal_convection)
# %%
records = parse_file(log_const_viscosity_thermal_convection)
df_records = pd.DataFrame(records)
df_log = fill_ogs_context(df_records)

# %%
# Use predefined analyses
# -----------------------
# :py:mod:`ogstools.logparser.time_step_vs_iterations` is one of many predefined
# analyses. All possibilities are shown here:
# :ref:`sphx_glr_auto_examples_howto_logparser_plot_logparser_analyses.py`.
#
# Here we are interested in every time step of the simulation and how many
# iterations have been needed.
# The predefined analyses only work with logs from `ogs` run with level `info` or finer (`debug`), like `ogs -l info` or `ogs - l debug`.
# (see
# `OGS Developer Guide - log and debug output
# <https://www.opengeosys.org/docs/devguide/advanced/log-and-debug-output>`_

df_ts_it = time_step_vs_iterations(df_log)
# The result is a pandas.DataFrame. You may manipulate the dataframe to your
# needs with pandas functionality.
pd.set_option("display.max_rows", 8)  # for visualization only
df_ts_it

# %%
# Pandas to plot
# --------------
# You can directly use
# `plot` `
# <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html>`_ from pandas.
df_ts_it.plot(grid=True)
