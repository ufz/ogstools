"""
Advanced topics
==================================

We cover:

1. Logs from parallel computation (OGS with MPI runs)

2. Performance tuning

3. Custom analyses

Although these topics are presented together they do not depend on each other and can be used separately.

"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from ogstools.logparser import (
    analysis_time_step,
    fill_ogs_context,
    parse_file,
)
from ogstools.logparser.examples import (
    const_viscosity_thermal_convection_log,
    parallel_log,
)

# %%
# 1. Logs from parallel computations (with MPI)
# ----------------------------------------------------------------
# The log file to be investigated in this example is the result of a mpirun (-np 3) from https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/EllipticPETSc/cube_1e3_XDMF_np3.prj


log = parallel_log
records = parse_file(log)
df_records = pd.DataFrame(records)
df_parallel = fill_ogs_context(df_records)
print(df_parallel.columns)
df_parallel  # noqa: B018

df_ts = analysis_time_step(df_parallel)
# For each mpi_process and each time_step we get the measurements (e.g. output_time)
df_ts  # noqa: B018
# %%
# 1.1. Aggregate measurements over all MPI processes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you are not particularly interested in the performance of each MPI_process pandas gives you all you need to further process data. However, for performance measurement it is recommended to consider always the slowest MPI_process for meaningful interpretation of overall performance (because of synchronization barriers in the evaluation scheme of OGS).
# Then the resulting DataFrame has the same structure like a DataFrame gained from serial OGS log.
df_ts.groupby("time_step").max()

# %%
df_ts[["output_time", "assembly_time"]].boxplot()

# %%
# 2. Performance tuning
# ----------------------------------------------------------------
#
# You can either (2.1) Reduce set of regular expressions when you exactly know what you final analysis will need AND / OR
# (2.2.) Save and load the pandas.DataFrame for the records.
#
# 2.1. Reduce regular expression
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The logparser tries to find matching regular expressions for each line. By default it iterates over all entries specified in :py:mod:`ogstools.logparser.ogs_regexes`.
# You can reduce it to the amount of entries you are actually interested in.
# For this example we are only interested in the number of iterations per time step.
# Because the parsing process is expensive, it is feasible to store the records to a file.


# %%
# 2.2. Save and load records
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We recommend to save the records by any of these methodes http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html.

# df_records.to_hdf("anyfilename.csv")
# pd.read_hdf("anyfilename.csv")


# %%
# 3. Custom analyses
# ----------------------------------------------------------------
# 3.1. Introduction into functions of the logparser
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The function :py:mod:`ogstools.logparser.parse_file` iterates over all lines in the log file. For a specific set of regular expressions it finds it creates a new entry into a list (here named records)
#
log = const_viscosity_thermal_convection_log

# Let us print the content of the log file in this example first.
with Path(log).open() as log_file:
    print(log_file.read())

# ToDo link to documentation
records = parse_file(log)
# The list of records can directly be transformed into a pandas.DataFrame for further inspections. It is the raw presentation of a filtered ogs log in pandas DataFrame format.
df_records = pd.DataFrame(records)
# The logparser is able to find the following entries:
print(df_records.columns)
# For each entry :py:mod:`ogstools.logparser.ogs_regexes` has added the type (corresponding to ogs log level) and value found to the result DataFrame.
df_records  # noqa: B018


# %%

# For each information (e.g. a time measurement or numerical metric) we need to know to which timestep, iteration_number, process, component it belongs.
# ToDo link to documentation, add this information to the table
df_log = fill_ogs_context(df_records)
df_log  # noqa: B018

# %%
# 3.2. Custom analyses - example
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We create a pivot_table where for each time step we see the step_size and the number of iterations.


df_custom = df_records.pivot_table(
    ["step_size", "iteration_number"], ["time_step"], aggfunc=np.max
)
df_custom  # noqa: B018
