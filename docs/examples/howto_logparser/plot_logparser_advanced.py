"""
Advanced topics
===============

We address to following:

1. Handling OGS logs from parallel computation (OGS with MPI runs).

2. Reduce computation time required for log processing.

3. Creating custom analyses.

Although these topics are discussed together, they are independent of each
other and can be utilized separately as needed.

"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from ogstools.examples import (
    log_const_viscosity_thermal_convection,
    log_parallel,
)
from ogstools.logparser import (
    analysis_time_step,
    fill_ogs_context,
    parse_file,
)

pd.set_option("display.max_rows", 8)  # for visualization only

# %%
# 1. Logs from parallel computations (with MPI)
# ---------------------------------------------
# The log file to be investigated in this example is the result of a mpirun (-np 3) from https://gitlab.opengeosys.org/ogs/ogs/-/blob/master/Tests/Data/EllipticPETSc/cube_1e3_XDMF_np3.prj


records = parse_file(log_parallel)
df_records = pd.DataFrame(records)
df_parallel = fill_ogs_context(df_records)
df_parallel

df_ts = analysis_time_step(df_parallel)
# For each mpi_process and each time_step we get the measurements (e.g. output_time)
df_ts
# %% [markdown]
# 1.1. Aggregate measurements over all MPI processes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you are not particularly interested in the performance of each MPI_process pandas gives you all you need to further process data. However, for performance measurement it is recommended to consider always the slowest MPI_process for meaningful interpretation of overall performance (because of synchronization barriers in the evaluation scheme of OGS).
# Then the resulting DataFrame has the same structure like a DataFrame gained from serial OGS log.
df_ts.groupby("time_step").max()

# %%
df_ts[["output_time", "assembly_time"]].boxplot()

# %% [markdown]
# 2. Reduce computation time to process logs
# ------------------------------------------
#
# To reduce the computation to evaluate the logs, you can either
# (2.1) Reduce set of regular expressions, when you exactly know,
# what you final analysis will need

# AND / OR

# (2.2.) Save and load the pandas.DataFrame for the records.
#
# 2.1. Reduce regular expression
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The logparser tries to find matching regular expressions for each line. By default it iterates over all entries specified in :py:mod:`ogstools.logparser.ogs_regexes`.
# You can reduce it to the amount of entries you are actually interested in.
# For this example we are only interested in the number of iterations per time step.
# Because the parsing process is expensive, it is useful to store the records to a file.
# According to :py:mod:`ogstools.logparser.parse_file`
# via parameter `regexes` a list of reduced or custom regexes can be provided.


# %% [markdown]
# 2.2. Save and load records
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We recommend saving the records by any of these methodes http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html.
# For example with HDF:
# ```python
# df_records.to_hdf("anyfilename.csv")
# pd.read_hdf("anyfilename.csv")
# ```

# %%
# 3. Custom analyses
# ------------------
# 3.1. Introduction into functions of the logparser
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The function :py:mod:`ogstools.logparser.parse_file` iterates over all lines in the log file. For a specific set of regular expressions it finds and creates a new entry into a list (here named records)
#

# Let us print the content of the log file in this example.
with Path(log_const_viscosity_thermal_convection).open() as log_file:
    print(log_file.read())

records = parse_file(log_const_viscosity_thermal_convection)
# The list of records can directly be transformed into a pandas.DataFrame for further inspections. It is the raw representation of a filtered OGS log in pandas DataFrame format.
df_records = pd.DataFrame(records)
# The logparser is able to find the following entries:
print(df_records.columns)
# For each entry :py:mod:`ogstools.logparser.ogs_regexes` has added the type (corresponding to OGS log level) and value found to the result DataFrame.
df_records


# %%

# For each information (e.g. a time measurement or numerical metric) we need to know to which timestep, iteration_number, process, component it belongs.
df_log = fill_ogs_context(df_records)
df_log

# %%
# 3.2. Custom analyses - example
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We create a pivot_table where for each time step we can see the step_size and the number of iterations.


df_custom = df_records.pivot_table(
    ["step_size", "iteration_number"], ["time_step"], aggfunc=np.max
)
df_custom
