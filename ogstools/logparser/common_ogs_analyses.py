# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#              See accompanying file LICENSE.txt or
#              http://www.opengeosys.org/project/license


from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from typeguard import typechecked


# Helper functions
def _check_input(
    df: pd.DataFrame, interest: list[str], context: list[str]
) -> None:
    diff = set(interest) - set(df.columns)
    if diff:
        msg = "Column(s) of interest ({}) is/are not present in table".format(
            ",".join(diff)
        )
        raise Exception(msg)
    diff = set(context) - set(df.columns)
    if diff:
        msg = "Column(s) of context ({}) is/are not present in table"
        raise Exception(
            msg,
            ",",
        )


def _check_output(
    pt: pd.DataFrame, interest: list[str], context: list[str]
) -> None:
    if pt.empty:
        msg = "The values of {} are not associated to all of {}. Call or see fill_ogs_context".format(
            ",".join(interest), ",".join(context)
        )
        raise Exception(msg)


# decorator for analyses
def pre_post_check(interest: list[str], context: list[str]) -> Callable:
    """
    A decorator for analyzing pandas DataFrames before and after applying a function.
    It checks the DataFrame against specified 'interest' and 'context' criteria both
    before and after the function is called.


    :param interest: indicates the columns of interest in the DataFrame.
    :param context: indicates the context columns in the DataFrame that should be checked.

    :returns: A decorator function that takes a function accepting a pandas DataFrame and
      returns a modified DataFrame, wrapping it with pre-check and post-check logic
      based on the specified 'interest' and 'context'.
    """

    def wrap(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def wrapped_f(df: Any) -> Any:
            _check_input(df, interest, context)
            pt = func(df)
            _check_output(pt, interest, context)
            return pt

        return wrapped_f

    return wrap


def analysis_time_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis with focus on computation time per time step. It combines time step specific measurements 'output time'
    and 'time step solution time' with iteration specific measurements 'assembly time', 'linear solver time', 'Dirichlet time'.
    Time from iteration are accumulated.
    """
    interest1 = ["output_time", "time_step_solution_time", "step_size"]
    interest2 = ["assembly_time", "linear_solver_time", "dirichlet_time"]
    interest = [*interest1, *interest2]
    context = ["mpi_process", "time_step"]
    _check_input(df, interest, context)

    dfe_ts = df.pivot_table(interest1, context)
    # accumulates coupling iterations and newton iterations
    dfe_tsi = df.pivot_table(interest2, context, aggfunc="sum")

    dfe = dfe_ts.merge(dfe_tsi, left_index=True, right_index=True)
    _check_output(dfe, interest, context)
    return dfe


def analysis_simulation(df: pd.DataFrame) -> pd.DataFrame:
    interest = ["execution_time"]  # 'start_time'
    context = ["mpi_process"]
    _check_input(df, interest, context)

    pt = df.pivot_table(interest, context)
    _check_output(pt, interest, context)
    return pt


def analysis_convergence_newton_iteration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convergence metrics need to be interpreted as norm `|x|`,`|dx|`, `|dx|/|x|` and are specific to
    defined <convergence_criterion> in prj - file.
    """
    dfe_newton_iteration = df.copy()
    interest = ["dx", "x", "dx_x"]
    if "coupling_iteration" in df:
        context = [
            "time_step",
            "coupling_iteration",
            "process",
            "iteration_number",
        ]
        if "component" in df.columns:
            context.append("component")
        _check_input(df, interest, context)
        # Eliminate all entries for coupling iteration (not of interest in this study)
        dfe_newton_iteration["coupling_iteration"] = (
            dfe_newton_iteration.groupby("mpi_process")[
                ["coupling_iteration"]
            ].bfill()
        )
        dfe_newton_iteration = dfe_newton_iteration[
            ~dfe_newton_iteration["coupling_iteration_process"].notna()
        ]
        dfe_newton_iteration = dfe_newton_iteration.dropna(subset=["x"])

        pt = dfe_newton_iteration.pivot_table(interest, context)

    else:
        context = ["time_step", "process", "iteration_number"]
        if "component" in df.columns:
            context.append("component")
        _check_input(df, interest, context)
        pt = dfe_newton_iteration.pivot_table(interest, context)

    _check_output(pt, interest, context)
    return pt


@pre_post_check(
    interest=["dx", "x", "dx_x"],
    context=["time_step", "coupling_iteration", "coupling_iteration_process"],
)
def analysis_convergence_coupling_iteration(df: pd.DataFrame) -> pd.DataFrame:
    # Coupling iteration column will be modified specific for coupling iteration analysis, modified data can not be used for other analysis ->copy!
    dfe_convergence_coupling_iteration = df.copy()
    interest = ["dx", "x", "dx_x"]
    context = ["time_step", "coupling_iteration", "coupling_iteration_process"]
    if "component" in df.columns:
        context.append("component")
    _check_input(df, interest, context)

    dfe_convergence_coupling_iteration["coupling_iteration"] = (
        dfe_convergence_coupling_iteration.groupby("mpi_process")[
            ["coupling_iteration"]
        ].ffill()
    )
    # All context log lines (iteration_number) have no values for dx, dx_x, x . From now on not needed -> dropped
    dfe_convergence_coupling_iteration = (
        dfe_convergence_coupling_iteration.dropna(
            subset=["coupling_iteration_process"]
        ).dropna(subset=["x"])
    )

    pt = dfe_convergence_coupling_iteration.pivot_table(interest, context)
    _check_output(pt, interest, context)
    return pt


def time_step_vs_iterations(df: pd.DataFrame) -> pd.DataFrame:
    interest = ["iteration_number"]
    context = ["time_step"]
    _check_input(df, interest, context)
    pt = df.pivot_table(["iteration_number"], ["time_step"], aggfunc="max")
    _check_output(pt, interest, context)
    return pt


@typechecked
def errors_per_ts_iteration(
    df: pd.DataFrame, metric: Literal["dx", "x", "dx_x"] = "dx"
) -> np.ndarray:
    return (
        analysis_convergence_newton_iteration(df)
        .pivot_table(
            index="iteration_number",
            columns="time_step",
            values=metric,
            fill_value=np.nan,
        )
        .to_numpy()
    )


@typechecked
def convergence_order_per_ts_iteration(
    df: pd.DataFrame,
    n: Literal[3, 4] = 3,
) -> np.ndarray:
    """Compute the convergence order of iterative solver errors.

    :math:`q(n=3) = \\frac{\\log | \\frac{x_{k-1}}{x_{k}} |}{\\log | \\frac{x_{k-2}}{x_{k-1}} |}`
    :math:`q(n=4) = \\frac{\\log | \\frac{x_{k+1}-x_{k}}{x_{k}-x_{k-1}} |}{\\log | \\frac{x_{k}-x_{k-1}}{x_{k-1}-x_{k-2}} |}`
    """
    errors = errors_per_ts_iteration(df)
    values = errors[1:] - errors[:-1] if n == 4 else errors
    log_ratios = np.log10(np.abs(values[1:] / values[:-1]))
    orders = log_ratios[1:] / log_ratios[:-1]
    orders = np.vstack((np.full((2, orders.shape[1]), np.nan), orders))
    if n == 4:
        orders = np.vstack((orders, np.full((1, orders.shape[1]), np.nan)))
    return orders


def model_and_clock_time(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe of an OGS log for inspection of data over time.

    The resulting dataframe's index is 'model_time'. The remaining columns
    consist of: 'time_step', 'step_size', 'iterations' and 'clock_time'.

    :param df:  The dataframe of an OGS log

    :returns:   The processed dataframe.

    """
    interest, context = (["step_start_time"], ["time_step", "step_size"])
    _check_input(df, interest, context)
    df_new = df.copy()
    df_time = df_new.pivot_table(interest, context, sort=False)
    _check_output(df_time, interest, context)
    df_time = df_time.reset_index().set_index("step_start_time")
    # NOTE: iteration_number may contain some faulty offset, but the
    # following aggregation works anyway, as we take the max value.
    interest, context = (["iteration_number"], ["time_step", "step_start_time"])
    _check_input(df_new, interest, context)
    df_new["step_start_time"] = df_new["step_start_time"].ffill()
    df_iter = df_new.pivot_table(interest, context, aggfunc=np.max, sort=False)
    # this trick handles the case when the data is one element short
    # which might be the case if the simulation is still running.
    iterations = np.zeros(len(df_time))
    iterations[-len(df_iter) :] = df_iter["iteration_number"].to_numpy()
    df_time["iterations"] = iterations
    # TODO: output_times + something else is still missing here
    sol_times = df_new["time_step_finished_time"].dropna().to_numpy()
    clock_time = np.zeros(len(df_time))
    clock_time[-len(sol_times) :] = np.cumsum(sol_times)
    df_time["clock_time"] = clock_time
    return df_time.rename_axis("model_time")


def analysis_simulation_termination(df: pd.DataFrame) -> pd.DataFrame:
    # For full print of messages consider setup jupyter notebook:
    # pd.set_option('display.max_colwidth', None)
    interest = ["message"]
    context = ["message", "line", "mpi_process", "type"]

    if "message" in df:
        _check_input(df, interest, context)
        df2 = df.dropna(subset=interest)[context]
        # ToDo Merge columns together and add a column for type (warning, error, critical)
        _check_output(df2, interest, context)
        return df2
    return pd.DataFrame()


def _types(df_raw_log: pd.DataFrame) -> pd.DataFrame:
    int_columns = [
        "line",
        "mpi_process",
        "time_step",
        "iteration_number",
        "coupling_iteration",
        "coupling_iteration_process",
        "component",
        "process",
    ]

    for column in df_raw_log.columns:
        if column in int_columns:
            try:
                df_raw_log[column] = df_raw_log[column].astype("Int64")
            except ValueError:
                print(
                    f"Could not convert column '{column}' to integer due to value error"
                )
            except TypeError:
                print(
                    f"Could not convert column '{column}' to integer due to type error"
                )
    return df_raw_log


def fill(df_raw_log: pd.DataFrame) -> pd.DataFrame:
    df_raw_log = _types(df_raw_log)
    df_raw_log["time_step"] = (
        df_raw_log.groupby("mpi_process")[["time_step"]].ffill().fillna(value=0)
    )

    df_raw_log["process"] = (
        df_raw_log.groupby("mpi_process")[["process"]].ffill().fillna(value=0)
    )

    return df_raw_log


def fill_ogs_context(df_raw_log: pd.DataFrame) -> pd.DataFrame:
    """
    Only needed for logs of Version 1.
    Fill missing values in OpenGeoSys (OGS) log DataFrame by context.
    This function fills missing values in an OpenGeoSys (OGS) log DataFrame by context.

    :param df_raw_log: DataFrame containing the raw OGS log data. Usually, the result of pd.DataFrame(parse_file(file))

    :returns: pd.DataFrame with missing values filled by context.

    References:
    Pandas documentation : https://pandas.pydata.org/pandas-docs/stable/user_guide/

    Notes:
    Some logs do not contain information about time_step and iteration. The
    information must be collected by context (by surrounding log lines from same mpi_process).
    Logs are grouped by mpi_process to get only surrounding log lines from same mpi_process.
    There are log lines that give the current time step (when time step starts).
    It can be assumed that in all following lines belong to this time steps, until next
    collected value of time step.
    Some columns that contain actual integer values are converted to float.
    See https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    ToDo list of columns with integer values are known from regular expression

    """
    df_raw_log = _types(df_raw_log)

    df_raw_log["time_step"] = (
        df_raw_log.groupby("mpi_process")[["time_step"]].ffill().fillna(value=0)
    )

    # Back fill, because iteration number can be found in logs at the END of the iteration
    df_raw_log["iteration_number"] = df_raw_log.groupby("mpi_process")[
        ["iteration_number"]
    ].bfill()

    if "component" in df_raw_log:
        df_raw_log["component"] = df_raw_log.groupby("mpi_process")[
            ["component"]
        ].transform(lambda x: x.fillna(-1))
    # Forward fill because process will be printed in the beginning - applied to all subsequent
    if "process" in df_raw_log:
        df_raw_log["process"] = df_raw_log.groupby("mpi_process")[
            ["process"]
        ].bfill()
    # Attention - coupling iteration applies to successor line and to all other predecessors - it needs further processing for specific analysis
    if "coupling_iteration_process" in df_raw_log:
        df_raw_log["coupling_iteration_process"] = df_raw_log.groupby(
            "mpi_process"
        )[["coupling_iteration_process"]].ffill(limit=1)
    return df_raw_log
