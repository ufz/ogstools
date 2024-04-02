# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#              See accompanying file LICENSE.txt or
#              http://www.opengeosys.org/project/license


from typing import Any, Callable

import numpy as np
import pandas as pd


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
            ",".format(),
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

    :return: A decorator function that takes a function accepting a pandas DataFrame and
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
        dfe_newton_iteration[
            "coupling_iteration"
        ] = dfe_newton_iteration.groupby("mpi_process")[
            ["coupling_iteration"]
        ].fillna(
            method="bfill"
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

    dfe_convergence_coupling_iteration[
        "coupling_iteration"
    ] = dfe_convergence_coupling_iteration.groupby("mpi_process")[
        ["coupling_iteration"]
    ].fillna(
        method="ffill"
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
    pt = df.pivot_table(["iteration_number"], ["time_step"], aggfunc=np.max)
    _check_output(pt, interest, context)
    return pt


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


def fill_ogs_context(df_raw_log: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in OpenGeoSys (OGS) log DataFrame by context.
    This function fills missing values in an OpenGeoSys (OGS) log DataFrame by context.

    :param df_raw_log: DataFrame containing the raw OGS log data. Usually, the result of pd.DataFrame(parse_file(file))

    :return: pd.DataFrame with missing values filled by context.

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

    df_raw_log["time_step"] = (
        df_raw_log.groupby("mpi_process")[["time_step"]]
        .fillna(method="ffill")
        .fillna(value=0)
    )

    # Back fill, because iteration number can be found in logs at the END of the iteration
    df_raw_log["iteration_number"] = df_raw_log.groupby("mpi_process")[
        ["iteration_number"]
    ].fillna(method="bfill")

    if "component" in df_raw_log:
        df_raw_log["component"] = df_raw_log.groupby("mpi_process")[
            ["component"]
        ].fillna(value=-1)
    # Forward fill because process will be printed in the beginning - applied to all subsequent
    if "process" in df_raw_log:
        df_raw_log["process"] = df_raw_log.groupby("mpi_process")[
            ["process"]
        ].fillna(method="bfill")
    # Attention - coupling iteration applies to successor line and to all other predecessors - it needs further processing for specific analysis
    if "coupling_iteration_process" in df_raw_log:
        df_raw_log["coupling_iteration_process"] = df_raw_log.groupby(
            "mpi_process"
        )[["coupling_iteration_process"]].fillna(method="ffill", limit=1)
    return df_raw_log
