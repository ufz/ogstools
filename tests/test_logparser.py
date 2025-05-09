from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dateutil import parser

from ogstools import logparser as lp
from ogstools.examples import (
    debug_parallel_3,
    info_parallel_1,
    log_adaptive_timestepping,
    serial_convergence_long,
    serial_critical,
    serial_info,
    serial_warning_only,
)


def log_types(records):
    d = defaultdict(list)
    for record in records:
        d[type(record)].append(record)
    return d


class TestLogparser:
    """Test cases for logparser."""

    def test_parallel_1_compare_serial_info(self):
        # Only for MPI execution with 1 process we need to tell the log parser by force_parallel=True!
        records_p = lp.parse_file(info_parallel_1, force_parallel=True)
        num_of_record_type_p = [len(i) for i in log_types(records_p).values()]

        records_s = lp.parse_file(serial_info)
        num_of_record_type_s = [len(i) for i in log_types(records_s).values()]

        assert (
            num_of_record_type_s == num_of_record_type_p
        ), "The number of logs for each type must be equal for parallel log and serial log"

    def test_parallel_3_debug(self):
        records = lp.parse_file(debug_parallel_3)
        mpi_processes = 3

        assert (
            len(records) % mpi_processes == 0
        ), "The number of logs should by a multiple of the number of processes)"

        num_of_record_type = [len(i) for i in log_types(records).values()]
        assert all(
            i % mpi_processes == 0 for i in num_of_record_type
        ), "The number of logs of each type should be a multiple of the number of processes"

        df_records = pd.DataFrame(records)
        df_records = lp.fill_ogs_context(df_records)
        df_ts = lp.analysis_time_step(df_records)

        # some specific values
        record_id = namedtuple("id", "mpi_process time_step")
        digits = 6
        assert df_ts.loc[
            record_id(mpi_process=0.0, time_step=1.0), "output_time"
        ] == pytest.approx(0.001871, digits)
        assert df_ts.loc[
            record_id(mpi_process=1.0, time_step=1.0), "output_time"
        ] == pytest.approx(0.001833, digits)
        assert df_ts.loc[
            record_id(mpi_process=0.0, time_step=1.0), "linear_solver_time"
        ] == pytest.approx(0.004982, digits)
        assert df_ts.loc[
            record_id(mpi_process=0.0, time_step=1.0), "assembly_time"
        ] == pytest.approx(0.002892, digits)
        assert df_ts.loc[
            record_id(mpi_process=1.0, time_step=1.0), "dirichlet_time"
        ] == pytest.approx(0.000250, digits)
        assert df_ts.loc[
            record_id(mpi_process=2.0, time_step=1.0),
            "time_step_solution_time",
        ] == pytest.approx(0.008504, digits)

    def test_serial_convergence_newton_iteration_long(self):
        records = lp.parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_records = lp.fill_ogs_context(df_records)
        df_cni = lp.analysis_convergence_newton_iteration(df_records)

        # some specific values
        record_id = namedtuple(
            "id",
            "time_step coupling_iteration process iteration_number component",
        )
        digits = 6
        assert df_cni.loc[
            record_id(
                time_step=1.0,
                coupling_iteration=0,
                process=0,
                iteration_number=1,
                component=-1,
            ),
            "dx",
        ] == pytest.approx(9.906900e05, digits)
        assert df_cni.loc[
            record_id(
                time_step=10.0,
                coupling_iteration=5,
                process=1,
                iteration_number=1,
                component=1,
            ),
            "x",
        ] == pytest.approx(1.066500e00, digits)

    def test_serial_convergence_coupling_iteration_long(self):
        records = lp.parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_st = lp.analysis_simulation_termination(df_records)
        status = len(df_st) == 2  # No errors assumed
        assert status  #
        if not (status):
            print(df_st)
        assert status  #
        df_records = lp.fill_ogs_context(df_records)
        df_st = lp.analysis_convergence_coupling_iteration(df_records)

        # some specific values
        record_id = namedtuple(
            "id",
            "time_step coupling_iteration coupling_iteration_process component",
        )
        digits = 6
        assert df_st.loc[
            record_id(
                time_step=1.0,
                coupling_iteration=1,
                coupling_iteration_process=0,
                component=-1,
            ),
            "dx",
        ] == pytest.approx(1.696400e03, digits)
        assert df_st.loc[
            record_id(
                time_step=10.0,
                coupling_iteration=5,
                coupling_iteration_process=1,
                component=-1,
            ),
            "x",
        ] == pytest.approx(1.066500e00, digits)

    def test_serial_critical(self):
        records = lp.parse_file(serial_critical)
        assert len(records) == 6
        df_records = pd.DataFrame(records)
        assert len(df_records) == 6
        df_st = lp.analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        assert has_errors
        if has_errors:
            print(df_st)

    def test_serial_warning_only(self):
        records = lp.parse_file(serial_warning_only)
        assert len(records) == 3
        df_records = pd.DataFrame(records)
        assert len(df_records) == 3
        df_st = lp.analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        assert has_errors
        if has_errors:
            print(df_st)

    def test_serial_time_vs_iterations(self):
        records = lp.parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_records = lp.fill_ogs_context(df_records)
        df_tsi = lp.time_step_vs_iterations(df_records)
        # some specific values
        assert df_tsi.loc[0, "iteration_number"] == 1
        assert df_tsi.loc[1, "iteration_number"] == 6
        assert df_tsi.loc[10, "iteration_number"] == 5

    def test_model_and_clock_time(self):
        records = lp.parse_file(log_adaptive_timestepping)
        df_log = lp.fill_ogs_context(pd.DataFrame(records))
        df_log_copy = df_log.copy()
        df_time = lp.model_and_clock_time(df_log)
        pd.testing.assert_frame_equal(df_log_copy, df_log)

        assert np.isclose(np.max(df_time["step_size"]), 0.48490317342720013), (
            f"Maximum step_size {np.max(df_time['step_size'])} does not match "
            "the value in the log."
        )
        assert (np.min(df_time["step_size"])) == 0.0001, (
            f"Minimum step_size {np.min(df_time['step_size'])} does not match "
            "the value in the log."
        )
        assert (np.max(df_time["iterations"])) == 17, (
            f"Maximum iterations {np.max(df_time['iterations'])} does not "
            "match the value in the log."
        )
        assert np.isclose(np.mean(df_time["iterations"]), 5.476190476190476), (
            f"Mean number of iterations {np.mean(df_time['iterations'])} does "
            "not add up to the expected value. Some data might be missing."
        )
        t_start, t_end = map(
            parser.parse, df_log["message"].to_numpy()[[0, -2]]
        )
        assert np.any(
            np.diff(df_time.index) < 0
        ), "No reduction in model_time measured, contrary to the example data."
        final_clock_time = df_time["clock_time"].to_numpy()[-1]
        run_time = (t_end - t_start).seconds
        assert np.isclose(final_clock_time, run_time, rtol=0.02), (
            f"Difference between final clock_time {final_clock_time} and "
            f"total_runtime {run_time} from timestamps is to large."
        )

    # TODO: graphical output is not yet tested
    def test_plot_error(self):
        records = lp.parse_file(log_adaptive_timestepping)
        df_log = lp.fill_ogs_context(pd.DataFrame(records))
        df_log_copy = df_log.copy()
        fig, axes = plt.subplots(3, figsize=(20, 30))
        lp.plot_convergence(df_log, "dx", fig=fig, ax=axes[0])
        lp.plot_convergence(df_log, "dx_x", fig=fig, ax=axes[1])
        lp.plot_convergence(df_log, "x", fig=fig, ax=axes[2])
        pd.testing.assert_frame_equal(df_log_copy, df_log)
        assert len(fig.axes) == 6  # 3 original axes + 3 axes for the colorbars

    # TODO: graphical output is not yet tested
    def test_plot_convergence_order(self):
        records = lp.parse_file(log_adaptive_timestepping)
        df_log = lp.fill_ogs_context(pd.DataFrame(records))
        df_log_copy = df_log.copy()
        fig = lp.plot_convergence_order(df_log, n=3, x_metric="time_step")
        fig = lp.plot_convergence_order(df_log, n=4, x_metric="model_time")
        pd.testing.assert_frame_equal(df_log_copy, df_log)
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "convergence order $q$"
