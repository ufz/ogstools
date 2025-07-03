import shutil
import sys
import tempfile
from collections import defaultdict, namedtuple
from pathlib import Path
from queue import Queue
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dateutil import parser
from watchdog.observers import Observer, ObserverType

from ogstools import logparser as lp
from ogstools.examples import (
    debug_parallel_3,
    info_parallel_1,
    log_adaptive_timestepping,
    log_petsc_mpi_1,
    log_petsc_mpi_2,
    serial_convergence_long,
    serial_critical,
    serial_info,
    serial_v2_coupled_ht,
    serial_warning_only,
)
from ogstools.logparser import (
    analysis_convergence_newton_iteration,
    analysis_time_step,
    parse_file,
    read_version,
    time_step_vs_iterations,
)
from ogstools.logparser.log_file_handler import (
    LogFileHandler,
    normalize_regex,
    parse_line,
    select_regex,
)
from ogstools.logparser.regexes import (
    Context,
    StepStatus,
    Termination,
    TimeStepEnd,
    TimeStepStart,
)


def log_types(records):
    d = defaultdict(list)
    for record in records:
        d[type(record)].append(record)
    return d


class TestLogparser:
    """Test cases for logparser. Until version 6.5.4"""

    def test_parallel_1_compare_serial_info(self):
        # Only for MPI execution with 1 process we need to tell the log parser by force_parallel=True!
        records_p = lp.parse_file(info_parallel_1, force_parallel=True)
        num_of_record_type_p = [len(i) for i in log_types(records_p).values()]

        records_s = lp.parse_file(serial_info)
        num_of_record_type_s = [len(i) for i in log_types(records_s).values()]

        assert (
            num_of_record_type_s == num_of_record_type_p
        ), f"The number of logs for each type must be equal for parallel log (got: {len(num_of_record_type_p)}) and serial log (got: {len(num_of_record_type_s)}))"

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
        assert len(records) == 7
        df_records = pd.DataFrame(records)
        assert len(df_records) == 7
        df_st = lp.analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        assert has_errors
        if has_errors:
            print(df_st)

    def test_serial_warning_only(self):
        records = lp.parse_file(serial_warning_only)
        assert len(records) == 4
        df_records = pd.DataFrame(records)
        assert len(df_records) == 4
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
        assert (
            df_tsi.loc[0, "iteration_number"] == 1
        ), f"Number of iterations in timestep 0 should be: 1, but got {df_tsi.loc[0, 'iteration_number']}."
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

        t_start, t_end = map(
            parser.parse, df_log["message"].to_numpy()[[1, -2]]
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
    @pytest.mark.parametrize("x_metric", ["time_step", "model_time"])
    @pytest.mark.parametrize(
        "log", [log_adaptive_timestepping, log_petsc_mpi_1, log_petsc_mpi_2]
    )
    def test_plot_error(self, x_metric: str, log: Path):
        records = lp.parse_file(log)
        df_log = lp.fill_ogs_context(pd.DataFrame(records))
        df_log_copy = df_log.copy()
        fig, axes = plt.subplots(3, figsize=(20, 30))
        lp.plot_convergence(df_log, "dx", x_metric, fig=fig, ax=axes[0])
        lp.plot_convergence(df_log, "dx_x", x_metric, fig=fig, ax=axes[1])
        lp.plot_convergence(df_log, "x", x_metric, fig=fig, ax=axes[2])
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

    def test_read_version(self):
        file = serial_v2_coupled_ht
        assert read_version(file) == 2


def consume(records: Queue) -> None:
    while True:
        item = records.get()
        if isinstance(item, Termination):
            print(f"Consumer: Termination signal ({item}) received. Exiting.")
            break
        print(f"Consumed: {item}")


def write_in_pieces(
    input_file: Path, output_file: Path, chunk_size: int, delay: float
):
    # Get the size of the input file
    input_file_size = input_file.stat().st_size

    # If chunk_size is larger than or equal to the input file size, copy the whole file
    if chunk_size >= input_file_size:
        print("copy")
        shutil.copy(input_file, output_file)
        return
    with input_file.open("r") as infile, output_file.open("a") as outfile:
        while chunk := infile.read(chunk_size):
            outfile.write(chunk)
            sleep(delay)


class TestLogparser_Version2:
    """Test cases for logparser. From OGS version 6.5.4"""

    @pytest.mark.skipif(sys.platform == "darwin", reason="Skipped on macOS")
    @pytest.mark.parametrize(
        "chunk_size",
        [20, 4095, 4096, 20000000000],
    )
    @pytest.mark.parametrize(
        "delay",
        [0, 0.001, 0.003],
    )
    def test_coupled_with_producer(self, chunk_size, delay):
        original_file = serial_v2_coupled_ht
        temp_dir = Path(
            tempfile.mkdtemp(
                f"test_v2_coupled_with_producer_{chunk_size}_{delay}"
            )
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        new_file = temp_dir / "ht.log"
        records: Queue = Queue()
        observer: ObserverType = Observer()
        status: Context = Context()
        shutil.rmtree(new_file, ignore_errors=True)
        handler = LogFileHandler(
            new_file,
            queue=records,
            status=status,
            stop_callback=lambda: (print("Stop Observer"), observer.stop()),
        )

        observer.schedule(handler, path=str(new_file.parent), recursive=False)

        print("Starting observer...")

        observer.start()

        # alternatively shutil.copyfile(original_file, new_file)
        write_in_pieces(
            original_file, new_file, chunk_size=chunk_size, delay=delay
        )

        # For real world application the following line should be commented out
        # consume(records)
        observer.join()
        num_expected = 49
        assert (
            records.qsize() == num_expected
        ), f"Expected {num_expected} records, got {records.qsize()} with {records}"
        # new_file.unlink() no clean up necessary

        assert status.process_step_status == StepStatus.TERMINATED
        assert status.time_step_status == StepStatus.TERMINATED
        assert status.simulation_status == StepStatus.TERMINATED

    # parameterized
    def test_version_select(self):
        original_file = serial_v2_coupled_ht
        ver = read_version(original_file)
        assert ver == 2, f"Expected version 2, but got {ver}"
        l_regexes = len(select_regex(ver))
        assert (
            l_regexes == 22
        ), f"Expected regexes version {ver},this is of length 22 but got {l_regexes}."

    # parameterized
    def test_parse_version(self):
        original_file = serial_v2_coupled_ht
        p = parse_file(original_file)
        l_records = list(p)
        print(l_records)
        assert (
            len(l_records) == 48
        ), f"Expected 48 records, but got {len(l_records)}"

    def test_parse_line(self):
        v2_regexes = normalize_regex(select_regex(2), False)
        line = "info: Time step #53 started. Time: 2000. Step size: 1000."
        ts_start_record = parse_line(
            v2_regexes,
            line,
            False,
            1,
        )
        assert ts_start_record is not None
        assert isinstance(ts_start_record, TimeStepStart)
        assert ts_start_record.time_step == 53
        assert ts_start_record.step_start_time == 2000
        assert ts_start_record.step_size == 1000

        line = "info: [time] Time step #99 took 0.1234 s."
        ts_end_record = parse_line(
            v2_regexes,
            line,
            False,
            1,
        )

        assert ts_end_record is not None
        assert isinstance(ts_end_record, TimeStepEnd)
        assert ts_end_record.time_step == 99
        assert ts_end_record.time_step_finished_time == 0.1234

        line = "info: Solving process #123 started."
        p_started = parse_line(
            v2_regexes,
            line,
            False,
            1,
        )
        assert p_started is not None
        assert p_started.process == 123

        line = "info: Convergence criterion, component 0: |dx|=2.6647e+00, |x|=1.1234e-01, |dx|/|x|=1.0000e+00"
        conv = parse_line(
            v2_regexes,
            line,
            False,
            1,
        )
        assert conv is not None
        assert conv.dx == 2.6647e00

    def test_construct_ts_good(self):
        ts_start_record = TimeStepStart(
            type="Info",
            line=1,
            mpi_process=0,
            time_step=53,
            step_start_time=2000.0,
            step_size=1000.0,
        )
        ts_end_record = TimeStepEnd(
            type="Info",
            line=1,
            mpi_process=0,
            time_step=53,
            time_step_finished_time=0.1234,
        )
        c = Context()

        assert c.time_step is None
        c.update(ts_start_record)
        assert c.time_step == 53
        assert c.time_step_status == StepStatus.RUNNING
        c.update(ts_end_record)
        assert c.time_step == 53
        assert c.time_step_status == StepStatus.TERMINATED

    def test_parse_good(self):
        original_file = serial_v2_coupled_ht
        records = parse_file(original_file)
        df_records = pd.DataFrame(records)
        df_tsit = time_step_vs_iterations(df_records)
        assert len(df_tsit) == 2
        df_ats = analysis_time_step(df_records)
        assert len(df_ats) == 3
        df_acni = analysis_convergence_newton_iteration(df_records)
        assert len(df_acni) == 10
