import shutil
import tempfile
from collections import defaultdict, namedtuple
from pathlib import Path
from queue import Queue
from time import sleep

import numpy as np
import pandas as pd
import pytest
from dateutil import parser
from watchdog.observers import Observer, ObserverType

from ogstools.examples import (
    debug_parallel_3,
    info_parallel_1,
    log_adaptive_timestepping,
    serial_convergence_long,
    serial_critical,
    serial_info,
    serial_warning_only,
)
from ogstools.logparser import (
    analysis_convergence_coupling_iteration,
    analysis_convergence_newton_iteration,
    analysis_simulation_termination,
    analysis_time_step,
    fill_ogs_context,
    model_and_clock_time,
    parse_file,
    time_step_vs_iterations,
)
from ogstools.logparser.log_file_handler import LogFileHandler

from ogstools.logparser.regexes import Termination


def log_types(records):
    d = defaultdict(list)
    for record in records:
        d[type(record)].append(record)
    return d


class TestLogparser:
    """Test cases for logparser. Until version TODO"""

    #    def test_new(self):
    #        records = parse_file("/home/meisel/gitlabrepos/ogstools/htstat.log")
    #        df_records = pd.DataFrame(records)
    #        df_filled_records = fill_ogs_context(df_records)
    #        df_filled_records.to_csv("test_new.csv")

    def test_parallel_1_compare_serial_info(self):
        # Only for MPI execution with 1 process we need to tell the log parser by force_parallel=True!
        records_p = parse_file(info_parallel_1, force_parallel=True)
        num_of_record_type_p = [len(i) for i in log_types(records_p).values()]

        records_s = parse_file(serial_info)
        num_of_record_type_s = [len(i) for i in log_types(records_s).values()]

        assert (
            num_of_record_type_s == num_of_record_type_p
        ), f"The number of logs for each type must be equal for parallel log (got: {len(num_of_record_type_p)}) and serial log (got: {len(num_of_record_type_s)}))"

    def test_parallel_3_debug(self):
        records = parse_file(debug_parallel_3)
        mpi_processes = 3

        assert (
            len(records) % mpi_processes == 0
        ), "The number of logs should by a multiple of the number of processes)"

        num_of_record_type = [len(i) for i in log_types(records).values()]
        assert all(
            i % mpi_processes == 0 for i in num_of_record_type
        ), "The number of logs of each type should be a multiple of the number of processes"

        df_records = pd.DataFrame(records)
        df_records = fill_ogs_context(df_records)
        df_ts = analysis_time_step(df_records)

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
        records = parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_records = fill_ogs_context(df_records)
        df_cni = analysis_convergence_newton_iteration(df_records)

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
        records = parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_st = analysis_simulation_termination(df_records)
        status = len(df_st) == 2  # No errors assumed
        assert status  #
        if not (status):
            print(df_st)
        assert status  #
        df_records = fill_ogs_context(df_records)
        df_st = analysis_convergence_coupling_iteration(df_records)

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
        records = parse_file(serial_critical)
        num_of_non_parsed_lines = 2  # 2 lines starting with Please re-run ,
        num_lines = sum(1 for _ in serial_critical.open("r", encoding="utf-8"))
        assert (
            len(records) == num_lines - num_of_non_parsed_lines
        ), f"Expected {num_lines-num_of_non_parsed_lines} parsed, but got {len(records)}"
        df_records = pd.DataFrame(records)
        assert len(df_records) == len(
            records
        ), f"Expected that all records are transformed to DataFrame, but got {len(df_records)}"
        df_st = analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        assert has_errors
        if has_errors:
            print(df_st)

    def test_serial_warning_only(self):
        records = parse_file(serial_warning_only)
        num_of_non_parsed_lines = 2  # 2 lines starting with Please re-run ,
        num_lines = sum(
            1 for _ in serial_warning_only.open("r", encoding="utf-8")
        )
        assert (
            len(records) == num_lines - num_of_non_parsed_lines
        ), f"Expected {num_lines-num_of_non_parsed_lines} parsed, but got {len(records)}"
        df_records = pd.DataFrame(records)
        assert len(df_records) == len(
            records
        ), f"Expected that all records are transformed to DataFrame, but got {len(df_records)}"
        df_st = analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        assert has_errors
        if has_errors:
            print(df_st)

    def test_serial_time_vs_iterations(self):
        records = parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_records.to_csv("test_serial_time_vs_iterations_records.csv")
        df_records = fill_ogs_context(df_records)
        df_records.to_csv("test_serial_time_vs_iterations_filled.csv")
        df_tsi = time_step_vs_iterations(df_records)
        # some specific values
        assert (
            df_tsi.loc[0, "iteration_number"] == 1
        ), f"Number of iterations in timestep 0 should be: 1, but got {df_tsi.loc[0, 'iteration_number']}."
        assert df_tsi.loc[1, "iteration_number"] == 6
        assert df_tsi.loc[10, "iteration_number"] == 5

    def test_model_and_clock_time(self):
        records = parse_file(log_adaptive_timestepping)
        df_log = fill_ogs_context(pd.DataFrame(records))
        df_time = model_and_clock_time(df_log)

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
        shutil.copy(input_file, output_file)
        return
    with input_file.open("r") as infile, output_file.open("a") as outfile:
        while chunk := infile.read(chunk_size):
            outfile.write(chunk)
            sleep(delay)


class TestLogparser_Version2:
    """Test cases for logparser. From OGS version 6.4.4"""

    @pytest.mark.parametrize(
         "chunk_size",
         [
             2000, 20000, 200000
         ],
     )
    @pytest.mark.parametrize(
         "delay",
         [
             0, 0.001, 0.05
         ],
     )
    def test_v2_coupled_with_producer(self, chunk_size, delay):
#    def test_v2_coupled_with_producer(self):
#        chunk_size = 5000000
#        delay = 0.001
        original_file = Path("/home/meisel/gitlabrepos/ogstools/ht2.log")
        temp_dir = Path(
            tempfile.mkdtemp(
                f"test_v2_coupled_with_producer_{chunk_size}_{delay}"
            )
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        new_file = temp_dir / "ht.log"
        records: Queue = Queue()
        observer: ObserverType = Observer()
        handler = LogFileHandler(
            new_file,
            queue=records,
            stop_callback=lambda: (print("Stop Observer"), observer.stop()),
        )

        observer.schedule(handler, path=str(new_file.parent), recursive=False)
        print("Starting observer...")
        observer.start()

        # emulating simulation run
        # shutil.copyfile(original_file, new_file)
        write_in_pieces(
            original_file, new_file, chunk_size=chunk_size, delay=delay
        )

        observer.join()
        # consume(records)
        assert (
            records.qsize() == 353
        ), f"Expected 353 records, got {records.qsize()} with {records}"


