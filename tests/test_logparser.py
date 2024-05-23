import unittest
from collections import defaultdict, namedtuple

import pandas as pd

from ogstools.examples import (
    debug_parallel_3,
    info_parallel_1,
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
    parse_file,
    time_step_vs_iterations,
)


def log_types(records):
    d = defaultdict(list)
    for record in records:
        d[type(record)].append(record)
    return d


class LogparserTest(unittest.TestCase):
    """Test cases for logparser."""

    def test_parallel_1_compare_serial_info(self):
        # Only for MPI execution with 1 process we need to tell the log parser by force_parallel=True!
        records_p = parse_file(info_parallel_1, force_parallel=True)
        num_of_record_type_p = [len(i) for i in log_types(records_p).values()]

        records_s = parse_file(serial_info)
        num_of_record_type_s = [len(i) for i in log_types(records_s).values()]

        self.assertSequenceEqual(
            num_of_record_type_s,
            num_of_record_type_p,
            "The number of logs for each type must be equal for parallel log and serial log",
        )

    def test_parallel_3_debug(self):
        records = parse_file(debug_parallel_3)
        mpi_processes = 3

        self.assertEqual(
            len(records) % mpi_processes,
            0,
            "The number of logs should by a multiple of the number of processes)",
        )

        num_of_record_type = [len(i) for i in log_types(records).values()]
        self.assertEqual(
            all(i % mpi_processes == 0 for i in num_of_record_type),
            True,
            "The number of logs of each type should be a multiple of the number of processes",
        )

        df_records = pd.DataFrame(records)
        df_records = fill_ogs_context(df_records)
        df_ts = analysis_time_step(df_records)

        # some specific values
        record_id = namedtuple("id", "mpi_process time_step")
        digits = 6
        self.assertAlmostEqual(
            df_ts.loc[record_id(mpi_process=0.0, time_step=1.0), "output_time"],
            0.001871,
            digits,
        )
        self.assertAlmostEqual(
            df_ts.loc[record_id(mpi_process=1.0, time_step=1.0), "output_time"],
            0.001833,
            digits,
        )
        self.assertAlmostEqual(
            df_ts.loc[
                record_id(mpi_process=0.0, time_step=1.0), "linear_solver_time"
            ],
            0.004982,
            digits,
        )
        self.assertAlmostEqual(
            df_ts.loc[
                record_id(mpi_process=0.0, time_step=1.0), "assembly_time"
            ],
            0.002892,
            digits,
        )
        self.assertAlmostEqual(
            df_ts.loc[
                record_id(mpi_process=1.0, time_step=1.0), "dirichlet_time"
            ],
            0.000250,
            digits,
        )
        self.assertAlmostEqual(
            df_ts.loc[
                record_id(mpi_process=2.0, time_step=1.0),
                "time_step_solution_time",
            ],
            0.008504,
            digits,
        )

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
        self.assertAlmostEqual(
            df_cni.loc[
                record_id(
                    time_step=1.0,
                    coupling_iteration=0,
                    process=0,
                    iteration_number=1,
                    component=-1,
                ),
                "dx",
            ],
            9.906900e05,
            digits,
        )
        self.assertAlmostEqual(
            df_cni.loc[
                record_id(
                    time_step=10.0,
                    coupling_iteration=5,
                    process=1,
                    iteration_number=1,
                    component=1,
                ),
                "x",
            ],
            1.066500e00,
            digits,
        )

    def test_serial_convergence_coupling_iteration_long(self):
        records = parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_st = analysis_simulation_termination(df_records)
        status = df_st.empty  # No errors assumed
        self.assertEqual(status, True)  #
        if not (status):
            print(df_st)
        self.assertEqual(status, True)  #
        df_records = fill_ogs_context(df_records)
        df_st = analysis_convergence_coupling_iteration(df_records)

        # some specific values
        record_id = namedtuple(
            "id",
            "time_step coupling_iteration coupling_iteration_process component",
        )
        digits = 6
        self.assertAlmostEqual(
            df_st.loc[
                record_id(
                    time_step=1.0,
                    coupling_iteration=1,
                    coupling_iteration_process=0,
                    component=-1,
                ),
                "dx",
            ],
            1.696400e03,
            digits,
        )
        self.assertAlmostEqual(
            df_st.loc[
                record_id(
                    time_step=10.0,
                    coupling_iteration=5,
                    coupling_iteration_process=1,
                    component=-1,
                ),
                "x",
            ],
            1.066500e00,
            digits,
        )

    def test_serial_critical(self):
        records = parse_file(serial_critical)
        self.assertEqual(len(records), 4)
        df_records = pd.DataFrame(records)
        self.assertEqual(len(df_records), 4)
        df_st = analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        self.assertEqual(has_errors, True)
        if has_errors:
            print(df_st)

    def test_serial_warning_only(self):
        records = parse_file(serial_warning_only)
        self.assertEqual(len(records), 1)
        df_records = pd.DataFrame(records)
        self.assertEqual(len(df_records), 1)
        df_st = analysis_simulation_termination(df_records)
        has_errors = not (df_st.empty)
        self.assertEqual(has_errors, True)
        if has_errors:
            print(df_st)

    def test_serial_time_vs_iterations(self):
        records = parse_file(serial_convergence_long)
        df_records = pd.DataFrame(records)
        df_records = fill_ogs_context(df_records)
        df_tsi = time_step_vs_iterations(df_records)
        # some specific values
        self.assertEqual(df_tsi.loc[0, "iteration_number"], 1)
        self.assertEqual(df_tsi.loc[1, "iteration_number"], 6)
        self.assertEqual(df_tsi.loc[10, "iteration_number"], 5)
