# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#              See accompanying file LICENSE.txt or
#              http://www.opengeosys.org/project/license

import re
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from ogstools.logparser.regexes import (
    Log,
    MPIProcess,
    NoRankOutput,
    Termination,
)


def _try_match_line(
    line: str,
    line_nr: int,
    regex: re.Pattern,
    log_type: type[Log],
    fill_mpi: bool,
) -> Any | None:
    if match := regex.match(line):
        # Line , Process, Type specific
        ts = log_type.type_str()
        types = (str, int, int) + tuple(log_type.__annotations__.values())
        optional_mpi_id = (0,) if fill_mpi else ()
        match_with_line = (ts, line_nr) + optional_mpi_id + match.groups()
        return [
            ctor(s) for ctor, s in zip(types, match_with_line, strict=False)
        ]
    return None


@dataclass
class OGSVersion:
    major: int
    minor: int
    patch: int
    temporary: int = 0
    commit: str = ""
    dirty: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OGSVersion):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, OGSVersion):
            return NotImplemented
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        return not self <= other

    def __ge__(self, other: object) -> bool:
        return not self < other


def read_version(file_name: str | Path) -> OGSVersion:
    return OGSVersion(6, 4, 4, 0)


def read_mpi_processes(file_name: str | Path) -> int:
    """
    Counts the number of MPI processes started by OpenGeoSys-6 by detecting
    specific log entries in a given file. It assumes that each MPI process
    will log two specific messages: "This is OpenGeoSys-6 version" and
    "OGS started on". The function counts occurrences of these messages and
    divides the count by two to estimate the number of MPI processes.

    :param file_name: The path to the log file, as either a string or a Path object.
    :returns: An integer representing the estimated number of MPI processes
             based on the log file's content.

    """
    occurrences = 0
    if isinstance(file_name, str):
        file_name = Path(file_name)
    with file_name.open() as file:
        lines = iter(file)
        # There is no synchronisation barrier between both info, we count both and divide
        while re.search(
            "info: This is OpenGeoSys-6 version|info: OGS started on",
            next(lines),
        ):
            occurrences = occurrences + 1
        return int(occurrences / 2)


def normalize_regex(
    ogs_res: list,
    parallel_log: bool = False,
) -> Any:  # ToDo

    patterns = []
    for regex, log_type in ogs_res:
        mpi_condition = (
            parallel_log
            and issubclass(log_type, MPIProcess)
            and not issubclass(log_type, NoRankOutput)
        )
        mpi_process_regex = "\\[(\\d+)\\]\\ " if mpi_condition else ""
        patterns.append((re.compile(mpi_process_regex + regex), log_type))
    return patterns


def simple_consumer(queue: Queue) -> None:
    print("[Consumer] Started")
    try:
        while True:
            try:
                item = queue.get(timeout=1)  # wait for a log item
                print(f"[Consumer] → {item}")
            except Empty:
                continue  # no data yet, just keep looping
    except KeyboardInterrupt:
        print("[Consumer] Interrupted, exiting...")


@dataclass
class Context:
    time_step = 0
    process = 0
    iteration = 0


def parse_line(
    patterns: list, line: str, parallel_log: bool, number_of_lines_read: int
) -> Log | Termination | None:

    for regex, log_type in patterns:
        has_mpi_process = parallel_log and issubclass(log_type, MPIProcess)
        fill_mpi = not has_mpi_process or issubclass(log_type, NoRankOutput)
        if r := _try_match_line(
            line,
            number_of_lines_read,  # ToDo should not be here
            regex,
            log_type,
            fill_mpi=fill_mpi,
        ):
            return log_type(*r)
    return None


def parse_file(
    file_name: str | Path,
    maximum_lines: int | None = None,
    force_parallel: bool = False,
) -> list[Any]:
    """
    Parses a log file from OGS, applying regex patterns to extract specific information,

    The function supports processing files in serial or parallel mode. In
    parallel mode, a specific regex is used to match log entries from different
    processes.

    :param file_name: The path to the log file, as a string or Path object.
    :param maximum_lines: Optional maximum number of lines to read from the file.
                          If not provided, the whole file is read.
    :param force_parallel: Should only be set to True if OGS run with MPI with a single core
    :returns: A list of extracted records based on the applied regex patterns.
             The exact type and structure of these records depend on the regex
             patterns and their associated processing functions.
    """
    if isinstance(file_name, str):
        file_name = Path(file_name)
    file_name = Path(file_name)

    parallel_log = force_parallel or read_mpi_processes(file_name) > 1
    version = read_version(file_name)
    if version < OGSVersion(6, 5, 4, 220):
        from ogstools.logparser.regexes import ogs_regexes

        regexes = ogs_regexes()
    else:
        from ogstools.logparser.regexes import new_regexes

        regexes = new_regexes()

    patterns = normalize_regex(regexes, parallel_log)

    number_of_lines_read = 0
    with file_name.open() as file:
        lines = iter(file)
        records = []
        for line in lines:
            number_of_lines_read += 1

            if (maximum_lines is not None) and (
                maximum_lines > number_of_lines_read
            ):
                break

            entry = parse_line(
                patterns, line, parallel_log, number_of_lines_read
            )

            if entry:
                records.append(entry)

    return records
