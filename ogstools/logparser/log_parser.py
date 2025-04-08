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
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from watchdog.events import (
    DirModifiedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer, ObserverType

from ogstools.logparser.regexes import (
    Log,
    MPIProcess,
    NoRankOutput,
    Termination,
    #    ogs_regexes,
    new_regexes,
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


def mpi_processes(file_name: str | Path) -> int:
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


def _normalize_regex(
    parallel_log: bool = False,
    ogs_res: list | None = None,
) -> Any:  # ToDo

    if ogs_res is None:
        ogs_res = new_regexes()
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


class LogFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        file_name: str | Path,
        patterns: Any,
        queue: Queue,
        stop_callback: Callable[[], tuple[None, Any]],
        force_parallel: bool = False,
        line_limit: int = 0,
    ):

        self.file_name = Path(file_name)

        self._file = self.file_name.open("r")
        self._file.seek(0, 0)
        self.queue = queue
        self.stop_callback = stop_callback
        self.line_num = 0
        self.line_limit = line_limit
        self.force_parallel = force_parallel
        self.patterns = patterns

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        if event.src_path == str(self.file_name):
            # print(f"{self.file_name} has been modified.")
            while True:
                self.line_num = self.line_num + 1
                # print("l:", self.line_num)
                line = self._file.readline()
                if not line or not line.endswith("\n"):
                    # print(line)
                    break  # Wait for complete line before processing

                log_entry = parse_line(
                    self.patterns,
                    line,
                    parallel_log=False,
                    number_of_lines_read=self.line_num,
                )

                if log_entry:
                    self.queue.put(log_entry)
                    print(f"{line}")

                if isinstance(log_entry, Termination):
                    print("===== Termination =====")
                    self.stop_callback()
                    break

                if self.line_limit > 0 and self.line_num > self.line_limit:
                    self.stop_callback()
                    break


def start_log(file_name: str | Path) -> ObserverType:
    queue: Queue = Queue()
    observer: ObserverType = Observer()
    handler = LogFileHandler(
        file_name,
        patterns=_normalize_regex(),
        queue=queue,
        stop_callback=lambda: (print("Stop Observer"), observer.stop()),
    )

    observer.schedule(handler, path=str(file_name), recursive=False)
    observer.start()

    # simple_consumer(queue)  # Start the consumer in the main thread
    return observer


def parse_line(
    patterns: list, line: str, parallel_log: bool, number_of_lines_read: int
) -> Log | None | Termination:

    for regex, log_type in patterns:
        has_mpi_process = parallel_log and issubclass(log_type, MPIProcess)
        fill_mpi = not has_mpi_process or issubclass(log_type, NoRankOutput)
        if r := _try_match_line(
            line,
            number_of_lines_read,
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
    ogs_res: list | None = None,
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

    parallel_log = force_parallel or mpi_processes(file_name) > 1
    patterns = _normalize_regex(parallel_log, ogs_res)

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
