# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any

from watchdog.events import (
    DirModifiedEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)

from ogstools.logparser.log_parser import (
    normalize_regex,
    parse_line,
    select_regex,
)
from ogstools.logparser.regexes import Context, Log, Termination


class LogFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        file_name: str | Path,
        queue: Queue,
        status: Context,
        stop_callback: Callable[[], tuple[None, Any]],
        force_parallel: bool = False,
        line_limit: int = 0,
    ):
        """
        :param file_name: The location of the log file to monitor.
        :param queue: The queue where log entries are put and to be consumed.
        :status: The status of the simulation (e.g. current time step).
        :stop_callback: A callback function to stop the simulation.
        :force_parallel: Only needed for MPI run with 1 process. Then it must be set to True.
        :line_limit: The number of lines to read before stopping the simulation. 0 means no limit.
        """

        self.file_name = Path(file_name)
        self.queue = queue
        self.status = status
        self.stop_callback = stop_callback
        self.force_parallel = force_parallel
        self.line_limit = line_limit

        self._file_read: bool = False
        self.num_lines_read: int = 0
        # real time monitoring is only working for log version 2 and serial logs or parallel sim without (ogs --parallel_log)
        self.patterns: list = normalize_regex(
            select_regex(version=2), parallel_log=False
        )

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.src_path != str(self.file_name):
            return

        print(f"{self.file_name} has been created.")
        self.process()

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        if event.src_path != str(self.file_name):
            return
        self.process()

    def process(self) -> None:
        if not self._file_read:
            try:
                self._file: Any = self.file_name.open("r")
                self._file.seek(0, 0)
                self._file_read = True
            except FileNotFoundError:
                print(f"File not found yet: {self.file_name}")
                return

        print(f"{self.file_name} has been modified.")
        while True:
            line = self._file.readline()
            num_lines_current = self.num_lines_read + 1
            if not line or not line.endswith("\n"):
                break  # Wait for complete line before processing

            log_entry: Log | Termination | None = parse_line(
                self.patterns,
                line,
                parallel_log=False,
                number_of_lines_read=num_lines_current,
            )

            if log_entry:
                assert isinstance(log_entry, Log | Termination)
                self.queue.put(log_entry)
                self.status.update(log_entry)

            if isinstance(log_entry, Termination):
                print("===== Termination =====")
                self.queue.put(log_entry)
                self.status.update(log_entry)
                self.stop_callback()
                self._file.close()
                break

            if self.line_limit > 0 and self.num_lines_read > self.line_limit:
                self.stop_callback()
                self._file.close()
                break
            self.num_lines_read = self.num_lines_read + 1
