# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


import threading
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
        line_limit: int = 0,
    ):
        """
        :param file_name: The location of the log file to monitor.
        :param queue: The queue where log entries are put and to be consumed.
        :status: The status of the simulation (e.g. current time step).
        :stop_callback: A callback function to stop the simulation.
        :line_limit: The number of lines to read before stopping the simulation. 0 means no limit.
        """

        self.file_name = Path(file_name)
        self.queue = queue
        self.status = status
        self.stop_callback = stop_callback
        self.line_limit = line_limit

        self._file_read: bool = False
        self._stopped: bool = False
        self._lock = threading.Lock()
        self.num_lines_read: int = 0
        # real time monitoring is only working for log version 2 and serial logs or parallel sim without (ogs --parallel_log)
        self.patterns: list = normalize_regex(
            select_regex(version=2), parallel_log=False
        )

    def on_created(self, event: FileCreatedEvent) -> None:
        if Path(event.src_path).resolve() != self.file_name.resolve():
            return
        self.process()

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        if Path(event.src_path).resolve() != self.file_name.resolve():
            return
        self.process()

    def process(self) -> None:
        with self._lock:
            if self._stopped:
                return

            if not self._file_read:
                try:
                    self._file: Any = self.file_name.open("r")
                    self._file.seek(0, 0)
                    self._file_read = True
                except FileNotFoundError:
                    print(f"File not found yet: {self.file_name}")
                    return

            # print(f"{self.file_name} has been modified.")
            while True:
                # file_pos_before = self._file.tell()
                line = self._file.readline()
                # file_pos_after = self._file.tell()
                num_lines_current = self.num_lines_read + 1
                if not line or not line.endswith("\n"):
                    break  # Wait for complete line before processing

                # Debug: print lines containing "Iteration"
                # if "Iteration" in line:
                #    print(
                #        f"DEBUG handler={id(self)} file={id(self._file)} LINE {num_lines_current} pos {file_pos_before}->{file_pos_after}: {line.strip()}"
                #    )

                log_entry: Log | Termination | None = parse_line(
                    self.patterns,
                    line,
                    parallel_log=False,
                    number_of_lines_read=num_lines_current,
                )

                # Debug: print what was parsed for Iteration lines
                # if "Iteration" in line:
                #    print(
                #        f"DEBUG PARSED: {type(log_entry).__name__ if log_entry else 'None'}"
                #    )

                if log_entry:
                    assert isinstance(log_entry, Log | Termination)
                    self.queue.put(log_entry)
                    self.status.update(log_entry)

                if isinstance(log_entry, Termination):
                    print("===== Termination =====")
                    self.queue.put(log_entry)
                    self.status.update(log_entry)
                    self._stopped = True
                    self._file.close()
                    self.stop_callback()
                    break

                if (
                    self.line_limit > 0
                    and self.num_lines_read > self.line_limit
                ):
                    self._stopped = True
                    self._file.close()
                    self.stop_callback()
                    break
                self.num_lines_read = self.num_lines_read + 1
