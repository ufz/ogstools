# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any

from watchdog.events import (
    DirCreatedEvent,
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

logger = logging.getLogger(__name__)


class LogFileHandler(FileSystemEventHandler):
    """Parse the lines that OGS appends to a log file and put the records into a queue."""

    def __init__(
        self,
        file_name: str | Path,
        queue: Queue,
        status: Context,
        stop_callback: Callable[[], None],
        line_limit: int = 0,
    ):
        """
        :param file_name: The location of the log file to monitor.
        :param queue: The queue where log entries are put and to be consumed.
        :param status: The status of the simulation (e.g. current time step).
        :param stop_callback: A callback function to stop the simulation.
        :param line_limit: The number of lines to read before stopping the simulation. 0 means no limit.
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

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """Process the log file when it is created.

        :param event: The file system event from watchdog.
        """
        if Path(event.src_path).resolve() != self.file_name.resolve():
            return
        self.process()

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        """Process the log file when it changes.

        :param event: The file system event from watchdog.
        """
        if Path(event.src_path).resolve() != self.file_name.resolve():
            return
        self.process()

    def process(self) -> None:
        """Parse the complete lines added since the last call.

        Stop at the termination record or after `line_limit` lines.
        """
        with self._lock:
            if self._stopped:
                return

            if not self._file_read:
                try:
                    self._file: Any = self.file_name.open("r")
                    self._file.seek(0, 0)
                    self._file_read = True
                except FileNotFoundError:
                    logger.info("File not found yet: %s", self.file_name)
                    return

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
                    logger.info("Simulation terminated: %s", self.file_name)
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
