from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any

from watchdog.events import (
    DirModifiedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)

from ogstools.logparser.log_parser import (
    normalize_regex,
    parse_line,
    read_mpi_processes,
    read_version,
    select_regex,
)
from ogstools.logparser.regexes import Termination


class LogFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        file_name: str | Path,
        queue: Queue,
        stop_callback: Callable[[], tuple[None, Any]],
        force_parallel: bool = False,
        line_limit: int = 0,
    ):

        self.file_name = Path(file_name)
        self._file_read = False
        self.patterns = None
        self.queue = queue
        self.stop_callback = stop_callback
        self.line_num = 0
        self.line_limit = line_limit
        self.force_parallel = force_parallel

        if self.patterns is None:
            # parallel_log = (
            #     self.force_parallel or read_mpi_processes(self.file_name) > 1
            # )
            parallel_log = False
            self.patterns = normalize_regex(
                select_regex(read_version(self.file_name)), parallel_log
            )

    def on_modified(self, event: FileModifiedEvent | DirModifiedEvent) -> None:
        if event.src_path != str(self.file_name):
            return

        if not self._file_read:
            try:
                self._file: Any = self.file_name.open("r")
                self._file.seek(0, 0)
            except FileNotFoundError:
                print(f"File not found yet: {self.file_name}")
                return

        print(f"{self.file_name} has been modified.")
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
                # print(f"{line}")

            if isinstance(log_entry, Termination):
                print("===== Termination =====")
                self.stop_callback()
                self._file.close()
                break

            if self.line_limit > 0 and self.line_num > self.line_limit:
                self.stop_callback()
                self._file.close()
                break
