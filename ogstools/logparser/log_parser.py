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
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ogstools.logparser.regexes import Log, ogs_regexes


def _try_match_parallel_line(
    line: str, line_nr: int, regex: re.Pattern, pattern_class: type[Log]
) -> Optional[Any]:
    if match := regex.match(line):
        # Line , Process, Type specific
        ts = pattern_class.type_str()
        types = (
            str,
            int,
            int,
        ) + tuple(pattern_class.__annotations__.values())
        match_with_line = (
            ts,
            line_nr,
        ) + match.groups()
        return [ctor(s) for ctor, s in zip(types, match_with_line)]
    return None


def _try_match_serial_line(
    line: str, line_nr: int, regex: re.Pattern, pattern_class: type[Log]
) -> Optional[list[tuple[str, Log]]]:
    if match := regex.match(line):
        # Line , Process, Type specific
        ts = pattern_class.type_str()
        types = (
            str,
            int,
            int,
        ) + tuple(pattern_class.__annotations__.values())
        match_with_line = (
            ts,
            line_nr,
            0,
        ) + match.groups()
        return [ctor(s) for ctor, s in zip(types, match_with_line)]
    return None


def mpi_processes(file_name: Union[str, Path]) -> int:
    """
    Counts the number of MPI processes started by OpenGeoSys-6 by detecting
    specific log entries in a given file. It assumes that each MPI process
    will log two specific messages: "This is OpenGeoSys-6 version" and
    "OGS started on". The function counts occurrences of these messages and
    divides the count by two to estimate the number of MPI processes.

    :param file_name: The path to the log file, as either a string or a Path object.
    :return: An integer representing the estimated number of MPI processes
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


def parse_file(
    file_name: Union[str, Path],
    maximum_lines: Optional[int] = None,
    force_parallel: bool = False,
    ogs_res: Optional[list] = None,
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
    :return: A list of extracted records based on the applied regex patterns.
             The exact type and structure of these records depend on the regex
             patterns and their associated processing functions.
    """
    if isinstance(file_name, str):
        file_name = Path(file_name)
    file_name = Path(file_name)
    parallel_log = force_parallel or mpi_processes(file_name) > 1

    if parallel_log:
        process_regex = "\\[(\\d+)\\]\\ "
        try_match = _try_match_parallel_line
    else:
        process_regex = ""
        try_match = _try_match_serial_line

    def compile_re_fn(mpi_process_regex: str) -> Callable[[str], re.Pattern]:
        return lambda regex: re.compile(mpi_process_regex + regex)

    compile_re = compile_re_fn(process_regex)

    if ogs_res is None:
        ogs_res = ogs_regexes()
    patterns = [(compile_re(k), v) for k, v in ogs_res]

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

            for key, value in patterns:
                if r := try_match(line, number_of_lines_read, key, value):
                    records.append(value(*r))
                    break

    return records
