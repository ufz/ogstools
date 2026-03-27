# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


import atexit
import importlib.util
import logging
import socket
import subprocess
import sys
import tempfile
import threading
from argparse import ArgumentParser
from enum import IntEnum
from pathlib import Path


class ExitCode(IntEnum):
    """Exit codes returned by the :func:`cli` function.

    Attributes
    ----------
    SUCCESS : int
        0 -- completed successfully.
    BOKEH_FAILED : int
        1 -- the bokeh subprocess exited with a non-zero return code.
    INVALID_INPUT : int
        2 -- invalid input, e.g. the specified JSON file was not found.
    """

    SUCCESS = 0
    BOKEH_FAILED = 1
    INVALID_INPUT = 2


logging.basicConfig()
logger = logging.getLogger(__name__)


def argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Monitor OpenGeoSys simulations via their log output.",
        epilog="Exit codes: 0 success, 1 bokeh failed, 2 invalid input.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        metavar="log-file",
        help="(Required)  OGS log file. Omit when piping: ogs ... | ogsmonitor",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="config-file",
        help="Optional JSON configuration file to fine-tune the displayed output.",
    )
    parser.add_argument(
        "-l", "--log", action="store_true", help="Enable verbose logging."
    )
    return parser


def _stream_stdin_to_file(dest: Path, done: threading.Event) -> None:
    """Read stdin line by line, echo to stderr, and write to dest. Runs in a daemon thread."""
    with dest.open("w") as f:
        for line in sys.stdin:
            sys.stderr.write(line)
            sys.stderr.flush()
            f.write(line)
            f.flush()
    done.set()


def cli() -> int:
    spec = importlib.util.find_spec("ogstools.logparser.monitor_app")
    if spec is None or spec.origin is None:
        msg = "Could not find module 'ogstools.logparser.monitor_app'"
        raise ImportError(msg)
    app_filename = spec.origin

    parser = argparser()
    args = parser.parse_args()
    logger.setLevel(logging.INFO if args.log else logging.ERROR)

    temp_file: Path | None = None
    stdin_subprocess_kwarg: dict = {}
    stdin_done: threading.Event | None = None
    pipe_mode = False

    if args.input:
        logfile_abs = Path(args.input).absolute()
    elif not sys.stdin.isatty():
        # Piped mode: ogs ... | ogsmonitor
        pipe_mode = True
        _, tmp_path = tempfile.mkstemp(suffix=".log", prefix="ogsmonitor_")
        temp_file = Path(tmp_path)
        logfile_abs = temp_file

        stdin_done = threading.Event()
        thread = threading.Thread(
            target=_stream_stdin_to_file,
            args=(temp_file, stdin_done),
            daemon=True,
        )
        thread.start()

        # Don't let bokeh serve inherit our stdin (it would consume the pipe)
        stdin_subprocess_kwarg = {"stdin": subprocess.DEVNULL}

        def _cleanup() -> None:
            if temp_file is not None and temp_file.exists():
                temp_file.unlink()

        atexit.register(_cleanup)
    else:
        parser.error(
            "Provide the filename (relative to current working directory or absolute) of the log file,"
            " or pipe stdin: ogs ... | ogsmonitor.\nUse -h for help."
        )

    json_file = None
    if args.config:
        json_file = Path(args.config).absolute()
        if not json_file.is_file():
            msg = f"Provided JSON file not found: {json_file}"
            logger.error(msg)
            return ExitCode.INVALID_INPUT
        logger.info("Using provided JSON configuration: %s", json_file)
    else:
        json_file = Path("monitor.json").absolute()
        if json_file.is_file():
            logger.info("Using JSON configuration found on disk: %s", json_file)
        else:
            json_file = None
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    cmd = (
        f"bokeh serve --show --port {port} {app_filename} --args {logfile_abs}"
    )
    if json_file is not None:
        cmd += f" {json_file}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            stderr=subprocess.STDOUT,
            **stdin_subprocess_kwarg,
        )
    except KeyboardInterrupt:
        if pipe_mode:
            print("\nOGS and ogsmonitor stopped.")
        else:
            print("\nogsmonitor stopped.")
        return ExitCode.SUCCESS
    if result.returncode != 0:
        logger.error(
            "Starting bokeh failed with returncode %d.", result.returncode
        )
        return ExitCode.BOKEH_FAILED
    return ExitCode.SUCCESS
