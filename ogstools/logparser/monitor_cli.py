# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


import atexit
import importlib.util
import subprocess
import sys
import tempfile
import threading
from argparse import ArgumentParser
from pathlib import Path

spec = importlib.util.find_spec(
    "ogstools.logparser.monitor_app"
)  # replace with any dotted name
if spec is None or spec.origin is None:
    msg = "Could not find module 'ogstools.logparser.monitor_app'"
    raise ImportError(msg)
app_filename = spec.origin
print(app_filename)
parser = ArgumentParser(description="This tool monitors OGS logfiles.")

parser.add_argument("-i", "--input", help="The path to the logfile.")

parser.add_argument("-j", "--json", help="The path to the json file.")


def _stream_stdin_to_file(dest: Path) -> None:
    """Read stdin line by line, echo to stderr, and write to dest. Runs in a daemon thread."""
    with dest.open("w") as f:
        for line in sys.stdin:
            sys.stderr.write(line)
            sys.stderr.flush()
            f.write(line)
            f.flush()


def cli() -> int:
    args = parser.parse_args()

    temp_file: Path | None = None
    stdin_subprocess_kwarg: dict = {}

    if args.input:
        logfile_abs = Path(args.input).absolute()
    elif not sys.stdin.isatty():
        # Piped mode: ogs ... | tee y.log | ogsmonitor
        _, tmp_path = tempfile.mkstemp(suffix=".log", prefix="ogsmonitor_")
        temp_file = Path(tmp_path)
        logfile_abs = temp_file

        thread = threading.Thread(
            target=_stream_stdin_to_file, args=(temp_file,), daemon=True
        )
        thread.start()

        # Don't let bokeh serve inherit our stdin (it would consume the pipe)
        stdin_subprocess_kwarg = {"stdin": subprocess.DEVNULL}

        def _cleanup() -> None:
            if temp_file is not None and temp_file.exists():
                temp_file.unlink()

        atexit.register(_cleanup)
    else:
        parser.error("Provide -i LOGFILE or pipe stdin: ogs ... | ogsmonitor")

    jsonfile = None
    if args.json:
        print("jsonfile provided")
        jsonfile = Path(args.json)
        jsonfile = jsonfile.absolute()
        if not jsonfile.is_file():
            msg = "Provided JSON file not found"
            raise FileNotFoundError(msg)
    else:
        jsonfile = Path("monitor.json")
        print("jsonfile found on disk")
        jsonfile = jsonfile.absolute()
        if not jsonfile.is_file():
            jsonfile = None
    cmd = f"bokeh serve --show {app_filename} --args {logfile_abs}"
    if jsonfile is not None:
        cmd += f" {jsonfile}"
    result = subprocess.run(
        cmd,
        shell=True,
        check=False,
        stderr=subprocess.STDOUT,
        **stdin_subprocess_kwarg,
    )
    return result.returncode
