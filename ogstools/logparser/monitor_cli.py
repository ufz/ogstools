import importlib.util
import subprocess
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
parser.add_argument("-j", "--json", help="The path to the logfile.")


def cli() -> int:
    args = parser.parse_args()
    logfile = Path(args.input)
    logfile_abs = logfile.absolute()
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
    )
    return result.returncode
