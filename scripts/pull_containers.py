#!/usr/bin/env python
"""Download OGS apptainer containers.

Run once before running the parallel simulation tests:

    python scripts/pull_containers.py
"""

import subprocess

from ogstools.core.execution import Execution


def _pull(url: str) -> None:
    if not url.startswith(("http://", "https://")):
        return
    print(f"Pulling {url} ...")
    subprocess.run(["apptainer", "pull", url], check=True)


def main() -> None:
    _pull(Execution.CONTAINER_SERIAL)
    _pull(Execution.CONTAINER_PARALLEL)
    print("Containers ready.")


if __name__ == "__main__":
    main()
