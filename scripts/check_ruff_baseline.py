#!/usr/bin/env python
"""Report entries of the ruff baseline that no longer hide a violation.

The baseline in `[tool.ruff.lint.extend-per-file-ignores]` exempts a file from
the rules it broke when they were enabled. An entry for a rule that the file no
longer breaks would hide new violations, so it has to go.

    python scripts/check_ruff_baseline.py
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def unneeded_codes(path: str, codes: list[str]) -> list[str]:
    """Return the baseline codes of a file that the file no longer violates.

    :param path: The file, relative to the repository root.
    :param codes: The rule codes the baseline exempts the file from.
    :returns: The codes without a violation, all of them for a missing file.
    """
    file = ROOT / path
    if not file.exists():
        return codes
    # The probe name ends like the file name, so permanent per-file-ignores
    # such as "*_cli.py" still match, but the baseline entry does not. It
    # keeps a leading "_", as the docstring rules skip private modules.
    private = file.name.startswith("_") and not file.name.startswith("__")
    probe = file.with_name(("_probe" if private else "probe_") + file.name)
    result = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--exit-zero",
            "--output-format=json",
            "--select",
            ",".join(codes),
            "--stdin-filename",
            str(probe),
            "-",
        ],
        input=file.read_text(),
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    found = {violation["code"] for violation in json.loads(result.stdout)}
    return [code for code in codes if code not in found]


def main() -> int:
    """Report the baseline entries that are no longer needed.

    :returns: 1 if there is such an entry, 0 otherwise.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    baseline = config["tool"]["ruff"]["lint"]["extend-per-file-ignores"]
    stale = {
        path: unneeded
        for path, codes in baseline.items()
        if (unneeded := unneeded_codes(path, codes))
    }
    for path, codes in stale.items():
        sys.stderr.write(
            f"{path}: remove {', '.join(codes)} from the baseline\n"
        )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
