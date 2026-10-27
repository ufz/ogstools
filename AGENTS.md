# Instructions for coding agents

Read the [coding standard](docs/development/coding-standards/index.md) before you change or review code: the shared checklist, the OGSTools checklist and the patterns.
They are the coding standard. This file adds only what applies to agents.

- Change only what the task needs. Bring existing code in line only where you touch it.
- Propose larger refactorings and design changes. Do not make them on your own.
- Run `pre-commit run --all-files` and fix what it reports.
- Never add a file to a baseline: `[tool.ruff.lint.extend-per-file-ignores]` in `pyproject.toml` or the `exclude:` of a hook in `.pre-commit-config.yaml`.
- Run the tests with `make test`.
- To review a change, follow [Check a change](docs/development/coding-standards/index.md#check-a-change) and [Review a merge request](docs/development/coding-standards/index.md#review-a-merge-request).
