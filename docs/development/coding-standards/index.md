# Coding standard

These pages say what code, tests and documentation in OGSTools and the other Python projects around OGS must meet in review.
How the standard itself is maintained is in the [maintainer guide](../maintainer.md#coding-standard).

```{toctree}
---
maxdepth: 1
---
Shared rules <shared>
OGSTools <ogstools>
Patterns <patterns>
```

The rules come in two checklists: one bullet per rule, grouped by area.

- [Shared rules](shared.md): rules for OGSTools, ogsworkflows and the other Python projects around OGS.
- [OGSTools](ogstools.md): rules only for OGSTools.
  The [patterns](patterns.md) describe its recurring families of components (file-backed objects, command-line tools, plot functions, …).
  For each family they say what every member has and which variations are allowed.

## How to apply it

- The rules apply to new code and to code that an MR changes.
  Existing code is brought in line when it is touched, not in a separate sweep.
  A sweep is its own MR.
- The checklists list only rules that no tool checks.
  ruff, mypy and the pre-commit hooks check the others: see `[tool.ruff]` (with the banned APIs) and `[tool.mypy]` in `pyproject.toml`, and the local hooks in `.pre-commit-config.yaml`.
  Run `pre-commit install` once; it also installs the hook that checks commit messages.
- Files that violated a rule when it was enabled are listed as a baseline: in `[tool.ruff.lint.extend-per-file-ignores]` and in the `exclude:` of a hook in `.pre-commit-config.yaml`.
  Remove a file from a baseline when you fix it. Never add a file.
- CI runs the pre-commit hooks, ruff included, on the files that an MR changes, and fails on their findings.
  It only reports mypy findings. Run `pre-commit run --all-files` before you ask for review.
- Where no rule applies, follow the conventions of OGS.
  Where OGS has none, follow those of pyvista, then Python best practice.

## Check a change

- `pre-commit run --from-ref origin/main --to-ref HEAD` runs the checks of CI on the commits of a branch.
- For the rules that no tool checks, read the checklists next to the diff, or let an LLM review it with the prompt below.
  Its findings are suggestions; you decide.

```text
Review `git diff origin/main...HEAD` against docs/development/coding-standards/index.md and the pages it links.
Comment only on added or changed lines. For each finding, name the rule and give file:line.
Skip what pre-commit checks.
```

## Review a merge request

- Once CI is green, the author sets `workflow::please review` (see [Contributing](../../contributing.md)) and requests one or two reviewers in the Reviewers field of the MR.
- The reviewer checks the change as in [Check a change](#check-a-change) and comments only on lines that the MR adds or changes.
  A comment names the rule it is based on. It is a proposal: the author may disagree and give a reason.
- The reviewer checks that every new code path has a test.
  In the diff of the MR, GitLab marks the lines that no test runs (coverage from the `tests coverage` CI job).
- The author answers each thread, by a change or a reply. The reviewer resolves the thread.
- When all threads are resolved, the reviewer approves the MR in GitLab.
- A maintainer merges: the author if they are one, otherwise the reviewer.
  The MR is merged without squashing, so the reviewed commits stay as they are.
