# Maintainer Guide

# Release procedure

- Make sure there is a complete changelog at `docs/releases` and added to the corresponding `index.md`.
  - Migration guide for breaking changes
  - Links to merge requests
  - Include date of release
- Update pinned dependencies in `pyproject.toml` and `uv.lock`
- In `docs/conf.py` update the binder branch (around line 218) to the new ogs / ogstools tag pair. Update OGSTools version in the [binder repo](https://github.com/bilke/binder-ogs-requirements). See this [example](https://github.com/bilke/binder-ogs-requirements/commit/7a5bdd0847c7ae41f92a943d3b1c7c94ed704eea)
- Change fallback_version in pyproject.toml
- Create a release tag.
- Manual trigger publish job (for PyPI release)
- Wait for the tag pipeline to complete. This will also run a pipeline in [ogs/tools/ogstools-docs](https://gitlab.opengeosys.org/ogs/tools/ogstools-docs)-repo. After finishing check if the updated docs on ogstools.opengeosys.org are shown and the version selector is working.
- Create [GitHub release](https://github.com/ufz/ogstools/releases) -> a Zenodo release is automatically created.
- Update authors on Zenodo release.
- Update Zenodo badge in repo.
- On https://github.com/conda-forge/ogstools-feedstock create a new issue with the title `@conda-forge-admin, please update version`
- Search the codebase for `TODO(ogs-version)` and simplify any now-obsolete ogs-version-dependent code/tests.

# Post-release procedure

- Update ogstools version in OGS (Benchmarks)
- Discourse announcement
