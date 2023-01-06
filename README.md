# ogstools

A collection of Python tools aimed at evolving into a modeling toolchain around OpenGeoSys.

In this project we'll collect existing pre and postprocessing scripts for
OpenGeoSys at first. Over time we'll develop an entire toolchain out of that
collection. Currently we are at the stage of code collection.

## Please donate your existing Python tools for OpenGeoSys

This can be done via merge request or via issue. For both we have set up
templates for the merge request or issue description that will ask you for some
details, e.g., purpose, features, application background of your contribution.

Use the merge request option if you want to contribute some tool that is very
likely useful for many users of OGS.

Use the issue option if you have a bunch of scripts in some repository, out of
which some might be useful for others and some are too special. However, the
merge request option is preferred.

## Minimum requirements for a donation

* The Python scripts must be syntactically correct code.
* There should be a short feature and usage description, e.g., in form of a README file
* Please do not add large amounts of data.
* You as a donator must have the right to donate the code, i.e., you are the
  sole author or all authors agree.

## Development setup

Create a virtual environment, activate it and install required packages:

```bash
python -m venv .venv
source .venv/bin/activate # run this in every new shell session
pip install -e ".[test]"

# enable basic style checks once:
pre-commit install
```

CLI scripts can now be simply run:

```bash
msh2vtu --help
```

### Testing with `tox` and `pytest`

Test environment are created and run with [`tox`](https://tox.wiki).
Tests are executed via [`pytest`](https://docs.pytest.org/en/7.2.x/)

To run the tests:

```bash
tox # parallelize with `tox -p`
```

You can view a coverage report by opening `htmlcov/index.html` in a browser.

You can also run a single test environment with e.g.:

```bash
tox -e py39
```

### Create a package

```bash
pyproject-build
```

Packages can then be found in `dist/`.
