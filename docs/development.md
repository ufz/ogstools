
# Development setup

Create a virtual environment, activate it and install required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# enable basic style checks once
pre-commit install
```

CLI scripts can now be simply run:

```bash
msh2vtu --help
```

:::{important}
Make sure to activate the virtual environment in every new shell session:

```bash
source .venv/bin/activate
```

If you want to automate this checkout [direnv](https://direnv.net).
:::

## Testing with `tox` and `pytest`

Test environment are created and run with [`tox`](https://tox.wiki).
Tests are executed via [`pytest`](https://docs.pytest.org/en/7.2.x/)

To run the tests:

```bash
tox
```

:::{note}
You can parallelize the tox tests with `tox -p`.
:::

You can view a test coverage report by opening `htmlcov/index.html` in a browser.

You can also run a single test environment with e.g.:

```bash
tox -e py39
```

The following environments are available:

```{command-output} tox list
```

## Create a package

```bash
pyproject-build
```

Packages can then be found in `dist/`.

## Build documentation

```bash
tox -e docs
make html
```

This will create the documentation files in `docs/_build/html`.

For development use the auto-generating and -reloading web server:

```bash
docs/toxin -e docs python docs/server.py
# open http://127.0.0.1:5500 in a web browser
```

- [MyST Markdown Parser](https://myst-parser.readthedocs.io/en/latest/)
- [MySt Markdown cheat sheet](https://jupyterbook.org/en/stable/reference/cheatsheet.html#math)
- [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
- [programoutput-extension](https://sphinxcontrib-programoutput.readthedocs.io/en/latest/#)
