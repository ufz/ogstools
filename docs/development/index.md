
# Development setup

Create a virtual environment, activate it and install required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# enable basic style checks once
pre-commit install
```

:::{important}
Make sure to activate the virtual environment in every new shell session:

```bash
source .venv/bin/activate
```

If you want to automate this checkout [direnv](https://direnv.net).

<h5><i class="fa-brands fa-windows"></i> Windows-specifics</h5>

On Windows the syntax for virtual environment activation is a bit different:

```powershell
# The following may need to be run once. Please check the docs for its consequences:
# https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policiess
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

# Activate via:
.venv\Scripts\Activate.ps1
```

:::

CLI scripts can now be simply run:

```bash
msh2vtu --help
```

## Testing with `tox` and `pytest`

Test environments are created and run with [`tox`](https://tox.wiki).
Tests are executed via [`pytest`](https://docs.pytest.org/en/7.2.x/).

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

Once tox has created an environment (you ran it once) you can use the auto-generating and -reloading web server (Linux / macOS only) for developing the documentation:

```bash
docs/toxin -e docs python docs/server.py
# open http://127.0.0.1:5500 in a web browser
```

For syntax references and extension usage see the following links:

- [Sphinx](https://www.sphinx-doc.org/en/master/)
- [MyST Markdown Parser](https://myst-parser.readthedocs.io/en/latest/)
- [MySt Markdown cheat sheet](https://jupyterbook.org/en/stable/reference/cheatsheet.html#math)
- [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
- [programoutput-extension](https://sphinxcontrib-programoutput.readthedocs.io/en/latest/#)
- [sphinx_design-extension](https://sphinx-design.readthedocs.io/en/furo-theme/index.html)
