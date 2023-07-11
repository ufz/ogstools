# Development setup

Create a virtual environment, activate it and install required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test,docs]"

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

:::{admonition} Using `make` for shortcuts!
:class: tip

Development-related tasks can also be done with `make` (requires a Bash shell with `make`). The above development setup can also be generated with:

```bash
make setup
```

To get all available `make`-targets run `make help`:

```{program-output} make --no-print-directory -C .. help
```

:::

## Testing with `pytest`

Tests are executed via [`pytest`](https://docs.pytest.org) (shortcut: `make test`):

```bash
pytest [--capture=tee-sys]
```

### Test coverage

The following commands run the tests and create a test coverage report (shortcut: `make coverage`):

```bash
coverage run -m pytest
coverage combine
coverage report --no-skip-covered
coverage html
...
TOTAL                                                                  1698    292    83%
coverage html
Wrote HTML report to htmlcov/index.html
```

You can view a test coverage report by opening `htmlcov/index.html` in a browser.

## Build documentation

For generating the documentation we use [Sphinx](https://www.sphinx-doc.org/en/master/?cmdf=sphinx) (shortcut: `make docs`):

```bash
cd docs
make html
```

This will create the documentation files in `docs/_build/html`.

You can use an auto-generating and -reloading web server (Linux / macOS only) for developing the documentation (shortcut: `make preview`):

```bash
make html -C docs # Generate docs once
python docs/server.py
# Open http://127.0.0.1:5500 in a web browser
# ...
# You can stop the server in the terminal with CTRL-D
```

### Galleries

Python files in `docs/examples` will be added to the Examples-gallery based on [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html).
Please note that text blocks are written [reStructuredText](https://docutils.sourceforge.io/rst.html)-format.
The examples can be downloaded from the final website as Jupyter notebook files.

You can interactively run and debug these files in Visual Studio Code, see the [Python Interactive window documentation](https://code.visualstudio.com/docs/python/jupyter-support-py).

### Further information

For syntax references and extension usage see the following links:

- [Sphinx](https://www.sphinx-doc.org/en/master/)
- [MyST Markdown Parser](https://myst-parser.readthedocs.io/en/latest/)
- [MySt Markdown cheat sheet](https://jupyterbook.org/en/stable/reference/cheatsheet.html#math)
- [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
- [programoutput-extension](https://sphinxcontrib-programoutput.readthedocs.io/en/latest/#)
- [sphinx_design-extension](https://sphinx-design.readthedocs.io/en/furo-theme/index.html)
- [sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html)

## Run checks

We use [pre-commit](https://pre-commit.com) to run various checks (shortcut `make check`):

```
pre-commit run --all-files
```

## Testing with `tox`

To test for different Python version you can use [`tox`](https://tox.wiki).

To run the tests:

```bash
tox
```

:::{note}
You can parallelize the tox tests with `tox -p`.
:::

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
