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

If you want to automate this install [direnv](https://direnv.net) and allow it once via `direnv allow` (see `.envrc` configuration file).

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
:name: make-shortcut

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

If you want to link to a gallery page from another page use the following syntax (prefix with `sphx_glr_`, replace directory separator with `_`):

```md
{ref}`meshlib example <sphx_glr_auto_examples_howto_meshlib_plot_meshlib_pyvista_input.py>`
```

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

## Create a package

```bash
pyproject-build
```

Packages can then be found in `dist/`.

# Development in a container with VSCode

A full-featured (including e.g. FEFLOW-functionality), ready-to-use development environment can be used via VSCode's [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) feature. You can do all development-related tasks with it, e.g. testing, previewing documentation or even debugging.

## Requirements

- [Docker](https://www.docker.com)
- [VSCode](https://code.visualstudio.com/) with [Remote Development extension pack](https://code.visualstudio.com/docs/remote/remote-overview) installed

## How-to

- Open the `ogstools`-project in VSCode
- Click the blue button in the left-bottom corner
- Click on `Reopen in Container`

Now you are inside the container. For example, you can open a new terminal (`Terminal` / `New Terminal`) and then run some tests with `pytest` or use the [`Testing`-sidebar](https://code.visualstudio.com/docs/python/testing#_run-tests) to select specific tests to debug.

## Container specification

[`.devcontainer/devcontainer.json`](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/tree/main/.devcontainer/devcontainer.json) (see also [available settings](https://containers.dev)):

:::{literalinclude} ../../.devcontainer/devcontainer.json
:language: json
:::

[`.devcontainer/Dockerfile`](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/tree/main/.devcontainer/Dockerfile):

:::{literalinclude} ../../.devcontainer/Dockerfile
:language: Dockerfile
:::

# Container usage without VSCode

:::{admonition} Advanced topic
:class: caution

If you are familiar with [Docker](https://www.docker.com), you can also start the container manually, e.g. with:

```bash
docker run --rm -it -v $PWD:$PWD -w $PWD registry.opengeosys.org/ogs/tools/ogstools/devcontainer-3.9 /bin/bash
# Inside the container:
make setup_devcontainer
pytest
```

Please also be aware of [file permission issues](../user-guide/docker.md#running-with-docker) when mounting your working directory.

______________________________________________________________________

To prevent these issues we recommend running via [Apptainer](https://apptainer.org):

```bash
apptainer shell docker://registry.opengeosys.org/ogs/tools/ogstools/devcontainer-3.9
# Inside the container:
make setup_devcontainer
pytest
```

:::
