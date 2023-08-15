# User Guide

This guide gives an overview over the `ogstools` package.
It explains important features, especially the usage of shipped command-line tools,
and should give new users a good start with this package.

## Getting Started

### Installation

It is a good practice to create a Python virtual environment in which your
Python packages will get installed. Create a virtual environment and activate
it:

::::{tab-set}

:::{tab-item} Linux & macOS
:sync: unix

```bash
python -m venv .venv
source .venv/bin/activate
```

:::

:::{tab-item} Windows
:sync: win

```powershell
python -m venv .venv

# The following may need to be run once. Please check the docs for its consequences:
# https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policiess
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

.venv\Scripts\Activate.ps1
```

:::

::::

:::{important}
Make sure to activate the virtual environment in every new shell session!

:::

The latest release of ogstools can be installed from [PyPI](https://pypi.org/project/ogstools/) using
`pip`:

```bash
pip install ogstools
```

:::{admonition} Install development version
:class: tip

You can also install the latest development version with `pip` (requires also [`git`](https://git-scm.com)):

```bash
pip install git+https://gitlab.opengeosys.org/ogs/tools/ogstools.git@main
```

:::

### First steps

See the following tool descriptions:

```{toctree}
---
maxdepth: 1
glob:
---
*
```
