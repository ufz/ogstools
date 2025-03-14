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
pip install ogstools[ogs]
```

:::{admonition} Install development version
:class: tip

You can also install the latest development version with `pip` (requires also [`git`](https://git-scm.com)):

```bash
pip install git+https://gitlab.opengeosys.org/ogs/tools/ogstools.git[ogs]
```

:::

#### With self-compiled (custom) OGS

With `pip install ogstools` an OGS wheel (it installs ogs binaries and tools) is installed as well.

If you want to use a self-compiled or custom-built version of OGS instead of the one provided by pip, follow these steps:

1. Install OGSTools without \[ogs\]

```bash
pip install ogstools
```

2. Set the Path to your custom OGS binaries

Specify the location of your self-compiled OGS binaries by setting the `OGS_BIN_PATH` environment variable to the folder that contains the ogs binary and other tools (e.g. vtkdiff).

Recommended: Add to your virtual environment's activate script:

```
export OGS_BIN_PATH="/<absolute_path_to_your_custom_ogs>/bin/"
```

Alternatively, but not recommended: ogs can be on your global `PATH`:

```
export PATH=path_to_your_custom_ogs/bin:$PATH
```

3. Test

```python
import ogstools as ot

ot.status(verbose=True)
```

### First steps

See the following tool descriptions:

```{toctree}
---
maxdepth: 0
glob: 0
---
../auto_user-guide/plot*
../user-guide/*

```
