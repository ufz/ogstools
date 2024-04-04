# Running in a container

```{eval-rst}
.. sectionauthor:: Lars Bilke (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

## Running with Docker

A prebuilt [Docker](https://www.docker.com) image with the latest (nightly build) `ogstools` and all features can be used:

::::{tab-set}
:::{tab-item} Linux/macOS
:sync: unix

```bash
docker run --rm -it -v $PWD:$PWD -w $PWD registry.opengeosys.org/ogs/tools/ogstools/main-3.9:feflow-8.0
# Now in the container:
ogs --version
# test if feflow2ogs is available:
feflow2ogs -h
...
python
# Now in a Python console:
import ogstools.meshplotlib as mpl
...
```

:::
:::{tab-item} Windows
:sync: win
This command works only in the powershell.

```powershell
docker run --rm -it -v ${PWD}:/working_dir -w /working_dir registry.opengeosys.org/ogs/tools/ogstools/main-3.9:feflow-8.0
# Now in the container:
ogs --version
# test if feflow2ogs is available:
feflow2ogs -h
...
python
# Now in a Python console:
import ogstools.meshplotlib as mpl
...
```

`/working_dir` is an arbitrary name of a path into which the current working directory (`PWD`) is mounted in the Docker container.
:::
::::

:::{danger}

Be aware that inside the container you are the `root`-user and if you write files they are owned by `root` too. When you exit the container and you are your regular user on your host again you will have no permissions to access these newly created files! There is no easy solution to this problem but you may consider using Apptainer ([see below](#running-with-apptainer--singularity)).

:::

::::{note}

The container is based on the [devcontainer](../development/index.md#container-specification) with `ogstools` installed:

:::{literalinclude} ../../Dockerfile
:language: Dockerfile
:::

::::

______________________________________________________________________

## Running with Apptainer / Singularity

The prebuilt Docker image can also be run with [Apptainer](https://apptainer.org) (formerly known as *Singularity*):

```bash
apptainer shell docker://registry.opengeosys.org/ogs/tools/ogstools/main-3.9
```

The above command will open a shell in the container. Your home-directory is automatically mounted and you are the same user as outside the container. There will be no file permission issues as with Docker.
