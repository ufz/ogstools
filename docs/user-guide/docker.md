# Running with Docker

```{eval-rst}
.. sectionauthor:: Lars Bilke (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

A prebuilt [Docker](https://www.docker.com) image with the latest (nightly build) `ogstools` and all features can be used:

```bash
docker run --rm -it -v $PWD:$PWD -w $PWD registry.opengeosys.org/ogs/tools/ogstools/main-3.9
# Now in the container:
ogs --version
...
python
# Now in a Python console:
import ogstools.meshplotlib as mpl
...
```

:::{note}

The container is based on the [devcontainer](../development/index.md#container-specification) with `ogstools` installed:

:::{literalinclude} ../../Dockerfile
:language: Dockerfile
:::

:::
