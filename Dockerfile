FROM registry.opengeosys.org/ogs/tools/ogstools/devcontainer-3.10

RUN --mount=target=/ogstools,type=bind,source=.,readwrite \
     pip install /ogstools[feflow,dev,docs,test,pinned] \
  && pip uninstall gmsh -y

RUN pip install -i https://gmsh.info/python-packages-dev-nox gmsh

ENTRYPOINT bash
