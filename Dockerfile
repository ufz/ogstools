FROM registry.opengeosys.org/ogs/tools/ogstools/devcontainer-3.9

RUN --mount=target=/ogstools,type=bind,source=.,readwrite \
     pip install /ogstools[feflow] \
  && pip uninstall vtk -y

RUN pip install --extra-index-url https://wheels.vtk.org vtk-osmesa \
  && pip install -i https://gmsh.info/python-packages-dev-nox gmsh

ENTRYPOINT bash
