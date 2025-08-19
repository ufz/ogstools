import pyvista as pv
import numpy as np

from pathlib import Path

import ogstools as ot
from ogs import mesh, simulator

gmsh_mesh_name = "rect.msh"
model_dir = Path("ogstools/examples/prj")
prj_path = model_dir / "SimpleLF.prj"


def generateModelMeshes():
    ot.meshlib.rect(
        lengths=(10, 2),
        n_edge_cells=(10, 4),
        n_layers=2,
        structured_grid=True,
        order=1,
        mixed_elements=False,
        jiggle=0.0,
        out_name=Path(gmsh_mesh_name),
    )

    meshes = ot.meshes_from_gmsh(gmsh_mesh_name)
    points_shape = np.shape(meshes["physical_group_left"].points)
    meshes["physical_group_left"].point_data["pressure"] = np.full(
        points_shape[0], 2.99e7
    )
    meshes["physical_group_right"].point_data["pressure"] = np.full(
        points_shape[0], 3e7
    )

    for name, mesh in meshes.items():
        pv.save_meshio(Path(model_dir, name + ".vtu"), mesh)


def executeSimulation():
    arguments = ["", str(prj_path), "-l debug", "-o /tmp/t/r1"]
    simulator.initialize(arguments)

    left_boundary = simulator.getMesh("physical_group_left")
    pressure = np.array(left_boundary.getPointDataArray("pressure", 1))

    # todo extract number of timesteps from project file
    # simulator.executeSimulation()
    for i in range(0, 15):
        # modify left boundary condition values
        if i < 10:
            pressure = np.full(pressure.shape, 2.99e7)
        else:
            pressure = np.full(pressure.shape, 3.01e7)
        left_boundary.setCellDataArray("pressure", pressure, 1)
        simulator.executeTimeStep()

    simulator.finalize()


def cleanupModelMeshes():
    print("todo: rm created meshes")


def main():
    generateModelMeshes()
    executeSimulation()
    cleanupModelMeshes()


if __name__ == "__main__":
    main()
