import unittest
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv

from ogstools.definitions import TESTS_DIR
from ogstools.meshlib.gmsh_meshing import gen_bhe_mesh

meshpath = TESTS_DIR / "data" / "meshlib"


def case_1(msh_file, mesh_type):
    return gen_bhe_mesh(
        length=50,
        width=50,
        layer=[50, 50, 20],
        groundwater=(
            -48,
            2,
            "-y",
        ),  # case for confinded aquifer, the top level of groundwater ends at soil layer transition
        BHE_Array=(25, 30, -5, -50, 0.076),
        meshing_type=mesh_type,
        out_name=msh_file,
    )


def case_2(msh_file, mesh_type):
    return gen_bhe_mesh(
        length=100,
        width=70,
        layer=[50, 50, 50],
        groundwater=(-50, 2, "+y"),
        BHE_Array=[
            (50, 40, -1, -60, 0.076),
            (50, 30, -1, -60, 0.076),
            (50, 50, -1, -52, 0.076),
        ],
        meshing_type=mesh_type,
        out_name=msh_file,
    )


def case_3(msh_file, mesh_type):
    return gen_bhe_mesh(
        length=120,
        width=60,
        layer=[50, 50, 40],
        groundwater=[(-3, 1, "-x"), (-130, 3, "+x")],
        BHE_Array=[
            (50, 25, -1, -60, 0.076),
            (50, 30, -1, -49, 0.076),
            (50, 35, -1, -60, 0.076),
        ],
        meshing_type=mesh_type,
        out_name=msh_file,
    )


def case_4(msh_file, mesh_type):
    return gen_bhe_mesh(
        length=80,
        width=30,
        layer=[50, 2, 48, 20],
        groundwater=[(-3, 1, "-x")],
        BHE_Array=[
            (40, 15, -1, -60, 0.076),
            (50, 15, -1, -49, 0.076),
            (60, 15, -1, -60, 0.076),
        ],
        meshing_type=mesh_type,
        out_name=msh_file,
        target_z_size_fine=1,
    )


class BHETest(unittest.TestCase):
    def test_BHE_Mesh_structured(self):
        tmp_dir = Path(mkdtemp())

        mesh_type = "structured"

        ##### Testcase 1 #####
        msh_file = tmp_dir / "bhe_test_1.vtu"

        meshes = case_1(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])  # f"bhe_gw_{mesh_type}_domain.vtu")

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 4)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

        ##### Testcase 2 #####
        msh_file = tmp_dir / "bhe_test_2.vtu"
        meshes = case_2(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 5)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

        ##### Testcase 3 #####
        msh_file = tmp_dir / "bhe_test_3.vtu"
        meshes = case_3(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 7)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

        ##### Testcase 4 #####
        msh_file = tmp_dir / "bhe_test_4.vtu"
        meshes = case_4(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 7)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

    def test_BHE_Mesh_prism(self):
        tmp_dir = Path(mkdtemp())

        mesh_type = "prism"

        ##### Testcase 1 #####
        msh_file = tmp_dir / "bhe_test_1.vtu"

        meshes = case_1(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])  # f"bhe_gw_{mesh_type}_domain.vtu")

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 4)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

        ##### Testcase 2 #####
        msh_file = tmp_dir / "bhe_test_2.vtu"
        meshes = case_2(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 5)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

        ##### Testcase 3 #####
        msh_file = tmp_dir / "bhe_test_3.vtu"
        meshes = case_3(msh_file=msh_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(tmp_dir / mesh))

        mesh = pv.read(tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 7)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain
