import unittest
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv

from ogstools.meshlib.gmsh_meshing import BHE, Groundwater, gen_bhe_mesh


def case_1(vtu_out_file_path: Path, mesh_type: str) -> list[str]:
    return gen_bhe_mesh(
        length=50,
        width=50,
        layer=[50, 50, 20],
        groundwater=Groundwater(
            -48,
            2,
            "-y",
        ),  # case for confinded aquifer, the top level of groundwater ends at soil layer transition
        BHE_Array=BHE(25, 30, -5, -50, 0.076),
        meshing_type=mesh_type,
        out_name=vtu_out_file_path,
    )


def case_2(vtu_out_file_path: Path, mesh_type: str) -> list[str]:
    return gen_bhe_mesh(
        length=100,
        width=70,
        layer=[50, 50, 50],
        groundwater=Groundwater(-50, 2, "+y"),
        BHE_Array=[
            BHE(50, 40, 0, -60, 0.076),
            BHE(50, 30, -1, -60, 0.076),
            BHE(50, 50, -1, -52, 0.076),
        ],
        meshing_type=mesh_type,
        out_name=vtu_out_file_path,
    )


def case_3(vtu_out_file_path: Path, mesh_type: str) -> list[str]:
    return gen_bhe_mesh(
        length=120,
        width=60,
        layer=[50, 50, 40],
        groundwater=[Groundwater(-3, 1, "-x"), Groundwater(-130, 3, "+x")],
        BHE_Array=[
            BHE(50, 25, -1, -60, 0.076),
            BHE(50, 30, -1, -49, 0.076),
            BHE(50, 35, -1, -60, 0.076),
        ],
        meshing_type=mesh_type,
        out_name=vtu_out_file_path,
    )


def case_4(vtu_out_file_path: Path, mesh_type: str) -> list[str]:
    return gen_bhe_mesh(
        length=80,
        width=30,
        layer=[50, 2, 48, 20],
        groundwater=[Groundwater(-3, 1, "-x")],
        BHE_Array=[
            BHE(40, 15, -1, -60, 0.076),
            BHE(50, 15, -1, -49, 0.076),
            BHE(60, 15, -1, -60, 0.076),
        ],
        meshing_type=mesh_type,
        out_name=vtu_out_file_path,
        target_z_size_fine=1,
    )


class BHETest(unittest.TestCase):
    tmp_dir = Path(mkdtemp())

    def test_BHE_Mesh_structured_case1(self):
        mesh_type = "structured"

        ##### Testcase 1 #####
        vtu_file = self.tmp_dir / "bhe_structured_1.vtu"

        meshes = case_1(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(self.tmp_dir / mesh))

        mesh = pv.read(
            self.tmp_dir / meshes[0]
        )  # f"bhe_gw_{mesh_type}_domain.vtu")

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 4)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

    def test_BHE_Mesh_structured_case2(self):
        ##### Testcase 2 #####
        vtu_file = self.tmp_dir / "bhe_structured_2.vtu"
        mesh_type = "structured"

        meshes = case_2(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(self.tmp_dir / mesh))

        mesh = pv.read(self.tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 5)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

    def test_BHE_Mesh_structured_case3(self):
        ##### Testcase 3 #####
        vtu_file = self.tmp_dir / "bhe_structured_3.vtu"
        mesh_type = "structured"
        meshes = case_3(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(self.tmp_dir / mesh))

        mesh = pv.read(self.tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 7)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

    def test_BHE_Mesh_structured_case4(self):
        ##### Testcase 4 #####
        vtu_file = self.tmp_dir / "bhe_structured_4.vtu"
        mesh_type = "structured"
        meshes = case_4(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(self.tmp_dir / mesh))

        mesh = pv.read(self.tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 7)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

    def test_BHE_Mesh_prism_case1(self):
        tmp_dir = Path(mkdtemp())

        mesh_type = "prism"

        ##### Testcase 1 #####
        vtu_file = tmp_dir / "bhe_prism_1.vtu"

        meshes = case_1(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

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

    def test_BHE_Mesh_prism_case2(self):
        ##### Testcase 2 #####
        vtu_file = self.tmp_dir / "bhe_prism_2.vtu"
        mesh_type = "prism"
        meshes = case_2(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(self.tmp_dir / mesh))

        mesh = pv.read(self.tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 5)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain

    def test_BHE_Mesh_prism_case3(self):
        ##### Testcase 3 #####
        mesh_type = "prism"
        vtu_file = self.tmp_dir / "bhe_prism_3.vtu"
        meshes = case_3(vtu_out_file_path=vtu_file, mesh_type=mesh_type)

        # check if all meshes are present in the directory
        for mesh in meshes:
            self.assertTrue(Path(self.tmp_dir / mesh))

        mesh = pv.read(self.tmp_dir / meshes[0])

        # check mat ID for Layers and BHE's
        self.assertEqual(max(mesh.cell_data["MaterialIDs"]), 7)

        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )

        self.assertTrue(
            np.isin(bhe_line.points, soil.points).all()
        )  # check if all BHE Line Nodes are also in the 3D Soil Domain
