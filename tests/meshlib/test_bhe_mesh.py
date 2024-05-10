import unittest
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv

from ogstools.definitions import TESTS_DIR
from ogstools.meshlib.gmsh_meshing import gen_bhe_mesh, gen_bhe_mesh_gmsh
from ogstools.msh2vtu import msh2vtu

meshpath = TESTS_DIR / "data" / "meshlib"


class BHETest(unittest.TestCase):
    def test_BHE_Mesh(self):
        tmp_dir = Path(mkdtemp())

        ###### Test of gen_bhe_mesh ######
        for mesh_type in [
            "prism",
            "structured",
        ]:  # every test for both supported meshing types
            msh_file = tmp_dir / f"bhe_gw_{mesh_type}.vtu"
            gen_bhe_mesh(
                length=150,
                width=100,
                layer=[50, 50, 50],
                groundwater=(
                    -50,
                    2,
                    "-x",
                ),  # case for confinded aquifer, the top level of groundwater ends at soil layer transition
                BHE_Array=[
                    (50, 40, -1, -60, 0.076),
                    (50, 50, -1, -60, 0.076),
                    (50, 60, -1, -60, 0.076),
                ],
                meshing_type=mesh_type,  # see for loop - test for both supported meshing types
                out_name=msh_file,
            )

            mesh = pv.read(tmp_dir / f"bhe_gw_{mesh_type}_domain.vtu")

            self.assertEqual(
                max(mesh.cell_data["MaterialIDs"]), 5
            )  # mat ID for Layers and BHE's

            # check if all submeshes are present

            top_path = Path(
                tmp_dir / f"bhe_gw_{mesh_type}_physical_group_Top_Surface.vtu"
            )
            bottom_path = Path(
                tmp_dir
                / f"bhe_gw_{mesh_type}_physical_group_Bottom_Surface.vtu"
            )
            gw_path = Path(
                tmp_dir
                / f"bhe_gw_{mesh_type}_physical_group_Groundwater_Inflow_0.vtu"
            )

            self.assertEqual(top_path.is_file(), True)
            self.assertEqual(bottom_path.is_file(), True)
            self.assertEqual(gw_path.is_file(), True)

            bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
            soil = mesh.extract_cells_by_type(
                [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
            )

            self.assertEqual(
                np.isin(bhe_line.points, soil.points).all(), True
            )  # check if all BHE Line Nodes are also in the 3D Soil Domain

            msh_file = tmp_dir / f"bhe_normal_{mesh_type}.vtu"
            gen_bhe_mesh(
                length=150,
                width=100,
                layer=[50, 50, 50],
                groundwater=(-30, 1, "+x"),
                BHE_Array=[
                    (50, 40, -1, -60, 0.076),
                    (50, 50, -1, -60, 0.076),
                    (50, 60, -1, -60, 0.076),
                ],
                meshing_type=mesh_type,
                out_name=msh_file,
            )

            mesh = pv.read(tmp_dir / f"bhe_normal_{mesh_type}_domain.vtu")

            self.assertEqual(
                max(mesh.cell_data["MaterialIDs"]), 6
            )  # mat ID for Layers and BHE's

            # check if all submeshes are present
            top_path = Path(
                tmp_dir
                / f"bhe_normal_{mesh_type}_physical_group_Top_Surface.vtu"
            )
            bottom_path = Path(
                tmp_dir
                / f"bhe_normal_{mesh_type}_physical_group_Bottom_Surface.vtu"
            )
            gw_path = Path(
                tmp_dir
                / f"bhe_normal_{mesh_type}_physical_group_Groundwater_Inflow_0.vtu"
            )

            self.assertEqual(top_path.is_file(), True)
            self.assertEqual(bottom_path.is_file(), True)
            self.assertEqual(gw_path.is_file(), True)

            bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
            soil = mesh.extract_cells_by_type(
                [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
            )

            self.assertEqual(
                np.isin(bhe_line.points, soil.points).all(), True
            )  # check if all BHE Line Nodes are also in the 3D Soil Domain

            msh_file = tmp_dir / f"bhe_two_gw_{mesh_type}.vtu"
            gen_bhe_mesh(
                length=150,
                width=100,
                layer=[50, 50, 50],
                groundwater=[(-48, 1, "-x"), (-130, 3, "+x")],
                BHE_Array=[
                    (50, 40, -1, -60, 0.076),
                    (50, 50, -1, -60, 0.076),
                    (50, 60, -1, -60, 0.076),
                ],
                meshing_type=mesh_type,
                out_name=msh_file,
            )

            mesh = pv.read(tmp_dir / f"bhe_two_gw_{mesh_type}_domain.vtu")

            self.assertEqual(
                max(mesh.cell_data["MaterialIDs"]), 7
            )  # mat ID for Layers and BHE's

            # check if all submeshes are present
            top_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_physical_group_Top_Surface.vtu"
            )
            bottom_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_physical_group_Bottom_Surface.vtu"
            )
            gw_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_physical_group_Groundwater_Inflow_0.vtu"
            )
            gw_1_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_physical_group_Groundwater_Inflow_1.vtu"
            )

            self.assertEqual(top_path.is_file(), True)
            self.assertEqual(bottom_path.is_file(), True)
            self.assertEqual(gw_path.is_file(), True)
            self.assertEqual(gw_1_path.is_file(), True)

            bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
            soil = mesh.extract_cells_by_type(
                [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
            )

            self.assertEqual(
                np.isin(bhe_line.points, soil.points).all(), True
            )  # check if all BHE Line Nodes are also in the 3D Soil Domain

            ###### Test of gen_bhe_mesh_gmsh ######
            msh_file = tmp_dir / f"bhe_gw_{mesh_type}_gmsh.msh"
            gen_bhe_mesh_gmsh(
                length=150,
                width=100,
                layer=[50, 50, 50],
                groundwater=(
                    -50,
                    2,
                    "-x",
                ),  # case for confinded aquifer, the top level of groundwater ends at soil layer transition
                BHE_Array=[
                    (50, 40, -1, -60, 0.076),
                    (50, 50, -1, -60, 0.076),
                    (50, 60, -1, -60, 0.076),
                ],
                meshing_type=mesh_type,  # see for loop - test for both supported meshing types
                out_name=msh_file,
            )

            msh2vtu(
                msh_file,
                output_path=tmp_dir,
                dim=[1, 3],
                reindex=True,
                log_level="ERROR",
            )

            mesh = pv.read(tmp_dir / f"bhe_gw_{mesh_type}_gmsh_domain.vtu")

            self.assertEqual(
                max(mesh.cell_data["MaterialIDs"]), 5
            )  # mat ID for Layers and BHE's

            # check if all submeshes are present
            top_path = Path(
                tmp_dir
                / f"bhe_gw_{mesh_type}_gmsh_physical_group_Top_Surface.vtu"
            )
            bottom_path = Path(
                tmp_dir
                / f"bhe_gw_{mesh_type}_gmsh_physical_group_Bottom_Surface.vtu"
            )
            gw_path = Path(
                tmp_dir
                / f"bhe_gw_{mesh_type}_gmsh_physical_group_Groundwater_Inflow_0.vtu"
            )

            self.assertEqual(top_path.is_file(), True)
            self.assertEqual(bottom_path.is_file(), True)
            self.assertEqual(gw_path.is_file(), True)

            bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
            soil = mesh.extract_cells_by_type(
                [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
            )

            self.assertEqual(
                np.isin(bhe_line.points, soil.points).all(), True
            )  # check if all BHE Line Nodes are also in the 3D Soil Domain

            msh_file = tmp_dir / f"bhe_normal_{mesh_type}_gmsh.msh"
            gen_bhe_mesh_gmsh(
                length=150,
                width=100,
                layer=[50, 50, 50],
                groundwater=(-30, 1, "+x"),
                BHE_Array=[
                    (50, 40, -1, -60, 0.076),
                    (50, 50, -1, -60, 0.076),
                    (50, 60, -1, -60, 0.076),
                ],
                meshing_type=mesh_type,
                out_name=msh_file,
            )

            msh2vtu(
                msh_file,
                output_path=tmp_dir,
                dim=[1, 3],
                reindex=True,
                log_level="ERROR",
            )

            mesh = pv.read(tmp_dir / f"bhe_normal_{mesh_type}_gmsh_domain.vtu")

            self.assertEqual(
                max(mesh.cell_data["MaterialIDs"]), 6
            )  # mat ID for Layers and BHE's

            # check if all submeshes are present
            top_path = Path(
                tmp_dir
                / f"bhe_normal_{mesh_type}_gmsh_physical_group_Top_Surface.vtu"
            )
            bottom_path = Path(
                tmp_dir
                / f"bhe_normal_{mesh_type}_gmsh_physical_group_Bottom_Surface.vtu"
            )
            gw_path = Path(
                tmp_dir
                / f"bhe_normal_{mesh_type}_gmsh_physical_group_Groundwater_Inflow_0.vtu"
            )

            self.assertEqual(top_path.is_file(), True)
            self.assertEqual(bottom_path.is_file(), True)
            self.assertEqual(gw_path.is_file(), True)

            bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
            soil = mesh.extract_cells_by_type(
                [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
            )

            self.assertEqual(
                np.isin(bhe_line.points, soil.points).all(), True
            )  # check if all BHE Line Nodes are also in the 3D Soil Domain

            msh_file = tmp_dir / f"bhe_two_gw_{mesh_type}_gmsh.msh"
            gen_bhe_mesh_gmsh(
                length=150,
                width=100,
                layer=[50, 50, 50],
                groundwater=[(-48, 1, "-x"), (-130, 3, "+x")],
                BHE_Array=[
                    (50, 40, -1, -60, 0.076),
                    (50, 50, -1, -60, 0.076),
                    (50, 60, -1, -60, 0.076),
                ],
                meshing_type=mesh_type,
                out_name=msh_file,
            )

            msh2vtu(
                msh_file,
                output_path=tmp_dir,
                dim=[1, 3],
                reindex=True,
                log_level="ERROR",
            )

            mesh = pv.read(tmp_dir / f"bhe_two_gw_{mesh_type}_gmsh_domain.vtu")

            self.assertEqual(
                max(mesh.cell_data["MaterialIDs"]), 7
            )  # mat ID for Layers and BHE's

            # check if all submeshes are present
            top_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_gmsh_physical_group_Top_Surface.vtu"
            )
            bottom_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_gmsh_physical_group_Bottom_Surface.vtu"
            )
            gw_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_gmsh_physical_group_Groundwater_Inflow_0.vtu"
            )
            gw_1_path = Path(
                tmp_dir
                / f"bhe_two_gw_{mesh_type}_gmsh_physical_group_Groundwater_Inflow_1.vtu"
            )

            self.assertEqual(top_path.is_file(), True)
            self.assertEqual(bottom_path.is_file(), True)
            self.assertEqual(gw_path.is_file(), True)
            self.assertEqual(gw_1_path.is_file(), True)

            bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
            soil = mesh.extract_cells_by_type(
                [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
            )

            self.assertEqual(
                np.isin(bhe_line.points, soil.points).all(), True
            )  # check if all BHE Line Nodes are also in the 3D Soil Domain
