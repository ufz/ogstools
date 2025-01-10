from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv
from parameterized import parameterized

from ogstools.meshlib.gmsh_BHE import BHE, Groundwater, gen_bhe_mesh


# confinded aquifer: top level of groundwater ends at soil layer transition
def case_1(vtu_out_file_path: Path, mesh_type: str) -> list[str]:
    return gen_bhe_mesh(
        length=50,
        width=50,
        layer=[50, 50, 20],
        groundwater=Groundwater(-48, 2, "-y"),
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


class TestBHE:
    tmp_dir = Path(mkdtemp())

    @parameterized.expand(((0, 4), (1, 5), (2, 7), (3, 7)))
    def test_bhe_mesh_structured(self, index: int, max_id: int):
        vtu_file = self.tmp_dir / f"bhe_structured_{index}.vtu"
        model = [case_1, case_2, case_3, case_4][index]
        meshes = model(vtu_out_file_path=vtu_file, mesh_type="structured")
        for mesh in meshes:
            assert Path(self.tmp_dir / mesh).is_file()
        mesh = pv.read(self.tmp_dir / meshes[0])
        assert max(mesh.cell_data["MaterialIDs"]) == max_id
        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )
        assert np.isin(bhe_line.points, soil.points).all()

    @parameterized.expand(((0, 4), (1, 5), (2, 7)))
    def test_bhe_mesh_prism(self, index: int, max_id: int):
        vtu_file = self.tmp_dir / f"bhe_prism_{index}.vtu"
        model = [case_1, case_2, case_3][index]
        meshes = model(vtu_out_file_path=vtu_file, mesh_type="prism")
        for mesh in meshes:
            assert Path(self.tmp_dir / mesh).is_file()
        mesh = pv.read(self.tmp_dir / meshes[0])
        assert max(mesh.cell_data["MaterialIDs"]) == max_id
        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )
        assert np.isin(bhe_line.points, soil.points).all()
