from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

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


def case_5(vtu_out_file_path: Path, mesh_type: str) -> list[str]:
    # test a 12x12 bhe array
    n_bhe_per_row = 12
    dist_bhe = 2.5
    border_distance = 2.5 * dist_bhe
    spacing_bhe = (
        np.linspace(0, (n_bhe_per_row - 1) * dist_bhe, n_bhe_per_row)
        + border_distance
    )

    bhe_array = [
        BHE(bhe_x, bhe_y, 0, -10, 0.076)
        for bhe_x, bhe_y in product(spacing_bhe, repeat=2)
    ]
    return gen_bhe_mesh(
        length=np.max(spacing_bhe) + 2 * border_distance,
        width=np.max(spacing_bhe) + 2 * border_distance,
        layer=[15],
        groundwater=[],
        BHE_Array=bhe_array,
        meshing_type=mesh_type,
        out_name=vtu_out_file_path,
        inner_mesh_size=10,
        outer_mesh_size=20,
        dist_box_x=4,
        dist_box_y=4,
        n_refinement_layers=0,
        target_z_size_coarse=10,
    )


class TestBHE:
    @pytest.mark.parametrize("mesh_type", ["structured", "prism"])
    @pytest.mark.parametrize(
        ("model", "max_id"),
        [(case_1, 4), (case_2, 5), (case_3, 7), (case_4, 7), (case_5, 144)],
    )
    def test_bhe_mesh(
        self, tmp_path, mesh_type: str, model: Callable, max_id: int
    ):
        vtu_file = tmp_path / "bhe_mesh.vtu"
        meshes = model(vtu_out_file_path=vtu_file, mesh_type=mesh_type)
        for mesh in meshes:
            assert Path(tmp_path / mesh).is_file()
        mesh = pv.read(tmp_path / meshes[0])
        assert max(mesh.cell_data["MaterialIDs"]) == max_id
        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )
        assert np.isin(bhe_line.points, soil.points).all()
