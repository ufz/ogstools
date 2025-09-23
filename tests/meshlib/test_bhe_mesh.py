from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
import pyvista as pv
from shapely import Polygon

from ogstools import Meshes
from ogstools.meshlib.gmsh_BHE import BHE, Groundwater, gen_bhe_mesh


# confinded aquifer: top level of groundwater ends at soil layer transition
def case_1(mesh_type: str) -> Meshes:
    bhe = BHE(x=25, y=30, z_begin=-5, z_end=-50, borehole_radius=0.076)
    distance_bhe_to_refinment_area_x = 10
    distance_bhe_to_refinment_area_y = 8
    return gen_bhe_mesh(
        model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=50, ymax=50),
        layer=[50, 50, 20],
        groundwater=Groundwater(
            begin=-48,
            isolation_layer_id=2,
            upstream=(89, 91),
            downstream=(269, 271),
        ),
        BHE_Array=bhe,
        refinement_area=Polygon.from_bounds(
            xmin=bhe.x - distance_bhe_to_refinment_area_x,
            ymin=bhe.y - distance_bhe_to_refinment_area_y,
            xmax=bhe.x + distance_bhe_to_refinment_area_x,
            ymax=bhe.y + distance_bhe_to_refinment_area_y,
        ),
        meshing_type=mesh_type,
    )


def case_2(mesh_type: str) -> Meshes:
    bhe_array = [
        BHE(x=50, y=40, z_begin=0, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=30, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=50, z_begin=-1, z_end=-52, borehole_radius=0.076),
    ]
    distance_bhe_to_refinment_area_x = 10
    distance_bhe_to_refinment_area_y = 8
    return gen_bhe_mesh(
        model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=100, ymax=70),
        layer=[50, 50, 50],
        groundwater=Groundwater(
            begin=-50,
            isolation_layer_id=2,
            upstream=(359, 1),
            downstream=(179, 181),
        ),
        BHE_Array=bhe_array,
        refinement_area=Polygon.from_bounds(
            xmin=min(bhe.x for bhe in bhe_array)
            - distance_bhe_to_refinment_area_x,
            ymin=min(bhe.y for bhe in bhe_array)
            - distance_bhe_to_refinment_area_y,
            xmax=max(bhe.x for bhe in bhe_array)
            + distance_bhe_to_refinment_area_x,
            ymax=max(bhe.y for bhe in bhe_array)
            + distance_bhe_to_refinment_area_y,
        ),
        meshing_type=mesh_type,
    )


def case_3(mesh_type: str) -> Meshes:
    bhe_array = [
        BHE(x=50, y=25, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=30, z_begin=-1, z_end=-49, borehole_radius=0.076),
        BHE(x=50, y=35, z_begin=-1, z_end=-60, borehole_radius=0.076),
    ]
    distance_bhe_to_refinment_area_x = 10
    distance_bhe_to_refinment_area_y = 8
    return gen_bhe_mesh(
        model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=120, ymax=60),
        layer=[50, 50, 40],
        groundwater=[
            Groundwater(
                begin=-3,
                isolation_layer_id=1,
                upstream=(179, 181),
                downstream=(359, 1),
            ),
            Groundwater(
                begin=-130,
                isolation_layer_id=3,
                upstream=(359, 1),
                downstream=(179, 181),
            ),
        ],
        BHE_Array=bhe_array,
        refinement_area=Polygon.from_bounds(
            xmin=min(bhe.x for bhe in bhe_array)
            - distance_bhe_to_refinment_area_x,
            ymin=min(bhe.y for bhe in bhe_array)
            - distance_bhe_to_refinment_area_y,
            xmax=max(bhe.x for bhe in bhe_array)
            + distance_bhe_to_refinment_area_x,
            ymax=max(bhe.y for bhe in bhe_array)
            + distance_bhe_to_refinment_area_y,
        ),
        meshing_type=mesh_type,
    )


def case_4(mesh_type: str) -> Meshes:
    bhe_array = [
        BHE(x=40, y=15, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=15, z_begin=-1, z_end=-49, borehole_radius=0.076),
        BHE(x=60, y=15, z_begin=-1, z_end=-60, borehole_radius=0.076),
    ]
    distance_bhe_to_refinment_area_x = 10
    distance_bhe_to_refinment_area_y = 4
    return gen_bhe_mesh(
        model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=80, ymax=30),
        layer=[50, 2, 48, 20],
        groundwater=[
            Groundwater(
                begin=-3,
                isolation_layer_id=1,
                upstream=(179, 181),
                downstream=(359, 1),
            )
        ],
        BHE_Array=bhe_array,
        refinement_area=Polygon.from_bounds(
            xmin=min(bhe.x for bhe in bhe_array)
            - distance_bhe_to_refinment_area_x,
            ymin=min(bhe.y for bhe in bhe_array)
            - distance_bhe_to_refinment_area_y,
            xmax=max(bhe.x for bhe in bhe_array)
            + distance_bhe_to_refinment_area_x,
            ymax=max(bhe.y for bhe in bhe_array)
            + distance_bhe_to_refinment_area_y,
        ),
        meshing_type=mesh_type,
        target_z_size_fine=1,
    )


def case_5(mesh_type: str) -> Meshes:
    # test a 12x12 bhe array
    n_bhe_per_row = 12
    dist_bhe = 2.5
    border_distance = 2.5 * dist_bhe
    spacing_bhe = (
        np.linspace(0, (n_bhe_per_row - 1) * dist_bhe, n_bhe_per_row)
        + border_distance
    )

    bhe_array = [
        BHE(x=bhe_x, y=bhe_y, z_begin=0, z_end=-10, borehole_radius=0.076)
        for bhe_x, bhe_y in product(spacing_bhe, repeat=2)
    ]
    distance_bhe_to_refinment_area = 4
    return gen_bhe_mesh(
        model_area=Polygon.from_bounds(
            xmin=0,
            ymin=0,
            xmax=np.max(spacing_bhe) + 2 * border_distance,
            ymax=np.max(spacing_bhe) + 2 * border_distance,
        ),
        layer=[15],
        groundwater=[],
        BHE_Array=bhe_array,
        meshing_type=mesh_type,
        inner_mesh_size=10,
        outer_mesh_size=20,
        refinement_area=Polygon.from_bounds(
            xmin=min(bhe.x for bhe in bhe_array)
            - distance_bhe_to_refinment_area,
            ymin=min(bhe.y for bhe in bhe_array)
            - distance_bhe_to_refinment_area,
            xmax=max(bhe.x for bhe in bhe_array)
            + distance_bhe_to_refinment_area,
            ymax=max(bhe.y for bhe in bhe_array)
            + distance_bhe_to_refinment_area,
        ),
        n_refinement_layers=0,
        target_z_size_coarse=10,
    )


class TestBHE:
    @pytest.mark.parametrize("mesh_type", ["structured", "prism"])
    @pytest.mark.parametrize(
        ("model", "max_id"),
        [(case_1, 4), (case_2, 5), (case_3, 7), (case_4, 7), (case_5, 144)],
    )
    def test_bhe_mesh(self, mesh_type: str, model: Callable, max_id: int):
        meshes = model(mesh_type=mesh_type)

        mesh = meshes.domain()
        assert max(mesh.cell_data["MaterialIDs"]) == max_id
        bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
        soil = mesh.extract_cells_by_type(
            [pv.CellType.HEXAHEDRON, pv.CellType.WEDGE]
        )
        assert np.isin(bhe_line.points, soil.points).all()
