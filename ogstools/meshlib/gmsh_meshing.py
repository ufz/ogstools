# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional, Union

import gmsh
import numpy as np

from ogstools.msh2vtu import msh2vtu


def _geo_square(
    geo: gmsh.model.geo,
    lengths: Union[float, list[float]],
    n_edge_cells: Union[int, list[int]],
    structured: bool,
) -> None:
    _lengths = lengths if isinstance(lengths, list) else [lengths] * 2
    _n = n_edge_cells if isinstance(n_edge_cells, list) else [n_edge_cells] * 2
    geo.addPoint(0, 0, 0, tag=1)
    geo.addPoint(_lengths[0], 0, 0, tag=2)
    geo.addPoint(_lengths[0], _lengths[1], 0, tag=3)
    geo.addPoint(0, _lengths[1], 0, tag=4)

    geo.addLine(1, 2, tag=1)
    geo.addLine(2, 3, tag=2)
    geo.addLine(3, 4, tag=3)
    geo.addLine(4, 1, tag=4)

    geo.addCurveLoop([1, 2, 3, 4], tag=1)
    geo.addPlaneSurface([1], tag=1)

    geo.mesh.setTransfiniteCurve(1, _n[0] + 1)
    geo.mesh.setTransfiniteCurve(2, _n[1] + 1)
    geo.mesh.setTransfiniteCurve(3, _n[0] + 1)
    geo.mesh.setTransfiniteCurve(4, _n[1] + 1)

    if structured:
        geo.mesh.setTransfiniteSurface(1)
        geo.mesh.setRecombine(dim=2, tag=1)


def rect(
    lengths: Union[float, list[float]] = 1.0,
    n_edge_cells: Union[int, list[int]] = 1,
    structured_grid: bool = True,
    order: int = 1,
    out_name: Path = Path("unit_square.msh"),
    msh_version: Optional[float] = None,
) -> None:
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
    if msh_version is not None:
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
    gmsh.model.add("unit_square")

    _geo_square(gmsh.model.geo, lengths, n_edge_cells, structured_grid)

    rectangle = gmsh.model.addPhysicalGroup(dim=2, tags=[1], tag=0)
    bottom = gmsh.model.addPhysicalGroup(dim=1, tags=[1])
    right = gmsh.model.addPhysicalGroup(dim=1, tags=[2])
    top = gmsh.model.addPhysicalGroup(dim=1, tags=[3])
    left = gmsh.model.addPhysicalGroup(dim=1, tags=[4])

    gmsh.model.setPhysicalName(dim=2, tag=rectangle, name="unit_square")
    gmsh.model.setPhysicalName(dim=1, tag=bottom, name="bottom")
    gmsh.model.setPhysicalName(dim=1, tag=right, name="right")
    gmsh.model.setPhysicalName(dim=1, tag=top, name="top")
    gmsh.model.setPhysicalName(dim=1, tag=left, name="left")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(out_name))

    gmsh.finalize()


def cuboid(
    lengths: Union[float, list[float]] = 1.0,
    n_edge_cells: Union[int, list[int]] = 1,
    structured_grid: bool = True,
    order: int = 1,
    out_name: Path = Path("unit_cube.msh"),
    msh_version: Optional[float] = None,
) -> None:
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
    if msh_version is not None:
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
    gmsh.model.add("unit_cube")
    _geo_square(gmsh.model.geo, lengths, n_edge_cells, structured_grid)

    dz = lengths if isinstance(lengths, float) else lengths[2]
    nz = n_edge_cells if isinstance(n_edge_cells, int) else n_edge_cells[2]
    newEntities = gmsh.model.geo.extrude(
        dimTags=[(2, 1)], dx=0, dy=0, dz=dz,
        numElements=[nz], recombine=structured_grid  # fmt: skip
    )

    top_tag = newEntities[0][1]
    vol_tag = newEntities[1][1]
    side_tags = [nE[1] for nE in newEntities[2:]]

    surf_tags = [1, top_tag] + side_tags
    surf_names = ["bottom", "top", "front", "right", "back", "left"]
    for surf_tag, surf_name in zip(surf_tags, surf_names):
        side_name = gmsh.model.addPhysicalGroup(dim=2, tags=[surf_tag])
        gmsh.model.setPhysicalName(dim=2, tag=side_name, name=surf_name)

    vol = gmsh.model.addPhysicalGroup(dim=3, tags=[vol_tag], tag=0)
    gmsh.model.setPhysicalName(dim=3, tag=vol, name="volume")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=3)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(out_name))
    gmsh.finalize()


@dataclass(frozen=True)
class Groundwater:
    begin: float = -30
    """ depth of groundwater begin (negative) in m """
    isolation_layer_id: int = 1
    """ number of the groundwater isolation layer (count starts with 0)"""
    flow_direction: str = "+x"
    """ groundwater inflow direction as string - supported '+x', '-x', '-y', '+y' """


@dataclass(frozen=True)
class BHE:
    """(B)orehole (H)eat (E)xchanger"""

    x: float = 50.0
    """x-coordinate of the BHE in m"""
    y: float = 50.0
    """y-coordinate of the BHE in m"""
    z_begin: float = -1.0
    """BHE begin depth (zero or negative) in m"""
    z_end: float = -60.0
    """BHE end depth (zero or negative) in m"""
    borehole_radius: float = 0.076
    """borehole radius in m"""


def gen_bhe_mesh_gmsh(
    length: float,
    width: float,
    layer: Union[float, list[float]],
    groundwater: Union[Groundwater, list[Groundwater]],
    BHE_Array: Union[
        BHE,
        list[BHE],
    ],
    target_z_size_coarse: float = 7.5,
    target_z_size_fine: float = 1.5,
    n_refinement_layers: int = 2,
    meshing_type: str = "structured",
    dist_box_x: float = 5.0,
    dist_box_y: float = 10.0,
    inner_mesh_size: float = 5.0,
    outer_mesh_size: float = 10.0,
    propagation: float = 1.1,
    order: int = 1,
    out_name: Path = Path("bhe_mesh.msh"),
) -> None:
    """
    Create a generic BHE mesh for the Heat_Transport_BHE-Process with additionally submeshes at the top, at the bottom and the groundwater inflow, which is exported in the Gmsh .msh format. For the usage in OGS, a mesh conversion with msh2vtu with dim-Tags [1,3] is needed. The mesh is defined by multiple input parameters. Refinement layers are placed at the BHE-begin, the BHE-end and the groundwater start/end. See detailed description of the parameters below:

    :param length: Length of the model area in m (x-dimension)
    :param width: Width of the model area in m (y-dimension)
    :param layer: List of the soil layer thickness in m
    :param groundwater: List of groundwater layers, where every is specified by a tuple of three entries: [depth of groundwater begin (negative), number of the groundwater isolation layer (count starts with 0), groundwater inflow direction as string - supported '+x', '-x', '-y', '+y'], empty list [] for no groundwater flow
    :param BHE_Array: List of BHEs, where every BHE is specified by a tuple of five floats: [x-coordinate BHE, y-coordinate BHE, BHE begin depth (zero or negative), BHE end depth (negative), borehole radius in m]
    :param target_z_size_coarse: maximum edge length of the elements in m in z-direction, if no refinemnt needed
    :param target_z_size_fine: maximum edge length of the elements in the refinement zone in m in z-direction
    :param n_refinement_layers: number of refinement layers which are evenly set above and beneath the refinemnt depths (see general description above)
    :param meshing_type: 'structured' and 'prism' are supported
    :param dist_box_x: distance in m in x-direction of the refinemnt box according to the BHE's
    :param dist_box_y: distance in m in y-direction of the refinemnt box according to the BHE's
    :param inner_mesh_size: mesh size inside the refinement box in m
    :param outer_mesh_size: mesh size outside of the refinement box in m
    :param propagation: growth of the outer_mesh_size, only supported by meshing_type 'structured'
    :param order: Define the order of the mesh: 1 for linear finite elements / 2 for quadratic finite elements
    :param out_name: name of the exported mesh, must end with .msh

    :returns: a gmsh .msh file
    """

    def _compute_layer_spacing(
        z_min: float,
        z_max: float,
        z_min_id: int,
        id_j: float,
        id_j_max: float,
    ) -> None:
        if z_min_id == 3:  # and id_j==1 - not needed, but logically safer
            if id_j == id_j_max:
                space = (
                    np.abs(z_min - z_max)
                    - n_refinement_layers * target_z_size_fine
                    - space_next_layer_refined
                )
                if space_next_layer_refined == 0:
                    if space <= target_z_size_fine:
                        absolute_height_of_layers.append(
                            np.abs(z_min - z_max)
                        )  # space
                        number_of_layers[len(number_of_layers) - 1].append(
                            math.ceil(
                                np.abs(z_min - z_max) / target_z_size_fine
                            )  # space
                        )
                    else:
                        absolute_height_of_layers.append(
                            n_refinement_layers * target_z_size_fine
                        )
                        number_of_layers[len(number_of_layers) - 1].append(
                            n_refinement_layers
                        )
                        absolute_height_of_layers.append(space)
                        number_of_layers[len(number_of_layers) - 1].append(
                            math.ceil((space) / target_z_size_coarse)
                        )
                else:
                    if space <= target_z_size_fine:
                        absolute_height_of_layers.append(
                            space
                            + n_refinement_layers * target_z_size_fine
                            + space_next_layer_refined
                        )
                        number_of_layers[len(number_of_layers) - 1].append(
                            math.ceil(
                                (
                                    space
                                    + n_refinement_layers * target_z_size_fine
                                    + space_next_layer_refined
                                )
                                / target_z_size_fine
                            )
                        )
                    else:
                        absolute_height_of_layers.append(
                            n_refinement_layers * target_z_size_fine
                        )
                        number_of_layers[len(number_of_layers) - 1].append(
                            n_refinement_layers
                        )
                        absolute_height_of_layers.append(space)
                        number_of_layers[len(number_of_layers) - 1].append(
                            math.ceil(space / target_z_size_coarse)
                        )
                        absolute_height_of_layers.append(
                            space_next_layer_refined
                        )
                        number_of_layers[len(number_of_layers) - 1].append(
                            math.ceil(
                                space_next_layer_refined / target_z_size_fine
                            )
                        )

            else:
                space = (
                    np.abs(z_min - z_max)
                    - n_refinement_layers * target_z_size_fine
                )  # all negative values, because of the extrusion in negative z-direction -> z_min -- End Layer, z_max -- start layer
                if (
                    np.abs(space)
                    <= (n_refinement_layers + 1) * target_z_size_fine
                ):
                    absolute_height_of_layers.append(
                        space + n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space + n_refinement_layers * target_z_size_fine)
                            / target_z_size_fine
                        )
                    )
                else:
                    absolute_height_of_layers.append(
                        n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        n_refinement_layers
                    )
                    absolute_height_of_layers.append(
                        space - n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - n_refinement_layers * target_z_size_fine)
                            / target_z_size_coarse
                        )
                    )
                    absolute_height_of_layers.append(
                        n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        n_refinement_layers
                    )

        elif (
            id_j == 1 and id_j != id_j_max
        ):  # erstes Mesh in jeweiligen Soil-Layer
            space = np.abs(z_min - z_max) - space_last_layer_refined
            if np.abs(space) <= (n_refinement_layers + 1) * target_z_size_fine:
                absolute_height_of_layers.append(
                    space + space_last_layer_refined
                )
                number_of_layers[len(number_of_layers) - 1].append(
                    math.ceil(
                        (space + space_last_layer_refined) / target_z_size_fine
                    )
                )
            else:
                if space_last_layer_refined == 0:
                    absolute_height_of_layers.append(
                        space - n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - n_refinement_layers * target_z_size_fine)
                            / target_z_size_coarse
                        )
                    )
                    absolute_height_of_layers.append(
                        n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        n_refinement_layers
                    )
                else:
                    absolute_height_of_layers.append(space_last_layer_refined)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space_last_layer_refined / target_z_size_fine)
                    )
                    absolute_height_of_layers.append(
                        space - n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - n_refinement_layers * target_z_size_fine)
                            / target_z_size_coarse
                        )
                    )
                    absolute_height_of_layers.append(2 * target_z_size_fine)
                    number_of_layers[len(number_of_layers) - 1].append(
                        n_refinement_layers
                    )

        elif id_j == id_j_max and id_j != 1:
            space = np.abs(z_min - z_max) - space_next_layer_refined
            if space <= (n_refinement_layers + 1) * target_z_size_fine:
                absolute_height_of_layers.append(
                    space + space_next_layer_refined
                )
                number_of_layers[len(number_of_layers) - 1].append(
                    math.ceil(
                        (space + space_next_layer_refined) / target_z_size_fine
                    )
                )
            else:
                if space_next_layer_refined == 0:
                    absolute_height_of_layers.append(
                        n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        n_refinement_layers
                    )
                    absolute_height_of_layers.append(
                        space - n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - n_refinement_layers * target_z_size_fine)
                            / target_z_size_coarse
                        )
                    )
                else:
                    absolute_height_of_layers.append(
                        n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        n_refinement_layers
                    )
                    absolute_height_of_layers.append(
                        space - n_refinement_layers * target_z_size_fine
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - n_refinement_layers * target_z_size_fine)
                            / target_z_size_coarse
                        )
                    )
                    absolute_height_of_layers.append(space_next_layer_refined)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space_next_layer_refined / target_z_size_fine)
                    )

        elif (
            id_j == id_j_max and id_j == 1
        ):  # Layer without a needed depth of BHE or Groundwater
            space = np.abs(z_min - z_max)
            if space_last_layer_refined == 0 and space_next_layer_refined == 0:
                absolute_height_of_layers.append(space)
                number_of_layers[len(number_of_layers) - 1].append(
                    math.ceil(space / target_z_size_coarse)
                )
            elif space_next_layer_refined == 0:
                if space - space_last_layer_refined <= target_z_size_fine:
                    absolute_height_of_layers.append(space)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space / target_z_size_fine)
                    )
                else:
                    absolute_height_of_layers.append(space_last_layer_refined)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space_last_layer_refined / target_z_size_fine)
                    )
                    absolute_height_of_layers.append(
                        space - space_last_layer_refined
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - space_last_layer_refined)
                            / target_z_size_coarse
                        )
                    )
            elif space_last_layer_refined == 0:
                if space - space_next_layer_refined <= target_z_size_fine:
                    absolute_height_of_layers.append(space)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space / target_z_size_fine)
                    )
                else:
                    absolute_height_of_layers.append(
                        space - space_next_layer_refined
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (space - space_next_layer_refined)
                            / target_z_size_coarse
                        )
                    )
                    absolute_height_of_layers.append(space_next_layer_refined)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space_next_layer_refined / target_z_size_fine)
                    )
            else:
                if (
                    space - space_next_layer_refined - space_last_layer_refined
                    <= target_z_size_fine
                ):
                    absolute_height_of_layers.append(space)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space / target_z_size_fine)
                    )
                else:
                    absolute_height_of_layers.append(space_next_layer_refined)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space_next_layer_refined / target_z_size_fine)
                    )
                    absolute_height_of_layers.append(
                        space
                        - space_next_layer_refined
                        - space_last_layer_refined
                    )
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(
                            (
                                space
                                - space_next_layer_refined
                                - space_last_layer_refined
                            )
                            / target_z_size_coarse
                        )
                    )
                    absolute_height_of_layers.append(space_last_layer_refined)
                    number_of_layers[len(number_of_layers) - 1].append(
                        math.ceil(space_last_layer_refined / target_z_size_fine)
                    )

        else:
            space = np.abs(z_min - z_max)
            if space <= (2 * n_refinement_layers + 1) * target_z_size_fine:
                absolute_height_of_layers.append(space)
                number_of_layers[len(number_of_layers) - 1].append(
                    math.ceil(space / target_z_size_fine)
                )
            else:
                absolute_height_of_layers.append(
                    n_refinement_layers * target_z_size_fine
                )
                number_of_layers[len(number_of_layers) - 1].append(
                    n_refinement_layers
                )
                absolute_height_of_layers.append(
                    space - 2 * n_refinement_layers * target_z_size_fine
                )
                number_of_layers[len(number_of_layers) - 1].append(
                    math.ceil(
                        (space - 2 * n_refinement_layers * target_z_size_fine)
                        / target_z_size_coarse
                    )
                )
                absolute_height_of_layers.append(
                    n_refinement_layers * target_z_size_fine
                )
                number_of_layers[len(number_of_layers) - 1].append(
                    n_refinement_layers
                )

    def _flatten_concatenation(
        matrix: list[list[float]],
    ) -> (
        list
    ):  # to flat a list, seems not so easy with a ready to use function --> this code is from https://realpython.com/python-flatten-list/
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    def _mesh_structured() -> None:
        point_id = 1
        line_id = 1
        curve_loop_id = 1
        surface_id = 1

        # define all points
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, tag=point_id)  # 1
        point_id += 1
        gmsh.model.geo.addPoint(x_min, 0.0, 0.0, tag=point_id)  # 2
        point_id += 1
        gmsh.model.geo.addPoint(x_max, 0.0, 0.0, tag=point_id)  # 3
        point_id += 1
        gmsh.model.geo.addPoint(length, 0.0, 0.0, tag=point_id)  # 4
        point_id += 1

        gmsh.model.geo.addPoint(0.0, y_min, 0.0, tag=point_id)  # 5
        point_id += 1
        gmsh.model.geo.addPoint(x_min, y_min, 0.0, tag=point_id)  # 6
        point_id += 1
        gmsh.model.geo.addPoint(x_max, y_min, 0.0, tag=point_id)  # 7
        point_id += 1
        gmsh.model.geo.addPoint(length, y_min, 0.0, tag=point_id)  # 8
        point_id += 1

        gmsh.model.geo.addPoint(0.0, y_max, 0.0, tag=point_id)  # 9
        point_id += 1
        gmsh.model.geo.addPoint(x_min, y_max, 0.0, tag=point_id)  # 10
        point_id += 1
        gmsh.model.geo.addPoint(x_max, y_max, 0.0, tag=point_id)  # 11
        point_id += 1
        gmsh.model.geo.addPoint(length, y_max, 0.0, tag=point_id)  # 12
        point_id += 1

        gmsh.model.geo.addPoint(0.0, width, 0.0, tag=point_id)  # 13
        point_id += 1
        gmsh.model.geo.addPoint(x_min, width, 0.0, tag=point_id)  # 14
        point_id += 1
        gmsh.model.geo.addPoint(x_max, width, 0.0, tag=point_id)  # 15
        point_id += 1
        gmsh.model.geo.addPoint(length, width, 0.0, tag=point_id)  # 16
        point_id += 1

        # define all lines in x-direction

        gmsh.model.geo.addLine(1, 2, line_id)  # 1
        line_id += 1
        gmsh.model.geo.addLine(2, 3, line_id)  # 2
        line_id += 1
        gmsh.model.geo.addLine(3, 4, line_id)  # 3
        line_id += 1

        gmsh.model.geo.addLine(5, 6, line_id)  # 4
        line_id += 1
        gmsh.model.geo.addLine(6, 7, line_id)  # 5
        line_id += 1
        gmsh.model.geo.addLine(7, 8, line_id)  # 6
        line_id += 1

        gmsh.model.geo.addLine(9, 10, line_id)  # 7
        line_id += 1
        gmsh.model.geo.addLine(10, 11, line_id)  # 8
        line_id += 1
        gmsh.model.geo.addLine(11, 12, line_id)  # 9
        line_id += 1

        gmsh.model.geo.addLine(13, 14, line_id)  # 10
        line_id += 1
        gmsh.model.geo.addLine(14, 15, line_id)  # 11
        line_id += 1
        gmsh.model.geo.addLine(15, 16, line_id)  # 12
        line_id += 1

        # define all lines in y-direction

        gmsh.model.geo.addLine(1, 5, line_id)  # 13
        line_id += 1
        gmsh.model.geo.addLine(2, 6, line_id)  # 14
        line_id += 1
        gmsh.model.geo.addLine(3, 7, line_id)  # 15
        line_id += 1
        gmsh.model.geo.addLine(4, 8, line_id)  # 16
        line_id += 1

        gmsh.model.geo.addLine(5, 9, line_id)  # 17
        line_id += 1
        gmsh.model.geo.addLine(6, 10, line_id)  # 18
        line_id += 1
        gmsh.model.geo.addLine(7, 11, line_id)  # 19
        line_id += 1
        gmsh.model.geo.addLine(8, 12, line_id)  # 20
        line_id += 1

        gmsh.model.geo.addLine(9, 13, line_id)  # 21
        line_id += 1
        gmsh.model.geo.addLine(10, 14, line_id)  # 22
        line_id += 1
        gmsh.model.geo.addLine(11, 15, line_id)  # 23
        line_id += 1
        gmsh.model.geo.addLine(12, 16, line_id)  # 24
        line_id += 1

        # add surfaces

        # first column
        gmsh.model.geo.addCurveLoop([1, 14, -4, -13], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 1
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        gmsh.model.geo.addCurveLoop([2, 15, -5, -14], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 2
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        gmsh.model.geo.addCurveLoop([3, 16, -6, -15], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 3
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        # second column
        gmsh.model.geo.addCurveLoop([4, 18, -7, -17], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 4
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        gmsh.model.geo.addCurveLoop([5, 19, -8, -18], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 5
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        gmsh.model.geo.addCurveLoop([6, 20, -9, -19], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 6
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        # third column
        gmsh.model.geo.addCurveLoop([7, 22, -10, -21], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 7
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        gmsh.model.geo.addCurveLoop([8, 23, -11, -22], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 8
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        gmsh.model.geo.addCurveLoop([9, 24, -12, -23], curve_loop_id)
        gmsh.model.geo.addPlaneSurface([curve_loop_id], surface_id)  # 9
        gmsh.model.geo.synchronize()
        curve_loop_id += 1
        surface_id += 1

        d = point_id  # first tag number of a bhe
        bhe_top_nodes = []

        # insert BHE's in the model
        for i in range(len(BHE_array)):
            X = BHE_array[i].x
            Y = BHE_array[i].y
            Z = 0
            delta = (
                alpha * BHE_array[i].borehole_radius
            )  # meshsize at BHE and distance of the surrounding optimal mesh points

            gmsh.model.geo.addPoint(
                X, Y, Z, delta, d
            )  # Diersch et al. 2011 Part 2

            gmsh.model.geo.addPoint(X, Y - delta, Z, delta, d + 1)
            gmsh.model.geo.addPoint(X, Y + delta, Z, delta, d + 2)

            gmsh.model.geo.addPoint(
                X + 0.866 * delta, Y + 0.5 * delta, Z, delta, d + 3
            )
            gmsh.model.geo.addPoint(
                X - 0.866 * delta, Y + 0.5 * delta, Z, delta, d + 4
            )

            gmsh.model.geo.addPoint(
                X + 0.866 * delta, Y - 0.5 * delta, Z, delta, d + 5
            )
            gmsh.model.geo.addPoint(
                X - 0.866 * delta, Y - 0.5 * delta, Z, delta, d + 6
            )

            if BHE_array[i].z_begin != 0:
                gmsh.model.geo.addPoint(
                    X, Y, BHE_array[i].z_begin, delta, d + 7
                )
                bhe_top_nodes.append(d + 7)
            else:
                bhe_top_nodes.append(d)

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.embed(
                0, [d, d + 1, d + 2, d + 3, d + 4, d + 5, d + 6], 2, 5
            )

            d = d + 8

        # Extrude the surface mesh according to the previously evaluated structure
        volumes_list_for_layers = []
        boundary_groundwater_list_plusx = []
        boundary_groundwater_list_minusx = []
        boundary_groundwater_list_plusy = []
        boundary_groundwater_list_minusy = []
        top_surface = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        surface_list = [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
        ]
        for j in range(0, len(number_of_layers)):
            # spacing of the each layer must be evaluated according to the implementation of the bhe
            R = gmsh.model.geo.extrude(
                surface_list,
                0,
                0,
                -depth_of_extrusion[j],
                number_of_layers[j],
                cummulative_height_of_layers[j],
                True,
            )  # soil 1
            # list of volume numbers and new bottom surfaces, which were extruded by the five surfaces
            volume_list = []
            surface_list = []

            # i=1
            for i in range(0, 9):
                volume_list.append(R[1 + i * 6][1])
                surface_list.append(R[i * 6])

            # export of the outer surfaces according their orientation

            boundary_groundwater_list_plusx.append(
                R[5 + 3 * 6][1]
            )  # R[5+3*6][1]
            boundary_groundwater_list_plusx.append(R[5 + 6 * 6][1])
            boundary_groundwater_list_plusx.append(R[5][1])

            boundary_groundwater_list_minusx.append(R[33][1])  # -x 33, 15, 51
            boundary_groundwater_list_minusx.append(
                R[15][1]
            )  # -y 40, 46, 52 #+y 2, 8, 14
            boundary_groundwater_list_minusx.append(R[51][1])

            boundary_groundwater_list_plusy.append(R[2][1])
            boundary_groundwater_list_plusy.append(R[8][1])
            boundary_groundwater_list_plusy.append(R[14][1])

            boundary_groundwater_list_minusy.append(R[40][1])
            boundary_groundwater_list_minusy.append(R[46][1])
            boundary_groundwater_list_minusy.append(R[52][1])

            volumes_list_for_layers.append(volume_list)

        k = 0
        BHE = []
        for i in range(0, len(BHE_array)):
            G = gmsh.model.geo.extrude(
                [(0, bhe_top_nodes[i])],
                0,
                0,
                BHE_array[i].z_end - BHE_array[i].z_begin,
                BHE_extrusion_layers[i],
                BHE_extrusion_depths[i],
                True,
            )
            BHE.append(G[1][1])

        gmsh.model.geo.synchronize()

        for i in range(0, len(number_of_layers)):
            gmsh.model.addPhysicalGroup(3, volumes_list_for_layers[i], i)
            k = i

        for i in range(0, len(BHE_array)):
            gmsh.model.addPhysicalGroup(1, [BHE[i]], k + 1)
            k += 1

        gmsh.model.addPhysicalGroup(2, top_surface, k + 1, name="Top_Surface")
        gmsh.model.addPhysicalGroup(
            2,
            np.array(surface_list)[:, 1].tolist(),
            k + 2,
            name="Bottom_Surface",
        )

        counter_for_gw_start_at_soil_layer = 0
        for i in range(0, len(groundwater_list)):
            # add loop for different groundwater flow directions
            if groundwater_list[i][4] == "+x":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusx[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        tag=k + 3,
                        name=f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusx[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                + 1
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                + 1
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        tag=k + 3,
                        name=f"Groundwater_Inflow_{i}",
                    )
            elif groundwater_list[i][4] == "-x":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    # gmsh.model.addPhysicalGroup(2,boundary_groundwater_list_minusx,k+3,f'Groundwater_Inflow_{i}')
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusx[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusx[
                            3
                            * (
                                groundwater_list[i][0]
                                + 1
                                + i
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + 1
                                + i
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
            elif groundwater_list[i][4] == "+y":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusy[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        tag=k + 3,
                        name=f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusy[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                + 1
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                + 1
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        tag=k + 3,
                        name=f"Groundwater_Inflow_{i}",
                    )
            elif groundwater_list[i][4] == "+x":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusy[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        tag=k + 3,
                        name=f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusy[
                            3
                            * (
                                groundwater_list[i][0]
                                + i
                                + 1
                                - counter_for_gw_start_at_soil_layer
                            ) : 3
                            * (
                                groundwater_list[i][3]
                                + i
                                + 1
                                - counter_for_gw_start_at_soil_layer
                            )
                        ],
                        tag=k + 3,
                        name=f"Groundwater_Inflow_{i}",
                    )
            k += 1

        # Sizing Functions and Transfinite Algorithm for Hexahedron meshing in wanted zones

        # inner square 5
        gmsh.model.mesh.setTransfiniteCurve(
            5, math.ceil((x_max - x_min) / minus_y_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            8, math.ceil((x_max - x_min) / minus_y_mesh_size) + 1
        )

        gmsh.model.mesh.setTransfiniteCurve(
            17, math.ceil((y_max - y_min) / minus_x_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            18, math.ceil((y_max - y_min) / minus_x_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            19, math.ceil((y_max - y_min) / plus_x_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            20, math.ceil((y_max - y_min) / plus_x_mesh_size) + 1
        )

        # outer squares 1,2,3,7,8,9
        gmsh.model.mesh.setTransfiniteCurve(
            13, math.ceil((y_min) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            14, math.ceil((y_min) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            15, math.ceil((y_min) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            16, math.ceil((y_min) / outer_mesh_size_inner) + 1
        )

        gmsh.model.mesh.setTransfiniteCurve(
            21, math.ceil((width - y_max) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            22, math.ceil((width - y_max) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            23, math.ceil((width - y_max) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            24, math.ceil((width - y_max) / outer_mesh_size_inner) + 1
        )

        gmsh.model.mesh.setTransfiniteCurve(
            2, math.ceil((x_max - x_min) / outer_mesh_size_inner) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            11, math.ceil((x_max - x_min) / outer_mesh_size_inner) + 1
        )

        # rectangular squares bgw
        gmsh.model.mesh.setTransfiniteCurve(
            1, math.ceil((x_min) / outer_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            4, math.ceil((x_min) / outer_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            7, math.ceil((x_min) / outer_mesh_size) + 1
        )
        gmsh.model.mesh.setTransfiniteCurve(
            10, math.ceil((x_min) / outer_mesh_size) + 1
        )

        gmsh.model.mesh.setTransfiniteCurve(
            3,
            math.ceil((length - x_max) / outer_mesh_size) + 1,
            "Progression",
            propagation,
        )
        gmsh.model.mesh.setTransfiniteCurve(
            6,
            math.ceil((length - x_max) / outer_mesh_size) + 1,
            "Progression",
            propagation,
        )
        gmsh.model.mesh.setTransfiniteCurve(
            9,
            math.ceil((length - x_max) / outer_mesh_size) + 1,
            "Progression",
            propagation,
        )
        gmsh.model.mesh.setTransfiniteCurve(
            12,
            math.ceil((length - x_max) / outer_mesh_size) + 1,
            "Progression",
            propagation,
        )

        gmsh.model.mesh.setTransfiniteSurface(
            1, arrangement="Left", cornerTags=[1, 2, 5, 6]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            3, arrangement="Left", cornerTags=[3, 4, 7, 8]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            4, arrangement="Left", cornerTags=[5, 6, 9, 10]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            6, arrangement="Left", cornerTags=[7, 8, 11, 12]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            7, arrangement="Left", cornerTags=[9, 10, 13, 14]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            9, arrangement="Left", cornerTags=[11, 12, 15, 16]
        )

        gmsh.model.mesh.setRecombine(2, 1)
        gmsh.model.mesh.setRecombine(2, 3)
        gmsh.model.mesh.setRecombine(2, 4)
        gmsh.model.mesh.setRecombine(2, 6)
        gmsh.model.mesh.setRecombine(2, 7)
        gmsh.model.mesh.setRecombine(2, 9)
        gmsh.model.mesh.recombine()

    def _mesh_prism() -> None:
        point_id = 1
        line_id = 1

        # define the outer boundaries square
        gmsh.model.geo.addPoint(
            0.0, 0.0, 0.0, outer_mesh_size, tag=point_id
        )  # 1
        point_id += 1
        gmsh.model.geo.addPoint(
            length, 0.0, 0.0, outer_mesh_size, tag=point_id
        )  # 2
        point_id += 1
        gmsh.model.geo.addPoint(
            length, width, 0.0, outer_mesh_size, tag=point_id
        )  # 3
        point_id += 1
        gmsh.model.geo.addPoint(
            0.0, width, 0.0, outer_mesh_size, tag=point_id
        )  # 4
        point_id += 1

        gmsh.model.geo.addLine(1, 2, line_id)  # 1
        line_id += 1
        gmsh.model.geo.addLine(2, 3, line_id)  # 2
        line_id += 1
        gmsh.model.geo.addLine(3, 4, line_id)  # 3
        line_id += 1
        gmsh.model.geo.addLine(4, 1, line_id)  # 4
        line_id += 1

        # inner points
        gmsh.model.geo.addPoint(
            x_min, y_min, 0.0, minus_x_mesh_size, tag=point_id
        )  # 5
        point_id += 1
        gmsh.model.geo.addPoint(
            x_max, y_min, 0.0, plus_x_mesh_size, tag=point_id
        )  # 6
        point_id += 1
        gmsh.model.geo.addPoint(
            x_max, y_max, 0.0, plus_x_mesh_size, tag=point_id
        )  # 7
        point_id += 1
        gmsh.model.geo.addPoint(
            x_min, y_max, 0.0, minus_x_mesh_size, tag=point_id
        )  # 8
        point_id += 1

        gmsh.model.geo.addLine(5, 6, line_id)  # 5
        line_id += 1
        gmsh.model.geo.addLine(6, 7, line_id)  # 6
        line_id += 1
        gmsh.model.geo.addLine(7, 8, line_id)  # 7
        line_id += 1
        gmsh.model.geo.addLine(8, 5, line_id)  # 8
        line_id += 1

        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.embed(
            1, [5, 6, 7, 8], 2, 1
        )  # embed the four lines of the inner sizing box

        d = point_id  # first tag number of a bhe

        bhe_top_nodes: list = []

        # insert BHE's in the model
        for i in range(len(BHE_array)):
            X = BHE_array[i].x
            Y = BHE_array[i].y
            Z = 0
            delta = (
                alpha * BHE_array[i].borehole_radius
            )  # meshsize at BHE and distance of the surrounding optimal mesh points

            gmsh.model.geo.addPoint(
                X, Y, Z, delta, d
            )  # Diersch et al. 2011 Part 2

            gmsh.model.geo.addPoint(X, Y - delta, Z, delta, d + 1)
            gmsh.model.geo.addPoint(X, Y + delta, Z, delta, d + 2)

            gmsh.model.geo.addPoint(
                X + 0.866 * delta, Y + 0.5 * delta, Z, delta, d + 3
            )
            gmsh.model.geo.addPoint(
                X - 0.866 * delta, Y + 0.5 * delta, Z, delta, d + 4
            )

            gmsh.model.geo.addPoint(
                X + 0.866 * delta, Y - 0.5 * delta, Z, delta, d + 5
            )
            gmsh.model.geo.addPoint(
                X - 0.866 * delta, Y - 0.5 * delta, Z, delta, d + 6
            )

            if BHE_array[i].z_begin != 0:
                gmsh.model.geo.addPoint(X, Y, BHE_array[i].z_begin, tag=d + 7)
                bhe_top_nodes.append(d + 7)
            else:
                bhe_top_nodes.append(d)

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.embed(
                0, [d, d + 1, d + 2, d + 3, d + 4, d + 5, d + 6], 2, 1
            )

            d = d + 8

        point_id = d
        # Extrude the surface mesh according to the previously evaluated structure
        volumes_list_for_layers = []
        top_surface = [1]

        boundary_groundwater_list_plusx = []
        boundary_groundwater_list_minusx = []
        boundary_groundwater_list_plusy = []
        boundary_groundwater_list_minusy = []

        surface_list = [(2, 1)]
        for j in range(0, len(number_of_layers)):
            # spacing of the each layer must be evaluated according to the implementation of the bhe
            R = gmsh.model.geo.extrude(
                surface_list,
                0,
                0,
                -depth_of_extrusion[j],
                number_of_layers[j],
                cummulative_height_of_layers[j],
                True,
            )  # soil 1

            # list of volume numbers and new bottom surfaces, which were extruded by the five surfaces
            volume_list = []
            surface_list = []

            volume_list.append(R[1][1])
            surface_list.append(R[0])

            boundary_groundwater_list_plusx.append(R[5][1])
            boundary_groundwater_list_minusx.append(R[3][1])
            boundary_groundwater_list_plusy.append(R[2][1])
            boundary_groundwater_list_minusy.append(R[4][1])

            volumes_list_for_layers.append(volume_list)

        BHE = []

        for i in range(0, len(BHE_array)):
            G = gmsh.model.geo.extrude(
                [(0, bhe_top_nodes[i])],
                0,
                0,
                BHE_array[i].z_end - BHE_array[i].z_begin,
                BHE_extrusion_layers[i],
                BHE_extrusion_depths[i],
            )
            BHE.append(G[1][1])

        gmsh.model.geo.synchronize()
        k = 0

        for i in range(
            0, len(number_of_layers)
        ):  # len(layer) is right, but len(number_of_layers) for testing only
            gmsh.model.addPhysicalGroup(3, volumes_list_for_layers[i], i)
            k = i

        for i in range(0, len(BHE_array)):
            gmsh.model.addPhysicalGroup(1, [BHE[i]], k + 1)
            k += 1

        gmsh.model.addPhysicalGroup(2, top_surface, k + 1, name="Top_Surface")
        gmsh.model.addPhysicalGroup(
            2,
            np.array(surface_list)[:, 1].tolist(),
            k + 2,
            name="Bottom_Surface",
        )

        counter_for_gw_start_at_soil_layer = 0
        for i in range(0, len(groundwater_list)):
            # add loop for different groundwater flow directions
            if groundwater_list[i][4] == "+x":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusx[
                            groundwater_list[i][0]
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusx[
                            groundwater_list[i][0]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
            elif groundwater_list[i][4] == "-x":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusx[
                            groundwater_list[i][0]
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusx[
                            groundwater_list[i][0]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
            elif groundwater_list[i][4] == "+y":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusy[
                            groundwater_list[i][0]
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_plusy[
                            groundwater_list[i][0]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
            elif groundwater_list[i][4] == "-y":
                if np.abs(groundwater_list[i][2]) in np.cumsum(layer):
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusy[
                            groundwater_list[i][0]
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
                    counter_for_gw_start_at_soil_layer += 1
                else:
                    gmsh.model.addPhysicalGroup(
                        2,
                        boundary_groundwater_list_minusy[
                            groundwater_list[i][0]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer : groundwater_list[
                                i
                            ][
                                3
                            ]
                            + 1
                            + i
                            - counter_for_gw_start_at_soil_layer
                        ],
                        k + 3,
                        f"Groundwater_Inflow_{i}",
                    )
            k += 1

        # gmsh.model.addPhysicalGroup(2,boundary_groundwater_list[start_groundwater+1:groundwater[1]+1],k+3,'Groundwater_Inflow')

    layer = layer if isinstance(layer, list) else [layer]

    groundwaters: list[Groundwater] = (
        [groundwater] if isinstance(groundwater, Groundwater) else groundwater
    )

    BHE_Array = [BHE_Array] if isinstance(BHE_Array, BHE) else BHE_Array

    # detect the soil layer, in which the groundwater flow starts
    groundwater_list: list = []
    for g in range(0, len(groundwaters)):
        start_groundwater = -1000
        icl: float = (
            -1
        )  # Index for critical layer structure, 0 - not critical, 1 - top critical, 2 - bottom critical, 3 - groundwater at layer transition
        # needed_medias_in_ogs=len(layer)+1
        for i in range(0, len(layer)):
            if (
                np.abs(groundwaters[g].begin) < np.sum(layer[: i + 1])
                and start_groundwater == -1000
            ):
                start_groundwater = i

                if (  # previous elif, one semantic block of different cases -> switch to if, because of ruff error
                    np.abs(groundwaters[g].begin)
                    - np.sum(layer[:start_groundwater])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - GW')
                    icl = 1
                    if np.abs(groundwaters[g].begin) == np.sum(
                        layer[:start_groundwater]
                    ):  # beginning of groundwater at a transition of two soil layers - special case
                        icl = 3
                        # needed_medias_in_ogs=len(layer) #needed_extrusions=len(layer)
                    elif (
                        np.sum(layer[: start_groundwater + 1])
                        - np.abs(groundwaters[g].begin)
                        < n_refinement_layers * target_z_size_fine
                    ):
                        icl = (
                            1.2  # for layers, which are top and bottom critical
                        )
                elif (
                    np.sum(layer[: start_groundwater + 1])
                    - np.abs(groundwaters[g].begin)
                    < n_refinement_layers * target_z_size_fine
                    and icl != 1.2
                ):
                    # print('critical at the bottom of the soil layer-GW')
                    icl = 2
                else:
                    icl = 0
        groundwater_list.append(
            [
                start_groundwater,
                icl,
                groundwaters[g].begin,
                groundwaters[g].isolation_layer_id,
                groundwaters[g].flow_direction,
            ]
        )

    ### Start of the algorithm ###
    BHE_array = np.array(BHE_Array)

    # detect the soil layer, in which the BHE starts - for the moment only for detection
    i = 0
    BHE_to_soil = np.zeros(
        shape=(len(BHE_array), 5), dtype=np.int8
    )  # [:,0] - index of BHE; Array, in welcher Schicht die jeweilige BHE anfängt [:,1] endet [:,3] und wo ein kritischer Übergang folgt [:,2] für BHE_Begin und [:,4] für BHE_End mit 0 - not critical, 1 - top critical, 2 - bottom critical, 3 - bhe at layer transition

    for j in range(0, len(BHE_array)):
        for i in range(0, len(layer)):
            if np.abs(BHE_array[j].z_begin) < np.sum(layer[: i + 1]) and np.abs(
                BHE_array[j].z_begin
            ) >= np.sum(
                layer[:i]
            ):  # Auswertung für BHE_Beginn
                BHE_to_soil[j, 0] = j
                BHE_to_soil[j, 1] = i
                if (
                    np.abs(BHE_array[j].z_end) - np.abs(BHE_array[j].z_begin)
                    <= n_refinement_layers * target_z_size_fine
                ):  # pragma: no cover
                    msg = "BHE to short, must be longer than n_refinement_layers * target_z_size_fine!"
                    raise Exception(msg)
                if (  # previous elif, one semantic block of different cases -> switch to if, because of ruff error
                    np.abs(BHE_array[j].z_begin) - np.sum(layer[:i])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - BHE  %d'%j)
                    BHE_to_soil[j, 2] = 1
                    if np.abs(BHE_array[j].z_begin) == np.sum(
                        layer[:i]
                    ):  # beginning a transition of two soil layers - special case
                        BHE_to_soil[j, 2] = 3
                elif (
                    np.sum(layer[: i + 1]) - np.abs(BHE_array[j].z_begin)
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('critical at the bottom of the soil layer - BHE %d'%j)
                    BHE_to_soil[j, 2] = 2
                else:
                    BHE_to_soil[j, 2] = 0

    # detect the soil layer, in which the BHE ends
    for j in range(0, len(BHE_array)):
        for i in range(0, len(layer)):
            if np.abs(BHE_array[j].z_end) < np.sum(layer[: i + 1]) and np.abs(
                BHE_array[j].z_end
            ) >= np.sum(layer[:i]):
                BHE_to_soil[j, 3] = i
                if (
                    np.abs(BHE_array[j].z_end) - np.sum(layer[:i])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - BHE  %d'%j)
                    BHE_to_soil[j, 4] = 1
                    if np.abs(BHE_array[j].z_end) == np.sum(
                        layer[:i]
                    ):  # beginning at a transition of two soil layers - special case
                        BHE_to_soil[j, 4] = 3

                    elif (
                        np.sum(layer[: i + 1]) - np.abs(BHE_array[j].z_end)
                        < n_refinement_layers * target_z_size_fine
                    ):
                        BHE_to_soil[
                            j, 4
                        ] = 1.2  # for layers, which are top and bottom critical
                elif (
                    np.sum(layer[: i + 1]) - np.abs(BHE_array[j].z_end)
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('critical at the bottom of the soil layer - BHE %d'%j)
                    BHE_to_soil[j, 4] = 2
                else:
                    BHE_to_soil[j, 4] = 0
            elif np.abs(BHE_array[j].z_end) >= np.sum(
                layer
            ):  # pragma: no cover
                raise Exception(
                    "BHE %d ends at bottom boundary or outside of the model area"
                    % j
                )

    needed_depths: list = []  # interesting depths
    for i in range(0, len(layer)):
        BHE_end_depths = (
            []
        )  # only the interesting depths in the i-th layer ToDo: Rename the variable

        # filter, which BHE's ends in this layer
        BHE_end_in_Layer = BHE_to_soil[BHE_to_soil[:, 3] == i]

        for k in BHE_end_in_Layer[:, 0]:
            BHE_end_depths.append([BHE_array[k].z_end, BHE_to_soil[k, 4]])

        # filter, which BHE's starts in this layer
        BHE_starts_in_Layer = BHE_to_soil[BHE_to_soil[:, 1] == i]

        for k in BHE_starts_in_Layer[:, 0]:
            BHE_end_depths.append([BHE_array[k].z_begin, BHE_to_soil[k, 2]])

        groundwater_list_0 = np.array(
            [inner_list[0] for inner_list in groundwater_list]
        )
        if i in groundwater_list_0:
            if len(np.argwhere(groundwater_list_0 == i)) == 1:
                BHE_end_depths.append(
                    [
                        groundwaters[
                            np.argwhere(groundwater_list_0 == i)[0, 0]
                        ].begin,
                        icl,
                    ]
                )
            else:  # pragma: no cover
                msg = "Two or more groundwater flows starts in the same soil layer, this is not allowed !"
                raise Exception(msg)

        # print(BHE_end_depths)
        groundwater_list_3 = np.array(
            [inner_list[3] for inner_list in groundwater_list]
        )
        if i in groundwater_list_3:
            BHE_end_depths.append([-np.sum(layer[:i]), 3])

        BHE_end_depths.append([-np.sum(layer[:i]), 0])
        BHE_end_depths.append([-np.sum(layer[: i + 1]), 0])
        depths = np.unique(BHE_end_depths, axis=0)  # [::-1]

        test, counts = np.unique(depths[:, 0], return_counts=True)
        arg_indexes = np.argwhere(counts > 1)

        if arg_indexes.size != 0:
            new_test = depths[depths[:, 0] == test[arg_indexes[0]]][:, 1]
            if np.argwhere(new_test > 0) == 1:
                duplicate_depth = np.argwhere(
                    depths[:, 0] == test[arg_indexes[0]]
                )
                not_needed_icl = np.argwhere(depths[:, 1] == 0)
                depths = np.delete(
                    depths,
                    np.intersect1d(duplicate_depth, not_needed_icl),
                    axis=0,
                )

            else:  # pragma: no cover
                msg = "Layering to difficult, groundwater, BHE depths and needed layers are very close - behaviour currently not implemented"
                raise Exception(msg)

        BHE_end_depths = depths[::-1]
        needed_depths.append(BHE_end_depths)

    number_of_layers: list = []
    cummulative_height_of_layers: list = []
    depth_of_extrusion: list = []
    for i in range(0, len(layer)):  # Schleife zum Berechnen der Layer-Struktur
        list_of_needed_depths = needed_depths[
            i
        ]  # all depths, which needs a node in the mesh

        # vorheriger_layer - Abstand für top-critical etc.
        if i > 0:
            if (
                2 in needed_depths[i - 1][:, 1]
                or 1.2 in needed_depths[i - 1][:, 1]
            ):
                space_last_layer_refined = (
                    needed_depths[i - 1][-1, 0]
                    - needed_depths[i - 1][-2, 0]
                    + n_refinement_layers * target_z_size_fine
                )
                if space_last_layer_refined < target_z_size_fine:
                    space_last_layer_refined = target_z_size_fine
                else:
                    space_last_layer_refined = space_last_layer_refined
            else:
                space_last_layer_refined = 0
        else:
            space_last_layer_refined = 0

        # nächster_Layer - Abstand für top-critical etc.
        if i < len(layer) - 1:
            if (
                1 in needed_depths[i + 1][:, 1]
                or 3 in needed_depths[i + 1][:, 1]
                or 1.2 in needed_depths[i + 1][:, 1]
            ):
                if 3 in needed_depths[i + 1][:, 1]:
                    space_next_layer_refined = (
                        n_refinement_layers * target_z_size_fine
                    )
                else:
                    space_next_layer_refined = (
                        needed_depths[i + 1][1, 0]
                        - needed_depths[i + 1][0, 0]
                        + n_refinement_layers * target_z_size_fine
                    )
                    if space_next_layer_refined < target_z_size_fine:
                        space_next_layer_refined = target_z_size_fine
                    else:
                        space_next_layer_refined = space_next_layer_refined
            else:
                space_next_layer_refined = 0  # nichts beim layering zu beachten
        elif i + 1 in groundwater_list_3:  # i+1 == groundwater[1]:
            space_next_layer_refined = (
                n_refinement_layers * target_z_size_fine
            )  # if, groundwater isolator is deeper than the model area
        else:
            space_next_layer_refined = 0

        absolute_height_of_layers: list = []
        number_of_layers.append([])

        # Evaluate Mesh for the Soil-Layers
        for j in range(
            1, len(list_of_needed_depths)
        ):  # not sure, hope to run up to the bounds of the layer
            groundwater_list_2 = np.array(
                [inner_list[2] for inner_list in groundwater_list]
            )
            if (
                list_of_needed_depths[j, 0] in groundwater_list_2
                and list_of_needed_depths[j, 0] != list_of_needed_depths[-1, 0]
            ):
                _compute_layer_spacing(
                    list_of_needed_depths[j - 1, 0],
                    list_of_needed_depths[j, 0],
                    list_of_needed_depths[j - 1, 1],
                    j,
                    len(list_of_needed_depths) - 1,
                )
                # Befehle zum Anlegen einer neuen Extrusion
                cummulative_height_of_layers.append(
                    (
                        np.cumsum(
                            np.array(absolute_height_of_layers)
                            / np.sum(absolute_height_of_layers)
                        )
                    ).tolist()
                )
                depth_of_extrusion.append(np.sum(absolute_height_of_layers))
                absolute_height_of_layers = []
                number_of_layers.append([])

            else:
                _compute_layer_spacing(
                    list_of_needed_depths[j - 1, 0],
                    list_of_needed_depths[j, 0],
                    list_of_needed_depths[j - 1, 1],
                    j,
                    len(list_of_needed_depths) - 1,
                )
                # print('no fine mesh at layer transition')

        cummulative_height_of_layers.append(
            (
                np.cumsum(
                    np.array(absolute_height_of_layers)
                    / np.sum(absolute_height_of_layers)
                )
            ).tolist()
        )
        depth_of_extrusion.append(np.sum(absolute_height_of_layers))

    # evaluate all extrusion depths with according num of layers
    All_extrusion_depths: list = []
    All_extrusion_layers: list = []

    last_height = 0
    for i in range(0, len(number_of_layers)):
        (
            depth_of_extrusion[i] * np.array(cummulative_height_of_layers[i])
            + last_height
        )
        All_extrusion_depths.append(
            (
                depth_of_extrusion[i]
                * np.array(cummulative_height_of_layers[i])
                + last_height
            ).tolist()
        )
        last_height = All_extrusion_depths[-1][-1]
        All_extrusion_layers.append(number_of_layers[i])

    all_extrusion = np.array(
        [
            _flatten_concatenation(All_extrusion_depths),
            _flatten_concatenation(All_extrusion_layers),
        ]
    ).transpose()

    BHE_extrusion_layers: list = []
    BHE_extrusion_depths: list = []
    # evaluate the extrusion for the BHE's
    for i in range(0, len(BHE_array)):
        needed_extrusion = all_extrusion[
            (
                (all_extrusion[:, 0] >= np.abs(BHE_array[i].z_begin))
                & (
                    all_extrusion[:, 0] <= np.abs(BHE_array[i].z_end) + 0.001
                )  # add little relax tolerance 0.001
            )
        ]

        BHE_extrusion_layers.append(needed_extrusion[:, 1])
        BHE_extrusion_depths.append(
            (needed_extrusion[:, 0] - np.abs(BHE_array[i].z_begin))
            / (needed_extrusion[-1, 0] - np.abs(BHE_array[i].z_begin))
        )

    # define the inner square with BHE inside
    # compute the box size from the BHE-Coordinates
    x_BHE = [BHE_array[i].x for i in range(0, len(BHE_array))]
    y_BHE = [BHE_array[i].y for i in range(0, len(BHE_array))]

    x_min = np.min(x_BHE) - dist_box_x
    x_max = np.max(x_BHE) + dist_box_x
    y_min = np.min(y_BHE) - dist_box_y
    y_max = np.max(y_BHE) + dist_box_y

    # Index for the right export of the groundwater inflow surface and adapt mesh sizes according to GW-flow
    plus_x_mesh_size = inner_mesh_size
    minus_x_mesh_size = inner_mesh_size
    # plus_y_mesh_size = inner_mesh_size
    minus_y_mesh_size = inner_mesh_size

    alpha = 6.134  # see Diersch et al. 2011 Part 2 for 6 surrounding nodes, not to be defined by user

    outer_mesh_size_inner = (outer_mesh_size + inner_mesh_size) / 2

    gmsh.initialize()
    gmsh.model.add(Path(out_name).stem)

    if meshing_type == "structured":
        _mesh_structured()
    elif meshing_type == "prism":
        _mesh_prism()
    else:  # pragma: no cover
        gmsh.finalize()
        msg = "Unknown meshing type! prism and structured supported"
        raise Exception(msg)

    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.removeDuplicateNodes()

    # delete zero-volume elements
    elem_tags, node_tags = gmsh.model.mesh.getElementsByType(
        1
    )  # 1 for line elements --> BHE's are the reason
    elem_qualities = gmsh.model.mesh.getElementQualities(
        elementTags=elem_tags, qualityName="volume"
    )

    zero_volume_elements_id = np.argwhere(elem_qualities == 0)

    # only possible with the hack over the visibilitiy, see https://gitlab.onelab.info/gmsh/gmsh/-/issues/2006
    gmsh.model.mesh.setVisibility(
        elem_tags[zero_volume_elements_id].flatten().tolist(), 0
    )
    gmsh.plugin.setNumber("Invisible", "DeleteElements", 1)
    gmsh.plugin.run("Invisible")

    gmsh.write(str(out_name))
    gmsh.finalize()


def gen_bhe_mesh(
    length: float,  # e.g. 150.0
    width: float,  # e.g. 100
    layer: Union[float, list[float]],  # e.g. 100
    groundwater: Union[Groundwater, list[Groundwater]],
    BHE_Array: Union[
        BHE,
        list[BHE],
    ],
    target_z_size_coarse: float = 7.5,
    target_z_size_fine: float = 1.5,
    n_refinement_layers: int = 2,
    meshing_type: str = "structured",
    dist_box_x: float = 5.0,
    dist_box_y: float = 10.0,
    inner_mesh_size: float = 5.0,
    outer_mesh_size: float = 10.0,
    propagation: float = 1.1,
    order: int = 1,
    out_name: Path = Path("bhe_mesh.vtu"),
) -> list[str]:
    """
    Create a generic BHE mesh for the Heat_Transport_BHE-Process with additionally
    submeshes at the top, at the bottom and the groundwater inflow, which is exported
    in the OGS readable .vtu format. Refinement layers are placed at the BHE-begin, the BHE-end and the groundwater start/end. See detailed description of the parameters below:

    :param length: Length of the model area in m (x-dimension)
    :param width: Width of the model area in m (y-dimension)
    :param layer: List of the soil layer thickness in m
    :param groundwater: List of groundwater layers, where every is specified by a tuple
        of three entries: [depth of groundwater begin (negative), number of the groundwater
        isolation layer (count starts with 0), groundwater inflow direction, as string - supported '+x', '-x', '-y', '+y'], empty list [] for no groundwater flow
    :param BHE_Array: List of BHEs, where every BHE is specified by a tuple of five floats:
        [x-coordinate BHE, y-coordinate BHE, BHE begin depth (zero or negative),
        BHE end depth (negative), borehole radius in m]
    :param target_z_size_coarse: maximum edge length of the elements in m in z-direction,
        if no refinemnt needed
    :param target_z_size_fine: maximum edge length of the elements in the refinement zone
        in m in z-direction
    :param n_refinement_layers: number of refinement layers which are evenly set above and
        beneath the refinemnt depths (see general description above)
    :param meshing_type: 'structured' and 'prism' are supported
    :param dist_box_x: distance in m in x-direction of the refinemnt box according to the BHE's
    :param dist_box_y: distance in m in y-direction of the refinemnt box according to the BHE's
    :param inner_mesh_size: mesh size inside the refinement box in m
    :param outer_mesh_size: mesh size outside of the refinement box in m
    :param propagation: growth of the outer_mesh_size, only supported by meshing_type
        'structured'
    :param order: Define the order of the mesh: 1 for linear finite elements / 2 for quadratic finite elements
    :param out_name: name of the exported mesh, must end with .vtu
    :return: list of filenames of the created vtu mesh files
    """

    tmp_dir = Path(mkdtemp())
    mesh_name = out_name.stem
    msh_file = tmp_dir / f"{mesh_name}.msh"

    # using gen_bhe_mesh_gmsh as basis function
    gen_bhe_mesh_gmsh(
        length=length,
        width=width,
        layer=layer,
        groundwater=groundwater,
        BHE_Array=BHE_Array,
        target_z_size_coarse=target_z_size_coarse,
        target_z_size_fine=target_z_size_fine,
        n_refinement_layers=n_refinement_layers,
        meshing_type=meshing_type,
        dist_box_x=dist_box_x,
        dist_box_y=dist_box_y,
        inner_mesh_size=inner_mesh_size,
        outer_mesh_size=outer_mesh_size,
        propagation=propagation,
        order=order,
        out_name=msh_file,
    )

    msh2vtu(
        msh_file,
        output_path=out_name.parents[0],
        dim=[1, 3],
        reindex=True,
        log_level="ERROR",
    )

    mesh_names = [
        f"{mesh_name}_domain.vtu",
        f"{mesh_name}_physical_group_Top_Surface.vtu",
        f"{mesh_name}_physical_group_Bottom_Surface.vtu",
    ]

    groundwater = (
        [groundwater] if isinstance(groundwater, Groundwater) else groundwater
    )

    for i in range(0, len(groundwater)):
        mesh_names.append(
            f"{mesh_name}_physical_group_Groundwater_Inflow_{i}.vtu"
        )

    return mesh_names
