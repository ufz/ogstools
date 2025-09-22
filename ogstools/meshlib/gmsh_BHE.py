# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from dataclasses import dataclass
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import gmsh
import numpy as np
import shapely

import ogstools as ot


@dataclass(frozen=True)
class Groundwater:
    begin: float = -30
    """ depth of groundwater begin (negative) in m from top surface """
    isolation_layer_id: int = 1
    """ number of the groundwater isolation layer (count starts with 0)"""
    upstream: tuple[float, float] = (160, 200)
    """
    Tuple of length 2 defining the angular range (in degrees) of groundwater inflow surfaces.
    Angles are measured on a 0 - 359° circle, where 0° corresponds to the +x axis direction and
    values increase counterclockwise. The first value defines the start angle, the second defines
    the end angle. If the start angle is larger than the end angle, the range wraps around 0°
    (e.g., (359, 1) covers 359° -> 0° -> 1°).
    """
    downstream: tuple[float, float] = (340, 20)
    """
    Tuple of length 2 defining the angular range (in degrees) of groundwater outflow surfaces.
    Angles are measured on a 0 - 359° circle, where 0° corresponds to the +x axis direction and
    values increase counterclockwise. The first value defines the start angle, the second defines
    the end angle. If the start angle is larger than the end angle, the range wraps around 0°
    (e.g., (340, 20) covers 340° -> 359° -> 0° -> 20°).
    """


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
    """BHE end depth (negative) in m"""
    borehole_radius: float = 0.076
    """borehole radius in m"""


def gen_bhe_mesh(
    model_area: shapely.Polygon,
    layer: float | list[float],  # e.g. 100
    groundwater: Groundwater | list[Groundwater],
    BHE_Array: BHE | list[BHE],
    refinement_area: shapely.Polygon,
    target_z_size_coarse: float = 7.5,
    target_z_size_fine: float = 1.5,
    n_refinement_layers: int = 2,
    meshing_type: Literal["prism", "structured"] = "prism",
    inner_mesh_size: float = 5.0,
    outer_mesh_size: float = 10.0,
    propagation: float = 1.1,
    order: int = 1,
    meshname: str = "bhe_mesh",
) -> ot.Meshes:
    """
    Create a generic BHE mesh for the Heat_Transport_BHE-Process with additionally
    submeshes at the top, at the bottom and the groundwater in- and outflow, which is returned as :py:mod:`ogstools.meshlib.Meshes`
    Refinement layers are placed at the BHE-begin, the BHE-end and the groundwater start/end. See detailed description of the parameters below:

    :param model_area: A shapely.Polygon (see https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html) of the model. No holes are allowed.
    :param layer: List of the soil layer thickness in m
    :param groundwater: List of groundwater layers, where every is specified by a tuple
        of three entries: [depth of groundwater begin (negative), number of the groundwater
        isolation layer (count starts with 0), groundwater upstream and downstream as tuple of 2 thresholds angles starting with 0 at +x (first value start, second end),  empty list [] for no groundwater flow
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
    :param refinement_area: A shapely.Polygon (see https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html) of the refinement_area. No holes are allowed.
    :param inner_mesh_size: mesh size inside the refinement area in m
    :param outer_mesh_size: mesh size outside of the refinement area in m
    :param propagation: growth of the outer_mesh_size, only supported by meshing_type
        'structured'
    :param order: Define the order of the mesh: 1 for linear finite elements / 2 for quadratic finite elements
    :param meshname: The name of the domain mesh.
    :returns: A ot.Meshes object

    # .. image:: ../../examples/howto_preprocessing/gen_bhe_mesh.png
    """

    def _compute_layer_spacing(
        z_min: float, z_max: float, z_min_id: int, id_j: float, id_j_max: float
    ) -> None:
        delta_z = np.abs(z_min - z_max)
        size_fine = n_refinement_layers * target_z_size_fine
        target_layer: list = number_of_layers[len(number_of_layers) - 1]
        if z_min_id == 3:  # and id_j==1 - not needed, but logically safer
            if id_j == id_j_max:
                space = delta_z - size_fine - space_next_layer_refined
                space_fine_next = space + size_fine + space_next_layer_refined
                if space_next_layer_refined == 0:
                    if space <= target_z_size_fine:
                        absolute_height_of_layers.append(delta_z)
                        target_layer.append(ceil(delta_z / target_z_size_fine))
                    else:
                        absolute_height_of_layers.extend([size_fine, space])
                        target_layer.append(n_refinement_layers)
                        target_layer.append(ceil(space / target_z_size_coarse))
                else:
                    if space <= target_z_size_fine:
                        absolute_height_of_layers.append(space_fine_next)
                        target_layer.append(
                            ceil(space_fine_next / target_z_size_fine)
                        )
                    else:
                        absolute_height_of_layers.append(size_fine)
                        target_layer.append(n_refinement_layers)
                        absolute_height_of_layers.append(space)
                        target_layer.append(ceil(space / target_z_size_coarse))
                        absolute_height_of_layers.append(
                            space_next_layer_refined
                        )
                        target_layer.append(
                            ceil(space_next_layer_refined / target_z_size_fine)
                        )

            else:
                # all negative values, because of the extrusion in negative z-direction -> z_min -- End Layer, z_max -- start layer
                space = delta_z - size_fine
                if (
                    np.abs(space)
                    <= (n_refinement_layers + 1) * target_z_size_fine
                ):
                    absolute_height_of_layers.append(space + size_fine)
                    target_layer.append(
                        ceil((space + size_fine) / target_z_size_fine)
                    )
                else:
                    absolute_height_of_layers.append(size_fine)
                    target_layer.append(n_refinement_layers)
                    absolute_height_of_layers.append(space - size_fine)
                    target_layer.append(
                        ceil((space - size_fine) / target_z_size_coarse)
                    )
                    absolute_height_of_layers.append(size_fine)
                    target_layer.append(n_refinement_layers)

        # erstes Mesh in jeweiligen Soil-Layer
        elif id_j == 1 and id_j != id_j_max:
            space = delta_z - space_last_layer_refined
            space_last = space + space_last_layer_refined
            if np.abs(space) <= (n_refinement_layers + 1) * target_z_size_fine:
                absolute_height_of_layers.append(space_last)
                target_layer.append(ceil(space_last / target_z_size_fine))
            else:
                if space_last_layer_refined == 0:
                    absolute_height_of_layers.append(space - size_fine)
                    target_layer.append(
                        ceil((space - size_fine) / target_z_size_coarse)
                    )
                    absolute_height_of_layers.append(size_fine)
                    target_layer.append(n_refinement_layers)
                else:
                    absolute_height_of_layers.append(space_last_layer_refined)
                    target_layer.append(
                        ceil(space_last_layer_refined / target_z_size_fine)
                    )
                    absolute_height_of_layers.append(space - size_fine)
                    target_layer.append(
                        ceil((space - size_fine) / target_z_size_coarse)
                    )
                    absolute_height_of_layers.append(2 * target_z_size_fine)
                    target_layer.append(n_refinement_layers)

        elif id_j == id_j_max and id_j != 1:
            space = delta_z - space_next_layer_refined
            space_next = space + space_next_layer_refined
            if space <= (n_refinement_layers + 1) * target_z_size_fine:
                absolute_height_of_layers.append(space_next)
                target_layer.append(ceil(space_next / target_z_size_fine))
            else:
                if space_next_layer_refined == 0:
                    absolute_height_of_layers.append(size_fine)
                    target_layer.append(n_refinement_layers)
                    absolute_height_of_layers.append(space - size_fine)
                    target_layer.append(
                        ceil((space - size_fine) / target_z_size_coarse)
                    )
                else:
                    absolute_height_of_layers.append(size_fine)
                    target_layer.append(n_refinement_layers)
                    absolute_height_of_layers.append(space - size_fine)
                    target_layer.append(
                        ceil((space - size_fine) / target_z_size_coarse)
                    )
                    absolute_height_of_layers.append(space_next_layer_refined)
                    target_layer.append(
                        ceil(space_next_layer_refined / target_z_size_fine)
                    )

        # Layer without a needed depth of BHE or Groundwater
        elif id_j == id_j_max and id_j == 1:
            space = delta_z
            space_next = space - space_next_layer_refined
            space_last = space - space_last_layer_refined
            space_nextlast = space_next - space_last_layer_refined
            if space_last_layer_refined == 0 and space_next_layer_refined == 0:
                absolute_height_of_layers.append(space)
                target_layer.append(ceil(space / target_z_size_coarse))
            elif space_next_layer_refined == 0:
                if space_last <= target_z_size_fine:
                    absolute_height_of_layers.append(space)
                    target_layer.append(ceil(space / target_z_size_fine))
                else:
                    absolute_height_of_layers.append(space_last_layer_refined)
                    target_layer.append(
                        ceil(space_last_layer_refined / target_z_size_fine)
                    )
                    absolute_height_of_layers.append(space_last)
                    target_layer.append(ceil(space_last / target_z_size_coarse))
            elif space_last_layer_refined == 0:
                if space_next <= target_z_size_fine:
                    absolute_height_of_layers.append(space)
                    target_layer.append(ceil(space / target_z_size_fine))
                else:
                    absolute_height_of_layers.append(space_next)
                    target_layer.append(ceil(space_next / target_z_size_coarse))
                    absolute_height_of_layers.append(space_next_layer_refined)
                    target_layer.append(
                        ceil(space_next_layer_refined / target_z_size_fine)
                    )
            else:
                if space_nextlast <= target_z_size_fine:
                    absolute_height_of_layers.append(space)
                    target_layer.append(ceil(space / target_z_size_fine))
                else:
                    absolute_height_of_layers.append(space_next_layer_refined)
                    target_layer.append(
                        ceil(space_next_layer_refined / target_z_size_fine)
                    )
                    absolute_height_of_layers.append(space_nextlast)
                    target_layer.append(
                        ceil(space_nextlast / target_z_size_coarse)
                    )
                    absolute_height_of_layers.append(space_last_layer_refined)
                    target_layer.append(
                        ceil(space_last_layer_refined / target_z_size_fine)
                    )

        else:
            space = delta_z
            if space <= (2 * n_refinement_layers + 1) * target_z_size_fine:
                absolute_height_of_layers.append(space)
                target_layer.append(ceil(space / target_z_size_fine))
            else:
                absolute_height_of_layers.append(size_fine)
                target_layer.append(n_refinement_layers)
                absolute_height_of_layers.append(space - 2 * size_fine)
                target_layer.append(
                    ceil((space - 2 * size_fine) / target_z_size_coarse)
                )
                absolute_height_of_layers.append(size_fine)
                target_layer.append(n_refinement_layers)

    # to flat a list, seems not so easy with a ready to use function --> this code is from https://realpython.com/python-flatten-list/
    def _flatten_concatenation(matrix: list[list[Any]]) -> list:
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    def _insert_BHE(inTag: int) -> list:
        bhe_top_nodes = []
        BHE_i: BHE
        for BHE_i in BHE_array:
            x, y, z = (BHE_i.x, BHE_i.y, 0)
            # meshsize at BHE and distance of the surrounding optimal mesh points
            # see Diersch et al. 2011 Part 2 for 6 surrounding nodes, not to be defined by user
            delta = 6.134 * BHE_i.borehole_radius

            bhe_center = gmsh.model.geo.addPoint(x, y, z, delta)
            gmsh.model.geo.addPoint(x, y - delta, z, delta)
            gmsh.model.geo.addPoint(x, y + delta, z, delta)
            dx, dy = (0.866 * delta, 0.5 * delta)
            gmsh.model.geo.addPoint(x + dx, y + dy, z, delta)
            gmsh.model.geo.addPoint(x - dx, y + dy, z, delta)
            gmsh.model.geo.addPoint(x + dx, y - dy, z, delta)
            gmsh.model.geo.addPoint(x - dx, y - dy, z, delta)

            if BHE_i.z_begin != 0:
                bhe_top_nodes.append(
                    gmsh.model.geo.addPoint(x, y, BHE_i.z_begin, delta)
                )
            else:
                bhe_top_nodes.append(bhe_center)

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.embed(
                0, list(range(bhe_center, bhe_center + 7)), 2, inTag
            )
        return bhe_top_nodes

    def _mesh_structured() -> None:
        if shapely.get_coordinate_dimension(
            refinement_area
        ) != shapely.get_coordinate_dimension(model_area):
            msg = f"Dimension of model area is {shapely.get_coordinate_dimension(model_area)}, but refinement area is of dimension {shapely.get_coordinate_dimension(refinement_area)}"
            raise ValueError(msg)

        point_registry_refinement_area: dict[frozenset[float], int] = {}
        line_registry_refinement_area = {}

        # define the refinement area
        point_ids_refinement = []
        if shapely.get_coordinate_dimension(refinement_area) == 2:
            for i, (x, y) in enumerate(refinement_area.exterior.coords[:-1]):
                point_ids_refinement.append(
                    geo.addPoint(x, y, 0, outer_mesh_size, tag=i)
                )
                point_registry_refinement_area[frozenset([x, y])] = (
                    point_ids_refinement[-1]
                )
        else:
            for i, (x, y, z) in enumerate(refinement_area.exterior.coords[:-1]):
                point_ids_refinement.append(
                    geo.addPoint(x, y, z, outer_mesh_size, tag=i)
                )
                point_registry_refinement_area[frozenset([x, y, z])] = (
                    point_ids_refinement[-1]
                )

        refinement_area_lines = list(
            zip(
                point_ids_refinement,
                point_ids_refinement[1:] + point_ids_refinement[:1],
                strict=True,
            )
        )
        refinement_area_line_ids = []
        for p1, p2 in refinement_area_lines:
            refinement_area_line_ids.append(geo.addLine(p1, p2))
            line_registry_refinement_area[frozenset([p1, p2])] = (
                refinement_area_line_ids[-1]
            )

        cl_refinement_area = geo.addCurveLoop(refinement_area_line_ids)
        surface_refinemment_area = geo.addPlaneSurface([cl_refinement_area])

        max_point_tag = geo.getMaxTag(dim=0) + 1
        # define the model area
        point_model_area_registry: dict[int, tuple[float, ...]] = {}
        if shapely.get_coordinate_dimension(model_area) == 2:
            for i, (x, y) in enumerate(model_area.exterior.coords[:-1]):
                point_model_area_registry[
                    geo.addPoint(
                        x, y, 0, outer_mesh_size, tag=max_point_tag + i
                    )
                ] = (x, y, 0)
        else:
            for i, (x, y, z) in enumerate(model_area.exterior.coords[:-1]):
                point_model_area_registry[
                    geo.addPoint(
                        x, y, z, outer_mesh_size, tag=max_point_tag + i
                    )
                ] = (x, y, z)

        point_ids_model_area = list(point_model_area_registry.keys())
        model_area_lines = list(
            zip(
                point_ids_model_area,
                point_ids_model_area[1:] + point_ids_model_area[:1],
                strict=True,
            )
        )

        model_surfaces = []
        model_surfaces_n_sides = [
            len(refinement_area_lines)
        ]  # TODO: Check this
        connection_lines = []
        transfinite_surfaces = []
        transfinite_curves: dict[int, tuple[int, float]] = {}
        connection_line_registry: dict[frozenset[int], int] = {}
        connection_line_transfinite_rings: list[list] = [[]]
        gw_upstream: list[list[int]] = [[] for _ in range(len(groundwaters))]
        gw_downstream: list[list[int]] = [[] for _ in range(len(groundwaters))]
        ref_vec = [1, 0]  # in positive x-dir
        for i, (p1, p2) in enumerate(model_area_lines):
            # determine, if a line corresponds to the groundwater
            p1_x, p1_y = point_model_area_registry[p1][:2]
            p2_x, p2_y = point_model_area_registry[p2][:2]
            dx = p2_x - p1_x
            dy = p2_y - p1_y
            normal_vec = [dy, -dx]

            # Calculate dot product
            dot_product = np.dot(ref_vec, normal_vec)

            # Calculate magnitudes (lengths of the vectors)
            magnitude_A = np.linalg.norm(ref_vec)
            magnitude_B = np.linalg.norm(normal_vec)

            # Calculate angle in radians
            angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))

            # Convert radians to degrees
            normal_vec_angle = (
                360 - np.degrees(angle_radians)
                if normal_vec[1] < 0
                else np.degrees(angle_radians)
            )
            for i_gw, groundwater in enumerate(groundwaters):
                if groundwater.upstream[0] > groundwater.upstream[1]:
                    if (
                        normal_vec_angle > groundwater.upstream[0]
                        or normal_vec_angle < groundwater.upstream[1]
                    ):
                        gw_upstream[i_gw].append(i)
                elif (
                    normal_vec_angle > groundwater.upstream[0]
                    and normal_vec_angle < groundwater.upstream[1]
                ):
                    gw_upstream[i_gw].append(i)

                if groundwater.downstream[0] > groundwater.downstream[1]:
                    if (
                        normal_vec_angle > groundwater.downstream[0]
                        or normal_vec_angle < groundwater.downstream[1]
                    ):
                        gw_downstream[i_gw].append(i)
                elif (
                    normal_vec_angle > groundwater.downstream[0]
                    and normal_vec_angle < groundwater.downstream[1]
                ):
                    gw_downstream[i_gw].append(i)

            model_line = geo.addLine(p1, p2)

            point1 = shapely.Point(
                model_area.exterior.coords[p1 - max_point_tag]
            )
            point2 = shapely.Point(
                model_area.exterior.coords[p2 - max_point_tag]
            )
            # find nearest point of the refinement area
            nearest_to_point1 = min(
                refinement_area.exterior.coords,
                key=lambda p: point1.distance(shapely.Point(p)),
            )
            nearest_to_point2 = min(
                refinement_area.exterior.coords,
                key=lambda p: point2.distance(shapely.Point(p)),
            )
            id_nearest_to_point1 = point_registry_refinement_area[
                frozenset(nearest_to_point1)
            ]
            id_nearest_to_point2 = point_registry_refinement_area[
                frozenset(nearest_to_point2)
            ]

            if nearest_to_point1 == nearest_to_point2 and i != 0:
                connection_line_transfinite_rings.append([])
                triangle_surface = True
            else:
                triangle_surface = False

            if (
                frozenset([p2, id_nearest_to_point2])
                in connection_line_registry
            ):
                connection_1 = connection_line_registry[
                    frozenset([p2, id_nearest_to_point2])
                ]
                if not triangle_surface:
                    connection_line_transfinite_rings[-1].append(connection_1)
            else:
                connection_lines.append(geo.addLine(p2, id_nearest_to_point2))
                connection_line_registry[
                    frozenset([p2, id_nearest_to_point2])
                ] = connection_lines[-1]
                connection_1 = connection_lines[-1]

            if (
                frozenset([p1, id_nearest_to_point1])
                in connection_line_registry
            ):
                connection_2 = connection_line_registry[
                    frozenset([p1, id_nearest_to_point1])
                ]
                if not triangle_surface:
                    connection_line_transfinite_rings[-1].append(connection_2)
            else:
                connection_lines.append(geo.addLine(p1, id_nearest_to_point1))
                connection_line_registry[
                    frozenset([p1, id_nearest_to_point1])
                ] = connection_lines[-1]
                connection_2 = connection_lines[-1]

            if nearest_to_point1 == nearest_to_point2:
                cl_surface = geo.addCurveLoop(
                    [model_line, connection_1, -connection_2]
                )
                model_surfaces.append(geo.addPlaneSurface([cl_surface]))
                model_surfaces_n_sides.append(3)
            else:
                if (
                    frozenset([id_nearest_to_point1, id_nearest_to_point2])
                    in line_registry_refinement_area
                ):
                    refinement_area_line = -line_registry_refinement_area[
                        frozenset([id_nearest_to_point1, id_nearest_to_point2])
                    ]
                elif (
                    frozenset([id_nearest_to_point2, id_nearest_to_point1])
                    in line_registry_refinement_area
                ):
                    refinement_area_line = line_registry_refinement_area[
                        frozenset([id_nearest_to_point2, id_nearest_to_point1])
                    ]
                else:
                    msg = "Something went wrong with building the surface line loops. Please check, that your model and refinement area defined properly."
                    raise Exception(msg)

                cl_surface = geo.addCurveLoop(
                    [
                        model_line,
                        connection_1,
                        refinement_area_line,
                        -connection_2,
                    ],
                    reorient=True,
                )
                model_surfaces.append(geo.addPlaneSurface([cl_surface]))
                model_surfaces_n_sides.append(4)
                geo.synchronize()
                dist_c1 = point1.distance(shapely.Point(nearest_to_point1))
                dist_c2 = point2.distance(shapely.Point(nearest_to_point2))
                dist_refinement_line = shapely.Point(
                    nearest_to_point1
                ).distance(shapely.Point(nearest_to_point2))
                dist_model_line = point1.distance(point2)

                n_elem_con_1 = (
                    transfinite_curves[connection_1][0]
                    if connection_1 in transfinite_curves
                    else ceil(dist_c1 / outer_mesh_size_inner) + 1
                )
                n_elem_con_2 = (
                    transfinite_curves[connection_2][0]
                    if connection_2 in transfinite_curves
                    else ceil(dist_c2 / outer_mesh_size_inner) + 1
                )
                max_element_n = max(n_elem_con_1, n_elem_con_2)
                transfinite_curves[connection_1] = (max_element_n, -propagation)
                transfinite_curves[connection_2] = (max_element_n, -propagation)

                num_elements_refine_and_model = max(
                    ceil(dist_refinement_line / inner_mesh_size) + 1,
                    ceil(dist_model_line / outer_mesh_size) + 1,
                )
                transfinite_curves[model_line] = (
                    num_elements_refine_and_model,
                    1.0,
                )
                transfinite_curves[refinement_area_line] = (
                    num_elements_refine_and_model,
                    1.0,
                )
                transfinite_surfaces.append(model_surfaces[-1])

        gmsh.model.geo.synchronize()

        if groundwaters:
            if not _flatten_concatenation(gw_downstream):
                msg = "No groundwater upstream surfaces detected, please check your specified angles!"
                raise ValueError(msg)
            if not _flatten_concatenation(gw_downstream):
                msg = "No groundwater downstream surfaces detected, please check your specified angles!"
                raise ValueError(msg)

        for connection_ring in connection_line_transfinite_rings:
            mesh_sizes = []
            for connection_line in connection_ring:
                mesh_sizes.append(transfinite_curves[connection_line][0])

            minimum_mesh_size = max(mesh_sizes)
            for connection_line in connection_ring:
                transfinite_curves[connection_line] = (
                    minimum_mesh_size,
                    -propagation,
                )

        bhe_top_nodes = _insert_BHE(inTag=1)

        # Extrude the surface mesh according to the previously evaluated structure
        volumes_list_for_layers = []

        top_surface = [
            surface_refinemment_area
        ] + model_surfaces  # list(range(1, 10))
        surface_list = [(2, tag) for tag in top_surface]

        gw_downstream_tags: list[list[int]] = [
            [] for _ in range(len(groundwaters))
        ]
        gw_upstream_tags: list[list[int]] = [
            [] for _ in range(len(groundwaters))
        ]

        for j, num_elements in enumerate(number_of_layers):
            # spacing of the each layer must be evaluated according to the implementation of the bhe
            extrusion_tags = geo.extrude(
                surface_list,
                0,
                0,
                -depth_of_extrusion[j],
                num_elements,
                cummulative_height_of_layers[j],
                True,
            )  # soil 1
            geo.synchronize()

            # list of volume numbers and new bottom surfaces, which were extruded by the five surfaces
            n_surfaces = (
                -1
            )  # the number of surfaces is number of model area lines + 1 for the refinement area
            volume_list = []
            surface_list = []
            for i, (dim, tag) in enumerate(extrusion_tags):
                if dim == 3:
                    volume_list.append(tag)
                    surface_list.append(extrusion_tags[i - 1])
                    for i_gw in range(len(groundwaters)):
                        if n_surfaces in gw_downstream[i_gw]:
                            gw_downstream_tags[i_gw].append(
                                extrusion_tags[i + 1][1]
                            )
                        if n_surfaces in gw_upstream[i_gw]:
                            gw_upstream_tags[i_gw].append(
                                extrusion_tags[i + 1][1]
                            )
                    n_surfaces += 1

            volumes_list_for_layers.append(volume_list)

        k = 0
        BHE_group = []
        for i, BHE_i in enumerate(BHE_array):
            extrusion_tags = geo.extrude(
                [(0, bhe_top_nodes[i])],
                0,
                0,
                BHE_i.z_end - BHE_i.z_begin,
                BHE_extrusion_layers[i],
                BHE_extrusion_depths[i],
                True,
            )
            BHE_group.append(extrusion_tags[1][1])

        geo.synchronize()

        for i, volume_i in enumerate(volumes_list_for_layers):
            model.addPhysicalGroup(3, volume_i, i)

        for k, BHE_group_k in enumerate(BHE_group, start=i + 1):
            model.addPhysicalGroup(1, [BHE_group_k], k)

        model.addPhysicalGroup(2, top_surface, k + 1, name="Top_Surface")
        model.addPhysicalGroup(
            2,
            np.array(surface_list)[:, 1].tolist(),
            k + 2,
            name="Bottom_Surface",
        )

        for key, (numNodes, prop) in transfinite_curves.items():
            mesh.setTransfiniteCurve(key, numNodes, coef=prop)

        for surface in transfinite_surfaces:
            mesh.setTransfiniteSurface(surface)
            mesh.setRecombine(2, surface)
        mesh.recombine()

        gw_counter = 0  # counter_for_gw_start_at_soil_layer
        for i, groundwater_entry in enumerate(groundwater_list):
            # add loop for different groundwater flow directions
            offset = np.abs(groundwater_entry[2]) in np.cumsum(layer)
            start_id = groundwater_entry[0] + i + int(not offset) - gw_counter
            end_id = groundwater_entry[3] + i + int(not offset) - gw_counter
            if offset:
                gw_counter += 1
            model.addPhysicalGroup(
                2,
                gw_downstream_tags[i][start_id:end_id],
                tag=k + 3,
                name=f"Groundwater_downstream_{i}",
            )
            model.addPhysicalGroup(
                2,
                gw_upstream_tags[i][start_id:end_id],
                tag=k + 4,
                name=f"Groundwater_upstream_{i}",
            )
            k += 2

    def _mesh_prism() -> None:
        # define the outer boundaries square
        point_model_area_registry: dict[int, tuple[float, ...]] = {}
        if shapely.get_coordinate_dimension(model_area) == 2:
            for x, y in model_area.exterior.coords[:-1]:
                point_model_area_registry[
                    geo.addPoint(x, y, 0, outer_mesh_size)
                ] = (x, y, 0)
        else:
            for x, y, z in model_area.exterior.coords[:-1]:
                point_model_area_registry[
                    geo.addPoint(x, y, z, outer_mesh_size)
                ] = (x, y, z)
        point_ids = list(point_model_area_registry.keys())
        lines = list(zip(point_ids, point_ids[1:] + point_ids[:1], strict=True))
        line_ids = []
        gw_upstream: list[list[int]] = [[] for _ in range(len(groundwaters))]
        gw_downstream: list[list[int]] = [[] for _ in range(len(groundwaters))]
        ref_vec = [1, 0]  # in positive x-dir
        for i, (p1, p2) in enumerate(lines):
            line_ids.append(geo.addLine(p1, p2))
            p1_x, p1_y = point_model_area_registry[p1][:2]
            p2_x, p2_y = point_model_area_registry[p2][:2]
            dx = p2_x - p1_x
            dy = p2_y - p1_y
            normal_vec = [dy, -dx]

            # Calculate dot product
            dot_product = np.dot(ref_vec, normal_vec)

            # Calculate magnitudes (lengths of the vectors)
            magnitude_A = np.linalg.norm(ref_vec)
            magnitude_B = np.linalg.norm(normal_vec)

            # Calculate angle in radians
            angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))

            # Convert radians to degrees
            normal_vec_angle = (
                360 - np.degrees(angle_radians)
                if normal_vec[1] < 0
                else np.degrees(angle_radians)
            )
            for i_gw, groundwater in enumerate(groundwaters):
                if groundwater.upstream[0] > groundwater.upstream[1]:
                    if (
                        normal_vec_angle > groundwater.upstream[0]
                        or normal_vec_angle < groundwater.upstream[1]
                    ):
                        gw_upstream[i_gw].append(i)
                elif (
                    normal_vec_angle > groundwater.upstream[0]
                    and normal_vec_angle < groundwater.upstream[1]
                ):
                    gw_upstream[i_gw].append(i)

                if groundwater.downstream[0] > groundwater.downstream[1]:
                    if (
                        normal_vec_angle > groundwater.downstream[0]
                        or normal_vec_angle < groundwater.downstream[1]
                    ):
                        gw_downstream[i_gw].append(i)
                elif (
                    normal_vec_angle > groundwater.downstream[0]
                    and normal_vec_angle < groundwater.downstream[1]
                ):
                    gw_downstream[i_gw].append(i)

        if groundwaters:
            if not _flatten_concatenation(gw_downstream):
                msg = "No groundwater upstream surfaces detected, please check your specified angles!"
                raise ValueError(msg)
            if not _flatten_concatenation(gw_downstream):
                msg = "No groundwater downstream surfaces detected, please check your specified angles!"
                raise ValueError(msg)

        geo.addCurveLoop(line_ids, 1)
        geo.addPlaneSurface([1], 1)
        geo.synchronize()

        bhe_top_nodes = _insert_BHE(inTag=1)

        # Extrude the surface mesh according to the previously evaluated structure
        volumes_list_for_layers = []
        top_surface = [1]

        surface_list = [(2, 1)]
        gw_downstream_tags: list[list[int]] = [
            [] for _ in range(len(groundwaters))
        ]
        gw_upstream_tags: list[list[int]] = [
            [] for _ in range(len(groundwaters))
        ]

        for j, num_elements in enumerate(number_of_layers):
            # spacing of the each layer must be evaluated according to the implementation of the bhe
            extrusion_tags = gmsh.model.geo.extrude(
                surface_list,
                0,
                0,
                -depth_of_extrusion[j],
                num_elements,
                cummulative_height_of_layers[j],
                True,
            )  # soil 1

            # list of new bottom surfaces, extruded by the five surfaces
            surface_list = [extrusion_tags[0]]

            for i_gw in range(len(groundwaters)):
                for i_tags in range(len(gw_upstream)):
                    for tag in gw_upstream[i_tags]:
                        gw_upstream_tags[i_gw].append(
                            extrusion_tags[tag + 2][1]
                        )
                    for tag in gw_downstream[i_tags]:
                        gw_downstream_tags[i_gw].append(
                            extrusion_tags[tag + 2][1]
                        )

            volumes_list_for_layers.append([extrusion_tags[1][1]])

        BHE_group = []

        for i, BHE_i in enumerate(BHE_array):
            extrusion_tags = gmsh.model.geo.extrude(
                [(0, bhe_top_nodes[i])],
                0,
                0,
                BHE_i.z_end - BHE_i.z_begin,
                BHE_extrusion_layers[i],
                BHE_extrusion_depths[i],
            )
            BHE_group.append(extrusion_tags[1][1])

        geo.synchronize()
        for i, volume_i in enumerate(volumes_list_for_layers):
            model.addPhysicalGroup(3, volume_i, i)

        for k, BHE_group_k in enumerate(BHE_group, start=i + 1):
            model.addPhysicalGroup(1, [BHE_group_k], k)

        model.addPhysicalGroup(2, top_surface, k + 1, name="Top_Surface")
        model.addPhysicalGroup(
            2,
            np.array(surface_list)[:, 1].tolist(),
            k + 2,
            name="Bottom_Surface",
        )

        gw_counter = 0  # counter_for_gw_start_at_soil_layer
        for i, groundwater_entry in enumerate(groundwater_list):
            # add loop for different groundwater flow directions
            offset = np.abs(groundwater_entry[2]) in np.cumsum(layer)
            start_id = groundwater_entry[0] + i + int(not offset) - gw_counter
            end_id = groundwater_entry[3] + i + int(not offset) - gw_counter
            if offset:
                gw_counter += 1
            model.addPhysicalGroup(
                2,
                gw_downstream_tags[i][start_id:end_id],
                tag=k + 3,
                name=f"Groundwater_downstream_{i}",
            )
            model.addPhysicalGroup(
                2,
                gw_upstream_tags[i][start_id:end_id],
                tag=k + 4,
                name=f"Groundwater_upstream_{i}",
            )
            k += 2

        def mesh_size_callback(
            _dim: int, _tag: int, x: float, y: float, z: float, lc: float
        ) -> float:
            if refinement_area.contains(shapely.Point(x, y, z)):
                return min(lc, inner_mesh_size)
            return min(lc, outer_mesh_size)

        mesh.setSizeCallback(callback=mesh_size_callback)

    layer = layer if isinstance(layer, list) else [layer]

    groundwaters: list[Groundwater] = (
        [groundwater] if isinstance(groundwater, Groundwater) else groundwater
    )

    BHE_Array = [BHE_Array] if isinstance(BHE_Array, BHE) else BHE_Array

    model_area = shapely.orient_polygons(model_area)
    refinement_area = shapely.orient_polygons(refinement_area)

    # detect the soil layer, in which the groundwater flow starts
    groundwater_list: list[list] = []
    for groundwater in groundwaters:
        start_groundwater = -1000
        # Index for critical layer structure, 0: not critical, 1: top critical,
        # 2: bottom critical, 3: groundwater at layer transition
        icl: float = -1
        # needed_medias_in_ogs=len(layer)+1
        for i, _ in enumerate(layer):
            if (
                np.abs(groundwater.begin) < np.sum(layer[: i + 1])
                and start_groundwater == -1000
            ):
                start_groundwater = i

                if (  # previous elif, one semantic block of different cases -> switch to if, because of ruff error
                    np.abs(groundwater.begin)
                    - np.sum(layer[:start_groundwater])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - GW')
                    icl = 1
                    # beginning of groundwater at a transition of two soil layers - special case
                    if np.abs(groundwater.begin) == np.sum(
                        layer[:start_groundwater]
                    ):
                        icl = 3
                        # needed_medias_in_ogs=len(layer)
                        # needed_extrusions=len(layer)
                    elif (
                        np.sum(layer[: start_groundwater + 1])
                        - np.abs(groundwater.begin)
                        < n_refinement_layers * target_z_size_fine
                    ):
                        # for layers, which are top and bottom critical
                        icl = 1.2
                elif (
                    np.sum(layer[: start_groundwater + 1])
                    - np.abs(groundwater.begin)
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
                groundwater.begin,
                groundwater.isolation_layer_id,
                groundwater.downstream,
                groundwater.upstream,
            ]
        )

    ### Start of the algorithm ###
    BHE_array = np.asarray(BHE_Array)

    i = 0

    BHE_to_soil = np.zeros(
        shape=(len(BHE_array),),
        dtype=[
            ("BHE_index", np.uint16),
            ("BHE_start_layer", np.uint8),
            # Soil layer, in which the respective BHE starts
            ("BHE_start_critical", np.float16),
            # define, where is a critical transition for BHE start with  0 - not critical, 1 - top critical, 2 - bottom critical, 3 - bhe at layer transition
            ("BHE_end_layer", np.uint8),
            # Soil layer, in which the respective BHE ends
            ("BHE_end_critical", np.float16),
            # define, where is a critical transition for BHE end with see BHE_start_critical plus 1.2 - top and bottom critical
        ],
    )

    # detect the soil layer, in which the BHE ends
    for j, BHE_j in enumerate(BHE_array):
        if BHE_j.z_end >= BHE_j.z_begin:  # pragma: no cover
            msg = f"BHE end depth must be smaller than BHE begin depth for BHE {j}"
            raise ValueError(msg)
        if BHE_j.z_begin > 0:  # pragma: no cover
            msg = "BHE begin depth must be zero or negative for BHE {j}"
            raise ValueError(msg)
        for i, _ in enumerate(layer):
            # detect the soil layer, in which the BHE starts - for the moment only for detection
            if np.abs(BHE_j.z_begin) < np.sum(layer[: i + 1]) and np.abs(
                BHE_j.z_begin
            ) >= np.sum(layer[:i]):
                BHE_to_soil[j]["BHE_index"] = j
                BHE_to_soil[j]["BHE_start_layer"] = i
                if (
                    np.abs(BHE_j.z_end) - np.abs(BHE_j.z_begin)
                    <= n_refinement_layers * target_z_size_fine
                ):  # pragma: no cover
                    msg = "BHE to short, must be longer than n_refinement_layers * target_z_size_fine!"
                    raise ValueError(msg)
                if (  # previous elif, one semantic block of different cases -> switch to if, because of ruff error
                    np.abs(BHE_j.z_begin) - np.sum(layer[:i])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - BHE  %d'%j)
                    # beginning a transition of two soil layers - special case
                    BHE_to_soil[j]["BHE_start_critical"] = 1
                    if np.abs(BHE_j.z_begin) == np.sum(layer[:i]):
                        BHE_to_soil[j]["BHE_start_critical"] = 3
                elif (
                    np.sum(layer[: i + 1]) - np.abs(BHE_j.z_begin)
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('critical at the bottom of the soil layer - BHE %d'%j)
                    BHE_to_soil[j]["BHE_start_critical"] = 2
                else:
                    BHE_to_soil[j]["BHE_start_critical"] = 0

            # detect the soil layer, in which the BHE ends
            if np.abs(BHE_j.z_end) < np.sum(layer[: i + 1]) and np.abs(
                BHE_j.z_end
            ) >= np.sum(layer[:i]):
                BHE_to_soil[j]["BHE_end_layer"] = i
                if (
                    np.abs(BHE_j.z_end) - np.sum(layer[:i])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - BHE  %d'%j)
                    BHE_to_soil[j]["BHE_end_critical"] = 1
                    # beginning at a transition of two soil layers - special case
                    if np.abs(BHE_j.z_end) == np.sum(layer[:i]):
                        BHE_to_soil[j]["BHE_end_critical"] = 3

                    elif (
                        np.sum(layer[: i + 1]) - np.abs(BHE_j.z_end)
                        < n_refinement_layers * target_z_size_fine
                    ):
                        # for layers, which are top and bottom critical
                        BHE_to_soil[j]["BHE_end_critical"] = 1.2
                elif (
                    np.sum(layer[: i + 1]) - np.abs(BHE_j.z_end)
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('critical at the bottom of the soil layer - BHE %d'%j)
                    BHE_to_soil[j]["BHE_end_critical"] = 2
                else:
                    BHE_to_soil[j]["BHE_end_critical"] = 0
            elif np.abs(BHE_j.z_end) >= np.sum(layer):  # pragma: no cover
                msg = f"BHE {j} ends at bottom boundary or outside of the model"
                raise ValueError(msg)

    needed_depths: list = []  # interesting depths
    for i, _ in enumerate(layer):
        # only the interesting depths in the i-th layer
        # TODO: Rename the variable
        BHE_end_depths = []

        # filter, which BHE's ends in this layer
        BHE_end_in_Layer = BHE_to_soil[BHE_to_soil["BHE_end_layer"] == i]

        for k in BHE_end_in_Layer["BHE_index"]:
            BHE_end_depths.append(
                [BHE_array[k].z_end, BHE_to_soil[k]["BHE_end_critical"]]
            )

        # filter, which BHE's starts in this layer
        BHE_starts_in_Layer = BHE_to_soil[BHE_to_soil["BHE_start_layer"] == i]

        for k in BHE_starts_in_Layer["BHE_index"]:
            BHE_end_depths.append(
                [BHE_array[k].z_begin, BHE_to_soil[k]["BHE_start_critical"]]
            )

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
    for i, _ in enumerate(layer):  # Schleife zum Berechnen der Layer-Struktur
        # all depths, which needs a node in the mesh
        list_of_needed_depths = needed_depths[i]

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
    for i, num_layer in enumerate(number_of_layers):
        All_extrusion_depths.append(
            (
                depth_of_extrusion[i]
                * np.array(cummulative_height_of_layers[i])
                + last_height
            ).tolist()
        )
        last_height = All_extrusion_depths[-1][-1]
        All_extrusion_layers.append(num_layer)

    all_extrusion = np.array(
        [
            _flatten_concatenation(All_extrusion_depths),
            _flatten_concatenation(All_extrusion_layers),
        ]
    ).transpose()

    BHE_extrusion_layers: list = []
    BHE_extrusion_depths: list = []
    # evaluate the extrusion for the BHE's
    for BHE_i in BHE_array:
        # add little relax tolerance 0.001
        needed_extrusion = all_extrusion[
            (
                (all_extrusion[:, 0] >= np.abs(BHE_i.z_begin))
                & (all_extrusion[:, 0] <= np.abs(BHE_i.z_end) + 0.001)
            )
        ]

        BHE_extrusion_layers.append(needed_extrusion[:, 1])
        BHE_extrusion_depths.append(
            (needed_extrusion[:, 0] - np.abs(BHE_i.z_begin))
            / (needed_extrusion[-1, 0] - np.abs(BHE_i.z_begin))
        )

    outer_mesh_size_inner = (outer_mesh_size + inner_mesh_size) / 2

    with TemporaryDirectory() as tmpdir:
        msh_file = Path(tmpdir) / f"{meshname}.msh"
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        model = gmsh.model
        geo = model.geo
        mesh = model.mesh

        model.add(msh_file.stem)

        if meshing_type == "structured":
            _mesh_structured()
        elif meshing_type == "prism":
            _mesh_prism()

        mesh.generate(3)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        mesh.setOrder(order)
        mesh.removeDuplicateNodes()

        # delete zero-volume elements
        # 1 for line elements --> BHE's are the reason
        elem_tags, node_tags = mesh.getElementsByType(1)
        elem_qualities = mesh.getElementQualities(
            elementTags=elem_tags, qualityName="volume"
        )
        zero_volume_elements_id = np.argwhere(elem_qualities == 0)

        # only possible with the hack over the visibilitiy, see https://gitlab.onelab.info/gmsh/gmsh/-/issues/2006
        mesh.setVisibility(
            elem_tags[zero_volume_elements_id].ravel().tolist(), 0
        )
        gmsh.plugin.setNumber("Invisible", "DeleteElements", 1)
        gmsh.plugin.run("Invisible")

        gmsh.write(str(msh_file))
        gmsh.finalize()
        return ot.Meshes.from_gmsh(
            msh_file, dim=[1, 3], log=False, meshname=meshname
        )
