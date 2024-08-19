# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Collection
from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from pathlib import Path
from tempfile import mkdtemp

import gmsh
import numpy as np
import pyvista as pv

from ogstools.msh2vtu import msh2vtu


def _line(
    geo: gmsh.model.geo, length: float, n_edge_cells: int, structured: bool
) -> int:
    geo.addPoint(0, 0, 0, tag=1)
    geo.addPoint(length, 0, 0, tag=2)
    line_tag = geo.addLine(1, 2, tag=1)
    geo.mesh.setTransfiniteCurve(1, n_edge_cells + 1)

    if structured:
        geo.mesh.setTransfiniteSurface(1)
        geo.mesh.setRecombine(dim=1, tag=1)
    return line_tag


def rect(
    lengths: float | tuple[float, float] = 1.0,
    n_edge_cells: int | tuple[int, int] = 1,
    n_layers: int = 1,
    structured_grid: bool = True,
    order: int = 1,
    mixed_elements: bool = False,
    jiggle: float = 0.0,
    out_name: Path | str = Path("rect.msh"),
    msh_version: float | None = None,
) -> None:
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
    if msh_version is not None:
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
    name = Path(out_name).stem
    gmsh.model.add(name)
    dx, dy = lengths if isinstance(lengths, Collection) else [lengths] * 2
    nx, ny = (
        n_edge_cells
        if isinstance(n_edge_cells, Collection)
        else [n_edge_cells] * 2
    )
    bottom_tag = _line(gmsh.model.geo, dx, nx, structured_grid)
    right_tags = []
    left_tags = []
    top_tag = bottom_tag
    for n in range(n_layers):
        recombine = n % 2 if mixed_elements else structured_grid
        newEntities = gmsh.model.geo.extrude(
            dimTags=[(1, top_tag)],
            dx=0,
            dy=dy / n_layers,
            dz=0,
            numElements=[ny] if structured_grid else [],
            recombine=recombine,  # fmt: skip
        )
        top_tag = abs(newEntities[0][1])
        plane_tag = abs(newEntities[1][1])
        layer_name = f"Layer {n}" if n_layers > 1 else name
        tag = -1 if n_layers > 1 else 0
        gmsh.model.addPhysicalGroup(
            dim=2, tags=[plane_tag], name=layer_name, tag=tag
        )
        right_tags += [abs(newEntities[2][1])]
        left_tags += [abs(newEntities[3][1])]

    gmsh.model.addPhysicalGroup(dim=1, tags=[bottom_tag], name="bottom")
    gmsh.model.addPhysicalGroup(dim=1, tags=[top_tag], name="top")
    gmsh.model.addPhysicalGroup(dim=1, tags=right_tags, name="right")
    gmsh.model.addPhysicalGroup(dim=1, tags=left_tags, name="left")

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    if not structured_grid:
        gmsh.option.setNumber("Mesh.MeshSizeMin", dy / ny)
        gmsh.option.setNumber("Mesh.MeshSizeMax", dy / ny)
    gmsh.model.mesh.generate(dim=2)
    if jiggle > 0.0:
        node_ids = gmsh.model.mesh.getNodes(dim=2)[0]
        for node_id in node_ids:
            offset = np.append(np.random.default_rng().random(2), 0.0)
            coord, parametric_coord, _, tag = gmsh.model.mesh.get_node(node_id)
            coord = np.asarray(coord) + offset * jiggle
            gmsh.model.mesh.set_node(node_id, coord, parametric_coord)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(out_name))
    gmsh.finalize()


def _square(
    geo: gmsh.model.geo,
    lengths: tuple[float, float],
    n_edge_cells: tuple[int, int],
    structured: bool,
) -> int:
    geo.addPoint(0, 0, 0, tag=1)
    geo.addPoint(lengths[0], 0, 0, tag=2)
    geo.addPoint(lengths[0], lengths[1], 0, tag=3)
    geo.addPoint(0, lengths[1], 0, tag=4)

    geo.addLine(1, 2, tag=1)
    geo.addLine(2, 3, tag=2)
    geo.addLine(3, 4, tag=3)
    geo.addLine(4, 1, tag=4)

    geo.addCurveLoop([1, 2, 3, 4], tag=1)
    plane_tag = geo.addPlaneSurface([1], tag=1)

    geo.mesh.setTransfiniteCurve(1, n_edge_cells[0] + 1)
    geo.mesh.setTransfiniteCurve(2, n_edge_cells[1] + 1)
    geo.mesh.setTransfiniteCurve(3, n_edge_cells[0] + 1)
    geo.mesh.setTransfiniteCurve(4, n_edge_cells[1] + 1)

    if structured:
        geo.mesh.setTransfiniteSurface(1)
        geo.mesh.setRecombine(dim=2, tag=1)
    return plane_tag


def cuboid(
    lengths: float | tuple[float, float, float] = 1.0,
    n_edge_cells: int | tuple[int, int, int] = 1,
    n_layers: int = 1,
    structured_grid: bool = True,
    order: int = 1,
    mixed_elements: bool = False,
    out_name: Path = Path("unit_cube.msh"),
    msh_version: float | None = None,
) -> None:
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
    if msh_version is not None:
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
    name = Path(out_name).stem
    gmsh.model.add(name)
    _lengths = lengths if isinstance(lengths, Collection) else (lengths,) * 3
    _n_edge_cells = (
        n_edge_cells
        if isinstance(n_edge_cells, Collection)
        else (n_edge_cells,) * 3
    )
    bottom_tag = _square(
        gmsh.model.geo,
        _lengths[1:],
        _n_edge_cells[1:],
        structured_grid and not mixed_elements,
    )

    dz = lengths if isinstance(lengths, float) else lengths[2]
    nz = n_edge_cells if isinstance(n_edge_cells, int) else n_edge_cells[2]
    front_tags = []
    right_tags = []
    back_tags = []
    left_tags = []
    top_tag = 1
    for n in range(n_layers):
        recombine = n % 2 if mixed_elements else structured_grid
        newEntities = gmsh.model.geo.extrude(
            dimTags=[(2, top_tag)],
            dx=0,
            dy=0,
            dz=dz / n_layers,
            numElements=[nz] if structured_grid else [],
            recombine=recombine,  # fmt: skip
        )
        top_tag = abs(newEntities[0][1])
        vol_tag = abs(newEntities[1][1])
        layer_name = f"Layer {n}" if n_layers > 1 else name
        tag = -1 if n_layers > 1 else 0
        gmsh.model.addPhysicalGroup(
            dim=3, tags=[vol_tag], name=layer_name, tag=tag
        )
        front_tags += [abs(newEntities[2][1])]
        right_tags += [abs(newEntities[3][1])]
        back_tags += [abs(newEntities[4][1])]
        left_tags += [abs(newEntities[5][1])]

    gmsh.model.addPhysicalGroup(dim=2, tags=[bottom_tag], name="bottom")
    gmsh.model.addPhysicalGroup(dim=2, tags=[top_tag], name="top")
    gmsh.model.addPhysicalGroup(dim=2, tags=front_tags, name="front")
    gmsh.model.addPhysicalGroup(dim=2, tags=right_tags, name="right")
    gmsh.model.addPhysicalGroup(dim=2, tags=back_tags, name="back")
    gmsh.model.addPhysicalGroup(dim=2, tags=left_tags, name="left")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=3)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    if not structured_grid:
        gmsh.option.setNumber(
            "Mesh.MeshSizeMin", _lengths[0] / _n_edge_cells[0]
        )
        gmsh.option.setNumber(
            "Mesh.MeshSizeMax", _lengths[0] / _n_edge_cells[0]
        )
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(out_name))
    gmsh.finalize()


def _ordered_edges(mesh: pv.UnstructuredGrid) -> np.ndarray:
    "Return edge elements ordered to form a contiguous array."
    edges = mesh.extract_feature_edges()
    n_cells = edges.n_cells
    # shape=(n_cells, 2, 3), the 2 is for pointA and pointB
    cell_pts = np.asarray([cell.points for cell in edges.cell])

    ordered_cell_ids = [0]
    cell_id = 0
    for _ in range(n_cells):
        next_id = np.argmax(
            np.equal(cell_pts[cell_id, 1], cell_pts[:, 0]).all(axis=1)
        )
        ordered_cell_ids += [int(next_id)]
        cell_id = int(next_id)
    return cell_pts[ordered_cell_ids[:-1], 0]


def remesh_with_triangle(
    mesh: pv.UnstructuredGrid,
    output_file: Path | str = Path() / "tri_mesh.msh",
    size_factor: float = 1.0,
    order: int = 1,
) -> None:
    """Discretizes a given Mesh with triangles and saves as gmsh .msh.

    Requires the mesh to be 2D and to contain 'MaterialIDs in the cell data.

    :param mesh:        The mesh which shall be discretized with triangles
    :param output_file: The full filepath to the resulting file
    :param size_factor: A factor to scale the element sizes.
    :param order:       The element order (1=linear, 2=quadratic, ...)
    """

    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
    gmsh.clear()
    gmsh.model.add("domain")

    mat_ids = np.unique(mesh["MaterialIDs"])
    region_edge_points = [
        _ordered_edges(mesh.threshold([m, m], "MaterialIDs")) for m in (mat_ids)
    ]
    for point in np.vstack(region_edge_points):
        gmsh.model.geo.addPoint(point[0], point[1], point[2])
    region_lengths = [len(pts) for pts in region_edge_points]
    region_start_tag = np.cumsum([0] + region_lengths) + 1
    for tag_0, tag_1 in pairwise(region_start_tag):
        for index in range(tag_0, tag_1 - 1):
            gmsh.model.geo.addLine(index, index + 1)
        gmsh.model.geo.addLine(tag_1 - 1, tag_0)

    field = gmsh.model.mesh.field
    for index, points in enumerate(region_edge_points):
        mat_id = mat_ids[index]
        line_tags = range(region_start_tag[index], region_start_tag[index + 1])
        gmsh.model.geo.addCurveLoop(line_tags, tag=mat_id)
        gmsh.model.geo.addPlaneSurface([mat_id], tag=mat_id)
        gmsh.model.addPhysicalGroup(
            dim=2, tags=[mat_id], name=f"Layer {mat_id}"
        )
        f1, f2 = (2 * mat_id, 2 * mat_id + 1)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        DEFAULT_SIZE = 6  # pylint: disable=C0103
        min_elem_size = size_factor * np.min(distances) * DEFAULT_SIZE
        max_elem_size = size_factor * np.max(distances) * DEFAULT_SIZE
        field.add("Distance", f1)
        field.setNumbers(f1, "CurvesList", line_tags)
        field.add("Threshold", f2)
        field.setNumber(f2, "IField", f1)
        field.setNumber(f2, "SizeMin", min_elem_size)
        field.setNumber(f2, "SizeMax", max_elem_size)
        field.setNumber(f2, "DistMin", min_elem_size)
        field.setNumber(f2, "DistMax", 3 * max_elem_size)

    field.add("Min", tag=f2 + 1)
    field.setNumbers(f2 + 1, "FieldsList", (mat_ids * 2 + 1).tolist())
    field.setAsBackgroundMesh(f2 + 1)

    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(output_file))
    gmsh.finalize()
    return


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
    layer: float | list[float],
    groundwater: Groundwater | list[Groundwater],
    BHE_Array: BHE | list[BHE],
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
    def _flatten_concatenation(matrix: list[list[float]]) -> list:
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
        for y_value in [0.0, y_min, y_max, width]:
            for x_value in [0.0, x_min, x_max, length]:
                gmsh.model.geo.addPoint(x_value, y_value, 0.0)

        # x-direction
        for i in range(14)[1::4]:
            for j in range(3):
                gmsh.model.geo.addLine(i + j, i + j + 1)

        # y-direction
        for i in range(1, 13):
            gmsh.model.geo.addLine(i, i + 4)

        # add surfaces
        for i in range(1, 10):
            i2 = i + 13 + (i - 1) // 3
            gmsh.model.geo.addCurveLoop([i, i2, -i - 3, -i2 + 1])
            gmsh.model.geo.addPlaneSurface([i])
            gmsh.model.geo.synchronize()

        bhe_top_nodes = _insert_BHE(inTag=5)

        # Extrude the surface mesh according to the previously evaluated structure
        volumes_list_for_layers = []
        bounds_gw: dict[str, list] = {"+x": [], "-x": [], "+y": [], "-y": []}
        boundaries_surfaces = {
            "+x": [23, 41, 5],
            "-x": [33, 15, 51],
            "+y": [2, 8, 14],
            "-y": [40, 46, 52],
        }
        top_surface = list(range(1, 10))
        surface_list = [(2, tag) for tag in top_surface]

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
            # list of volume numbers and new bottom surfaces, which were extruded by the five surfaces
            volume_list = [extrusion_tags[1 + i * 6][1] for i in range(9)]
            surface_list = [extrusion_tags[i * 6] for i in range(9)]

            for direction, boundary_tags in bounds_gw.items():
                for surf in boundaries_surfaces[direction]:
                    boundary_tags.append(extrusion_tags[surf][1])

            volumes_list_for_layers.append(volume_list)

        k = 0
        BHE_group = []
        for i, BHE_i in enumerate(BHE_array):
            extrusion_tags = gmsh.model.geo.extrude(
                [(0, bhe_top_nodes[i])],
                0,
                0,
                BHE_i.z_end - BHE_i.z_begin,
                BHE_extrusion_layers[i],
                BHE_extrusion_depths[i],
                True,
            )
            BHE_group.append(extrusion_tags[1][1])

        gmsh.model.geo.synchronize()

        for i in range(len(number_of_layers)):
            gmsh.model.addPhysicalGroup(3, volumes_list_for_layers[i], i)
            k = i

        for i in range(len(BHE_array)):
            gmsh.model.addPhysicalGroup(1, [BHE_group[i]], k + 1)
            k += 1

        gmsh.model.addPhysicalGroup(2, top_surface, k + 1, name="Top_Surface")
        gmsh.model.addPhysicalGroup(
            2,
            np.array(surface_list)[:, 1].tolist(),
            k + 2,
            name="Bottom_Surface",
        )

        gw_counter = 0  # counter_for_gw_start_at_soil_layer
        for i, groundwater in enumerate(groundwater_list):
            # add loop for different groundwater flow directions
            offset = np.abs(groundwater[2]) in np.cumsum(layer)
            start_id = 3 * (groundwater[0] + i + int(not offset) - gw_counter)
            end_id = 3 * (groundwater[3] + i + int(not offset) - gw_counter)
            if offset:
                gw_counter += 1
            gmsh.model.addPhysicalGroup(
                2,
                bounds_gw[groundwater[4]][start_id:end_id],
                tag=k + 3,
                name=f"Groundwater_Inflow_{i}",
            )
            k += 1

        mesh = gmsh.model.mesh
        # Sizing Functions and Transfinite Algorithm for Hexahedron meshing in wanted zones
        # inner square
        delta_x, delta_y = (x_max - x_min, y_max - y_min)
        mesh.setTransfiniteCurve(5, ceil(delta_x / inner_mesh_size) + 1)
        mesh.setTransfiniteCurve(8, ceil(delta_x / inner_mesh_size) + 1)

        mesh.setTransfiniteCurve(17, ceil(delta_y / inner_mesh_size) + 1)
        mesh.setTransfiniteCurve(18, ceil(delta_y / inner_mesh_size) + 1)
        mesh.setTransfiniteCurve(19, ceil(delta_y / inner_mesh_size) + 1)
        mesh.setTransfiniteCurve(20, ceil(delta_y / inner_mesh_size) + 1)

        mesh.setTransfiniteCurve(13, ceil(y_min / outer_mesh_size_inner) + 1)
        mesh.setTransfiniteCurve(14, ceil(y_min / outer_mesh_size_inner) + 1)
        mesh.setTransfiniteCurve(15, ceil(y_min / outer_mesh_size_inner) + 1)
        mesh.setTransfiniteCurve(16, ceil(y_min / outer_mesh_size_inner) + 1)

        ny = ceil((width - y_max) / outer_mesh_size_inner) + 1
        mesh.setTransfiniteCurve(21, ny)
        mesh.setTransfiniteCurve(22, ny)
        mesh.setTransfiniteCurve(23, ny)
        mesh.setTransfiniteCurve(24, ny)

        mesh.setTransfiniteCurve(2, ceil(delta_x / outer_mesh_size_inner) + 1)
        mesh.setTransfiniteCurve(11, ceil(delta_x / outer_mesh_size_inner) + 1)

        # rectangular squares bgw
        mesh.setTransfiniteCurve(1, ceil(x_min / outer_mesh_size) + 1)
        mesh.setTransfiniteCurve(4, ceil(x_min / outer_mesh_size) + 1)
        mesh.setTransfiniteCurve(7, ceil(x_min / outer_mesh_size) + 1)
        mesh.setTransfiniteCurve(10, ceil(x_min / outer_mesh_size) + 1)

        num_nodes = ceil((length - x_max) / outer_mesh_size) + 1
        mesh.setTransfiniteCurve(3, num_nodes, "Progression", propagation)
        mesh.setTransfiniteCurve(6, num_nodes, "Progression", propagation)
        mesh.setTransfiniteCurve(9, num_nodes, "Progression", propagation)
        mesh.setTransfiniteCurve(12, num_nodes, "Progression", propagation)

        for i, surface_tag in enumerate([1, 3, 4, 6, 7, 9]):
            j = 2 * i
            corner_tags = [1 + j, 2 + j, 5 + j, 6 + j]
            mesh.setTransfiniteSurface(surface_tag, cornerTags=corner_tags)
            mesh.setRecombine(2, surface_tag)
        mesh.recombine()

    def _mesh_prism() -> None:
        # define the outer boundaries square
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, outer_mesh_size)
        gmsh.model.geo.addPoint(length, 0.0, 0.0, outer_mesh_size)
        gmsh.model.geo.addPoint(length, width, 0.0, outer_mesh_size)
        gmsh.model.geo.addPoint(0.0, width, 0.0, outer_mesh_size)

        gmsh.model.geo.addLine(1, 2)
        gmsh.model.geo.addLine(2, 3)
        gmsh.model.geo.addLine(3, 4)
        gmsh.model.geo.addLine(4, 1)

        # inner points
        gmsh.model.geo.addPoint(x_min, y_min, 0.0, inner_mesh_size)
        gmsh.model.geo.addPoint(x_max, y_min, 0.0, inner_mesh_size)
        gmsh.model.geo.addPoint(x_max, y_max, 0.0, inner_mesh_size)
        gmsh.model.geo.addPoint(x_min, y_max, 0.0, inner_mesh_size)

        gmsh.model.geo.addLine(5, 6)
        gmsh.model.geo.addLine(6, 7)
        gmsh.model.geo.addLine(7, 8)
        gmsh.model.geo.addLine(8, 5)

        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()

        # embed the four lines of the inner sizing box
        gmsh.model.mesh.embed(1, [5, 6, 7, 8], 2, 1)
        bhe_top_nodes = _insert_BHE(inTag=1)

        # Extrude the surface mesh according to the previously evaluated structure
        volumes_list_for_layers = []
        top_surface = [1]

        surface_list = [(2, 1)]
        bounds_gw: dict[str, list] = {"+x": [], "-x": [], "+y": [], "-y": []}
        boundaries_surfaces = {"+x": [5], "-x": [3], "+y": [2], "-y": [4]}
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

            for direction, boundary_tags in bounds_gw.items():
                for surf in boundaries_surfaces[direction]:
                    boundary_tags.append(extrusion_tags[surf][1])

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

        gmsh.model.geo.synchronize()
        k = 0

        # len(layer) is right, but len(number_of_layers) for testing only
        for i in range(len(number_of_layers)):
            gmsh.model.addPhysicalGroup(3, volumes_list_for_layers[i], i)
            k = i

        for i in range(len(BHE_array)):
            gmsh.model.addPhysicalGroup(1, [BHE_group[i]], k + 1)
            k += 1

        gmsh.model.addPhysicalGroup(2, top_surface, k + 1, name="Top_Surface")
        gmsh.model.addPhysicalGroup(
            2,
            np.array(surface_list)[:, 1].tolist(),
            k + 2,
            name="Bottom_Surface",
        )

        gw_counter = 0  # counter_for_gw_start_at_soil_layer
        for i, groundwater in enumerate(groundwater_list):
            # add loop for different groundwater flow directions
            offset = np.abs(groundwater[2]) in np.cumsum(layer)
            start_id = groundwater[0] + i + int(not offset) - gw_counter
            end_id = groundwater[3] + i + int(not offset) - gw_counter
            if offset:
                gw_counter += 1
            gmsh.model.addPhysicalGroup(
                2,
                bounds_gw[groundwater[4]][start_id:end_id],
                tag=k + 3,
                name=f"Groundwater_Inflow_{i}",
            )
            k += 1

    layer = layer if isinstance(layer, list) else [layer]

    groundwaters: list[Groundwater] = (
        [groundwater] if isinstance(groundwater, Groundwater) else groundwater
    )

    BHE_Array = [BHE_Array] if isinstance(BHE_Array, BHE) else BHE_Array

    # detect the soil layer, in which the groundwater flow starts
    groundwater_list: list = []
    for groundwater in groundwaters:
        start_groundwater = -1000
        # Index for critical layer structure, 0: not critical, 1: top critical,
        # 2: bottom critical, 3: groundwater at layer transition
        icl: float = -1
        # needed_medias_in_ogs=len(layer)+1
        for i in range(len(layer)):
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
                groundwater.flow_direction,
            ]
        )

    ### Start of the algorithm ###
    BHE_array = np.asarray(BHE_Array)

    # detect the soil layer, in which the BHE starts - for the moment only for detection
    i = 0
    # [:,0] - index of BHE; Array, in welcher Schicht die jeweilige BHE anfängt [:,1] endet [:,3] und wo ein kritischer Übergang folgt [:,2] für BHE_Begin und [:,4] für BHE_End mit 0 - not critical, 1 - top critical, 2 - bottom critical, 3 - bhe at layer transition
    BHE_to_soil = np.zeros(shape=(len(BHE_array), 5), dtype=np.int8)

    for j, BHE_j in enumerate(BHE_array):
        for i in range(len(layer)):
            # Auswertung für BHE_Beginn
            if np.abs(BHE_j.z_begin) < np.sum(layer[: i + 1]) and np.abs(
                BHE_j.z_begin
            ) >= np.sum(layer[:i]):
                BHE_to_soil[j, 0] = j
                BHE_to_soil[j, 1] = i
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
                    BHE_to_soil[j, 2] = 1
                    if np.abs(BHE_j.z_begin) == np.sum(layer[:i]):
                        BHE_to_soil[j, 2] = 3
                elif (
                    np.sum(layer[: i + 1]) - np.abs(BHE_j.z_begin)
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('critical at the bottom of the soil layer - BHE %d'%j)
                    BHE_to_soil[j, 2] = 2
                else:
                    BHE_to_soil[j, 2] = 0

    # detect the soil layer, in which the BHE ends
    for j, BHE_j in enumerate(BHE_array):
        for i in range(len(layer)):
            if np.abs(BHE_j.z_end) < np.sum(layer[: i + 1]) and np.abs(
                BHE_j.z_end
            ) >= np.sum(layer[:i]):
                BHE_to_soil[j, 3] = i
                if (
                    np.abs(BHE_j.z_end) - np.sum(layer[:i])
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('difficult meshing at the top of the soil layer - BHE  %d'%j)
                    BHE_to_soil[j, 4] = 1
                    # beginning at a transition of two soil layers - special case
                    if np.abs(BHE_j.z_end) == np.sum(layer[:i]):
                        BHE_to_soil[j, 4] = 3

                    elif (
                        np.sum(layer[: i + 1]) - np.abs(BHE_j.z_end)
                        < n_refinement_layers * target_z_size_fine
                    ):
                        # for layers, which are top and bottom critical
                        BHE_to_soil[j, 4] = 1.2
                elif (
                    np.sum(layer[: i + 1]) - np.abs(BHE_j.z_end)
                    < n_refinement_layers * target_z_size_fine
                ):
                    # print('critical at the bottom of the soil layer - BHE %d'%j)
                    BHE_to_soil[j, 4] = 2
                else:
                    BHE_to_soil[j, 4] = 0
            elif np.abs(BHE_j.z_end) >= np.sum(layer):  # pragma: no cover
                msg = f"BHE {j} ends at bottom boundary or outside of the model"
                raise ValueError(msg)

    needed_depths: list = []  # interesting depths
    for i in range(len(layer)):
        # only the interesting depths in the i-th layer
        # TODO: Rename the variable
        BHE_end_depths = []

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
    for i in range(len(layer)):  # Schleife zum Berechnen der Layer-Struktur
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

    # define the inner square with BHE inside
    # compute the box size from the BHE-Coordinates
    x_BHE = [BHE_i.x for BHE_i in BHE_array]
    y_BHE = [BHE_i.y for BHE_i in BHE_array]

    x_min = np.min(x_BHE) - dist_box_x
    x_max = np.max(x_BHE) + dist_box_x
    y_min = np.min(y_BHE) - dist_box_y
    y_max = np.max(y_BHE) + dist_box_y

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
    # 1 for line elements --> BHE's are the reason
    elem_tags, node_tags = gmsh.model.mesh.getElementsByType(1)
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
    layer: float | list[float],  # e.g. 100
    groundwater: Groundwater | list[Groundwater],
    BHE_Array: BHE | list[BHE],
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

    for i in range(len(groundwater)):
        mesh_names.append(
            f"{mesh_name}_physical_group_Groundwater_Inflow_{i}.vtu"
        )

    return mesh_names
