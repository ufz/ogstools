# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections.abc import Collection
from itertools import pairwise
from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv


def _line(
    geo: gmsh.model.geo, length: float, n_edge_cells: int, structured: bool
) -> int:
    geo.addPoint(0, 0, 0, tag=1)
    geo.addPoint(length, 0, 0, tag=2)
    line_tag = geo.addLine(1, 2, tag=1)

    if structured:
        geo.mesh.setTransfiniteCurve(1, n_edge_cells + 1)
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
    layer_ids: list | None = None,
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
    if layer_ids is None:
        layer_ids = list(range(n_layers))
    assert isinstance(layer_ids, list)
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
        tag = layer_ids[n]
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


def remesh_with_triangles(
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
    # gmsh requires the domain ids to start at 1
    id_offset = 1 if 0 in mat_ids else 0
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
        mat_id = mat_ids[index] + id_offset
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
