# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
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

from ogstools.core.storage import _date_temp_path


def optional_default_file(
    filepath: Path | str | None, class_id: str, suffix: str
) -> Path:
    filepath = Path(filepath) if filepath else _date_temp_path(class_id, suffix)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def _line(
    geo: type[gmsh.model.geo],
    length: float,
    n_edge_cells: int,
    structured: bool,
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
    out_name: Path | str | None = None,
    msh_version: float | None = None,
    layer_ids: list | None = None,
) -> Path:
    """
    Generates a rectangular mesh using gmsh.

    :param lengths: Length of the rectangle in x and y direction. Provide a tuple (x, y) or a scalar for a square. All values must be >= 1e-7 and <= 1e12.
    :param n_edge_cells: Number of edge cells in x and y direction. Provide a tuple (x, y) or a scalar for a square. All values must be >= 1.
    :param n_layers: Number of layers in y direction. Must be >= 1.
    :param structured_grid: If True, the mesh will be structured. If False, the mesh will be unstructured.
    :param order: Order of the mesh elements. 1 for linear, 2 for quadratic.
    :param mixed_elements: If True, the mesh will be mixed elements. If False, the mesh will be structured.
    :param jiggle: Amount of random displacement to apply to the mesh nodes. Default is 0.0 (no displacement).
    :param out_name: Name of the output mesh file. Default is "rect.msh".
    :param msh_version: Version of the GMSH mesh file format. Default is None (use the default version).
    :param layer_ids: List of layer IDs for the physical groups. If None, the IDs will be generated automatically.
    """
    out_name = optional_default_file(out_name, "gmsh_rect", ".msh")

    if not all(
        1e-5 <= length <= 1e9
        for length in (lengths if isinstance(lengths, tuple) else (lengths,))
    ):
        # Numerical restriction for gmsh (discovered by testing)
        msg = f"All lengths must be >= 1e-5 and <= 1e9, got: {lengths}"
        raise ValueError(msg)

    if not all(
        n_cell >= 1
        for n_cell in (
            n_edge_cells if isinstance(n_edge_cells, tuple) else (n_edge_cells,)
        )
    ):
        msg = f"All n_edge_cells must be >= 1: {n_edge_cells}"
        raise ValueError(msg)

    if not n_layers >= 1:
        msg = f"n_layers must be >= 1: {n_layers}"
        raise ValueError(msg)

    if not all(
        length / np.max(n_edge_cells) >= 1e-10
        for length in (lengths if isinstance(lengths, tuple) else (lengths,))
    ):
        print(
            "Warning: The length of the rectangle divided by the number of edge cells is smaller than 1e-10. This may lead to unexpected results."
        )

    gmsh.initialize(["-noenv"])
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
            recombine=recombine,
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
    return Path(out_name)


def _square(
    geo: type[gmsh.model.geo],
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
    out_name: Path | str | None = None,
    msh_version: float | None = None,
) -> Path:
    out_name = optional_default_file(out_name, "gmsh_cuboid", ".msh")

    gmsh.initialize(["-noenv"])
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
            numElements=[nz],
            recombine=recombine,
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
    return out_name


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
    output_file: Path | str | None = None,
    refinement: dict | None = None,
    local_ref: list[dict] | dict | None = None,
    mesh_opts: dict | None = None,
) -> Path:
    """Discretizes a given Mesh with triangles and saves as gmsh .msh.

    Requires the mesh to be 2D and to contain 'MaterialIDs in the cell data.

    :param mesh:        The mesh which shall be discretized with triangles
    :param output_file: The full filepath to the resulting file
    :param refinement: specification for refinement of region edges

        You can provide a dict with the following data:
        - 'SizeMin': Minimum element size (default: median point distance)
        - 'SizeMax': Maximum element size (default: 3 * median point distance)
        - 'DistMin': Distance until which SizeMin is used (default: 0)
        - 'DistMax': Distance after which SizeMax is used (default: 3 * median point distance)
    :param local_ref: specification/s of local refinement

        Allows the same refinement options as in `refinement`, but the dict
        has to contain an entry named 'pts' which provides the points around
        which the refinement is performed.
    :param mesh_opts: Meshing options. Will be passed to
        `gmsh.option.setNumber(f"Mesh.{key}", value)`. Additionally pass
        'order' to set the element order (1=linear, 2=quadratic, ...).
    """
    output_file = optional_default_file(output_file, "gmsh_remesh", ".msh")
    gmsh.initialize(["-noenv"])
    gmsh.option.set_number("General.Verbosity", 0)
    gmsh.clear()
    gmsh.model.add("domain")

    mat_ids = np.unique(mesh["MaterialIDs"])
    # gmsh requires the domain ids to start at 1
    id_offset = 1 if 0 in mat_ids else 0
    region_edge_points = [
        _ordered_edges(mesh.threshold([m, m], scalars="MaterialIDs"))
        for m in (mat_ids)
    ]
    point_stack = np.vstack(region_edge_points)
    for point in point_stack:
        gmsh.model.geo.addPoint(point[0], point[1], point[2])

    region_lengths = [len(pts) for pts in region_edge_points]
    region_start_tag = np.cumsum([0] + region_lengths) + 1
    for tag_0, tag_1 in pairwise(region_start_tag):
        for index in range(tag_0, tag_1 - 1):
            gmsh.model.geo.addLine(index, index + 1)
        gmsh.model.geo.addLine(tag_1 - 1, tag_0)

    field = gmsh.model.mesh.field
    ref = refinement if isinstance(refinement, dict) else {}
    mdists = []
    for index, points in enumerate(region_edge_points):
        mat_id = mat_ids[index] + id_offset
        line_tags = range(region_start_tag[index], region_start_tag[index + 1])
        gmsh.model.geo.addCurveLoop(line_tags, tag=mat_id)
        gmsh.model.geo.addPlaneSurface([mat_id], tag=mat_id)
        gmsh.model.addPhysicalGroup(
            dim=2, tags=[mat_id], name=f"Layer {mat_id}"
        )
        f_1, f_2 = (2 * mat_id, 2 * mat_id + 1)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        mdist = np.median(distances)
        mdists.append(mdist)
        field.add("Distance", f_1)
        field.setNumbers(f_1, "CurvesList", line_tags)
        field.add("Threshold", f_2)
        field.setNumber(f_2, "IField", f_1)
        field.setNumber(f_2, "SizeMin", ref.get("SizeMin", mdist))
        field.setNumber(f_2, "SizeMax", ref.get("SizeMax", mdist * 3))
        field.setNumber(f_2, "DistMin", ref.get("DistMin", 0))
        field.setNumber(f_2, "DistMax", ref.get("DistMax", mdist * 3))

    f_min = f_2 + 1
    field.add("Min", tag=f_min)
    field.setNumbers(f_min, "FieldsList", (mat_ids * 2 + 1).tolist())

    if local_ref is None:
        field.setAsBackgroundMesh(f_min)
    else:
        from scipy.spatial import KDTree

        ref_pts_ = local_ref if isinstance(local_ref, list) else [local_ref]
        for subdomain_id, ref in enumerate(ref_pts_):
            if one_pt := np.asarray(ref["pts"]).ndim == 1:
                mdist = np.median(mdists)
            else:
                distances = np.linalg.norm(np.diff(ref["pts"], axis=0), axis=1)
                mdist = np.median(distances)

            tree = KDTree(point_stack)
            # as the boundaries in the gmsh geometry of different regions may
            # overlap, we have to find more then one closest neighbour.
            # Otherwise the size field might not have an effect, due the later
            # call of removeAllDuplicates.
            pt_dists, pt_ids = tree.query(ref["pts"], k=2)
            if np.isscalar(pt_ids):
                pt_ids_ = [pt_ids]
            elif np.asarray(pt_ids).ndim == 1:
                matches = np.isclose(pt_dists, pt_dists[0])
                pt_ids_ = np.asarray(pt_ids)[matches]
            else:
                matches = np.isclose(pt_dists.T, pt_dists[:, 0]).T
                pt_ids_ = np.asarray(pt_ids)[matches]

            f_1 = f_min + subdomain_id * 2 + 1
            f_2 = f_1 + 1
            field.add("Distance", f_1)
            list_type = "PointsList" if one_pt else "CurvesList"
            field.setNumbers(f_1, list_type, pt_ids_)
            field.add("Threshold", f_2)
            field.setNumber(f_2, "IField", f_1)
            field.setNumber(f_2, "SizeMin", ref.get("SizeMin", mdist))
            field.setNumber(f_2, "SizeMax", ref.get("SizeMax", mdist * 3))
            field.setNumber(f_2, "DistMin", ref.get("DistMin", 0))
            field.setNumber(f_2, "DistMax", ref.get("DistMax", mdist * 3))
        f4 = f_2 + 1
        field.add("Min", tag=f4)
        field.setNumbers(f4, "FieldsList", list(range(f_min, f4, 2)))
        field.setAsBackgroundMesh(f4)

    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()
    opts = {} if mesh_opts is None else mesh_opts
    opts.setdefault("SecondOrderIncomplete", 1)
    opts.setdefault("MeshSizeExtendFromBoundary", 0)
    opts.setdefault("MeshSizeFromPoints", 0)
    opts.setdefault("MeshSizeFromCurvature", 0)
    opts.setdefault("Smoothing", 1)
    opts.setdefault("Algorithm", 0)
    for setting, value in opts.items():
        gmsh.option.setNumber(f"Mesh.{setting}", value)
    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.setOrder(opts.get("order", 1))
    gmsh.write(str(output_file))
    gmsh.finalize()
    return output_file
