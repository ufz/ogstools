from pathlib import Path
from typing import Union

import gmsh


def _geo_square(
    geo: gmsh.model.geo,
    lengths: Union[float, list[float]],
    n_edge_cells: Union[int, list[int]],
    structured: bool,
):
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
):
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
    gmsh.model.add("unit_square")

    _geo_square(gmsh.model.geo, lengths, n_edge_cells, structured_grid)

    bottom = gmsh.model.addPhysicalGroup(dim=1, tags=[1])
    right = gmsh.model.addPhysicalGroup(dim=1, tags=[2])
    top = gmsh.model.addPhysicalGroup(dim=1, tags=[3])
    left = gmsh.model.addPhysicalGroup(dim=1, tags=[4])
    rectangle = gmsh.model.addPhysicalGroup(dim=2, tags=[1])

    gmsh.model.setPhysicalName(dim=1, tag=bottom, name="bottom")
    gmsh.model.setPhysicalName(dim=1, tag=right, name="right")
    gmsh.model.setPhysicalName(dim=1, tag=top, name="top")
    gmsh.model.setPhysicalName(dim=1, tag=left, name="left")
    gmsh.model.setPhysicalName(dim=2, tag=rectangle, name="unit_square")

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
):
    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 0)
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

    vol = gmsh.model.addPhysicalGroup(dim=3, tags=[vol_tag])
    gmsh.model.setPhysicalName(dim=3, tag=vol, name="volume")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=3)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(out_name))
    gmsh.finalize()
