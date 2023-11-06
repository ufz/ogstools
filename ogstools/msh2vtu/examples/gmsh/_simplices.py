from pathlib import Path

import gmsh


def _geo_square(geo: gmsh.model.geo, element_size: float, structured: bool):
    geo.addPoint(0, 0, 0, meshSize=element_size, tag=1)
    geo.addPoint(1, 0, 0, meshSize=element_size, tag=2)
    geo.addPoint(1, 1, 0, meshSize=element_size, tag=3)
    geo.addPoint(0, 1, 0, meshSize=element_size, tag=4)

    geo.addLine(1, 2, tag=1)
    geo.addLine(2, 3, tag=2)
    geo.addLine(3, 4, tag=3)
    geo.addLine(4, 1, tag=4)

    geo.addCurveLoop([1, 2, 3, 4], tag=1)
    geo.addPlaneSurface([1], tag=1)

    if structured:
        n_points = 1 + round(1.0 / element_size)
        geo.mesh.setTransfiniteCurve(1, n_points)
        geo.mesh.setTransfiniteCurve(2, n_points)
        geo.mesh.setTransfiniteCurve(3, n_points)
        geo.mesh.setTransfiniteCurve(4, n_points)
        geo.mesh.setTransfiniteSurface(1)
        geo.mesh.setRecombine(dim=2, tag=1)


def unit_square(
    element_size: float = 0.5,
    structured_grid: bool = True,
    order: int = 1,
    out_name: Path = Path("unit_square.msh"),
):
    gmsh.initialize()
    gmsh.model.add("unit_square")

    _geo_square(gmsh.model.geo, element_size, structured_grid)

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


def unit_cube(
    element_size: float = 1.0,
    structured_grid: bool = True,
    order: int = 1,
    out_name: Path = Path("unit_cube.msh"),
):
    gmsh.initialize()
    gmsh.model.add("unit_cube")
    _geo_square(gmsh.model.geo, element_size, structured_grid)

    num_el = [round(1.0 / element_size)] if structured_grid else []
    rec = structured_grid

    newEntities = gmsh.model.geo.extrude(
        dimTags=[(2, 1)], dx=0, dy=0, dz=1, numElements=num_el, recombine=rec
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
