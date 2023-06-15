# mesh two connected unit cubes with different elements:
# tetra, pyramid, wedge (vtu name "wedge" in gmsh named "prism")
# tetra10, pyramid14 (pyramid13 fails, meshio bug?), wedge18 (wedge 15)
#
# force pyramids by a quad mesh at right side and tri meshes at the remaining sides
#
# force wedges (gmsh: prisms) by extrusions (with Recombine=True) of tri mesh

import gmsh

# init
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("cubes_mixed")

dim1 = 1
dim2 = 2
dim3 = 3
lc = 0.5  # mesh size

# opposite vertices define cuboid
x0 = 0.0
y0 = 0.0
z0 = 0.0
x1 = 1.0
y1 = 1.0
z1 = 1.0


# vertices
gmsh.model.geo.addPoint(x1, y0, z0, lc, 1)
gmsh.model.geo.addPoint(x1, y1, z0, lc, 2)
gmsh.model.geo.addPoint(x0, y1, z0, lc, 3)
gmsh.model.geo.addPoint(x0, y0, z1, lc, 4)
gmsh.model.geo.addPoint(x1, y0, z1, lc, 5)
gmsh.model.geo.addPoint(x1, y1, z1, lc, 6)
gmsh.model.geo.addPoint(x0, y1, z1, lc, 7)
gmsh.model.geo.addPoint(x0, y0, z0, lc, 8)


# edges
gmsh.model.geo.addLine(7, 6, 1)
gmsh.model.geo.addLine(6, 5, 2)
gmsh.model.geo.addLine(5, 1, 3)
gmsh.model.geo.addLine(1, 8, 4)
gmsh.model.geo.addLine(8, 3, 5)
gmsh.model.geo.addLine(3, 7, 6)
gmsh.model.geo.addLine(7, 4, 7)
gmsh.model.geo.addLine(4, 8, 8)
gmsh.model.geo.addLine(4, 5, 9)
gmsh.model.geo.addLine(2, 1, 10)
gmsh.model.geo.addLine(2, 6, 11)
gmsh.model.geo.addLine(2, 3, 12)


# faces
gmsh.model.geo.addCurveLoop([6, 1, -11, 12], 1)
gmsh.model.geo.addPlaneSurface([1], 1)  # right (normal ey)
nx = 3  # number of mesh divisions in x-direction
ny = 3  # .. and y
gmsh.model.geo.mesh.setTransfiniteCurve(6, ny)
gmsh.model.geo.mesh.setTransfiniteCurve(1, nx)
gmsh.model.geo.mesh.setTransfiniteCurve(11, ny)
gmsh.model.geo.mesh.setTransfiniteCurve(12, nx)
gmsh.model.geo.mesh.setTransfiniteSurface(1)
gmsh.model.geo.mesh.setRecombine(dim2, 1)

gmsh.model.geo.addCurveLoop([11, 2, 3, -10], 2)
gmsh.model.geo.addPlaneSurface([2], 2)  # front (normal ex)

gmsh.model.geo.addCurveLoop([2, -9, -7, 1], 3)
gmsh.model.geo.addPlaneSurface([3], -3)  # top (normal ez)

gmsh.model.geo.addCurveLoop([6, 7, 8, 5], 4)
gmsh.model.geo.addPlaneSurface([4], -4)  # back (normal -ex)

gmsh.model.geo.addCurveLoop([8, -4, -3, -9], 5)
gmsh.model.geo.addPlaneSurface([5], 5)  # left (normal -ey)

gmsh.model.geo.addCurveLoop([10, 4, 5, -12], 6)
gmsh.model.geo.addPlaneSurface([6], 6)  # bottom (normal -ez)


# volume
gmsh.model.geo.addSurfaceLoop([6, 2, 1, 4, 3, 5], 1)
gmsh.model.geo.addVolume([1], 1)  # first unit cube

top_surface_id = (
    3  # in direction [0,0,1] extrude 2 layers of height 0.5, recombination=True
)
newEntities = gmsh.model.geo.extrude(
    [(dim2, 5)], 0, -1, 0, [2], [1], True
)  # second unit cube


# mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(dim3)
# gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1) # serendipity elements (pyramid13 unknown to meshio, bug?)
# gmsh.model.mesh.setOrder(2)   # higher order, for simplex (tri, tet) no difference between Lagrange and Serendipity elements

gmsh.write("cube_mixed.msh")
gmsh.finalize()
