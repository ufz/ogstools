# mesh unit cube with tetraeders
import gmsh

# init
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("cube")

dim1 = 1
dim2 = 2
dim3 = 3
lc = 150.0  # mesh size

# opposite vertices
x0 = 0.0
y0 = 0.0
z0 = 0.0
x1 = 1000.0
y1 = 1000.0
z1 = 1000.0


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
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.addCurveLoop([11, 2, 3, -10], 2)
gmsh.model.geo.addPlaneSurface([2], 2)

gmsh.model.geo.addCurveLoop([2, -9, -7, 1], 3)
gmsh.model.geo.addPlaneSurface([3], -3)

gmsh.model.geo.addCurveLoop([6, 7, 8, 5], 4)
gmsh.model.geo.addPlaneSurface([4], -4)

gmsh.model.geo.addCurveLoop([8, -4, -3, -9], 5)
gmsh.model.geo.addPlaneSurface([5], 5)

gmsh.model.geo.addCurveLoop([10, 4, 5, -12], 6)
gmsh.model.geo.addPlaneSurface([6], 6)


# volume
gmsh.model.geo.addSurfaceLoop([6, 2, 1, 4, 3, 5], 1)
gmsh.model.geo.addVolume([1], 1)


# physical groups
D = gmsh.model.addPhysicalGroup(dim2, [4])
gmsh.model.setPhysicalName(dim2, D, "west")

C = gmsh.model.addPhysicalGroup(dim2, [3])
gmsh.model.setPhysicalName(dim2, C, "top")

B = gmsh.model.addPhysicalGroup(dim2, [2])
gmsh.model.setPhysicalName(dim2, B, "east")

F = gmsh.model.addPhysicalGroup(dim2, [6])
gmsh.model.setPhysicalName(dim2, F, "bottom")

A = gmsh.model.addPhysicalGroup(dim2, [1])
gmsh.model.setPhysicalName(dim2, A, "north")

E = gmsh.model.addPhysicalGroup(dim2, [5])
gmsh.model.setPhysicalName(dim2, E, "south")

W = gmsh.model.addPhysicalGroup(dim3, [1])
gmsh.model.setPhysicalName(dim3, W, "volume")


# mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(dim3)
# higher order, for simplex (tetra) no difference between Lagrange and
# Serendipity elements
gmsh.model.mesh.setOrder(2)

gmsh.write("cube_tet.msh")
gmsh.finalize()
