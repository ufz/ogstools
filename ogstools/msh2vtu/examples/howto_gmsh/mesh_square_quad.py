# mesh unit square with quadrilaterals (of higher two)
import gmsh

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("square")

# Dimensions
dim1 = 1
dim2 = 2

lc = 0.5  # characteristic length for meshing
# Define some corner points. All points should have different tags:
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

# Lines connecting points
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# The third elementary entity is the surface. In order to define a surface
# from the curves defined above, a curve loop has first to be defined.
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

# Add plane surfaces defined by one or more curve loops.
gmsh.model.geo.addPlaneSurface([1], 1)

#  prepare structured grid
nEhori = 3
nEvert = 3
gmsh.model.geo.mesh.setTransfiniteCurve(1, nEhori)
gmsh.model.geo.mesh.setTransfiniteCurve(2, nEvert)
gmsh.model.geo.mesh.setTransfiniteCurve(3, nEhori)
gmsh.model.geo.mesh.setTransfiniteCurve(4, nEvert)
gmsh.model.geo.mesh.setTransfiniteSurface(1, "Alternate")
gmsh.model.geo.mesh.setRecombine(dim2, 1)


# physical groups (only this gets saved to file per default)
Bottom = gmsh.model.addPhysicalGroup(dim1, [1])
gmsh.model.setPhysicalName(dim1, Bottom, "Bottom")

Right = gmsh.model.addPhysicalGroup(dim1, [2])
gmsh.model.setPhysicalName(dim1, Right, "Right")

Top = gmsh.model.addPhysicalGroup(dim1, [3])
gmsh.model.setPhysicalName(dim1, Top, "Top")

Left = gmsh.model.addPhysicalGroup(dim1, [4])
gmsh.model.setPhysicalName(dim1, Left, "Left")

Rectangle = gmsh.model.addPhysicalGroup(dim2, [1])
gmsh.model.setPhysicalName(dim2, Rectangle, "UnitSquare")

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(dim2)
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)  # serendipity elements
gmsh.model.mesh.setOrder(2)  # higher order elements (quadratic)
gmsh.write("square_quad.msh")

gmsh.finalize()
