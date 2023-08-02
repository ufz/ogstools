import geopandas as gp
import pandamesh as pm
import meshio

from argparse import ArgumentParser, RawTextHelpFormatter


parser = ArgumentParser(
    description="This tool allows meshing of shapefiles.",
    formatter_class=RawTextHelpFormatter,
)

parser.add_argument("-i", "--input", help="The path to the input shape-file.")
parser.add_argument(
    "-o",
    "--output",
    help="The path to the output file.\n"
    "The extension defines the format according to meshio",
)
parser.add_argument(
    "-c", "--cellsize", help="The cellsize for the mesh.", type=float
)
parser.add_argument(
    "meshing",
    choices=["Triangle", "GMSH"],
    default="Triangle",
    type=str,
    help="Either Triangle or GMSH can be choosen for meshing.",
    nargs="?",
    const=1,
)

parser.add_argument(
    "simplify",
    choices=["simplified", "original"],
    default="original",
    type=str,
    help="Either the shapefiles are kept unchanged or they can be simplified.",
    nargs="?",
    const=1,
)
args = parser.parse_args()

gdf = gp.read_file(args.input)

if "MultiPolygon" in gdf["geometry"].geom_type.values:
    # break down multipolygon to multiple polygons
    exploded_data = gdf.explode()
    # from geoseries to geodataframe
    gdf = gp.GeoDataFrame(
        geometry=[polygon for polygon in exploded_data["geometry"]]
    )
# get rid off intersections
union = gdf.unary_union
# union to geodataframe
union_gdf = gp.GeoDataFrame(geometry=[union])
# breakdown multipolygon again
exploded_union = union_gdf.explode()
# create final geodataframe
final_gdf = gp.GeoDataFrame(
    geometry=[polygon for polygon in exploded_union["geometry"]]
)
# define cellsize
final_gdf["cellsize"] = args.cellsize
if args.simplify == "simplified":
    # simplified = final_gdf.copy()
    final_gdf.geometry = final_gdf.geometry.buffer(args.cellsize / 20)
    final_gdf["dissolve_column"] = 0
    final_gdf = final_gdf.dissolve(by="dissolve_column")
    final_gdf.geometry = final_gdf.geometry.simplify(args.cellsize / 2)
    exploded_union = final_gdf.explode()
    # create final geodataframe
    final_gdf = gp.GeoDataFrame(
        geometry=[polygon for polygon in exploded_union["geometry"]]
    )
    final_gdf["cellsize"] = args.cellsize

# choose the meshing algorithm: also GmshMesher() possible, but requires installation of gmsh or triangle
if args.meshing == "Triangle":
    mesher = pm.TriangleMesher(final_gdf)
elif args.meshing == "GMSH":
    mesher = pm.GmshMesher(final_gdf)
vertices, faces = mesher.generate()
# write mesh to file, also vtk-format possible
mesh = meshio.Mesh(vertices, [("triangle", faces)])
mesh.write(args.output)
