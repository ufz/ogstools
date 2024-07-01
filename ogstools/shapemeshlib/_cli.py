from argparse import ArgumentParser, RawTextHelpFormatter

import geopandas as gp

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

mesh = geodataframe_meshing()
mesh.write(args.output)
