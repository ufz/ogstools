from argparse import ArgumentParser, RawTextHelpFormatter

from ogstools.shapemeshlib import (
    create_pyvista_mesh,
    geodataframe_meshing,
    prepare_shp_for_meshing,
)

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
    "-c",
    "--cellsize",
    help="The cellsize for the mesh.",
    type=float,
    default=None,
)
parser.add_argument(
    "meshing",
    choices=["Triangle", "GMSH"],
    default="Triangle",
    type=str,
    help="Either Triangle or GMSH can be chosen for meshing.",
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


def cli() -> None:
    args = parser.parse_args()
    simple = "simplified" in args.simplify
    triangle = "Triangle" in args.meshing
    geodataframe = prepare_shp_for_meshing(args.input)
    points_cells = geodataframe_meshing(
        geodataframe, simple, triangle, args.cellsize
    )
    pyvista_mesh = create_pyvista_mesh(
        points=points_cells[0], cells=points_cells[1]
    )
    pyvista_mesh.save(args.output)
