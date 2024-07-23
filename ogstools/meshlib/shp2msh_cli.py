from argparse import ArgumentParser, RawTextHelpFormatter

import ogstools.meshlib as ml

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
    choices=["triangle", "gmsh"],
    default="triangle",
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

    ml.read_shape(
        args.input,
        simplify=simple,
        mesh_generator=args.meshing,
        cellsize=args.cellsize,
    ).save(args.output)
