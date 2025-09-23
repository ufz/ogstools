# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import sys
from pathlib import Path

import click

import ogstools as ot
from ogstools.meshlib.meshes_from_yaml import meshes_from_yaml


@click.command(name="meshes-from-yaml")
@click.option(
    "-i",
    "--input",
    "geometry_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML geometry definition file",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output"),
    show_default=True,
    help="Output directory for mesh and VTU files",
)
def cli(geometry_file: Path, output: Path) -> None:
    """Generate a Gmsh mesh (.msh) from a YAML geometry file and save VTUs."""
    try:
        msh_file = meshes_from_yaml(geometry_file, output)
        click.echo(f"Mesh written to {msh_file}")

        meshes = ot.Meshes.from_gmsh(msh_file, log=False)
        meshes.save(output)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
