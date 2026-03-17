# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import sys
from pathlib import Path

import click


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
        from ogstools import Meshes

        meshes = Meshes.from_yaml(geometry_file)
        meshes.save(output)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
