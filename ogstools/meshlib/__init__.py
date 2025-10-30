# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from ._utils import reindex_material_ids
from .boundary import Boundary, Layer, LocationFrame, Raster
from .boundary_set import LayerSet
from .boundary_subset import Gaussian2D, Surface
from .cosim import mesh_from_simulator
from .data_processing import difference, difference_matrix, difference_pairwise
from .geo import depth, p_fluid
from .gmsh_converter import meshes_from_gmsh
from .gmsh_meshing import cuboid, rect
from .ip_mesh import to_ip_mesh, to_ip_point_cloud
from .mesh_series import MeshSeries
from .meshes import Meshes
from .region import (
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)

__all__ = [
    "Boundary",
    "Gaussian2D",
    "Layer",
    "LayerSet",
    "LocationFrame",
    "MeshSeries",
    "Meshes",
    "Raster",
    "Surface",
    "cuboid",
    "depth",
    "difference",
    "difference_matrix",
    "difference_pairwise",
    "mesh_from_simulator",
    "meshes_from_gmsh",  # deprecated
    "p_fluid",
    "rect",
    "reindex_material_ids",
    "to_ip_mesh",
    "to_ip_point_cloud",
    "to_region_prism",
    "to_region_simplified",
    "to_region_tetraeder",
    "to_region_voxel",
]
