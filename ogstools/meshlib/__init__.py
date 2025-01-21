# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from .boundary import Boundary, Layer, LocationFrame, Raster
from .boundary_set import LayerSet
from .boundary_subset import Gaussian2D, Surface
from .data_processing import difference, difference_matrix, difference_pairwise
from .gmsh_converter import meshes_from_gmsh
from .gmsh_meshing import cuboid, rect
from .ip_mesh import to_ip_mesh, to_ip_point_cloud
from .mesh import Mesh
from .mesh_series import MeshSeries
from .region import (
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)
from .shape_meshing import read_shape

__all__ = [
    "Boundary",
    "Gaussian2D",
    "Layer",
    "LayerSet",
    "LocationFrame",
    "Mesh",
    "MeshSeries",
    "Raster",
    "Surface",
    "cuboid",
    "difference",
    "difference_matrix",
    "difference_pairwise",
    "meshes_from_gmsh",
    "read_shape",
    "rect",
    "to_ip_mesh",
    "to_ip_point_cloud",
    "to_region_prism",
    "to_region_simplified",
    "to_region_tetraeder",
    "to_region_voxel",
]
