# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from .region import (
    RegionSet,
    to_boundary,
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)

__all__ = [
    "to_region_prism",
    "to_region_simplified",
    "to_region_tetraeder",
    "to_region_voxel",
    "RegionSet",
    "to_boundary",
]
