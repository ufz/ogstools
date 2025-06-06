#!/usr/bin/env python3

# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


"""
Created on Thu Sep 23 13:54:03 2021

@author: Dominik Kern
"""
import pyvista as pv
from PVGeo.grids import SurferGridReader

# READ IN
filename = "relief.grd"
dem = SurferGridReader().apply(filename)  # GRD reader
x0, x1, y0, y1, z0, z1 = dem.GetBounds()

# RESAMPLE
# import numpy as np
# x = np.linspace(x0, x1, num=10)
# y = np.linspace(y0, y1, num=10)
# xx, yy, zz = np.meshgrid(x, y, [0])
# grid = pv.StructuredGrid(xx, yy, zz)
#
# dem_resampled = grid.sample(dem)

# SAVE
# relief = dem_resampled.warp_by_scalar(scale_factor=1.)
relief = dem.warp_by_scalar(scale_factor=1.0)
relief.plot(show_edges=True)
pv.save_meshio("relief.stl", relief.triangulate())  # STL writer
