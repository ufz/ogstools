# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import pyvista as pv

from ogstools.definitions import ROOT_DIR
from ogstools.meshlib import MeshSeries

# Will probably be replaced with some dynamically generated example
examples = ROOT_DIR / "_examples"
meshseries_THM_2D = MeshSeries(str(examples / "2D.pvd"), time_unit="s")
meshseries_CT_2D = MeshSeries(str(examples / "elder.xdmf"), time_unit="s")
meshseries_XDMF = MeshSeries(
    str(examples / "2D_single_fracture_HT_2D_single_fracture.xdmf"),
    time_unit="s",
)
mesh_mechanics = pv.XMLUnstructuredGridReader(
    str(examples / "mechanics_example.vtu")
).read()
