# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Joerg Buchwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

from .ogs import OGS
from .build_tree import BuildTree
from .curves import Curves
from .geo import Geo
from .linsolvers import LinSolvers
from .local_coordinate_system import LocalCoordinateSystem
from .media import Media
from .mesh import Mesh
from .nonlinsolvers import NonLinSolvers
from .parameters import Parameters
from .processes import Processes
from .processvars import ProcessVars
from . import properties
from .python_script import PythonScript
from .timeloop import TimeLoop

__all__ = [
    "OGS"
]
