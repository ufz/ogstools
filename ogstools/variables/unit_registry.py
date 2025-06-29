# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pint import UnitRegistry

u_reg: UnitRegistry = UnitRegistry(
    preprocessors=[lambda s: s.replace("%", "percent")]
)
u_reg.formatter.default_format = "~.12g"
u_reg.setup_matplotlib(True)
