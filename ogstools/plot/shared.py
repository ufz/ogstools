# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from .plot_setup import PlotSetup
from .plot_setup_defaults import setup_dict

setup = PlotSetup.from_dict(setup_dict)
