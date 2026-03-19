# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


from .plot_setup import PlotSetup
from .plot_setup_defaults import setup_dict

setup = PlotSetup.from_dict(setup_dict)
