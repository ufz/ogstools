# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pint import UnitRegistry

u_reg: UnitRegistry = UnitRegistry(
    preprocessors=[lambda s: s.replace("%", "percent")]
)
u_reg.formatter.default_format = "~.12g"
u_reg.setup_matplotlib(True)
