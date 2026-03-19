# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pint import UnitRegistry

ureg: UnitRegistry = UnitRegistry()
Q_ = ureg.Quantity
