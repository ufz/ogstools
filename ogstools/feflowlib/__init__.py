# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging as log
from importlib.util import find_spec
from sys import stdout

if find_spec("ifm") is None:
    msg = "Could not import ifm. A working FEFLOW Python setup is required!"
    raise ImportError(msg)

from ._tools import (
    assign_bulk_ids,
    remove_bulk_ids,
)
from .feflow_model import FeflowModel

__all__ = [
    "assign_bulk_ids",
    "FeflowModel",
    "remove_bulk_ids",
]

# log configuration
log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=log.WARN,
    stream=stdout,
    datefmt="%d.%m.%Y %H:%M:%S",
)
