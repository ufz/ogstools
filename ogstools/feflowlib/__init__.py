# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import logging as log
from importlib.util import find_spec
from sys import stdout
from typing import TYPE_CHECKING

from ._tools import assign_bulk_ids, remove_bulk_ids

if TYPE_CHECKING:
    from ._feflowlib import read_feflow
    from .feflow_model import FeflowModel

__all__ = ["FeflowModel", "assign_bulk_ids", "read_feflow", "remove_bulk_ids"]


def __getattr__(name: str) -> object:
    """Load the FEFLOW interface only when a reader is requested.

    :param name: Exported reader or model class.
    :returns: The requested export.
    """
    if name not in {"FeflowModel", "read_feflow"}:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    if find_spec("ifm") is None:
        msg = "Could not import ifm. A working FEFLOW Python setup is required!"
        raise ImportError(msg)
    if name == "read_feflow":
        from ._feflowlib import read_feflow

        return read_feflow
    from .feflow_model import FeflowModel

    return FeflowModel


if find_spec("ifm") is not None:
    log.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
        level=log.WARN,
        stream=stdout,
        datefmt="%d.%m.%Y %H:%M:%S",
    )
