# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging

""".. noindex::"""

logging.basicConfig(
    level=logging.INFO,  # could be DEBUG or WARNING later
    format="%(levelname)s: %(message)s",
)


def set_log_level(level: str = "INFO") -> None:
    """Change the global log level for materiallib."""
    logging.getLogger().setLevel(level.upper())
