# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


def _conductivity_tensor(mesh: "pv.UnstructuredGrid") -> np.ndarray:
    """Compute the planar hydraulic conductivity in global coordinates.

    FEFLOW defines P_COND as the maximum principal conductivity, P_ANIS
    as K_min / K_max, and P_ANGL in degrees from x to the maximum direction.
    See the flow material properties in the public FEFLOW manual:
    https://download.feflow.com/html/help110/feflow/09_Parameters/Material_Properties/flow_parameters.html

    :param mesh: Mesh with cell arrays P_COND, P_ANIS and P_ANGL. Conductivity
        must already use the output units; this function does not convert units.
    :returns: One row-major, four-component conductivity tensor per cell.
    """
    maximum = np.asarray(mesh.cell_data["P_COND"])
    minimum = maximum * mesh.cell_data["P_ANIS"]
    angle = np.deg2rad(mesh.cell_data["P_ANGL"])
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = (maximum - minimum) * cosine * sine
    return np.column_stack(
        (
            maximum * cosine**2 + minimum * sine**2,
            cross,
            cross,
            maximum * sine**2 + minimum * cosine**2,
        )
    )
