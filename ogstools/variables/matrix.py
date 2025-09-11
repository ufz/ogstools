# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pyvista import UnstructuredGrid
from typeguard import typechecked

from ogstools.variables import tensor_math
from ogstools.variables.mesh_dependent import angles
from ogstools.variables.variable import Scalar, Variable
from ogstools.variables.vector import Vector, VectorList


class Matrix(Variable):
    """Represent a matrix variable.

    Matrix variables should contain either 4 (2D) or 6 (3D) components.
    Matrix components can be accesses with brackets e.g. stress[0]
    """

    @typechecked
    def __getitem__(
        self,
        index: (
            int
            | Literal["xx", "yy", "zz", "xy", "yz", "xz"]
            | Literal["rr", "tt", "pp", "rt", "tp", "rp"]
        ),
    ) -> Scalar:
        """A scalar variable as a matrix component.

        The following index values correspond to a polar coordinate system:

        rr: radial component
        tt: angular component in theta (azimuthal) direction
        pp: angular component in phi (polar) direction
        rt: shear component in the radial-azimuthal plane
        tp: shear component in the azimuthal-polar plane
        rp: shear component in the radial-polar plane
        """
        cartesian_keys = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5}
        polar_keys = {"rr": 0, "tt": 1, "pp": 2, "rt": 3, "tp": 4, "rp": 5}
        key_map = cartesian_keys | polar_keys
        int_index = key_map.get(str(index), index)

        return Scalar.from_variable(
            self,
            output_name=self.output_name + f"_{index}",
            symbol=f"{{{self.symbol}}}_{{{index}}}",
            func=lambda x: self.func(x)[..., int_index],
            bilinear_cmap=True,
        )

    def to_polar(
        self, center: Sequence = (0, 0, 0), normal: Sequence = (0, 0, 1)
    ) -> Matrix:
        """Return the Matrix converted to a polar coordinate system.

        For 3D only spherical coordinate system is implemented for now.
        """

        def theta(mesh: UnstructuredGrid) -> np.ndarray | None:
            "Calculate the azimuth angle with regards to the z-axis"
            if np.shape(mesh[self.data_name])[-1] == 4:  # 2D
                return None
            pts, z = (mesh.points, mesh.points[:, 2])
            r = np.hypot(*pts[:, [0, 1]].T)
            return np.arctan(
                np.divide(r, z, out=np.ones_like(z) * 1e12, where=z != 0.0)
            )

        return self.replace(
            mesh_dependent=True,
            func=lambda mesh: (
                tensor_math.to_polar(
                    self.func(self._get_data(mesh)),
                    angles(mesh, center, normal),
                    theta(mesh),
                )
            ),
        )

    @property
    def magnitude(self) -> Scalar:
        "A scalar variable as the frobenius norm of the matrix."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_magnitude",
            symbol=rf"||{{{self.symbol}}}||_\mathrm{{F}}",
            func=lambda x: tensor_math.frobenius_norm(self.func(x)),
        )

    @property
    def trace(self) -> Scalar:
        "A scalar variable as the trace of the matrix."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_trace",
            symbol=rf"\mathrm{{tr}}({{{self.symbol}}})",
            func=tensor_math.trace,
        )

    @property
    def eigenvalues(self) -> Vector:
        "A vector variable as the eigenvalues of the matrix."
        return Vector.from_variable(
            self,
            output_name=self.output_name + "_eigenvalues",
            symbol=r"\lambda",
            func=lambda x: tensor_math.eigenvalues(self.func(x)),
        )

    @property
    def eigenvectors(self) -> VectorList:
        "A vector variable as the eigenvectors of the matrix."
        return VectorList.from_variable(
            self,
            output_name=self.output_name + "_eigenvectors",
            symbol="v",
            data_unit="",
            output_unit="",
            func=lambda x: tensor_math.eigenvectors(self.func(x)),
        )

    @property
    def det(self) -> Scalar:
        "A scalar variable as the determinant of the matrix."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_det",
            symbol=rf"\mathrm{{det}} {{{self.symbol}}}",
            func=lambda x: tensor_math.det(self.func(x)),
        )

    @property
    def invariant_1(self) -> Scalar:
        "A scalar variable as the first invariant of the matrix."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_I1",
            func=lambda x: tensor_math.invariant_1(self.func(x)),
        )

    @property
    def invariant_2(self) -> Scalar:
        "A scalar variable as the second invariant of the matrix."
        return Scalar.from_variable(
            self,
            output_unit=self.output_unit + "^2",
            output_name=self.output_name + "_I2",
            func=lambda x: tensor_math.invariant_2(self.func(x)),
            process_with_units=True,
        )

    @property
    def invariant_3(self) -> Scalar:
        "A scalar variable as the third invariant of the matrix."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_I3",
            func=lambda x: tensor_math.invariant_3(self.func(x)),
        )

    @property
    def mean(self) -> Scalar:
        "A scalar variable as the mean value of the matrix."
        return Scalar.from_variable(
            self,
            output_name="mean_" + self.output_name,
            symbol=r"\pi",
            func=lambda x: tensor_math.mean(self.func(x)),
        )

    @property
    def hydrostatic_component(self) -> Matrix:
        "A vector variable as the effective pressure of the matrix."
        return Matrix.from_variable(
            self,
            output_name="hydrostatic_" + self.output_name + "_component",
            symbol=rf"p^{{{self.symbol}}}",
            func=lambda x: tensor_math.hydrostatic_component(self.func(x)),
        )

    @property
    def deviator(self) -> Matrix:
        "A vector variable as the deviator of the matrix."
        return Matrix.from_variable(
            self,
            output_name=self.output_name + "_deviator",
            symbol=rf"s^{{{self.symbol}}}",
            func=lambda x: tensor_math.deviator(self.func(x)),
        )

    @property
    def deviator_invariant_1(self) -> Scalar:
        "A scalar variable as the first invariant of the matrix deviator."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_J1",
            func=lambda x: tensor_math.deviator_invariant_1(self.func(x)),
        )

    @property
    def deviator_invariant_2(self) -> Scalar:
        "A scalar variable as the second invariant of the matrix deviator."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_J2",
            func=lambda x: tensor_math.deviator_invariant_2(self.func(x)),
        )

    @property
    def deviator_invariant_3(self) -> Scalar:
        "A scalar variable as the third invariant of the matrix deviator."
        return Scalar.from_variable(
            self,
            output_name=self.output_name + "_J3",
            func=lambda x: tensor_math.deviator_invariant_3(self.func(x)),
        )

    @property
    def octahedral_shear(self) -> Scalar:
        "A scalar variable as the octahedral shear component of the matrix."
        return Scalar.from_variable(
            self,
            output_name="octahedral_shear_" + self.output_name,
            symbol=r"\tau_\mathrm{oct}",
            func=lambda x: tensor_math.octahedral_shear(self.func(x)),
        )

    @property
    def von_Mises(self) -> Scalar:
        "A scalar variable as the von Mises stress."
        return Scalar.from_variable(
            self,
            output_name="von_Mises_" + self.output_name,
            symbol=rf"{{{self.symbol}}}_\mathrm{{v}}",
            func=lambda x: tensor_math.von_mises(self.func(x)),
        )

    @property
    def qp_ratio(self) -> Scalar:
        "A scalar variable as the qp stress ratio."
        return Scalar.from_variable(
            self,
            output_name="qp_ratio",
            output_unit="percent",
            symbol="qp",
            func=lambda x: tensor_math.qp_ratio(self.func(x)),
            process_with_units=True,
        )
