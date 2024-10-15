# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from dataclasses import dataclass
from typing import Literal

from ogstools.variables import tensor_math
from ogstools.variables.variable import Scalar, Variable
from ogstools.variables.vector import Vector, VectorList


@dataclass
class Matrix(Variable):
    """Represent a matrix variable.

    Matrix variables should contain either 4 (2D) or 6 (3D) components.
    Matrix components can be accesses with brackets e.g. stress[0]
    """

    def __getitem__(
        self, index: int | Literal["xx", "yy", "zz", "xy", "yz", "xz"]
    ) -> Scalar:
        "A scalar variable as a matrix component."
        int_index = (
            index
            if isinstance(index, int)
            else ["xx", "yy", "zz", "xy", "yz", "xz"].index(index)
        )
        return Scalar.from_variable(
            self,
            output_name=self.output_name + f"_{index}",
            symbol=f"{{{self.symbol}}}_{{{index}}}",
            func=lambda x: self.func(x)[..., int_index],
            bilinear_cmap=True,
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
    def hydrostatic_component(self) -> "Matrix":
        "A vector variable as the effective pressure of the matrix."
        return Matrix.from_variable(
            self,
            output_name="hydrostatic_" + self.output_name + "_component",
            symbol=rf"p^{{{self.symbol}}}",
            func=lambda x: tensor_math.hydrostatic_component(self.func(x)),
        )

    @property
    def deviator(self) -> "Matrix":
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
