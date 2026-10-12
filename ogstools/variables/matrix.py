# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal

from ogstools.mesh.utils import angles, azimuth
from ogstools.variables import tensor_math
from ogstools.variables.variable import Scalar, Variable
from ogstools.variables.vector import Vector, VectorList

from .func import Function


class Matrix(Variable):
    """Represent a matrix variable.

    Matrix variables should contain either 4 (2D) or 6 (3D) components.
    Matrix components can be accesses with brackets e.g. stress[0]
    """

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
        if not isinstance(index, int) and index not in key_map:
            allowed = list(key_map.keys())
            msg = f"Matrix index can only be an int or one of {allowed}."
            raise KeyError(msg)
        int_index = key_map.get(str(index), index)

        return Scalar.from_variable(
            self,
            func=lambda x: x[..., int_index],
            suffix=index,
            symbol=f"{{{self.symbol}}}_{{{index}}}",
            bilinear_cmap=True,
        )

    def to_polar(
        self, center: Sequence = (0, 0, 0), normal: Sequence = (0, 0, 1)
    ) -> Matrix:
        """Return the Matrix converted to a polar coordinate system.

        For 3D only spherical coordinate system is implemented for now.
        """
        return Matrix.from_variable(
            self,
            func=Function(
                tensor_math.to_polar,
                [partial(angles, center=center, normal=normal), azimuth],
            ),
            prefix="polar",
        )

    @property
    def magnitude(self) -> Scalar:
        "A scalar variable as the frobenius norm of the matrix."
        return Scalar.from_variable(
            self,
            func=tensor_math.frobenius_norm,
            suffix="magnitude",
            symbol=rf"||{{{self.symbol}}}||_\mathrm{{F}}",
        )

    @property
    def trace(self) -> Scalar:
        "A scalar variable as the trace of the matrix."
        return Scalar.from_variable(
            self,
            func=tensor_math.trace,
            suffix="trace",
            symbol=rf"\mathrm{{tr}}({{{self.symbol}}})",
        )

    @property
    def eigenvalues(self) -> Vector:
        "A vector variable as the eigenvalues of the matrix."
        return Vector.from_variable(
            self,
            func=tensor_math.eigenvalues,
            suffix="eigenvalues",
            symbol=r"\lambda",
        )

    @property
    def eigenvectors(self) -> VectorList:
        "A vector variable as the eigenvectors of the matrix."
        return VectorList.from_variable(
            self,
            func=tensor_math.eigenvectors,
            suffix="eigenvectors",
            symbol="v",
            data_unit="",
            output_unit="",
        )

    @property
    def det(self) -> Scalar:
        "A scalar variable as the determinant of the matrix."
        return Scalar.from_variable(
            self,
            func=tensor_math.det,
            prefix="det",
            output_unit=self.output_unit + "^2",
            symbol=rf"\mathrm{{det}} {{{self.symbol}}}",
            process_with_units=True,
        )

    @property
    def invariant_1(self) -> Scalar:
        "A scalar variable as the first invariant of the matrix."
        return Scalar.from_variable(
            self, func=tensor_math.invariant_1, suffix="I1"
        )

    @property
    def invariant_2(self) -> Scalar:
        "A scalar variable as the second invariant of the matrix."
        return Scalar.from_variable(
            self,
            func=tensor_math.invariant_2,
            suffix="I2",
            output_unit=self.output_unit + "^2",
            process_with_units=True,
        )

    @property
    def invariant_3(self) -> Scalar:
        "A scalar variable as the third invariant of the matrix."
        return Scalar.from_variable(
            self, func=tensor_math.invariant_3, suffix="I3"
        )

    @property
    def tensor_mean(self) -> Scalar:
        "A scalar variable as the mean value of the matrix."
        return Scalar.from_variable(
            self, func=tensor_math.mean, prefix="mean", symbol=r"\pi"
        )

    @property
    def hydrostatic_component(self) -> Matrix:
        "A vector variable as the effective pressure of the matrix."
        return Matrix.from_variable(
            self,
            func=tensor_math.hydrostatic_component,
            prefix="hydrostatic",
            suffix="component",
            symbol=rf"p^{{{self.symbol}}}",
        )

    @property
    def deviator(self) -> Matrix:
        "A vector variable as the deviator of the matrix."
        return Matrix.from_variable(
            self,
            func=tensor_math.deviator,
            suffix="deviator",
            symbol=rf"s^{{{self.symbol}}}",
        )

    @property
    def deviator_invariant_1(self) -> Scalar:
        "A scalar variable as the first invariant of the matrix deviator."
        return Scalar.from_variable(
            self, func=tensor_math.deviator_invariant_1, suffix="J1"
        )

    @property
    def deviator_invariant_2(self) -> Scalar:
        "A scalar variable as the second invariant of the matrix deviator."
        return Scalar.from_variable(
            self, func=tensor_math.deviator_invariant_2, suffix="J2"
        )

    @property
    def deviator_invariant_3(self) -> Scalar:
        "A scalar variable as the third invariant of the matrix deviator."
        return Scalar.from_variable(
            self, func=tensor_math.deviator_invariant_3, suffix="J3"
        )

    @property
    def octahedral_shear(self) -> Scalar:
        "A scalar variable as the octahedral shear component of the matrix."
        return Scalar.from_variable(
            self,
            func=tensor_math.octahedral_shear,
            prefix="octahedral_shear",
            symbol=r"\tau_\mathrm{oct}",
        )

    @property
    def von_Mises(self) -> Scalar:
        "A scalar variable as the von Mises stress."
        return Scalar.from_variable(
            self,
            func=tensor_math.von_mises,
            prefix="von_Mises",
            symbol=rf"{{{self.symbol}}}_\mathrm{{v}}",
        )

    @property
    def qp_ratio(self) -> Scalar:
        "A scalar variable as the qp stress ratio."
        return Scalar.from_variable(
            self,
            func=tensor_math.qp_ratio,
            prefix="qp_ratio (",
            suffix=")",
            output_unit="%",
            symbol="qp",
            process_with_units=True,
        )
