# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Common tensor transformation operations.

They can be used as they are, but are also part of
:py:obj:`ogstools.propertylib.matrix.Matrix` properties.
All input arrays are expected to be in vector notation of symmetric tensors in
the form of:

[xx, yy, zz, xy] for 2D and

[xx, yy, zz, xy, yz, xz] for 3D.

This notation style is the default output of OGS:

<https://www.opengeosys.org/docs/userguide/basics/conventions/#a-namesymmetric-tensorsa--symmetric-tensors-and-kelvin-mapping>

A better overview for the theoretical background of the equations can be found
here, for example:

<https://en.wikipedia.org/wiki/Cauchy_stress_tensor#Cauchy's_stress_theorem%E2%80%94stress_tensor>
"""

from typing import Optional, TypeVar, Union

import numpy as np
from numpy import linalg as LA
from pint.facets.plain import PlainQuantity, PlainUnit

from .unit_registry import u_reg

ValType = Union[PlainQuantity, np.ndarray]


T = TypeVar("T")


def _split_quantity(values: ValType) -> tuple[np.ndarray, Optional[PlainUnit]]:
    if isinstance(values, PlainQuantity):
        return values.magnitude, values.units
    return values, None


def _to_quantity(
    values: np.ndarray, unit: Optional[PlainUnit]
) -> Union[np.ndarray, PlainQuantity]:
    return values if unit is None else u_reg.Quantity(values, unit)


def identity(vals: T) -> T:
    ":returns: The input values."
    return vals


def sym_tensor_to_mat(vals: np.ndarray) -> np.ndarray:
    "Convert an symmetric tensor to a 3x3 matrix."
    assert np.shape(vals)[-1] in [4, 6]
    shape = list(np.shape(vals))[:-1] + [3, 3]
    mat = np.zeros(shape)
    idx = np.asarray([[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [0, 2]])
    for i in range(np.shape(vals)[-1]):
        mat[..., idx[i, 0], idx[i, 1]] = vals[..., i]
    return mat


def trace(values: ValType) -> ValType:
    """Return the trace.

    :math:`tr(\\mathbf{\\sigma}) = \\sum\\limits_{i=1}^3 \\sigma_{ii}`
    """
    vals, unit = _split_quantity(values)
    return _to_quantity(np.sum(vals[..., :3], axis=-1), unit)


def eigenvalues(values: ValType) -> ValType:
    "Return the eigenvalues."
    vals, unit = _split_quantity(values)
    eigvals: np.ndarray = LA.eigvals(sym_tensor_to_mat(vals))
    eigvals.sort(axis=-1)
    assert np.all(eigvals[..., 0] <= eigvals[..., 1])
    assert np.all(eigvals[..., 1] <= eigvals[..., 2])
    return _to_quantity(eigvals, unit)


def eigenvectors(values: ValType) -> ValType:
    "Return the eigenvectors."
    vals, unit = _split_quantity(values)
    eigvals, eigvecs = LA.eig(sym_tensor_to_mat(vals))
    ids = eigvals.argsort(axis=-1)
    eigvals = np.take_along_axis(eigvals, ids, axis=-1)
    eigvecs = np.take_along_axis(eigvecs, ids[:, np.newaxis], axis=-1)
    assert np.all(eigvals[..., 0] <= eigvals[..., 1])
    assert np.all(eigvals[..., 1] <= eigvals[..., 2])
    return _to_quantity(eigvecs, unit)


def det(values: ValType) -> ValType:
    "Return the determinants."
    vals, unit = _split_quantity(values)
    return _to_quantity(np.linalg.det(sym_tensor_to_mat(vals)), unit)


def frobenius_norm(values: ValType) -> ValType:
    """Return the Frobenius norm.

    :math:`||\\mathbf{\\sigma}||_F = \\sqrt{ \\sum\\limits_{i=1}^m \\sum\\limits_{j=1}^n |\\sigma_{ij}|^2 }`
    """
    vals, unit = _split_quantity(values)
    return _to_quantity(
        np.linalg.norm(sym_tensor_to_mat(vals), axis=(-2, -1)), unit
    )


def invariant_1(values: ValType) -> ValType:
    """Return the first invariant.

    :math:`I1 = tr(\\mathbf{\\sigma})`
    """
    return trace(values)


def invariant_2(values: ValType) -> ValType:
    """Return the second invariant.

    :math:`I2 = \\frac{1}{2} \\left[(tr(\\mathbf{\\sigma}))^2 - tr(\\mathbf{\\sigma}^2) \\right]`
    """
    return 0.5 * (trace(values) ** 2 - trace(values**2))


def invariant_3(values: ValType) -> ValType:
    """Return the third invariant.

    :math:`I3 = det(\\mathbf{\\sigma})`
    """
    return det(values)


def mean(values: ValType) -> ValType:
    """Return the mean value.
    Also called hydrostatic component or octahedral normal component.

    :math:`\\pi = \\frac{1}{3} I1`
    """
    return (1.0 / 3.0) * invariant_1(values)


def effective_pressure(values: ValType) -> ValType:
    """Return the effective pressure.

    :math:`\\pi = -\\frac{1}{3} I1`
    """
    return -mean(values)


def hydrostatic_component(values: ValType) -> ValType:
    """Return the hydrostatic component.

    :math:`p_{ij} = \\pi \\delta_{ij}`
    """
    vals, unit = _split_quantity(values)
    result = vals * 0.0
    result[..., :3] = _split_quantity(mean(vals))[0][..., np.newaxis]
    return _to_quantity(result, unit)


def deviator(values: ValType) -> ValType:
    """Return the deviator.

    :math:`s_{ij} = \\sigma_{ij} - \\pi \\delta_{ij}`
    """
    return values - hydrostatic_component(values)


def deviator_invariant_1(values: ValType) -> ValType:
    """Return the first invariant of the deviator.

    :math:`J1 = 0`
    """
    return trace(deviator(values))


def deviator_invariant_2(values: ValType) -> ValType:
    """Return the second invariant of the deviator.

    :math:`J2 = \\frac{1}{2} tr(\\mathbf{s}^2)`
    """
    return 0.5 * trace(deviator(values) ** 2)


def deviator_invariant_3(values: ValType) -> ValType:
    """Return the third invariant of the deviator.

    :math:`J3 = \\frac{1}{3} tr(\\mathbf{s}^3)`
    """
    return (1.0 / 3.0) * trace(deviator(values) ** 3)


def octahedral_shear(values: ValType) -> ValType:
    """Return the octahedral shear value.

    :math:`\\tau_{oct} = \\sqrt{\\frac{2}{3} J2}`
    """
    return np.sqrt((2.0 / 3.0) * deviator_invariant_2(values))


def von_mises(values: ValType) -> ValType:
    """Return the von Mises stress.

    :math:`\\sigma_{Mises} = \\sqrt{3 J2}`
    """
    return np.sqrt(3.0 * deviator_invariant_2(values))


def qp_ratio(values: ValType) -> ValType:
    """Return the qp ratio (von Mises stress / effective pressure).

    :math:`qp = \\sigma_{Mises} / \\pi`
    """
    return von_mises(values) / effective_pressure(values)
