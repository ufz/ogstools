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

from typing import TypeVar, Union

import numpy as np
from numpy import linalg as LA
from pint.facets.plain import PlainQuantity

from .unit_registry import u_reg

ValType = Union[PlainQuantity, np.ndarray]


T = TypeVar("T")


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


def trace(vals: ValType) -> ValType:
    """Return the trace.

    :math:`tr(\\mathbf{\\sigma}) = \\sum\\limits_{i=1}^3 \\sigma_{ii}`
    """
    return np.sum(vals[..., :3], axis=-1)


def eigenvalues(vals: ValType) -> ValType:
    "Return the eigenvalues."
    if isinstance(vals, PlainQuantity):
        unit = vals.units
        vals = vals.magnitude
    else:
        unit = None
    eigvals: np.ndarray = LA.eigvals(sym_tensor_to_mat(vals))
    eigvals.sort(axis=-1)
    assert np.all(eigvals[..., 0] <= eigvals[..., 1])
    assert np.all(eigvals[..., 1] <= eigvals[..., 2])
    return eigvals if unit is None else u_reg.Quantity(eigvals, unit)


def eigenvectors(vals: ValType) -> ValType:
    "Return the eigenvectors."
    if isinstance(vals, PlainQuantity):
        vals = vals.magnitude
    eigvals, eigvecs = LA.eig(sym_tensor_to_mat(vals))
    ids = eigvals.argsort(axis=-1)
    eigvals = np.take_along_axis(eigvals, ids, axis=-1)
    eigvecs = np.take_along_axis(eigvecs, ids[:, np.newaxis], axis=-1)
    assert np.all(eigvals[..., 0] <= eigvals[..., 1])
    assert np.all(eigvals[..., 1] <= eigvals[..., 2])
    return eigvecs


def det(vals: ValType) -> ValType:
    "Return the determinants."
    if isinstance(vals, PlainQuantity):
        unit = vals.units
        vals = vals.magnitude
    else:
        unit = None
    result = np.linalg.det(sym_tensor_to_mat(vals))
    return result if unit is None else u_reg.Quantity(result, unit)


def frobenius_norm(val: ValType) -> ValType:
    """Return the Frobenius norm.

    :math:`||\\mathbf{\\sigma}||_F = \\sqrt{ \\sum\\limits_{i=1}^m \\sum\\limits_{j=1}^n |\\sigma_{ij}|^2 }`
    """
    return np.linalg.norm(sym_tensor_to_mat(val), axis=(-2, -1))


def I1(vals: ValType) -> ValType:  # pylint: disable=invalid-name
    """Return the first invariant.

    :math:`I1 = tr(\\mathbf{\\sigma})`
    """
    return trace(vals)


def I2(vals: ValType) -> ValType:  # pylint: disable=invalid-name
    """Return the second invariant.

    :math:`I2 = \\frac{1}{2} \\left[(tr(\\mathbf{\\sigma}))^2 - tr(\\mathbf{\\sigma}^2) \\right]`
    """
    return 0.5 * (trace(vals) ** 2 - trace(vals**2))


def I3(vals: ValType) -> ValType:  # pylint: disable=invalid-name
    """Return the third invariant.

    :math:`I3 = det(\\mathbf{\\sigma})`
    """
    return det(vals)


def mean(vals: ValType) -> ValType:
    """Return the mean value.
    Also called hydrostatic component or octahedral normal component.

    :math:`\\pi = \\frac{1}{3} I1`
    """
    return (1.0 / 3.0) * I1(vals)


def effective_pressure(vals: ValType) -> ValType:
    """Return the effective pressure.

    :math:`\\pi = -\\frac{1}{3} I1`
    """
    return -mean(vals)


def hydrostatic_component(vals: ValType) -> ValType:
    """Return the hydrostatic component.

    :math:`p_{ij} = \\pi \\delta_{ij}`
    """
    if isinstance(vals, PlainQuantity):
        unit = vals.units
        vals = vals.magnitude
    else:
        unit = None
    result = vals * 0.0
    result[..., :3] = mean(vals)[..., np.newaxis]
    return result if unit is None else u_reg.Quantity(result, unit)


def deviator(vals: ValType) -> ValType:
    """Return the deviator.

    :math:`s_{ij} = \\sigma_{ij} - \\pi \\delta_{ij}`
    """
    return vals - hydrostatic_component(vals)


def J1(vals: ValType) -> ValType:  # pylint: disable=invalid-name
    """Return the first invariant of the deviator.

    :math:`J1 = 0`
    """
    return trace(deviator(vals))


def J2(vals: ValType) -> ValType:  # pylint: disable=invalid-name
    """Return the second invariant of the deviator.

    :math:`J2 = \\frac{1}{2} tr(\\mathbf{s}^2)`
    """
    return 0.5 * trace(deviator(vals) ** 2)


def J3(vals: ValType) -> ValType:  # pylint: disable=invalid-name
    """Return the third invariant of the deviator.

    :math:`J3 = \\frac{1}{3} tr(\\mathbf{s}^3)`
    """
    return (1.0 / 3.0) * trace(deviator(vals) ** 3)


def octahedral_shear(vals: ValType) -> ValType:
    """Return the octahedral shear value.

    :math:`\\tau_{oct} = \\sqrt{\\frac{2}{3} J2}`
    """
    return np.sqrt((2.0 / 3.0) * J2(vals))


def von_mises(vals: ValType) -> ValType:
    """Return the von Mises stress.

    :math:`\\sigma_{Mises} = \\sqrt{3 J2}`
    """
    return np.sqrt(3.0 * J2(vals))


def qp_ratio(vals: ValType) -> ValType:
    """Return the qp ratio (von Mises stress / effective pressure).

    :math:`qp = \\sigma_{Mises} / \\pi`
    """
    return von_mises(vals) / effective_pressure(vals)
