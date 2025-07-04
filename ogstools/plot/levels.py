# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Utilities to create nicely spaced levels."""

from math import nextafter
from typing import Any

import numpy as np
from pyvista import UnstructuredGrid

from ogstools.plot.shared import setup
from ogstools.variables import Variable


def nice_num(val: float) -> float:
    """
    Return the closest number of the form 10**x * {1,2,4,5}.

    Fractions containing only these number are ensured to have
    terminating decimal representations.
    """
    pow10 = 10 ** np.floor(np.log10(val))
    vals = np.array([1.0, 2.0, 4.0, 5.0, 10.0])
    return pow10 * vals[np.argmin(np.abs(val / pow10 - vals))]


def nice_range(lower: float, upper: float, n_ticks: float) -> np.ndarray:
    """
    Return an array in the interval (lower, upper) with terminating decimals.

    The length of the arrays will be close to n_ticks.
    """
    base = nice_num(upper - lower)
    tick_spacing = nice_num(base / (n_ticks - 1))
    nice_lower = np.ceil(lower / tick_spacing) * tick_spacing
    nice_upper = np.ceil(upper / tick_spacing) * tick_spacing
    res = np.arange(nice_lower, nice_upper, tick_spacing)
    return res[(res > lower) & (res < upper)]


def adaptive_rounding(vals: np.ndarray, precision: int) -> np.ndarray:
    """
    Return the given values rounded to significant digits.

    The significant digits are based of the median decimal exponent and the
    given precision.
    """
    if vals.size == 0:
        return vals
    median_exp = median_exponent(vals)
    rounded_vals = np.stack([np.round(v, precision - median_exp) for v in vals])
    if len(set(rounded_vals)) > 1:
        return rounded_vals
    return np.stack([np.round(v, 12 - median_exp) for v in vals])


def compute_levels(lower: float, upper: float, n_ticks: int) -> np.ndarray:
    """
    Return an array in the interval [lower, upper] with terminating decimals.

    The length of the arrays will be close to n_ticks.
    At the boundaries the tickspacing may differ from the remaining array.
    """
    if lower == upper:
        return np.asarray([lower, nextafter(lower, np.inf)])
    result = nice_range(lower, upper, n_ticks)
    levels = np.unique(
        adaptive_rounding(
            np.append(np.append(lower, result), upper), precision=3
        )
    )
    if levels[0] == levels[-1]:
        return np.array([levels[0], nextafter(levels[0], np.inf)])
    return levels


def median_exponent(vals: np.ndarray) -> int:
    "Get the median exponent from an array of numbers."
    if np.issubdtype(vals.dtype, np.integer):
        return 0
    log = np.log10(np.abs(vals), out=np.zeros_like(vals), where=(vals != 0.0))
    exponents = np.floor(log).astype(int)
    return int(np.median(exponents))


def combined_levels(
    meshes: np.ndarray, variable: Variable | str, **kwargs: Any
) -> np.ndarray:
    """
    Calculate well spaced levels for the encompassing variable range in meshes.
    """
    variable = Variable.find(variable, meshes.ravel()[0])
    vmin, vmax = np.inf, -np.inf
    VMIN = kwargs.get("vmin", setup.vmin)
    VMAX = kwargs.get("vmax", setup.vmax)
    unique_vals = np.array([])
    mesh: UnstructuredGrid
    for mesh in np.ravel(meshes):
        values = variable.magnitude.transform(
            mesh.ctp(True).threshold(value=[1, 1], scalars=variable.mask)
            if variable.mask_used(mesh)
            else mesh
        )
        if (
            kwargs.get("log_scaled", setup.log_scaled)
            and not variable.is_mask()
        ):
            values = np.log10(np.where(values > 1e-14, values, 1e-14))
        vmin = min(vmin, np.nanmin(values)) if VMIN is None else vmin
        vmax = max(vmax, np.nanmax(values)) if VMAX is None else vmax
        unique_vals = np.unique(
            np.concatenate((unique_vals, np.unique(values)))
        )
    vmin = vmin if VMIN is None else VMIN
    vmax = vmax if VMAX is None else VMAX
    if vmin == vmax:
        return np.array([vmin, nextafter(vmax, np.inf)])
    if (
        all(val.is_integer() for val in unique_vals)
        and VMIN is None
        and VMAX is None
        and len(unique_vals) <= setup.num_levels
    ):
        return unique_vals[(vmin <= unique_vals) & (unique_vals <= vmax)]
    return compute_levels(
        vmin, vmax, kwargs.get("num_levels", setup.num_levels)
    )


def level_boundaries(levels: np.ndarray) -> np.ndarray:
    return np.array(
        [
            levels[0] - 0.5 * (levels[1] - levels[0]),
            *0.5 * (levels[:-1] + levels[1:]),
            levels[-1] + 0.5 * (levels[-1] - levels[-2]),
        ]
    )
