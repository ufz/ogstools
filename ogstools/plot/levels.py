# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Utilities to create nicely spaced levels."""

from math import nextafter

import numpy as np

from ogstools.plot.shared import setup
from ogstools.propertylib import Property
from ogstools.propertylib.properties import get_preset


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
    return np.unique(
        adaptive_rounding(
            np.append(np.append(lower, result), upper), precision=3
        )
    )


def median_exponent(vals: np.ndarray) -> int:
    "Get the median exponent from an array of numbers."
    if np.issubdtype(vals.dtype, np.integer):
        return 0
    log = np.log10(np.abs(vals), out=np.zeros_like(vals), where=(vals != 0.0))
    exponents = np.floor(log).astype(int)
    return int(np.median(exponents))


def combined_levels(
    meshes: np.ndarray, mesh_property: Property | str
) -> np.ndarray:
    """
    Calculate well spaced levels for the encompassing property range in meshes.
    """
    mesh_property = get_preset(mesh_property, meshes.ravel()[0])
    p_min, p_max = np.inf, -np.inf
    unique_vals = np.array([])
    for mesh in np.ravel(meshes):
        values = mesh_property.magnitude.transform(mesh)
        if setup.log_scaled:  # TODO: can be improved
            values = np.log10(np.where(values > 1e-14, values, 1e-14))
        p_min = min(p_min, np.nanmin(values)) if setup.p_min is None else p_min
        p_max = max(p_max, np.nanmax(values)) if setup.p_max is None else p_max
        unique_vals = np.unique(
            np.concatenate((unique_vals, np.unique(values)))
        )
    p_min = setup.p_min if setup.p_min is not None else p_min
    p_max = setup.p_max if setup.p_max is not None else p_max
    if p_min == p_max:
        return np.array([p_min, p_max + 1e-12])
    if (
        all(val.is_integer() for val in unique_vals)
        and setup.p_min is None
        and setup.p_max is None
    ):
        return unique_vals[(p_min <= unique_vals) & (unique_vals <= p_max)]
    return compute_levels(p_min, p_max, setup.num_levels)


def level_boundaries(levels: np.ndarray) -> np.ndarray:
    return np.array(
        [
            levels[0] - 0.5 * (levels[1] - levels[0]),
            *0.5 * (levels[:-1] + levels[1:]),
            levels[-1] + 0.5 * (levels[-1] - levels[-2]),
        ]
    )
