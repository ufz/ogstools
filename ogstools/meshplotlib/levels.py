"""Utilities to create nicely spaced levels."""

import numpy as np


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
    nice_range = nice_num(upper - lower)
    tick_spacing = nice_num(nice_range / (n_ticks - 1.0))
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
    log = np.log10(np.abs(vals), out=np.zeros_like(vals), where=(vals != 0.0))
    exponents = np.floor(log).astype(int)
    median_exp = int(np.median(exponents))
    return np.stack([np.round(v, precision - median_exp) for v in vals])


def get_levels(lower: float, upper: float, n_ticks: int) -> np.ndarray:
    """
    Return an array in the interval [lower, upper] with terminating decimals.

    The length of the arrays will be close to n_ticks.
    At the boundaries the tickspacing may differ from the remaining array.
    """
    if np.abs(upper - lower) <= 1e-6:
        return lower + np.array([0.0, 1e-6])
    levels = nice_range(lower, upper, n_ticks)
    return np.append(np.append(lower, adaptive_rounding(levels, 6)), upper)
