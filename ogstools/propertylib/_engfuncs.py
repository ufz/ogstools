"""Common engineering transformation functions."""

import numpy as np

from ._mathfuncs import trace


def effective_pressure(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the effective pressure based on the input array.

    :param values: The input array.

    :returns: The effective pressure values.
    """
    return -(1.0 / 3.0) * trace(vals)


def von_mises(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the von Mises stress based on the input array.

    :param values: The input array.

    :returns: The von Mises stress values.
    """
    return np.sqrt(
        0.5 * np.sum(np.square(np.diff(vals[..., :3], append=vals[..., 0])), -1)
        + 3 * np.sum(np.square(vals[..., 3:]), -1)
    )


def qp_ratio(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the QP ratio (von Mises stress / effective pressure).

    :param values: The input array.

    :returns: The QP ratios.
    """
    return von_mises(vals) / effective_pressure(vals)
