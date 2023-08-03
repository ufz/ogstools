"""Common engineering transformation functions."""

import numpy as np

from .utils import dim_from_len


def trace(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the trace of each vector in the input array.

    :param vals: The input array of vectors.

    :returns: The trace values of the vectors.
    """
    return np.sum(vals[..., : dim_from_len(vals.shape[-1])], axis=-1)


def effective_pressure(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the effective pressure based on the input array.

    :param vals: The input array.

    :returns: The effective pressure values.
    """
    return -(1.0 / 3.0) * trace(vals)


def von_mises(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the von Mises stress based on the input array.

    :param vals: The input array.

    :returns: The von Mises stress values.
    """
    return np.sqrt(
        0.5
        * np.sum(np.square(np.diff(vals[..., :3], append=vals[..., :1])), -1)
        + 3 * np.sum(np.square(vals[..., 3:]), -1)
    )


def qp_ratio(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the QP ratio (von Mises stress / effective pressure).

    :param vals: The input array.

    :returns: The QP ratios.
    """
    return von_mises(vals) / effective_pressure(vals)
