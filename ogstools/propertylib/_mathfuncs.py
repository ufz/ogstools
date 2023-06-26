"""Common mathematical transformation functions."""

import numpy as np

from .utils import dim_from_len


def component(vals: np.ndarray, id: int) -> np.ndarray:
    """
    Extract a specific component from the input array.

    :param values: The input array.
    :param index: The index of the component to extract.

    :returns: The extracted component.
    """
    if len(np.shape(vals)) > 0:
        if id >= np.shape(vals)[-1]:
            raise ValueError(
                (
                    f"Requested index {id} is out of bounds for ",
                    f"array with shape {vals.shape}",
                )
            )
        return vals[..., id]
    return vals


def magnitude(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the magnitude of each vector in the input array.

    :param values: The input array of vectors.

    :returns: The magnitudes of the vectors.
    """
    return np.sqrt(np.sum(np.square(vals), axis=-1))


def log_magnitude(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the logarithm of the magnitude of each vector in the input array.

    :param values: The input array of vectors.

    :returns: The logarithms of the magnitudes of the vectors.
    """
    return np.log10(np.sqrt(np.sum(np.square(vals), axis=-1)))


def trace(vals: np.ndarray) -> np.ndarray:
    """
    Calculate the trace of each vector in the input array.

    :param values: The input array of vectors.

    :returns: The trace values of the vectors.
    """
    return np.sum(vals[..., : dim_from_len(vals.shape[-1])], axis=-1)
