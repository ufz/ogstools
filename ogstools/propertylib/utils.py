from typing import TypeVar

import numpy as np


def dim_from_len(len: int):
    ":returns: The dimension corresponding to the length. (2|4 -> 2, 3|6 -> 3)"
    dim_map = {2: 2, 3: 3, 4: 2, 6: 3}
    if len in dim_map:
        return dim_map[len]
    raise ValueError("Can't determine dimension for length " + str(len))


def sym_tensor_to_mat(vals: np.ndarray) -> np.ndarray:
    "Convert an symmetric tensor to a 3x3 matrix."
    assert np.shape(vals)[-1] in [4, 6]
    shape = list(np.shape(vals))[:-1] + [3, 3]
    mat = np.zeros(shape)
    idx = {0: [0, 0], 1: [1, 1], 2: [2, 2], 3: [0, 1], 4: [1, 2], 5: [0, 2]}
    for i in range(np.shape(vals)[-1]):
        mat[..., idx[i][0], idx[i][1]] = vals[..., i]
    return mat


T = TypeVar("T")


def identity(vals: T) -> T:
    ":returns: The input values."
    return vals
