def dim_from_len(len: int):
    """
    Determine the dimension based on the length.

    :param len: The length of the array.

    :returns: The corresponding dimension.
    """
    dim_map = {2: 2, 3: 3, 4: 2, 6: 3}
    if len in dim_map:
        return dim_map[len]
    return len


def identity(vals):
    """
    Return the input values without modification.

    :param values: The input values.

    :returns: input values.
    """
    return vals
