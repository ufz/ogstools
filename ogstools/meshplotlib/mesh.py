"""A class to handle Meshseries data."""

import pyvista as pv


class Mesh(pv.UnstructuredGrid):
    """
    A wrapper around pyvista.

    Will be replaced by own module with more functionality.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(cls, filepath: str):
        """Read the mesh from a filepath."""
        return cls(pv.get_reader(filepath).read())
