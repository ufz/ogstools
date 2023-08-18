import tempfile
from pathlib import Path

import pyvista as pv


class Mesh:
    @property
    def filename(self) -> Path:
        """
        :return path and filename of the mesh
        """
        return self._filename

    @property
    def mesh(self) -> pv.DataSet:
        """
        A base class for representing mesh data and operations.

        This abstract class provides methods to manage and work with mesh data.
        Subclasses are expected to implement specific mesh transformation methods.

        Attributes:
            _filename (Path): The path and filename of the mesh.
            _mesh (pv.DataSet): The mesh data itself.

        Note:
            Do not instantiate an object from this class. Mesh is an abstract class.
            You may use the specific mesh transformation methods provided by subclasses.
        """
        return self._mesh

    def __init__(self, filename, mesh):
        """
        Do not instantiate an object from this class. Mesh is an abstract class. You may use:
        to_region_prism, to_region_tetraeder, to_region_simplified, to_region_voxel instead
        """
        self._filename = filename
        self._mesh = mesh

    def as_pyvista(self) -> pv.UnstructuredGrid:
        """
        Transforms the mesh into a pyvista Unstructured Grid, only if it is not already a pyvista mesh.

        mesh = mesh.as_pyvista()

        :returns pyvista UnstructuredGrid
        """

        if self.mesh is None:
            self._mesh = pv.read(self.filename)
        return self.mesh

    def as_file(self) -> Path:
        """
        Save the mesh to a file or return the existing filename.

        If the mesh is not already saved to a file, it will be saved to a temporary file with
        a .vtu extension.

        Returns:
            Path: The path and filename of the saved mesh file.
        """
        if self._filename:
            return self.filename
        if self.mesh:
            self._filename = Path(tempfile.mkstemp(".vtu", "mesh")[1])
            pv.save_meshio(filename=self.filename, mesh=self.mesh)
            return self.filename
        msg = "Neither mesh nor file given"
        raise ValueError(msg)
