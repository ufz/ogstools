# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import tempfile
from collections.abc import ItemsView, KeysView, ValuesView
from pathlib import Path

import pyvista as pv

from .gmsh_converter import meshes_from_gmsh
from .mesh import Mesh
from .meshes_from_yaml import meshes_from_yaml
from .subdomains import (
    get_dim,
    identify_subdomains,
    named_boundaries,
    split_by_threshold_angle,
    split_by_vertical_lateral_edges,
)


class Meshes:
    """
    OGS input mesh. Refers to prj - file section <meshes>
    """

    def __init__(self, meshes: dict[str, pv.UnstructuredGrid]) -> None:
        """
        Initialize a Meshes object.
            :param meshes:      List of Mesh objects (pyvista UnstructuredGrid)
                                The first mesh is the domain mesh.
                                All following meshes represent subdomains, and their points must align with points on the domain mesh.
                                If needed, the domain mesh itself can also be added again as a subdomain.
            :returns:           A Meshes object
        """
        self._meshes: dict[str, Mesh] = {
            name: Mesh(mesh) for name, mesh in meshes.items()
        }
        self.has_identified_subdomains: bool = False
        self._accessed: bool = False
        self._tmp_dir = Path(tempfile.mkdtemp("meshes"))

    def __getitem__(self, key: str) -> Mesh:
        if key not in self._meshes:
            msg = f"Key {key!r} not found"
            raise KeyError(msg)
        return self._meshes[key]

    def __setitem__(self, key: str, mesh: pv.UnstructuredGrid) -> None:
        self.has_identified_subdomains = False
        self._meshes[key] = Mesh(mesh)

    def __len__(self) -> int:
        return len(self._meshes)

    @classmethod
    def from_yaml(cls, geometry_file: Path) -> "Meshes":
        """ """

        gmsh_file = meshes_from_yaml(geometry_file)
        print(f"Info: Mesh written to {gmsh_file}")

        meshes_dict = meshes_from_gmsh(gmsh_file)
        meshes_obj = cls(meshes_dict)
        meshes_obj.has_identified_subdomains = True
        return meshes_obj

    @classmethod
    def from_gmsh(
        cls,
        filename: Path,
        dim: int | list[int] = 0,
        reindex: bool = True,
        log: bool = True,
        meshname: str = "domain",
    ) -> "Meshes":
        """
        Generates pyvista unstructured grids from a gmsh mesh (.msh).

        Extracts domain-, boundary- and physical group-submeshes.

        :param filename:    Gmsh mesh file (.msh) as input data
        :param dim: Spatial dimension (1, 2 or 3), trying automatic detection,
                    if not given. If multiple dimensions are provided, all elements
                    of these dimensions are embedded in the resulting domain mesh.
        :param reindex: Physical groups / regions / Material IDs to be
                        renumbered consecutively beginning with zero.
        :param log:     If False, silence log messages
        :param meshname:    The name of the domain mesh and used as a prefix for subdomain meshes.

        :returns: A dictionary of names and corresponding meshes
        """
        meshes_dict = meshes_from_gmsh(filename, dim, reindex, log, meshname)
        meshes_obj = cls(meshes_dict)
        meshes_obj.has_identified_subdomains = True
        return meshes_obj

    @classmethod
    def from_mesh(
        cls,
        mesh: pv.UnstructuredGrid,
        threshold_angle: float | None = 15.0,
        domain_name: str = "domain",
    ) -> "Meshes":
        """Extract 1D boundaries of a 2D mesh.

        :param mesh:            The 2D domain
        :param threshold_angle: If None, the boundary will be split by the
                                assumption of vertical lateral boundaries. Otherwise
                                it represents the angle (in degrees) between
                                neighbouring elements which - if exceeded -
                                determines the corners of the boundary mesh.
        :param domain_name:     The name of the domain mesh.
        :returns:               A Meshes object.
        """

        dim = get_dim(mesh)
        assert dim == 2, f"Expected a mesh of dim 2, but given mesh has {dim=}"
        boundary = mesh.extract_feature_edges()
        if threshold_angle is None:
            subdomains = split_by_vertical_lateral_edges(boundary)
        else:
            subdomains = split_by_threshold_angle(boundary, threshold_angle)

        sub_meshes_dict = named_boundaries(subdomains)

        meshes_dict = {domain_name: mesh} | sub_meshes_dict
        meshes_obj = cls(meshes_dict)
        meshes_obj.has_identified_subdomains = False
        return meshes_obj

    def keys(self) -> KeysView[str]:
        """
        Get the names of all meshes.

        :returns: All mesh names
        """
        return self._meshes.keys()

    def values(self) -> ValuesView[Mesh]:
        """
        Get all Mesh objects (pyvista.UnstructuredGrid).

        :returns: All Mesh objects
        """
        return self._meshes.values()

    def items(self) -> ItemsView[str, Mesh]:
        """
        Get all meshnames-Mesh pairs.

        Each item is a tuple of (name, Mesh)

        :returns: All (name, Mesh) pairs
        """
        return self._meshes.items()

    def pop(self, key: str) -> Mesh:
        """
        Remove a mesh by name and return it.

        This removes the mesh from the internal dictionary.

        :param key: The name of the mesh to remove

        :returns: The Mesh object that was removed
        """
        return self._meshes.pop(key)

    def domain(self) -> Mesh:
        """
        Get the domain mesh.

        By convention, the domain mesh is the **first mesh** in the dictionary
        of meshes when the `Meshes` object was constructed.
        The domain mesh is expected to be constant. e.g. Do not: myobject.domain = pv.Sphere()


        :returns: The domain mesh
        """
        return next(iter(self._meshes.values()))

    def domain_name(self) -> str:
        """
        Get the name of the domain mesh.

        By convention, the domain mesh is the **first mesh** in the dictionary
        of meshes when the `Meshes` object was constructed.

        :returns: The name of the domain mesh
        """
        return next(iter(self._meshes.keys()))

    def subdomains(self) -> dict[str, Mesh]:
        """
        Get the subdomain meshes.

        By convention, all meshes **except the first** one are considered subdomains.
        This returns a list of those subdomain meshes.

        :returns: A dictionary of {name: Mesh} for all subdomains
        """
        items = list(self._meshes.items())
        return dict(items[1:])  # by convention: first mesh is domain

    def save(
        self, meshes_path: Path | None = None, overwrite: bool = False
    ) -> list[Path]:
        """
        Save all meshes.

        This function will perform identifySubdomains, if not yet been done.

        :param meshes_path: Optional path to the directory where meshes
                            should be saved. If None, a temporary directory
                            will be used.

        :param overwrite: If True, existing mesh files will be overwritten.
                          If False, an error is raised if any file already exists.

        :returns: A list of Paths pointing to the saved mesh files
        """
        meshes_path = meshes_path or self._tmp_dir
        meshes_path.mkdir(parents=True, exist_ok=True)

        if not self.has_identified_subdomains:
            identify_subdomains(self.domain(), list(self.subdomains().values()))
            self.has_identified_subdomains = True

        output_files = [meshes_path / f"{name}.vtu" for name in self._meshes]
        existing_files = [f for f in output_files if f.exists()]
        if existing_files and not overwrite:
            existing_list = "\n".join(str(f) for f in existing_files)
            msg = f"The following mesh files already exist:{existing_list}. Set `overwrite=True` to overwrite them, or choose a different `meshes_path`."

            raise FileExistsError(msg)

        for name, mesh in self._meshes.items():
            mesh.filepath = meshes_path / f"{name}.vtu"
            pv.save_meshio(mesh.filepath, mesh)

        return [meshes_path / f"{name}.vtu" for name in self._meshes]
