# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import os
import tempfile
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from pathlib import Path

import pyvista as pv

from .gmsh_converter import meshes_from_gmsh
from .mesh import Mesh
from .meshes_from_yaml import meshes_from_yaml
from .subdomains import (
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

    def __iter__(self) -> Iterator[str]:
        yield from self._meshes

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

        dim = mesh.GetMaxSpatialDimension()
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

    @classmethod
    def from_gml(
        cls,
        domain_path: Path,
        gml_path: Path,
        out_dir: Path | None = None,
        tolerance: float = 1e-12,
    ) -> "Meshes":
        """Create Meshes from geometry definition in the gml file.

        :param domain_path: Path to the domain mesh.
        :param gml_file:    Path to the gml file.
        :param out_dir:     Where to write the gml meshes (default: gml dir)
        :param tolerance:   search length for node search algorithm

        :returns:           A Meshes object.
        """
        out_dir = gml_path.parent if out_dir is None else out_dir

        cur_dir = Path.cwd()
        os.chdir(out_dir)

        from ogstools._find_ogs import cli

        prev_files = set(out_dir.glob("*.vtu"))
        cli().constructMeshesFromGeometry(
            "-g", gml_path, "-m", domain_path, "-s", str(tolerance)
        )
        vtu_files = sorted(set(out_dir.glob("*.vtu")).difference(prev_files))
        os.chdir(cur_dir)
        return cls(
            {file.stem: Mesh(file) for file in [domain_path] + vtu_files}
        )

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

    def identify_subdomain(self) -> None:
        identify_subdomains(self.domain(), list(self.subdomains().values()))
        self.has_identified_subdomains = True

    def _partmesh_prepare(self, meshes_path: Path) -> Path:
        from ogstools import cli

        domain_file_name = self.domain_name() + ".vtu"
        domain_file = meshes_path / domain_file_name

        parallel_path = meshes_path / "p"
        cli().partmesh(o=parallel_path, i=domain_file, ogs2metis=True)
        return parallel_path / self.domain_name()

    def _partmesh_single(
        self, num_partitions: int, base_file: Path
    ) -> list[Path]:
        from ogstools import cli

        partition_path = base_file.parent / str(num_partitions)
        partition_path.mkdir(parents=True, exist_ok=True)
        meshes_path = base_file.parent.parent

        subdomain_files = [
            f"{meshes_path}/{subdomain}.vtu" for subdomain in self.subdomains()
        ]

        domain_file_name = self.domain_name() + ".vtu"
        domain_file = meshes_path / domain_file_name

        cli().partmesh(
            o=partition_path,
            i=domain_file,
            m=True,
            n=num_partitions,
            x=base_file,
            *subdomain_files,  # noqa: B026
        )
        return list(partition_path.glob("*"))

    def _partmesh_save_all(
        self, partitions: list[int], meshes_path: Path
    ) -> dict[int, list[Path]]:
        """
        Creates a folder with decomposed / partitioned mesh suitable for parallel OGS execution

        :param num_partitions:  The number of partitions (internally passed to OGS bin tool partmesh)

        :param meshes_path:     The folder where the original serial mesh is ALREADY stored.
                                E.g. `save` must be called before.

        """

        base_file = self._partmesh_prepare(meshes_path)
        parallel_files: dict[int, list[Path]] = {
            partition: self._partmesh_single(partition, base_file)
            for partition in partitions
        }

        return parallel_files

    def save(
        self,
        meshes_path: Path | None = None,
        overwrite: bool = False,
        partitions: list | None = None,
    ) -> dict[int, list[Path]]:
        """
        Save all meshes.

        This function will perform identifySubdomains, if not yet been done.

        :param meshes_path: Optional path to the directory where meshes
                            should be saved. If None, a temporary directory
                            will be used.

        :param overwrite:   If True, existing mesh files will be overwritten.
                            If False, an error is raised if any file already exists.

        :param partitions:  List of integers > 1 that indicate the number of partitions
                            similar to the OGS binary tool partmesh.
                            The serial mesh will always be generated
                            Example: partitions = [2,4,8,16]

        :returns: A list of Paths pointing to the saved mesh files
        """
        meshes_path = meshes_path or self._tmp_dir
        meshes_path.mkdir(parents=True, exist_ok=True)

        partitions = partitions or []

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

        serial_files = {
            1: [meshes_path / f"{name}.vtu" for name in self._meshes]
        }

        parallel_files = self._partmesh_save_all(partitions, meshes_path)

        return serial_files | parallel_files
