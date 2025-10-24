# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import os
import tempfile
from collections.abc import ItemsView, Iterator, KeysView, Sequence, ValuesView
from pathlib import Path

import pyvista as pv

from ogstools._internal import deprecated

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
        dim: int | Sequence[int] = 0,
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

    @property
    def domain(self) -> Mesh:
        """
        Get the domain mesh.

        By convention, the domain mesh is the **first mesh** in the dictionary
        of meshes when the `Meshes` object was constructed.
        The domain mesh is expected to be constant. e.g. Do not: myobject.domain = pv.Sphere()


        :returns: The domain mesh
        """
        return next(iter(self._meshes.values()))

    @property
    def domain_name(self) -> str:
        """
        Get the name of the domain mesh.

        By convention, the domain mesh is the **first mesh** in the dictionary
        of meshes when the `Meshes` object was constructed.

        :returns: The name of the domain mesh
        """
        return next(iter(self._meshes.keys()))

    @domain_name.setter
    def domain_name(self, name: str) -> None:
        list_meshes = list(self._meshes.items())
        _, domain_mesh = list_meshes[0]
        self._meshes = {name: domain_mesh} | dict(list_meshes[1:])

    @property
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
        identify_subdomains(self.domain(), list(self.subdomains.values()))
        self.has_identified_subdomains = True

    def rename_subdomains(self, rename_map: dict[str, str]) -> None:
        """
        Rename subdomain meshes according to the provided mapping.

        :param rename_map:  A dictionary mapping old subdomain names -> new names.
                            e.g. {'left':'subdomain_left'}
        """

        items = list(self._meshes.items())
        domain_name, domain_mesh = items[0]
        subdomains = dict(items[1:])

        invalid = [name for name in rename_map if name not in subdomains]
        if invalid:
            msg = f"Invalid subdomain names: {invalid}. Valid names: {list(subdomains)}"
            raise KeyError(msg)

        new_subdomains = {
            rename_map.get(name, name): mesh
            for name, mesh in subdomains.items()
        }

        self._meshes = {domain_name: domain_mesh} | new_subdomains

    @deprecated(
        """
    Please rename the groups in the original meshes - containing physical_group OR (better)
    Use the shorter names (without "physical_group") -> renaming in prj-files and scripts necessary.
    """
    )
    def rename_subdomains_legacy(self) -> None:
        """
        Add to the name physical_group to restore legacy convention
        """
        rename_map = {
            name: f"physical_group_{name}"
            for name in self.subdomains
            if not name.startswith("physical_group_")
        }
        self.rename_subdomains(rename_map)

    @staticmethod
    def partmesh_prepare(
        domain_file: Path | str, output_path: Path | str, dry_run: bool = False
    ) -> Path:
        """
        Creates a metis files. This file is needed for partitioning the OGS input mesh (for parallel OGS compution).

        :param domain_file: A Path to existing domain mesh file (.vtu extension)
        :param output:      A Path to existing folder. Here the resulting metis file will be stored (.mesh)
        :param dry_run:     If True: Writes no files, but returns the list of files expected to be created
                            If False: Writes files and returns the list of created files

        :returns:           Path to the generated metis file.
        """

        from ogstools import cli

        if not domain_file.exists():
            msg = f"File does not exist: {domain_file}"
            if not dry_run:
                raise FileExistsError(msg)
            print(msg)

        cmd = f"partmesh -o {output_path} -i {domain_file} --ogs2metis"
        if dry_run:
            print(cmd)
        else:
            ret = cli().partmesh(o=output_path, i=domain_file, ogs2metis=True)
            if ret:
                msg = f"partmesh -o {output_path} -i {domain_file} --ogs2metis failed with return value {ret}"
                raise ValueError(msg)

        return output_path / (Path(domain_file).stem + ".mesh")

    @staticmethod
    def partmesh(
        num_partitions: int,
        metis_file: Path | str,
        domain_file: Path | str,
        subdomain_files: Sequence[Path | str],
        dry_run: bool = False,
    ) -> list[Path]:
        """
        Creates a folder in the folder where the metis_file is. Puts .bin files into this folder that are needed
        as input files for running OGS parallel (MPI).
        Wrapper around command line tool partmesh, adding file checks, dry-run option, normalized behaviour for partition == 1
        Only use this function directly when you want to bypass creating the Meshes object
        (e.g. files for domain and subdomains are already present)

        :param num_partitions:  List of integers or a single integer that indicate the number of partitions
                            similar to the OGS binary tool partmesh.
                            The serial mesh will always be generated
                            Example 1: num_partitions = [1,2,4,8,16]
                            Example 2: num_partitions = 2

        :param meshes_path: Optional path to the directory where meshes
                            should be saved. It must already exist (will not be created).
                            If None, a temporary directory will be used.

        :param metis_file:   A path to existing metis partitioned file (.mesh extension).

        :param domain_file: A path to existing domain mesh file (.vtu extension)

        :param dry_run:     If True: Writes no files, but returns the list of files expected to be created
                            If False: Writes files and returns the list of created files

        :returns:           A list of Paths pointing to the saved mesh files, if num_partitions are given (also just [1]),
                            then it returns
                            A dict, with keys representing the number of partitions and values A list of Paths (like above)
        """

        from ogstools import cli

        partition_path = metis_file.parent / str(num_partitions)

        missing_files = [
            f
            for f in subdomain_files + [domain_file] + [metis_file]
            if not f.exists()
        ]

        if missing_files:
            missing_str = ", ".join(str(f) for f in missing_files)
            msg = f"The following files do not exist: {missing_str}"
            if not dry_run:
                raise FileExistsError(msg)
            print(dry_run)

        if not dry_run:
            partition_path.mkdir(parents=True, exist_ok=True)

        if num_partitions == 1:
            files = [
                file.parent / "partition" / "1" / file.name
                for file in subdomain_files + [domain_file]
            ]
            if dry_run:
                return files
            for file_link in files:

                if file_link.exists() or file_link.is_symlink():
                    file_link.unlink()
                file_link.symlink_to(Path("../..") / file_link.name)

            return list(partition_path.glob("*"))

        file_names = [
            "node_properties_val",
            "node_properties_cfg",
            "cell_properties_val",
            "cell_properties_cfg",
            "msh_nod",
            "msh_ele",
            "msh_ele_g",
            "msh_cfg",
        ]

        if dry_run:
            subdomain_files = [
                Path(f"{subdomain_file.stem}_{file_part}{num_partitions}.bin")
                for file_part in file_names
                for subdomain_file in subdomain_files
            ]
            domain_files = [
                Path(f"{domain_file.stem}_{file_part}{num_partitions}.bin")
                for file_part in file_names[2:]
            ]

            return subdomain_files + domain_files

        cli().partmesh(
            o=partition_path,
            i=domain_file,
            m=True,
            n=num_partitions,
            x=metis_file.parent / metis_file.stem,  # without .mesh extension
            *subdomain_file_paths,  # noqa: B026
        )
        return list(partition_path.glob("*"))

    def _partmesh_save_all(
        self,
        num_partitions: Sequence[int],
        meshes_path: Path | str,
        dry_run: bool = False,
    ) -> dict[int, list[Path]]:
        """
        Creates a folder with decomposed / partitioned mesh suitable for parallel OGS execution

        :param num_partitions:  The number of partitions (internally passed to OGS bin tool partmesh)

        :param meshes_path:     Path to the folder where the original serial mesh files are already stored.
                                E.g. `save` must be called before.
        :param dry_run:     If True: Writes no files, but returns the list of files expected to be created
                            If False: Writes files and returns the list of created files

        :returns:           A dict with key indicating the number of partition and value a list of
                            Paths of the generated files
        """

        meshes_path = Path(meshes_path)
        domain_file_name = self.domain_name + ".vtu"
        domain_file = meshes_path / domain_file_name
        parallel_path = meshes_path / "partition"
        metis_file = self.partmesh_prepare(domain_file, parallel_path, dry_run)

        meshes_path = metis_file.parent.parent
        subdomain_files = [
            meshes_path / (subdomain + ".vtu") for subdomain in self.subdomains
        ]

        parallel_files: dict[int, list[Path]] = {
            partition: self.partmesh(
                partition, metis_file, domain_file, subdomain_files, dry_run
            )
            for partition in num_partitions
        }

        return parallel_files

    def save(
        self,
        meshes_path: Path | str | None = None,
        overwrite: bool = False,
        num_partitions: int | Sequence[int] | None = None,
        dry_run: bool = False,
    ) -> list[Path] | dict[int, list[Path]]:
        """
        Save all meshes.

        This function will perform identifySubdomains, if not yet been done.

        :param meshes_path: Optional path to the directory where meshes
                            should be saved. It must already exist (will not be created).
                            If None, a temporary directory will be used.

        :param overwrite:   If True, existing mesh files will be overwritten.
                            If False, an error is raised if any file already exists.

        :param num_partitions:  List of integers or a single integer that indicate the number of partitions
                            similar to the OGS binary tool partmesh.
                            The serial mesh will always be generated
                            Example 1: num_partitions = [1,2,4,8,16]
                            Example 2: num_partitions = 2

        :param dry_run:     If True: Writes no files, but returns the list of files expected to be created
                            If False: Writes files and returns the list of created files

        :returns:           A list of Paths pointing to the saved mesh files, if num_partitions are given (also just [1]),
                            then it returns
                            A dict, with keys representing the number of partitions and values A list of Paths (like above)
        """
        meshes_path = meshes_path or self._tmp_dir
        if isinstance(num_partitions, int):
            num_partitions = [num_partitions]
        serial_files = [meshes_path / f"{name}.vtu" for name in self._meshes]

        if not dry_run:
            meshes_path.mkdir(parents=True, exist_ok=True)

            if not self.has_identified_subdomains:
                identify_subdomains(self.domain, list(self.subdomains.values()))
                self.has_identified_subdomains = True

            output_files = [
                meshes_path / f"{name}.vtu" for name in self._meshes
            ]
            existing_files = [f for f in output_files if f.exists()]
            if existing_files and not overwrite:
                existing_list = "\n".join(str(f) for f in existing_files)
                msg = f"The following mesh files already exist:{existing_list}. Set `overwrite=True` to overwrite them, or choose a different `meshes_path`."

                raise FileExistsError(msg)

            for name, mesh in self._meshes.items():
                mesh.filepath = meshes_path / f"{name}.vtu"
                pv.save_meshio(mesh.filepath, mesh)

        if not num_partitions:
            return serial_files

        parallel_files = self._partmesh_save_all(
            num_partitions, meshes_path, dry_run
        )
        return {1: serial_files} | parallel_files
