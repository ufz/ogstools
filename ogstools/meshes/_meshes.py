# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

import copy
import logging as log
import os
from collections.abc import Iterator, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import yaml
from matplotlib import pyplot as plt
from typing_extensions import Self

import ogstools.mesh as Mesh
from ogstools._internal import deprecated
from ogstools.core.storage import StorageBase
from ogstools.mesh import check_datatypes, utils

logger = log.getLogger(__name__)


class Meshes(MutableMapping, StorageBase):
    """
    OGS input mesh. Refers to prj - file section <meshes>
    """

    def __init__(
        self,
        meshes: dict[str, pv.UnstructuredGrid],
        id: str | None = None,
    ) -> None:
        """
        Initialize a Meshes object.
            :param meshes:      List of Mesh objects (pyvista UnstructuredGrid)
                                The first mesh is the domain mesh.
                                All following meshes represent subdomains, and their points must align with points on the domain mesh.
                                If needed, the domain mesh itself can also be added again as a subdomain.
            :returns:           A Meshes object
        """
        super().__init__("Meshes", "", id)
        self._meshes = meshes
        self.has_identified_subdomains: bool = False
        self.num_partitions: int | Sequence[int] | None = None

    @classmethod
    def from_folder(cls, filepath: str | Path) -> Meshes:
        """
        Create a Meshes object from a folder of already save Meshes.
        Reverse of .save. It need a meta.yaml file in the specified folder.
        """
        filepath = Path(filepath)
        import yaml

        meta_file_path = filepath / "meta.yaml"

        assert meta_file_path.exists()
        with Path(meta_file_path).open("r") as f:
            restored_data = yaml.safe_load(f)

        mesh_names = list(restored_data["meshes"])
        has_identified_subdomain = restored_data["has_identified_subdomains"]
        num_partitions = restored_data["num_partitions"]

        meshes = cls.from_files(
            [filepath / (mesh + ".vtu") for mesh in mesh_names],
            domain_key=mesh_names[0],
        )
        meshes.has_identified_subdomains = has_identified_subdomain
        meshes.num_partitions = num_partitions

        return meshes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Meshes):
            return NotImplemented

        # Quick structural checks first
        if (
            len(self._meshes) != len(other._meshes)
            or self.has_identified_subdomains != other.has_identified_subdomains
            or self.num_partitions != other.num_partitions
            or len(self.subdomains) != len(other.subdomains)
            # TODO: use mesh compare
            # or self.domain != other.domain
            or self.domain_name != other.domain_name
        ):
            return False

        for key, mesh1 in self.subdomains.items():
            mesh2 = other.subdomains[key]

            mesh2.active_scalars_name = None
            mesh1.active_scalars_name = None
            if mesh1.point_data != mesh2.point_data:
                return False
            if mesh1.cell_data != mesh2.cell_data:
                return False

            # TODO: when merge with MeshSeries compare
            if mesh1 != mesh2:  # or mesh1.equal(mesh2) if you have that
                return False

        self.domain.active_scalars_name = None
        other.domain.active_scalars_name = None
        if self.domain.cell_data != other.domain.cell_data:
            return False
        if self.domain.point_data != other.domain.point_data:
            return False

        return True

    def __deepcopy__(self, memo: dict) -> Meshes:
        if id(self) in memo:
            return memo[id(self)]

        # Deep-copy each mesh explicitly (pyvista objects are mutable)
        meshes_copy = {
            name: mesh.copy(deep=True) for name, mesh in self._meshes.items()
        }

        new = self.__class__(meshes=meshes_copy)
        new.has_identified_subdomains = self.has_identified_subdomains
        new.num_partitions = copy.deepcopy(self.num_partitions, memo)

        memo[id(self)] = new
        return new

    def __getitem__(self, key: str) -> pv.UnstructuredGrid:
        if key not in self._meshes:
            msg = f"Key {key!r} not found"
            raise KeyError(msg)
        return self._meshes[key]

    def __setitem__(self, key: str, mesh: pv.UnstructuredGrid) -> None:
        self.has_identified_subdomains = False
        self._meshes[key] = mesh

    def __delitem__(self, key: str) -> None:
        del self._meshes[key]

    def __len__(self) -> int:
        return len(self._meshes)

    def __iter__(self) -> Iterator[str]:
        yield from self._meshes

    @classmethod
    def from_files(
        cls, filepaths: Sequence[str | Path], domain_key: str = "domain"
    ) -> Self:
        """Initialize a Meshes object from a Sequence of existing files.

        :param filepaths:   Sequence of Mesh files (.vtu)
                            The first mesh is the domain mesh.
                            All following meshes represent subdomains, and their
                            points must align with points on the domain mesh.
        :param domain_key:  String which is only in the domain filepath

        """

        meshes = cls(
            {
                Path(m).stem: pv.read(m)
                for m in sorted(
                    filepaths, key=lambda fp: str(fp).replace(domain_key, "")
                )
            }
        )
        meshes._bind_to_path(Path(filepaths[0]).parent)
        return meshes

    @classmethod
    def from_file(cls, meta_file: str | Path) -> Self:
        """
        Restore a Meshes object from a meta.yaml file.

        :param meta_file: Path to the meta.yaml file written by Meshes.save()
        :returns:         A restored Meshes object
        """
        meta_file = Path(meta_file)
        if not meta_file.exists():
            msg = f"Meta file does not exist: {meta_file}"
            raise FileNotFoundError(msg)

        base_path = meta_file.parent

        with meta_file.open("r") as f:
            restored_data = yaml.safe_load(f)

        mesh_names = restored_data.get("meshes", [])
        if not mesh_names:
            msg = "meta.yaml does not contain any mesh entries"
            raise ValueError(msg)

        meshes = cls.from_files(
            [base_path / f"{name}.vtu" for name in mesh_names],
            domain_key=mesh_names[0],
        )

        meshes.has_identified_subdomains = restored_data.get(
            "has_identified_subdomains", False
        )
        meshes.num_partitions = restored_data.get("num_partitions", None)

        return meshes

    @classmethod
    def from_id(cls, meshes_id: str) -> Self:
        """
        Load Meshes from the user storage path using its ID. Storage.UserPath must be set

        :param meshes_id: The unique ID of the Meshes object to load.
        :returns:         A Meshes instance restored from disk.
        """
        meshes_folder = StorageBase.saving_path() / "Meshes" / meshes_id
        if not meshes_folder.exists():
            msg = f"No meshes found at {meshes_folder}"
            raise FileNotFoundError(msg)

        meta_file = meshes_folder / "meta.yaml"
        if not meta_file.exists():
            msg = f"Missing meta.yaml in meshes folder: {meshes_folder}"
            raise FileNotFoundError(msg)

        meshes = cls.from_file(meta_file)
        meshes._id = meshes_id
        return meshes

    @classmethod
    def from_gmsh(
        cls,
        filename: Path,
        dim: int | Sequence[int] = 0,
        reindex: bool = True,
        log: bool = True,
        meshname: str = "domain",
    ) -> Self:
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
        from ogstools.meshes.gmsh_converter import meshes_from_gmsh

        meshes_dict = meshes_from_gmsh(filename, dim, reindex, log, meshname)
        meshes_obj = cls(meshes_dict)
        meshes_obj.has_identified_subdomains = True
        return meshes_obj

    @classmethod
    def from_yaml(cls, geometry_file: Path) -> Self:
        """ """

        from ogstools.meshes._meshes_from_yaml import meshes_from_yaml

        gmsh_file = meshes_from_yaml(geometry_file)
        print(f"Info: Mesh written to {gmsh_file}")
        return cls.from_gmsh(gmsh_file)

    @classmethod
    def from_mesh(
        cls,
        mesh: pv.UnstructuredGrid,
        threshold_angle: float | None = 15.0,
        domain_name: str = "domain",
    ) -> Self:
        """Create Meshes by extract boundaries of domain mesh.

        The provided mesh must be already conforming to OGS standards.

        :param mesh:            The domain mesh
        :param threshold_angle: If None, the boundary will be split by the
                                assumption of vertical lateral boundaries. Otherwise
                                it represents the angle (in degrees) between
                                neighbouring elements which - if exceeded -
                                determines the corners of the boundary mesh.
        :param domain_name:     The name of the domain mesh.
        :returns:               A Meshes object.
        """
        from ogstools.meshes.subdomains import extract_boundaries

        sub_meshes_dict = extract_boundaries(mesh, threshold_angle)

        meshes_dict = {domain_name: mesh} | sub_meshes_dict
        meshes_obj = cls(meshes_dict)
        meshes_obj.has_identified_subdomains = False
        return meshes_obj

    def add_gml_subdomains(
        self,
        domain_path: Path,  # TODO: get from self?
        gml_path: Path,
        out_dir: Path | None = None,
        tolerance: float = 1e-12,
    ) -> None:
        """Add Meshes from geometry definition in the gml file to subdomains.

        :param gml_file:    Path to the gml file.
        :param out_dir:     Where to write the gml meshes (default: gml dir)
        :param tolerance:   search length for node search algorithm
        """
        from ogstools._find_ogs import cli

        out_dir = gml_path.parent if out_dir is None else out_dir
        prev_files = set(out_dir.glob("*.vtu"))

        cur_dir = Path.cwd()
        os.chdir(out_dir)
        assert gml_path.exists()
        assert domain_path.exists()

        cli().constructMeshesFromGeometry(
            "-g", gml_path, "-m", domain_path, "-s", str(tolerance)
        )
        os.chdir(cur_dir)

        gml_meshes = sorted(set(out_dir.glob("*.vtu")).difference(prev_files))
        for file in gml_meshes:
            filename = str(file.stem)
            prefix = f"{gml_path.stem}_geometry_"
            prefix_offset = len(prefix) if filename.startswith(prefix) else 0
            self[filename[prefix_offset:]] = pv.read(file)

    def sort(self) -> None:
        "Sort the subdomains alphanumerically."
        self._meshes = self._meshes
        sorted_subdomains = dict(sorted(self.subdomains.items()))
        self._meshes = {self.domain_name: self.domain} | sorted_subdomains

    def _save_impl(self, dry_run: bool, **kwargs: Any) -> list[Path]:
        active_path = Path(self.next_target)

        if isinstance(self.num_partitions, int):
            self.num_partitions = [self.num_partitions]
        serial_files = [active_path / f"{name}.vtu" for name in self._meshes]
        meta_file = active_path / "meta.yaml"

        if not dry_run:
            active_path.mkdir(parents=True, exist_ok=True)

            if not self.has_identified_subdomains:
                self.identify_subdomain()

            set_pv_attr = getattr(pv, "set_new_attribute", setattr)
            for name, mesh in self._meshes.items():
                check_datatypes(mesh, strict=True, meshname=name)
                filepath = active_path / f"{name}.vtu"
                set_pv_attr(mesh, "filepath", filepath)
                Mesh.save(mesh, filepath, **kwargs)

            meta_dict = {
                "meshes": list(self._meshes.keys()),
                "has_identified_subdomains": self.has_identified_subdomains,
                "num_partitions": self.num_partitions,
            }

            yaml_string = yaml.dump(meta_dict)

            with meta_file.open("w") as f:
                f.write(yaml_string)

        if not self.num_partitions:
            return serial_files + [meta_file]

        parallel_files = self._partmesh_save_all(
            self.num_partitions, active_path, dry_run
        )
        flattened_parallel_files = [
            file for partition in parallel_files.values() for file in partition
        ]

        return serial_files + [meta_file] + flattened_parallel_files

    def save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        archive: bool = False,
        id: str | None = None,
        **kwargs: Any,
    ) -> list[Path]:
        """
        Save all meshes. If num_partitions is specified (e.g. obj.num_partition = [2,3]) is also create paritioned meshes for each partition.

        This function will perform identifySubdomains, if not yet been done.

        :param target:      Optional path to the folder where meshes
                            should be saved. If None, a temporary folder will be used.

        :param overwrite:   If True, existing mesh files will be overwritten.
                            If False, an error is raised if any file already exists.

        :param dry_run:     If True: Writes no files, but returns the list of files expected to be created
                            If False: Writes files and returns the list of created files.

        :param archive:     If True: The folder specified by path contains no symlinks. It copies all referenced data (which might take time and space).

        :param id:          Optional identifier. Mutually exclusive with path.

        :returns:           A list of Paths pointing to the saved mesh files (including files for partitions).
        """

        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run, **kwargs)
        self._post_save(user_defined, archive, dry_run)
        return files

    def _propagate_target(self) -> None:
        pass

    def validate(self, strict: bool = True) -> bool:
        """
        Validate all meshes for conformity with OGS requirements.

        Meshes must be saved before validation can be performed.

        :param strict: If True, raise a UserWarning if validation fails.
        :returns: True if all meshes pass validation, False otherwise.
        :raises ValueError: If meshes have not been saved yet.
        """
        if not self.is_saved:
            msg = "Meshes must be saved before validation."
            raise ValueError(msg)
        return all(
            utils.validate(mesh.filepath, strict=strict)
            for mesh in self._meshes.values()
        )

    @property
    def domain(self) -> pv.UnstructuredGrid:
        """
        Get the domain mesh.

        By convention, the domain mesh is the **first mesh** in the dictionary
        of meshes when the `Meshes` object was constructed.
        The domain mesh is expected to be constant. e.g. Do not: myobject.domain = pv.Sphere()


        :returns: The domain mesh
        """
        return next(iter(self._meshes.values()))

    @domain.setter
    def domain(self, new_domain: pv.UnstructuredGrid) -> None:
        self._meshes[self.domain_name] = new_domain

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
    def subdomains(self) -> dict[str, pv.UnstructuredGrid]:
        """
        Get the subdomain meshes.

        By convention, all meshes **except the first** one are considered subdomains.
        This returns a list of those subdomain meshes.

        :returns: A dictionary of {name: Mesh} for all subdomains
        """
        items = list(self._meshes.items())
        return dict(items[1:])  # by convention: first mesh is domain

    def identify_subdomain(self, include_domain: bool = False) -> None:
        from ogstools.meshes.subdomains import identify_subdomains

        if include_domain:
            identify_subdomains(self.domain, list(self.values()))
        else:
            identify_subdomains(self.domain, list(self.subdomains.values()))
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

    def modify_names(self, prefix: str = "", suffix: str = "") -> None:
        """
        Add prefix and/or suffix to names of meshes.
        Separators (underscore, etc.) have to be provided by user
        as part of prefix and/or suffix parameters.

        :param prefix: Prefix to be added at the beginning of the mesh name
        :param suffix: Suffix to be added after the mesh name
        """
        items = list(self._meshes.items())
        self._meshes = {f"{prefix}{name}{suffix}": mesh for name, mesh in items}

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
        self.modify_names(prefix="physical_group_")

    @staticmethod
    def create_metis(
        domain_file: Path | str, output_path: Path | str, dry_run: bool = False
    ) -> Path:
        """
        Creates a metis files. This file is needed to partition the OGS input mesh (for parallel OGS compution)
        using the OGS cmd line tool partmesh.

        :param domain_file: A Path to existing domain mesh file (.vtu extension)
        :param output:      A Path to existing folder. Here the resulting metis file will be stored (.mesh)
        :param dry_run:     If True: Metis file is not written
                            If False: Metis file is written

        :returns:           Path to the generated metis file.
        """

        from ogstools import cli

        domain_file = Path(domain_file)
        output_path = Path(output_path)

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
    def create_partitioning(
        num_partitions: int,
        domain_file: Path | str,
        subdomain_files: Sequence[Path | str],
        metis_file: Path | str | None = None,
        dry_run: bool = False,
    ) -> list[Path]:
        """
        Creates a subfolder in the metis_file' folder. Puts .bin files into this folder that are needed
        as input files for running OGS parallel (MPI).
        Wrapper around command line tool partmesh, adding file checks, dry-run option, normalized behaviour for partition == 1
        Only use this function directly when you want to bypass creating the Meshes object
        (e.g. files for domain and subdomains are already present)

        :param num_partitions:  List of integers or a single integer that indicate the number of partitions
                            similar to the OGS binary tool partmesh.
                            The serial mesh will always be generated
                            Example 1: num_partitions = [1,2,4,8,16]
                            Example 2: num_partitions = 2

        :param domain_file: A Path to existing domain mesh file (.vtu extension)

        :param subdomain_files:
                            A list of Path to existing subdomain files (.vtu extensions)

        :param metis_file:  A Path to existing metis partitioned file (.mesh extension).

        :param dry_run:     If True: Writes no files, but returns the list of files expected to be created
                            If False: Writes files and returns the list of created files

        :returns:           A list of Paths pointing to the saved mesh files, if num_partitions are given (also just [1]),
                            then it returns
                            A dict, with keys representing the number of partitions and values A list of Paths (like above)
        """

        from ogstools import cli

        domain_file = Path(domain_file)
        if not metis_file:
            parallel_path = domain_file.parent / "partition"
            metis_file = Meshes.create_metis(
                domain_file=domain_file,
                output_path=parallel_path,
                dry_run=dry_run,
            )

        metis_file = Path(metis_file)
        subdomain_file_paths: list = [Path(file) for file in subdomain_files]
        partition_path = metis_file.parent / str(num_partitions)

        missing_files = [
            f
            for f in subdomain_file_paths + [domain_file] + [metis_file]
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
                (partition_path / file.name, file)
                for file in subdomain_file_paths + [domain_file]
            ]
            if dry_run:
                return [symlink for symlink, _ in files]
            for file_link, file in files:
                if file_link.exists() or file_link.is_symlink():
                    file_link.unlink()
                rel = os.path.relpath(file.parent, file_link.parent)
                file_link.symlink_to(Path(rel) / file.name)

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
            subdomain_file_paths = [
                Path(f"{subdomain_file.stem}_{file_part}{num_partitions}.bin")
                for file_part in file_names
                for subdomain_file in subdomain_file_paths
            ]
            domain_files = [
                Path(f"{domain_file.stem}_{file_part}{num_partitions}.bin")
                for file_part in file_names
            ]

            return subdomain_file_paths + domain_files

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
        metis_file = self.create_metis(domain_file, parallel_path, dry_run)

        meshes_path = metis_file.parent.parent
        subdomain_files = [
            meshes_path / (subdomain + ".vtu") for subdomain in self.subdomains
        ]

        parallel_files: dict[int, list[Path]] = {
            partition: self.create_partitioning(
                partition, domain_file, subdomain_files, metis_file, dry_run
            )
            for partition in num_partitions
        }

        return parallel_files

    def plot(self, **kwargs: Any) -> plt.Figure:
        """Plot the domain mesh and the subdomains.

        keyword arguments: see :func:`ogstools.plot.contourf`
        """
        self.sort()

        from ogstools import plot

        fontsize = kwargs.pop("fontsize", plot.setup.fontsize)
        lw = kwargs.get("lw", kwargs.get("linewidth", 2))
        show_edges = kwargs.pop("show_edges", True)

        if len(np.unique(self.domain.cell_data.get("MaterialIDs", []))) > 1:
            var = "MaterialIDs"
        else:
            var = "None"
        cbar = kwargs.pop("cbar", var != "None")
        fig = plot.contourf(
            self.domain, var, show_edges=show_edges, cbar=cbar, **kwargs
        )

        if fig is None:
            assert "fig" in kwargs, "Only provide ax together with fig."
            fig = kwargs.get("fig")
        assert isinstance(fig, plt.Figure)
        ax: plt.Axes = fig.axes[0]

        for i, (name, mesh) in enumerate(self.items()):
            color = kwargs.get("color", plt.get_cmap("Set2")(i))

            # TODO: 1D, 3D
            if mesh.GetMaxSpatialDimension() == 1:
                plot.line(
                    mesh, ax=ax, label=name, lw=lw, color=color,
                    fontsize=fontsize, clip_on=False
                )  # fmt: skip
            else:
                if name == self.domain_name:
                    ax.plot([], [], "s", label=name, c="lightgrey", ms=16 * lw)
                else:
                    axes = plot.utils.get_projection(self.domain)[:2]
                    ax.plot(
                        *mesh.points[:, axes].T, "o",
                        label=name, clip_on=False, color=color, ms=8 * lw
                    )  # fmt: skip

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=0.9 * fontsize,
            borderaxespad=0.0,
            numpoints=1,
        )
        return fig

    def remove_material(
        self, mat_id: int | Sequence[int], tolerance: float = 1e-12
    ) -> None:
        """Remove material from meshes and update integration point data.

        :param mat_id:      MaterialID/s to be removed from domain, subdomain
                            elements, which only belonged to this material are
                            also removed. If given as a sequence, then it must
                            be of length 2 and all ids in between are removed.
        :param tolerance:   Absolute distance threshold to check if subdomain
                            nodes still have a corresponding domain node after
                            removal of the designated material.
        """
        from scipy.spatial import KDTree as scikdtree

        from ogstools._find_ogs import cli
        from ogstools.core.storage import _date_temp_path
        from ogstools.mesh.ip_mesh import ip_data_threshold

        mat_id = (mat_id, mat_id) if isinstance(mat_id, int) else mat_id

        cut_material: pv.UnstructuredGrid = self.domain.threshold(
            mat_id, scalars="MaterialIDs"
        )
        cut_material.clear_data()
        cut_material_file = Mesh.save(cut_material)

        # pyvista's extract_feature_edges doesn't handle quadratic elements
        boundary_file = _date_temp_path("Mesh_boundary", "vtu")
        boundary_file.parent.mkdir(exist_ok=True, parents=True)
        cli().ExtractBoundary(
            i=str(cut_material_file),
            o=str(boundary_file),
        )
        cut_boundary = pv.read(boundary_file)

        self["cut_boundary"] = cut_boundary

        new_ip_data = ip_data_threshold(self.domain, mat_id, invert=True)
        self.domain.field_data.update(new_ip_data)
        self.domain = self.domain.threshold(
            mat_id, scalars="MaterialIDs", invert=True
        )

        tree = scikdtree(self.domain.points)

        for name, subdomain in self.subdomains.items():
            distances = tree.query(subdomain.points)[0]
            new_subdomain = subdomain.extract_points(
                distances <= tolerance, adjacent_cells=False
            )
            if new_subdomain.n_cells == 0:
                msg = (
                    f"subdomain {name} has no remaining cells after removal of "
                    f"material {mat_id}. Thus, it is removed from meshes."
                )
                logger.warning(msg)
                self.pop(name)
            else:
                self[name] = new_subdomain

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        base_repr = super().__repr__()

        if self.user_specified_id:
            construct = f'{cls_name}.from_id("{self._id}")'
        elif self.is_saved:
            construct = f"{cls_name}.from_folder({str(self.active_target)!r})"
        else:
            mesh_info = {
                name: f"<Mesh: {mesh.n_points} points, {mesh.n_cells} cells>"
                for name, mesh in self._meshes.items()
            }
            construct = (
                f"{cls_name}(\n"
                f"  meshes={mesh_info},\n"
                f"  has_identified_subdomains={self.has_identified_subdomains},\n"
                f"  num_partitions={self.num_partitions}\n"
                f")"
            )

        return f"{construct}\n{base_repr}"

    def __str__(self) -> str:
        base_repr = super().__str__()

        lines = [
            f"{base_repr}",
            f"  Domain: {self.domain_name} "
            f"(cells={self.domain.n_cells}, points={self.domain.n_points})",
        ]
        if self.subdomains:
            lines.append("  Subdomains:")
            for name, mesh in self.subdomains.items():
                lines.append(
                    f"    - {name} (cells={mesh.n_cells}, points={mesh.n_points})"
                )
        else:
            lines.append("  Subdomains: none")

        lines.append(
            f"  Identified subdomains: {self.has_identified_subdomains}"
        )
        lines.append(f"  Number of partitions: {self.num_partitions}")

        return "\n".join(lines)
