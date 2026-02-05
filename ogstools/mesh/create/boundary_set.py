# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import chain, pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from ogstools import mesh

from .boundary import Layer, LocationFrame, Raster
from .boundary_subset import Surface
from .region import RegionSet


class BoundarySet(ABC):
    """
    Abstract base class representing a collection of boundaries with constraints.

    A BoundarySet is composed of multiple boundaries linked with constraints:
    - Free of gaps and overlaps.
    - Distinguished by markers to identify different boundaries.
    - Adherence to rules of `piecewise linear complex (PLC)`.
    """

    @abstractmethod
    def bounds(self) -> list:
        return []

    @abstractmethod
    def filenames(self) -> list[Path]:
        return []


class LayerSet(BoundarySet):
    """
    Collection of geological layers stacked to represent subsurface arrangements.

    In a geological information system, multiple layers can be stacked vertically to
    represent the subsurface arrangement. This class provides methods to manage and
    work with layered geological data.
    """

    def __init__(self, layers: list[Layer]):
        """
        Initializes a LayerSet. It checks if the list of provided layers are given in a top to bottom order.
        In neighboring layers, layers share the same surface (upper bottom == low top).
        """

        for upper, lower in pairwise(layers):
            if upper.bottom != lower.top:
                msg = "Layerset is not consistent."
                raise ValueError(msg)
        self.layers = layers

    def bounds(self) -> list:
        return list(self.layers[0].top.mesh.bounds)

    def filenames(self) -> list[Path]:
        layer_filenames = [layer.bottom.filename for layer in self.layers]
        layer_filenames.insert(0, self.layers[0].top.filename)  # file interface
        return layer_filenames

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "LayerSet":
        """Create a LayerSet from a Pandas DataFrame."""
        Row = namedtuple("Row", ["material_id", "mesh", "resolution"])
        surfaces = [
            Row(
                material_id=surface._asdict()["material_id"],
                mesh=surface._asdict()["filename"],
                resolution=surface._asdict()["resolution"],
            )
            for surface in df.itertuples(index=False)
        ]
        base_layer = [
            Layer(
                top=Surface(top.mesh, material_id=top.material_id),
                bottom=Surface(bottom.mesh, material_id=bottom.material_id),
                num_subdivisions=bottom.resolution,
            )
            for top, bottom in pairwise(surfaces)
        ]
        return cls(layers=base_layer)

    def create_raster(
        self, resolution: float, margin: float = 0.0
    ) -> tuple[Path, Path]:
        """
        Create raster representations for the LayerSet.

        This method generates raster files at a specified resolution for each layer's
        top and bottom boundaries and returns paths to the raster files.

        :param resolution: The resolution for raster creation.
        :param margin: ratio by which to shrink the raster boundary (0.01 == 1%)
        """
        bounds = self.layers[0].top.mesh.bounds
        raster_set = self.create_rasters(resolution=resolution)

        locFrame = LocationFrame(
            xmin=bounds[0] * (1 + margin),
            xmax=bounds[1] * (1 - margin),
            ymin=bounds[2] * (1 + margin),
            ymax=bounds[3] * (1 - margin),
        )
        # Raster needs to be finer then asc files. Otherwise Mesh2Raster fails
        # TODO: it should also be possible to have a raster which is not
        # aligned with the cardinal directions, OR do an automatic rotation, if
        # the surfaces are rectangular, but rotated.
        raster = Raster(locFrame, resolution=resolution * 0.95)
        raster_vtu = Path(tempfile.mkstemp(".vtu", "raster")[1])
        raster.as_vtu(raster_vtu)

        rastered_layers_txt = Path(
            tempfile.mkstemp(".txt", "rastered_layers")[1]
        )
        with rastered_layers_txt.open("w") as file:
            file.write("\n".join(str(item) for item in raster_set))
        return raster_vtu, rastered_layers_txt

    def create_rasters(self, resolution: float) -> list[Path]:
        """
        For each surface a (temporary) raster file with given resolution is created.

        :param resolution: The resolution for raster creation.
        """
        rasters = [self.layers[0].top.create_raster_file(resolution=resolution)]
        for layer in self.layers:
            r = layer.create_raster(resolution=resolution)
            rasters.extend(r[1:])

        return list(dict.fromkeys(rasters))

    def refine(self, factor: int) -> "LayerSet":
        """
        Refine the LayerSet by increasing the number of subdivisions.

        This function refines the LayerSet by increasing the number of subdivisions
        in each layer. The factor parameter determines the degree of refinement.

        :param layerset: The original LayerSet to be refined.
        :param factor:   The refinement factor for the number of subdivisions.

        :returns: A new LayerSet with increased subdivisions for each layer.
        """

        def refined_num_subsections(num_subsections: int, factor: int) -> int:
            return (num_subsections + 1) * factor - 1

        out = [
            Layer(
                layer.top,
                layer.bottom,
                material_id=layer.material_id,
                num_subdivisions=refined_num_subsections(
                    layer.num_subdivisions, factor
                ),
            )
            for layer in self.layers
        ]

        return LayerSet(layers=out)

    def to_region_prism(
        self, resolution: float, margin: float = 0.0
    ) -> RegionSet:
        """
        Convert a layered geological structure into a RegionSet using prism meshing.

        This function takes a :class:`boundary_set.LayerSet` and converts it into a :class:`region.RegionSet` object using prism or tetrahedral meshing technique.
        The function will use prism elements for meshing if possible; otherwise, it will use
        tetrahedral elements.

        :param resolution: The desired resolution in [meter] for meshing. It must greater than 0.
        :param margin: ratio by which to shrink the raster boundary (0.01 == 1%)

        :returns: A :class:`boundary_set.LayerSet` object containing the meshed representation of the geological structure.

        raises:
            ValueError: If an error occurs during the meshing process.

        example:
            layer_set = LayerSet(...)
            resolution = 0.1
            region_set = layer_set.to_region_prism(resolution)
        """

        raster_vtu, rastered_layers_txt = self.create_raster(
            resolution=resolution, margin=margin
        )

        outfile = Path(tempfile.mkstemp(".vtu", "region_prism")[1])

        from ogstools._find_ogs import cli

        ret = cli().createLayeredMeshFromRasters(  # type: ignore[union-attr]
            i=raster_vtu, r=rastered_layers_txt, o=outfile
        )
        if ret:
            raise ValueError

        materials_in_domain: list[int] = list(
            chain.from_iterable(
                [layer.material_id] * (layer.num_subdivisions + 1)
                for layer in self.layers
            )
        )

        pv_mesh = mesh.read(outfile)
        intermediate_vtu_ids = list(set(pv_mesh.cell_data["MaterialIDs"]))
        # reversed bc createLayeredMeshFromRasters starts numbering from the bottom
        # up, but we number the layers from top to bottom
        id_mapping = dict(
            zip(intermediate_vtu_ids, materials_in_domain[::-1], strict=False)
        )
        new_ids = [
            id_mapping[old_id] for old_id in pv_mesh.cell_data["MaterialIDs"]
        ]
        pv_mesh.cell_data["MaterialIDs"].setfield(new_ids, np.uint32)

        return RegionSet(input=pv_mesh)

    def to_region_simplified(
        self, xy_resolution: float, rank: int
    ) -> RegionSet:
        """
        Convert a layered geological structure to a simplified meshed region.

        This function converts a layered geological structure represented by a `LayerSet` into a
        simplified meshed region using the specified `xy_resolution` and `rank`.

        :param xy_resolution (float):   The desired spatial resolution of the mesh in the XY plane.
        :param rank (int):              The rank of the mesh (2 for 2D, 3 for 3D).

        :returns: A `RegionSet` object containing the simplified meshed representation of the geological structure.

        raises:
            AssertionError: If the length of the `bounds` retrieved from the `layer_set` is not 6.

        example:
            layer_set = LayerSet(...)
            xy_resolution = 0.1  # Example resolution in XY plane
            rank = 2  # Mesh will be 2D
            region_set = layer_set.to_region_simplified(xy_resolution, rank)
        """

        bounds = self.bounds()

        assert len(bounds) == 6
        simple_meshes = [
            layer.to_simplified_mesh(
                resolution=xy_resolution, rank=rank, bounds=bounds
            )
            for layer in self.layers
        ]
        merged_mesh = simple_meshes[0].merge(
            simple_meshes[1:], merge_points=True
        )

        from ogstools._find_ogs import cli

        tmp_dir = Path(tempfile.mkdtemp("to_region_simplified"))
        mesh.save(merged_mesh, outfile := (tmp_dir / "domain.vtu"))
        cli().NodeReordering(i=str(outfile), o=str(outfile))

        return RegionSet(input=mesh.read(outfile))

    def to_region_tetrahedron(
        self, resolution: int, margin: float = 0.0
    ) -> RegionSet:
        """
        Convert a layered geological structure to a tetrahedral meshed region.

        This function converts a layered geological structure represented by a `LayerSet`
        into a tetrahedral meshed region using the specified `resolution`.

        :param resolution: The desired resolution for meshing.
        :param margin: ratio by which to shrink the raster boundary (0.01 == 1%)

        :returns: A `RegionSet` object containing the tetrahedral meshed representation of the geological structure.

        raises:
            ValueError: If an error occurs during the meshing process.

        notes:
            - The `resolution` parameter determines the desired spatial resolution of the mesh.
            - The function utilizes tetrahedral meshing using Tetgen software to create the meshed representation.
            - The resulting mesh is tetrahedral, and material IDs are assigned to mesh cells based on the geological layers.

        example:
            layer_set = LayerSet(...)
            resolution = 1  # Example resolution for meshing
            region_set = layer_set.to_region_tetrahedron(resolution)
        """

        raster_vtu, rastered_layers_txt1 = self.create_raster(
            resolution=resolution, margin=margin
        )

        smesh_file = Path(tempfile.mkstemp(".smesh", "region_tetrahedron")[1])

        from ogstools._find_ogs import cli

        ret_smesh = cli().createTetgenSmeshFromRasters(  # type: ignore[union-attr]
            i=raster_vtu, r=rastered_layers_txt1, o=smesh_file
        )
        if ret_smesh:
            raise ValueError
        try:
            subprocess.run(
                ["tetgen", "-pkABEFN", str(smesh_file)],
                check=True,
                capture_output=True,
                text=True,
            )

        except subprocess.CalledProcessError as e:
            m = f"Tetgen failed with return code {e.returncode}"
            raise ValueError(m) from e

        except FileNotFoundError as e:
            m = f"Tetgen could not be found. You need to install it on your system. Error: {e}"
            raise ValueError(m) from e

        except Exception as e:
            m = f"An unexpected error occurred when calling tetgen: {e}"
            raise ValueError(m) from e

        outfile = smesh_file.with_suffix(".1.vtk")
        if not outfile.exists():
            path = outfile.parent
            os.listdir(path)

        materials_in_domain: list[int] = list(
            chain.from_iterable(
                [layer.material_id] * (layer.num_subdivisions + 1)
                for layer in self.layers
            )
        )

        pv_mesh = mesh.read(outfile)
        region_attribute_name = "cell_scalars"
        if region_attribute_name in pv_mesh.cell_data:
            pv_mesh.cell_data["MaterialIDs"] = pv_mesh.cell_data.pop(
                region_attribute_name
            )

            intermediate_vtu_ids = sorted(
                dict.fromkeys(pv_mesh.cell_data["MaterialIDs"])
            )
            # reversed bc createLayeredMeshFromRasters starts numbering from the bottom
            # up, but we number the layers from top to bottom
            id_mapping = dict(
                zip(
                    intermediate_vtu_ids,
                    materials_in_domain[::-1],
                    strict=False,
                )
            )
            new_ids = [
                id_mapping[old_id]
                for old_id in pv_mesh.cell_data["MaterialIDs"]
            ]
            pv_mesh.cell_data["MaterialIDs"].setfield(new_ids, np.uint32)

        return RegionSet(input=pv_mesh)

    def to_region_voxel(self, resolution: list) -> RegionSet:
        """
        Convert a layered geological structure to a voxelized mesh.

        This function converts a layered geological structure represented by a `LayerSet`
        into a voxelized mesh using the specified `resolution`.

        :param resolution: A list of [x_resolution, y_resolution, z_resolution] for voxelization.

        :returns: A `Mesh` object containing the voxelized mesh representation of the geological structure.

        raises:
            ValueError: If an error occurs during the voxelization process.

        example:
            layer_set = LayerSet(...)
            resolution = [0.1, 0.1, 0.1]  # Example voxelization resolutions in x, y, and z dimensions
            voxel_mesh = layer_set.to_region_voxel(resolution)
        """

        layers_txt = Path(tempfile.mkstemp(".txt", "layers")[1])
        layer_filenames = self.filenames()
        with layers_txt.open("w") as file:
            file.write("\n".join(str(filename) for filename in layer_filenames))
        outfile = Path(tempfile.mkstemp(".vtu", "region_voxel")[1])

        from ogstools._find_ogs import cli

        ret = cli().Layers2Grid(  # type: ignore[union-attr]
            i=layers_txt,
            o=outfile,
            x=resolution[0],
            y=resolution[1],
            z=resolution[2],
        )
        if ret:
            raise ValueError()

        materials_in_domain = list(
            chain.from_iterable([layer.material_id] for layer in self.layers)
        )

        region_attribute_name = "MaterialIDs"

        pv_mesh = mesh.read(outfile)
        if region_attribute_name not in pv_mesh.cell_data:
            pv_mesh.cell_data[region_attribute_name] = pv_mesh.cell_data[
                region_attribute_name
            ]

        intermediate_vtu_ids = sorted(
            dict.fromkeys(pv_mesh.cell_data["MaterialIDs"])
        )
        # reversed bc createLayeredMeshFromRasters starts numbering from the bottom
        # up, but we number the layers from top to bottom
        k = zip(intermediate_vtu_ids, materials_in_domain[::-1], strict=False)
        id_mapping = dict(k)
        new_ids = [
            id_mapping[old_id] for old_id in pv_mesh.cell_data["MaterialIDs"]
        ]
        pv_mesh.cell_data["MaterialIDs"].setfield(new_ids, np.uint32)

        return RegionSet(input=pv_mesh)
