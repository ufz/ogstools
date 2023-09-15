import tempfile
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path

import pandas as pd

from ogstools.meshlib.boundary import Layer, LocationFrame, Raster
from ogstools.meshlib.boundary_subset import Surface


class BoundarySet(ABC):
    """
    Abstract base class representing a collection of boundaries with constraints.

    A BoundarySet is composed of multiple boundaries linked with constraints:
    - Free of gaps and overlaps.
    - Distinguished by markers to identify different boundaries.
    - Adherence to rules of `piecewise linear complex (PLC)`.
    """

    @abstractmethod
    def bounds(self):
        return

    @abstractmethod
    def filenames(self):
        return


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

        for upper, lower in zip(layers, layers[1:]):
            if upper.bottom != lower.top:
                msg = "Layerset is not consistent."
                raise ValueError(msg)
        self.layers = layers

    def bounds(self):
        return list(self.layers[0].top.mesh.bounds)

    def filenames(self):
        layer_filenames = [layer.bottom.filename for layer in self.layers]
        layer_filenames.insert(0, self.layers[0].top.filename)  # file interface
        return layer_filenames

    @classmethod
    def from_pandas(cls, df: pd.DataFrame):
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
            for top, bottom in zip(surfaces, surfaces[1:])
        ]
        return cls(layers=base_layer)

    def create_raster(self, resolution):
        """
        Create raster representations for the LayerSet.

        This method generates raster files at a specified resolution for each layer's
        top and bottom boundaries and returns paths to the raster files.
        """
        bounds = self.layers[0].top.mesh.bounds
        raster_set = self.create_rasters(resolution=resolution)

        locFrame = LocationFrame(
            xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3]
        )
        # Raster needs to be finer then asc files. Otherwise Mesh2Raster fails
        raster = Raster(locFrame, resolution=resolution * 0.95)
        raster_vtu = Path(tempfile.mkstemp(".vtu", "raster")[1])
        raster.as_vtu(raster_vtu)
        rastered_layers_txt = Path(
            tempfile.mkstemp(".txt", "rastered_layers")[1]
        )
        with rastered_layers_txt.open("w") as file:
            file.write("\n".join(str(item) for item in raster_set))
        return raster_vtu, rastered_layers_txt

    def create_rasters(self, resolution: int) -> list[Path]:
        """
        For each surface a (temporary) raster file with given resolution is created.
        """
        rasters = [self.layers[0].top.create_raster_file(resolution=resolution)]
        for layer in self.layers:
            r = layer.create_raster(resolution=resolution)
            rasters.extend(r[1:])

        return list(dict.fromkeys(rasters))
