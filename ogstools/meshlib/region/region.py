import os
import tempfile
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pyvista as pv
from ogs import cli

from ..boundary_set import LayerSet


class RegionSet:
    """
    A class representing a set of regions composed of subsets, each identified by MaterialID.

    The RegionSet class represents a collection of regions, where each region is composed of
    subsets. Each subset within a region is uniquely identified by "MaterialID".
    """

    def __init__(self, input: Union[Path, pv.DataSet]):
        if type(input) is Path:
            self.filename = input
            self.mesh = None
        else:
            self.filename = Path(tempfile.mkstemp(".vtu", "region_set")[1])
            self.mesh = input

    def box_boundaries(self):
        """
        Retrieve the boundaries of the mesh in local coordinate system (u, v, w).

        This function extracts the boundaries of the mesh along the u, v, and w directions
        of the local coordinate system. The u-axis corresponds to the x-coordinate, the v-axis
        corresponds to the y-coordinate, and the w-axis corresponds to the z-coordinate.

        Returns:
            tuple: A tuple (u_min, u_max, v_min, v_max, w_min, w_max) representing the boundaries
            of the mesh in the local coordinate system.

        Notes:
            - If the original mesh was created from boundaries, this function returns the original boundaries.
            - The returned boundaries adhere to the definition of [Pyvista Box](https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Box.html).

        Example:
            mesh = ...
            u_min, u_max, v_min, v_max, w_min, w_max = mesh.box_boundaries()
        """
        surface = self.mesh.extract_surface()
        u_max = to_boundary(surface, lambda normals: normals[:, 0] > 0.5)
        u_min = to_boundary(surface, lambda normals: normals[:, 0] < -0.5)
        v_max = to_boundary(surface, lambda normals: normals[:, 1] > 0.5)
        v_min = to_boundary(surface, lambda normals: normals[:, 1] < -0.5)
        w_max = to_boundary(surface, lambda normals: normals[:, 2] > 0.5)
        w_min = to_boundary(surface, lambda normals: normals[:, 2] < -0.5)

        return (u_min, u_max, v_min, v_max, w_min, w_max)


def to_boundary(
    surface_mesh: pv.PolyData,
    filter_condition: Callable[[np.ndarray], np.ndarray],
) -> pv.UnstructuredGrid:
    """
    Extract cells from a surface mesh that meet a filter condition for normals.

    This function takes a surface mesh represented by a `pv.PolyData` object and extracts
    cells that match a specified filter condition based on the normals of the mesh.

    :param surface_mesh:        The input surface mesh.
    :param filter_condition:    A callable filter condition that takes an array
                                of normals as input and returns an array
                                indicating whether the condition is met.

    Returns:
        pv.UnstructuredGrid: A mesh containing only the cells that meet the filter condition.

    Example:
        surface_mesh = ...
        specific_cells = to_boundary(surface_mesh, lambda normals: [n[2] > 0.5 for n in normals])
    """

    surface_mesh = surface_mesh.compute_normals(
        cell_normals=True, point_normals=True
    )

    ids = np.arange(surface_mesh.n_cells)[
        filter_condition(surface_mesh["Normals"])
    ]

    surface_mesh.rename_array("vtkOriginalPointIds", "BULK_NODE_ID")
    surface_mesh.rename_array("vtkOriginalCellIds", "BULK_ELEMENT_ID")
    surface_mesh.cell_data.remove("Normals")
    specific_cells = surface_mesh.extract_cells(ids)
    specific_cells.cell_data.remove("vtkOriginalCellIds")
    specific_cells.point_data.remove("vtkOriginalPointIds")
    return specific_cells


def to_region_prism(layer_set: LayerSet, resolution: float) -> RegionSet:
    """
    Convert a layered geological structure into a RegionSet using prism meshing.

    This function takes a :class:`boundary_set.LayerSet` and converts it into a :class:`region.RegionSet` object using prism or tetrahedral meshing technique.
    The function will use prism elements for meshing if possible; otherwise, it will use
    tetrahedral elements.

    Parameters:
        layer_set (LayerSet): A :class:`boundary_set.LayerSet`.
        resolution (float): The desired resolution in [meter] for meshing. It must greater than 0.

    Returns:
        RegionSet: A :class:`boundary_set.LayerSet` object containing the meshed representation of the geological structure.

    Raises:
        ValueError: If an error occurs during the meshing process.

    Example:
        layer_set = LayerSet(...)
        resolution = 0.1
        region_set = to_region_prism(layer_set, resolution)
    """

    raster_vtu, rastered_layers_txt = layer_set.create_raster(
        resolution=resolution
    )

    outfile = Path(tempfile.mkstemp(".vtu", "region_prism")[1])
    ret = cli.createLayeredMeshFromRasters(
        i=raster_vtu, r=rastered_layers_txt, o=outfile
    )
    if ret:
        raise ValueError

    materials_in_domain: list[int] = sum(
        [
            [layer.material_id] * (layer.num_subdivisions + 1)
            for layer in layer_set.layers
        ],
        [],
    )

    pv_mesh = pv.XMLUnstructuredGridReader(outfile).read()
    intermediate_vtu_ids = list(set(pv_mesh.cell_data["MaterialIDs"]))
    # reversed bc createLayeredMeshFromRasters starts numbering from the bottom
    # up, but we number the layers from top to bottom
    id_mapping = dict(zip(intermediate_vtu_ids, materials_in_domain[::-1]))
    new_ids = [
        id_mapping[old_id] for old_id in pv_mesh.cell_data["MaterialIDs"]
    ]
    pv_mesh.cell_data["MaterialIDs"].setfield(new_ids, np.uint32)

    return RegionSet(input=pv_mesh)


def layer_to_simplified_mesh(
    layer, resolution: float, rank: int, bounds: list[float]
) -> pv.UnstructuredGrid:
    """Convert a geological layer to a simplified mesh.

    This function converts a geological `layer` into a simplified mesh using the specified
    `resolution`, `rank`, and bounding `bounds`.


    :param layer:               The geological layer to be converted to a mesh.
    :param resolution (float):  The desired spatial resolution of the mesh in units of the geological structure.
                                Be cautious with very high resolutions, as they may lead to distorted or incomplete meshes,
                                making them unsuitable for further analysis.
    :param rank (int):          he rank of the mesh (2 for 2D, 3 for 3D). The mesh dimensionality must be consistent
                                with the provided `bounds`.
    :param bounds (list[float]):    A list of bounding values [min_x, max_x, min_y, max_y] for 2D mesh or
                                    [min_x, max_x, min_y, max_y, min_z, max_z] for 3D mesh.
                                    The `bounds` define the region of the geological structure that will be meshed.

    Returns:
        pv.UnstructuredGrid: A simplified unstructured grid mesh representing the layer.

    Raises:
        Exception: If the specified `rank` is not 2 or 3, indicating an invalid mesh dimensionality.

    Example:
        layer = ...
        resolution = 1.5  # Example resolution in geological units
        rank = 2  # Mesh will be 2D
        bounds = [0, 10, 0, 10]  # Bounding box [min_x, max_x, min_y, max_y]
        mesh = layer_to_simplified_mesh(layer, resolution, rank, bounds)
    """

    axis0_range = np.arange(bounds[0], bounds[1], resolution)
    axis1_range = np.arange(bounds[2], bounds[3], resolution)

    heights = np.linspace(
        layer.bottom.mesh.points[:, 2].mean(),
        layer.top.mesh.points[:, 2].mean(),
        num=layer.num_subdivisions + 2,
    )

    if rank == 2:
        AXIS1, AXIS2 = np.meshgrid(axis1_range, heights)
        AXIS0 = 0 * (AXIS1 + AXIS2)
        # resulting mesh is in xy - plane
        X, Y, Z = (AXIS1, AXIS2, AXIS0)
    elif rank == 3:
        X, Y, Z = np.meshgrid(axis0_range, axis1_range, heights)
    else:
        msg = "rank is {rank}, but must be 2 or 3."
        raise Exception(msg)

    grid = pv.StructuredGrid(X, Y, Z)
    material = layer.material_id
    grid.cell_data["MaterialIDs"] = (np.ones(grid.n_cells) * material).astype(
        np.int32
    )
    return grid


def to_region_simplified(
    layer_set: LayerSet, xy_resolution: float, rank: int
) -> RegionSet:
    """
    Convert a layered geological structure to a simplified meshed region.

    This function converts a layered geological structure represented by a `LayerSet` into a
    simplified meshed region using the specified `xy_resolution` and `rank`.

    :param layer_set (LayerSet):    A `LayerSet` object representing the layered geological structure.
    :param xy_resolution (float):   The desired spatial resolution of the mesh in the XY plane.
    :param rank (int):              The rank of the mesh (2 for 2D, 3 for 3D).

    Returns:
        RegionSet: A `RegionSet` object containing the simplified meshed representation of the geological structure.

    Raises:
        AssertionError: If the length of the `bounds` retrieved from the `layer_set` is not 6.

    Example:
        layer_set = LayerSet(...)
        xy_resolution = 0.1  # Example resolution in XY plane
        rank = 2  # Mesh will be 2D
        region_set = to_region_simplified(layer_set, xy_resolution, rank)
    """

    bounds = layer_set.bounds()

    assert len(bounds) == 6
    simple_meshes = [
        layer_to_simplified_mesh(
            layer, resolution=xy_resolution, rank=rank, bounds=bounds
        )
        for layer in layer_set.layers
    ]
    mesh = simple_meshes[0].merge(simple_meshes[1:], merge_points=True)

    return RegionSet(input=mesh)


def to_region_tetraeder(layer_set: LayerSet, resolution: int) -> RegionSet:
    raster_vtu, rastered_layers_txt1 = layer_set.create_raster(
        resolution=resolution
    )

    """
    Convert a layered geological structure to a tetrahedral meshed region.

    This function converts a layered geological structure represented by a `LayerSet`
    into a tetrahedral meshed region using the specified `resolution`.

    Parameters:
        layer_set (LayerSet): A `LayerSet` object representing the layered geological structure.
        resolution (int): The desired resolution for meshing.

    Returns:
        RegionSet: A `RegionSet` object containing the tetrahedral meshed representation of the geological structure.

    Raises:
        ValueError: If an error occurs during the meshing process.

    Notes:
        - The `LayerSet` object contains information about the layers in the geological structure.
        - The `resolution` parameter determines the desired spatial resolution of the mesh.
        - The function utilizes tetrahedral meshing using Tetgen software to create the meshed representation.
        - The resulting mesh is tetrahedral, and material IDs are assigned to mesh cells based on the geological layers.

    Example:
        layer_set = LayerSet(...)
        resolution = 1  # Example resolution for meshing
        region_set = to_region_tetraeder(layer_set, resolution)
    """

    smesh_file = Path(tempfile.mkstemp(".smesh", "region_tetraeder")[1])

    ret_smesh = cli.createTetgenSmeshFromRasters(
        i=raster_vtu, r=rastered_layers_txt1, o=smesh_file
    )
    if ret_smesh:
        raise ValueError
    ret_tetgen = cli.tetgen("-pkABEFN", smesh_file)
    if ret_tetgen:
        print("Tetgen returns: ", ret_tetgen)
        raise ValueError

    outfile = smesh_file.with_suffix(".1.vtk")
    if not outfile.exists():
        path = outfile.parent
        os.listdir(path)

    materials_in_domain: list[int] = sum(
        [
            [layer.material_id] * (layer.num_subdivisions + 1)
            for layer in layer_set.layers
        ],
        [],
    )

    region_attribute_name = "cell_scalars"

    pv_mesh = pv.read(outfile)
    if region_attribute_name:
        pv_mesh.cell_data["MaterialIDs"] = pv_mesh.cell_data[
            region_attribute_name
        ]

    intermediate_vtu_ids = sorted(
        dict.fromkeys(pv_mesh.cell_data["MaterialIDs"])
    )
    # reversed bc createLayeredMeshFromRasters starts numbering from the bottom
    # up, but we number the layers from top to bottom
    id_mapping = dict(zip(intermediate_vtu_ids, materials_in_domain[::-1]))
    new_ids = [
        id_mapping[old_id] for old_id in pv_mesh.cell_data["MaterialIDs"]
    ]
    pv_mesh.cell_data["MaterialIDs"].setfield(new_ids, np.uint32)

    return RegionSet(input=pv_mesh)


def to_region_voxel(layer_set: LayerSet, resolution: list) -> RegionSet:
    """
    Convert a layered geological structure to a voxelized mesh.

    This function converts a layered geological structure represented by a `LayerSet`
    into a voxelized mesh using the specified `resolution`.

    Parameters:
        layer_set (LayerSet): A `LayerSet` object representing the layered geological structure.
        resolution (list): A list of [x_resolution, y_resolution, z_resolution] for voxelization.

    Returns:
        Mesh: A `Mesh` object containing the voxelized mesh representation of the geological structure.

    Raises:
        ValueError: If an error occurs during the voxelization process.

    Example:
        layer_set = LayerSet(...)
        resolution = [0.1, 0.1, 0.1]  # Example voxelization resolutions in x, y, and z dimensions
        voxel_mesh = to_region_voxel(layer_set, resolution)
    """

    layers_txt = Path(tempfile.mkstemp(".txt", "layers")[1])
    layer_filenames = layer_set.filenames()
    with layers_txt.open("w") as file:
        file.write("\n".join(str(filename) for filename in layer_filenames))
    outfile = Path(tempfile.mkstemp(".vtu", "region_voxel")[1])
    ret = cli.Layers2Grid(
        i=layers_txt,
        o=outfile,
        x=resolution[0],
        y=resolution[1],
        z=resolution[2],
    )
    if ret:
        raise ValueError()

    materials_in_domain: list[int] = sum(
        [[layer.material_id] for layer in layer_set.layers],
        [],
    )

    region_attribute_name = "MaterialIDs"
    pv_mesh = pv.read(outfile)
    if region_attribute_name not in pv_mesh.cell_data:
        pv_mesh.cell_data[region_attribute_name] = pv_mesh.cell_data[
            region_attribute_name
        ]

    intermediate_vtu_ids = sorted(
        dict.fromkeys(pv_mesh.cell_data["MaterialIDs"])
    )
    # reversed bc createLayeredMeshFromRasters starts numbering from the bottom
    # up, but we number the layers from top to bottom
    k = zip(intermediate_vtu_ids, materials_in_domain[::-1])
    id_mapping = dict(k)
    new_ids = [
        id_mapping[old_id] for old_id in pv_mesh.cell_data["MaterialIDs"]
    ]
    pv_mesh.cell_data["MaterialIDs"].setfield(new_ids, np.uint32)

    return RegionSet(input=pv_mesh)
