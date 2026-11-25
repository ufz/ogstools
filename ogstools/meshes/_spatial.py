# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import gmsh
import numpy as np
import pandas as pd
import shapely

from ogstools.definitions import temp_file

if TYPE_CHECKING:
    from ogstools.meshes._meshes import Meshes


def _gmsh_add_lines(
    lines: shapely.LineString | shapely.LinearRing,
    coord_tags: dict[tuple[float, float], int],
    line_ids: dict[tuple[int, int], int],
) -> tuple[list[int], list[int]]:
    """Add the segments of a line to the gmsh model, reusing existing segments.

    gmsh must already be initialised. This function does not finalise it.

    :param lines:       The line whose segments are added.
    :param coord_tags:  Mapping from point coordinates to gmsh point tags.
    :param line_ids:    Mapping from pairs of point tags to gmsh line tags.
                        It records the lines already added, so that no line is
                        added twice. New lines are added to it.

    :returns:   The gmsh line tags of all segments of the line, and the gmsh
                line tags of the newly created segments.
                A segment that reuses an existing line in reverse direction
                has a negative tag in the first list.
    """
    result_ids = []
    new_ids = []
    for start, end in zip(
        lines.coords[:-1],
        lines.coords[1:],
        strict=True,
    ):
        if (coord_tags[start], coord_tags[end]) in line_ids:
            result_ids.append(line_ids[(coord_tags[start], coord_tags[end])])
        elif (coord_tags[end], coord_tags[start]) in line_ids:
            result_ids.append(-line_ids[(coord_tags[end], coord_tags[start])])
        else:
            line_id = gmsh.model.geo.add_line(
                coord_tags[start], coord_tags[end]
            )
            result_ids.append(line_id)
            new_ids.append(line_id)
            line_ids[(coord_tags[start], coord_tags[end])] = line_id
    return result_ids, new_ids


def _collect_3D_coords(
    layer_dict: dict[str, gpd.GeoDataFrame], height_data: np.ndarray | None
) -> np.ndarray:
    coords_3d = pd.concat(
        [
            layer_dict[name][layer_dict[name].geometry.has_z].get_coordinates(
                include_z=True
            )
            for name in layer_dict
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["x", "y", "z"])
    return (
        np.vstack((coords_3d[["x", "y", "z"]].to_numpy(), height_data))
        if height_data is not None
        else coords_3d[["x", "y", "z"]].to_numpy()
    )


def _preprocess_geometry(
    layer_dict: dict[str, gpd.GeoDataFrame],
    cellsize: float | None = None,
    domain: str | None = None,
) -> tuple[dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
    project_crs = next(iter(layer_dict.values())).crs
    geom_dict: dict[str, gpd.GeoDataFrame] = {}
    for geom_type in ["Polygon", "LineString", "Point"]:
        geom_dict[geom_type] = gpd.GeoDataFrame(
            pd.concat(
                [
                    layer[layer.geom_type == geom_type]
                    for _, layer in layer_dict.items()
                ],
                ignore_index=True,
            ),
            crs=project_crs,
        )
    if domain is not None:
        geom_dict["Polygon"] = gpd.clip(
            geom_dict["Polygon"], layer_dict[domain], keep_geom_type=True
        ).explode()
    geom_dict["LineString"] = gpd.clip(
        geom_dict["LineString"], geom_dict["Polygon"], keep_geom_type=True
    ).explode()
    geom_dict["Point"] = gpd.clip(
        geom_dict["Point"], geom_dict["Polygon"], keep_geom_type=True
    )
    if cellsize is None:
        cellsize = np.sqrt(geom_dict["Polygon"].union_all().area) / 100
    for _, frame in geom_dict.items():
        if "cellsize" in frame.columns:
            frame["cellsize"] = frame["cellsize"].fillna(cellsize)
        else:
            frame["cellsize"] = cellsize
    boundaries = gpd.GeoDataFrame(
        geometry=[
            shapely.union_all(
                [
                    polygon.boundary
                    for polygon in geom_dict["Polygon"]["geometry"]
                ]
            )
        ],
        crs=project_crs,
    )
    dissolved_polys = gpd.GeoDataFrame(
        geometry=shapely.get_parts(
            shapely.polygonize([shapely.union_all(boundaries.geometry)])
        ),
        crs=project_crs,
    )
    dissolved_polys["rep_point"] = (
        dissolved_polys.geometry.representative_point()
    )
    dissolved_polys.set_geometry("rep_point", inplace=True)
    poly_sj = gpd.GeoDataFrame(
        gpd.sjoin(
            left_df=dissolved_polys,
            right_df=geom_dict["Polygon"],
            how="left",
            predicate="within",
        )
        .dropna(subset=["original_layer"])
        .groupby(level=0)
        .agg({"geometry": "first", "original_layer": list, "cellsize": "min"})
        .reset_index(drop=True),
        geometry="geometry",
        crs=project_crs,
    )
    return geom_dict, poly_sj


def _mesh_with_gmsh(
    geom_dict: dict[str, gpd.GeoDataFrame],
    poly_sj: gpd.GeoDataFrame,
    cellsize_func: str | Callable | None = None,
    domain: str | None = None,
    log: bool = False,
) -> Path:
    gmsh.initialize()
    if isinstance(cellsize_func, str):
        math_field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(math_field, "F", cellsize_func)
        gmsh.model.mesh.field.setAsBackgroundMesh(math_field)
    elif callable(cellsize_func):
        gmsh.model.mesh.setSizeCallback(cellsize_func)
    if not log:
        gmsh.option.setNumber("General.Terminal", 0)
    coords = (
        pd.concat(
            [
                poly_sj.get_coordinates(include_z=False, index_parts=True)
                .reset_index()
                .merge(
                    poly_sj["cellsize"], left_on="level_0", right_index=True
                ),
                geom_dict["LineString"]
                .get_coordinates(include_z=False, index_parts=True)
                .reset_index()
                .merge(
                    geom_dict["LineString"]["cellsize"],
                    left_on="level_0",
                    right_index=True,
                ),
                geom_dict["Point"]
                .get_coordinates(include_z=False, index_parts=True)
                .reset_index()
                .merge(
                    geom_dict["Point"]["cellsize"],
                    left_on="level_0",
                    right_index=True,
                ),
            ],
            ignore_index=True,
        )
        .sort_values("cellsize", ascending=True)
        .drop_duplicates(subset=["x", "y"], keep="first")
    )
    coord_tags: dict[tuple[float, float], int] = {}
    for point in coords.itertuples():
        coord_tags[(point.x, point.y)] = gmsh.model.geo.add_point(
            point.x, point.y, 0.0, point.cellsize
        )
    line_ids: dict[tuple[int, int], int] = {}
    poly_ids: dict[int, int] = {}
    poly_groups: dict[str, list[int]] = {
        name: [] for name in geom_dict["Polygon"]["original_layer"].unique()
    }
    for index, row in poly_sj.iterrows():
        outer_boundary_ids, _ = _gmsh_add_lines(
            row["geometry"].exterior, coord_tags, line_ids
        )
        outer_id = gmsh.model.geo.addCurveLoop(outer_boundary_ids)
        inner_ids = []
        for ring in row["geometry"].interiors:
            ring_ids, _ = _gmsh_add_lines(ring, coord_tags, line_ids)
            inner_ids.append(gmsh.model.geo.addCurveLoop(ring_ids))
        result = gmsh.model.geo.addPlaneSurface([outer_id, *inner_ids])
        poly_ids[index] = result
        for name in row["original_layer"]:  # phys_group
            poly_groups[name].append(result)
    # Since we can only embed in polygon interiors we need to keep track
    # which lines actually can/need to be embedded and which polygon they
    # lie in
    line_embed: dict[int, list[int]] = {name: [] for name in poly_sj.index}
    lines_groups: dict[str, list[int]] = {
        name: [] for name in geom_dict["LineString"]["original_layer"].unique()
    }
    point_groups = {
        name: points.get_coordinates(include_z=False)
        .apply(
            lambda row: coord_tags[(row["x"], row["y"])],
            axis=1,
        )
        .tolist()
        for name, points in geom_dict["Point"].groupby("original_layer")
    }
    point_sj = gpd.sjoin(
        geom_dict["Point"],
        poly_sj,
        how="left",
        predicate="within",
    )
    point_embed = dict(
        zip(
            poly_sj.index,
            [
                [
                    coord_tags[(row.geometry.x, row.geometry.y)]
                    for _, row in point_sj[
                        point_sj["index_right"] == i
                    ].iterrows()
                ]
                for i in poly_sj.index
            ],
            strict=True,
        )
    )
    line_sj = gpd.sjoin(
        geom_dict["LineString"],
        poly_sj,
        how="left",
        predicate="within",
    ).reset_index()
    for _, line in line_sj.iterrows():
        line_segments, new_segments = _gmsh_add_lines(
            line["geometry"],
            coord_tags=coord_tags,
            line_ids=line_ids,
        )
        if new_segments:
            line_embed[line["index_right"]] += new_segments
        lines_groups[line["original_layer_left"]] += [
            abs(line) for line in line_segments
        ]
    gmsh.model.geo.synchronize()
    for poly, line_tags in line_embed.items():
        gmsh.model.mesh.embed(1, line_tags, 2, poly_ids[poly])
    for name, line_tags in lines_groups.items():
        gmsh.model.addPhysicalGroup(1, line_tags, name=name)
    for poly, point_tags in point_embed.items():
        gmsh.model.mesh.embed(0, point_tags, 2, poly_ids[poly])
    for name, point_tags in point_groups.items():
        gmsh.model.addPhysicalGroup(0, point_tags, name=name)
    for name, poly_tags in poly_groups.items():
        gmsh.model.addPhysicalGroup(2, poly_tags, name=name)
    gmsh.model.mesh.generate(2)
    msh_file = temp_file(
        ".msh", "spatial_gen_mesh", "" if domain is None else domain
    )
    gmsh.write(str(msh_file))
    gmsh.finalize()
    return msh_file


def _read_spatial_data(
    spatial_data: list[
        Path | str | gpd.GeoDataFrame | tuple[str, gpd.GeoDataFrame]
    ],
) -> dict[str, gpd.GeoDataFrame]:
    layer_dict: dict[str, gpd.GeoDataFrame] = {}
    gdf_number = 1
    for file in spatial_data:
        if isinstance(file, gpd.GeoDataFrame):
            layer_dict[f"frame_{gdf_number}"] = file.copy().explode()
            gdf_number += 1
        elif isinstance(file, tuple):
            layer_dict[file[0]] = file[1].copy().explode()
        else:
            filepath = Path(file)
            if filepath.suffix not in [".gpkg", ".shp", ".geojson"]:
                msg = f"Only .gpkg, .shp and .geojson are currently supported, not {filepath.suffix}. To mesh other file types read into geodataframe first."
                raise ValueError(msg)
            for layer in gpd.list_layers(filepath)["name"]:
                name = (
                    f"{filepath.stem}_{layer}"
                    if filepath.suffix == ".gpkg"
                    else filepath.stem
                )
                layer_dict[name] = gpd.read_file(
                    filepath, layer=layer
                ).explode()
    return layer_dict


def _interp_height_data(meshes: Meshes, height_data: np.ndarray) -> None:
    from scipy.interpolate import LinearNDInterpolator

    height_interp = LinearNDInterpolator(
        height_data[:, :2], height_data[:, 2], np.nan
    )
    for name in meshes:
        meshes[name].points[:, 2] = height_interp(meshes[name].points[:, :2])
