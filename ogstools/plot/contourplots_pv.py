# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""3D Plotting functions."""

from typing import Any

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

from ogstools.plot import setup, utils
from ogstools.plot.levels import combined_levels
from ogstools.variables import Variable


def contourf_pv(
    mesh: pv.UnstructuredGrid,
    variable: str | Variable,
    show_edges: bool = True,
    opacities: dict[int, float] | None = None,
    lighting: bool = False,
    categoric: bool | None = None,
    **kwargs: Any,
) -> pv.Plotter:
    """General 3D plot using pyvista

    :param mesh:        The mesh to be plotted with pyvista.
    :param variable:    The variable which should be shown.
    :param show_edges:  If True, draw edges as black lines
    :param opacities:   Dictionary, mapping opacities to material ids.
                        Default None (all opaque),
                        Example: {1:0.0 # transparent, 2:1 # opaque, 3:0.5}
                        All not provided ids are rendered as opaque.
    :param lighting:    If True, use lighting in the visualization.
    :param categoric:   If True, use a categoric colormap. By default it uses
                        the variable to determine if this should be True.

    :returns:           A pyvista Plotter object. Use .show() to display the
                        scene.
    """
    plot_mesh: pv.UnstructuredGrid = mesh.copy()
    variable_ = Variable.find(variable, mesh)

    # converting to float as the interactive plot seems to have trouble with int
    # taking magnitude of pint quantities to prevent numpy warning
    plot_var = variable_.replace(func=lambda x: variable_.func(x).astype(float))

    levels = combined_levels(np.array([mesh]), plot_var)
    if categoric is None:
        categoric = plot_var.categoric
    pv_levels = (
        np.arange(len(levels))
        if categoric
        else np.linspace(levels[0], levels[-1], 255)
    )
    pv_levels = np.asarray(kwargs.get("levels", pv_levels))
    cmap = utils.get_cmap_norm(pv_levels, plot_var)[0]

    plotter = pv.Plotter(
        off_screen=True, window_size=(720, 400), border=False, image_scale=2
    )

    plot_mesh[plot_var.output_name] = plot_var.transform(mesh)
    if plot_var.mask_used(plot_mesh):
        plot_mesh = plot_mesh.ctp(True).threshold(
            value=[1, 1], scalars=variable_.mask
        )

    if opacities is None:
        plotter.add_mesh(
            plot_mesh, scalars=plot_var.output_name, show_scalar_bar=False,
            show_edges=show_edges, cmap=cmap, culling=True,
            categories=categoric, lighting=lighting, **kwargs
        )  # fmt: skip
    else:
        # Plot each region with its individual opacity
        for idx, matID in enumerate(np.unique(plot_mesh[plot_var.output_name])):
            region = plot_mesh.threshold([matID, matID], plot_var.output_name)
            region[plot_var.output_name] = plot_var.transform(region)
            opacity_kwarg = (
                {"opacity": opacities[matID]} if matID in opacities else {}
            )
            plotter.add_mesh(
                region, scalars=plot_var.output_name, show_scalar_bar=False,
                show_edges=show_edges and matID not in opacities, culling=True,
                lighting=lighting, cmap=ListedColormap([cmap(int(idx))]),
                categories=categoric, clim=pv_levels[[0, -1]],
                **opacity_kwarg, **kwargs
            )  # fmt:skip

    if categoric:
        # pyvista needs strings as data for proper rendering of categoric data
        # unfortunately the categoric values are then sorted alphanumerically,
        # thus we need to add leading zeros. As this is can be performance heavy
        # we do it for the smallest viable subset.
        categoric_var = plot_var.replace(
            func=lambda x: np.char.zfill(
                plot_var.func(getattr(x, "magnitude", x))
                .astype(int).astype(str), 2
            )
        )  # fmt: skip
        plot_mesh = plot_mesh.extract_cells(
            np.unique(plot_mesh[plot_var.data_name], return_index=True)[1]
        )
        plot_mesh[plot_var.output_name] = categoric_var.transform(plot_mesh)

    # Latex in vtk renderer is broken for now, thus removing special characters
    # see https://github.com/pyvista/pyvista/discussions/2928
    label = plot_var.get_label()
    for s in ["$", "\\", "{", "}"]:
        label = label.replace(s, "")
    scalar_bar_args = dict(  # noqa: C408
        vertical=True, position_x=0.8, position_y=0.05, height=0.9, title=label
    )
    # optionally adding regions names
    if plot_var.data_name == "MaterialIDs" and setup.material_names is not None:
        plot_mesh["MaterialIDs"] = [
            ":".join([matID, setup.material_names.get(int(matID), "")])
            for matID in plot_mesh["MaterialIDs"]
        ]

    # Finally, adding an invisible mesh solely for the scalarbar.
    plotter.add_mesh(
        plot_mesh, scalars=plot_var.output_name, show_edges=show_edges,
        cmap=cmap, culling=True, categories=categoric, lighting=lighting,
        scalar_bar_args=scalar_bar_args, opacity=0.0, **kwargs
    )  # fmt: skip

    plotter.show_axes()
    return plotter
