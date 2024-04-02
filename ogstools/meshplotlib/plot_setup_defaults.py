# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""
.. literalinclude:: ../../ogstools/meshplotlib/plot_setup_defaults.py
   :language: python
   :linenos:
   :lines: 9-

"""

setup_dict = {
    "default_cmap": "RdBu_r",
    "dpi": 120,
    "fig_scale": 1.0,
    "min_ax_aspect": 0.5,
    "max_ax_aspect": 2.0,
    "invert_colorbar": False,
    "layout": "compressed",
    "length": ["m", "m"],
    "material_names": None,
    "num_levels": 11,
    "num_streamline_interp_pts": 50,
    "p_min": None,
    "p_max": None,
    "scale_type": "auto",
    "show_element_edges": "MaterialIDs",
    "show_region_bounds": True,
    "embedded_region_names_color": None,
    "title_center": "",
    "title_left": "",
    "title_right": "",
    "x_label": None,
    "y_label": None,
    "log_scaled": False,
    "combined_colorbar": True,
    "rcParams": {
        "font.weight": "normal",
        "font.family": "sans-serif",
        "font.size": 32,
        "axes.titlesize": "medium",
        "axes.labelsize": "medium",
        "axes.labelpad": 12,
        "axes.grid": False,
        "legend.fontsize": "medium",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "axes.linewidth": 1,
        "lines.linewidth": 1,
        "legend.framealpha": 1,
        "mathtext.fontset": "dejavuserif",
        "xtick.major.pad": 12,
        "xtick.major.size": 12,
        "xtick.direction": "in",
        "ytick.major.pad": 12,
        "ytick.major.size": 12,
        "ytick.direction": "in",
    },
}
