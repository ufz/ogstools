# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""
.. literalinclude:: ../../ogstools/plot/plot_setup_defaults.py
   :language: python
   :linenos:
   :lines: 9-

"""

setup_dict = {
    "default_cmap": "RdBu_r",
    "custom_cmap": None,
    "dpi": 120,
    "min_ax_aspect": 0.5,
    "max_ax_aspect": 2.0,
    "invert_colorbar": False,
    "layout": "compressed",
    "material_names": None,
    "num_levels": 11,
    "num_streamline_interp_pts": 50,
    "vmin": None,
    "vmax": None,
    "scale_type": "auto",
    "show_element_edges": "MaterialIDs",
    "show_region_bounds": True,
    "log_scaled": False,
    "combined_colorbar": False,
    "tick_pad": 14,
    "tick_length": 14,
    "fontsize": 32,
    "linewidth": 1,
    "label_split": 37,
    "spatial_unit": "m",
    "time_unit": "s",
}
