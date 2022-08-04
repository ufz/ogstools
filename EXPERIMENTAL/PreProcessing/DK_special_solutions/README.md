``appendPointData.py``
``appendCellData.py``
These scripts are for the specific case to assign position-dependent permeabilites (defined by ``w_c_reader.py``) to the elements.
Generally they maybe used to assign node and element data from other data sources (programs, databases).

``estimate_initial_u_and_initial_p.py``
estimates initial pressure and initial displacements due to gravitation for HM-simulations assuming linear, isotropic elasticity and fixed boundaries except free boundary on top.


**Further preprocessing scripts are part of the repositories**

``joergbuchwald/VTUinterface/tools/spatial_transformation/``
applying a spatial coordinate transformation to a mesh (vtu)

``joergbuchwald/ogs6py/examples/compose_prj_from_csv/``
composing the media section in an input file (prj) from a data table (csv)

``joergbuchwald/ogs6py/examples/optimize_coupling_scheme_parameter``
finding the optimal coupling parameter (algorithmic parameter) for HM-simulations with the staggered scheme
