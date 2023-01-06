"""
Extracts some boundary values from a result file (domain mesh) and adds them as
point data to a boundary mesh (subdomain mesh).
This is frequent task for simulation sequences with OpenGeoSys.
"""

import meshio

# TODO: Replace hardcoded path with cli args
domain_mesh = meshio.read("result.vtu")
subdomain_mesh = meshio.read("boundary.vtu")

# look up in the subdomain mesh what the node numbers in the domain mesh are
# ("bulk_node_ids" is OGS nomenclature)
selection = subdomain_mesh.point_data["bulk_node_ids"]

# index: displacement, selected nodes, all of them, x-component
displacement_x_values = domain_mesh.point_data["displacement"][selection][:, 0]
# add selected data to subdomain mesh (point data title "displacement_x" is
# arbitrary)
subdomain_mesh.point_data["displacement_x"] = displacement_x_values

subdomain_mesh.write("boundary_with_added_data.vtu")
