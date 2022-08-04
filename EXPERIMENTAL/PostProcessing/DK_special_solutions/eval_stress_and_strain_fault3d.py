#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 08:58:34 2022

@author: dominik

   Currently evaluate friction law F_t < F_n tan(p) + c on faults (point_data: faultID)
   Faults are volumes (3D) this makes permeability changes easy

   Alternatively use Paraview filter "Resample with Dataset"

   INPUT (hard coded): file names and unit normal vector (fault plane, matType=1)

   TODO
      selection via point_data: faultID!
      find most critical fault state in simulated history and eval k then
"""

import numpy as np
import meshio
import pyvista as pv

###   ENTER HERE FILE NAMES AND MESH PARAMETERS   ###
# SD/linear
#cell_type = 'triangle'
#mixed_mesh_filename = 'mini_domain.vtu'
#results_mesh_filename =  'resultsSD_ts_1_t_1.000000.vtu'  # volume-only

# HM/quadratic
cell_data_key = 'MaterialIDs'
cell_data_value = 1     # 0=rock, 1=fault, 2=dgr
cell_type = 'tetra10'    # 'triangle6'
mixed_mesh_filename   = 'ys12_normal_domain.vtu'   # volume and possibly lower dimensional cells
initial_mesh_filename = 'results_equilibrium_ts_10_t_315360000000.000000.vtu'  # volume-only
results_mesh_filename = 'results_loadcase_ts_100_t_3153600000000.000000.vtu'  # volume-only
updated_mesh_filename = 'ys12_faulted_domain.vtu'

###   ENTER HERE NORMAL VECTOR AND MOHR-COULOMB PARAMETERS   ###
fault_normal_vector = np.array([ 0.9995065603657315, 0.0, 0.03141075907812829 ])   # unit vector
friction_angle = 0.3*np.pi
cohesion = 10000
###   INPUT END   ###


''' --- obtain geometry description of fault plane --- '''
mixed_mesh = meshio.read(mixed_mesh_filename)

selection_index = mixed_mesh.cell_data_dict[cell_data_key][cell_type] == cell_data_value
selection_cells_values = mixed_mesh.cells_dict[cell_type][selection_index]
unique_fault_nodes = np.unique(selection_cells_values)  # a cell is a set of node numbers

nx, ny, nz = fault_normal_vector
nxn = np.outer(fault_normal_vector, fault_normal_vector)
Vnxn = np.array([nxn[0,0], nxn[1,1], nxn[2,2], 2*nxn[0,1], 2*nxn[1,2], 2*nxn[0,2]])  # xx, yy, zz, xy, yz, xz
Mn = np.array([[nx,0,0,ny,0,nz], [0,ny,0,nx,nz,0], [0,0,nz,0,ny,nx]])   # to calculate traction vector on fault


''' --- evaluate results on fault geometry --- '''
volume_mesh = meshio.read(results_mesh_filename)   # SD
number_of_nodes = len(volume_mesh.points)

fault_sigma = volume_mesh.point_data['sigma'][unique_fault_nodes]
fault_sigma_n = np.inner(fault_sigma, Vnxn)
fault_sigma_Vt = np.inner(fault_sigma, Mn) - np.outer(fault_sigma_n, fault_normal_vector)  # traction vector
fault_abs_sigma_t = np.linalg.norm(fault_sigma_Vt, axis=1)
max_abs_sigma_t = cohesion - np.tan(friction_angle) * fault_sigma_n   # compressive stress has negative sign

# point data
pd_cfs = np.zeros(number_of_nodes)
pd_sigma_n = np.zeros(number_of_nodes)
pd_abs_sigma_t = np.zeros(number_of_nodes)

pd_cfs[unique_fault_nodes] = np.abs(fault_abs_sigma_t)/max_abs_sigma_t
pd_sigma_n[unique_fault_nodes] = fault_sigma_n
pd_abs_sigma_t[unique_fault_nodes] = fault_abs_sigma_t

mixed_mesh.point_data['cfs'] = pd_cfs
mixed_mesh.point_data['sigma_n'] = pd_sigma_n
mixed_mesh.point_data['abs_sigma_t'] = pd_abs_sigma_t

fault_epsilon = volume_mesh.point_data['epsilon'][unique_fault_nodes]
fault_epsilon_n = np.inner(fault_epsilon, Vnxn)


''' --- extract submesh of fault for plot --- '''
subdomain_cells = []		# list
#subdomain_cell_data = {}	# dict

selection_cells_block = (cell_type, selection_cells_values)
subdomain_cells.append(selection_cells_block)

#selection_cell_data_values = mesh.cell_data_dict[CELL_DATA_KEY][cell_type][selection_index]
#subdomain_cell_data[CELL_DATA_KEY].append(selection_cell_data_values)

submesh = meshio.Mesh(points=mixed_mesh.points, point_data=mixed_mesh.point_data, cells=subdomain_cells)
submesh.write('results_on_fault.vtu')  # conversion to pv via write-read, probably there is a better way

pv_mesh = pv.read(results_mesh_filename) # domain
pv_submesh = pv.read('results_on_fault.vtu') # fault
# pl = pv.Plotter()
# pv.set_plot_theme("document")
# pl.add_mesh(pv_mesh, show_edges=True, color="mintcream", opacity=0.2)
# pl.add_mesh(pv_submesh, scalars=pv_submesh['cfs'], scalar_bar_args={'title': "CFS"})   #
# pl.show(screenshot = "cfs.png")

print(" fault_sigma_n  min = {}   max = {}".format(min(fault_sigma_n), max(fault_sigma_n)))
print("|fault_sigma_t| min = {}   max = {}".format(min(fault_abs_sigma_t), max(fault_abs_sigma_t)))

print(" fault_epsilon_n  min = {}   max = {}   mean = {}".format(min(fault_epsilon_n), max(fault_epsilon_n), np.mean(fault_epsilon_n)))


''' --- update permeability (field name: vm_permeability) --- '''
initial_mesh = meshio.read(initial_mesh_filename)
fault_epsilon0 = volume_mesh.point_data['epsilon'][unique_fault_nodes]
fault_epsilon0_n = np.inner(fault_epsilon, Vnxn)

pd_eps_n = np.zeros(number_of_nodes)
pd_eps0_n =  np.zeros(number_of_nodes)

pd_eps_n[unique_fault_nodes] = fault_epsilon_n
pd_eps0_n[unique_fault_nodes] = fault_epsilon0_n

scd_eps_n = np.zeros(len(selection_cells_values))   # Selected Cell Data
scd_eps0_n = np.zeros(len(selection_cells_values))   # Selected Cell Data
for cell_index, cell_values in enumerate(selection_cells_values):
    scd_eps_n[cell_index]  = np.mean(pd_eps_n[cell_values])
    scd_eps0_n[cell_index] = np.mean(pd_eps0_n[cell_values])

k_old = volume_mesh.cell_data_dict['vm_permeability'][cell_type][selection_index]
a = 1e-5
b = np.sqrt(12*k_old) + a*np.heaviside(scd_eps_n - scd_eps0_n, 0)
k_new = 1e2*k_old #+ (b/a)*((b**2)/12 - k_old) + 1

new_cell_data = initial_mesh.cell_data
cell_block = 0
new_cell_data['vm_permeability'][cell_block][selection_index] = k_new

updated_mesh = meshio.Mesh( points=initial_mesh.points, point_data=initial_mesh.point_data, cells=initial_mesh.cells, cell_data=new_cell_data)
updated_mesh.write(updated_mesh_filename)
