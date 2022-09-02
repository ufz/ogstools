#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 08:58:34 2022

@author: dominik

   Evaluate friction law F_t < F_n tan(p) + c on given faults (point_data: faultIDs)
   Faults are lower-dimensional (2D) so no thickness is required and it is easier to mesh than thin layers

   Alternatively use Paraview filter "Resample with Dataset"

   INPUT (hard coded): file names and unit normal vector (fault plane, matType=1)

   TODO
      delta_rCFS = loaded_rCFS - initial_rCFS
      restructure: run absolute CFS stress in function and then difference and relative

      pass fault (MatID, normal vector, filename) as structure/class
      def __init__(self, faultMatID, normal_vector, name)

      find most critical fault state in simulated history and eval permeability then
      how to update permeabilities consequently?  --> ys12_faulted_domain.vtu
"""

import numpy as np
import meshio
import pyvista as pv
import os
import matplotlib.pyplot as plt

###   ENTER HERE FILE NAMES AND MESH PARAMETERS   ###

initial_mesh_filename = 'results_equilibrium_ts_10_t_315360000000.000000.vtu'  # volume-only
#results_mesh_filename = 'results_loadcase_ts_100_t_3153600000000.000000.vtu'  # volume-only
#TODO read into list

###   ENTER HERE FAULT IDs AND NORMAL VECTORs   ###
FaultMatID_A = 2
FaultMatID_B = 3
fault_normal_vector_A = np.array([ 0.9510565162951536, 0.0, 0.30901699437494745 ])   # fault A (unit vector)
fault_normal_vector_B = np.array([ 0.9505872273155842, -0.3088645131422603, 0.03141075907812835 ])   # fault B (unit vector)

### AND MOHR-COULOMB PARAMETERS   ###
friction_angle = (30/180)*np.pi
cohesion = 10000

###   INPUT END   ###


def cfs(mesh_filename, faultID, fault_normal_vector, show_results=False, screenshot_filename="cfs.png"):
    '''Coulomb Failure Stress'''

    ''' --- obtain geometry description of fault plane --- '''
    nx, ny, nz = fault_normal_vector
    nxn = np.outer(fault_normal_vector, fault_normal_vector)
    Vnxn = np.array([nxn[0,0], nxn[1,1], nxn[2,2], 2*nxn[0,1], 2*nxn[1,2], 2*nxn[0,2]])  # xx, yy, zz, xy, yz, xz
    Mn = np.array([[nx,0,0,ny,0,nz], [0,ny,0,nx,nz,0], [0,0,nz,0,ny,nx]])   # to calculate traction vector on fault

    ''' --- evaluate results on fault geometry --- '''
    mesh = meshio.read(mesh_filename)
    fault_points_index = mesh.point_data["faultIDs"] == faultID

    fault_sigma = mesh.point_data['sigma'][fault_points_index]
    fault_sigma_n = np.inner(fault_sigma, Vnxn)
    fault_sigma_Vt = np.inner(fault_sigma, Mn) - np.outer(fault_sigma_n, fault_normal_vector)  # traction vector
    fault_abs_sigma_t = np.linalg.norm(fault_sigma_Vt, axis=1)
    max_abs_sigma_t = cohesion - np.tan(friction_angle) * fault_sigma_n   # compressive stress has negative sign
    fault_cfs = 1 - np.abs(fault_abs_sigma_t)/max_abs_sigma_t

    if show_results:
        ''' --- add results as point data --- '''
        number_of_nodes = len(mesh.points)
        pd_cfs = np.zeros(number_of_nodes)
        pd_sigma_n = np.zeros(number_of_nodes)
        pd_abs_sigma_t = np.zeros(number_of_nodes)

        pd_cfs[fault_points_index] = fault_cfs
        pd_sigma_n[fault_points_index] = fault_sigma_n
        pd_abs_sigma_t[fault_points_index] = fault_abs_sigma_t

        mesh.point_data['cfs'] = pd_cfs
        mesh.point_data['sigma_n'] = pd_sigma_n
        mesh.point_data['abs_sigma_t'] = pd_abs_sigma_t

        fault_epsilon = mesh.point_data['epsilon'][fault_points_index]
        fault_epsilon_n = np.inner(fault_epsilon, Vnxn)

        ''' --- extract submesh of fault for plot --- '''
        point_cloud = pv.PolyData(mesh.points[fault_points_index])
        point_cloud["cfs"] = fault_cfs
        fault_mesh = point_cloud.delaunay_2d()     # reconstruct_surface() reduces points !?
        fault_mesh.point_data["cfs"] = fault_cfs

        pv_mesh = pv.read(mesh_filename) # domain
        pl = pv.Plotter()
        pv.set_plot_theme("document")
        pl.add_mesh(pv_mesh, show_edges=True, color="mintcream", opacity=0.2)
        pl.add_mesh(fault_mesh, scalars=fault_mesh['cfs'], scalar_bar_args={'title': "$x$ Delta CFS", 'vertical':True, 'position_x':0.85, 'position_y':0.05}, cmap='jet', clim=[-1, 1])   # , show_scalar_bar=False
        pl.show(screenshot = screenshot_filename)

        print(" fault_sigma_n  min = {}   max = {}".format(min(fault_sigma_n), max(fault_sigma_n)))
        print("|fault_sigma_t| min = {}   max = {}".format(min(fault_abs_sigma_t), max(fault_abs_sigma_t)))
        print(" CFS  min = {}   max = {}".format(min(fault_cfs), max(fault_cfs)))
        print(" fault_epsilon_n  min = {}   max = {}   mean = {}".format(min(fault_epsilon_n), max(fault_epsilon_n), np.mean(fault_epsilon_n)))
    return min(fault_cfs)


# results_mesh_filenames = os.popen("ls results_loadcase_*.vtu").readlines()
# print("find temporal minimum of spatial CFS minimum")
# minmin_cfs = 1
# for results_mesh_filename in results_mesh_filenames:
#     min_cfs = cfs(initial_mesh_filename, FaultMatID_A, fault_normal_vector_A, show_results=False, screenshot_filename="cfsA.png")
#     print(min_cfs)
#     if min_cfs < minmin_cfs:
#         minmin_cfs = min_cfs
#         mincfs_results_mesh_filename = results_mesh_filename
#         print(results_mesh_filename)

plt.rcParams['text.usetex'] = True
min_cfs = cfs(initial_mesh_filename, FaultMatID_A, fault_normal_vector_A, show_results=True, screenshot_filename="cfsA.png")

#min_cfs = cfs(initial_mesh_filename, FaultMatID_B, fault_normal_vector_B, show_results=True, screenshot_filename="cfsB.png")
