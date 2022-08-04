#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:08:53 2021

Calculates the initial displacement and pressure of a fully saturated,
homogenous, isotropic elastic cube in a gravitational field
with incompressible grains, and incompressible fluid (alpha=1).
BC-H: pressure on top (atmospheric), no flux elsewhere
BC-M: stress on top (atmospheric), roller elsewhere

@author: dominik kern
"""

# PARAMETERS (average values, constant)
mu = 1.0e-3       # fluid viscosity
k =  1.2e-15       # permeability
E =  7.9e10        # Youngs modulus (bulk)
nu = 0.27       # Poisson ratio (bulk)
a =  1000.0      # characteristic length (to estimate time scale)
rhof = 1000.0   # fluid density
rhos = 2000.0   # solid density
phi = 0.0015     # porosity
g =  9.81        # gravitational acceleration (abs) assumed in negative z-dir.
pa = 1.0e5        # atmospheric pressure
zr_avg = 140.0   # averaged height of top surface (to simplify domain to a cuboid)
z_bottom = -1000  # assuming plane bottom

h_max =zr_avg-z_bottom # maximal depth
rho = phi*rhof + (1-phi)*rhos   # bulk density
M = E*(1-nu)/((1+nu)*(1-2*nu))   # P-wave modulus  M = K+(4/3)*G
snu = nu/(1-nu)   # common factor for stress relation


# TIMESCALE
cv = k*M/mu   # [Verruijt: Soil Mechanics] eq. (15.16)
t_ref = (a**2)/cv
print("reference time t_ref = {:2.3e} s".format(t_ref))   # to guide time step
print("--")

# FLUID PRESSURE
# some test values to check for plausibility
h0 = 0.0*h_max        # depth of 1. point (given in meters below top)
h1 = 0.5*h_max         # depth of 2. point (given in meters below top)
h2 = 1.0*h_max         # depth of 3. point (given in meters below top)
p0 = rhof*g*h0 + pa
p1 = rhof*g*h1 + pa
p2 = rhof*g*h2 + pa
print("depth = {:2.3f} m:   p = {:2.3e} Pa".format(h0, p0))
print("depth = {:2.3f} m:   p = {:2.3e} Pa".format(h1, p1))
print("depth = {:2.3f} m:   p = {:2.3e} Pa".format(h2, p2))
# expression for prj-file
print("prj initial_p:   {:e}*max({:f}-z, 0) + {:e}   (C++ExprTk)".format(rhof*g, zr_avg, pa))   # max() to avoid negative pressures in points above average level
print("--")


# STRESS AND STRAIN
# total stress sigma_zz, some test values to check for plausibility
S0 = -rho*g*h0 - pa
S1 = -rho*g*h1 - pa
S2 = -rho*g*h2 - pa
# effective stress sigma'_zz, some test values to check for plausibility
Se0= S0 + p0
Se1= S1 + p1
Se2= S2 + p2
# output total stresses sigma_xx and sigma_yy
print("depth = {:2.3f} m:   sigma_xx = sigma_yy = {:2.3e} Pa,   sigma_zz = {:2.3e} Pa".format(h0, snu*Se0-p0, S0))
print("depth = {:2.3f} m:   sigma_xx = sigma_yy = {:2.3e} Pa,   sigma_zz = {:2.3e} Pa".format(h1, snu*Se1-p1, S1))
print("depth = {:2.3f} m:   sigma_xx = sigma_yy = {:2.3e} Pa,   sigma_zz = {:2.3e} Pa".format(h2, snu*Se2-p2, S2))
# output effective stresses sigma'_xx and sigma'_yy
print("depth = {:2.3f} m:   sigma'_xx = sigma'_yy = {:2.3e} Pa,   sigma'_zz = {:2.3e} Pa".format(h0, snu*Se0, Se0))
print("depth = {:2.3f} m:   sigma'_xx = sigma'_yy = {:2.3e} Pa,   sigma'_zz = {:2.3e} Pa".format(h1, snu*Se1, Se1))
print("depth = {:2.3f} m:   sigma'_xx = sigma'_yy = {:2.3e} Pa,   sigma'_zz = {:2.3e} Pa".format(h2, snu*Se2, Se2))
# expression for prj-file, only needed when fixed displacement (Dirichlet) is to be enforced approximately by the corresponding stress (Neumann)
print("( prj sigma_xx=sigma_yy:   {:2.3e}*({:2.3f}-z) - {:2.3e} )".format( g*(snu*(rhof-rho) - rhof), zr_avg, pa))
print("--")


# DISPLACEMENT (u in z-direction)
uz = lambda z: 0.5*(rho-rhof)*(g/M)*((zr_avg-z)**2 - h_max**2)
print("depth = {:2.3f} m:   u_z = {:2.4f} m".format(h0, uz(zr_avg-h0)))
print("depth = {:2.3f} m:   u_z = {:2.4f} m".format(h1, uz(zr_avg-h1)))
print("depth = {:2.3f} m:   u_z = {:2.4f} m".format(h2, uz(zr_avg-h2)))
print("prj initial_uz:   {:e}*(({}-z)^2-{}^2)   (C++ExprTk)".format(0.5*(rho-rhof)*g/M, zr_avg, h_max))
