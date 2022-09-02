'''
read VM's fortran results (3D) and add interpolations (w_c) to cells of a given mesh

runfile('/home/dominik/projects/python/vm_data/src/appendCellData.py',
        wdir='/home/dominik/projects/python/vm_data/src',
        args='ys11_normal_domain_volume_only.vtu ys11_normal_domain_appended_cell_data.vtu')

'''
import numpy as np      # for operations and data types
from scipy.interpolate import RegularGridInterpolator   # for 3D interpolation
import meshio   # read/write meshes
import vmware   # Victor Malkovsky data fortran wrapper (by D. Kern)
import w_c_reader   # read w_c_distr (ascii file)
# argv.py
import sys

if len(sys.argv)==3:
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
else:
    print("Error, wrong number of command line arguments!")
    print("usage: python3 appendCellData.py INPUTFILE OUTPUTFILE")

# READ IN VM-DATA
VMFLAG = False # True = read FORTRAN data, False = read w_c_distr

if VMFLAG:
    print(vmware.w_cond.__doc__)   # to check if library found (should be in working directory)
    nnn = np.array([0,0,0], 'i')  # nx, ny, nz  division in respective coordinate directions
    xyz = np.array([0.0,0.0,0.0,0.0,0.0,0.0], 'f')    # x1r,x2r, y1r,y2r, zbas,z2r
    w = vmware.w_cond(nnn, xyz)  # function results are returned in w, nnn and xyz
else:
    vm_w_c = w_c_reader.w_c_reader()
    vm_w_c.read('w_c_distr')
    nnn = vm_w_c.nnn
    xyz = vm_w_c.xyz
    w   = vm_w_c.w


# vm data in domain: [X1R,X2R] [Y1R,Y2R] [ZBAS,Z2R]
NXW = nnn[0]
NYW = nnn[1]
NZW = nnn[2]
x = np.linspace(0.0, xyz[1]-xyz[0], NXW) # shifted to x=0 as mesh does
y = np.linspace(0.0, xyz[3]-xyz[2], NYW) # shifted to y=0 as mesh does
z = np.linspace(0.0, xyz[5]-xyz[4], NZW) # shifted to z=0 as mesh does
print("VMgrid nxw, nyw, nzw: ", NXW, NYW, NZW)
print("VMgrid x1,x2, y1,y2, zb,zr: ", xyz)

# READ IN MESH AND ADD INTERPOLATED DATA
mesh = meshio.read(inputfile)
print(f'MESH: {len(mesh.points)} points')	# array
print(f'MESH: {len(mesh.cells[0].data)} cells')   # dictionary
#print(mesh)

# add interpolated w_c_values to mesh ([0,LX]  [0,LY] [0,LZ])
interpolating_function = RegularGridInterpolator((x, y, z),
                                                 w[0:NXW, 0:NYW, 0:NZW],
                                                 method='nearest',
                                                 bounds_error=False,
                                                 fill_value=None)   # method='nearest', 'linear'

cells_count = len(mesh.cells[0].data)
centerpoints=np.zeros((cells_count, 3))
for cell_index, cellblock_cell in enumerate(mesh.cells[0].data):
    centerpoints[cell_index] = np.sum(mesh.points[cellblock_cell], axis=0) / len(cellblock_cell)

w_interpolation = interpolating_function(centerpoints)
k_cell = w_interpolation * 1e-7   # k=w_c*mu/(rho*g)
k_min = np.min(w[w>0]) * 1e-7  # ignore w=0 of blocks outside domain
k_max = np.max(w) * 1e-7
dk = k_max - k_min
print("k_mean = {:e}".format(np.mean(k_cell)))

# linear interpolation (increasing) for porosity
phi_min = 0.0015
phi_max = 0.0050
dphi = phi_max - phi_min
phi_cell = phi_min + dphi*(k_cell-k_min)/dk
print("phi_mean = {:e}".format(np.mean(phi_cell)))

# linear interpolation (decreasing) for Young's modulus
E_min = 5.630e10
E_max = 7.870e10
dE = E_max - E_min
E_cell = E_max - dE*(k_cell-k_min)/dk
print("E_mean = {:e}".format(np.mean(E_cell)))

min_k_cell = min(k_cell)
if min_k_cell < 0:
    print("Warning, negative value k={} found, shifted to zero.".format(min_k_cell))
    k_cell -= min_k_cell  # shift to nonnegative values

mesh.cell_data['vm_permeability'] = [np.array(k_cell)]   # append data
mesh.cell_data['vm_porosity'] = [np.array(phi_cell)]   # append data
mesh.cell_data['vm_youngs_modulus'] = [np.array(E_cell)]   # append data


# write to file
meshio.write(outputfile, mesh)
# seems compressed, as it is smaller than original file (ogs/generatemesh) although data were added
