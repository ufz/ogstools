# read VM's fortran results (3D) and add interpolations (w_c) to nodes of a given mesh
import numpy as np      # for operations and data types
from scipy.interpolate import RegularGridInterpolator   # for 3D interpolation
import meshio   # read/write meshes
import vmware   # Victor Malkovsky data for distribution of water conductivity (by D. Kern)
# argv.py
import sys

# TODO make usage fail-safe
inputfile = sys.argv[1]
outputfile = sys.argv[2]



# READ IN VM-DATA
print(vmware.w_cond.__doc__)   # to check if library found (should be in working directory)
nnn = np.array([0,0,0],'i')  # nx, ny, nz  division in respective coordinate directions
xyz = np.array([0.0,0.0,0.0,0.0,0.0,0.0],'f')    # x1r,x2r, y1r,y2r, zbas,z2r
w = vmware.w_cond(nnn, xyz)  # function results are returned in w, nnn and xyz

# vm data in domain: [X1R,X2R] [Y1R,Y2R] [ZBAS,Z2R]
NXW = nnn[0]
NYW = nnn[1]
NZW = nnn[2]
x = np.linspace(0.0, xyz[1]-xyz[0], NXW) # shifted to x=0 as mesh does
y = np.linspace(0.0, xyz[3]-xyz[2], NYW) # shifted to y=0 as mesh does
z = np.linspace(0.0, xyz[5]-xyz[4], NZW) # shifted to z=0 as mesh does
print("VM nxw, nyw, nzw: ", NXW, NYW, NZW)

# READ IN MESH AND ADD INTERPOLATED DATA
#~/OGS/build/bin/generateStructuredMesh -e hex -o hexmesh.vtu --lx 12499.2 --ly 18200 --lz 1429.9
#[2020-07-23 10:22:54.240] [ogs] [info] Mesh created: 1331 nodes, 1000 elements.
mesh = meshio.read(inputfile)
# cell_data          cells              get_cells_type()   point_data         read()
# cell_data_dict     cells_dict         gmsh_periodic      point_sets         sets_to_int_data()
# cell_sets          field_data         info               points             write()
# cell_sets_dict     get_cell_data()    int_data_to_sets() prune()
print(f'MESH: {len(mesh.points)} points')	# array
print(f'MESH: {len(mesh.cells[0].data)} cells')   # dictionary

# add interpolated w_c_values to mesh ([0,LX]  [0,LY] [0,LZ])
interpolating_function = RegularGridInterpolator((x, y, z), w[0:NXW,0:NYW,0:NZW], method='linear', bounds_error=False, fill_value=None)
w_interpolation = interpolating_function(mesh.points)
mesh.point_data = {'w_c': w_interpolation}   # give data a name, this will be visible in paraview

# write to file
meshio.write(outputfile ,mesh)
# seems compressed, as it is smaller than original file (ogs/generatemesh) although data were added
