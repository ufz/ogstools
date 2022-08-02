import numpy as np
import time
from scipy import interpolate
import ranmedi1D
import ranmedi2D
import ranmedi3D
import vtuIO

class RanmediVTU(object):
    def __init__(self, vtufile, xi=250, eps=20, lx=64, ly=64, lz=64, kappa=0.1, seed=42, meditype="gaussian", dim=2):
        self.dim = dim
        self.lx = lx
        self.ly= ly
        self.lz = lz
        print(f"Gerating {meditype} random field for {dim} dimensions")
        if dim == 1:
            self.rm = ranmedi1D.Ranmedi1D(xi,eps,lx=lx,kappa=kappa, seed=seed, meditype=meditype)
        elif dim == 2:
            self.rm = ranmedi2D.Ranmedi2D(xi,eps,lx=lx, lz=ly, kappa=kappa, seed=seed, meditype=meditype)
        elif dim == 3:
            self.rm = ranmedi3D.Ranmedi3D(xi,eps,lx=lx, ly=ly, lz=lz, kappa=kappa, seed=seed, meditype=meditype)
        print("reading in VTU file")
        self.vtufile = vtuIO.VTUIO(vtufile)
        self.points = self.vtufile.points
        if dim ==2:
            self.field = self.scipyinterpd2d()
        if dim ==3:
            self.field = self.scipyinterpd3d()
    def scipyinterpd2d(self):
        start = time.time()
        field = np.zeros(len(self.points[:,0]))
        print("starting interpolation")
        x = [i for i in range(self.lx)]
        y = [i for i in range(self.ly)]
        f = interpolate.interp2d(x, y, self.rm.ranmedi, kind='cubic')
        x_offset = np.min(self.points[:,0])
        x_max = np.max(self.points[:,0])
        y_offset = np.min(self.points[:,1])
        y_max = np.max(self.points[:,1])
        X_predict = self.points.copy()
        X_predict[:,0] = self.points[:,0] - x_offset
        X_predict[:,0] = (X_predict[:,0] * self.lx / (x_max-x_offset))
        X_predict[:,1] = self.points[:,1] - y_offset
        X_predict[:,1] = (X_predict[:,1] * self.ly / (y_max-y_offset))
        print("applying to new datapoints")
        for i in range(len(self.points[:,0])):
            field[i] = f(X_predict[i,0], X_predict[i,1])
        stop = time.time()
        diff = stop-start
        print(f"interpolation took {diff}s")
        return field
    def scipyinterpd3d(self):
        start = time.time()
        field = np.zeros(len(self.points[:,0]))
        x = [i for i in range(self.lx)]
        y = [i for i in range(self.ly)]
        z = [i for i in range(self.lz)]
        f = interpolate.RegularGridInterpolator((x, y, z), self.rm.ranmedi)
        x_offset = np.min(self.points[:,0])
        x_max = np.max(self.points[:,0])
        y_offset = np.min(self.points[:,1])
        y_max = np.max(self.points[:,1])
        z_offset = np.min(self.points[:,2])
        z_max = np.max(self.points[:,2])
        X_predict = self.points.copy()
        X_predict[:,0] = self.points[:,0] - x_offset
        X_predict[:,0] = (X_predict[:,0] * (self.lx-1) / (x_max-x_offset))
        X_predict[:,1] = self.points[:,1] - y_offset
        X_predict[:,1] = (X_predict[:,1] * (self.ly-1) / (y_max-y_offset))
        X_predict[:,2] = self.points[:,2] - z_offset
        X_predict[:,2] = (X_predict[:,2] * (self.lz-1) / (z_max-z_offset))
        for i in range(len(self.points[:,0])):
            print((X_predict[i,0], X_predict[i,1], X_predict[i,2]))
            field[i] = f((X_predict[i,0], X_predict[i,1], X_predict[i,2]))
        stop = time.time()
        diff = stop-start
        print(f"interpolation took {diff}s")
        return field

    def writefield(self,fieldname, ofilename, celldata=False):
        print("writing VTU file")
        self.vtufile.write_field(self.field, fieldname, ofilename)
        if celldata is True:
            self.vtufile.point_data_to_cell_data(fieldname, ofilename)


if __name__== '__main__':
  #  for i in range(6):
  #      rm = RanmediVTU("square2d_random.vtu", xi=12, seed=i)
  #      rm.writefield(f"gaussian_field_{i}", "square2d_random.vtu", celldata=True)
    rm = RanmediVTU("tunnel.vtu", xi=10, lx=128,ly=128, eps=1, dim=2)
    porosityfield = rm.field+0.18
    permeabilityfield = 2.1407407407407408e-15 * porosityfield**3/(1-porosityfield)**2
    rm.vtufile.write_field(porosityfield,"porosity", "tunnel_random.vtu")
    rm.vtufile.write_field(permeabilityfield, "permeability", "tunnel_random.vtu")
    rm.vtufile.point_data_to_cell_data("permeability", "tunnel_random.vtu")
