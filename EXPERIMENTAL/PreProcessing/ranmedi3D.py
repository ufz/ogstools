# ranmedi - generating 3D random field from PSDF
# Contributing author: Joerg Buchwald

import numpy as np
import ranmedi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Ranmedi3D(ranmedi.Ranmedi):
    def __init__(self, xi, eps, lx=512, ly=512, lz=512, kappa=0.1, seed=42, meditype="gaussian"):
        self.xi = xi
        self.eps = eps
        self.kappa = kappa
        self.lx = lx
        self.ly = ly
        self.lz = lz
        np.random.seed(seed)
        self.psdf = np.zeros((lx,ly, lz), dtype=np.complex128)
        if meditype == "gaussian":
            self.gaussian()
            self.label = f"{meditype}, xi={xi}, eps={eps}"
        elif meditype == "exponential":
            self.exponential()
            self.label = f"{meditype}, xi={xi}, eps={eps}"
        elif meditype == "vkarman":
            self.vkarman()
            self.label = f"{meditype}, xi={xi}, eps={eps}, kappa={kappa}"
        self.ranmedi = self.getfield()
    def plot(self):
        init_z = 0
        fig, ax = plt.subplots()
        im = plt.imshow(self.ranmedi[:,:,init_z], cmap=plt.cm.jet)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        axcolor = 'lightgoldenrodyellow'
        ax.margins(x=0)
        # Make a horizontal slider to control the frequency.
        ax_z = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        z_slider = Slider(
            ax=ax_z,
            label='z coordinate',
            valmin=0,
            valmax=self.lz,
            valinit=init_z,
        )
        def update(val):
            im.set_data(self.ranmedi[:,:,int(z_slider.val)])
        z_slider.on_changed(update)
        fig.canvas.draw_idle()
        plt.colorbar(im)
        plt.title(self.label)
        plt.show()
    def krsq(self,i,j,k):
        return (i*2*np.pi/self.lx)**2+(j*2*np.pi/self.ly)**2+(k*2*np.pi/self.lz)**2
    def getfield(self):
        return np.real((np.fft.ifftn(self.psdf)))
    def fluk1(self, psdfi):
        return np.sqrt(psdfi) * np.exp(1.j * self.random)
    def fluk2(self,psdfi):
        return -np.conjugate(np.sqrt(psdfi) * np.exp(1.j*self.random))

    def gaussian(self):
        for i in range(0,int(self.lx/2)):
            for j in range(0,int(self.ly/2)):
                for l in range(0,int(self.lz/2)):
                    psdfi = self.eps**2 * np.sqrt(np.pi)**3 * (self.xi)**3 * np.exp(-self.krsq(i,j,l)*(self.xi)**2/4)
                    self.psdf[i,j,l] = self.fluk1(psdfi)
                    self.psdf[i,self.ly-j-1,l] = self.fluk2(psdfi)
                    self.psdf[self.lx-i-1,j,l] = self.fluk2(psdfi)
                    self.psdf[i,j,self.lz-l-1] = self.fluk2(psdfi)

if __name__=='__main__':
    rm = Ranmedi3D(15, 20)
    rm.plot()
