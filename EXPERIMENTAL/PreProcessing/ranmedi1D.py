# ranmedi - generating 1D random field from PSDF
# Contributing author: Joerg Buchwald

import numpy as np
import ranmedi
import matplotlib.pyplot as plt

class Ranmedi1D(ranmedi.Ranmedi):
    def __init__(self, xi, eps, lx=512, kappa=0.1, seed=42, meditype="gaussian"):
        self.xi = xi
        self.eps = eps
        self.kappa = kappa
        self.lx = lx
        np.random.seed(seed)
        self.psdf = np.zeros(lx, dtype=np.complex128)
        if meditype == "gaussian":
            self.gaussian()
            self.label = f"{meditype}, xi={xi}, eps={eps}"
#        elif meditype == "exponential":
#            self.exponential()
#            self.label = f"{meditype}, xi={xi}, eps={eps}"
#        elif meditype == "vkarman":
#            self.vkarman()
#            self.label = f"{meditype}, xi={xi}, eps={eps}, kappa={kappa}"
        self.ranmedi = self.getfield()
    def plot(self):
        plt.plot(range(self.lx),self.ranmedi)
        plt.title(self.label)
        plt.show()
    def krsq(self,i):
        return (i*2*np.pi/self.lx)**2
    def getfield(self):
        return np.real((np.fft.ifft(self.psdf)))
    def fluk1(self, psdfi):
        return np.sqrt(psdfi) * np.exp(1.j * self.random)
    def fluk2(self,psdfi):
        return -np.conjugate(np.sqrt(psdfi) * np.exp(1.j*self.random))

    def gaussian(self):
        for i in range(0, int(self.lx/2)):
            psdfi = self.eps**2*np.sqrt(np.pi) * self.xi * np.exp(-self.krsq(i) * (self.xi)**2/4)
            self.psdf[i] = self.fluk1(psdfi)
            self.psdf[self.lx-i-1] = self.fluk2(psdfi)


if __name__=='__main__':
    rm = Ranmedi1D(15, 20)
    rm.plot()
