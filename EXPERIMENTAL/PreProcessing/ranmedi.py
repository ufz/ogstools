# ranmedi - generating 2D random field from PSDF
# Contributing author: Joerg Buchwald

import numpy as np

class Ranmedi(object):
    def __init__(self, xi, eps, lx=512, ly=512, lz=512, kappa=0.1, seed=42):
        self.xi = xi
        self.eps = eps
        self.kappa = kappa
        self.lx = lx
        self.ly = ly
        self.lz = lz
        np.random.seed(seed)
    @property
    def random(self):
        return 2*np.pi*(np.random.rand()-0.5)
    def fluk1(self, psdfi):
        return np.sqrt(psdfi) * np.exp(1.j * self.random)
    def fluk2(self,psdfi):
        return -np.conjugate(np.sqrt(psdfi) * np.exp(1.j*self.random))
