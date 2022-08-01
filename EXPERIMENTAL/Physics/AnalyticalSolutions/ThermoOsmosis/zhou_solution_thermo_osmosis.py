# -*- coding: utf-8 -*-
"""
Analytical solution for thermo-osmosis in 1D column.
See [1] for details.

[1] Zhou, Y., Rajapakse, R. K. N. D., & Graham, J. (1998).
International Journal of Solids and Structures, 35(34-35), 4659-4683.


Copyright (c) 2012-2022, OpenGeoSys Community (http://www.opengeosys.org)
            Distributed under a Modified BSD License.
              See accompanying file LICENSE or
              http://www.opengeosys.org/project/license

"""

# pylint: disable=C0103, R0902, R0914, R0913, C0301, C0114, C0115, C0116

import numpy as np
from scipy import special

class ANASOL:
    """ Analytical solution for
        Themo-Osmosis in a 1D collumn

    """
    def __init__(self, T0=0,T1=50,l=99.5):
        #material properties
        self.aw = 3.0e-4 #K^-1
        self.as_ = 3.0e-6 #K^-1
        self.alpha = 3.0e-6 # K^-1
        self.Cs = 937. #J/kg/K
        self.Cw = 4186.0 #J/kg
        self.rhow = 1000. #kg/m^3
        self.rhos = 2610. #kg/m^3
        self.betaw = 3.3e9 # Pa
        self.Ks = 59e9 # Pa
        self.lambdas = 3.29 # J/(s m K)
        self.lambdaw = 0.582 #J/(s m K)
        self.Sw = 2.7e-10 # m / (sK)
        self.k = 5e-14 # m^5/(Js)=m^2/(Pa s)
        self.E = 2.88e6 # Pa
        self.nu = 0.2
        self.n = 0.375

        self.T0 = T0
        self.T1 = T1
        self.l = l

    @property
    def Kprime(self):
        return self.E/(3*(1-2*self.nu))
    @property
    def lambda_(self):
        return (1-self.n)*self.lambdas + self.lambdaw*self.n
    @property
    def Cv(self):
        return (1-self.n)*self.rhos*self.Cs+self.n*self.rhow*self.Cw
    @property
    def G(self):
        return self.E/(2*(1+self.nu))
    @property
    def c1(self):
        return 1 - self.Kprime/self.Ks
    @property
    def xi(self):
        return self.c1
    @property
    def c2(self):
        return self.n*self.aw + (1-self.n)*self.as_-self.alpha*self.Kprime/self.Ks
    @property
    def c3(self):
        return self.n/self.betaw+(1-self.n)/self.Ks
    @property
    def a1(self):
        return -self.T0*self.Kprime*self.alpha*self.Sw/self.c1 - self.lambda_ + self.T0*self.aw*self.betaw*self.Sw
    @property
    def a2(self):
        return -self.T0*self.Kprime*self.alpha*self.k/self.c1 - self.T0*(self.Sw-self.aw*self.betaw*self.k)
    @property
    def a3(self):
        return -self.T0*self.Kprime*self.alpha*self.c2/self.c1+self.Cv
    @property
    def a4(self):
        return self.T0*self.Kprime*self.alpha*self.c3/self.c1
    @property
    def b1(self):
        return (1-self.nu)*2*self.G*self.Sw/((1-2*self.nu)*self.c1)
    @property
    def b2(self):
        return (1-self.nu)*2*self.G*self.k/((1-2*self.nu)*self.c1)
    @property
    def b3(self):
        return (1-self.nu)*2*self.G*self.c2/((1-2*self.nu)*self.c1) - self.Kprime*self.alpha
    @property
    def b4(self):
        return -(1-self.nu)*2*self.G*self.c3/((1-2*self.nu)*self.c1)-self.xi
    @property
    def h1(self):
        return -(self.a1*self.b2-self.b1*self.b2)/(self.a1*self.b3-self.a3*self.b1)
    @property
    def h2(self):
        return -(self.a1*self.b4-self.b1*self.a4)/(self.a1*self.b3-self.a3*self.b1)
    @property
    def h3(self):
        return self.a1/(self.a1*self.b3-self.a3*self.b1)
    @property
    def g1(self):
        return self.h1
    @property
    def g2(self):
        return self.a2/self.a1+self.h2+self.a3/self.a1*self.h1
    @property
    def g3(self):
        return self.a4/self.a1+self.a3/self.a1*self.h2
    @property
    def gamma1(self):
        return (-self.g2-np.sqrt(self.g2**2-4*self.g1*self.g3))/(2*self.g1)
    @property
    def gamma2(self):
        return (-self.g2+np.sqrt(self.g2**2-4*self.g1*self.g3))/(2*self.g1)
    @property
    def h0(self):
        return -self.a3*self.h3/(self.a1*self.g3)
#    @property
#    def E1(self):
#        return (self.xi+self.Kprime*self.alpha*(self.h1*self.gamma1**2+self.h2))/((1-self.nu)*self.G/(1-2*self.nu))
#    @property
#    def E2(self):
#        return (self.xi+self.Kprime*self.alpha*(self.h1*self.gamma2**2+self.h2))/((1-self.nu)*self.G/(1-2*self.nu))
#    @property
#    def F1(self):
#        pass
#    @property
#    def F2(self):
#        pass
#    @classmethod
#    def phi(cls,x,t):
#        return 2 * np.sqrt(t/np.pi) * np.exp(-x**2 / (4*t)) - x * special.erfc( x/(2*np.sqrt(t)))

    def T(self, x, t, mmax):
        """ temperature

        Parameters
        ----------
        x : `float`
        t : `float`
        mmax : `int`
        """
        prefactor = -self.T1/(self.h1*(self.gamma1-self.gamma2))
        summe = 0.0
        for m in range(mmax):
            summe = summe + (self.h1*self.gamma1+self.h2)*(special.erfc( ((2*m+2)*self.l-x)*np.sqrt(self.gamma1)/(2*np.sqrt(t))  )-special.erfc( (2*m*self.l+x)*np.sqrt(self.gamma1)/(2*np.sqrt(t)) ) ) - (self.h1*self.gamma2+self.h2)*(special.erfc( ((2*m+2)*self.l-x)*np.sqrt(self.gamma2)/(2*np.sqrt(t))  ) - special.erfc( (2*m*self.l+x)*np.sqrt(self.gamma2)/(2*np.sqrt(t))  ))
        return prefactor*summe
    def p(self, x, t, mmax):
        """ pressure

        Parameters
        ----------
        x : `float`
        t : `float`
        mmax : `int`
        """
        prefactor = -self.T1/(self.h1*(self.gamma1-self.gamma2))
        summe = 0.0
        for m in range(mmax):
            summe = summe + (special.erfc( ((2*m+2)*self.l-x)*np.sqrt(self.gamma1)/(2*np.sqrt(t))  )-special.erfc( (2*m*self.l+x)*np.sqrt(self.gamma1)/(2*np.sqrt(t)) ) ) - (special.erfc( ((2*m+2)*self.l-x)*np.sqrt(self.gamma2)/(2*np.sqrt(t))  ) - special.erfc( (2*m*self.l+x)*np.sqrt(self.gamma2)/(2*np.sqrt(t))  ))
        return prefactor*summe
