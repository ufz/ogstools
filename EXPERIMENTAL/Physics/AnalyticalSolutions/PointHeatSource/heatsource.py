import numpy as np
from scipy import special
import matplotlib.pyplot as plt

class ANASOL(object):
    def __init__(self):
        #material properties
        self.E = 5.e9 #Youngs modulus
        self.nu = 0.3 #Poisson ratio
        self.aprime = 1.5e-5 # coefficient of volume expansion of the soil a_u = a_s if no structural changes occur
        self.Q=300 # [Q]=W strength of the heat source
        self.n = 0.16 #porosity of soil
        self.rho_w = 999.1 #denstiy of pore water
        self.c_w = 4280 #specifict heat of pore water
        self.K_w = 0.6 # thermal conductivity of pore water
        self.rho_s = 2290.0 #density of the solid
        self.c_s = 917.654 #specific heat capacity
        self.K_s = 1.838 #themal conductivity of solid
        self.k = 2e-20 #coefficient of permeability
        self.gravity = 9.81 #gravity
        self.vis = 1e-3 #viscosity water at 20 deg
        self.a_s = 1.5e-5 # coefficient of volume expansion of skeletal material (beta_s)
        self.a_w = 4.0e-4 # coefficient of volume expansion of pore water (beta_w)
        self.T0 = 273.15

        self.Init()

    def f(self, ka, R, t):
        return special.erfc(R/(2*np.sqrt(ka*t)))

    def g(self, ka, R, t):
        return (ka*t/R**2+(1/2-ka*t/R**2)*special.erfc(R/(2*np.sqrt(ka*t)))-np.sqrt(ka*t/(np.pi*R**2))*np.exp(-R**2/(4*ka*t)))

    def fstar(self,R,t):
        return (self.Y*self.f(self.kappa,R,t)-self.Z*self.f(self.c,R,t))

    def gstar(self,R,t):
        return (self.Y*self.g(self.kappa,R,t)-self.Z*self.g(self.c,R,t))

    def temperature(self,x,y,z,t):
        R = self.R(x, y, z)
        return (self.Q/(4*np.pi*self.K*R)*self.f(self.kappa,R,t)+self.T0)

    def porepressure(self,x,y,z,t):
        R = self.R(x, y, z)
        return (self.X/(1-self.c/self.kappa)*self.Q/(4*np.pi*self.K*R)*(self.f(self.kappa,R,t)-self.f(self.c,R,t)))

    def u_i(self,x,y,z,t,i):
        R = self.R(x, y, z)
        index = {"x": x, "y": y, "z": z}
        return self.a_u*index[i]*self.Q/(4*np.pi*self.K*R)*self.gstar(R,t)

    def R(self,x,y,z):
        return np.sqrt(x**2+y**2+z**2)

    def dg_dR(self,ka,i,R,t):
        return ((2*i/R**3)*np.sqrt(ka*t/np.pi)*np.exp(-R*R/(4*ka*t))+(2*i*ka*t/R**4)*(self.f(ka,R,t)-1))

    def dgstar_dR(self,i,R,t): # Subscript R means derivative w.r.t R
        return (self.Y*self.dg_dR(self.kappa,i,R,t)-self.Z*self.dg_dR(self.c,i,R,t))

    def sigma_ii(self,x,y,z,t,ii): # for normal components
        R = self.R(x, y, z)
        index = {"xx": x, "yy": y, "zz": z}
        return ((self.Q*self.a_u/(4*np.pi*self.K*R))*(2*self.G*(self.gstar(R,t)*(1-index[ii]**2/R**2)+index[ii]*self.dgstar_dR(index[ii],R,t))
                        +self.lambd*(x*self.dgstar_dR(x,R,t)+y*self.dgstar_dR(y,R,t)+z*self.dgstar_dR(z,R,t)+2*self.gstar(R,t)))
                        -self.bprime*(self.temperature(x,y,z,t)-self.T0))

    def sigma_ij(self,x,y,z,t,i,j): # for shear components
        R = self.R(x, y, z)
        index = {"x": x, "y": y, "z": z}
        return ((self.Q*self.a_u/(4*np.pi*self.K*R))*(2*self.G*
                        (index[i]*self.dgstar_dR(index[j],R,t)/2+index[j]*self.dgstar_dR(index[i],R,t)/2-index[i]*index[j]*self.gstar(R,t)/R**2)))
    def Init(self):
        #derived constants
        self.gamma_w=self.gravity*self.rho_w #unit weight of water
        self.lambd=self.E*self.nu/((1+self.nu)*(1-2*self.nu))#lame constant
        self.G=self.E/(2*(1+self.nu)) # shear constant
        self.K=self.n*self.K_w+(1-self.n)*self.K_s #thermal conductivity
        self.bprime=(self.lambd+2*self.G/3)*self.aprime
        self.m=self.n*self.rho_w*self.c_w+(1-self.n)*self.rho_s*self.c_s
        self.kappa=self.K/self.m #scaled heat conductivity
        self.K_hydr=self.k*self.rho_w*self.gravity/self.vis #hydraulic conductivity
        self.a_u=self.a_s*(1-self.n)+self.a_w*self.n
        self.c=self.K_hydr*(self.lambd+2*self.G)/self.gamma_w #coefficient of consolidation
        self.X=self.a_u*(self.lambd+2*self.G)-self.bprime
        self.Y=1/(self.lambd+2*self.G) * (self.X/((1-self.c/self.kappa)*self.a_u)+self.bprime/self.a_u)
        self.Z=1/(self.lambd+2*self.G) * (self.X/((1-self.c/self.kappa)*self.a_u))
