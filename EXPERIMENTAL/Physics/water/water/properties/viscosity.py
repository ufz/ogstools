"""
Copyright (c) 2012-2021, OpenGeoSys Comunity (http://www.opengeosys.org)
              Distributed under a Modified BSD License.
                See accompanying file LICENSE or
                http://www.opengeosys.org/project/license

"""
# pylint: disable=C0103, R0902, R0914, R0913
import numpy as np
from water.properties import template
T0 = 273.15

class viscosity_1(template.PROPERTY):
    def value(self, temperature):
        # source: https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html
        temp = np.array([10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]) + T0
        if (np.min(temperature) < temp[0]) or (np.max(temperature) > temp[-1]):
            print("The temperature is not within the defined domain")
        vis = np.array([
            0.0013, 0.001, 0.0007978, 0.0006531, 0.0005471, 0.0004658,
            0.0004044, 0.000355, 0.000315, 0.00002822
        ])
        return np.interp(temperature, temp, vis)
    def dvalue(self, temperature):
        return np.gradient(self.value(temperature),temperature)
    def exprtk_value(self):
        raise NotImplementedError
    def exprtk_dvalue(self):
        raise NotImplementedError

class viscosity_2(template.PROPERTY):
    def value(self, temperature):
        # use only one-liner to keep it parsable:
        return -2.01570959e-09 * temperature**3 + 2.11402803e-06 * temperature**2 - 7.43932150e-04 * temperature +  8.82066680e-02
    def dvalue(self, temperature):
        # use only one-liner to keep it parsable:
        return -2.01570959e-09 * 3 * temperature**2 + 2.11402803e-06 * 2 * temperature - 7.43932150e-04
    def dvaluenum(self,temperature):
        return np.gradient(self.value(temperature),temperature)
    def exprtk_value(self):
        string = self._getcodeasstring("value", None)
        string = self._convertpythontoexprtk(string)
        return string
    def exprtk_dvalue(self):
        string = self._getcodeasstring("dvalue",None)
        string = self._convertpythontoexprtk(string)
        return string

class viscosity_3(template.PROPERTY):
    def value(self, temperature):
        return 2.414*10**(-5)*10**(247.8/(temperature-140))
    def dvalue(self, temperature):
        return -2.414*10**(-5)*247.8*np.log(10) * 10**(247.8/(temperature-140))/(temperature-140)**2
    def dvaluenum(self,temperature):
        return np.gradient(self.value(temperature),temperature)
    def exprtk_value(self):
        string = self._getcodeasstring("value", None)
        string = self._convertpythontoexprtk(string)
        return string
    def exprtk_dvalue(self):
        string = self._getcodeasstring("dvalue",None)
        string = self._convertpythontoexprtk(string)
        return string

class viscosity_4(template.PROPERTY):
    def value(self, temperature):
        return 4.2844e-5+1/(0.157 * (temperature-208.157) * (temperature-208.157)-91.296)
    def dvalue(self, temperature):
        raise NotImplementedError
    def dvaluenum(self,temperature):
        return np.gradient(self.value(temperature),temperature)
    def exprtk_value(self):
        string = self._getcodeasstring("value", None)
        string = self._convertpythontoexprtk(string)
        return string
    def exprtk_dvalue(self):
        string = self._getcodeasstring("dvalue", None)
        string = self._convertpythontoexprtk(string)
        return string
