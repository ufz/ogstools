"""
Copyright (c) 2012-2021, OpenGeoSys Community (http://www.opengeosys.org)
              Distributed under a Modified BSD License.
                See accompanying file LICENSE or
                http://www.opengeosys.org/project/license

"""
# pylint: disable=C0103, R0902, R0914, R0913
import numpy as np
from water.properties import template
T0 = 273.15

class specificheatcapacity_1(template.PROPERTY):
    def value(self, temperature):
        # source: https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
        temp = np.array([0.01, 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]) + T0
        if (np.min(temperature) < temp[0]) or (np.max(temperature) > temp[-1]):
            print("The temperature is not within the defined domain")
        cw = np.array([
            4217., 4191., 4157., 4118., 4074., 4026., 3977., 3925., 3873.,
            3820., 3768
        ])
        return np.interp(temperature, temp, cw)
    def dvalue(self, temperature):
        return np.gradient(self.value(temperature),temperature)
    def exprtk_value(self):
        raise NotImplementedError
    def exprtk_dvalue(self):
        raise NotImplementedError

class specificheatcapacity_2(template.PROPERTY):
    def value(self, temperature):
        # use only one-liner to keep it parsable:
        return  1.55452794e-04 * temperature**3 - 1.64289282e-01 * temperature**2 + 5.25998084e+01 * temperature - 1.06049924e+03
    def dvalue(self, temperature):
        # use only one-liner to keep it parsable:
        return 1.55452794e-04 * 3 * temperature**2 - 1.64289282e-01 * 2 * temperature + 5.25998084e+01
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
