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

class expansivity_1(template.PROPERTY):
    def value(self, temperature):
        # source: https://www.engineeringtoolbox.com/water-density-specific-weight-d_595.html
        temp = np.array([
            0.0, 4., 10., 20., 30., 40., 50., 60., 70., 80., 90., 140., 200.,
            260.
        ]) + T0
        if (np.min(temperature) < temp[0]) or (np.max(temperature) > temp[-1]):
            print("The temperature is not within the defined domain")
        beta = np.array([
            -5.e-5, 0.003e-4, 8.8e-5, 2.07e-4, 3.03e-4, 3.85e-4, 4.57e-4, 5.22e-4,
            5.82e-4, 6.40e-4, 6.95e-4, 9.75e-4, 1.59e-3, 2.21e-3
        ])
        return np.interp(temperature, temp, beta)
    def dvalue(self, temperature):
        return np.gradient(self.value(temperature), temperature)
    def exprtk_value(self):
        raise NotImplementedError
    def exprtk_dvalue(self):
        raise NotImplementedError

class expansivity_2(template.PROPERTY):
    def value(self, temperature):
        # use only one-liner to keep it parsable:
        return  -1.46673616e-12 * temperature**4 + 2.46256925e-09 * temperature**3 - 1.51400613e-06 * temperature**2 + 4.11530308e-04 * temperature - 4.15284133e-02
    def dvalue(self, temperature):
        # use only one-liner to keep it parsable:
        return -1.46673616e-12 * 4 * temperature**3 + 2.46256925e-09 * 3 * temperature**2 - 1.51400613e-06 * 2 * temperature + 4.11530308e-04
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
