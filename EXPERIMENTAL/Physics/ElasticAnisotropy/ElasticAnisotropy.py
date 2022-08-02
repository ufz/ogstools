import numpy as np

class ElasticAnisotropy(object):
    """Class for conversion and rotation of elastic constants

    Parameters
    ----------
    E_nu_const : `dict` containing elastic moduli
    """
    def __init__(self, E_nu_const):

        E_1 = E_nu_const['E_1']
        E_2 = E_nu_const['E_2']
        E_3 = E_nu_const['E_3']
        nu_12 = E_nu_const['nu_12']
        nu_23 = E_nu_const['nu_23']
        nu_13 = E_nu_const['nu_13']
        G_12 = E_nu_const['G_12']
        G_23 = E_nu_const['G_23']
        G_13 = E_nu_const['G_13']
        nu_21 = nu_12 * E_2 / E_1
        nu_32 = nu_23 * E_3 / E_2
        nu_31 = nu_13 * E_3 / E_1
        D = (1 - nu_12*nu_21-nu_23*nu_32-nu_31*nu_13-2*nu_12*nu_23*nu_31)/(E_1 * E_2 * E_3)
        self.C = np.array([[(1-nu_23*nu_32)/(E_2*E_3*D), (nu_21+nu_31*nu_23)/(E_2*E_3*D), (nu_31+nu_21*nu_32)/(E_2*E_3*D), 0, 0, 0],
                [(nu_12+nu_13*nu_32)/(E_1*E_3*D), (1-nu_31*nu_13)/(E_1*E_3*D), (nu_32+nu_31*nu_12)/(E_1*E_3*D), 0, 0, 0],
                [(nu_13+nu_12*nu_23)/(E_1*E_2*D), (nu_23+nu_13*nu_21)/(E_1*E_2*D), (1-nu_12*nu_21)/(E_1*E_2*D), 0, 0, 0],
                [0, 0, 0, G_23, 0, 0],
                [0, 0, 0, 0, G_13, 0],
                [0, 0, 0, 0, 0, G_12]])

    def getRmatrix(self, n):
        """Compute a rotation matrix from vector n

        Parameters
        ----------
        n : `list`, `numpy.array` or `tuple`
        """
        e3 = np.array([n[0], n[1], n[2]])
        e1 = np.array([n[2]/(n[0]*np.sqrt(1+(n[2]/n[0])**2)), 0, -1.0/np.sqrt(1+(n[2]/n[0])**2)])
        e2 = np.cross(e3,e1)
        R = np.array([[e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]]])
        return R

    #TODO: can be a property or general fct that takes a 6x6 matrix
    def getTensor(self):
        """Construct a tensor from Voigt matrix given by self.C

        """
        cijkl = np.zeros((3,3,3,3))
        def getCijkl(i,j,k,l):
            def getvoigtindex(i,j):
                if i == j:
                    return i
                elif (i == 1 and j==2) or (i == 2 and j==1):
                    return 3
                elif (i == 0 and j==2) or (i == 2 and j==0):
                    return 4
                elif (i == 0 and j==1) or (i == 1 and j==0):
                    return 5
            m = getvoigtindex(i,j)
            n = getvoigtindex(k,l)
            return self.C[m,n]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        cijkl[i,j,k,l] = getCijkl(i, j,k,l)
        return cijkl

    def getVoigtMatrix(self, cijkl):
        """Convert tensor to Voigt matrix

        Parameters
        ----------
        cijkl : `numpy.array`
        """
        C = np.zeros((6,6))
        def getCij(i,j):
            def gettensorindex(i):
                if (i < 3):
                    return i, i
                elif i == 3:
                    return 1, 2
                elif i == 4:
                    return 0, 2
                elif i == 5:
                    return 0, 1
            k, l = gettensorindex(i)
            m, n = gettensorindex(j)
            return cijkl[k,l,m,n]
        for j in range(6):
            for i in range(6):
                C[i,j] = getCij(i,j)
        return C

    def rotateTensor(self, cijkl, R):
        """Rotate tensor with rotation matrix

        Parameters
        ----------
        cijkl : `numpy.array`
        R : `numpy.array`
        """
        cijklnew = np.zeros((3,3,3,3))
        def transformTensor(c,R,i,j,k,l):
            c_new = 0.0
            for r in range(3):
                for s in range(3):
                    for t in range(3):
                        for u in range(3):
                            c_new += R[i,r]*R[j,s]*R[k,t]*R[l,u]*c[r,s,t,u]
            return c_new
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        cijklnew[i,j,k,l] = transformTensor(cijkl,R,i,j,k,l)
        return cijklnew
