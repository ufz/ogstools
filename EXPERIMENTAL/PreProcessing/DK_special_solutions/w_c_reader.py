#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:34:11 2021

@author: dominik

reading w_c_distr in addition to VMWARE to access original VMdata
without possible periodicity due to BBS pseudorandom generator

"""
import numpy as np
import sys


class w_c_reader:

    def __init__(self):
        self.nnn = np.zeros(3)
        self.xyz = np.zeros(6)
        self.w = 0


    def read(self, filename):
        nx = 0
        ny = 0
        nz = 0
        with open(filename) as file:
            for lineno, line in enumerate(file):

                if lineno==0:
                    headerarray=line.split()
                    NX = int(headerarray[0])
                    NY = int(headerarray[1])
                    NZ = int(headerarray[2])
                    self.nnn = np.array([NX, NY, NZ])
                    self.w = np.zeros((NX, NY, NZ))

                elif lineno==1:
                    headerarray=line.split()
                    X1 = float(headerarray[0])
                    X2 = float(headerarray[1])
                    Y1 = float(headerarray[2])
                    Y2 = float(headerarray[3])
                    ZB = float(headerarray[4])
                    ZR = float(headerarray[5])
                    self.xyz = np.array([X1, X2, Y1, Y2, ZB, ZR])

                else:
                    nx = nx % NX
                    ny = ny % NY
                    data = line.split()
                    L = len(data)
                    nxEnd = nx + L
                    if nxEnd > NX:
                        print("Inconsistent data, more points than declared in header!")
                        #sys.exit()
                    else:
                        self.w[nx:nxEnd, ny, nz] = list(map(float, data[0:L]))
                        nx = nxEnd

                    if nx == NX:
                        ny += 1
                        if ny == NY:
                            nz += 1
                            if nz > NZ:
                                print("Inconsistent data, more points than declared in header!")
                                #sys.exit()

        if nx<NX or ny<NY or nz<NZ:
            print("Inconsistent data, less points than declared in header!")
        return 0


# for debugging
# vm_w_c = w_c_reader()
# vm_w_c.read('w_c_distr')
