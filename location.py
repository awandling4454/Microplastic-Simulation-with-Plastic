# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
Editted 6/21/2021
Allison Wandling
"""
from utilities import SetLatticeConstant
import numpy as np

###############################################################################
class Location:
    def __init__(self):
        self.zyxLoc = np.full(3, 0.0, dtype="float64")
        
###############################################################################
    def SetLocation(self, latticeConstant, x, y, z): 
        [self.zyxLoc[0], self.zyxLoc[1], self.zyxLoc[2]] = SetLatticeConstant(latticeConstant, x, y, z)
        return
        
###############################################################################
    def GetLocation(self):
        return(self.zyxLoc[0], self.zyxLoc[1], self.zyxLoc[2])
###############################################################################
    def GetNpLocation(self):
        return(self.zyxLoc)
        
##############################################################################
    def ScaleLocation(self, scale):        
        self.zyxLoc = np.multiply(self.zyxLoc, scale)
        return
        
###############################################################################
    def PrintLocation(self, parameters):
        print("  x= %7.1e, y= %7.1e, z= %7.1e" % (self.zyxLoc[0], self.zyxLoc[1], self.zyxLoc[2]))
        return
         
###############################################################################
