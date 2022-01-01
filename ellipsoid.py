# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""

import copy
import numpy as np
    
from location import Location
import random
#from polymer import Polymer

###############################################################################
class Ellipsoid():
    
    def __init__(self, parameters):
        self.location = Location()
        self.tangleSet = {}
        self.particleCount = 0
        self.r = 0
        self.monomerCount = 0
        self.xCenter = 0
        self.yCenter = 0
        self.zCenter = 0
        self.Q_11 = 0
        self.Q_12 = 0
        self.Q_13 = 0
        self.Q_22 = 0
        self.Q_23 = 0
        self.Q_33 = 0
        
###############################################################################
    def GetLocation(self):
        return(self.location.GetLocation())
        
###############################################################################
    # need to add code to move Monomers
    def SetLocation(self, parameters, x, y, z):
        self.location.SetLocation(parameters.latticeConstant, x, y, z)
    
###############################################################################
    def SetRandomLocation(self, parameters):
        x = random.random() * parameters.oldPolymerBoxSize
        y = random.random() * parameters.oldPolymerBoxSize
        z = random.random() * parameters.oldPolymerBoxSize
        
        #print("x= ", x, "y= ", y, "z= ", z)

        self.location.SetLocation(parameters.latticeConstant, x, y, z) 
        return

###############################################################################
    def ScaleLocation(self, scaler):
        self.location.ScaleLocation(scaler)
        return
    
###############################################################################
    # DUG Ver 0.01.00 Added code to move polymer as a whole
    def DeltaMoveEllipsoid(self, parameters, deltaX, deltaY, deltaZ):
        [x, y, z] = self.location.GetLocation()
        x += deltaX
        y += deltaY
        z += deltaZ
        self.location.SetLocation(parameters.latticeConstant, x, y, z)
        return
###############################################################################
    def SetDebugPrintStatus(self, debugPrintsEnabled = False):
        self.debugPrintStatus = debugPrintsEnabled
              
###############################################################################
    def ClearTangleSet(self):
        self.tangleSet = {}
        return
    
###############################################################################
    def SetTangleSet(self, tangleSet):
        self.tangleSet = copy.deepcopy(tangleSet)
        return
       
###############################################################################
    def GetTangleSet(self):
        return (self.tangleSet)

###############################################################################
    def ClearParticles(self):
        self.particleCount = 0
        return
    
###############################################################################
    def AddParticle(self):
        self.particleCount += 1
        return(self.particleCount)

###############################################################################
    def GetParticleCount(self):
        return (self.particleCount)

###############################################################################
    def CalcRadius(self, parameters, allPolymers):
        xLoc = []
        yLoc = []
        zLoc = []
        
        for i in self.tangleSet:
            [x, y, z] = allPolymers[i].GetXYZAbsLoc()
            xLoc = np.concatenate((xLoc, x))
            yLoc = np.concatenate((yLoc, y))
            zLoc = np.concatenate((zLoc, z))
            
        xDistMatrix = np.subtract.outer(xLoc, xLoc)
        yDistMatrix = np.subtract.outer(yLoc, yLoc)
        zDistMatrix = np.subtract.outer(zLoc, zLoc)
               
        distanceMatrix=   np.sqrt(np.add(       np.square(xDistMatrix), \
                                         np.add(np.square(yDistMatrix), \
                                          np.square(zDistMatrix))))
        self.r = np.amax(distanceMatrix) / 2.0
        
        return

###############################################################################
    def GetRadius(self):
        reductionFactor = .9  # if < 1 then some monomers will be out of the spere
        return(self.r * reductionFactor) 
        
###############################################################################
    def CalcQ(self, R_a, R_b, R_acm, R_bcm):
        N = len(R_a)
        
        total = 0.0
        
        for i in range(N):
            total += (R_a[i] - R_acm) * (R_b[i] - R_bcm)
            
        result = total / float(N)
        return(result)
        
###############################################################################
    def PrintQs(self):
        print("Q_11= %12.5e" %                             (self.Q_11))
        print("Q_12= %12.5e, Q_22= %12.5e" %               (self.Q_12, self.Q_22))
        print("Q_13= %12.5e, Q_23= %12.5e, Q_33= %12.5e" % (self.Q_13, self.Q_23, self.Q_33))
        
        return
    
###############################################################################
    def GetCenterOfMassLocation(self):
        return(self.xCenter, self.yCenter, self.zCenter)
        
###############################################################################
    def CalcQs(self, xLoc, yLoc, zLoc):
        self.Q_11 = self.CalcQ(xLoc, xLoc, self.xCenter, self.xCenter)
        self.Q_22 = self.CalcQ(yLoc, yLoc, self.yCenter, self.yCenter)
        self.Q_33 = self.CalcQ(zLoc, zLoc, self.zCenter, self.zCenter)
        self.Q_12 = self.CalcQ(xLoc, yLoc, self.xCenter, self.yCenter)
        self.Q_13 = self.CalcQ(xLoc, zLoc, self.xCenter, self.zCenter)
        self.Q_23 = self.CalcQ(yLoc, zLoc, self.yCenter, self.zCenter)
        return
    
# Used for ellipsoids in non-drying case
###############################################################################
    def CalcForcedCenterOfMass(self, parameters, allPolymers):
        self.xCenter = 0.0
        self.yCenter = 0.0
        self.zCenter = 0.0
        self.monomerCount = 0
        xLoc = []
        yLoc = []
        zLoc = []
        
        self.xCenter = random.random() * parameters.boxSize
        self.yCenter = random.random() * parameters.boxSize
        self.zCenter = random.random() * parameters.boxSize
        
        for i in self.tangleSet:
            allPolymers[i].SetForcedCenterOfMass(1e-15, self.xCenter, self.yCenter, self.zCenter)
            self.monomerCount += allPolymers[i].GetNumberOfMonomers(parameters)
            
            [x, y, z] = allPolymers[i].GetXYZBaseLoc()
            
            x = np.add(x, self.xCenter)
            y = np.add(y, self.yCenter)
            z = np.add(z, self.zCenter)
            
            xLoc = np.concatenate((xLoc, x))
            yLoc = np.concatenate((yLoc, y))
            zLoc = np.concatenate((zLoc, z))

        self.CalcQs(xLoc, yLoc, zLoc)

        return
    
###############################################################################
    def CalcCenterOfMass(self, parameters, allPolymers):
        self.xCenter = 0.0
        self.yCenter = 0.0
        self.zCenter = 0.0
        self.monomerCount = 0
        xLoc = []
        yLoc = []
        zLoc = []
        
        for i in self.tangleSet:
            self.monomerCount += allPolymers[i].GetNumberOfMonomers(parameters)
            
            [x, y, z] = allPolymers[i].GetXYZAbsLoc()
            xLoc = np.concatenate((xLoc, x))
            yLoc = np.concatenate((yLoc, y))
            zLoc = np.concatenate((zLoc, z))

        self.xCenter = np.sum(xLoc) / self.monomerCount
        self.yCenter = np.sum(yLoc) / self.monomerCount
        self.zCenter = np.sum(zLoc) / self.monomerCount
        
        self.CalcQs(xLoc, yLoc, zLoc)

        return
    
###############################################################################
    def IsInEllipsoid(self, particle):
        [x, y, z] = particle.GetLocation()
        x_c = self.xCenter
        y_c = self.yCenter
        z_c = self.zCenter
        
        #factor = 1/7.81
        factor = 1.0/4.0
        
        Q_bar =       self.Q_11 * self.Q_22 * self.Q_33 + \
                2.0 * self.Q_12 * self.Q_13 * self.Q_23 - \
                      self.Q_23 * self.Q_23 * self.Q_11 - \
                      self.Q_13 * self.Q_13 * self.Q_22 - \
                      self.Q_12 * self.Q_12 * self.Q_33
        
        result = (((factor       * ((self.Q_22 * self.Q_33) - (self.Q_23 * self.Q_23))) / Q_bar) * ((x - x_c) * (x - x_c))) + \
                 (((factor       * ((self.Q_11 * self.Q_33) - (self.Q_13 * self.Q_13))) / Q_bar) * ((y - y_c) * (y - y_c))) + \
                 (((factor       * ((self.Q_11 * self.Q_22) - (self.Q_12 * self.Q_12))) / Q_bar) * ((z - z_c) * (z - z_c))) + \
                 (((factor * 2.0 * ((self.Q_13 * self.Q_23) - (self.Q_12 * self.Q_33))) / Q_bar) * ((x - x_c) * (y - y_c))) + \
                 (((factor * 2.0 * ((self.Q_12 * self.Q_23) - (self.Q_13 * self.Q_22))) / Q_bar) * ((x - x_c) * (z - z_c))) + \
                 (((factor * 2.0 * ((self.Q_13 * self.Q_12) - (self.Q_23 * self.Q_11))) / Q_bar) * ((y - y_c) * (z - z_c)))
                                  
        return(result < 1)