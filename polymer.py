# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""
import random 
import copy
import numpy as np

from location import Location
from polymerMonomers import PolymerMonomers

    
###############################################################################
class Polymer:
    
###############################################################################    
    def __init__(self, parameters, x, y, z):
        self.startingMonomer = Location()
        self.monomerChain = PolymerMonomers(parameters)
        self.SetStartingLocation(parameters, x, y, z)
        
###############################################################################
    def Clone(self, parameters):
        polymer = Polymer(parameters, 0, 0, 0)
        polymer.monomerChain = copy.copy(self.monomerChain)
        return(polymer)

###############################################################################
    def GetStartingLocation(self):
        return(self.startingMonomer.GetLocation())
        
###############################################################################
    # need to add code to move Monomers
    def SetStartingLocation(self, parameters, x, y, z):
        scale = parameters.oldPolymerBoxSize / parameters.boxSize
        #print("Scaler = %f" % (scale))
        self.startingMonomer.SetLocation(parameters.latticeConstant, x * scale, y * scale, z * scale)
    
###############################################################################
    def SetRandomLocation(self, parameters):
        x = random.random() * parameters.oldPolymerBoxSize
        y = random.random() * parameters.oldPolymerBoxSize
        z = random.random() * parameters.oldPolymerBoxSize
        
        self.startingMonomer.SetLocation(parameters.latticeConstant, x, y, z) 
        return

###############################################################################
    def AddMonomer(self, parameters, x, y, z):
        self.monomerChain.AddMonomer(parameters, x, y, z)
        return
        
###############################################################################
    def SetEllipsoidIndex(self, index):
        self.ellipsoidIndex = index
        return
###############################################################################
    def GetEllipsoidIndex(self):
        try:
            return(self.ellipsoidIndex)
        except:
            return(None)
    
###############################################################################
    def SetBasePolymerIndex(self, index):
        self.monomerChain.SetBasePolymerIndex(index)
        return
    
###############################################################################
    def GetBasePolymerIndex(self):
        return(self.monomerChain.GetBasePolymerIndex())
    
###############################################################################
    def GetNpArray(self, whichOne):
        return(self.monomerChain.GetNpArray(whichOne))
###############################################################################
    def SetNpArray(self, npArray, whichOne):
        self.monomerChain.SetNpArray(npArray, whichOne)
        return
        
###############################################################################
    def SetDebugPrintStatus(self, debugPrintsEnabled = False):
        self.monomerChain.SetDebugPrintStatus(debugPrintsEnabled)
              
###############################################################################
    def SetTangleSet(self, tangleSet):
        self.tangleSet = copy.deepcopy(tangleSet)
        return
    
###############################################################################
# passing the index of this polymer may look funny..  But it saves lots of memory
    def GetTangleSet(self, index = -1):
        if (index == -1):
            raise ValueError("******\Error:  \nYou need to pass the polymer index as a parameter.\n*****") 
            
        try:
            return(self.tangleSet)
        except:
            return ({index})

###############################################################################
    def CalcRadius(self, parameters):
        self.monomerChain.CalcRadius(parameters)
        return

###############################################################################
    def GetGyrationRadius(self):
        result = self.monomerChain.GetGyrationRadius()
        return(result) 
        
###############################################################################
    def GetTangleRadius(self):
        reductionFactor = .6  # if < 1 then some monomers will be out of the spere
        result = self.monomerChain.GetGyrationRadius() * reductionFactor
        return(result) 
        
###############################################################################
    def GetCenterOfMass(self):
        [xM, yM, zM] = self.monomerChain.GetCenterOfMass()
        [xL, yL, zL] = self.startingMonomer.GetLocation()
        return(xL + xM, yL + yM, zL + zM)

    
###############################################################################
    def GetCenterOfMassAndTangleRadius(self):
        [x,y,z] = self.GetCenterOfMass()
        r = self.GetTangleRadius()
        return(x,y,z,r)
###############################################################################
    def SetCenterOfMass(self, parameters, x, y, z):
        scale = parameters.oldPolymerBoxSize / parameters.boxSize
        self.monomerChain.SetLocation(parameters.latticeConstant, x * scale, y * scale, z * scale)
                    
###############################################################################
    def SetForcedCenterOfMass(self, latticeConstant, xCenter, yCenter, zCenter):
        [xMass, yMass, zMass] = self.monomerChain.GetCenterOfMass()
        xStart = xCenter - xMass
        yStart = yCenter - yMass
        zStart = zCenter - zMass
        self.startingMonomer.SetLocation(latticeConstant, xStart, yStart, zStart)
        return
############################################################.###################
    def GetMonomerLocation(self, i):
        [x, y, z] = self.monomerChain.GetMonomerLocation(i)
        [xI, yI, zI] = self.startingMonomer.GetLocation()
       
        return(x + xI, y + yI, z + zI)
        
###############################################################################
    def GetNumberOfMonomers(self, parameters):
        return(parameters.polymerLength)
                
###############################################################################
    def GetXYZAbsLoc(self):
        [xStart, yStart, zStart] = self.startingMonomer.GetLocation()
        [xLoc, yLoc, zLoc] = self.monomerChain.GetXYZLoc()
        
        xAbsLoc = np.add(xLoc, xStart)
        yAbsLoc = np.add(yLoc, yStart)
        zAbsLoc = np.add(zLoc, zStart)
        
        return(xAbsLoc, yAbsLoc, zAbsLoc)
        
###############################################################################
    def GetXYZBaseLoc(self):
        return(self.monomerChain.GetXYZLoc())
        
###############################################################################
    def RemoveSomeMonomers(self, count):
        self.monomerChain.RemoveSomeMonomers(count)
                
###############################################################################
    # DUG Ver 0.01.00 Added code to move polymer as a whole
    def DeltaMovePolymer(self, parameters, deltaX, deltaY, deltaZ):
        [x, y, z] = self.startingMonomer.GetLocation()
        x += deltaX
        y += deltaY
        z += deltaZ
        self.startingMonomer.SetLocation(parameters.latticeConstant, x, y, z)
        return

###############################################################################
    def ScaleLocation(self, parameters, scaler):
        self.startingMonomer.ScaleLocation(scaler)
        return

###############################################################################
    def CreateVector(self):
        self.monomerChain.CreateVector(self.startingMonomer)
        return
    
###############################################################################
    def DebugPrintChain(self, parameters):
        self.monomerChain.DebugPrintChain(parameters)
        
###############################################################################
        
    def CreatePolymerChain(self, parameters): 
        self.monomerChain.CreatePolymerChain(parameters)   
