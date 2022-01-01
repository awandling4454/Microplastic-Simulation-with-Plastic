# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""
import numpy as np

from utilities import CalculateRadiusGyration
from utilities import CreateRandomVector
from utilities import NormalizeVector
from location import Location

###############################################################################
class Monomer(Location):
    def __init__(self):
       Location.__init__(self)  
    
###############################################################################
class PolymerMonomers:
    
    def __init__(self, parameters):
        self.xCenterOfMass = 0
        self.yCenterOfMass = 0
        self.zCenterOfMass = 0
        self.xLoc = np.full(parameters.polymerLength, 0, dtype=parameters.doMathIn)
        self.yLoc = np.full(parameters.polymerLength, 0, dtype=parameters.doMathIn)
        self.zLoc = np.full(parameters.polymerLength, 0, dtype=parameters.doMathIn)
        self._chain = []
        
        self.R_g = 0                    # Radius of gyration
        #self.debugPrintsEnabled = False
            
###############################################################################
    def GetNpArray(self, whichOne):
        if whichOne == "x":            
            return(self.xLoc)
            
        if whichOne == "y":            
            return(self.yLoc)
            
        if whichOne == "z":            
            return(self.zLoc)
            
        raise
###############################################################################
    def SetNpArray(self, npArray, whichOne):
        if whichOne == "x":            
            self.xLoc = npArray            
        elif whichOne == "y":            
            self.yLoc = npArray            
        elif whichOne == "z":            
            self.zLoc = npArray 
        else:
            raise
        
###############################################################################
    def SetBasePolymerIndex(self, index):
        self.basePolymerIndex = index
        return
    
###############################################################################
    def GetBasePolymerIndex(self):
        return(self.basePolymerIndex)
    
###############################################################################
        
    def AddMonomer(self, parameters, x, y, z):
        monomer = Monomer()
        monomer.SetLocation(parameters.latticeConstant, x, y, z)
        self._chain.append(monomer)
        return
        
###############################################################################
    def SetDebugPrintStatus(self, debugPrintsEnabled = False):
        #self.debugPrintsEnabled = debugPrintsEnabled
        return(False)

###############################################################################
    def CalcRadius(self, parameters):
        self.R_g = CalculateRadiusGyration(parameters.polymerLength)
        return

###############################################################################
    def GetGyrationRadius(self):
        return(self.R_g) 
        
###############################################################################
    def CalcCenterOfMass(self, parameters):
        self.xCenterOfMass = np.sum(self.xLoc) / parameters.polymerLength
        self.yCenterOfMass = np.sum(self.yLoc) / parameters.polymerLength
        self.zCenterOfMass = np.sum(self.zLoc) / parameters.polymerLength
        self.CalcRadius(parameters)
        return
    
###############################################################################
    def GetCenterOfMass(self):
        return(self.xCenterOfMass, self.yCenterOfMass, self.zCenterOfMass)
                    
###############################################################################
    def GetMonomerLocation(self, i):
        return(self.xLoc[i], self.yLoc[i], self.zLoc[i])
        
###############################################################################
    def RemoveSomeMonomers(self, count):
            delIndex = 0
            index = len(self._chain) - 1
            
            while (delIndex < count) and (len(self._chain) > 0):
                #if ((delIndex == 0) & (self.debugPrintsEnabled)):
                    #print("  len(self._chain)= %d, index= %d" % (len(self._chain), index))
                [x, y, z] = self._chain[index].GetLocation()
                self.xCenterOfMass -= x
                self.yCenterOfMass -= y
                self.zCenterOfMass -= z
                del self._chain[index]
                index -= 1
                delIndex += 1
                

###############################################################################
    def CreateVector(self):
        for i in range(len(self._chain)):
            [self.xLoc[i], self.yLoc[i], self.zLoc[i]] = self._chain[i].GetLocation()
                
        return            
###############################################################################
    def GetXYZLoc(self):            
        return(self.xLoc, self.yLoc, self.zLoc)
        
        
###############################################################################
    def FreeCreationMemory(self):
        del self._chain
        return
        
###############################################################################
    def CreatePolymerChain(self, parameters): 
        xMonomerLoc = 0
        yMonomerLoc = 0
        zMonomerLoc = 0

        # Controls direction polymer will go.  sameFactor + newFactor MUST = 1.0
        rememberFactor = parameters.polymerStiffness       # The higher the value the less wiggely the chain
        newFactor= 1.0 - rememberFactor
        
        #  The initial direction of the chain direction
        [xDistance, yDistance, zDistance] = CreateRandomVector(parameters, 2.0 * parameters.monomerRad)
        
        while ((len(self._chain) < parameters.polymerLength)):
                    
            # add a new link in the chain
            temp = Monomer()
            temp.SetLocation(parameters.latticeConstant, xMonomerLoc, yMonomerLoc, zMonomerLoc)
            self._chain.append(temp)
       
            # Chain need to go in the "somewhat" same direction
            # caclculate the new direction
            #[xDistanceNew, yDistanceNew, zDistanceNew] = xDistanceVector[x], yDistanceVector[x], zDistanceVector[x] 
            # now add the change of direction to the old direction
            # So new direction is mostly part old direction + part new direction            
            [xDistanceVector, yDistanceVector, zDistanceVector] = CreateRandomVector(parameters, 2.0 * parameters.monomerRad)
            
            xDistance =  (xDistance * rememberFactor) + (newFactor * xDistanceVector)
            yDistance =  (yDistance * rememberFactor) + (newFactor * yDistanceVector)
            zDistance =  (zDistance * rememberFactor) + (newFactor * zDistanceVector)
                 
            [xDistance, yDistance, zDistance] = NormalizeVector(2.0 * parameters.monomerRad, xDistance, yDistance, zDistance)
            
            xMonomerLoc += xDistance
            yMonomerLoc += yDistance
            zMonomerLoc += zDistance
                                   
        #if (self.debugPrintsEnabled): print("polymer created with %3d pull backs and length %6d" % (currentRetries, len(self._chain)))
        
        self.CreateVector()
        self.CalcCenterOfMass(parameters)
        self.FreeCreationMemory()
    
        # set the event to let main program know we are done and to 
        # read the queue with results of our calculation
        #if (self.debugPrintsEnabled): print("Polymer: Overlap count=", overlapCount)
        # I didn't want to toll old code, so False for mutiprocessing code
        return
        