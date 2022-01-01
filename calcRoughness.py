# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017
aggDistanceOutBottom
@author: Michelle
"""

import random
import math

from myCsv     import MyCsv
from utilities import CalculateVectorLength

##############################################################################
def CalculateRoughness(parameters, data, fileName, indexs, logResults = True, displayToConsole = False):
    if logResults:
        data.log.Log("\nRoughness calculation created file %s." % (fileName))
        data.log.Log("Roughness at levels from percent bottom of box.")
        
    if displayToConsole:
        print("\nRoughness calculation created file %s." % (fileName))
        print("Roughness at levels from percent bottom of box.")

    tLog = MyCsv(fileName, 10)
    tLog.Log(0, "Percent, Roughness, Candidates, Candidate/sample pairs")
    numberSamples = 1000000
    
    for i in indexs:
        zPlain = (float(i) / 100.0) * parameters.boxSize
        candidates = []
        
        #Get list of Candidate Particles
        for particle in data.allParticles:
            [x, y, z] = particle.GetLocation()
            r = particle.GetRadius()
            
            if ((z < zPlain) & ((z + r) > zPlain)):
                candidates.append(particle)
                
        total = 0.0
        hits = 0
        
        
        # points on the surface of a sphere has the formula r = (x**2 + y**2 + z**2)**.5
        # we know all values except z.  So we rewrite the equation to solve for z
        # Rewrite result: 
        # r = (x**2 + y**2 + z**2)**.5
        # r**2 = x**2 + y**2 + z**2
        # r**2 - (x**2 + y**2) = x**2 + y**2 + z**2 - (x**2 + y**2)
        # z = (r**2 - (x**2 + y**2)**.5
        
        for sample in range(numberSamples):
            xRandom = random.random() * parameters.boxSize
            yRandom = random.random() * parameters.boxSize
            
            for candidate in candidates:
                radius = candidate.GetRadius()
                [xLoc, yLoc, zLoc] = candidate.GetLocation()
                deltaX = xRandom - xLoc
                deltaY = yRandom - yLoc
                
                if (radius > CalculateVectorLength(deltaX, deltaY, 0)):
                    zAboveCenter = math.sqrt(radius**2 - (deltaX**2 + deltaY**2))
                    #print("Found with zAboveCenter= %e" % (zAboveCenter))
                    
                    if ((zAboveCenter + zLoc) > zPlain):
                        total += ((zAboveCenter + zLoc) - zPlain)**2
                        hits += 1
        
        if logResults:
            data.log.Log("%2d: %e (for %2d candidates and %5d sample/particle pairs)" % 
                         (i, math.sqrt(total / float(numberSamples)), len(candidates), hits))
            
        if displayToConsole:
            print("%2d: %e (for %2d candidates and %5d sample/particle pairs)" % 
                         (i, math.sqrt(total / float(numberSamples)), len(candidates), hits))
            
        tLog.Log(0, "%2d, %e, %d, %d" % 
                         (i, math.sqrt(total / float(numberSamples)), len(candidates), hits))
        
    tLog.Close()
        
