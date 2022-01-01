# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017
aggDistanceOutBottom
@author: Michelle
editted on 6/21/2021
by Allison Wandling
"""

# \/ \/ \/ \/ \/ \/ \/ \/\/ \/ \/ \/ \/ \/ \/ \/\/ \/ \/ \/ \/ \/ \/ \/ 
# comment out the following line and set availableCpus = 1 in runParameters.txt
#  if you are not using pp.
#import pp
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\

import timeit
import time
import random 
import math
import sys
import os
#import resource      

#from numba import jit
import numpy as np
import copy

from polymer import Polymer
from aggregate import Aggregate     
from readFile import FileParameters  

from holdAggregates import HoldAggregates  
from polymerBins import PolymerBins
from myLogger import MyLogger
from myLogger import LogDestination
from ellipsoid import Ellipsoid
from nearObjectsStatusLine import NearObjectsStatusLine
from calcRoughness import CalculateRoughness

from utilities import CreatePolymerFileName
from utilities import CreateRandomVector
from utilities import CalculateSmallestRadius
from utilities import CalculateRadiusGyration
from utilities import CalculateLargestRadius
from utilities import CalculatePolymerLength
from utilities import CalculateTotalParticle
from utilities import GetParticleIndexesLessThen
from utilities import NormalizeVector
from utilities import RecalculateBoxSize
from utilities import Sec2HMS
from utilities import FormatNumber
from myCsv     import MyCsv
from data import Data 
#############################################################################
def CalcAggregateRadius(parameters, newC):
    volumeAllParticles = newC * (4.0/3.0) * math.pi * (parameters.particleRad ** 3)
    newR = ((volumeAllParticles / parameters.packingFactor) / (math.pi * (4.0/3.0))) ** (1.0/3.0)
    return(newR)
    
###########################################################################
def WriteDump(parameters, data, fileName, displayDebug = True):
    totalMonomers = 0
    
    dumpStart = timeit.default_timer() 
    if (displayDebug): data.log.Log("", LogDestination.CONSOLE)
    if (displayDebug): data.log.Log(("Starting data dump to file %s" % fileName), LogDestination.CONSOLE)
        
    # fill in box with more polymers
    # debug Switch
    if False:
        if (parameters.drying):
            if (parameters.generatePolymers):
                RecalculateBoxSize(parameters, data, True)
                CreateNewPolymers(parameters, data)
    
            print("len(allParticles)= %d"  % (len(data.allParticles)))
            print("total particles= %e"    % (sum(particle.c for particle in data.allParticles)))
            print("len(allPolymers)= %d" % (len(data.allPolymers)))        
            print("parameters.polymersToGenerate= %d" % (parameters.polymersToGenerate))
            print("parameters.boxSize= %d" % (parameters.boxSize))
            
    if (parameters.writeDumpPolymers):
        for i in range(len(data.allPolymers)):
            totalMonomers += data.allPolymers[i].GetNumberOfMonomers(parameters)
            
    if (parameters.writeDumpTangles):
        for i in range(len(data.allEllipsoids)):
            tangleList = list(data.allEllipsoids[i].GetTangleSet())
            for j in tangleList:
                totalMonomers += data.allPolymers[j].GetNumberOfMonomers(parameters)

    #if (displayDebug): print("parameters.monomerRad= %e" %(parameters.monomerRad))
    #if (displayDebug): print("totalMonomers= %d" %(totalMonomers))
    #if (displayDebug): s = ("Packing density by volume of polymers being dumped is %1.5e" % (CalcPackingDensity(parameters.monomerRad, totalMonomers, parameters.boxSize)))
    #if (displayDebug): log.Log(s, LogDestination.CONSOLE)
    #if (displayDebug): s = ("Packing density by volume of particles being dumped is %1.5f" % (CalcPackingDensity(parameters.particleRad, totalParticles, parameters.boxSize)))
    #if (displayDebug): log.Log(s, LogDestination.CONSOLE)
    totalObjects = len(data.allParticles) + totalMonomers
    
    if (displayDebug): print("len(allParticles)= %d"  % (len(data.allParticles)))
    if (displayDebug): print("total particles= %e"    % (sum(particle.c for particle in data.allParticles)))
#    if (displayDebug): print("Number of polymers= %d" % (len(allPolymers)))

    dumpFile=open(fileName,'w')
    dumpFile.write('ITEM: TIMESTEP\n')    
    dumpFile.write(str(parameters.clock)+'\n')    
    dumpFile.write('ITEM: NUMBER OF ATOMS\n')    
    dumpFile.write(str(totalObjects)+'\n')    
    dumpFile.write('ITEM: BOX BOUNDS pp pp ss\n') 
    
    dumpFile.write(str(0)+' '+str(parameters.boxSize)+'\n')    
    dumpFile.write(str(0)+' '+str(parameters.boxSize)+'\n')    
    dumpFile.write(str(0)+' '+str(parameters.boxSize)+'\n')    
    dumpFile.write('ITEM: ATOMS x y z radius\n') 
              
    for j in range(len(data.allParticles)):
        [x, y, z] = data.allParticles[j].GetLocation()  
        dumpFile.write("%20.14e %20.14e %20.14e %16.10e\n" % (x, y, z, data.allParticles[j].r))

    if (parameters.writeDumpPolymers):        
        for i in range(len(data.allPolymers)):
            poly = data.allPolymers[i]
            
            for j in range(parameters.polymerLength):
                if (parameters.polymerLength != poly.GetNumberOfMonomers(parameters)):
                    print("****** WriteDump: Write arg *****")
                    
                [x, y, z] = poly.GetMonomerLocation(j)
                dumpFile.write("%20.14e %20.14e %20.14e %16.10e\n" % (x, y, z, parameters.monomerRad))

    if (parameters.writeDumpTangles):        
        for i in range(len(data.allEllipsoids)):
            tangleList = list(data.allEllipsoids[i].GetTangleSet())
            
            for j in tangleList:
                poly = data.allPolymers[j]
                
                for j in range(parameters.polymerLength):
                    if (parameters.polymerLength != poly.GetNumberOfMonomers(parameters)):
                        print("****** WriteDump: Write arg *****")
                        
                    [x, y, z] = poly.GetMonomerLocation(j)
                    dumpFile.write("%20.14e %20.14e %20.14e %16.10e\n" % (x, y, z, parameters.monomerRad))
      
    dumpFile.close()
    
    dumpStop = timeit.default_timer()
    if (displayDebug): data.log.Log(("Dumped file in %s H:M:S." % (Sec2HMS(dumpStop - dumpStart))), LogDestination.CONSOLE)

    
###############################################################################
def CalcDistance(i, j):
    [x1, y1, z1] = i.GetLocation()
    [x2, y2, z2] = j.GetLocation()
    distance=math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return(distance)
              
###############################################################################
def CalcSumParcelRadiusMatrix(parameters, allParticles,subParticleList, allSalts):
    particle_rLst = np.full(len(subParticleList), 0, dtype = parameters.doMathIn)
    salt_rLst = np.full(len(allSalts), 0, dtype = parameters.doMathIn)
    for k in range(len(subParticleList)):
        particle_rLst[k] = subParticleList[k].r
    for m in range(len(allSalts)):
        salt_rLst[m] = allSalts[m].r
        
    matrix = np.add.outer(particle_rLst, salt_rLst)
    return(matrix)

#used in the VdW matrix
################################################################################
def CalcSumRadiusMatrix(parameters, allParticles):
    particle_rLst = np.full(len(allParticles),0, dtype = parameters.doMathIn)
    for k in range(len(allParticles)):
        particle_rLst = allParticles[k].r
    matrix = np.add.outer(particle_rLst, particle_rLst)
    return(matrix)
################################################################################
def CalcDifferenceRadiusMatrix(parameters,allParticles,subParticleList):
    rLst = np.full(len(subParticleList),0, dtype=parameters.doMathIn)
    
    for k in range(len(subParticleList)):
            rLst[k] = allParticles[subParticleList[k]].r
            
    matrix = np.subtract.outer(rLst, rLst)
    return(matrix)
##############################################################################
def CalcParcelDifferenceRadiusMatrix(parameters, allParticles, subParticleList, allSalts):
    particle_rLst = np.full(len(subParticleList), 0, dtype = parameters.doMathIn)
    salt_rLst = np.full(len(allSalts), 0, dtype = parameters.doMathIn)
    
    for k in range(len(subParticleList)):
        particle_rLst[k] = subParticleList[k].r
    for m in range(len(allSalts)):
        salt_rLst[m] = allSalts[m].r
    
    matrix = np.subtract.outer(particle_rLst, salt_rLst)
    return(matrix)
#used in the VdW matrix

##############################################################################
def CalcProductRadiusMatrix(parameters,allParticles,subParticleList):
    rLst = np.full(len(subParticleList),0,dtype= parameters.doMathIn)
    for k in range(len(subParticleList)):
        rLst[k]= allParticles[subParticleList[k]].r
        
    matrix = np.multiply.outer(rLst,rLst)
    return(matrix)
##############################################################################
def CalcParcelProductRadiusMatrix(parameters, allParticles, subParticleList, allSalts):
    #multiplies the two radii together
    particles_rLst = np.full(len(subParticleList), 0, dtype = parameters.doMathIn)
    #creates an empty list the size of the subParticcleList
    salt_rLst = np.full(len(allSalts), 0, dtype = parameters.doMathIn)   
    
    for k in range(len(subParticleList)):
       particles_rLst[k] = subParticleList[k].r #subParticeList contains indexes of th particles in all Particles that touch the parcel
    for m in range(len(allSalts)):
       salt_rLst[m] = allSalts[m].r
            
    matrix = np.multiply.outer(particles_rLst, salt_rLst)
    return(matrix)
#used in the VdW matrix

##############################################################################
def CalcRadDivision(parameters,allParticles):
    sumRadiusMatrix = CalcSumRadiusMatrix(parameters,allParticles)
    productRadiusMatrix = CalcProductRadiusMatrix(parameters,allParticles)
    
    radDivisionMatrix = np.divide(productRadiusMatrix, sumRadiusMatrix)
    return(radDivisionMatrix)
##############################################################################
def CalcRadParcelDivision(parameters, allParticles, allSalts):
        #(ri*rj)/(ri+rj)
    sumRadiusMatrix = CalcSumParcelRadiusMatrix(parameters, allParticles, allSalts)
    productRadiusMatrix = CalcParcelProductRadiusMatrix(parameters, allParticles, allSalts)
        
    radDivisionMatrix = np.divide(productRadiusMatrix,sumRadiusMatrix)
    return(radDivisionMatrix)

   
###############################################################################
def CalcSSParcelDistanceMatrix(parameters, allParticles, allSalts):   
    partRadSum = CalcSumParcelRadiusMatrix(parameters, allParticles, allSalts)    
    distanceMatrix=  CalcParcelDistanceMatrix(parameters, allParticles, allSalts)    
    gapMatrix = np.subtract(distanceMatrix, partRadSum)
    return(gapMatrix)
    
###############################################################################
def CalcSSDistanceMatrix(parameters,allParticles):
    partRadSum =  CalcSumRadiusMatrix(parameters,allParticles)
    distanceMatrix = CalcDistanceMatrix(parameters,allParticles)
    gapMatrix = np.subtract(distanceMatrix,partRadSum)
    return(gapMatrix)
###############################################################################
def CalcDistanceMatrix(parameters, allParticles):   
    # do a quick check to see if any nudging needs to be done
    # this is generally the case.... saves a lot of time
    xLst = np.full(len(allParticles), 0.0, dtype=parameters.doMathIn)
    yLst = np.full(len(allParticles), 0.0, dtype=parameters.doMathIn)
    zLst = np.full(len(allParticles), 0.0, dtype=parameters.doMathIn)
    
    for k in range(len(allParticles)):
        [xLst[k], yLst[k], zLst[k]] = allParticles[k].GetLocation()
        
    xSqDistMatrix = np.square(np.subtract.outer(xLst, xLst))
    ySqDistMatrix = np.square(np.subtract.outer(yLst, yLst))
    zSqDistMatrix = np.square(np.subtract.outer(zLst, zLst))
            
    distanceMatrix=   np.sqrt(np.add(xSqDistMatrix, np.add(ySqDistMatrix, zSqDistMatrix))) 
    return(distanceMatrix)
###############################################################################
def CalcParcelDistanceMatrix(parameters, allParticles,allSalts):
    # particlesinParcel is allParticles
    xLst = np.full(len(allParticles), 0.0, dtype=parameters.doMathIn)
    yLst = np.full(len(allParticles), 0.0, dtype=parameters.doMathIn)
    zLst = np.full(len(allParticles), 0.0, dtype=parameters.doMathIn)
    s_xLst = np.full(len(allSalts), 0.0, dtype = parameters.doMathIn)
    #salts xLst
    s_yLst = np.full(len(allSalts), 0.0, dtype = parameters.doMathIn)
    #salts yLst
    s_zLst = np.full(len(allSalts), 0.0, dtype = parameters.doMathIn)
    #salts zLst
    
    for k in range(len(allParticles)):
        [xLst[k], yLst[k], zLst[k]] = allParticles[k].GetLocation()
    
    for m in range(len(allSalts)):
        [s_xLst[m], s_yLst[m], s_zLst[m]] = allSalts[m].GetLocation()
        
    xSqDistMatrix = np.square(np.subtract.outer(xLst, s_xLst))
    ySqDistMatrix = np.square(np.subtract.outer(yLst, s_yLst))
    zSqDistMatrix = np.square(np.subtract.outer(zLst, s_zLst))
            
    distanceMatrix=   np.sqrt(np.add(xSqDistMatrix, np.add(ySqDistMatrix, zSqDistMatrix))) 
    return(distanceMatrix) 
###############################################################################
def CalcNaturalLogMatrix(parameters, allParticles, allSalts):
    #ln(1+e^(-kH))
    SSDistanceMatrix = CalcSSParcelDistanceMatrix(parameters, allParticles, allSalts)
    kappa = calcDebyeLength(parameters)
    negativekappa = -1*kappa
    kappaSSDistanceMatrix = np.multiply(negativekappa,SSDistanceMatrix)
    #because kH is too big for exp function multiply kH by 1e-6
    kappaSSDistanceMatrix = np.multiply(kappaSSDistanceMatrix, 1e-6)
    exponentiated = np.exp(kappaSSDistanceMatrix, dtype = np.float64)
    #because kH*1e-6 the one needs to be changed to 1**1e-6
    raisedOne = 1**(1e-6)
    plusOne = np.add(raisedOne,exponentiated) 
    NaturalLogMatrix = np.log(plusOne)
    
    return(NaturalLogMatrix)
#################################################################################
def calcDebyeLength(parameters):
    #calculates the dybe length of the particles based on the salt concentration
    Ef = parameters.Ef #dielectric constant of water
    Eo = parameters.Eo #vaccum permitivity
    boltz = parameters.boltz
    temp= parameters.temp #in Kelvin
    Fe_ions = parameters.Fe_ions
    Fe_charge = parameters.Fe_charge
    Cl_ions = parameters.Cl_ions
    Cl_charge = parameters.Cl_charge
    electronCharge = parameters.electronCharge
    Fe_and_Cl_Sum = ((Fe_ions*(Fe_charge**2)*electronCharge**2)+(Cl_ions*(Cl_charge**2)*electronCharge**2))
    Concentration = parameters.Concentration #mmol/L
    Concentration_Converted = Concentration*1000*6.02E23/1000 #mmol/L to mol/m^3
    debyeLength = ((Ef*Eo*boltz*temp)/(Fe_and_Cl_Sum*Concentration_Converted))**(1/2)
    kappa = 1/debyeLength
    return(kappa)

################################################################################
def EDLConstants(parameters): #look up how the salt concentration influences surface potential
    surfacePotential = -41.63*math.log(parameters.pH)+52.75 
    #function to calculate the zeta potential in mV
    surfacePotentialV = surfacePotential*1e-3    
    EDLConstants = ((4*math.pi*parameters.Ef*parameters.Eo*(surfacePotential)**2)*1e-6)
    return(EDLConstants)

##############################################################################
def EDLMatrix (parameters, subParticleList, allSalts):
    #completing the full EDL equation 
    #only need to worry about kr>5 because the concentration would have to be
    #unlikely small for kr<5
    NaturalLogMatrix = CalcNaturalLogMatrix(parameters, subParticleList, allSalts)
    radDivisionMatrix = CalcRadDivisionMatrix(parameters, subParticleList, allSalts)
    radDiviisionMatrix = np.multiply(radDivisionMatrix,1e-6) #because whole equation is multiplied by 1e-6
    NaturalLogAndRad =  np.multiply(NaturalLogMatrix,radDivisionMatrix)
    EDLConstant = EDLConstants(parameters)
    EDLConstant = EDLConstant*1e-6
    #equation is multiplied by 1e-6 because of kH
    EDLMatrix= np.multiply(EDLConstant,NaturalLogAndRad)
    EDLMatrix = np.multiply(EDLMatrix,1e6) #makes all the equation normal again
    #gives an answer in V #the more positive the number then the less likely to agglomerate
    
    return(EDLMatrix)    
    
###############################################################################
def CalcOverlapParticles(parameters, allParticles, minimumOverlapGap, includeUpperTriangle = True):    
    gapMatrix = CalcSSDistanceMatrix(parameters, allParticles)
    
    if includeUpperTriangle:
        np.fill_diagonal(gapMatrix, 1.0)      # make self very far apart so they don't show up in results (1 meter should be enough)
    else:
        upperIndexes = np.triu_indices(len(allParticles))
        gapMatrix[upperIndexes] = 1.0         # diagnal and semmetric indexes should not be considered
        
    x = np.where(gapMatrix <= minimumOverlapGap)
    return(x)

###############################################################################
def CalcOverlapParticleParcelPair(parameters, data):
    allParticles = data.allParticles
    allParcels = data.allParcels
        
    if (len(allParcels) == 0):
        return([[], []])
    
    if False:
        x = [[], []] #[0] = particles, [1] = parcel
        
        for i in range(len(allParticles)):
            for j in range(len(allParcels)):
                [x1, y1, z1] = allParticles[i].GetLocation()
                [x2, y2, z2] = allParcels[j].GetLocation()
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                twoRadius = allParticles[i].GetRadius() + allParcels[j].GetRadius()
                
                if (distance < twoRadius):
                    x[0].append(i)
                    x[1].append(j)
    else:
        xPLst = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
        yPLst = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
        zPLst = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
        pRadius = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
    
        for k in range(len(allParticles)):
            [xPLst[k], yPLst[k], zPLst[k]] = allParticles[k].GetLocation()
            pRadius[k] = allParticles[k].GetRadius()
            
        xELst = np.full(len(allParcels), 0, dtype=parameters.doMathIn)
        yELst = np.full(len(allParcels), 0, dtype=parameters.doMathIn)
        zELst = np.full(len(allParcels), 0, dtype=parameters.doMathIn)
        eRadius = np.full(len(allParcels), 0, dtype=parameters.doMathIn)
    
        for k in range(len(allParcels)):
            [xELst[k], yELst[k], zELst[k]] = allParcels[k].GetLocation()
            eRadius[k] = allParcels[k].GetRadius()

        xSqDistMatrix = np.square(np.subtract.outer(xPLst, xELst))
        ySqDistMatrix = np.square(np.subtract.outer(yPLst, yELst))
        zSqDistMatrix = np.square(np.subtract.outer(zPLst, zELst))
            
        distanceMatrix = np.sqrt(np.add(xSqDistMatrix, np.add(ySqDistMatrix, zSqDistMatrix))) 
        radiusSum = np.add.outer(pRadius, eRadius)
    
        gapMatrix = np.subtract(distanceMatrix, radiusSum)
        x = np.where(gapMatrix <= 0) #where parcel and particle are touching
        print(x)
    return(x)
    
###############################################################################
def CalcOverlapParticleEllipsoidPair(parameters, data):
    allParticles = data.allParticles
    allEllipsoids = data.allEllipsoids
        
    if (len(allEllipsoids) == 0):
        return([[], []])
    
    if False:
        x = [[], []] #[0] = particles, [1] = ellipsoid
        
        for i in range(len(allParticles)):
            for j in range(len(allEllipsoids)):
                [x1, y1, z1] = allParticles[i].GetLocation()
                [x2, y2, z2] = allEllipsoids[j].GetLocation()
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                twoRadius = allParticles[i].GetRadius() + allEllipsoids[j].GetRadius()
                
                if (distance < twoRadius):
                    x[0].append(i)
                    x[1].append(j)
    else:
        xPLst = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
        yPLst = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
        zPLst = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
        pRadius = np.full(len(allParticles), 0, dtype=parameters.doMathIn)
    
        for k in range(len(allParticles)):
            [xPLst[k], yPLst[k], zPLst[k]] = allParticles[k].GetLocation()
            pRadius[k] = allParticles[k].GetRadius()
            
        xELst = np.full(len(allEllipsoids), 0, dtype=parameters.doMathIn)
        yELst = np.full(len(allEllipsoids), 0, dtype=parameters.doMathIn)
        zELst = np.full(len(allEllipsoids), 0, dtype=parameters.doMathIn)
        eRadius = np.full(len(allEllipsoids), 0, dtype=parameters.doMathIn)
    
        for k in range(len(allEllipsoids)):
            [xELst[k], yELst[k], zELst[k]] = allEllipsoids[k].GetLocation()
            eRadius[k] = allEllipsoids[k].GetRadius()

        xSqDistMatrix = np.square(np.subtract.outer(xPLst, xELst))
        ySqDistMatrix = np.square(np.subtract.outer(yPLst, yELst))
        zSqDistMatrix = np.square(np.subtract.outer(zPLst, zELst))
            
        distanceMatrix = np.sqrt(np.add(xSqDistMatrix, np.add(ySqDistMatrix, zSqDistMatrix))) 
        radiusSum = np.add.outer(pRadius, eRadius)
    
        gapMatrix = np.subtract(distanceMatrix, radiusSum)
        x = np.where(gapMatrix <= 0)
        
    return(x)
    
###############################################################################
def NudgeOneParticlePair(parameters, data, nudgeDistance, minimumOverlapGap, i, j, vector, redo, iRedo, nudges): 
    allParticles = data.allParticles
    failed = False
    maxRetries = 100
    retries = 0
    [xDistance, yDistance, zDistance] = vector
    
    # make initial vector moving small particle away from larger particle
    [x,  y,  z]  = allParticles[i].GetLocation()
    [xj, yj, zj] = allParticles[j].GetLocation()
                            
    overlapped = True
    
    while ((overlapped == True) & (failed == False)):
        overlapped = False  # assume all will be good
        
        # Now check that we moved it enough
        # Since i is the one moving only check all other particles
        distance = CalcDistance(allParticles[i], allParticles[j])
             
        ## Check to see if particals overlap... If so move it come more
        if ((distance - (allParticles[i].r + allParticles[j].r)) <= minimumOverlapGap):
            # move the partical
            x += xDistance
            y += yDistance
            z += zDistance
            allParticles[i].SetLocation(parameters, x, y, z)
            #allParticles[i].SetRandomLocation(parameters)
            #print("Moving in vector. I= %d, j= %d" % (i, j))
            nudges += 1
            redo = True
            overlapped = True;
                
        if (allParticles[i].isOutOfBox(parameters)):
            allParticles[i].SetRandomLocation(parameters)
            [x, y, z] = allParticles[i].GetLocation()
            vector = CreateRandomVector(parameters, nudgeDistance)
            [xDistance, yDistance, zDistance] = vector
            redo = iRedo = True
            overlapped = False
            retries += 1
            if (retries > maxRetries): failed = True
 
    return(redo, iRedo, nudges, failed, vector)
    
###############################################################################
def NudgeOneEllipsoidParticlesPair(parameters, data, nudgeDistance, i, j, redo, nudges): 
    allParticles = data.allParticles
    allEllipsoids = data.allEllipsoids
    failed = False
    maxRetries = 100
    retries = 0
    
    # make initial vector moving small particle away from larger particle
    [x,  y,  z]  = allParticles[i].GetLocation()
    [xj, yj, zj] = allEllipsoids[j].GetLocation()
    [xDistance, yDistance, zDistance] = NormalizeVector(nudgeDistance, x - xj, y - yj, z - zj)
                            
    overlapped = True
    
    while ((overlapped == True) & (failed == False)):
        overlapped = False  # assume all will be good
        
        ## Check to see if particals overlap... If so move it come more
        if (allEllipsoids[j].IsInEllipsoid(allParticles[i])):
            # move the partical
            x += xDistance
            y += yDistance
            z += zDistance
            allParticles[i].SetLocation(parameters, x, y, z)
            nudges += 1
            redo = True
            overlapped = True;
                
        if (allParticles[i].isOutOfBox(parameters)):
            allParticles[i].SetRandomLocation(parameters)
            [x, y, z] = allParticles[i].GetLocation()
            [xDistance, yDistance, zDistance] = CreateRandomVector(parameters, nudgeDistance)
            redo = True
            overlapped = False
            retries += 1
            if (retries > maxRetries): failed = True
 
    return(redo, nudges, failed)
    
###############################################################################
# nudgeDistance is the distance that a particle should be moved... if it has to move
# checks to see if particals overlap.  If they do then nudge them to unoverlap them
def NudgeParticles(parameters, data, nudgeDistance, minimumOverlapGap, printNudgeStatus = False): 
    if parameters.graphForcesTest:
        return(0, False)
    
    nudges = 0
    startNudgeStatus = time.time()
    lastNudgeStatus = startNudgeStatus
    nextNudgeStatus = startNudgeStatus + parameters.statInterval
    failed = False
    vectorUsed = []
    iteration = 0
    
    allParticles = data.allParticles
    redo = True
    
    startTimer = timeit.default_timer()
    
    while (redo & (failed == False)):
        redo = False
        
        # Check for overlaping particles
        overlappingElements = CalcOverlapParticles(parameters, allParticles, minimumOverlapGap, False)        
        p1List = overlappingElements[0]
           
        #if number of elements == 0 then nothing to nudge
        if (len(p1List) != 0):           
            if True:
            #if (parameters.drying == False):
                for mm in range(len(p1List)):
                    i = p1List[mm]  
                    j = overlappingElements[1][mm] 
                           
                    # make initial vector moving small particle away from larger particle
                    [xi, yi, zi]  = allParticles[i].GetLocation()
                    [xj, yj, zj] = allParticles[j].GetLocation()
                    ci  = allParticles[i].GetParticleCount()
                    cj = allParticles[j].GetParticleCount()
                    nudgeIt = max(nudgeDistance, min(allParticles[i].GetRadius(), allParticles[j].GetRadius()) / 4.0)
                    
                    cTotal = float(ci + cj)
                    [xDistancei, yDistancei, zDistancei] = NormalizeVector(nudgeIt * (float(cj) / cTotal), xi - xj, yi - yj, zi - zj)
                    [xDistancej, yDistancej, zDistancej] = NormalizeVector(nudgeIt * (float(ci) / cTotal), xj - xi, yj - yi, zj - zi)
                    
                    xi += xDistancei
                    yi += yDistancei
                    zi += zDistancei
                    xj += xDistancej
                    yj += yDistancej
                    zj += zDistancej
                    
                    allParticles[i].SetLocation(parameters, xi, yi, zi)
                    allParticles[j].SetLocation(parameters, xj, yj, zj)
                    redo = True
                    iteration += 1
                            
                    if (nextNudgeStatus < time.time()):
                        deltaNudgeStatus = time.time() - lastNudgeStatus
                        lastNudgeStatus = time.time()
                        print("NudgeParticles: %s(%+3d H:M:S i= %d, j= %d" % 
                                                  (Sec2HMS(time.time() - startNudgeStatus), deltaNudgeStatus, i, j))
                        nextNudgeStatus = lastNudgeStatus + parameters.statInterval
                        [xi, yi, zi] = allParticles[i].GetLocation()
                        [xj, yj, zj] = allParticles[j].GetLocation()
                        distance = CalcDistance(allParticles[i], allParticles[j])
                        print("NudgeParticles:   xi= %7.1e, yi= %7.1e, zi= %7.1e" % (xi, yi, zi))
                        print("NudgeParticles:   xj= %7.1e, yj= %7.1e, zj= %7.1e" % (xj, yj, zj))
                        print("NudgeParticles:   xi_v= %7.1e, yi_v= %7.1e, zi_v= %7.1e" % (xDistancei, yDistancei, zDistancei))
                        print("NudgeParticles:   xj_v= %7.1e, yj_v= %7.1e, zj_v= %7.1e" % (xDistancej, yDistancej, zDistancej))
                        print("NudgeParticles:   nudgeDistance= %7.1e, distance= %7.1f, it= %d" % (nudgeDistance, distance, iteration))
                        LogToCsv(parameters, data, -1, True, True)  
                        WriteDump(parameters, data, (parameters.fileNameBase + "_NudgeParticles_%04d.dump" % (int(parameters.clock))))
            else:
                #print(p1List)
                
                # Only particles in overlappingElements need to be nudged someware
                # If it's not overlapping it doesn't need to be played with.
                for mm in range(len(p1List)):
                    i = p1List[mm]                               
        
                    vectorUsed = CreateRandomVector(parameters, nudgeDistance)
                    
                    nn = 0
                    iRedo = True
                    
                    while iRedo:
                        iRedo = False
                        
                        for nn in range(len(allParticles)): 
                            j = nn
                            nn += 1
                            
                            if (nextNudgeStatus < time.time()):
                                deltaNudgeStatus = time.time() - lastNudgeStatus
                                lastNudgeStatus = time.time()
                                print("NudgeParticles: %s(%+3d H:M:S i= %d, j= %d" % 
                                                          (Sec2HMS(time.time() - startNudgeStatus), deltaNudgeStatus, i, j))
                                nextNudgeStatus = lastNudgeStatus + parameters.statInterval
                                [xi, yi, zi] = allParticles[i].GetLocation()
                                [xj, yj, zj] = allParticles[j].GetLocation()
                                distance = CalcDistance(allParticles[i], allParticles[j])
                                print("NudgeParticles:   xi= %7.1e, yi= %7.1e, zi= %7.1e" % (xi, yi, zi))
                                print("NudgeParticles:   xj= %7.1e, yj= %7.1e, zj= %7.1e" % (xj, yj, zj))
                                print("NudgeParticles:   xv= %7.1e, yv= %7.1e, zv= %7.1e" % (vectorUsed[0], vectorUsed[1], vectorUsed[2]))
                                print("NudgeParticles:   nudgeDistance= %7.1e, distance= %7.1f, it= %d" % (nudgeDistance, distance, iteration))
                                LogToCsv(parameters, data, -1, True, True)  
                                WriteDump(parameters, data, (parameters.fileNameBase + "_NudgeParticles.dump"))
                       
                            # don't process self against self
                            if (i != j):                                                                                  
                                [redo, iRedo, nudges, failed, vectorUsed] = NudgeOneParticlePair(parameters, data, nudgeDistance, minimumOverlapGap, i, j, vectorUsed, redo, iRedo, nudges)
                                
                            if failed: 
                                data.log.Log("****\nNudging failed\n****")
                                break
                               
            if failed == False:
                # Check for overlapping Particles to Ellipsoids
                # returns:
                #   overlappingElements[0][:] are the particles
                #   overlappingElements[1][:] are the ellipsoids
                overlappingElements = CalcOverlapParticleEllipsoidPair(parameters, data)
                
                #if number of elements == 0 then nothing to nudge
                if (len(overlappingElements[0]) != 0): 
                    
                    # Only particles in overlappingElements need to be nudged someware
                    # If it's not overlapping it doesn't need to be played with.
                    # overlap between i = particle & j = ellipsoid
                    for mm in range(len(overlappingElements[0])):
                        i = overlappingElements[0][mm]
                        j = overlappingElements[1][mm]                              
                        
                        [redo, nudges, failed] = NudgeOneEllipsoidParticlesPair(parameters, data, nudgeDistance, i, j, redo, nudges)
                        if failed: 
                            data.Log("****\nNudging failed\n****")
                            break
                        
    parameters.timeInNudgeParticles += timeit.default_timer() - startTimer
    
    return(nudges, failed)
#############################################################################
def CreateParticles(parameters, data):
    if parameters.graphForcesTest:
        agg = Aggregate(parameters.particleRad)
        agg.SetLocation(parameters, 0, 0, 0)
        data.allParticles.append(agg)
        x = (parameters.particleRad + parameters.stericLayerThickness) * 2 + parameters.spacing
    
    for index in range(parameters.particleCount):
        agg = Aggregate(parameters.particleRad)
        
        if parameters.graphForcesTest == False:
            agg.SetRandomLocation(parameters)
        else:
            agg.SetLocation(parameters, x, 0, 0)
            x += parameters.spacing
            
        data.allParticles.append(agg)
       
    if parameters.graphForcesTest == False:
        RecalculateBoxSize(parameters, data, False)
    #[nudges, failed] = NudgeParticles(parameters, data, parameters.particleRad * .5, parameters.particleRad * .25)
        
    data.log.Log("")
    data.log.Log("Initialized particles.")
    
###############################################################################
def ParcelRad(parameters):
    parcelRad = ((.64*parameters.boxSize*3)/(4*parameters.numParcels*math.pi))**(1/3)
    return(parcelRad)

##################################################################################
def CreateParcels(parameters,data):
    #determine size of the parcels
    #.64(TotalVolume) = Np(4/3*pi*R^3)
    #r = ((.64*totalVolume*3)/(4Np*pi))^1/3
    parcelRad = ((.64*parameters.boxSize*3)/(4*parameters.numParcels*math.pi))**(1/3)
    #should i take the boxSize from the parameters #make sure it should be numParcels
    
    if parameters.graphForcesTest:
        agg = Aggregate(parcelRad)
        agg.SetLocation(parameters, 0, 0, 0)
        data.allParcels.append(agg)
        x = (parcelRad + parameters.stericLayerThickness) * 2 + parameters.spacing
    
    for index in range(parameters.numParcels):
        agg = Aggregate(parcelRad)
        
        if parameters.graphForcesTest == False:
            agg.SetRandomLocation(parameters)
        else:
            agg.SetLocation(parameters, x, 0, 0)
            x += parameters.spacing
            
        data.allParcels.append(agg)
       
    if parameters.graphForcesTest == False:
        RecalculateBoxSize(parameters, data, False)
    #[nudges, failed] = NudgeParticles(parameters, data, parameters.particleRad * .5, parameters.particleRad * .25)
        
    data.log.Log("")
    data.log.Log("Initialized parcels.")


#################################################################################
def CreateSalts(parametes,data, location, parcelRad):
    
   
    concentration = parameters.Concentration*6.02E23/1000 
    #concentration converted from mol/L to molecules/m^3
    boxSize = parameters.boxSize
    totalParticles = concentration*boxSize
    
    
    particlesPerParcel = totalParticles/parameters.numParcels
    #totalParticles = Fe + Cl
    #Cl = 3Fe becuase FeCl3
    #totalParticles = 4Fe
    
    Fe_Num = math.trunc(particlesPerParcel/4) 
    #math.trunc removes the decimal place
    Cl_Num = math.trunc(particlesPerParcel-Fe_Num)
    
    parcelRad = parcelRad = ((.64*parameters.boxSize*3)/(4*parameters.numParcels*math.pi))**(1/3) #needed for salt location inside parcel
    
    if parameters.graphForcesTest:
        x = (parameters.FeRad + parameters.stericLayerThickness) * 2 + parameters.spacing
    
    for index in range(Fe_Num):
        agg = Aggregate(parameters.FeRad)
        #salt particles will have an assumed raidus of 0 because so much smaller than the plastic particles
        
        if parameters.graphForcesTest == False:
            agg.SetRandomRangedLocation(parameters,location,parcelRad) #location comes from where the parcel is created because particles are inside parcel
        else:
            agg.SetLocation(parameters, x, 0, 0)
            #based on parcel volume
            x+= parameters.spacing
            
        data.allSalts.append(agg)
     
    for index in range(Cl_Num):
        agg = Aggregate(parameters.ClRad)
        #salt particles will have an assumed raidus of 0 because so much smaller than the plastic particles
        
        if parameters.graphForcesTest == False:
            agg.SetRandomRangedLocation(parameters,location, parcelRad) #location comes from where the parcel is created because particles are inside parcel
        else:
            agg.SetLocation(parameters, x, 0, 0)
            #based on parcel volume
            x+= parameters.spacing
            
        data.allSalts.append(agg)
       
    if parameters.graphForcesTest == False:
        RecalculateBoxSize(parameters, data, False)
    #[nudges, failed] = NudgeParticles(parameters, data, parameters.particleRad * .5, parameters.particleRad * .25)
        
    #data.log.Log(" ")
    #data.log.Log("Initialized Salts")
        
  
    
###############################################################################
        
    
#############################################################################
def CreateBasePolymers(parameters, data, onlyBase): 
    allPolymers = data.allPolymers
    allParticles = data.allParticles
    
    #print("CreateBasePolymers entered")
    start = timeit.default_timer()
    showNext=125
    polymersDone = len(allPolymers)
    statusInterval = 1000  # interval between polymer count
        
    totalParticles = CalculateTotalParticle(parameters, allParticles)
    parameters.polymersToGenerate = int(totalParticles * parameters.chainsPerParticle)
    if (parameters.generateBasePolymerCount > parameters.polymersToGenerate):
        parameters.generateBasePolymerCount = parameters.polymersToGenerate
    created = 0
     
    polymersToGenerate = parameters.polymersToGenerate
    
    if onlyBase:
        polymersToGenerate = min(polymersToGenerate, parameters.generateBasePolymerCount)
    
    # didn't want to toss old code.... So True if for non-multiprocessing
    while polymersDone < polymersToGenerate:
        polymer = Polymer(parameters, 0, 0, 0)
        
        if (len(allPolymers) >= parameters.generateBasePolymerCount):
            polymer = allPolymers[random.randint(0, parameters.generateBasePolymerCount - 1)].Clone(parameters)  
            polymer.SetRandomLocation(parameters)
        else:
            polymer.SetBasePolymerIndex(polymersDone)
            polymer.SetRandomLocation(parameters)
            polymer.SetDebugPrintStatus(polymersDone == 0)
            polymer.CreatePolymerChain(parameters)
                
        #polymer.SetTangleSet({polymersDone})
        data.polymerBins.AddToBin(parameters, polymer, polymersDone)
        
        # if the fail flag is True then something went wrong and this
        # polymer is invalid (needs to be regenerated)
        # If valid polymer then save it with the others
        allPolymers.append(polymer)
        created += 1
        polymersDone += 1
        parameters.maxPolymerRadius = max(parameters.maxPolymerRadius, polymer.GetTangleRadius())
        
        if (polymersDone == parameters.generateBasePolymerCount):
            xyz = np.full((3, parameters.generateBasePolymerCount, parameters.polymerLength), 0, dtype=parameters.doMathIn)
            
            for j in range(parameters.generateBasePolymerCount):
                xyz[0][j] = allPolymers[j].GetNpArray('x')
                xyz[1][j] = allPolymers[j].GetNpArray('y')
                xyz[2][j] = allPolymers[j].GetNpArray('z')
            
            np.save(CreatePolymerFileName(parameters.tempDir), xyz)
                    
        if ((len(allPolymers) >= showNext) or (polymersDone == polymersToGenerate)):
            currentTime = timeit.default_timer()
            
            outLine = ("%s polymers created in %s H:M:S." % (
                        FormatNumber(len(allPolymers)),
                        Sec2HMS(currentTime - start)))
                
            if (len(allPolymers) >= statusInterval):
                timePerPolymer = (currentTime - start) / polymersDone
                polymersRemaining = polymersToGenerate - polymersDone
                timeRemaining = int(timePerPolymer * polymersRemaining)
                outLine += ("  Estimated time remaining %s H:M:S." % Sec2HMS(timeRemaining))
                
            if (showNext <= statusInterval):
                showNext = showNext * 2.0
            else:
                showNext = showNext + statusInterval
                        
    #print("CreateBasePolymers exiting")
    return    
#############################################################################
def CreateNewPolymers(parameters, data): 
    
    if ((parameters.polymerVolFrac <= 0) | (parameters.polymersToGenerate == 0)):
        return
    
    if (parameters.generatePolymers == False):
        #CreateBasePolymers(parameters, data, True)
        #CreateNewEllipsoids(parameters, data)
        return
      
    CreateBasePolymers(parameters, data, False)
    
    return
    
#############################################################################
#def CreateNewEllipsoids(parameters, data, displayPrints = False):
#    if parameters.generatePolymers:
#        return
#    
#    requiredEllipsoids = CalcRequiredEllipsoids(parameters)
#    
#    for i in range(len(data.allEllipsoids), requiredEllipsoids):
#        AddEllipsoidToListNonDrying(parameters, data)
#        
#    return
#
###########################################################################
def WriteDumpEllipsoid(parameters, fileName, polymerList, particleList, box):
    totalMonomers = 0
    
    dumpStart = timeit.default_timer() 
        
            
    for i in range(len(polymerList)):
        totalMonomers += polymerList[i].GetNumberOfMonomers(parameters)

    totalObjects = len(particleList) + totalMonomers
    #totalObjects = len(particleList)
    
    dumpFile=open(fileName,'w')
    dumpFile.write('ITEM: TIMESTEP\n')    
    dumpFile.write(str(parameters.clock)+'\n')    
    dumpFile.write('ITEM: NUMBER OF ATOMS\n')    
    dumpFile.write(str(totalObjects)+'\n')    
    dumpFile.write('ITEM: BOX BOUNDS pp pp ss\n') 
    
    dumpFile.write(str(box[0][0])+' '+str(box[0][1])+'\n')    
    dumpFile.write(str(box[1][0])+' '+str(box[1][1])+'\n')    
    dumpFile.write(str(box[2][0])+' '+str(box[2][1])+'\n')    
    dumpFile.write('ITEM: ATOMS x y z radius\n') 

    for i in range(len(polymerList)):
        poly = polymerList[i]
        
        for j in range(parameters.polymerLength):
            [x, y, z] = poly.GetMonomerLocation(j)
            dumpFile.write(str(x)+' '+str(y)+' '+str(z)+' '+str(parameters.monomerRad)+'\n')
              
    for j in range(len(particleList)):
        [x, y, z] = particleList[j].GetLocation()    
        dumpFile.write(str(x)+' '+str(y)+' '+str(z)+' '+str(particleList[j].r)+'\n')
      
    dumpFile.close()
    
    dumpStop = timeit.default_timer()
    print("Dumped file in %s H:M:S." % (Sec2HMS(dumpStop - dumpStart))) 
    
    return
    
###############################################################################
def DumpElipsoid(parameters, data, ellipsoid):
    
    data.downEscapiesNotUsed = HoldAggregates(1)
    
    print("*********\nStart data dump:")
    [x,y,z] = ellipsoid.GetCenterOfMassLocation()
    print("Center of Mass x= %e, y= %e, z= %e" % (x, y, z))
    print("Box Size= %e" % (parameters.boxSize))
    r = ellipsoid.GetRadius()
    print("Rough spherical radius= %e" % (r))
    ellipsoid.PrintQs()
    
    bins = 30
    binSize = r / bins
    width = 3 * bins
    particleList = []
    polymerList = []
    tangleSet = ellipsoid.GetTangleSet()
    
    for i in tangleSet:
        polymerList.append(data.allPolymers[i])
        
    print("Generating ellipsoid visulation data")
    for i in range(width):
        for j in range(width):
            for k in range(width):
                xLoc = ((i - (width / 2.0)) * binSize) + x
                yLoc = ((j - (width / 2.0)) * binSize) + y
                zLoc = ((k - (width / 2.0)) * binSize) + z
                aggregate = Aggregate(binSize / 20.0)
                aggregate.SetLocation(parameters, xLoc, yLoc, zLoc)
                
                if (ellipsoid.IsInEllipsoid(aggregate)):
                    #print("True at x= %e, y= %e, z= %e" % (xLoc, yLoc, zLoc))
                    particleList.append(aggregate)
                    
    monomersIn = 0
    monomerCount = 0
    print("Generating polymer in ellipsoid percentage")
    
    for i in tangleSet:
        monomerCount += data.allPolymers[i].GetNumberOfMonomers(parameters)
        
        for j in range(parameters.polymerLength):
            [x, y, z] = data.allPolymers[i].GetMonomerLocation(j)
            aggregate.SetLocation(1e-15, x, y, z)
                
            if (ellipsoid.IsInEllipsoid(aggregate)):
                monomersIn += 1
            
    print("monomer percent in ellipsoid is %4.1f" % ((monomersIn / monomerCount) * 100))
    
    box = [[0 for x in range(2)] for y in range(3)]
    
    box[0][0] = x - (1.5*r)
    box[0][1] = x + (1.5*r)
    
    box[1][0] = y - (1.5*r)
    box[1][1] = y + (1.5*r)
    
    box[2][0] = z - (1.5*r)
    box[2][1] = z + (1.5*r)
        
    WriteDumpEllipsoid(parameters, "Ellipsoid.dump", polymerList, particleList, box)
    print("End data dump\n*********")
    
    raise ValueError("******\nEllipsoid generated.  See ellipsoid.dump for visulation. program Terminated\n*****")
    return

###############################################################################
#from numba import jit
#@jit(cache=True)
def FilterCandidatePolymerTangles(iteration, tempDir, polymerLength, monomerRad, minimumNumberOfCrossings, generateBasePolymerCount,
                                  elementsFloat, startingLocsFn, baseIndexFn, areCloseFn):
    import os
    import numpy as np
    import timeit
    #from numba import jit
    
    from utilities import CreatePolymerFileName
    
    ###########################################################################
    #@jit()

    def IsTangled2(polymerLength, monomerRad, minimumNumberOfCrossings, 
                   bi, bj, kk, 
                   xyz, startingLocs):  
        tangled = False
        
        if True:                   
            xBaseMonomerLocs1 = xyz[0][bi]
            yBaseMonomerLocs1 = xyz[1][bi]
            zBaseMonomerLocs1 = xyz[2][bi]
            xBaseMonomerLocs2 = xyz[0][bj]
            yBaseMonomerLocs2 = xyz[1][bj]
            zBaseMonomerLocs2 = xyz[1][bj]
    
            xLoc1 = startingLocs[kk][0]
            yLoc1 = startingLocs[kk][1]
            zLoc1 = startingLocs[kk][2]
            xLoc2 = startingLocs[kk][3] 
            yLoc2 = startingLocs[kk][4] 
            zLoc2 = startingLocs[kk][5]
            
            tangled = IsTangled(polymerLength, monomerRad, minimumNumberOfCrossings,  
                             xBaseMonomerLocs1, yBaseMonomerLocs1, zBaseMonomerLocs1,  
                             xBaseMonomerLocs2, yBaseMonomerLocs2, zBaseMonomerLocs2,  
                             xLoc1, yLoc1, zLoc1,  xLoc2, yLoc2, zLoc2)
        return(tangled)
        
    ###########################################################################
    def IsTangled(polymerLength, monomerRad, minimumNumberOfCrossings,  xBaseMonomerLocs1, yBaseMonomerLocs1, zBaseMonomerLocs1,  xBaseMonomerLocs2, yBaseMonomerLocs2, zBaseMonomerLocs2,  xLoc1, yLoc1, zLoc1,  xLoc2, yLoc2, zLoc2):           
        import numpy as np
        mc = polymerLength      
        skipCount = [100, 10, 2]
        
        x1 = np.add(xBaseMonomerLocs1, xLoc1)
        y1 = np.add(yBaseMonomerLocs1, yLoc1)
        z1 = np.add(zBaseMonomerLocs1, zLoc1)
        x2 = np.add(xBaseMonomerLocs2, xLoc2)
        y2 = np.add(yBaseMonomerLocs2, yLoc2)
        z2 = np.add(zBaseMonomerLocs2, zLoc2)
        
        i = 0
        polymersInProximity = True
        
        while ((i < len(skipCount)) & (polymersInProximity == True)):
            sc = skipCount[i]
            
            # Only process if skipCount is less then polymer length
            if (sc <= mc):
                xSqDiff = np.square(np.subtract.outer(x1[0:mc:sc], x2[0:mc:sc]))
                ySqDiff = np.square(np.subtract.outer(y1[0:mc:sc], y2[0:mc:sc]))                
                zSqDiff = np.square(np.subtract.outer(z1[0:mc:sc], z2[0:mc:sc]))
                xyzSqDiff = np.add(xSqDiff, np.add(ySqDiff, zSqDiff))
                xSqDiff = None; ySqDiff = None;  zSqDiff= None# Use memory then free it up 
                
                distanceMatrix = np.sqrt(xyzSqDiff);  
                xyzSqDiff = None # Use memory then free it up 
                
                # if False; Faster, but misses over 50% of tangles...
                # I need to figure out why before Code can become operational
                if True:
                    np.fill_diagonal(distanceMatrix, 1.0)      # don't want self to match
                else:
                    upperIndexes = np.triu_indices(int(mc/sc))
                    distanceMatrix[upperIndexes] = 1.0        # diagnal and semmetric indexes should not be considered
        
                areClose = np.where(distanceMatrix < (monomerRad * ((sc + .25) * 2.0)))
                elements = len(areClose[0])
                
                polymersInProximity = (elements != 0)
                
                i += 1
                
        # If in proximity then process the polymer in detail  
        crossings = 0
        
        if (polymersInProximity & (elements >= minimumNumberOfCrossings)):
             polymerList = areClose[0]
             polymerList.sort()
    
             lastCrossing = polymerList[0]  # We know this value is at a crossing
             crossings = 0
             gapBetweenCrossings = 50       # once crossing detected, the monomer number before another crossing counts
                      
             for i in range(len(polymerList)):
                 if ((abs(polymerList[i] - lastCrossing) > gapBetweenCrossings)):
                     lastCrossing = polymerList[i]
                     crossings += 1
                      
        tangled = (crossings >= minimumNumberOfCrossings)
        
        #if (self.debugPrintsEnabled & (self.debugPrintsEnabled & (tangled))): print("%4d monomers are in close proximity" % elements)            
        return(tangled)
    ###########################################################################
    
    grossTimeStart = timeit.default_timer()    
    computationalTimeStart = timeit.default_timer()
    #print("%2d: Just entered FilterCandidatePolymerTangles" % (int(timeit.default_timer() - grossTimeStart)))
    
    
    if True:
        startingLocs = np.load(startingLocsFn)
        baseIndex = np.load(baseIndexFn)
        areClose = np.load(areCloseFn)
        os.remove(startingLocsFn)
        os.remove(baseIndexFn)
        os.remove(areCloseFn)
        #print("%2d: Files Read" % (int(timeit.default_timer() - grossTimeStart)))
       
        elements = int(elementsFloat + .5)
        
        #print("FilterCandidatePolymerTangles: Elements= ", elements)
        #print("FilterCandidatePolymerTangles: startingLocs.shape= ", startingLocs.shape)
        #print("FilterCandidatePolymerTangles: baseIndex.shape= ", baseIndex.shape)
        #print("FilterCandidatePolymerTangles: areClose.shape= ", areClose.shape)

        xyz = np.load(CreatePolymerFileName(tempDir))
            
        filteredAreClose = []  
    
        for kk in range(elements):
            #if kk >= elements:
            #    print("FilterCandidatePolymerTangles: kk= %d > elements= %d" % (kk, elements))
                
            i = int(areClose[kk][0] + .5)
            j = int(areClose[kk][1] + .5)
            bi = baseIndex[kk][0]
            bj = baseIndex[kk][1]
        
            #if (kk < 2):
            #    print("%2d: kk= %d, i= %d, j= %d, bi= %d, bj= %d" % (int(timeit.default_timer() - grossTimeStart), kk, i, j, bi, bj))
            
            if False:
                try:
                    polymersAreTangled = IsTangled(polymerLength, monomerRad, minimumNumberOfCrossings, 
                                                   xyz[0][bi], xyz[1][bi], xyz[2][bi], 
                                                   xyz[0][bj], xyz[1][bj], xyz[2][bj],
                                                   startingLocs[kk][0], startingLocs[kk][1], startingLocs[kk][2], 
                                                   startingLocs[kk][3], startingLocs[kk][4], startingLocs[kk][5])
                except:
                    print("IsTangled Exception: elements= %d, kk= %d, i= %d, j= %d, bi= %d, bj= %d" % (elements, kk, i, j, bi, bj))
                    raise ValueError("Probably index out of bounds.")
            else:
                polymersAreTangled = IsTangled2(polymerLength, monomerRad, minimumNumberOfCrossings, 
                                               bi, bj, kk, 
                                               xyz, startingLocs)
        
            if (polymersAreTangled):
                # print("i= %4d, j= %4d are tangled" % (i, j))
                filteredAreClose.append([i, j])
    else:
        filteredAreClose = [[1,1], [2,2]]
        print("in Fake Code")
        
    if len(filteredAreClose) == 0: 
        filteredAreClose = [[-1, -1], [-2, -2]]
        
    #print("%2d: Leaving FilterCandidatePolymerTangles. len(filteredAreClose)= %d"% (int(timeit.default_timer() - grossTimeStart), len(filteredAreClose)))
    return(iteration, filteredAreClose, timeit.default_timer() - grossTimeStart, timeit.default_timer() - computationalTimeStart)
   
###############################################################################
#@jit()
def ProcessOneBlock(shifter, mainBinLength, currentWork, partialSummaryOfPolymers):    
    
    #print("ProcessOneBlock: started")
    #print("ProcessOneBlock: mainBinLength= %d, len(partialSummaryOfPolymers)= %d" % (mainBinLength, len(partialSummaryOfPolymers)))
    startTime = timeit.default_timer()
    areClose = []
    areCloseProtoList = [[], []]
    
    if True:
        x1 = partialSummaryOfPolymers[0:mainBinLength, 0]        
        y1 = partialSummaryOfPolymers[0:mainBinLength, 1]        
        z1 = partialSummaryOfPolymers[0:mainBinLength, 2]        
        r1 = partialSummaryOfPolymers[0:mainBinLength, 3]
        
        x2 = partialSummaryOfPolymers[:, 0]        
        y2 = partialSummaryOfPolymers[:, 1]        
        z2 = partialSummaryOfPolymers[:, 2]        
        r2 = partialSummaryOfPolymers[:, 3] 
        
        xDistMatrix = np.subtract.outer(x1, x2)
        yDistMatrix = np.subtract.outer(y1, y2)
        zDistMatrix = np.subtract.outer(z1, z2)
        rMatrix =     np.add.outer(r1, r2)
        
        distanceMatrix=   np.sqrt(np.add( np.square(xDistMatrix), 
                                   np.add(np.square(yDistMatrix), 
                                          np.square(zDistMatrix))))
    
        SSdistanceMatrix = np.subtract(distanceMatrix, rMatrix)
                
        upperIndexes = np.triu_indices(mainBinLength)
        SSdistanceMatrix[upperIndexes] = 1.0        # diagnal and semmetric indexes should not be considered
    
        areCloseProtoList = np.where(SSdistanceMatrix < 0.0)
        #areCloseProtoList = [[],[]]
        
        #for i in range(len(areCloseProtoList1[0])):
        #    if areCloseProtoList1[0][i] > areCloseProtoList1[1][i]:
        #        areCloseProtoList[0].append(areCloseProtoList1[0][i])
        #        areCloseProtoList[1].append(areCloseProtoList1[1][i])
        
        elements = len(areCloseProtoList[0])
        for i in range(elements):
            areClose.append(currentWork[areCloseProtoList[0][i]] + (shifter * currentWork[areCloseProtoList[1][i]]))
        #areClose = [(currentWork[areCloseProtoList[0][i]] + (shifter * currentWork[areCloseProtoList[1][i]])) for i in range(elements)]
            
        #print("ProcessOneBlock: len(x1)= ", len(x1), "len(x2)= ", len(x2))
        #print("ProcessOneBlock: len(areCloseProtoList[0])= ", len(areCloseProtoList[0]))
        #print("ProcessOneBlock: len(areClose)= ", len(areClose))
    else:
        areClose = [1,1]
        
    if (len(areClose) == 0):
        areClose = [-1,-1]
            
    #print("ProcessOneBlock: Ending. len(areClose[0]= ", len(areClose[0]), ", len(areClose[1])= ", len(areClose[1]))
    return(areClose, timeit.default_timer() - startTime)   
         
###############################################################################
#@jit()
def BuildParameters(parameters, allPolymers, fullAreClose, start, end, finalAreClose, baseIndex, locs):    
    #print("arrayLength= %d, start= %d, stop= %d" % (arrayLength, start, stop))
    
    elements = 0
    
    for kk in range(start, end):
        i = int(fullAreClose[kk] % data.shifter)
        j = int(fullAreClose[kk] / data.shifter)
        
        baseIndex[elements]  = [allPolymers[i].monomerChain.basePolymerIndex, allPolymers[j].monomerChain.basePolymerIndex]
        finalAreClose[elements] = [i, j]
        
        locs[elements, 0:3] = allPolymers[i].startingMonomer.zyxLoc
        locs[elements, 3:6] = allPolymers[j].startingMonomer.zyxLoc
        
        #if kk < start + 2:
        #    print("i= %d, j= %d" % (i, j))
    
        elements += 1
                   
    return(elements)
        
###############################################################################
#@jit()
def SecondStageTangleCalculation(parameters, data, nearObjectsStatusLine):
    allPolymers = data.allPolymers
    iteration = 0
    moreJobs = []
    
    baseIndex =     np.full((parameters.processOneBlockMaxSize, 2), 0, dtype="int")
    locs =          np.full((parameters.processOneBlockMaxSize, 6), 0, dtype=parameters.doMathIn)
    finalAreClose = np.full((parameters.processOneBlockMaxSize, 2), 0, dtype="int")
    
    for i in range(0, data.lenNpHoldAreClose, parameters.processOneBlockMaxSize):  
        start = i
        end = i + parameters.processOneBlockMaxSize
        
        if (end > data.lenNpHoldAreClose):
            end = data.lenNpHoldAreClose
            
        #print("start= %d, end=%d, data.lenNpHoldAreClose= %d, parameters.processOneBlockMaxSize= %d" % (start, end, data.lenNpHoldAreClose, parameters.processOneBlockMaxSize))
        if start < end:
            #print("SecondStageTangleCalculation: Before FilterAndBuildParameters len(areClose[0]= ", len(areClose[0]))FilterCandidatePolymerTangles
            elements = BuildParameters(parameters, allPolymers, data.npHoldAreClose, start, end, finalAreClose, baseIndex, locs)
            #print("SecondStageTangleCalculation: After len(areClose[0]= ", len(areClose[0]))
                
            locsFn = ("%s/locs_%04d.npy" % (parameters.tempDir, iteration))
            baseIndexFn = ("%s/baseIndex_%04d.npy" % (parameters.tempDir, iteration))
            areCloseFn = ("%s/areCLose_%04d.npy" % (parameters.tempDir, iteration))
            
            np.save(areCloseFn, finalAreClose)
            np.save(locsFn, locs)
            np.save(baseIndexFn, baseIndex)
            
            #NOTE: areClose goes as a 2xN array and comes out as a Mx2 array.  WHere N >= (M * 2)
            #areCloseFinal = FilterCandidatePolymerTangles(parameters, locs, baseIndex, areClose)
            #print("SecondStageTangleCalculation: moreJobs Qed. len(areClose[0])= ", len(areClose[0]))
            
            if (parameters.availableCpus == 1):
                [myIteration, areCloseFinal, grossTime, computationalTime] = \
                        FilterCandidatePolymerTangles(iteration, parameters.tempDir, parameters.polymerLength, parameters.monomerRad, \
                                parameters.minimumNumberOfCrossings, \
                                parameters.generateBasePolymerCount, elements, locsFn, baseIndexFn, areCloseFn)
                        
                if areCloseFinal[0][0] >= 0:
                    data.finalHoldAreClose += areCloseFinal
            else:
                moreJobs.append(data.job_server.submit(FilterCandidatePolymerTangles, ( \
                                                  iteration, parameters.tempDir, parameters.polymerLength, parameters.monomerRad, parameters.minimumNumberOfCrossings, \
                                                  parameters.generateBasePolymerCount, elements, locsFn, baseIndexFn, areCloseFn)))   
 
            nearObjectsStatusLine.Print(parameters, data, iteration)
            iteration += 1
               
    return(moreJobs)

###############################################################################
def ClearFirstStageJobs(parameters, data, jobs):
    
    for job in jobs:             
        [areClose, grossTime] = job()
        
        if areClose[0] >= 0:
            data.npHoldAreClose[data.lenNpHoldAreClose:(data.lenNpHoldAreClose+len(areClose))] = areClose
            data.lenNpHoldAreClose += len(areClose)
    
    return()
    
###############################################################################
def CalcNearObjects(parameters, data):
    moreJobs = []
    
    #print("Entering CalcNearObjects")
    allPolymers = data.allPolymers
    polymerBins = data.polymerBins
    nearObjectsStatusLine = NearObjectsStatusLine(parameters.statInterval)

    summaryOfPolymers = np.full((4, len(allPolymers)), 0.0, dtype=parameters.doMathIn)
    polymerBins.ResetBinCounters()
    fileName = ("%s_debug_binLens.txt" % (parameters.fileNameBase))            
    polymerBins.SaveBinLen(fileName) 
    
    for j in range(len(allPolymers)):
        summaryOfPolymers[:,j] = allPolymers[j].GetCenterOfMassAndTangleRadius() 
        
        #print("j= ", j, ", allPolymers[j].GetCenterOfMass()", allPolymers[j].GetCenterOfMass())    
    
    #print("Entering CalcNearObjects. Processing %d blocks" % (blocksToProcess))

    #for i in range(len(allPolymers)):         
    #    allPolymers[i].SetTangleSet({i})  # everything start untangled

    # tuple of all parallel python servers to connect with  

    #ppservers = (content[0],content[1],)
    #job_server = pp.Server(ncpus, ppservers=ppservers)
    jobs = []  
    startStage1 = timeit.default_timer()      
    
    while (polymerBins.IsMoreWork()):
        suspendTime = CheckForSuspension(data.log)
        startStage1 += suspendTime

        mainBinLength = polymerBins.GetCurrentBinCount()
        
        if (mainBinLength == 0):
            polymerBins.IncrementIndexes()
        else:            
            currentWork =  polymerBins.GetMoreWork()
            
            try:
                #x = timeit.default_timer()
                partialSummaryOfPolymers = np.array([summaryOfPolymers[:,i] for i in currentWork])
                #print(">>>>>>>> pp data pre time=", timeit.default_timer() - x)
            except:
                raise ValueError("****** Error:\nProblem in creating partialSummaryOfPolymers matrix.\n*****")
                    
            # Overhead of scheduling task is > then work performed.
            if True:
            #if (parameters.availableCpus == 1):
                [areClose, grossTime] = ProcessOneBlock(data.shifter, mainBinLength, currentWork, partialSummaryOfPolymers)
                
                if areClose[0] >= 0:
                    data.npHoldAreClose[data.lenNpHoldAreClose:(data.lenNpHoldAreClose + len(areClose))] = areClose
                    #print(">>data.lenNpHoldAreClose= %d" % (data.lenNpHoldAreClose))
                    #print(areClose[0:5])
                    #print(data.npHoldAreClose[data.lenNpHoldAreClose:(data.lenNpHoldAreClose + 5)])
                    data.lenNpHoldAreClose += len(areClose)
             
            else:
                jobs.append(data.job_server.submit(ProcessOneBlock, (data.shifter, mainBinLength, currentWork, partialSummaryOfPolymers)))
                #print(">>>>>>>> ProcessOneBlock schedule time=", timeit.default_timer() - x)
            
            nearObjectsStatusLine.Print(parameters, data, 0)
            
            depth = 10.0   # depth = keep all CPUs happy with X data packets to crunch.
                        # The larger the depth the more effecient the process... But
                        # the more memory it takes.  This parameter must be tuned to the system it's running on.
            
            if (len(jobs) > (depth * parameters.availableCpus)):
                suspendTime = CheckForSuspension(data.log)
                startStage1 += suspendTime
                ClearFirstStageJobs(parameters, data, jobs)
                jobs = []
            
    ClearFirstStageJobs(parameters, data, jobs)  
    data.maxLenNpHoldAreClose = max(data.maxLenNpHoldAreClose, data.lenNpHoldAreClose)
    
    jobs = []
    data.timeInStageOne += timeit.default_timer() - startStage1
    print("Stage 1 took %s H:M:S. %s items queued up." % (Sec2HMS(timeit.default_timer() - startStage1, True), FormatNumber(data.lenNpHoldAreClose)))
    
    data.tangleCandidates = data.lenNpHoldAreClose

    startStage2 = timeit.default_timer()      
                
    if (data.lenNpHoldAreClose > 0):
        moreJobs = SecondStageTangleCalculation(parameters, data, nearObjectsStatusLine)
        
    print("Stage 2 took %s H:M:S to create work packets." % (Sec2HMS(timeit.default_timer() - startStage2, True)))
       
        #print(">>>>>>>> fractional group processed. len(areClose[0])= %d, processOneBlockMaxSize= %d" % \
        #          (data.lenNpHoldAreClose), parameters.processOneBlockMaxSize))
    
    if (len(moreJobs) != 0):
        if (len(moreJobs) < parameters.availableCpus):
            parameters.processOneBlockMaxSize = max(10000,  int(parameters.processOneBlockMaxSize * .9))
        elif (len(moreJobs) > (parameters.availableCpus * 4)):
            parameters.processOneBlockMaxSize = min(100000, int(parameters.processOneBlockMaxSize * 1.1))
     
        data.log.Log("CalcNearObjects: processOneBlockMaxSize set to %d for %d jobs" % 
                         (parameters.processOneBlockMaxSize, len(moreJobs)))
        
    secondStageIteration = 0
    
    for job in moreJobs:             
        #print("About to pull second stage job")
        suspendTime = CheckForSuspension(data.log)
        startStage2 += suspendTime
        
        nearObjectsStatusLine.Print(parameters, data, secondStageIteration)
        [myIteration, areCloseFinal, grossTime, computationalTime] = job()
        #print(">>>>>>>> second state deque time= %f (+%f)" % (timeit.default_timer() - x, timeit.default_timer() - last))
        #last = timeit.default_timer()
        #print(">>>>>>>> grossTime= %f, computationalTime= %f, file read time= %f" % (groseTime, computationalTime, groseTime - computationalTime))
        #print(">>>>>>>> job pulled. Len(areCloseFinal)= ", len(areCloseFinal))
        #print("job pulled. areCloseFinal:\n", areCloseFinal)
    
        if areCloseFinal[0][0] >= 0:
            data.finalHoldAreClose += areCloseFinal
            
        secondStageIteration += 1
                
    print("Stage 2 took %s H:M:S to process packets" % (Sec2HMS(timeit.default_timer() - startStage2, True)))
    data.timeInStageTwo += timeit.default_timer() - startStage2
    
    # Print final time when done
    nearObjectsStatusLine.Print(parameters, data, secondStageIteration, True)

    #print("Exiting CalcNearObjects.")
    #print("******\Just wanted to stop the program.\n*****")
    #raise ValueError("******\Just wanted to stop the program.\n*****")
    return
    
###############################################################################,
def GrowEllipsoid(parameters, data, ellipsoid, tangleSet):
    ellipsoid.CalcCenterOfMass(parameters, data.allPolymers)
    ellipsoid.CalcRadius(parameters, data.allPolymers)
    
    parameters.maxTangleRadius = max(parameters.maxTangleRadius, ellipsoid.GetRadius())    
    return()
    
###############################################################################,
def AddEllipsoidToListDrying(parameters, data, tangleSet):
    ellipsoid = Ellipsoid(parameters)
    ellipsoid.SetTangleSet(tangleSet)
    ellipsoid.CalcCenterOfMass(parameters, data.allPolymers)
    ellipsoid.CalcRadius(parameters, data.allPolymers)
    
    returnIndex = len(data.allEllipsoids)    
    data.allEllipsoids.append(ellipsoid)
    parameters.maxTangleRadius = max(parameters.maxTangleRadius, ellipsoid.GetRadius())
    return(returnIndex)

###############################################################################,
def AddEllipsoidToListNonDrying(parameters, data):
    # need at least 2 ellipsoids to make a polymer
    if (len(data.allPolymers) < 2):
        return
    
    i = j = random.randint(0,len(data.allPolymers))
    
    while (i == j):
        j = random.randint(0,len(data.allPolymers) - 1)
        
    ellipsoid = Ellipsoid(parameters)
    ellipsoid.SetTangleSet({1,j})
    ellipsoid.CalcForcedCenterOfMass(parameters, data.allPolymers)
    ellipsoid.CalcRadius(parameters, data.allPolymers)
    
    data.allEllipsoids.append(ellipsoid)
    parameters.maxTangleRadius = max(parameters.maxTangleRadius, ellipsoid.GetTangleRadius())

    #DumpElipsoid(parameters, data, ellipsoid)
    return

###############################################################################
def CheckForParticleInclusion(parameters, data):
    
    listOfParticles = GetParticleIndexesLessThen(parameters, data.allParticles, parameters.maxPolymerRadius)
    
    for ellipsoid in data.allEllipsoids:
        for ii in range(len(listOfParticles)):
            i = listOfParticles[ii]
            
            if data.allParticles[i].GetRadius() < ellipsoid.GetRadius():
                try:
                    if ellipsoid.IsInEllipsoid(data.allParticles[i]):
                        data.log.Log("Capturing Particle i=%d" % (i))
                        data.capturedParticles.PushAgglomerate(data.allParticles[i])
                        del data.allParticles[i]
                        ellipsoid.AddParticle()
                        #print("allParticles[i].GetRadius()= %e, ellipsoid.GetRadius()= %e" % (data.allParticles[i].GetRadius(), ellipsoid.GetRadius()))
                        
                        for j in range(len(listOfParticles)):
                            listOfParticles[j] -= 1
                except:
                    print("******\Error:  \ni=%d is out of Range.\n*****" % (i))
                    raise ValueError("******\Error:  \ni is out of Range.\n*****") 
              
    return

###############################################################################
#@jit()
def TurnTanglesIntoEllipsoids(parameters, data, checkToCreateEllipsoids):
    allPolymers = data.allPolymers
    totalNewTangles = 0
    totalPolymersInTangled = 0
    needsProcessing = [True] * len(allPolymers)
    existingTangles = 0
    k = 0
            
    for i in range(len(allPolymers)):
        if allPolymers[i].GetEllipsoidIndex() != None:
            existingTangles += 1
            
    #print(">> len(allPolymers)= %d, existingTangles= %d" % (len(allPolymers), existingTangles / 2))
    existingTangles = 0
        
    for i in checkToCreateEllipsoids:
        if (needsProcessing[i]):
            ellipsoidList = []
            tangleSet = allPolymers[i].GetTangleSet(i)
            ellipsoidIndex = 0
            # create a list of all ellipsoids associated with new tangle
            for vv in tangleSet:
                ellipsoidIndex = allPolymers[vv].GetEllipsoidIndex()
                #print("polymer %d" % (vv))
                
                if ellipsoidIndex != None:
                    ellipsoidList.append(ellipsoidIndex)
                    #print("ellipsoidIndex= %d for polymer %d" % (ellipsoidIndex, vv))
             
            # Get rid of duplicates
            ellipsoidList = list(set(ellipsoidList))
            #print("ellipsoidList=", ellipsoidList)
            
            # if No polymers pointing to ellipsoids, then create ellipsoid tangle
            if (len(ellipsoidList) == 0):
                totalNewTangles += 1
                opps = False
                
                # if this tangle set exists in ellipsoid list then bug.
                for k in range(len(data.allEllipsoids)):
                    if (tangleSet == data.allEllipsoids[k].GetTangleSet()):
                        opps = True
                        
                if opps:
                    print("****\nEllipsoid already with this tangle set. Elliipsoid= %d\n***" % (k))
                else:
                    ellipsoidIndex = AddEllipsoidToListDrying(parameters, data, tangleSet)
                    #print("New Ellipsoid")
            else:
                ellipsoidIndex = min(ellipsoidList)
                
                # If multiply tangles now associated then bump stastic
                if (len(ellipsoidList) > 1):
                    #print("Ellipsoids Merged!!!!!!")
                    data.ellipsoidMerges += 1
                    
                GrowEllipsoid(parameters, data, data.allEllipsoids[ellipsoidIndex], tangleSet)
                existingTangles += 1
                #print(">>>>> Grow Ellipsoid")
                
            # mark all polymers in the tangle as processed.
            for k in tangleSet:
                if (needsProcessing[k] == True):
                    totalPolymersInTangled += 1   
                    
                    needsProcessing[k] = False
                    allPolymers[k].SetEllipsoidIndex(ellipsoidIndex)
                    #print("SetEllipsoidIndex: ellipsoidIndex= %d in polymer %d" % (ellipsoidIndex, k))
                    
    #print(">> Existing Tangles= %d, New Tangles= %d" % (existingTangles, totalNewTangles))
    #existingTangles = 0
    #        
    #for i in range(len(allPolymers)):
    #    if allPolymers[i].GetEllipsoidIndex() != None:
    #        existingTangles += 1
    #        
    #print(">> len(allPolymers)= %d, existingTangles= %d." % (len(allPolymers), existingTangles / 2))
    return(totalNewTangles, totalPolymersInTangled)

###############################################################################
#@jit()
def CheckForPolymerTangles(parameters, data):
    if (parameters.drying == False):
        #CreateNewEllipsoids(parameters, data)
        return(0, 0)
        
    
    if ('data.npHoldAreClose' not in dir()):    
        data.npHoldAreClose = np.full(int(data.shifter), 0, dtype="uint64")
        
    totalPolymersInTangled = 0
    totalNewTangles = 0
        
    print("\nStarting CheckForPolymerTangles at %s with %s polymers" % (Sec2HMS(parameters.clock), FormatNumber(len(data.allPolymers))))
    data.lenNpHoldAreClose = 0
    data.finalHoldAreClose = []
    allPolymers = data.allPolymers
    startCheckForPolymerTangles = timeit.default_timer()
   
    RecalculateBoxSize(parameters, data, True)
    
    CalcPolymerBrownianMotion(parameters, data, parameters.clock - data.lastPolymerBrownianMotionTime)
    data.lastPolymerBrownianMotionTime = parameters.clock
    
    CreateNewPolymers(parameters, data)
    data.polymerBins.BoxResized(parameters, data.allPolymers)
    
    if (len(data.allPolymers) == 0):
        print("len(allPolymers) = 0")
    else:
        # returns data.npHoldAreCloseFinal
        CalcNearObjects(parameters, data)
    
        checkToCreateEllipsoids = []
    
        #if elements == 0 then nothing to do
        if data.lenNpHoldAreClose != 0:
            #print("numberTangled= %d (%8.5f%% of elements to check)." % (len(areClose), len(areClose) / elements * 100.0))
            #if (len(areCloseFinal) != 0): print("%d tangled polymers." % (len(areCloseFinal)))
    
            # Mark the tangled Polymers as such
            for v in range(len(data.finalHoldAreClose)):
                i = data.finalHoldAreClose[v][0]
                j = data.finalHoldAreClose[v][1]
                checkToCreateEllipsoids.append(i)
                checkToCreateEllipsoids.append(j)
                
                iTangleSet = allPolymers[i].GetTangleSet(i)
                jTangleSet = allPolymers[j].GetTangleSet(j)
                tangleSet = iTangleSet | jTangleSet  # combine lists and get rid of duplicates
    
                # Let everyone know who they are tangled with
                for vv in tangleSet:
                    allPolymers[vv].SetTangleSet(tangleSet)
    
        [totalNewTangles, totalPolymersInTangled] = TurnTanglesIntoEllipsoids(parameters, data, checkToCreateEllipsoids)
        
        parameters.timeInCheckForPolymerTangles += timeit.default_timer() - startCheckForPolymerTangles
        
        #if (parameters.availableCpus != 1):
        #    data.job_server.print_stats()    
                    
        data.finalHoldAreClose = []
        data.polymerBins.BoxResized(parameters, allPolymers, 0, len(allPolymers))
   
    if False:     
        print("***\nCheckForPolymerTangles: allEllipsoids:")
        
        for i in range(len(data.allEllipsoids)):
            tangleSet_i = data.allEllipsoids[i].GetTangleSet()
            
            for j in range(i + 1, len(data.allEllipsoids)):
                tangleSet_j = data.allEllipsoids[j].GetTangleSet()
                if tangleSet_i == tangleSet_j:
                    print("TS= ", tangleSet_i, ", i=", i, ", j=", j)
                    
        print("CheckForPolymerTangles: Done\n***:")
        
    # if npHoldAreClose is getting full then make it bigger
    if (data.maxLenNpHoldAreClose > (data.npHoldAreClose[0] * .8)):
        data.shifter = int(data.shifter * 1.25)  # 1.25 = 1 / .8
        data.npHoldAreClose = np.full(int(data.shifter), 0, dtype="uint64")
        
    
    data.polymerBins.SaveBinLen(parameters.baseDir + "polymerBinsLens.txt")
    print("End CheckForPolymerTangles(%s). New tangles= %d, polymers in tangled= %d, TE= %d.\n" % \
              (Sec2HMS(timeit.default_timer() - startCheckForPolymerTangles), totalNewTangles, totalPolymersInTangled, len(data.allEllipsoids)))
    return(totalNewTangles, totalPolymersInTangled)

###############################################################################
def CalcAverageAggRad(allParticles):
    # display the average aggregate size
    total = 0
    
    total = sum(particle.r for particle in allParticles)
               
    if (total == 0):
        averageR = 0
    else:
        averageR = total / len(allParticles)
        
    return(averageR)
    
###############################################################################
def _remove_leading_zero(string):
    if string.startswith("0."):
        return string[1:]
    
    if string.startswith("-0."):
        return "-" + string[2:] 
    
    return(string)

###############################################################################
def FormatIt(floatNumber, precision = 5):
    if (floatNumber < 1):
        fmt = ("%%%d.%df" % (precision, precision - 1))
        tightFormat = (fmt % (floatNumber))
        tightFormat = _remove_leading_zero(tightFormat)
    elif (floatNumber < 10):
        fmt = ("%%%d.%df" % (precision, precision - 2))
        tightFormat = (fmt % (floatNumber))
    elif (floatNumber < 100):
        fmt = ("%%%d.%df" % (precision, precision - 3))
        tightFormat = (fmt % (floatNumber))
    elif (floatNumber < 1000):
        fmt = ("%%%d.%df" % (precision, precision - 4))
        tightFormat = (fmt % (floatNumber))
    else:
        fmt = ("%%%dd" % (precision))
        tightFormat = (fmt % (int(floatNumber)))
        
    return(tightFormat)

    
###############################################################################
def PrintStatusLine(parameters, data, log, lastStats, startTime, iteration = 0, deltaIt = 0):
    timeDelta = int(time.time() - startTime) - lastStats
    lastStats = int(time.time() - startTime)
       
    averageR = CalcAverageAggRad(data.allParticles)  
    
    if (averageR == 0):
        aar = ("avg agg rad = <No aggregate left>")
    else:
        aar = ("AAR=%8.2e" % (averageR))
        
    it = ("it=%+5d" % (deltaIt))
            
    totalParticles = CalculateTotalParticle(parameters, data.allParticles)
    
    timeDeltaString = ("+" + Sec2HMS(timeDelta))
        
        
    if parameters.generatePolymers:
        tPolymers = len(data.allPolymers)
    else:
        tPolymers = totalParticles * parameters.chainsPerParticle
        
    if parameters.drying:
        TE = (", TE= %3d" % (len(data.allEllipsoids)))
    else:
        TE = ""
    
    statusLine = ("%s(%s), %s: clk= %s, SL= %s, Pa= %6s, Po= %6s, %s%s, BS= %8.2e" % \
                      (Sec2HMS(lastStats), timeDeltaString, it,  \
                       FormatIt(parameters.clock, 5), FormatIt(parameters.solidsLoading, 5), \
                       FormatNumber(totalParticles), FormatNumber(tPolymers), aar, \
                       TE, parameters.boxSize))
    log.Log(statusLine)
    return(lastStats)
    
###############################################################################
def CalcParticleBrownianMotion(parameters, data, timeStep):
    allParticles = data.allParticles
    # Brownian motion + other movement factors for Particles/Aggregates.
    ##determine settling distance for particles, need to actually use real stuff from paper###
    # DUG Ver 0.00.02: Added correct settling equation.  the underbar is used to denote
    # subscripted variables.
    start = timeit.default_timer()
    
    for i in range(len(allParticles)):
        D = (parameters.boltz * parameters.temperature) / (6 * math.pi * parameters.fluidViscosity * allParticles[i].r) # diffusionConstant
        t_agg = timeStep            # delta time since last iteration
        rho_p = parameters.PEDensity
        rho_f = parameters.fluidDensity
        phi_2 = parameters.solidsLoading * (1.0 - parameters.polymerPercent)
        g = parameters.gravitationalConstant
        mu = parameters.fluidViscosity
        r = allParticles[i].r
        
        [Px, Py, Pz] = CreateRandomVector(parameters, random.random())
        #print(D, parameters.fluidViscosity, parameters.solidsLoading)
        settlingAmount = (((2.0 * (rho_p - rho_f) * (1.0 - phi_2) * g * r) / (9.0 * mu)) * t_agg)
        
        deltaX = (Px * math.sqrt(2.0 * D * t_agg))
        deltaY = (Py * math.sqrt(2.0 * D * t_agg))
        deltaZ = (Pz * math.sqrt(2.0 * D * t_agg))
            
        [x, y, z] = allParticles[i].GetLocation()
        x -= deltaX
        y -= deltaY 
        z -= deltaZ + settlingAmount
        allParticles[i].SetLocation(parameters, x, y, z)
        
        movement = abs(deltaZ)
        data.brownianCount += 1
        data.brownianTotal += movement
        data.brownianMax = max(data.brownianMax, movement)
            
        distance = abs(settlingAmount)
        data.stokesCount += 1
        data.stokesTotal += distance
        data.stokesMax = max(data.stokesMax , distance)

    parameters.timeInCalcParticleBrownianMotion += timeit.default_timer() - start
    return
    
###############################################################################
def CalcPolymerBrownianDistance(parameters, N, timeStep, maxMovement = False):
    t_agg = timeStep             # delta time since last iteration
    k = parameters.boltz
    T = parameters.temperature
    eta = parameters.fluidViscosity
    
    R_g = CalculateRadiusGyration(N)  
    D = (k * T) / (6.0 * math.pi * eta * R_g)
    
    sigma_1 = math.sqrt(2 * D * t_agg)
    mu = 0.0
    
    if (maxMovement):
        P = 1.0
    else:
        P = np.random.normal(mu, sigma_1)
   
    return(P * sigma_1)
###############################################################################

def CalcPolymerBrownianMotion(parameters, data, timeStep):
    start = timeit.default_timer()
    
    allPolymers = data.allPolymers
    polymerLength = GetPolymerLength(parameters)
    
    needsProcessing = [True] * len(allPolymers)
    parameters.runningTotalMaxBrownian += CalcPolymerBrownianDistance(parameters, polymerLength, timeStep, True)
    doCheck = (parameters.runningTotalMaxBrownian > (parameters.monomerRad * 4.0))
    
    #print("Clock=%8.3f(+%f), runningTotalMaxBrownian= %e, doCheck %r" % (parameters.clock, timeStep, parameters.runningTotalMaxBrownian, doCheck))
    
    if doCheck:   
        #print("Calculating polymer Brownian motion runningTotalMaxBrownian(= %e) > (monomerRad(=%e) * 4" % (parameters.runningTotalMaxBrownian, parameters.monomerRad))
        parameters.runningTotalMaxBrownian = 0.0
        
        for i in range(len(allPolymers)):
            if needsProcessing[i]:
                N = polymerLength * len(allPolymers[i].GetTangleSet(i))
                distanceMoved = CalcPolymerBrownianDistance(parameters, N, timeStep)
                
                [x,y,z] = CreateRandomVector(parameters, distanceMoved)
                
                tangleSet = allPolymers[i].GetTangleSet(i)
                
                # all tangles get moved as a group
                for j in tangleSet:
                    allPolymers[j].DeltaMovePolymer(parameters, x, y, z)
                    needsProcessing[j] = False
                    
                if (len(tangleSet) > 1):
                    for tangleIndex in tangleSet: break
                    ellipsoidIndex = allPolymers[tangleIndex].GetEllipsoidIndex()
                    data. allEllipsoids[ellipsoidIndex].DeltaMoveEllipsoid(parameters, x, y, z)
        
    parameters.timeInCalcPolymerBrownianMotion += timeit.default_timer() - start
    return

###############################################################################
# Calculate Van Der Waals equation
def CalcV_daMatrix(parameters, distanceMatrix, lamda, d, phi_1, k, T, sigmaD_g):
    # DUG Ver 0.00.06: Need to add new factor from Paper from interactMatrix      
    
    # DUG Ver 0.00.06: Added an additional Close distance factor
    # V_d = ((-3 * d * phi * k * T * lamda**2) / (2 * sigmaD_g)) + 
    #        (((d * phi**2 * k * T) / (10 * sigmaD_g)) * (12 - 45 * lamda - 60 * lamda**2)) 
    # temp1 = ((-3 * d * phi * k * T * lamda**2) / (2 * sigmaD_g))
    # temp2b = 60 * lamda**2
    # temp2b = 45 * lamda
    # temp2c = 12 - temp2b - temp2a (i.e. (12 - 45 * lamda - 60 * lamda**2))
    # temp2 = ((d * phi**2 * k * T) / (10 * sigmaD_g)) * temp2c
    # V_da = temp1 + temp2
    
    temp1 = np.multiply(np.square(lamda), ((-3.0 * d * phi_1 * k * T) / (2.0 * sigmaD_g)))
    temp2 = (d * phi_1**2 * k * T) / (10.0 * sigmaD_g)
    temp3a = np.subtract(12.0, np.multiply(45.0, lamda))
    temp3 = np.subtract(temp3a, np.multiply(60.0, np.square(lamda))) 
    temp3a = None
    
    
    V_da = np.add(temp1, np.multiply(temp2, temp3))
    V_da[np.isnan(V_da)]=0.0
            
    return(V_da)
    
###############################################################################
def CalcV_dbMatrix(parameters, distanceMatrix, phi_1, d, k, T, sigmaD_g, lamda):
    temp1 = np.multiply(np.square(phi_1), ((d * k * T) / (10.0 * sigmaD_g)))
    temp2  = np.subtract(12.0, np.multiply(45.0, lamda));
    temp2  = np.add(temp2, np.multiply(60.0, np.square(lamda))); 
    temp2  = np.subtract(temp2, np.multiply(30.0, np.power(lamda, 3.0)));
    temp2  = np.subtract(temp2, np.multiply(3.0, np.power(lamda, 5.0)));
    
    V_db = np.multiply(temp1, temp2)
    V_db[np.isnan(V_db)]=0.0

    return(V_db)
    
###############################################################################
def CalcV_dcMatrix(parameters, allParticles, arrayDimension):
    # DUG Ver 0.00.07: Added an additional far distance factor
    # V_d = 0
    
    if (arrayDimension == 1):
        V_dc = np.zeros((len(allParticles))) 
    else:           
        V_dc = np.zeros((len(allParticles), len(allParticles)))            

    return(V_dc)
    
###############################################################################
def GetPolymerLength(parameters):
    if parameters.drying:
        polymerLength = parameters.polymerLength
    else:
        polymerLength = CalculatePolymerLength(parameters)
    return(polymerLength)
    
###############################################################################
def CalcV_dMatrixWorker(parameters, allParticles, distanceMatrix, sumRadiusMatrix, arrayDimension = 2):  
    polymerLength = GetPolymerLength(parameters)
    sigmaD_g = CalculateRadiusGyration(polymerLength) * 2.0
    k = parameters.boltz
    phi_1 = parameters.solidsLoading * parameters.polymerPercent
    #print(distanceMatrix)
    #print("Diameter of gyration is ", sigmaD_g)
    
    h = np.subtract(distanceMatrix, sumRadiusMatrix)  # Surface to Surface distance
    d = CalcAverageAggRad(allParticles) * 2.0
    lamda = np.divide(np.subtract(h, sigmaD_g), sigmaD_g)  # lamda = (h - sigmaD_g) / sigmaD_g
    T = parameters.temperature
    
    V_da = CalcV_daMatrix(parameters, distanceMatrix, lamda, d, phi_1, k, T, sigmaD_g)
    V_db = CalcV_dbMatrix(parameters, distanceMatrix, phi_1, d, k, T, sigmaD_g, lamda)
    V_dc = CalcV_dcMatrix(parameters, distanceMatrix, arrayDimension)

    V_dam = np.ma.array(V_da, mask=( h >= sigmaD_g));
    V_dbm = np.ma.array(V_db, mask=((h <  sigmaD_g) | (h > 2.0 * sigmaD_g)));
    V_dcm = np.ma.array(V_dc, mask=( h <= (2.0 * sigmaD_g)));
    
    V_d = np.ma.array(np.dstack((V_dam, V_dbm, V_dcm)), mask=np.dstack((V_dam.mask, V_dbm.mask, V_dcm.mask)));
    V_d = np.sum(V_d, axis=2)
    
    # The above operation may result in NaN results.  Change them to 0
    # Need to see if one of the two below solution lines runs faster
    # V_d = np.nan_to_num(V_d)
    # V_d[np.isnan(V_d)]=0
    #print(V_dam)
    #print(V_dbm)
    #print(V_dcm)
    np.fill_diagonal(V_d, 0.0)
    #print(V_d)
    return(V_d)
    
###############################################################################
# Polymer Interact
def CalcV_dMatrix(parameters, allParticles, distanceMatrix):
    sumRadiusMatrix = CalcSumRadiusMatrix(parameters, allParticles)
    V_d = CalcV_dMatrixWorker(parameters, allParticles, distanceMatrix, sumRadiusMatrix)

    return(V_d)
    
###############################################################################
###############################################################################
def CalcVdWMatrixWorker(parameters, allParticles,subParticleList, partRadProd, partRadSum, partRadDiff, distanceSqMatrix, clearNaNErrors = False):
    A = parameters.A
    #np.reshape(partRadProd,(10,10)) #had to reshape because it was outputting a (10,10) and (1000,) matrix    
    """temp1 = np.multiply(partRadProd, 2.0)
    temp2 = np.square(partRadSum)
    temp3 = np.square(partRadDiff)
    temp4 = distanceSqMatrix"""
    #we do not want all of the particle to be calcualted on ly the one that are in the contact with the parcel
    temp1 = CalcParcelProductRadiusMatrix(parameters, allParticles, subParticleList, data.allSalts)
    temp2 = np.square(CalcSumParcelRadiusMatrix(parameters, allParticles, subParticleList, data.allSalts))
    temp3 = np.square(CalcParcelDifferenceRadiusMatrix(parameters, allParticles,subParticleList, data.allSalts))
     
    temp4 = distanceSqMatrix
    

    x = np.where(np.subtract(temp4, temp2) == 0)
    y = np.where(np.subtract(temp4, temp3) == 0)
    
    if (len(x[0]) != 0): raise ValueError("******\Error:  \nabout to divide by 0.\n*****")
    if (len(y[0]) != 0): raise ValueError("******\Error:  \nabout to divide by 0.\n*****")
    
    temp5 = np.divide(temp1, np.subtract(temp4, temp2))
    temp6 = np.divide(temp1, np.subtract(temp4, temp3))
    
    
    temp7 = np.divide(np.subtract(temp4, temp2), np.subtract(temp4, temp3))
    
    # for this test some particles can overlap... We don't care about them
    # so just adjust the data so it works
    if clearNaNErrors:
        temp7[temp7 <= 0] = .1
        
    temp7 = np.log(temp7)
    
    
    VdW = np.multiply(A / 6.0, np.add(temp5, np.add(temp6, temp7)))
    VdW = np.negative(VdW)
    
    return(VdW)
    
    
###############################################################################
def CalcVdWMatrix(parameters, subParticleList, allParticles, allSalts, distanceSqMatrix, clearNaNErrors = False):
    # Van der Waals equation
    # Where:
    #    R = center-to-center distance of two particles
    #    r_i = radius of particle 1
    #    r_j = radius of particle 2
    #    A = hamaker constant.  This is caculated in readFile.py and used here
    #
    # VdW= -(A/6) * \
    #           (((2*r_i*r_j) / (R**2-((r_i+r_j)**2))) + \
    #            ((2*r_i*r_j) / (R**2-((r_i-r_j)**2))) + \
    #          ln(R**2-((r_i+r_j)**2) / (R**2-((r_i-r_j)**2))))
    #
    # Now lets build code.  Build common parts first
    #
    # temp1 = 2*r_i*R_j
    # temp2 = (r_i+r_j)**2)
    # temp3 = (r_i-r_j)**2)
    # temp4 = R**2
    #
    # Substuting you get:
    # 
    # VdW= -(A/6) * \
    #           (((temp1) / (temp4-temp2)) + \
    #            ((temp1) / (temp4-temp3)) + \
    #          ln((temp4-temp2) / (temp4-temp3)))
    #
    # temp5 = ((temp1) / (temp4-temp2))
    # temp4 = ((temp1) / (temp4-temp3))
    # temp7 = ln((temp4-temp2) / (temp4-temp3)) 
    #
    # Final Subtuting:
    #
    # VdW= -(A/6) * (temp5 + temp6 + temp7)
    #
    # Since the diagnal is zero we will divide by zero, so ignore the exception  
    partPRads = []
     
    partPRads = [part.r for part in subParticleList]
        
    #particle radii of only the particles touching that specific parcel
    partSRads = []
    partSRads = [part.r for part in allSalts]
    
    
    partRadDiff = np.subtract.outer(partPRads, partSRads)
    partRadSum = np.add.outer(partPRads, partSRads) 
    partRadProd = np.multiply.outer(partPRads, partSRads)
    
#put in subParticle list because we do not want all of hte particles to b calclated      
    VdW = CalcVdWMatrixWorker(parameters, allParticles, subParticleList,partRadProd, partRadSum, partRadDiff, distanceSqMatrix, clearNaNErrors)

    return(VdW)
 
###############################################################################
def CalcTwoParticlesToAgglomerate(parameters, allParticles, interactMatrix):
    rList = [particle.r for particle in allParticles]
    oneOverRList = [1 / r for r in rList]
    
    rSummedListMatrix = np.add.outer(rList, rList)
    oneOverRSummedListMatrix = np.add.outer(oneOverRList, oneOverRList)
    k = parameters.boltz
    T = parameters.temperature
    mu = parameters.fluidViscosity
    
    colFreqMatrix = np.multiply((2.0*k*T) / (3.0*mu), np.multiply(rSummedListMatrix, oneOverRSummedListMatrix))
    
    temp = np.nan_to_num(np.divide(interactMatrix, parameters.boltz * parameters.temperature))
    temp = temp*1e-9
    biggest = 50    # close particles can be very repuslive.  This number
                    # represents a large number without crashing the program
    temp[temp > biggest] = biggest
    stabRatMatrix = np.exp(temp)
    stabRatMatrix = np.multiply(stabRatMatrix,1e9)
    print(stabRatMatrix)
    
        
    aggFreqMatrix = np.divide(colFreqMatrix, stabRatMatrix)
    np.fill_diagonal(aggFreqMatrix, 0.0)      # make sure i != j below threshold
    sumFreqRows = np.sum(aggFreqMatrix, 1)
    sumFreq = np.sum(sumFreqRows)
    
    ## select particle pair of interest ##
    goal = random.random() * sumFreq
    sumAggFreq=0
    i = 0
    
    # Lets get in the ball park and just fly through the rows
    while sumAggFreq < goal:
        if (sumAggFreq + sumFreqRows[i]) < goal:
            sumAggFreq += sumFreqRows[i]
        else:
            break
        
        i += 1
        
    # Now lets take out time and go through the row to find the exact partical pair
    j = 0;
    
    while sumAggFreq < goal:
        if (sumAggFreq + aggFreqMatrix[i][j]) < goal:
            sumAggFreq += aggFreqMatrix[i][j]
        else:
            break
        
        j += 1
        
    #averageAggFreq = aggFreqMatrix[i][j]
    averageAggFreq = sumFreq / len(allParticles)**2
    return(i, j, averageAggFreq)
    
###############################################################################
def WillNewAgglomerateFit(parameters, data, newCombinedAgg):
    willFit = False
    tries = 0
    maxTries = 2000
    
    allParticles = data.allParticles
    allEllipsoids = data.allEllipsoids
    
    while (willFit == False) & (tries < maxTries):
        willFit = True
        [x1, y1, z1] = newCombinedAgg.GetLocation()
        r1 = newCombinedAgg.GetRadius()
        i = 0
        
        while ((i < len(allParticles)) & (tries < maxTries) & (willFit == True)):
            [x2, y2, z2] = allParticles[i].GetLocation()
            
            if (i >= len(allParticles)):
                raise ValueError("******\Error:  \ni is out of Range... But wait previous line worked.\n*****")
            r2 = allParticles[i].GetRadius()
            
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            
            if (distance < (r1 + r2)):
                newCombinedAgg.SetRandomLocation(parameters)
                tries += 1
                willFit = False
                
            i += 1
            
        if (willFit):
            i = 0
        
            while ((i < len(allEllipsoids)) & (tries < maxTries) & (willFit == True)):
                [x2, y2, z2] = allEllipsoids[i].GetLocation()
                r2 = allEllipsoids[i].GetRadius()
                
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                
                if (distance < (r1 + r2)):
                    newCombinedAgg.SetRandomLocation(parameters)
                    tries += 1
                    willFit = False
                    
                i += 1
        
    willFit = (tries < maxTries)
    
    if willFit == False:
        allParticles.append(newCombinedAgg)
        fileName = ("%s_willFitFail_%04d.dump" % (parameters.fileNameBase, data.willFitDumpIndex))  
        data.willFitDumpIndex += 1        
        WriteDump(parameters, data, fileName)
        del allParticles[len(allParticles) - 1]
        
    data.maxWillFit = max(data.maxWillFit, tries)
    return(willFit)
    
###############################################################################
def WillTwoParticlesNotReject(parameters, data, i, j, currentPotential):
    numberDivisions = parameters.particleCount
    
    ri = data.allParticles[i].GetRadius()    
    rj = data.allParticles[j].GetRadius()
    SLT = parameters.stericLayerThickness
    
    SS_adjustment = ri + rj + (SLT * 2.0)
    SS_distance = CalcDistance(data.allParticles[i], data.allParticles[j]) - SS_adjustment
    vectorLength = min(SS_distance, 10e-9)
        
    privateParticleList = []
    
    if False:    
        xDistances = [(float(index) / float(numberDivisions) * vectorLength) + SS_adjustment for index in range(1, numberDivisions + 1)]    
        
        agg = Aggregate(ri)
        agg.SetLocation(parameters, 0, 0, 0)
        privateParticleList.append(agg)
       
        for index in range(numberDivisions):  
            agg = Aggregate(rj)
            agg.SetLocation(parameters, xDistances[index], 0, 0)
            privateParticleList.append(agg)
    
        
        distanceMatrix = CalcDistanceMatrix(parameters, privateParticleList)
        np.fill_diagonal(distanceMatrix, 1.0)  # this will keep particles away from self
        distanceSqMatrix = np.square(distanceMatrix)    
        VdW = CalcVdWMatrix(parameters, privateParticleList, distanceSqMatrix, True)
        interactMatrix = VdW
        
        EDL = EDLMatrix(parameters, data.allParticles)
        interactMatrix = np.add(interactMatrix, EDL)
        
        
        if parameters.V_dCalc == True:
            V_d = CalcV_dMatrix(parameters, privateParticleList, distanceMatrix)
            interactMatrix = np.add(interactMatrix, V_d)

    else:
        xDistances = [(float(index) / float(numberDivisions) * vectorLength) + SS_adjustment for index in range(1, numberDivisions + 1)]   
        
        #print("")
        #for index in range(10):
        #    print("Vector= ", CalculateVectorLength((xi + SS_adjustment)- xj[index], (yi + SS_adjustment) - yj[index], (zi + SS_adjustment) - zj[index]))
        
        xiLst = np.full(1, 0.0, dtype=parameters.doMathIn)
        yiLst = np.full(1, 0.0, dtype=parameters.doMathIn)
        ziLst = np.full(1, 0.0, dtype=parameters.doMathIn)
        xjLst = np.full(numberDivisions, 0.0, dtype=parameters.doMathIn)
        yjLst = np.full(numberDivisions, 0.0, dtype=parameters.doMathIn)
        zjLst = np.full(numberDivisions, 0.0, dtype=parameters.doMathIn)
        rjLst = np.full(numberDivisions, 0.0, dtype=parameters.doMathIn)
        
        xiLst[0] = 0.0
        yiLst[0] = 0.0
        ziLst[0] = 0.0
       
        for index in range(numberDivisions):  
            agg = Aggregate(rj)
            agg.SetLocation(parameters, xDistances[index], 0, 0)
            privateParticleList.append(agg)
    
            xjLst[index] = xDistances[index]
            yjLst[index] = 0.0
            zjLst[index] = 0.0
            rjLst[index] = rj
           
        
        xSqDistMatrix = np.square(np.subtract(xiLst, xjLst))
        ySqDistMatrix = np.square(np.subtract(yiLst, yjLst))
        zSqDistMatrix = np.square(np.subtract(ziLst, zjLst))
    
        distanceSqMatrix = np.add(xSqDistMatrix, np.add(ySqDistMatrix, zSqDistMatrix))
    
        distanceMatrix   = np.sqrt(distanceSqMatrix)
        
        partRadDiff = np.subtract(ri, rjLst)
        partRadSum = np.add(ri, rjLst) 
        partRadProd = np.multiply(ri, rjLst)
        
        #if data.firstTime:
        #    data.firstTime = False
        #    print("partRadProd:\n", partRadProd)
        #    print("partRadSum:\n", partRadSum)
        #    print("partRadDiff:\n", partRadDiff)
        #    print("distanceSqMatrix:\n", distanceSqMatrix)
    
        VdW = CalcVdWMatrixWorker(parameters, partRadProd, partRadSum, partRadDiff, distanceSqMatrix, True)
        interactMatrix = VdW
        
        # #polymer Interact 
        # V_d is build from three equations to each describing what happens if particles are close to far.
        # need to build final matrix using the 3 distance matrices (V_da, V_db, and V_dc)
        #
        # Equation for code came from the paper "Depletion Stabilization in 
        # Nanoparticles-Polymer Suspensions: Multi-Length-Scale Analysis of Microstructure"
        # NOTE: [ri + rj] * numberDivisions nenerates a list of numberDivisions length with each element ri + rj
        sumRadiusMatrix = np.array([ri + rj] * numberDivisions)
        #V_d = CalcV_dMatrixWorker(parameters, privateParticleList, np.sqrt(distanceSqMatrix), sumRadiusMatrix, 1)
        #print("V_d:\n", V_d)
        if parameters.V_dCalc == True:
            V_d = CalcV_dMatrixWorker(parameters, privateParticleList, distanceMatrix, sumRadiusMatrix, 1)
            interactMatrix = np.add(interactMatrix, V_d)
            
        EDL = EDLMatrix(parameters, data.allParticles)
        interactMatrix = np.add(interactMatrix, EDL)
        
        
        #print("V_d:\n", V_d)
        #print("interactMatrix:\n", interactMatrix)  
    
    #print("")
    #print("distanceMatrix:\n", distanceMatrix)
    #print("VdW:\n", VdW)
    #print("V_d:\n", V_d)
    #print("interactMatrix:\n", interactMatrix)
    
    #DumpMatrixToCSV(parameters, VdW,             "reject_VdW.csv", privateParticleList)
    #DumpMatrixToCSV(parameters, V_d,             "reject_V_d.csv", privateParticleList)
    #DumpMatrixToCSV(parameters, interactMatrix,  "reject_interactMatrix.csv", privateParticleList)
    
    maxPotential = np.max(interactMatrix[0,1:len(privateParticleList)])
    minPotential = np.min(interactMatrix[0,1:len(privateParticleList)])
    
    #print("currentPotential", currentPotential)
    #print("maxPotential", maxPotential)
    #print("minPotential", minPotential)
    
    k = parameters.boltz
    T = parameters.temperature
        
    ap = math.e**-((maxPotential-currentPotential)/ (k * T))
    ep = math.e**-((0-minPotential)/ (k * T))
    approachProbability = min(1.0, ap)
    escapeProbability = min(1.0, ep)
   
    #print("ap=", ap)
    #print("approachProbability=", approachProbability)
    #print("ep=", ep)
    #print("escapeProbability=", escapeProbability)
    
    accepted = True
    data.sumApproachProb += approachProbability
    data.sumEscapeProb += escapeProbability
    data.rejectProbEvents += 1
    
    if (random.random() > approachProbability):
        data.particlesDidNotApproach += 1
        data.agglomerationsRejected += 1
        accepted = False
    else:
        if (random.random() < escapeProbability):
            data.particlesEscaped += 1
            data.agglomerationsRejected += 1
            accepted = False
        else:
            data.agglomerationsAccepted += 1
    
    return(accepted)
        
###############################################################################
def AggregateTwoParticles(parameters, data, subParticleList, i, j, currentPotential): 
    allParticles = data.allParticles
    
    newC=(allParticles[i].c + allParticles[j].c)
    newR = CalcAggregateRadius(parameters, newC)
                
    newCombinedAgg = Aggregate(newR)
    newCombinedAgg.SetRandomLocation(parameters)
    newCombinedAgg.r = newR
    newCombinedAgg.c = newC
    
    if parameters.drying == False:
        agglomerationSucceded = True
        
        if (WillTwoParticlesNotReject(parameters, data, i, j, currentPotential)):
            newSingleAgg = copy.deepcopy(subParticleList[int(random.random() * len(subParticleList))])
            newSingleAgg.SetRandomLocation(parameters)
    
            allParticles[i] = newCombinedAgg
            allParticles[j] = newSingleAgg
            
            RecalculateBoxSize(parameters, data, False)
        
            minClearanceScale = .05         # percent to scal of gap between particles variable
            
            [nudged, failed] = NudgeParticles(parameters, data, parameters.smallestParticleRadius, parameters.particleRad * minClearanceScale)
    else:                 
        # Check to see if new particle will fit
        agglomerationSucceded = WillNewAgglomerateFit(parameters, data, newCombinedAgg)
        
        if (agglomerationSucceded):
            newSingleAgg = copy.deepcopy(subParticleList[int(random.random() * len(subParticleList))])
            newSingleAgg.SetRandomLocation(parameters)
    
            agglomerationSucceded = WillNewAgglomerateFit(parameters, data, newSingleAgg)
            
            if (agglomerationSucceded):
                if (WillTwoParticlesNotReject(parameters, data, i, j, currentPotential)):
                    del allParticles[max(i,j)]
                    del allParticles[min(i,j)]
                
                    allParticles.append(newCombinedAgg)
                    allParticles.append(newSingleAgg)
                    
                    RecalculateBoxSize(parameters, data, False)
            else:
                newR = newSingleAgg.GetRadius()
            
    return(agglomerationSucceded, newR)

###############################################################################
def SettlementParticleCheck(parameters, data):
    # DUG Particles settling out of the bottom get new particles and put them someware
    # Don't worry about overlap here since NudgeParticles will seperate them
    i = 0
    allParticles = data.allParticles
    downEscapies = data.downEscapies
    
    # DUG Ver 0.00.02: Changed how bottom of box is processed.  Now bottom
    # can be opened or closed.
    while i < len(allParticles):
        [x,y,z] = allParticles[i].GetLocation()
        
        # if particle in the box then process the next one
        if (allParticles[i].isOutBottom(parameters) == False):
            i += 1
        else:
            downEscapies.PushAgglomerate(allParticles[i])
            del allParticles[i]
            
            if (downEscapies.CanPop()):
                thisAggregate = downEscapies.PopAgglomerate()    
                thisAggregate.SetRandomLocation(parameters)
                #debugLog.Log((" Removed downEscapies agglomerate of size %d") % (thisAggregate.c))
                allParticles.append(copy.deepcopy(thisAggregate))
            
            RecalculateBoxSize(parameters, data, False)
    return

###############################################################################
def CheckForOutOfBox(parameters, allParticles):
    for i in range(len(allParticles)):
        if (allParticles[i].isOutOfBox(parameters)):
            allParticles[i].SetRandomLocation(parameters)
            
###############################################################################
def CalcV_sMatrix(parameters, allParticles):
    V_sMatrix = np.zeros((len(allParticles), len(allParticles)))
    
    if (parameters.stericLayerThickness == 0):
        return(V_sMatrix)
    
    overlapList = CalcOverlapParticles(parameters, allParticles, 2.0e-9)
    
    for index in range(len(overlapList[0])):
        i = overlapList[0][index]    
        j = overlapList[1][index]   
        
        if (allParticles[i].r > allParticles[j].r):
            i,j = j,i
            
        r_s = allParticles[i].r + parameters.stericLayerThickness
        r_b = allParticles[j].r + parameters.stericLayerThickness
        d = CalcDistance(allParticles[i], allParticles[j])
        surfaceToSurface_d = d - (allParticles[i].r + allParticles[j].r)
                
        GAMMA = 13790.0
        L = 1.0e-9
        k = parameters.boltz
        T = parameters.temperature
        SS_d = surfaceToSurface_d   
        CC_d = d
        
        h_s = ((r_s - r_b + CC_d) * (r_s + r_b - CC_d)) / (2.0 * CC_d)
        h_b = ((r_b - r_s + CC_d) * (r_b + r_s - CC_d)) / (2.0 * CC_d)
        
        overlapVolume = ((1.0/3.0) * math.pi * h_s**2 * (3.0 * r_s - h_s)) + ((1.0/3.0) * math.pi * h_b**2 * (3.0 * r_b - h_b))
        
        PI = ((k * T) / math.sqrt(1.0/GAMMA)) * ((((2.0 * L) / SS_d)**(9.0/4.0)) - ((SS_d / (2.0 * L))**(3.0/4.0)))
        
        V_s = PI / overlapVolume
            
        V_sMatrix[i][j] = V_s
        V_sMatrix[j][i] = V_s
        #print(V_s)
    
    return(V_sMatrix)

###############################################################################
def CheckForSuspension(log):
    printSuspendMessage = True
    timeitCheckForSuspension = timeit.default_timer()
                  
    # If file "suspend" exects in the working directory then the program execution is suspended
    while os.path.isfile("suspend"):            
        if printSuspendMessage:
            log.Log("Simulation suspended", LogDestination.BOTH, True)
            printSuspendMessage = False
            
        time.sleep(2)
        
    if (printSuspendMessage == False):
        log.Log("Simulation resumed", LogDestination.BOTH, True)
        return(timeit.default_timer() - timeitCheckForSuspension)
    else:
        return(0)

############################################################################### 
def LogToCsv(parameters, data, thisIteration, forceEntry = False, skipPolymers = False):
    #log FluidViscosity & solidsLoading over time…15
    691
    
    allEllipsoids = data.allEllipsoids
    allParticles = data.allParticles
    allPolymers = data.allPolymers
    polymerBins = data.polymerBins
    
    largestTangle = 0
    AAR = CalcAverageAggRad(allParticles)
        
    # Tangles over time
    if parameters.generatePolymers & (skipPolymers == False):
        if forceEntry | parameters.tanglesOverTimeCsv.isNextSimTimeStep(parameters.clock):
            [tangleCount, totalPolymersTangled] = CheckForPolymerTangles(parameters, data)
            totalPolymersTangled = 0
            percentOfPolymersInTangles = 0
            polymersTangleCount = 0

            totalOOBCount = 0
            totalOOBDistance = 0 
            oobMax = 0
             
            if (len(allPolymers) != 0):
                for i in range(len(data.allEllipsoids)):
                    polymersTangleCount = len(data.allEllipsoids[i].GetTangleSet())
                    totalPolymersTangled += polymersTangleCount
                    largestTangle = max(largestTangle, polymersTangleCount)
                
                percentOfPolymersInTangles = totalPolymersTangled / len(allPolymers)
                
                for i in range(len(data.allPolymers)):
                    [x, y, z] = allPolymers[i].GetCenterOfMass() 
                    
                    if (x < 0):
                         totalOOBCount += 1
                         totalOOBDistance += abs(x)
                         oobMax = max(oobMax, abs(x))
                    elif (x > parameters.boxSize):
                         totalOOBCount += 1
                         totalOOBDistance += abs(x - parameters.boxSize)
                         oobMax = max(oobMax, x - parameters.boxSize)
                    elif (y < 0):
                         totalOOBCount += 1
                         totalOOBDistance += abs(y)
                         oobMax = max(oobMax, abs(y))
                    elif (y > parameters.boxSize):
                         totalOOBCount += 1
                         totalOOBDistance += abs(y - parameters.boxSize)
                         oobMax = max(oobMax, y - parameters.boxSize)
                    elif (z < 0):
                         totalOOBCount += 1
                         totalOOBDistance += abs(z)
                         oobMax = max(oobMax, abs(z))
                    elif (z > parameters.boxSize):
                         totalOOBCount += 1
                         totalOOBDistance += abs(z - parameters.boxSize)
                         oobMax = max(oobMax, z - parameters.boxSize)
              
            if (totalOOBCount == 0):
                totalOOBDistance_div_totalOOBCount = 0
            else:
                totalOOBDistance_div_totalOOBCount = totalOOBDistance / totalOOBCount
                
            s = ("%f, %f, %e, %e, %d, %d, %d, %d, %d, %f, %d, %e, %e" %
                     (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR,
                     len(allPolymers), data.tangleCandidates, totalPolymersTangled, polymersTangleCount, 
                    largestTangle, percentOfPolymersInTangles,
                    totalOOBCount, totalOOBDistance_div_totalOOBCount, oobMax))
            
            parameters.tanglesOverTimeCsv.Log(parameters.clock, s)
     
                         
    # log the sim time clock verses average particle size
    if forceEntry | parameters.particalSizeDistributionCsv.isNextSimTimeStep(parameters.clock):
        s = ("%f, %f, %e, %e" % (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR))
        
        for ii in range(len(allParticles)):
            s += (", %e" % (allParticles[ii].r))
                
        #for ii in range(downEscapies.GetQueueLen()):
        #    aggregate = downEscapies.PeekAgglomerate(ii)
        #    s += (", %e" % (aggregate.r))
        parameters.particalSizeDistributionCsv.Log(parameters.clock, s)

    # Ellipsoids
    totalVolume = 0
    NumberOfEllipsoidsWithParticles = 0
    
    if forceEntry | parameters.ellipsoidsOverTimeCsv.isNextSimTimeStep(parameters.clock):
        CheckForParticleInclusion(parameters, data)
        
        for i in allEllipsoids:
            particleSetCount = i.GetParticleCount()
            
            if (particleSetCount > 0): 
                NumberOfEllipsoidsWithParticles += 1
            
            oneVolume = (4.0/3.0) * math.pi * i.GetRadius()**3
            totalVolume += oneVolume
            
        volFrac = totalVolume / parameters.boxSize**3
            
        s = ("%f, %f, %e, %e, %d, %f, %d, %d, %d" % (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR, 
                                             len(allEllipsoids), volFrac, 
                                             data.capturedParticles.GetQueueLen(), NumberOfEllipsoidsWithParticles, largestTangle))
        parameters.ellipsoidsOverTimeCsv.Log(parameters.clock, s)
     
    if forceEntry | parameters.radiusOverTimeCsv.isNextSimTimeStep(parameters.clock):
        if data.rejectProbEvents == 0:
            averageApproachProb = 0
            averageEscaoeProb = 0
        else:
            averageApproachProb = data.sumApproachProb / float(data.rejectProbEvents)
            averageEscaoeProb = data.sumEscapeProb / float(data.rejectProbEvents)
            
        s = ("%f, %f, %e, %e, %d, %e, %e, %e, %e, %e, %d, %d, %d, %e, %e, %e, %5.2f, %5.2f" %
                (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR,
                thisIteration,
                parameters.smallestParticleRadius, CalcAverageAggRad(allParticles), parameters.largestParticleRadius,
                parameters.maxPolymerRadius, parameters.maxTangleRadius,
                data.agglomerationsAccepted, data.agglomerationsRejected, data.particlesDidNotApproach,
                data.lastDim, data.lastAAF, CalculateRadiusGyration(CalculatePolymerLength(parameters)),
                averageApproachProb, averageEscaoeProb))
        parameters.radiusOverTimeCsv.Log(parameters.clock, s)
        data.rejectProbEvents = 0
        data.sumApproachProb = 0.0
        data.sumEscapeProb = 0.0
                         
    if forceEntry | parameters.debugCsv.isNextSimTimeStep(parameters.clock):
        xTot = 0
        yTot = 0
        zTot = 0
        
        for i in range(len(data.allParticles)):
            [x,y,z] = data.allParticles[i].GetLocation()
            xTot += x
            yTot += y
            zTot += z
            
        averageX = (xTot / len(data.allParticles)) / parameters.boxSize
        averageY = (xTot / len(data.allParticles)) / parameters.boxSize
        averageZ = (yTot / len(data.allParticles)) / parameters.boxSize
        
        s = ("%f, %f, %e, %e, %f, %f, %f, %e, %d, %d, %d, %d, %f" % 
                 (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR, 
                      averageX, averageY, averageZ,
                      parameters.largestParticleRadius, data.lengthOfSubparticleList, data.ellipsoidMerges,
                      data.maxWillFit, data.maxLenNpHoldAreClose, parameters.viscosityScaler))
        parameters.debugCsv.Log(parameters.clock, s)
        data.maxWillFit = 0
        
    if forceEntry | parameters.fluidViscositySolidsLoadingOverTimeCsv.isNextSimTimeStep(parameters.clock):
        s = ("%f, %f, %e, %e, %e, %d, %f" % 
                 (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR, 
                  parameters.fluidViscosity,
                  polymerBins.GetBinsPerDimension(), parameters.polymersPerBin))
        parameters.fluidViscositySolidsLoadingOverTimeCsv.Log(parameters.clock, s)
        
    if forceEntry | parameters.executionTimesCsv.isNextSimTimeStep(parameters.clock):
        notAccountedFor = (timeit.default_timer() - data.pgmStart) - \
            (parameters.timeInCheckForPolymerTangles + 
             parameters.timeInRunSimulation +
             parameters.timeInCalcParticleBrownianMotion + 
             parameters.timeInCalcPolymerBrownianMotion +
             parameters.timeInNudgeParticles)
        s = ("%f, %f, %e, %e, %7.1f, %7.1f, %7.1f, %7.1f, %7.1f, %7.1f, %7.1f, %7.1f, %7.1f" % 
                 (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR, 
                  (timeit.default_timer() - data.pgmStart), notAccountedFor,
                  parameters.timeInCheckForPolymerTangles, data.timeInStageOne, data.timeInStageTwo, 
                  parameters.timeInRunSimulation, 
                  parameters.timeInCalcParticleBrownianMotion, parameters.timeInCalcPolymerBrownianMotion,
                  parameters.timeInNudgeParticles))
        parameters.executionTimesCsv.Log(parameters.clock, s)
        
    if forceEntry | parameters.settlingCsv.isNextSimTimeStep(parameters.clock):
        if data.stokesCount == 0:
            stokesAverage = 0.0
        else:
            stokesAverage = data.stokesTotal / data.stokesCount
            
        if data.brownianCount == 0:
            brownianAverage = 0.0
        else:
            brownianAverage = data.brownianTotal / data.brownianCount
            
        s = ("%f, %f, %e, %e, %d, %e, %e, %d, %e, %e" % 
                 (parameters.clock, parameters.solidsLoading, parameters.boxSize, AAR,
                  data.stokesCount, stokesAverage, data.stokesMax,
                  data.brownianCount, brownianAverage, data.brownianMax))
        parameters.settlingCsv.Log(parameters.clock, s)
        data.stokesCount = data.stokesTotal = data.stokesMax = 0
        data.brownianCount = data.brownianTotal = data.brownianMax = 0
                   
    return

###############################################################################
def DumpMatrixToCSV(parameters, matrix, fileName, allParticles):
    print("\nProcessing", fileName)
    tLog = MyCsv(parameters.baseDir + fileName, 10)

    for i in range(1, len(allParticles)):
        distance_SS = CalcDistance(allParticles[0], allParticles[i]) - (parameters.particleRad * 2)
        line = ("%e, %e, " % (distance_SS, matrix[0][i]))
        tLog.Log(0, line)
        
    print("max=", np.max(matrix[0, 1:len(allParticles)]))
    print("min=", np.min(matrix[0, 1:len(allParticles)]))
    
###############################################################################
###############################################################################
def RunSimulation(parameters, data):
    nudges = 0
    
    allParticles = data.allParticles
    #allPolymers = data.allPolymers
    downEscapies = data.downEscapies
    allSalts = data.allSalts
    allParcels = data.allParcels
    log = data.log
    
    thisIteration = 0;              # Number of times through sim loop
    lastIteration = 0               # iteration of last status line
    nextWrite = 1                   # next write of the dump file based on iteration
    nextWriteIncrement = 1
    
    nudgeMovementScale = .5         # percent of nudge factor to scale
    minClearanceScale = .05         # percent to scal of gap between particles variable
    
    data.log.Log("\nRunning Simulation")
    parameters.radiusOverTimeCsv.Log(parameters.clock,
                    "Clock, Solids Loading, BoxSize, AAR, " + \
                    "Iteration, Smallest Particle Radius, AAR, " + \
                    "Largest Particle Radius, Max. Polymer Radius, Max. Tangle Radius, " +
                    "agglomerationsAccepted, agglomerationsRejected, particlesDidNotApproach, " +
                    "last Dim, last AAF, Radius of Gyration, " +
                    "Average approach prob, Average escape prob")
    parameters.fluidViscositySolidsLoadingOverTimeCsv.Log(parameters.clock, \
                    "Clock, Solids Loading, Box Size, AAR, " + \
                    "Fluid Viscosity, Bins per Dimension, Polymers per Bin")  
    parameters.particalSizeDistributionCsv.Log(parameters.clock, "Clock, Solids Loading, Box Size, " + \
                    "1-N Particle Radius")  
    parameters.ellipsoidsOverTimeCsv.Log(parameters.clock, \
                    "Clock, Solids Loading, Box Size, AAR, " + \
                    "Ellipsoid count, Ellipsoid VolFrac, " + \
                    "Number of Captured Particles, Number of Tangles with Particles, Most Tangles in Ellipsoid")  
    parameters.debugCsv.Log(parameters.clock, \
                    "Clock, Solids Loading, Box Size, AAR, " + \
                    "average X Loc, average Y Loc, average Z Loc, " + \
                    "Largest Particle Radius, lengthOfSubparticleList, " \
                    "Total Ellipsoids that Merged, Max. WillFit retries, " \
                    "maxLenNpHoldAreClose, Viscosity Scaler" )  
    parameters.executionTimesCsv.Log(parameters.clock, \
                    "Clock, Solids Loading, Box Size, AAR, " + \
                    "RT Clock, Not accounted for, CheckForPolymerTangles, Stage One, Stage Two, " + \
                    "RunSimulation, CalcParticleBrownianMotion, CalcPolymerBrownianMotion, NudgeParticles")  
    parameters.settlingCsv.Log(parameters.clock, \
                    "Clock, Solids Loading, Box Size, AAR, " + \
                    "Stokes Count, Stokes Average, Stokes Max, " + \
                    "Brownian Count, Brownian Average, Brownian Max")

    if (parameters.generatePolymers): 
        parameters.tanglesOverTimeCsv.Log(parameters.clock, \
                    "Clock, Solids Loading, Box Size, AAR, " + \
                    "Polymer Count, Tangle Candidates, Tangled polymers, Tangles, " + \
                    "Most polymers in Tangle, Percent of polymers in tangles, " + \
                    "Polymers OOB, average polymer OOB distance, max polymer OOB distance")  
    
    # Make sure particals are sufficently apart from each other.  Must have some gap between particles
    if parameters.graphForcesTest == False:
        [nudged, failed] = NudgeParticles(parameters, data, parameters.particleRad * nudgeMovementScale, parameters.particleRad * minClearanceScale)
        nudges += nudged
        
    #make first log entry. forceEntry must be true to make sure it happens.
    LogToCsv(parameters, data, thisIteration, True)
    
    #bigDataCsv = MyCsv(parameters.baseDir + "bigData.csv", 0)
    #bigDataCsv.Log(parameters.clock,
    #                "Clock, Solids Loading, BoxSize, AAR, " + \
    #                "Iteration, dim, AAF")
                
    # initialize run time statistics timers
    startTime = time.time()
    lastStats = 0                   # last clock time status line was printed
    nextStats = parameters.statInterval
    lastStats = PrintStatusLine(parameters, data, log, lastStats, startTime, 0, thisIteration)
    
    # this is just for the visualization. it is not necessary for the code to run
    #end when clock runs out or if only one partical
    #DUG added exit condition of 1 partical in the list

    while (parameters.clock < parameters.simTime) and \
          (parameters.solidsLoading < 1.0) and \
          (data.lengthOfSubparticleList > (parameters.particleCount * .2)): 
            
        #if (parameters.clock > 30):
        #    raise ValueError("******\Error:  \nSet termination based on clock.  Program aborted.\n*****")
        startTime += int(CheckForSuspension(log))
        
        #print((time.time() - startTime), ">", nextStats)
        if ((time.time() - startTime) > nextStats):            
            deltaIt = thisIteration - lastIteration
            lastIteration = thisIteration

            RecalculateBoxSize(parameters, data, True)
            CreateNewPolymers(parameters, data)
            lastStats = PrintStatusLine(parameters, data, log, lastStats, startTime, thisIteration, deltaIt)
            nextStats = (time.time() - startTime) + parameters.statInterval
        
        if (parameters.writeParticlesEvery > 0) and (parameters.clock >= nextWrite):
            nextWrite += nextWriteIncrement
            nextWriteIncrement = min(nextWriteIncrement * 2.0, parameters.writeParticlesEvery) 
            
            fileName = ("%s_%04d.dump" % (parameters.fileNameBase, int(parameters.clock)))            
            #print("writing file", fileName)
            WriteDump(parameters, data, fileName)
        
        if (len(parameters.writeOnSL) > 0):
            if (parameters.solidsLoading >= float(parameters.writeOnSL[0])):
                fileName = ("%s_%s.dump" % (parameters.fileNameBase, float(parameters.writeOnSL[0])))
                del parameters.writeOnSL[0]            
                #print("writing file", fileName)
                WriteDump(parameters, data, fileName)
                     
        LogToCsv(parameters, data, thisIteration)
                     
        thisIteration += 1
        startTimerRunSimulator = timeit.default_timer()        
        agglomerationSucceded = False
            
        while (agglomerationSucceded == False) and (data.lengthOfSubparticleList > (parameters.particleCount * .2)):
            subParticleList = [] #particles that are smaller than the largest particle to aggregate
            subParticleListIndexes = []
                        
            for i in range(len(allParticles)):
                if (allParticles[i].r < data.largestParticleToAggregate):
                    subParticleList.append(allParticles[i])
                    subParticleListIndexes.append(i)
            
            data.lengthOfSubparticleList = len(subParticleList)
            
            overlappingElements= CalcOverlapParticleParcelPair(parameters, data)
            p1List = overlappingElements[0] # creatse a list of only the particles
            overlappedParticleList = [] #particles that are in or touching the parcel
            particlesinParcel = []
            
                
        
            if len(p1List)!=0: #if particles are touching the parcel
                for k in range(parameters.numParcels): #iterates throughe each parcel
                    for mm in range(len(p1List)): #iterates through each particle touching the parcel
                        i = p1List[mm] #index of the particle in allParticles that is touching th parcel
                        j = overlappingElements[1][mm] #index of the parcel that is touching the particle
                        if j ==k:
                            particlesinParcel.append(allParticles[i]) #particles in that parcel
                            overlappedParticleList.append(i) #index of particles that are specifically in that parcel
                    [xj,yj,zj] = allParcels[j].GetLocation() #find the location of a specific parcel
                    parcelRad = ((.64*parameters.boxSize*3)/(4*parameters.numParcels*math.pi))**(1/3)
                    CreateSalts(parameters,data,[xj,yj,zj],parcelRad) #creatse the number of salts in one parcel
    
                    distanceMatrix = CalcParcelDistanceMatrix(parameters, particlesinParcel, allSalts)
                    np.fill_diagonal(distanceMatrix, 1.0)  # this will keep particles away from self
                    distanceSqMatrix = np.square(distanceMatrix)
            #np.square just squares the individual elements
            
            # Van der Waals
                    VdW = CalcVdWMatrix(parameters, particlesinParcel,allParticles, allSalts, distanceSqMatrix, False)
                    interactMatrix = VdW
            
            # Add V_s 
                    if parameters.graphForcesTest == False:   
                        if parameters.V_sCalc == True:            #not being calculated             
                            V_s = CalcV_sMatrix(parameters, overlappedParticleList)
                            interactMatrix = np.add(interactMatrix, V_s)
            
            # #polymer Interact 
            # V_d is build from three equations to each describing what happens if particles are close to far.
            # need to build final matrix using the 3 distance matrices (V_da, V_db, and V_dc)
            #
            # Equation for code came from the paper "Depletion Stabilization in 
            # Nanoparticles-Polymer Suspensions: Multi-Length-Scale Analysis of Microstructure"
                if parameters.V_dCalc == True:
                    V_d = CalcV_dMatrix(parameters, overlappedParticleList, distanceMatrix)
                    interactMatrix = np.add(interactMatrix, V_d)
                
                EDL = EDLMatrix(parameters, overlappedParticleList, allSalts)
            
                interactMatrix = np.add(interactMatrix, EDL) 
            
            #DumpMatrixToCSV(parameters, VdW,  "matrix_VdW.csv", data.allParticles)
            #DumpMatrixToCSV(parameters, V_s,  "matrix_V_s.csv", data.allParticles)
            #DumpMatrixToCSV(parameters, V_d,  "matrix_V_d.csv", data.allParticles)
            #DumpMatrixToCSV(parameters, interactMatrix,  "matrix_interact.csv", data.allParticles)
            
                if parameters.graphForcesTest:            
                    DumpMatrixToCSV(parameters, VdW,  "matrix_VdW.csv", data.allParticles)
                    #DumpMatrixToCSV(parameters, V_s,  "matrix_V_s.csv", data.allParticles)
                    DumpMatrixToCSV(parameters, V_d,  "matrix_V_d.csv", data.allParticles)
                    DumpMatrixToCSV(parameters, interactMatrix,  "matrix_interact.csv", data.allParticles)
                    raise ValueError("******\graph Force test over.  See matrix*.csv for results. program Terminated\n*****")
            
                [i, j, averageAggFreq] = CalcTwoParticlesToAgglomerate(parameters, particlesinParcel, allSalts,interactMatrix)
                [agglomerationSucceded, newMaxRadius] = AggregateTwoParticles(parameters, data, particlesinParcel, overlappedParticleList[i], overlappedParticleList[j], interactMatrix[i][j])
            #need to change aggregateTwoParticles 
                if (agglomerationSucceded == False):
                    data.largestParticleToAggregate =  min(data.largestParticleToAggregate, newMaxRadius)
                    data.log.Log("****\nAgglomeration failed: New Minimum Radius= %e\n****" % (newMaxRadius))

            dim = ((len(allParticles)/(parameters.boxSize**3)) * len(allParticles) * averageAggFreq)
            data.lastDim = dim
            data.lastAAF = averageAggFreq
            parameters.clock += (2.0 / dim)
        
        #s = ("%f, %f, %e, %e, %d, %e, %e" %
        #        (parameters.clock, parameters.solidsLoading, parameters.boxSize, CalcAverageAggRad(data.allParticles),
        #        thisIteration, dim, averageAggFreq))
        #bigDataCsv.Log(parameters.clock, s)
        
            parameters.timeInRunSimulation += timeit.default_timer() - startTimerRunSimulator
        
            # Modify locations per Brownian motion
            CalcParticleBrownianMotion(parameters, data, 2.0 / dim)  
        # CalcPolymerBrownianMotion(parameters, data, 2 / dim)   # Done in CheckForPolymerTangles      
            SettlementParticleCheck(parameters, data)
        
            CheckForOutOfBox(parameters, allParticles)
            
            #add parcel Brownian Motion
            CalcParcelBrownianMotion(parameters,data,2.0/dim)
            #add Settlement parcel check
            SettlementParcelCheck(parameters, data)
            #check for out of box for parcel
            CheckForOutOfBox(parameters, allParticles)

                        
            interactMatrix = None  # release memory
            distanceMatrix = None  # release memory
            distanceSqMatrix = None  # release memory
            allSalts = None
            overlappedParticleList = None #release memory
        ## DUG
        ## Need to make sure no two particles are overlapping to each other
        ## If they are move them + or - in x, y, or z direction until
        ## there are no overlaping particals
        
        # Must have some gap between particles    
            RecalculateBoxSize(parameters, data, False)
            parameters.smallestParticleRadius = CalculateSmallestRadius(parameters, allParticles)
            parameters.largestParticleRadius = CalculateLargestRadius(parameters, allParticles)
        
        #if parameters.runningTotalMaxBrownian > parameters.monomerRad:
        #    CreatePolymers(parameters, log, allParticles, allPolymers, polymerBins)
        #    print("Checking for tangles %e" % (parameters.runningTotalOfBrownian))
        #    lastTangleCount = CheckForPolymerTangles(parameters, log, allParticles, allPolymers, polymerBins, downEscapies)
        #    parameters.runningTotalOfBrownian = 0
        #else:
            #print("Not checking for tangles %e" % (parameters.runningTotalMaxBrownian))
            [nudged, failed] = NudgeParticles(parameters, data, parameters.smallestParticleRadius * nudgeMovementScale, parameters.particleRad * minClearanceScale)
            nudges += nudges
    
        print("RunSimulation: Loop Termination Conditions. clock= %f, SL= %f, lengthOfSubparticleList= %d" % \
              (parameters.clock, parameters.solidsLoading, data.lengthOfSubparticleList))
            
        if (parameters.clock < parameters.simTime) and (parameters.solidsLoading < 1.0) and (parameters.drying):
                data.log.Log("****\nTo many large particles..... Program terminated early.\n****")
                parameters.clock = 1505.19 + parameters.dryingDelay # will cause RecalculateBoxSize to set solidsloading to 1.00003
                RecalculateBoxSize(parameters, data, True)

        while (downEscapies.GetQueueLen() > 0):        
            aggregate = downEscapies.PopAgglomerate()
            aggregate.SetRandomLocation(parameters)
            allParticles.append(aggregate)
        
        parameters.smallestParticleRadius = CalculateSmallestRadius(parameters, allParticles)
        [nudged, failed] = NudgeParticles(parameters, data, parameters.smallestParticleRadius * nudgeMovementScale, parameters.particleRad * minClearanceScale)
        nudges += nudged

    # final log to CSV file
    LogToCsv(parameters, data, thisIteration, True)
        
    #print out the final program status line
    deltaIt = thisIteration - lastIteration
    PrintStatusLine(parameters, data, log, lastStats, startTime, thisIteration, deltaIt)
    
    stopTime = time.time()
    data.log.Log("Simulation finished in %s H:M:S," % (Sec2HMS(stopTime - startTime)))
    
##############################################################################
def InitializeProgram(parameters, data):
    #calculate the number of particles to generate.AsciiBins
    maxParticlesAllowed = 15.0e3
            
    if parameters.particleCount > maxParticlesAllowed:
        data.log.Log("Calculated %d particles to generate.  %d is maximum allowed." % (parameters.particleCount, maxParticlesAllowed)); 
        parameters.particleCount = maxParticlesAllowed
        
    parameters.polymersToGenerate = int(parameters.particleCount * parameters.chainsPerParticle)

    data.log.Log("%d particles to generate. VolFrac = %f" % (parameters.particleCount, parameters.particleVolFrac))
    data.log.Log("%d polymers to generate. VolFrac = %f" % (parameters.polymersToGenerate, parameters.polymerVolFrac))  
        
    # Set to false while I'm testing reading input parameters and don't want the
    # parameters to be used.  Of couse they must also be valid
    if (parameters.valid):
        # Generate the two types of atom groups
        CreateParticles(parameters, data)
        CreateParcels(parameters, data)
        data.log.Log("Creating Polymers.")
        CreateNewPolymers(parameters, data)
        data.log.Log("Finished creating Polymers.")
        RecalculateBoxSize(parameters, data, True)
        
        if (parameters.saveInitialAtoms):
            WriteDump(parameters, data, (parameters.fileNameBase + "_InitialAtoms.dump"))
                        
        return(True)
    else:
        s = ("\n*******\n  Warning:\n  Invalid parameter file.  Fix before program will run\n*******\n"); 
        data.log.Log(s) 
        return(False)

##############################################################################
def PrintRunTimes(parameters, whichLog):
    whichLog.Log("\nTimeing stats with simulation at clock = %s seconds" % (FormatIt(parameters.clock, 5)))
    whichLog.Log("  Spent %s H:M:S in RecalculateBoxSize" % (Sec2HMS(parameters.timeInRecalculateBoxSize)))
    whichLog.Log("  Spent %s H:M:S in CheckForPolymerTangles" % (Sec2HMS(parameters.timeInCheckForPolymerTangles)))
    whichLog.Log("    Spent %s H:M:S in CalcPolymerBrownianMotion" % (Sec2HMS(parameters.timeInCalcPolymerBrownianMotion)))
    whichLog.Log("  Spent %s H:M:S in CalcParticleBrownianMotion" % (Sec2HMS(parameters.timeInCalcParticleBrownianMotion)))
    whichLog.Log("  Spent %s H:M:S in RunSimulation" % (Sec2HMS(parameters.timeInRunSimulation)))
    return
##############################################################################
######################  Start of Program  #################################### 
##############################################################################
args = sys.argv[1:]

      
if len(args) == 0:
    args.append("runParameters.txt")

randomSeed = random.randrange(sys.maxsize)
#randomSeed = 583422507507101411; print("\n******\nFixed seed (%d) being used\n******\n" % (randomSeed))
random.seed(randomSeed)

for arg in args:   
    # Declare base variables
    data = Data()
    data.arg = arg
    data.allParticles = []
    data.allSalts = []
    data.allParcels = []
    data.allPolymers = []
    data.allEllipsoids = []
    data.newPolymers = []
    data.pgmStart = timeit.default_timer()
    data.timeInStageOne = 0
    data.timeInStageTwo = 0
    
    parameters = FileParameters()
    parameters.ReadParametersFromFile(arg)
    
    data.log = MyLogger(parameters.fileNameBase + ".txt", LogDestination.BOTH)     # Need to figure out how system logger works
    data.debugLog = MyLogger(parameters.fileNameBase + "_debugLog.txt", LogDestination.FILE)
               
    data.log.Log("Program Started", LogDestination.BOTH, True)      # We are off and running
    data.log.Log("Python version:\n %s" % (sys.version))
    data.log.Log("Program version: 2.09")
    data.log.Log("The random number generator seed is %d" % (randomSeed))
    data.log.Log("parameters file is %s." % (arg))
    if ((parameters.availableCpus != 1) & (parameters.generatePolymers)):
        data.log.Log("Initializing multi-core processing")
        ppservers = ()

        data.job_server = pp.Server("autodetect", ppservers=ppservers)
        #data.log.Log("Number of CPU detected is %d." % (data.job_server.get_ncpus()))
        data.job_server.set_ncpus(parameters.availableCpus)
        #data.log.Log("Number of CPU used is %d." % (data.job_server.get_ncpus()))
    
    parameters.PrintParameters(data)
    data.downEscapies = HoldAggregates(parameters.downEscapedAggregates)
    data.capturedParticles = HoldAggregates(0)
    data.polymerBins = PolymerBins(parameters)
    data.lengthOfSubparticleList = parameters.particleCount
    data.npHoldAreClose = None
    
    # Initialize program
    if (InitializeProgram(parameters, data)):    
        # Finally what we are here to do
        if (parameters.valid):
            RecalculateBoxSize(parameters, data)
            RunSimulation(parameters, data)
            fileName = parameters.fileNameBase + "_roughness.csv"
            #CalculateRoughness(parameters, data, fileName, range(95, 0, -5))
            
            if (parameters.drying):
                
                if (parameters.generatePolymers):
                    os.remove(CreatePolymerFileName(parameters.tempDir)) 
            
            # add any captured particles to allParticles    
            while data.capturedParticles.GetQueueLen() > 0:
                print("poping captured particle")
                data.allParticles.append(data.capturedParticles.PopAgglomerate())
                
            # Dump atoms to a file
            WriteDump(parameters, data, (parameters.fileNameBase + '.dump'))
        else:
            if (parameters.valid == False):
                data.log.Log("** Invalid parameters. Program execution aborted") 
    else:
        data.log.Log("*******\nProgram Initialization failed. Program execution aborted.\n*******")
    
    # Terminate the program
    pgmStop = timeit.default_timer()
    
    data.debugLog.Log("\nFinal timer Stastics:")
    PrintRunTimes(parameters, data.debugLog)
    other = (pgmStop - data.pgmStart) - \
                      (parameters.timeInRecalculateBoxSize + \
                       parameters.timeInCalcParticleBrownianMotion + \
                       parameters.timeInCalcPolymerBrownianMotion + \
                       parameters.timeInRunSimulation) 
    data.debugLog.Log("    Spent %s H:M:S on other Parts" % (Sec2HMS(other)))
    
    data.debugLog.Log("  Final total particles = %d. VolFrac = %f" % (CalculateTotalParticle(parameters, data.allParticles), parameters.particleVolFrac))   
    data.debugLog.Log("  Final polymers = %d. VolFrac = %f" % (parameters.polymersToGenerate, parameters.polymerVolFrac))
     
    if (parameters.generatePolymers & (parameters.numberOfPolymerBrownian != 0)):
        data.debugLog.Log("  Maximum any polymer moved due to Brownian motion = %e" % (parameters.maxPolymerBrownian))
        data.debugLog.Log("  Average all polymer moved due to Brownian motion = %e" % (parameters.totalPolymerBrownian / parameters.numberOfPolymerBrownian))
       
    
    data.log.Log("")
    data.log.Log("Program executed in %s H:M:S." % (Sec2HMS(pgmStop - data.pgmStart)))
    data.log.Log("Program Terminated", LogDestination.BOTH, True)
    data.log.Close()
    data.debugLog.Close()
    
    if ((parameters.availableCpus != 1) & (parameters.generatePolymers)):
        data.job_server.destroy()