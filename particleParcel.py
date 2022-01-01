# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:14:10 2021

@author: allis
"""
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

###############################################################################
def CalcDistance(i, j):
    [x1, y1, z1] = i.GetLocation()
    [x2, y2, z2] = j.GetLocation()
    distance=math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return(distance)

##############################################################################
def CalcProductRadiusMatrix(parameters, allParticles, subParticleList, allSalts):
    #multiplies the two radii together
    particles_rLst = np.full(len(subParticleList), 0, dtype = parameters.doMathIn)
    #creates an empty list the size of the subParticcleList
    salt_rLst = np.full(len(allSalts), 0, dtype = parameters.doMathIn)   
    
    for k in range(len(subParticleList)):
       particles_rLst[k] = allParticles[subParticleList[k]].r #subParticeList contains indexes of th particles in all Particles that touch the parcel
       salt_rLst = allSalts[k].r
            
    matrix = np.multiply.outer(particles_rLst, salt_rLst)
    return(matrix)
#used in the VdW matrix
###############################################################################
def CalcSumRadiusMatrix(parameters, allParticles,subParticleList, allSalts):
    particle_rLst = np.full(len(subParticleList), 0, dtype = parameters.doMathIn)
    salt_rLst = np.full(len(allSalts), 0, dtype = parameters.doMathIn)
    for k in range(len(subParticleList)):
        particle_rLst[k] = allParticles[subParticleList[k]].r
        salt_rLst = allSalts[k].r
        
    matrix = np.add.outer(particle_rLst, salt_rLst)
    return(matrix)

#used in the VdW matrix
##############################################################################
def CalcDifferenceRadiusMatrix(parameters, allParticles, subParticleList, allSalts):
    particle_rLst = np.full(len(subParticleList), 0, dtype = parameters.doMathIn)
    salt_rLst = np.full(len(allSalts), 0, dtype = parameters.doMathIn)
    
    for k in range(len(subParticleList)):
        particle_rLst[k] = allParticles[subParticleList[k]].r
        salt_rLst[k] = allSalts[k].r
    
    matrix = np.subtract.outer(particle_rLst, salt_rLst)
    return(matrix)
#used in the VdW matrix
###############################################################################
def CalcVdWMatrixWorker(parameters, allParticles,subParticleList, partRadProd, partRadSum, partRadDiff, distanceSqMatrix, clearNaNErrors = False):
    A = parameters.A
    #np.reshape(partRadProd,(10,10)) #had to reshape because it was outputting a (10,10) and (1000,) matrix    
    """temp1 = np.multiply(partRadProd, 2.0)
    temp2 = np.square(partRadSum)
    temp3 = np.square(partRadDiff)
    temp4 = distanceSqMatrix"""
    #we do not want all of the particle to be calcualted on ly the one that are in the contact with the parcel
    temp1 = CalcProductRadiusMatrix(parameters, allParticles, subParticleList, data.allSalts)
    temp2 = np.square(CalcSumRadiusMatrix(parameters, allParticles, subParticleList, data.allSalts))
    temp3 = np.square(CalcDifferenceRadiusMatrix(parameters, allParticles,subParticleList, data.allSalts))
     
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
    for i in subParticleList:
        partPRads.append(part.r for part in allParticles(i))
    #particle radii of only the particles touching that specific parcel
    partSRads = [part.r for part in allSalts] #salts radii
    
    partRadDiff = np.subtract.outer(partPRads, partSRads)
    partRadSum = np.add.outer(partPRads, partSRads) 
    partRadProd = np.multiply.outer(partPRads, partSRads)
    
#put in subParticle list because we do not want all of hte particles to b calclated      
    VdW = CalcVdWMatrixWorker(parameters, allParticles, subParticleList,partRadProd, partRadSum, partRadDiff, distanceSqMatrix, clearNaNErrors)

    return(VdW)
 
###############################################################################
###############################################################################
def CalcSSDistanceMatrix(parameters, allParticles, allSalts):   
    partRadSum = CalcSumRadiusMatrix(parameters, allParticles, allSalts)    
    distanceMatrix=  CalcDistanceMatrix(parameters, allParticles, allSalts)    
    gapMatrix = np.subtract(distanceMatrix, partRadSum)
    return(gapMatrix)

##############################################################################
def calcDybeLength(parameters):
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
###############################################################################
def CalcNaturalLogMatrix(parameters, allParticles, allSalts):
    #ln(1+e^(-kH))
    SSDistanceMatrix = CalcSSDistanceMatrix(parameters, allParticles, allSalts)
    kappa = calcDybeLength(parameters)
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
##############################################################################
def CalcRadDivisionMatrix(parameters, allParticles, allSalts):
        #(ri*rj)/(ri+rj)
    sumRadiusMatrix = CalcSumRadiusMatrix(parameters, allParticles, allSalts)
    productRadiusMatrix = CalcProductRadiusMatrix(parameters, allParticles, allSalts)
        
    radDivisionMatrix = np.divide(productRadiusMatrix,sumRadiusMatrix)
    return(radDivisionMatrix)

##############################################################################
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
def CalcDistanceMatrix(parameters, allParticles, allSalts):   
    # do a quick check to see if any nudging needs to be done
    # this is generally the case.... saves a lot of time
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
        [s_xLst[k], s_yLst[k], s_zLst[k]] = allSalts[k].GetLocation()
        
    xSqDistMatrix = np.square(np.subtract.outer(xLst, s_xLst))
    ySqDistMatrix = np.square(np.subtract.outer(yLst, s_yLst))
    zSqDistMatrix = np.square(np.subtract.outer(zLst, s_zLst))
            
    distanceMatrix=   np.sqrt(np.add(xSqDistMatrix, np.add(ySqDistMatrix, zSqDistMatrix))) 
    return(distanceMatrix)

###############################################################################

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
    particlesPerParcel = 20 #testing to see if locations are given to  particles
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
        
  
#####################################################################################
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
    allSalts = data.allSalts
    redo = True
    
    startTimer = timeit.default_timer()
    
    while (redo & (failed == False)):
        redo = False
        
        # Check for overlaping particles
        overlappingElements = CalcOverlapParticles(parameters, allParticles, allSalts, minimumOverlapGap, False)        
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
###############################################################################
def CalcTwoParticlesToAgglomerate(parameters, allParticles,interactMatrix):
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
#############################################################################
def CalcAggregateRadius(parameters,newC):
    #how to account for salt rad
    volumeAllParticles = newC * (4.0/3.0) * math.pi * (parameters.particleRad ** 3)
    newR = ((volumeAllParticles / parameters.packingFactor) / (math.pi * (4.0/3.0))) ** (1.0/3.0)
    return(newR)
###############################################################################
def WillTwoParticlesNotReject(parameters, data, i, j, currentPotential):
    numberDivisions = parameters.particleCount
    
    ri = data.allParticles[i].GetRadius()    
    rj = data.allSalts[j].GetRadius()
    SLT = parameters.stericLayerThickness
    
    SS_adjustment = ri + rj + (SLT * 2.0)
    SS_distance = CalcDistance(data.allParticles[i], data.allSalts[j]) - SS_adjustment
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
    
        
        distanceMatrix = CalcDistanceMatrix(parameters, privateParticleList, allSalts)
        np.fill_diagonal(distanceMatrix, 1.0)  # this will keep particles away from self
        distanceSqMatrix = np.square(distanceMatrix)    
        VdW = CalcVdWMatrix(parameters, privateParticleList, allSalts, distanceSqMatrix, True)
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
            
        EDL = EDLMatrix(parameters, data.allParticles, data.allSalts)
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
def AggregateTwoParticles(parameters, data, allSalts, subParticleList, i, j, currentPotential): 
    allParticles = data.allParticles
    
    newC=(allParticles[i].c + allSalts[j].c)
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
def DumpMatrixToCSV(parameters, matrix, fileName, allParticles):
    print("\nProcessing", fileName)
    tLog = MyCsv(parameters.baseDir + fileName, 10)

    for i in range(1, len(allParticles)):
        distance_SS = CalcDistance(allParticles[0], allParticles[i]) - (parameters.particleRad * 2)
        line = ("%e, %e, " % (distance_SS, matrix[0][i]))
        tLog.Log(0, line)
        
    print("max=", np.max(matrix[0, 1:len(allParticles)]))
    print("min=", np.min(matrix[0, 1:len(allParticles)]))
#####################################################################################

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
    data.allParcels =[]
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
    
allParticles = data.allParticles
allParcels = data.allParcels
allSalts = data.allSalts

ParcelRad = ParcelRad(parameters)

CreateParticles(parameters,data)
CreateParcels(parameters,data)

###############################################################################
overlappingElements = CalcOverlapParticleParcelPair(parameters, data)
p1List = overlappingElements[0] #creates a list of only the particles
#p1List = [1 ,50] #check to see if it works
subParticleList = []

if len(p1List) !=0: #creating too many salts because creates salts for every particle touching the parcel
    for k in range(parameters.numParcels):
        for mm in range(len(p1List)):
            i = p1List[mm] #index of the particle in allParticles that is toucing the parcel
            j = overlappingElements[1][mm] #index of the parcel that is touching the particle
            if j ==k:
                subParticleList.append(i) #particles that acting specifically with that parcel
        #need to determine which particles contact the same parcels
        
        [xj,yj,zj] = allParcels[j].GetLocation() #findin the location of a specific parcel
        
        parcelRad =  ((.64*parameters.boxSize*3)/(4*parameters.numParcels*math.pi))**(1/3)
    
        CreateSalts(parameters,data, [xj,yj,zj], parcelRad) #creates the number of salt in one parcel
    
        #calc DistanceMatrix of particles in contact with the parcel and the salts inside
        distanceMatrix = CalcDistanceMatrix(parameters,subParticleList, allSalts) 
        
    #VdW matrix #van der Waals matrix for just that parcel
        VdW = CalcVdWMatrix(parameters,subParticleList, allSalts)
        interactMatrix = VdW #calc EDL matrix
    #set VdW matrix equal to the interact matrix
    
        EDL = EDLMatrix(parameters,subParticleList, allSalts)
        interactMatrix = np.add(interactMatrix,EDL)
        #add EDL matrix to the interact matrix
        
        #see if there is any agglomeration
        if parameters.graphForcesTest:            
            DumpMatrixToCSV(parameters, VdW,  "matrix_VdW.csv", data.allParticles)
        #DumpMatrixToCSV(parameters, V_s,  "matrix_V_s.csv", data.allParticles)
       
            DumpMatrixToCSV(parameters, interactMatrix,  "matrix_interact.csv", data.allParticles)
            raise ValueError("******\graph Force test over.  See matrix*.csv for results. program Terminated\n*****")
            
        [i, j, averageAggFreq] = CalcTwoParticlesToAgglomerate(parameters, subParticleList, interactMatrix)
        [agglomerationSucceded, newMaxRadius] = AggregateTwoParticles(parameters, data, allSalts,subParticleList, subParticleListIndexes[i], subParticleListIndexes[j], interactMatrix[i][j])
        #how to concatanate a matrix
    subParticleList = None #release the memory of all sub particles
    allSalts = None #release the memory of all salts because different salts in each particle
    
    
            
    #calc particle Brownian motion
    
    #cal parcel Brownian motion
    #check for particle settling
    #check for parcel settling
    #check to see if particle or parcel is out of the box
    #clear memory of the interact matrix, distance matrix, and distanceSqMatrix
    #recalculate box size
    #calculate smallest radius
    #calculate largest radius
    #nudge particles
    #look at the rest of the steps in run simulation
    
    
    


