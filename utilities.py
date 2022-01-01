# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle

Editted 6/21/2021
Allison Wandling
"""
import random
import math
import timeit

###############################################################################
def FormatNumber(number):
    units = ["K", "M", "G", "T", "Q"]
    ui = -1
    on = ""
    
    if (number < 1000):
        on = ("%d" % (number))
    else:
        while number >= 1000:
            ui += 1
            number /= 1000.0
        
        if (number >= 100):
            on = ("%5.1f%s" % (number, units[ui]))
        elif (number >= 10):
            on = ("%5.2f%s" % (number, units[ui]))
        else:
            on = ("%5.3f%s" % (number, units[ui]))
            
    return(on)
    
###############################################################################
def CalculateRadiusGyration(polymerLength):
    #R_g = 1.7e-9 * polymerLength**(3.0/5.0)
    #R_g = 3e-9
    R_g = (polymerLength/6)**(3.0/5.0) * 1.5e-10
    return(R_g)
    
###############################################################################
def CalculatePolymerLength(parameters):  
    if (parameters.variableLengthPolymers):
        polymerTimeFactor = min(parameters.clock / parameters.polymerizationTime, 1.0)
        polymerLength = int(parameters.polymerLength * polymerTimeFactor + 1.0) 
    else:
        polymerLength = parameters.polymerLength
        
    return(min(polymerLength, parameters.polymerLength))
    
###############################################################################
def SetLatticeConstant(latticeConstant, x, y, z):
    """try:
        x = latticeConstant * (int(x / latticeConstant) + .5)
        y = latticeConstant * (int(y / latticeConstant) + .5)
        z = latticeConstant * (int(z / latticeConstant) + .5)
    except:
        print("SetlatticeConstant: x=%f, y=%f, z=%f" % (x, y, z))
        raise""" #lattice Constant = 0
    return(x, y, z)
    
#############################################################################
def Sec2HMS(x, withFraction = False):  
    seconds = int(x)
    fraction = x - seconds
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    hours = int(minutes / 60)
    minutes = minutes % 60
    days = int(hours / 24)
    hours = int(hours % 24)
    
    if days == 0:
        HMS =        '{:2d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    else:
        HMS = '{:d}:{:02d}:{:02d}:{:02d}'.format(days, hours, minutes, seconds)
        
    if withFraction:
        HMS = ("%s.%02d" %(HMS, int(fraction * 100)))
        
    return(HMS)
    
###############################################################################
def CreatePolymerFileName(tempDir):
    fn = ("%s/allBasePolymers.npy" % (tempDir))        
    return(fn)
###############################################################################
def CalculateVectorLength(x, y, z):
    # returns the length of the vector
    vector = math.sqrt(x*x + y*y + z*z)
    return(vector)
    
###############################################################################
def NormalizeVector(scale, x, y, z):
    # normalize the vector to the nudgeDistance
    vectorLength = math.sqrt(x*x + y*y + z*z)
    scaleFactor = scale / vectorLength
    x *= scaleFactor
    y *= scaleFactor
    z *= scaleFactor
    return(x, y, z)
    
###############################################################################
def CreateRandomVector(parameters, scale):
    xVal = (random.random() - .5)
    yVal = (random.random() - .5)
    zVal = (random.random() - .5)
    
    [xDistance, yDistance, zDistance] = NormalizeVector(scale, xVal, yVal, zVal)
    
    return(xDistance, yDistance, zDistance)
    
###############################################################################
def CalculateTotalParticle(parameters, allParticles):
    return(sum(particle.c for particle in allParticles))
    
###############################################################################
def GetParticleIndexesLessThen(parameters, allParticles, size):
    list = [x for x in range(len(allParticles)) if allParticles[x].r < size]
    return(list)
    
###############################################################################
def RecalculateBoxSize(parameters, data, alsoDoPolymers = False):
    if parameters.graphForcesTest:
        return
    
    startTime = timeit.default_timer()
    
    allParticles = data.allParticles
    allPolymers = data.allPolymers
    allEllipsoids = data.allEllipsoids
    
    totalParticles = sum(particle.c for particle in allParticles) + data.downEscapies.CalcTotalParticles()
    #total particles = allParticles + allSalts - escapedParticles
    oldBoxSize = parameters.boxSize
    
    CalculateVolFracs(parameters)
    newBoxSize = ((totalParticles * ((4.0/3.0) * math.pi * parameters.particleRad**3)) / parameters.particleVolFrac) ** (1.0/3.0)
    #print("total particles= %d, newBoxSize= %10.3e" % (totalParticles, newBoxSize))
    
    parameters.boxSize = newBoxSize
    scaler = newBoxSize / oldBoxSize
    
    if (newBoxSize != oldBoxSize):
        #print("newBoxSize= %e" % (newBoxSize))
        if scaler == 0:
            print("len(allParticles)= %e" % (len(allParticles)))
            print("len(allPolymers)= %e" % (len(allPolymers)))
            print("totalParticles= %e" % (totalParticles))
            print("parameters.particleRad= %e" % (parameters.particleRad))
            print("parameters.particleVolFrac= %e" % (parameters.particleVolFrac))
            print("parameters.polymerVolFrac= %e" % (parameters.polymerVolFrac))
            raise ValueError("******\Error:  \nRecalculateBoxSize.Particles: New box size is zero.  Program aborted.\n*****")
        
        for i in range(len(allParticles)):
            [x, y, z] = allParticles[i].GetLocation()
            
            if False:
                radius = allParticles[i].GetRadius() * parameters.particleRadScale
                tScale = (newBoxSize -  radius * 2.0) / (oldBoxSize -  radius * 2.0)
                x = ((x - radius) * tScale) + radius
                y = ((y - radius) * tScale) + radius
                z = ((z - radius) * tScale) + radius
            else:
                x *= scaler
                y *= scaler
                z *= scaler
                
            allParticles[i].SetLocation(parameters, x, y, z)
            
        for i in range(len(allEllipsoids)):
            allEllipsoids[i].ScaleLocation(scaler)
        
    if alsoDoPolymers:
        oldBoxSize = parameters.oldPolymerBoxSize 
        scaler = newBoxSize / oldBoxSize
        parameters.oldPolymerBoxSize = newBoxSize
        
        if scaler == 0:
            print("len(allParticles)= %e" % (len(allParticles)))
            print("len(allPolymers)= %e" % (len(allPolymers)))
            print("totalParticles= %e" % (totalParticles))
            print("parameters.particleRad= %e" % (parameters.particleRad))
            print("parameters.particleVolFrac= %e" % (parameters.particleVolFrac))
            print("parameters.polymerVolFrac= %e" % (parameters.polymerVolFrac))
            raise ValueError("******\Error:  \nRecalculateBoxSize.Polymers: New box size is zero.  Program aborted.\n*****")
        
        if (newBoxSize != oldBoxSize):
            polymerNeedsProcessing = [True] * len(allPolymers)
            
            xOriginal = yOriginal = zOriginal = xNew = yNew = zNew = 0
            
            for i in range(len(allPolymers)):
                if (polymerNeedsProcessing[i] == True):
                    try:
                        tangleSet = allPolymers[i].GetTangleSet(i)
                        tangleList = list(tangleSet)
                        [xOriginal, yOriginal, zOriginal] = allPolymers[i].GetStartingLocation()
                        allPolymers[i].ScaleLocation(parameters.latticeConstant, scaler)
                        [xNew, yNew, zNew] = allPolymers[i].GetStartingLocation()
                        polymerNeedsProcessing[i] = False
                    except:
                        print("i= %d, scaler= %e" % (i, scaler))
                        print("xOriginal=%e, yOriginal=%e, zOriginal=%e" % (xOriginal, yOriginal, zOriginal))
                        print("xNew=%e, yNew=%e, yNew=%e" % (xNew, yNew, zNew))
                        raise ValueError("******\Error:  \nRecalculateBoxSize.Polymers: Tangle list had a problem.  Program aborted.\n*****")

                    if (len(tangleList) > 1):
                        try:
                            xDelta = xNew - xOriginal
                            yDelta = yNew - yOriginal
                            zDelta = zNew - zOriginal
                
                            for j in range(len(tangleList)):
                                if (j != i):
                                    allPolymers[j].DeltaMovePolymer(parameters, xDelta, yDelta, zDelta)
                                    polymerNeedsProcessing[j] = False
                        except:
                            print("i= %d, j= %d" % (i, j))
                            print("xOriginal=%f, yOriginal=%f, zOriginal=%f" % (xOriginal, yOriginal, zOriginal))
                            print("xNew=%f, yNew=%f, yNew=%f" % (xNew, yNew, zNew))
                            raise
                        
            #data.polymerBins.BoxResized(parameters, data.allPolymers, 0, data.beginingOfNewPolymers)
            data.polymerBins.BoxResized(parameters, data.allPolymers, 0, len(allPolymers))
        
    parameters.timeInRecalculateBoxSize += timeit.default_timer() - startTime
    return

###############################################################################
def CalculateVolFracs(parameters):
    
    if (parameters.drying):
        x = max(parameters.clock - parameters.dryingDelay, 0.0)
    else:
        x = 0.0
        
    # Solid loading changes over time.  Calculate current solids loading
    # initial solidsLoading should be 0.02
    parameters.solidsLoading = (-2e-10*(x**3)) + (4e-7*(x*x)) + (0.0005*x) + parameters.initialSolidsLoading
    _CalculateFluidViscosity(parameters)  # Dependent on solidsLoading
    
    # solidsLoading = particleVolFrac + polymerVolFrac
    parameters.polymerVolFrac = parameters.polymerPercent * parameters.solidsLoading
    parameters.particleVolFrac = parameters.solidsLoading - parameters.polymerVolFrac    
        
    return

###############################################################################
def _CalculateFluidViscosity(parameters):
    if (parameters.drying):
        x = parameters.solidsLoading * parameters.polymerPercent
    else:
        x = 0.0
        
    parameters.viscosityScaler = 0
    #parameters.fluidViscosity = ((167.55 * (x*x)) + (0.0 * x) + 0.00098) * parameters.viscosityScaler
    parameters.fluidViscosity = .00890 #fluid viscosity of water
      
    return
    
###############################################################################
def CalculateSmallestRadius(parameters, allParticles):
    return(min(particle.r for particle in allParticles))
    
###############################################################################
def CalculateLargestRadius(parameters, allParticles):
    return(max(particle.r for particle in allParticles))
    
###############################################################################
