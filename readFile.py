# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017variableLengthPolymers

@author: Michelle

Editted 6/21/2021
by Allison Wandling
"""
import multiprocessing
import math
import os
import shutil
from shutil import copyfile

from myCsv import MyCsv
from myLogger import LogDestination
from utilities import CalculateVolFracs

###############################################################################
class FileParameters:
    
    # set defaults
    def __init__(self):
        self.boxSize = 5
        self.oldPolymerBoxSize = self.boxSize
        self.particleCount = 2,000
        # self.hamaker =	8.0e-14	# Hamaker constant of polyethylene in water
        self.A =	8.0e-21	 
        self.temperature = 300         # temperature in K
        self.statInterval = 60	# Number of seconds between sim status reports to console
       
        self.particleRad = 500e-9
        self.PEDensity = .975
        self.FeRad = 63E-12
        self.FeCl3Density = 2900 #in kg/m^3
        self.ClRad = 181E-12
        self.numParcels = 20000
        
        self.monomerRad= 0
        self.monomerDiameter= self.monomerRad * 2
        self.monomerBeadDiameter= 0
        self.latticeConstant = ((self.monomerRad * 2) - self.monomerBeadDiameter) / 4
        self.polymerLength = 0
        self.polymerStiffness = 0
        self.monomerMolecularWeight = 0
        self.initialSolidsLoading = .0232
        
        self.latticeConstant = 0
        
        self.fileNameBase = "default"
        self.addUniqueNumber = False
        self.simtTime = 36000
        
        
        # Misc. parameters
        self.simTime = 36000
        self.doMathIn = "Float64"
        self.minimumNumberOfCrossings = 3       # Number of crossing before two polymers are consider tangled.
        self.polymersPerBin = 500               # Number of polymers in a processing bin
        self.writeDumpPolymers = False
        self.writeDumpTangles = False
        self.generatePolymers = False
        self.drying = False
        self.dryingDelay = 0
        self.polymerPercent = 0
        self.packingFactor = .64                # for random spheres packing density
        self.fluidDensity = .997         #density of water         # diffusion constont.  .1 is totally blind guess of a good value
        self.availableCpus = 0
        self.gravitationalConstant = 9.8        # gravitational Constant
        self.planksConstant = 6.62607004e-34
        self.avogadroNumber = 6.0221409e+23
        self.boltz= 1.38e-23                     # boltzmann constant e
        self.Ef = 76
        self.Ep = 2.3 #polyethylene electrical permitivity
        self.Eo = 8.854e-12 # vaccuum permittivity
        self.particleRefractiveIndex = 1.525
        self.fluidRefractiveIndex = 1.33
        self.rotationalFrequency = 1
        self.downEscapedAggregates = 20
        self.writeParticlesEvery = 0
        self.stericLayerThickness = 0
        self.writeOnSL=[]
        self.maxPolymerBrownian = 0
        self.totalPolymerBrownian = 0
        self.numberOfPolymerBrownian = 0
        self.runningTotalMaxBrownian = 0
        self.variableLengthPolymers = False

        # initialize status flags
        self.valid = True
        self.CheckForTanglesCountDown = -1.0  # for the Tangle Check the first iteration
        self.clock = 0
        self.lastPolymerBrownianCheck = 0
        
        self.timeInRecalculateBoxSize = 0
        self.maxPolymerRadius = 0
        self.maxTangleRadius = 0
        self.smallestParticleRadius = self.particleRad
        self.largestParticleRadius = self.particleRad
        self.timeInCheckForPolymerTangles = 0
        self.timeInCalcParticleBrownianMotion = 0
        self.timeInCalcParcelBrownianMotion = 0
        self.timeInCalcPolymerBrownianMotion = 0
        self.timeInRunSimulation = 0
        self.timeInNudgeParticles = 0
        
        # program Tuning parameters
        self.polymersPerBin = 500               # Number of polymers in a processing bin
        self.generateBasePolymerCount = 0
        self.processOneBlockMaxSize =  10000
        self.polymerizationTime =  0 # 5 hours
 
        # Debug       
        self.graphForcesTest = False       
        self.saveInitialAtoms = False
        
        self.V_dCalc = False
        self.V_sCalc = False

#pH constants
        self.pH = 3.5                   
        # 3.5505945 creates almost 0 surface potential
        #use pH of 2.5, 3, 3.5, 4, and 4.5

#EDLConstants
        self.Eo = 8.854e-12 #vaccum permitivity
        self.Ef = 76 #dielectric constant of water
        self.boltz= 1.38e-23
        self.temp = 300 
        self.Fe_ions = 1
        self.Fe_charge = 3
        self.Cl_ions = 3
        self.Cl_charge = 1
        self.electronCharge = 1.6e-19
        #self.Fe_and_Cl_sum = 3.072e-37
        #((1*(3**2)*self.electronCharge**2)+(3*(1**2)*self.electronCharge**2))
        self.Concentration = .0005 #.5 originally #mmol/L
        #self.Concentration_Converted = 3.01e23
        #.5/1000*(6.02e23)/1000 #changing mmol to molecules and L to m^3
        #self.debyeLength = 5.635e-9
        #((self.Ef*self.Eo*self.boltz*self.temp)/(self.Fe_and_Cl_sum*self.Concentration_Converted))**(1/2)
        #self.kappa = 1.775e8
        #1/self.debyeLength
        
###############################################################################
    def ReadParametersFromFile(self, fileName=["runParameters.txt"]):          
        print("Reading file %s" % (fileName))
        
        with open(fileName) as f:
            content = f.readlines()
        f.close()
        
        content = [x.strip() for x in content] 
        
        for i in range(len(content)):
            line = content[i].rstrip()
            
            # Check to see if the entire line is a comment
            if ((len(line) > 0) and (line[0] != "#")):
                position = line.find("#")
                
                # if position == -1 then no comment in line
                if (position == -1):
                    withoutComment = line
                else:
                    withoutComment = line[0:position]
                 
                # get rid of troublesum leading and trailing white space
                withoutComment = withoutComment.strip()
                position = withoutComment.find("=")
                
                # if no = in line then something is just plain wrong.
                if (position == -1):
                    print("<<<< Unknown command line: %s >>>>>" % (line))
                else:
                    tag = withoutComment[0:position].lower()
                    value = withoutComment[position+1:]
                    
                    # strip off white space on both sides
                    value = value.strip()
                    tag = tag.strip()
                    #print("tag = %s, value = %s" %(tag, value))
            
                    if (tag == "particleCount".lower()):
                        self.particleCount = int(value)
                        #print("particleCount=", self.particleCount)
                    elif (tag == "temperature".lower()):
                        self.temperature = float(value)
                        #print("temperature=", self.temperature)
                        
                    elif (tag == "particleElictricalPermitivity".lower()):
                        self.particleElictricalPermitivity = float(value)
                        #print("particleElictricalPermitivity=", self.particleElictricalPermitivity)
                    elif (tag == "fluidElictricalPermitivity".lower()):
                        self.fluidElictricalPermitivity = float(value)
                        #print("fluidElictricalPermitivity=", self.fluidElictricalPermitivity)
                    elif (tag == "particleRefractiveIndex".lower()):
                        self.particleRefractiveIndex = float(value)
                        #print("particleRefractiveIndex=", self.particleRefractiveIndex)
                    elif (tag == "fluidRefractiveIndex".lower()):
                        self.fluidRefractiveIndex = float(value)
                        #print("fluidRefractiveIndex=", self.fluidRefractiveIndex)
                    elif (tag == "rotationalFrequency".lower()):
                        self.rotationalFrequency = float(value)
                        #print("rotationalFrequency=", self.rotationalFrequency)
                    elif (tag == "stericLayerThickness".lower()):
                        self.stericLayerThickness = float(value)
                        #print("stericLayerThickness=", self.stericLayerThickness)
                        
                    elif (tag == "particleRad".lower()):
                        self.particleRad = float(value)
                        #print("particleRad=", self.particleRad)
                    elif (tag == "monomerRad".lower()):
                        self.monomerRad = float(value)
                        self.monomerDiameter= self.monomerRad * 2
                        #print("monomerRad=", self.monomerRad)
                    elif (tag == "monomerBeadDiameter".lower()):
                        self.monomerBeadDiameter = float(value)
                        #print("monomerBeadDiameter=", self.monomerBeadDiameter)
                    elif (tag == "latticeConstant".lower()):
                        self.latticeConstant = float(value)
                        #print("latticeConstant=", self.latticeConstant)
                    elif (tag == "polymerLength".lower()):
                        self.polymerLength = int(value)
                        #print("polymerLength=", self.polymerLength)
                    elif (tag == "polymerStiffness".lower()):
                        self.polymerStiffness = float(value)
                        if self.polymerStiffness < 0: self.polymerStiffness = 0
                        if self.polymerStiffness > 1: self.polymerStiffness = 1
                        #print("polymerStiffness=", self.polymerStiffness)
                    elif (tag == "monomerMolecularWeight".lower()):
                        self.monomerMolecularWeight = float(value)
                        #print("monomerMolecularWeight=", self.monomerMolecularWeight)
                           
                    elif (tag == "writeDumpTangles".lower()):
                        if (value.lower() == "true"):
                            self.writeDumpTangles = True
                        else:
                            self.writeDumpTangles = False
                        #print("writeDumpTangles=", self.writeDumpTangles)
                    elif (tag == "writeDumpPolymers".lower()):
                        if (value.lower() == "true"):
                            self.writeDumpPolymers = True 
                        else:
                            self.writeDumpPolymers = False
                    elif (tag == "generatePolymers".lower()):
                        if (value.lower() == "true"):
                            self.generatePolymers = True
                        else:
                            self.generatePolymers = False
                    elif (tag == "variableLengthPolymers".lower()):
                        if (value.lower() == "true"):
                            self.variableLengthPolymers = True
                        else:
                            self.variableLengthPolymers = False
                    elif (tag == "drying".lower()):
                        if (value.lower() == "false"):
                            self.drying = False
                            self.dryingDelay = 0
                        else:
                            self.drying = True
                            self.dryingDelay = int(value)
                    elif (tag == "polymerPercent".lower()):
                        self.polymerPercent = float(value)
                        #print("polymerPercent=", self.polymerPercent)  
                    elif (tag == "initialSolidsLoading".lower()):
                        self.initialSolidsLoading = float(value)
                        #print("initialSolidsLoading=", self.initialSolidsLoading)  
                    elif (tag == "availableCpus".lower()):
                        self.availableCpus = int(value)
                        if (self.availableCpus >= multiprocessing.cpu_count()): self.availableCpus = multiprocessing.cpu_count()
                        if (self.availableCpus <= 0): self.availableCpus = multiprocessing.cpu_count()  + self.availableCpus # + a negitive number
                        if (self.availableCpus <= 0): self.availableCpus = 1
                        #print("availableCpus=", self.availableCpus)

                    elif (tag == "fluidDensity".lower()):
                        self.fluidDensity = float(value)
                        #print("fluidDensity=", self.fluidDensity)
                    elif (tag == "downEscapedAggregates".lower()):
                        self.downEscapedAggregates = int(value)
                        #print("downEscapedAggregates=", self.downEscapedAggregates)

                    elif (tag == "latticeConstant".lower()):
                        self.latticeConstant = float(value)
                        #print("latticeConstant=", self.latticeConstant)
                    elif (tag == "fileNameBase".lower()):
                        self.fileNameBase = str(value)
                        #print("fileNameBase=", self.fileNameBase)
                    elif (tag == "writeParticlesEvery".lower()):
                        self.writeParticlesEvery = float(value)
                        #print("writeParticlesEvery=", self.writeParticlesEvery)
                    elif (tag == "writeonsl".lower()):
                        if (len(value) > 0):
                            splitValue = value.split(",")
                            
                            for i in range(len(splitValue)):
                                splitValue[i] = splitValue[i].strip()
                                
                            if (len(splitValue) > 0):
                                splitValue.sort()
                                self.writeOnSL = splitValue                            
                        #print("writeOnSL=", self.writeOnSL)  

                    elif (tag == "minimumNumberOfCrossings".lower()):
                        self.minimumNumberOfCrossings = int(value)
                        #print("minimumNumberOfCrossings=", self.minimumNumberOfCrossings)
                    elif (tag == "doMathIn".lower()):
                        if (value.lower() == "float32"):
                            self.doMathIn = "Float32"
                        elif (value.lower() == "float64"):
                            self.doMathIn = "Float64"
                        else:
                            print("<<<< Unknown math type, command line: %s. doMathIn defaulted to %s >>>>>" % (line, self.doMathIn))
                        #print("doMathIn=", self.doMathIn)
                    elif (tag == "simTime".lower()):
                        self.simTime = float(value)
                        #print("simTime=", self.simTime)
                    elif (tag == "statInterval".lower()):
                        self.statInterval = float(value)
                        #print("statInterval=", self.statInterval)
                        
                    elif (tag == "saveInitialAtoms".lower()):
                        if (value.lower() == "true"):
                            self.saveInitialAtoms = True
                        elif (value.lower() == "false"):
                            self.saveInitialAtoms = False
                    elif (tag == "graphForcesTest".lower()): 
                        if (value.lower() == "true"):
                            self.graphForcesTest = True
                            self.particleCount = 1000
                            self.particleRad = 15e-9
                            self.spacing= .1e-9
                            self.stericLayerThickness = 1e-9
                            self.saveInitialAtoms = True
                        elif (value.lower() == "false"):
                            self.graphForcesTest = False
                    elif (tag == "addUniqueNumber".lower()):
                        if (value.lower() == "true"):
                            self.addUniqueNumber = True
                        elif (value.lower() == "false"):
                            self.addUniqueNumber = False
                    elif (tag == "V_dCalc".lower()):
                        if (value.lower() == "true"):
                            self.V_dCalc = True
                        elif (value.lower() == "false"):
                            self.V_dCalc = False
                    elif (tag == "V_sCalc".lower()):
                        if (value.lower() == "true"):
                            self.V_sCalc = True
                        elif (value.lower() == "false"):
                            self.V_sCalc = False
                            
                    # program tuning parameters 
                    elif (tag == "polymersPerBin".lower()): 
                        self.polymersPerBin = int(value)
                        #print("polymersPerBin=", self.polymersPerBin)
                    elif (tag == "generateBasePolymerCount".lower()): 
                        self.generateBasePolymerCount = int(value)
                        #print("generateBasePolymerCount=", self.generateBasePolymerCount) 
                    elif (tag == "processOneBlockMaxSize".lower()): 
                        self.processOneBlockMaxSize = int(value)
                        #print("processOneBlockMaxSize=", self.processOneBlockMaxSize)
                    elif (tag == "polymerizationTime".lower()): 
                        self.polymerizationTime = float(value)
                        #print("polymerizationTime=", self.polymerizationTime)
                    
                    #EDL Constants
                    elif (tag == "Eo".lower()):
                        self.Eo = float(value)
                    elif (tag == "Ef".lower()):
                        self.Ef = float(value)
                    elif (tag == "surfacePotential".lower()):
                        self.surfacePotential = float(value)
                    elif (tag == "boltz".lower()):
                        self.boltz = float(value)
                    elif (tag == "temp".lower()):
                        self.temp = int(value)
                    elif (tag == "Fe_ions".lower()):
                        self.Fe_ions = int(value)
                    elif (tag == "Fe_charge".lower()):
                        self.Fe_charge = int(value)
                    elif (tag == "Cl_ions".lower()):
                        self.Cl_ions = int(value)
                    elif (tag == "Cl_charge".lower()):
                        self.Cl_charge = int(value)
                    elif (tag == "electronCharge".lower()):
                        self.electronCharage = float(value)
                    elif (tag == "Fe_and_Cl_sum".lower()):
                        self.Fe_and_Cl_sum = float(value)
                    elif (tag == "Concentration".lower()):
                        self.Concentration= float(value)
                    elif (tag == "Concentration_Converted".lower()):
                        self.Concentration_Converted = float(value)
                    elif (tag == "debyeLength".lower()):
                        self.debyeLength = float(value)
                    elif (tag == "kappa".lower()):
                        self.kappa = float(value)
                    elif (tag == "EDLConstants".lower()):
                        self.EDLConstants = float(value)
                    elif (tag == "pH".lower()):
                        self.pH = float(value)
                        
                    
                    else:
                        print("<<<< Unknown command line: %s >>>>>" % (line))
                        self.valid = False #change later
                    
        self.oldPolymerBoxSize = self.boxSize
        self.latticeConstantDiv2 = self.latticeConstant / 2.0
        self.monomerBondLength = (self.monomerRad * 2.0) - self.monomerBeadDiameter
        self.polymerMolecularWeight = self.polymerLength * self.monomerMolecularWeight
        
        #self.massOf1_ZnO_particle = 7.06858352e-20
        #self.density_ZnO_particle = 5.61
        
        self.particleRadScale = 0.5
        
        self.chainsPerParticle = 0
                                   
        CalculateVolFracs(self)  # also Calculates solidsLoading
        
        # Calculate hamaker constant
        T = self.temperature
        k = self.boltz
        hBar = self.planksConstant
        epslon_1 = 2.25
        epslon_3 = self.Ef
        omega = self.rotationalFrequency
        n_1 = 1.51
        n_3 = 1.33
        self.A = ((3.0*k*T/4.0) * (((epslon_1 - epslon_3)/(epslon_1 + epslon_3))**2)) + \
                        (((3.0*hBar*omega)/(32.0*math.pi*math.sqrt(2.0))) * ((n_1**2 - n_3**2)**2) / ((n_1**2 + n_3**2)**(3.0/2.0)))        
          
        baseDir = "Results" 
             
        if not os.path.exists(baseDir):
            os.makedirs(baseDir)
            
        if (self.addUniqueNumber):
            uniqueNumber = 1
            foundNumberNotUsed = False
            
            while (foundNumberNotUsed == False):
                asciiUniqueNumber = ("_%04d" % (uniqueNumber))
                newBaseDir = baseDir + "/" + self.fileNameBase + asciiUniqueNumber + "/"
                
                if (os.path.exists(newBaseDir) == True): 
                    uniqueNumber += 1
                else:
                    foundNumberNotUsed = True
                    baseDir = newBaseDir
        else:
            baseDir += "/" + self.fileNameBase + "/"
                
            if (os.path.exists(baseDir) == True): 
                shutil.rmtree(baseDir) 
                         
        if not os.path.exists(baseDir):
            os.makedirs(baseDir)
            
        self.baseDir = baseDir
        copyfile(fileName, baseDir + fileName)
            
        self.fileNameBase = baseDir + self.fileNameBase
        self.tempDir = baseDir + "temp/"
        
        if not os.path.exists(self.tempDir):
            os.makedirs(self.tempDir)
        
        logInterval = 10
        self.particalSizeDistributionCsv = MyCsv(baseDir + "partSizeDist.csv", logInterval)
        self.radiusOverTimeCsv = MyCsv(baseDir + "radiusOverTime.csv", logInterval)
        self.fluidViscositySolidsLoadingOverTimeCsv = MyCsv(baseDir + "fluidSolidsTime.csv", logInterval)
        self.ellipsoidsOverTimeCsv = MyCsv(baseDir + "ellipsoidsTime.csv", logInterval)
        self.debugCsv = MyCsv(baseDir + "debug.csv", logInterval)
        self.executionTimesCsv = MyCsv(baseDir + "executionTime.csv", logInterval)
        self.settlingCsv = MyCsv(baseDir + "settling.csv", logInterval)
        self.particalSizeDistributionCsv = MyCsv(baseDir + "partSizeDist.csv", 3600)
                
        if (self.generatePolymers):
            self.tanglesOverTimeCsv = MyCsv(baseDir + "TanglesTime.csv", logInterval)
            
        #self.availableCpus = 1

        
###############################################################################
    def PrintParameters(self, data):
        log = data.log
        
        log.Log("\n%s file parameters:" % (data.arg))
        log.Log("  Constants:")
        log.Log("    A= %3.5e" % (self.A))
        log.Log("    boltz= %4.2e" % (self.boltz))
        log.Log("    fluidElictricalPermitivity= %3.5e" % (self.Ef))
        log.Log("    fluidRefractiveIndex= %3.5e" % (self.fluidRefractiveIndex))
        log.Log("    gravitationalConstant= %3.5e" % (self.gravitationalConstant))
        log.Log("    particleElictricalPermitivity= %3.5e" % (self.Ep))
        log.Log("    planksConstant= %3.5e" % (self.planksConstant))
        log.Log("    particleRefractiveIndex= %3.5e" % (self.particleRefractiveIndex))
        log.Log("    rotationalFrequency= %3.5e" % (self.rotationalFrequency))
        log.Log("")
        log.Log("  Simulation parameters:")
        log.Log("    addUniqueNumber= %r" % (self.addUniqueNumber))
        log.Log("    availableCpus= %d" % (self.availableCpus))
        log.Log("    downEscapedAggregates= %d" % (self.downEscapedAggregates))
        log.Log("    drying= %r" % (self.drying))
        log.Log("    dryingDelay= %d" % (self.dryingDelay))
        log.Log("    fluidDensity= %5.2f" % (self.fluidDensity))
        log.Log("    generatePolymers= %r" % (self.generatePolymers))
        log.Log("    initialSolidsLoading= %6.4f" % (self.initialSolidsLoading))
        log.Log("    latticeConstant= %3.2e" % (self.latticeConstant))
        log.Log("    minimumNumberOfCrossings= %s" % (self.minimumNumberOfCrossings)) 
        log.Log("    particleCount= %d" % (self.particleCount))
        log.Log("    polymerizationTime= %4.1f" % (self.polymerizationTime))
        log.Log("    polymerPercent= %4.2f" % (self.polymerPercent))
        log.Log("    simTime= %.1f" % (self.simTime))
        log.Log("    temperature= %3.0fC" % (self.temperature))
        log.Log("    variableLengthPolymers= %r" % (self.variableLengthPolymers))
        log.Log("    writeDumpPolymers= %r" % (self.writeDumpPolymers))
        log.Log("    writeDumpTangles= %r" % (self.writeDumpTangles))
        log.Log("")
        log.Log("  Particle parameters")
        log.Log("    particleRad= %3.2e" % (self.particleRad))
        log.Log("    particleVolFrac= %7.5f" % (self.particleVolFrac))
        log.Log("    stericLayerThickness= %3.3e" % (self.stericLayerThickness))
        log.Log("")
        log.Log("  Polymer parameters")
        log.Log("    monomerDiameter= %3.2e" % (self.monomerDiameter))
        log.Log("    monomerMolecularWeight= %5f" % (self.monomerMolecularWeight))
        log.Log("    monomerRad= %3.2e" % (self.monomerRad))
        log.Log("    particleVolFrac= %7.5f" % (self.polymerVolFrac))
        log.Log("    polymerLength= %5d" % (self.polymerLength))
        log.Log("    polymerStiffness= %3.2f" % (self.polymerStiffness))
        log.Log("")
        log.Log("  Program tuning Parameters")
        log.Log("    doMathIn= %s" % (self.doMathIn))
        log.Log("    fileNameBase= %s" % (self.fileNameBase))
        log.Log("    generateBasePolymerCount= %s" % (self.generateBasePolymerCount)) 
        log.Log("    polymersPerBin= %s" % (self.polymersPerBin))
        log.Log("    processOneBlockMaxSize= %s" % (self.processOneBlockMaxSize))
        log.Log("    statInterval= %4d" % (self.statInterval))
        log.Log("")
        log.Log("  Debug Parameters")
        log.Log("    graphForcesTest= %r" % (self.graphForcesTest)) 
        log.Log("    saveInitialAtoms= %r" % (self.saveInitialAtoms)) 
        log.Log("    writeOnSL= %s" % (self.writeOnSL))
        log.Log("    writeParticlesEvery= %s" % (self.writeParticlesEvery))
        log.Log("")  
        log.Log(("Using %d of %d CPUs." % (self.availableCpus, multiprocessing.cpu_count())))
