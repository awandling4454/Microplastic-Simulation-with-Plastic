# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:52:27 2021

@author: allis
"""
import math

Ef = 76 #dielectric constant of water
Eo = 8.854e-12 #vaccum permitivity
boltz = 1.38E-23
temp= 300 #in Kelvin
Fe_and_Cl_Sum = 3.027E-37
Concentration = 3 #mmol/L
Concentration_Converted = Concentration*1000*6.02E23/1000 #mmol/L to mol/m^3
debyeLength = ((Ef*Eo*boltz*temp)/(Fe_and_Cl_Sum*Concentration_Converted))**(1/2)
print (debyeLength)

#how to calculate number of particles
avogadroNumber = 6.0221409e+23
boxSize = 7.12E-7 #m^3
totalParticles = Concentration_Converted*boxSize
#assume Fe and Cl ions do not form any new compounds
Fe_Num = math.trunc(totalParticles/4) #math.trunc removes the decimal place
#totalParticles = Fe + Cl
#Cl = 3Fe becuase FeCl3
#totalParticles = 4Fe
Cl_Num = math.trunc(totalParticles - Fe_Num) #math.trucn removes the decimal place
print(Fe_Num)
print(Cl_Num)
numParcels = 10000
particlesPerParcel = totalParticles/numParcels
print(particlesPerParcel)