# -*- coding: utf-8 -*-
import math
from readFile import FileParameters




def EDLConstants(parameters):
    surfacePotential = -41.63*math.log(parameters.pH)+52.75
    print(surfacePotential)
    debyeLength = ((parameters.Ef*parameters.Eo*parameters.boltz*parameters.temp)/(parameters.Fe_and_Cl_sum*parameters.Concentration_Converted))**(1/2)
    kappa = 1/debyeLength
    EDLConstants = ((4*math.pi*parameters.Ef*parameters.Eo*(surfacePotential)**2)*1e-6)
    return(EDLConstants)

parameters = FileParameters()
EDLConstants = EDLConstants(parameters)
print(EDLConstants)