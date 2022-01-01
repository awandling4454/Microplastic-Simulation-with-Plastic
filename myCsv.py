# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""
#from datetime import datetime
     
class MyCsv():       
    # set defaults
    # location 1 = debug file only, 2 = console only,  3 = both, otherwise nowhere
    def __init__(self, fileName=[], simTimeStep = 60):
        self.outputFile = fileName
        self.simTimeStep = .25  # Gets doubled two times in initializion to 1.
        self.maxSimTimeStep = simTimeStep
        self.lastSimTimeStep = 0
        
        if (len(self.outputFile) != 0):
            self.f = open(self.outputFile, 'w')
        
###############################################################################
    def __del__(self):
        if (len(self.outputFile) != 0):
            self.f.close()
        
###############################################################################
    def isNextSimTimeStep(self, currentClock):
        return ((self.lastSimTimeStep + self.simTimeStep) < currentClock)
    
###############################################################################
    def Log(self, currentClock, string):
        self.f.write(string + "\n") 
        self.f.flush()
        self.lastSimTimeStep = currentClock
        self.simTimeStep = min(self.simTimeStep * 2, self.maxSimTimeStep)
                        
###############################################################################
    def Flush(self):
        if (len(self.outputFile) != 0):
            self.f.flush()
            
###############################################################################
    def Close(self):
        if (len(self.outputFile) != 0):
            self.f.close()
 ###############################################################################
           
