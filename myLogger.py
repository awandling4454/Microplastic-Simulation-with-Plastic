# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""
#from datetime import datetime
from time import gmtime, strftime
from enum import Enum

###############################################################################
class LogDestination(Enum):
     FILE = 1
     CONSOLE = 2
     BOTH = 3
     
###############################################################################
class MyLogger():       
    # set defaults
    # location 1 = debug file only, 2 = console only,  3 = both, otherwise nowhere
    def __init__(self, fileName=[], location=3, withTimeStamp=False):
        self.whereTo = 3
        if   (location == LogDestination.FILE):    self.whereTo = 1
        elif (location == LogDestination.CONSOLE): self.whereTo = 2
        elif (location == LogDestination.BOTH):    self.whereTo = 3
        
        self.outputFile = fileName
        self.timeStamp=withTimeStamp
        
        if (len(self.outputFile) != 0):
            self.f = open(self.outputFile, 'w')
        
###############################################################################
    def __del__(self):
        if (len(self.outputFile) != 0):
            self.f.close()
        
###############################################################################
    def Log(self, string, location=-1, withTimeStamp=-1):
        if (location == LogDestination.FILE): location = 1
        elif (location == LogDestination.CONSOLE): location = 2
        elif (location == LogDestination.BOTH): location = 3
        elif (location == -1): location = self.whereTo
        else:
            print("MyLogger: invalid location %s" % location)
            
        if (withTimeStamp == -1):
            withTimeStamp = self.timeStamp
                        
        if withTimeStamp:
            #formattedString = datetime.now().isoformat(timespec='seconds') + ": " + string
            formattedString = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ": " + string
        else:
            formattedString = string

        if ((location == 1) or (location == 3)):
            if (len(self.outputFile) != 0):
                self.f.write(formattedString + "\n") 
                self.f.flush()
            
        if ((location == 2) or (location == 3)):
            print(formattedString)
            
###############################################################################
    def Flush(self):
        if (len(self.outputFile) != 0):
            self.f.flush()
            
###############################################################################
    def Close(self):
        if (len(self.outputFile) != 0):
            self.f.close()
            
