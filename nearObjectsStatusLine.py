# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""

import timeit

from myLogger import MyLogger
from utilities import Sec2HMS
from myLogger import LogDestination
from utilities import FormatNumber
     
###############################################################################
class NearObjectsStatusLine():       
    # set defaults
    # location 1 = debug file only, 2 = console only,  3 = both, otherwise nowhere
    def __init__(self, interval = 60):
        self.SetInterval(interval)
        self.Reset()
        
###############################################################################
    def Reset(self):
        self.lastClock = timeit.default_timer()
        self.nextPrint = self.lastClock + self.timeInterval
        self.start = self.lastClock
        
        return
    
###############################################################################
    def SetInterval(self, interval = 60):
        self.timeInterval = interval
        return
    
###############################################################################
    def Print(self, parameters, data, iteration, force = False, destination = LogDestination.CONSOLE):
        if (force | (self.nextPrint < timeit.default_timer())):
            deltaTime = timeit.default_timer() - self.lastClock
            self.lastClock = timeit.default_timer()
            self.nextPrint = self.lastClock + self.timeInterval
            log = data.log
            lenCombinedList = len(data.finalHoldAreClose)
            sinceStart = self.lastClock - self.start
                
            if iteration < 0:
                iterationString = ""
            else:
                iterationString = (", it= %d" % (iteration))
                
            if deltaTime < 1000:
                deltaTimeStr = str(int(deltaTime))
            else:
                deltaTimeStr = Sec2HMS(deltaTime)
                
            statusLine = ("CalcNearObjects: %s(+%3s), len(areClose)= %6s, len(CombinedList)= %6s%s" % 
                          (Sec2HMS(sinceStart), deltaTimeStr, FormatNumber(data.lenNpHoldAreClose), FormatNumber(lenCombinedList), iterationString))
            log.Log(statusLine, destination)
            
        return
                        
 ###############################################################################
           
