# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""

import copy
    
###############################################################################
class PolymerBins:
###############################################################################
    
    def __init__(self, parameters):
        self.binPerDimension = 2
        self.binSize = parameters.boxSize / self.binPerDimension
        self.bins = self._CreateBins()
        self.ResetBinCounters()
        self.lastBoxSize = parameters.boxSize
        return
        
###############################################################################
    def _CalculateBinPerDimensionAndBinSize(self, parameters, allPolymers):
        cellsPerSide = int((len(allPolymers) / parameters.polymersPerBin)**(1.0/3.0))
        
        if cellsPerSide < 2:
            cellsPerSide = 2
            
        newBinSize = parameters.boxSize / cellsPerSide
        
        while ((newBinSize < (parameters.maxPolymerRadius * 4.0)) & (cellsPerSide > 2.0)):
            cellsPerSide -= 1
            newBinSize = parameters.boxSize / cellsPerSide
            parameters.polymersPerBin = len(allPolymers) / (cellsPerSide**3)
            print("****\n Adjusting bin size due to minimum requirement. New polymers per bin is %5.1f\n****" % (parameters.polymersPerBin))
            
        return(cellsPerSide, newBinSize)
        
###############################################################################
    def _CreateBins(self):   
        a = [copy.deepcopy([]) for i in range(self.binPerDimension)]
        b = [copy.deepcopy(a) for i in range(self.binPerDimension)]
        bins = [copy.deepcopy(b) for i in range(self.binPerDimension)]
        return(bins)
            
###############################################################################
    def AddToBin(self, parameters, onePolymer, polymerIndex):        
        [xLoc, yLoc, zLoc] = onePolymer.GetStartingLocation()
            
        xBin = int(xLoc / self.binSize)
        yBin = int(yLoc / self.binSize)
        zBin = int(zLoc / self.binSize)
        
        #print("xBin= %d, yBin= %d, zBin= %d" % (xBin, yBin, zBin))
        #print("xLoc= %e, yLoc= %e, zLoc= %e" % (xLoc, yLoc, zLoc))
        if xBin >= self.binPerDimension: xBin = (self.binPerDimension - 1)
        if yBin >= self.binPerDimension: yBin = (self.binPerDimension - 1)
        if zBin >= self.binPerDimension: zBin = (self.binPerDimension - 1)
        if xBin <  0: xBin = 0
        if yBin <  0: yBin = 0
        if zBin <  0: zBin = 0
        
        try:
            self.bins[xBin][yBin][zBin].append(int(polymerIndex))
        except:
            print(parameters.boxSize)
            print(xLoc, yLoc, zLoc)
            print(xBin, yBin, zBin)
            raise
            
        return
            
###############################################################################
    def GetBinsPerDimension(self):
        return(self.binPerDimension)
        
###############################################################################
    def ResetBinCounters(self):
        self.xIndex = 0;
        self.yIndex = 0
        self.zIndex = 0
        
###############################################################################
    def CalcBinLen(self):
        dimension = len(self.bins[0][0])
        total = 0

        binLens = self._CreateBins()
        
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    total += len(self.bins[i][j][k])
                    binLens[i][j][k] = len(self.bins[i][j][k])
                    
        average = int(total / dimension**3)
                    
        #if (total >= 1000000):
        #    s = ("%4.1fM" % (total / 1000000))
        #elif (total >= 1000):
        #    s = ("%4.1fK" % (total / 1000))
        #else:
        #    s = ("%4d" % (total))
        #print("PrintBinLen: dimension= %d, self.binPerDimension= %d, Bin size %8.2e, total polymers= %s, average= %d" % \
        #          (dimension, self.binPerDimension, self.binSize, s, average))
        
        if average > 1000: raise ValueError("******\Error:  \nAverage way to big (%d).  Program aborted.\n*****" % (average))
            
        return(binLens)
    
###############################################################################
    def PrintBinLen(self, printBins = False):        
        if printBins:
            print(self.CalcBinLen())
            
        return
    
###############################################################################
    def SaveBinLen(self, fileName = "debug_binLens.txt"):        
        a = str(self.CalcBinLen())
            
        #print(a)
        f = open(fileName, "w")
        f.write(a)
        f.close() 
        return
    
###############################################################################
    def GetNextBins(self):
        return(self.xIndex, self.yIndex, self.zIndex)
        
###############################################################################
    def GetBinProperties(self):
        return([self.binPerDimension, self.binSize, self.lastBoxSize])

###############################################################################
    def SetBinProperties(self, properties):
        
        self.binPerDimension = properties[0]
        self.binSize = properties[1]
        self.lastBoxSize = properties[2]
        
        self.bins = self._CreateBins()
        self.ResetBinCounters()
        return

###############################################################################
    def GetCurrentBinCount(self):
        return(len(self.bins[self.xIndex][self.yIndex][self.zIndex]))
        
###############################################################################
    def IsMoreWork(self):
        return(self.zIndex < self.binPerDimension)
    
###############################################################################
    def GetMoreWork(self):      
        if (self.IsMoreWork() == False):
            return([])
            
        results      = copy.deepcopy(self.bins[self.xIndex  ][self.yIndex  ][self.zIndex  ])
        
        if (self.xIndex < (self.binPerDimension - 1)):
            results += self.bins[self.xIndex+1][self.yIndex  ][self.zIndex  ]
            
        if (self.yIndex < (self.binPerDimension - 1)):
            results += self.bins[self.xIndex  ][self.yIndex+1][self.zIndex  ]
            
            if (self.xIndex < (self.binPerDimension - 1)):
                results += self.bins[self.xIndex+1][self.yIndex+1][self.zIndex  ]

        if (self.zIndex < (self.binPerDimension - 1)):
            results +=     self.bins[self.xIndex  ][self.yIndex  ][self.zIndex+1]
            
            if (self.xIndex < (self.binPerDimension - 1)):
                results += self.bins[self.xIndex+1][self.yIndex  ][self.zIndex+1]
            
            if (self.yIndex < (self.binPerDimension - 1)):
                results += self.bins[self.xIndex  ][self.yIndex+1][self.zIndex+1]
            
                if (self.xIndex < (self.binPerDimension - 1)):
                    results += self.bins[self.xIndex+1][self.yIndex+1][self.zIndex+1]
            
        self.IncrementIndexes()
           
        return(results)
        
###############################################################################
    def IncrementIndexes(self):
        if (self.IsMoreWork() == False):
            return
            
        self.xIndex += 1;
        
        if self.xIndex == self.binPerDimension:
            self.xIndex = 0
            self.yIndex += 1
        
        if self.yIndex == self.binPerDimension:
            self.yIndex = 0
            self.zIndex += 1
            
        return
###############################################################################
        
    def _ReBin(self, parameters, allPolymers, firstPolymer, lastPolymer):        
        [self.binPerDimension, self.binSize] = self._CalculateBinPerDimensionAndBinSize(parameters, allPolymers)
        self.bins = self._CreateBins()
        self.ResetBinCounters()
        self.lastBoxSize = parameters.boxSize

        #print("***** ReBining %d->%d of %d polymers *****" % (firstPolymer, lastPolymer, len(allPolymers)))
        
        for i in range(firstPolymer, lastPolymer):
            self.AddToBin(parameters, allPolymers[i], i)
            
        average = (lastPolymer - firstPolymer) / self.binPerDimension**3 
        #print("_ReBin: Done, polymers= %d->%d, maxPolymerRad= %8.2e, binSize= %8.2e, binsPerSide= %d, polymers/bin= %3d" % \
        #      (firstPolymer, lastPolymer, parameters.maxPolymerRadius, self.binSize, self.binPerDimension, average))
        if average > 1000: raise ValueError("******\Error:  \nAverage way to big.  Program aborted.\n*****")

        return
    
###############################################################################
    def BoxResized(self, parameters, allPolymers, firstPolymer = -1, lastPolymer = -1):
        if (parameters.maxPolymerRadius == 0): return
        
        if (firstPolymer == -1):
            firstPolymer = 0
        
        if (lastPolymer == -1):
            lastPolymer = len(allPolymers)
        
        oldBinPerDimension = self.binPerDimension
        [self.binPerDimension, self.binSize] = self._CalculateBinPerDimensionAndBinSize(parameters, allPolymers)
        
        if (oldBinPerDimension != self.binPerDimension):
            self._ReBin(parameters, allPolymers, firstPolymer, lastPolymer)
        return
 ###############################################################################
