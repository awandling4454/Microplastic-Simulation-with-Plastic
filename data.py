# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle

Editted 6/21/2021
by Allison Wandling
"""



###############################################################################    
class Data:    
    
###############################################################################    
    def __init__(self):
        self.holdAreClose = []
        self.lastPolymerBrownianMotionTime = 0.0
        self.largestParticleToAggregate = 1.0   # = 1 meter i.e. aggregate all particles
        self.lengthOfSubparticleList = 0
        self.ellipsoidMerges = 0
        self.maxWillFit = 0
        self.brownianCount = 0
        self.brownianTotal = 0
        self.brownianMax = 0
        self.stokesCount = 0
        self.stokesTotal = 0
        self.stokesMax = 0
        self.tangleCandidates = 0
        self.willFitDumpIndex = 1
        self.shifter = int(2.0 * 10**9)
        self.maxLenNpHoldAreClose = 0
        self.agglomerationsAccepted = 0
        self.agglomerationsRejected = 0
        self.particlesDidNotApproach = 0
        self.particlesEscaped = 0
        self.lastDim = 0
        self.lastAAF = 0
        self.sumApproachProb = 0.0
        self.sumEscapeProb = 0.0
        self.rejectProbEvents = 0
        self.firstTime = True

        return        
###############################################################################    
