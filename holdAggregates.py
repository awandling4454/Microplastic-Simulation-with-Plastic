# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:22:24 2017

@author: Michelle
"""
import random
import copy


class HoldAggregates():
          
    # set defaults
    def __init__(self, queueLength):
        self.aggregates = []
        self.maxAggregates = queueLength
        
##############################################################################
    def     PushAgglomerate(self, agglomerate):
        self.aggregates.append(copy.deepcopy(agglomerate))

##############################################################################
    def     CanPop(self):
        return(self.maxAggregates < len(self.aggregates))
        
##############################################################################
    def     PopAgglomerate(self):
        index = int(random.random() * len(self.aggregates))
        
        rc = copy.deepcopy(self.aggregates[index])
        del self.aggregates[index]
        return(rc)
        
##############################################################################
    def     PeekAgglomerate(self, index):
        rc = copy.deepcopy(self.aggregates[index])
        return(rc)

##############################################################################
    def     GetQueueLen(self):
        return(len(self.aggregates))

##############################################################################
    def     CalcTotalParticles(self):        
        return(sum(particle.c for particle in self.aggregates))

##############################################################################
