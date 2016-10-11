# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:38:31 2016

@author: mmorel, haiming, Colin Eason
"""
import numpy as np
import cv2
import copy
import math
from matplotlib import pyplot as plt
from trackBase import TrackBase

class TrackCV (TrackBase):
    
    def __init__(self, trackId, algorithm, roi, img, mask):
        super(TrackCV,self).__init__(trackId,roi,img,mask)
        self.seedCount = 0
        self.requiredSeeds = 0
        self.algorithm = algorithm
        self.initTracker(roi,img)
        self.stuckCount = 0
              
    def initTracker(self,roi,img):
        # known working algorithms:    MIL, TLD, BOOSTING, MEDIANFLOW
        # only works in MultiTracker:  KCF
        if self.algorithm.lower() == 'mflow':
            self.tracker = cv2.Tracker_create("MEDIANFLOW")
        else: #default to boost
            self.tracker = cv2.Tracker_create("BOOSTING")
        self.tracker.init(img, roi)          
            
    def doNonOccludedSearch(self, img):
        #if self.stuckCount > 5:
        #    self.active = False
        #    return True, (0,0,0,0)
            
        # set search window so we're not performing a global search
        roi = self.resizeRect(self.getNewestRect(),(4,4))
        x,y,w,h = roi
        self.searchWindow = roi
        window = img[y:y+h, x:x+w]        
        ok, newroi = self.tracker.update(img)
        x2,y2,w2,h2 = newroi
        newroi = (int(x2),int(y2),int(w2),int(h2))
        #newroi = (x+x2,y+y2,w2,h2)
        if ok and not w2 == 0 and not h2 == 0 :
            x,y,w,h = newroi
            #return False, newroi
            #histMatch, score = self.histogramsMatch(img[y:y+h, x:x+w], self.getOldestTemplate(), 0.4)
            tempMatch, tscore = self.templatesMatch(img[y:y+h, x:x+w], self.getNewestTemplate(), 0.8)
            if newroi == self.getNewestRect():
                self.stuckCount = self.stuckCount + 1
            else:
                self.stuckCount = 0
                
            return not tempMatch, newroi
            
            if tempMatch:           
                if newroi == self.getNewestRect():
                    #self.active = False
                    return True, newroi     
                else:                          
                    return False, newroi
            else:
                return True, newroi
        else : # track failed
            return True, newroi
            
    def doOccludedSearch(self, img): 
        occluded, newroi = super(TrackCV,self).doOccludedSearch(img)
        # re-initialize tracker if we're coming out of occlusion
        if not occluded:
            self.initTracker(newroi,img)
        return occluded, newroi
                           