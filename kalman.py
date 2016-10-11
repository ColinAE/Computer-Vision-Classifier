# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:02:51 2016

@author: mmorel, haiming, Colin Eason
"""

import numpy as np
import cv2
import copy
import math

# Wrapper class for an OpenCV 2 measurement Kalman filter that provides some
# history about predictions and measurements
class Kalman:
    meas=[] # save measured values
    pred=[] # save predicted values
    kalman = None # actual OpenCV Kalman filter
    lastPred = None
    lastCorr = None
    stuck = False # TODO: might not be needed
    averageValue = (0,0)
    averageVelocity = (0,0)
    def __init__(self,point,q=0.03,r=0.00003):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.statePre[0,0]  = point[0]
        self.kalman.statePre[1,0]  = point[1]
        self.kalman.statePre[2,0]  = 0
        self.kalman.statePre[3,0]  = 0
        # TODO: determine if this is necessary
        #self.kalman.statePost[0,0]  = point[0]
        
        #self.kalman.statePost[1,0]  = point[1]
        
        #self.kalman.statePost[2,0]  = 0
        
        #self.kalman.statePost[3,0]  = 0
        
        # Set Kalman parameter matricies
        # H
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        
        # A
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        
        # Q
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * q
        
        # R
        self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * r 
        
        self.predict()
        self.correct(point)           
        
    # provide an actual measurement and update model
    def correct(self, point):
        mp = np.array([[np.float32(point[0])],[np.float32(point[1])]])
        cp = self.kalman.correct(mp)
        point = (int(cp[0]),int(cp[1]))
        self.meas.append(point)
        self.lastCorr = (int(point[0]),int(point[1]))
        return point
        
    # call when sample could not be found
    def predict(self):            
        tp = self.kalman.predict()
        point = (int(tp[0]),int(tp[1]))
        self.lastPred = (point)
        self.pred.append(point)
        N = len(self.pred)        
        
        #update running averages
        self.averageValue = ((self.averageValue[0] * ((N - 1)/N)) + point[0]/N, (self.averageValue[1] * ((N - 1)/N)) + point[1]/N)
        self.averageVelocity = ((self.averageVelocity[0] * ((N - 1)/N)) + self.kalman.statePost[2,0]/N,(self.averageVelocity[1] * ((N - 1)/N)) + self.kalman.statePost[3,0]/N)
        
        
        #if self.lastPred == point :
        #    self.stuck = True
        #else:
        #    self.stuck = False
        
        return point
        
    def getPath(self):
        return self.meas;
        
    def getPredicted(self):
        return self.pred
        
    def getLastPrediction(self):
        return self.lastPred
        
    def getLastCorrected(self):
        return self.lastCorr
        
    def isStuck(self):
        return self.stuck
        
    def getVelocity(self):
        return (self.kalman.statePost[2,0], self.kalman.statePost[3,0])
        
    def getAverageValue(self):
        return self.averageValue
        
    def getAverageVelocity(self):
        return self.averageVelocity