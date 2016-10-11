# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:54:19 2016

@author: mmorel, haiming, Colin Eason
"""

import numpy as np
import cv2
import copy
import math

class ObjectDetector:
    
    bgSub = None
    lastImgBgSubtract = None
    lastImgFiltered = None
    lastImgContours = None
    
    def __init__(self):
        #self.bgSub = cv2.bgsegm.createBackgroundSubtractorGMG() 
        #self.bgSub = cv2.bgsegm.createBackgroundSubtractorMOG() 
        self.bgSub = cv2.createBackgroundSubtractorMOG2() 
        self.bgSub.setDetectShadows(False)    
        
    def getLastFrameBgSubtract(self):
        return self.lastImgBgSubtract
        
    def getLastFrameFiltered(self):
        return self.lastImgFiltered
        
    def getLastFrameContours(self):
        return self.lastImgContours
    
    def detectObjects(self,frame): 
        
        img = copy.deepcopy(frame)
        
        # Separate foreground from background
        self.lastImgBgSubtract = self.bgSub.apply(img)
        
        # Clean off noise and close large blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                      
        result = cv2.morphologyEx(self.lastImgBgSubtract, cv2.MORPH_OPEN, kernel,iterations = 2)              
        
        result = cv2.blur(result, (15,15))  
        
        thresh1, self.lastImgFiltered = cv2.threshold(result,0,255,cv2.THRESH_OTSU)

        # TODO: Matt or Haiming judged these lines to be default
        #result = cv2.morphologyEx(self.lastImgBgSubtract, cv2.MORPH_DILATE, kernel,iterations = 2)
        #result = cv2.morphologyEx(self.lastImgBgSubtract, cv2.MORPH_CLOSE, kernel,iterations = 2)  
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        #self.lastImgFiltered = cv2.morphologyEx(self.lastImgFiltered, cv2.MORPH_ERODE, kernel,iterations = 1) 
       
        
        
        if False:              
            #next, find the rough centers of blobs        
            dist = cv2.distanceTransform(self.lastImgFiltered,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
            dist = np.uint8(dist)
            #lastly, cut away just the centers so not so many blobs merge    
            thresh1, self.lastImgFiltered = cv2.threshold(dist,0.7*dist.max(),255,cv2.THRESH_BINARY)   
        
        # Find blob locations by their outlines
        self.lastImgContours = copy.deepcopy(self.lastImgFiltered)
        contours = cv2.findContours(self.lastImgContours,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    
        return contours
        
    def drawBounds(self,title,points,img,color=(0,255,0),enfilter=True,thresh=50):
        count = 0
        cntlist = points[1]
        height, width = img.shape[:2]
        for cnt in cntlist:
            count = count + 1
            rect = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < thresh and enfilter == True:
                continue
            x,y,w,h = rect
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            lbl = '#' + repr(count) + ' ('+ repr(x) + ',' + repr(y) + ') ' + repr(int(cv2.contourArea(cnt))) + 'px'
            cv2.putText(img,lbl,(x,y+h+10),0,0.3,color) 
        frmLbl = 'Total objects: ' + repr(count)
        cv2.putText(img,title,(15,15),0,0.5,color) 
        cv2.putText(img,frmLbl,(15,30),0,0.5,color) 
        return img
        