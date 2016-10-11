# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:22:53 2016

@author: mmorel, haiming, Colin Eason
"""
import numpy as np
import cv2
import copy
import math
from statistics import mode
from statistics import StatisticsError
from kalman import Kalman

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

class TrackBase:           
             
    def __init__(self, trackId, rect, img, mask):
        self.trackId = trackId
        self.lastClass = 'unknown'
        self.lastKeypoints = None
        self.rect = []
        self.searchWindow = rect
        self.template = []
        self.tracklet = []
        self.trackletRects = []
        self.direction = Enum(['LEFT','RIGHT','UNKNOWN'])
        self.actions = []
        self.actions.append('unknown')
        self.classHistory = []
        self.active = True
        self.occluded = False
        self.foundCount = 0
        self.numTemplates = 6
        self.outDimens = None
        #self.trackOutputStream = None
        self.doTrackWrite = False
        self.savedFrames = []
        self.outFps = 30
        height, width = img.shape[:2]
        self.imgBounds = (width,height)
        x,y,w,h = rect
        x,y,w,h = rect = self.resizeRect(rect,(1.0,1.0),(0.0,0.0))

        #x,y,w,h = rect = self.resizeRect(rect,(1.0,0.75),(-0.15,-0.2))
              
        #crop syntax is [starty:endy, startx:endx]
        template = img[y:y+h, x:x+w]
        
        #maskFiltered = mask[y:y+h, x:x+w]
                    
        # 3 Kalman filters per research document            
        self.kalmanFast = Kalman((x+(w/2),y+(h/2)))
        self.kalmanArea = Kalman((w,h) ) #  ,0.003,0.3)
        self.kalmanSlow = Kalman((x+(w/2),y+(h/2)), 0.003, 0.03)
        self.updateTemplate(rect,template)  
        
        #x,y,w,h = rect = self.resizeRect(rect,(1.5,1.3),(0,0.5))
        
        self.trackletSize = (0,0)
        
        self.updateTracklet(rect,img)
        
        # set up the ROI for tracking
        #roi = template
        
        #hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        #maskFiltered = cv2.inRange(hsv_roi, np.array((0., 150.,60.)), np.array((180.,255.,255.)))        
        #cv2.imshow('maskFiltered',maskFiltered)
        
        #self.roi_hist = cv2.calcHist([hsv_roi],[0,1],maskFiltered,[36,50],[0,180,0,255])    
        
        #cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        
    # returns true if the provided point plus current track bounds go over the edge of the frame, false otherwise
    def isInBounds(self, point, bounds):
        bh, bw = bounds
        px,py = point
        x,y,w,h = self.getNewestRect()
        if px + w/2 > bw or px - w/2 < 0:
            return False
        if py + h/2 > bh or py - h/2 < 0:
            return False
        return True            
        
    # search for this track in the provided frame by whatever concrete algorithm the track implements   
    def update(self, img):
        self.savedFrames.append(img)
        # always do predictions before measurements
        pLoc = self.kalmanFast.predict()
        self.kalmanSlow.predict()
        self.kalmanArea.predict()
        # make sure we haven't already gone off the frame
        if not self.isInBounds(pLoc,img.shape[:2]):
            self.active = False           
            print('Track #' + repr(self.trackId) + ' ' +  repr(self.occluded) + ' ENDING TRACK: gone off frame ' + repr(pLoc))
            return 
        newroi = (0,0,0,0)
        # search phase
        if not self.occluded : 
            self.occluded, newroi = self.doNonOccludedSearch(img)
        else :
            self.occluded, newroi = self.doOccludedSearch(img)
            
        # update phase
        if not self.occluded :
            self.doNonOccludedUpdate(newroi,img)
        else :
            self.doOccludedUpdate(newroi,img)
            
        #if not self.trackOutputStream is None and self.doTrackWrite:
            #w,h = self.outDimens
                    
            #resized = cv2.resize(self.getNewestTemplate(), (w,h))
                    
        #    self.trackOutputStream.write(self.getNewestTracklet())
        
        #print('Track #' + repr(self.trackId) + ' ' +  repr(self.occluded) + ' ROI: ' + repr(self.rect[0]))
        
    # perform search for tracked object with the assumption it was occluded in the previous frame
    # default implementation provides a global template match for previously known good template
    def doOccludedSearch(self, img):    
        
        # set search window so we're not performing a global search
        roi = self.resizeRect(self.getNewestRect(),(4,4))
        x,y,w,h = roi
        newroi = (0,0,0,0)
        self.searchWindow = roi
        window = img[y:y+h, x:x+w]
        temp = self.getOldestTemplate()
        
        # make sure we haven't already gone off the frame
        if window.shape[:2][0] <= 0 or window.shape[:2][0] < temp.shape[:2][0] or window.shape[:2][1] <= 0 or window.shape[:2][1] < temp.shape[:2][1]:
            self.active = False           
            print('Track #' + repr(self.trackId) + ' ' +  repr(self.occluded) + ' ENDING TRACK: match window off frame ' + repr(window.shape[:2]))
            return True, newroi           
        
        #print('Track #' + repr(self.trackId) + ' window:' + repr(window.shape[:2]) + ' temp:' + repr(temp.shape[:2]))
        
        #cv2.imshow('template',temp) 
        
        # perform the template matching search
        res = cv2.matchTemplate(window,temp,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.7 : #found match
            px, py = max_loc
            x,y,w,h = roi
            x2,y2,w2,h2 = self.getOldestRect()
            newroi = (px+x,py+y,w2,h2)
            return False, newroi
        else :
            return True, newroi
            
        #print('Track #' + repr(self.trackId) + ' ' +  repr(self.occluded) + ' CORR: ' + repr(max_val))
        
    # perform search for tracked object with the assumption it was not occluded in the previous frame     
    # all subclasses should implement this method with their specific tracking algorithm
    def doNonOccludedSearch(self,img):
        return
        
    # updates the track when it was not found in the search frame using Kalman filter predictions
    def doOccludedUpdate(self, roi, img):
        if self.foundCount < 10:
            pLocS = self.kalmanFast.getLastPrediction()
        else:
            pLocS = self.kalmanSlow.getLastPrediction()
        x,y,w,h = self.getOldestRect()
        newRoi = self.resizeRect((int(pLocS[0]-(w/2)),int(pLocS[1]-(h/2)), w,h), (1.0,1.0))
        self.updateTemplate( newRoi, None)
        self.updateTracklet( newRoi, img)  # don't want missing frames in tracklet
        
    # updates the track with a real measurement when it was found in the search frame
    def doNonOccludedUpdate(self, roi, img):
        self.foundCount = self.foundCount + 1
        x,y,w,h = roi
        
        # correct Kalman filters
        self.kalmanFast.correct((x+(w/2),y+(h/2)))
        self.kalmanSlow.correct((x+(w/2),y+(h/2)))
        self.kalmanArea.correct((w,h))
        self.updateTemplate( roi, img[y:y+h, x:x+w])
        self.updateTracklet( roi, img)
    # TODO: remove when deemed no longer necessary
    # callback when a potential seed roi is detected near this
    def onFindIntersect(self,roi,img):
        return               
        
    # returns true if provided roi is near this track position
    def matches(self, roi, img):
        
        if self.intersects(roi, 100) :           
            self.onFindIntersect(roi,img)    
            return True
        else :
            return False
        
    def updateTemplate(self, rect, template):            
        x,y,w,h = rect
        #self.rect.insert( 0,rect )
        
        if not rect is None:
            self.rect.append(rect)
            if len(self.rect) > self.numTemplates:
                del self.rect[0]
            #del self.rect[self.numTemplates-1]
        
        #self.template.insert( 0,template ) 
        
        if not template is None:
            self.template.append( template ) 
            if len(self.template) > self.numTemplates:
                del self.template[0]
            #del self.template[self.numTemplates-1]
                
    def updateTracklet(self, rect, img):
        #x,y,w,h = rect = self.resizeRect(rect,(1.3,1.5),(-0.15,0.25)) #wave
        x,y,w,h = rect = self.resizeRect(rect,(1.2,1.5),(0.0,0.25))
        if self.trackletSize[0] == 0:
            self.trackletSize = (w,h)
        else:
            w,h = self.trackletSize
        #x = rect[0]
        #y = rect[1]
        roiGray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.tracklet.append( roiGray)
        self.trackletRects.append((x,y,w,h))
        #if not self.trackOutputStream is None and self.doTrackWrite:
        #    self.trackOutputStream.write(img[y:y+h, x:x+w])
    # returns true is the provided rectangle touches this track's known rectangle
    # based on a circle drawn from the center
    def intersects(self, rect, dmin=100):            
        x1,y1,w1,h1 = rect 
        x2,y2,w2,h2 = self.getNewestRect()
        center1 = (int(x1 + w1/2), int(y1+h1/2))
        center2 = (int(x2 + w2/2), int(y2+h2/2))
        # radius of search area
        radius1 = int(math.hypot(w1, h1)  / 2)
        radius2 = int(math.hypot(w2, h2)  / 2)
        # distance between centers
        dist = int(math.hypot(center1[0] - center2[0], center1[1] - center2[1]))
        #cv2.line(img, center1, center2, (0,255,255),1)
        #cv2.circle(img, center1, radius1, (0,0,255), 1)
        #cv2.circle(img, center2, radius2, (0,255,255), 1)
        return dist < (radius1+radius2) or dist < dmin  
    
    # return the input rectangle resized by given area coefficient and bounds checked against provided bounds    
    def resizeRect(self, rect, scale=(1.0,1.0), offsets=(0,0)):
        x,y,w,h = rect 
        width, height = self.imgBounds
        if x < 0 :
            x = 0        
        sx = x - (w*(scale[0]-1.0))/2
        sx = sx + offsets[0]*w
        if sx < 0 :
            sx = 0     
        if sx >= width:
            sx = width - 1       
        ex = sx + (w*scale[0])             
        if ex >= width:
            ex = width - 1
        if y < 0 :
            y = 0
        sy = y - (h*(scale[1]-1.0))/2
        sy = sy + offsets[1]*h
        if sy < 0 :
            sy = 0
        if sy >= height:
            sy = height - 1
        ey = sy + (h*scale[1])
        if ey >= height:
            ey = height - 1
        return (int(sx),int(sy),int(ex-sx),int(ey-sy))
        
    # Returns a measure of how close the provided image's histogram matches this one.  Lower value = more similar     
    def histogramsMatch(self, img1, img2, thresh=0.6):
        hsv_roi1 =  cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv_roi1], [0, 1], None, [36,50],[0,180,0,255])
        cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)
        hsv_roi2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        hist2 = cv2.calcHist([hsv_roi2], [0, 1], None, [36,50],[0,180,0,255])
        cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return score <= thresh, score
        
    # TODO: resize templates more intelligently    
    # returns true if the provided images match, false otherwise
    def templatesMatch(self,img1,img2,thresh=0.3):
        #cv2.imshow('template',img) 
        h1,w1 = img1.shape[:2]       
        h2,w2 = img2.shape[:2]    
        w = min((w1,w2))
        h = min((h1,h2))
        if w <=0 or h <=0:
            return False, 0.0
        x1 = (w1-w)/2
        y1 = (h1-h)/2
        x2 = (w2-w)/2
        y2 = (h2-h)/2
        crop1 = img1[y1:y1+h, x1:x1+w]
        crop2 = img2[y2:y2+h, x2:x2+w]
        #resized = cv2.resize(img2, (w,h))
        #print('Track #' + repr(self.trackId) + ' img1:' + repr(img1.shape[:2]) + ' img2:' + repr(img2.shape[:2]) + ' resize:' + repr(resized.shape[:2]))
        #cv2.imshow('resized',resized)             
        res = cv2.matchTemplate(crop1,crop2,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val >= thresh, max_val
        
    def getNewestRect(self):
        #return self.rect[0]
        return self.rect[-1]   
        
    def getTrackletRect(self):
        return self.trackletRects[-1]
        
    def getTrackletRects(self):
        return self.trackletRects
        
    def saveLastTracklet(self):
        lasttracklet = self.getLastTracklet(45)
        if not lasttracklet is None:
            h,w = lasttracklet[0].shape[:2]
            outputStream = cv2.VideoWriter('track' + repr(self.trackId) +'.avi',cv2.VideoWriter_fourcc(*'DIVX'), self.outFps, (w,h))
            for frame in lasttracklet:               
                outputStream.write(frame)
            outputStream.release()
        
    def getLastTracklet(self, numFrames):
        if len(self.trackletRects) > numFrames:
            h,w = self.savedFrames[0].shape[:2]
            temptracklet = []
            #temprects = []
            xmax,xmin,ymax,ymin = (0,w,0,h)
            for rect in self.trackletRects[-numFrames:]:
                x,y,w,h = rect
                xmin = x if x < xmin else xmin
                xmax = (x + w) if (x + w) > xmax else xmax
                ymin = y if y < ymin else ymin
                ymax = (y + h) if (y + h) > ymax else ymax
            for frame in self.savedFrames[-numFrames:]:
                temptracklet.append(frame[ymin:ymin+(ymax-ymin), xmin:xmin+(xmax-xmin)])
            return temptracklet
        else:
            return None
        
    def getNewestTracklet(self):
        return self.tracklet[-1]
        
    def getOldestRect(self):
        return self.rect[0]
        #if len(self.rect) < self.numTemplates :
        #    return self.rect[len(self.rect)-1]   
        #else:
        #    return self.rect[self.numTemplates-1]
        
    def getNewestTemplate(self):
        return self.template[-1]
        
    def getOldestTemplate(self):
        return self.template[0]
        #if len(self.template) < self.numTemplates :
        #    return self.template[len(self.template)-1]   
        #else:
        #    return self.template[self.numTemplates-1]
        
    def getPath(self):
        return self.kalmanFast.getPath()
        
    def getPredicted(self):
        return self.kalmanFast.getPredicted()
        
    def getTrackId(self):
        return self.trackId
        
    def isOccluded(self):
        return self.occluded
        
    def getSearchWindow(self):
        return self.searchWindow
        
    def isActive(self):
        return self.active        
        
    def updateClass(self,classType,kp):
        self.lastKeypoints = kp
        self.lastClass = classType
        self.classHistory.append(classType)
        try:
            lc = mode(self.classHistory)
            self.lastClass = lc
            return self.lastClass
        except StatisticsError:
            return self.lastClass
            
    def getLastClass(self):
        return self.lastClass
        
    def getLastKeypoints(self):
        return self.lastKeypoints
        
    def getTracklet(self):
        return self.tracklet
        
    def updateAction(self,action):
        self.actions.append(action)
        
        return self.actions[-1]
        
    def getLastAction(self):
        return self.actions[-1]
        
    def setDoTrackWrite(self,doWrite):
        self.doTrackWrite = doWrite
        
    def setOutFps(self,fps):
        self.outFps = fps
        
    #def cleanup(self):
        #if not self.trackOutputStream is None:
        #    self.trackOutputStream.release()
        
    #def setTrackOutputStream(self, fps, dimens):
        
        #w,h,t = dimens
        
        #self.outDimens = (w,h)
        
        #h,w = self.getNewestTracklet().shape[:2]
        
        #self.trackOutputStream = cv2.VideoWriter('track' + repr(self.trackId) +'.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (w,h))
    
        #if not self.trackOutputStream.isOpened():   
            
        #    self.trackOutputStream = None
            
        #    print('Error opening output video stream')        