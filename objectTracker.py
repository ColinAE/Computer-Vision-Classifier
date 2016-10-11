# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:48:34 2016

@author: mmorel, haiming, Colin Eason
"""

import numpy as np
import cv2
import copy
import math
from trackCV import TrackCV
from trackMosse import TrackMosse
from objectClassifier import ObjectClassifier
from actionRecognize import ActionRecognize
from common import RectSelector

def getTrackInst(trackId, algorithm, roi, img, mask):
    
    if algorithm.lower() == 'mflow' or algorithm.lower() == 'boost':
        return TrackCV(trackId, algorithm, roi, img, mask )
    else: #default to mosse
        return TrackMosse(trackId, roi, img, mask )

class ObjectTracker:
    
    def __init__(self, outputDest, trackAlgorithm='mosse', drawKp=False,doClassify=False,doActRecog=True,outFps=30,retrain=False,matchEdges=False,filterVelocity=True):
        self.tracks = []    
        self.trackCount = 0        
        self.doClassify = doClassify
        self.doActRecog = doActRecog        
        self.drawKeypoints = drawKp        
        self.trackAlgorithm = trackAlgorithm
        self.vCubeDimens = (75,100,45)        
        self.outFps = outFps
        self.doTrackWrite = False
        self.rect_sel = RectSelector('frame', self.onrect)
        self.lastFrame = None
        if self.doClassify:
            self.classifier = ObjectClassifier()    
        if self.doActRecog:
            self.actionRecognizer = ActionRecognize('actions/',self.vCubeDimens,matchEdges=matchEdges,retrain=retrain,filterVelocity=filterVelocity)
        self.outputFile = open(outputDest, 'w')
        self.frameCount = 0
        
    def seedTracks(self,img,mask,points):
        self.lastFrame = img
        contourlist = points[1]
        for contour in contourlist:

            rect = cv2.boundingRect(contour)
            x,y,w,h = rect 
            
            # TBD: remove this once we can fix whole first frame being detected as track
            if w > 10000 or h < 100 :#cv2.contourArea(contour) <= 5000:
                continue
                                                                        
            # is this blob close to a known object?
            foundTrack = False
                               
            for track in self.tracks :    
                if track.isActive() and track.matches(rect,img) :
                    foundTrack = True                   
                    continue
           
            # start new track                     
            if foundTrack == False and not self.isNearEdge(rect,img):
                self.startTrack(img,rect,mask)
                
    def onrect(self, rect): 
        # API delivers top left/bottom right corners so we need to convert
        x1,y1,x2,y2 = rect
        trackRect = (x1,y1,x2-x1,y2-y1)
        self.startTrack(self.lastFrame,trackRect)
        
    def startTrack(self,img,rect,mask=None):
           self.trackCount = self.trackCount + 1 
           ntrck = getTrackInst(self.trackCount, self.trackAlgorithm, rect, img, mask )
           ntrck.setOutFps(self.outFps)        
           self.tracks.append(ntrck)
           print('New track #' + repr(self.trackCount) + ' ' + repr(rect)) 
                
    def writeTracks(self,frame):
        for track in self.tracks:
            x,y,w,h = track.getNewestRect()
            lost = '1' if not track.isActive() else '0'
            occluded = '1' if track.isOccluded() else '0'
            generated = '1' if track.isOccluded() else '0'
            self.outputFile.write( '' + repr(track.getTrackId()-1) + ' ' + repr(x)  + ' ' + repr(y) + ' ' + repr(x+w) + ' ' + repr(y+w) + ' ' + repr(frame) + ' ' + lost + ' ' + occluded + ' ' + generated + ' "'+ track.getLastAction() + '"' )
            self.outputFile.write("\n")
            
    def saveTracks(self):
        for track in self.tracks:
            track.saveLastTracklet()
            
    def clearTracks(self):
        self.tracks = []
        #self.trackCount = 0
        
    def update(self,img):
        self.frameCount = self.frameCount + 1
        for track in self.tracks :
            if track.isActive():    
                track.update(img)   
                if self.doActRecog and self.frameCount % 5 == 0:
                    action = self.actionRecognizer.getAction(track.getLastTracklet(45)) 
                    track.updateAction(action)
                
                if self.doClassify:
                    ttype,kp = self.classifier.classify(img,track.getNewestRect())             
                    track.updateClass(ttype,kp)  
                    
    def toggleDoTrackWrite(self):
        self.doTrackWrite = not self.doTrackWrite
        #for track in self.tracks:
        #    track.setDoTrackWrite(self.doTrackWrite)
                
    def isNearEdge(self,rect,img,thresh=50):
        x,y,w,h = rect 
        ih,iw = img.shape[:2]
        return x < thresh or y < thresh or x+w > iw - thresh or y+h > ih - thresh                             
        
    def drawBounds(self,title,img,color=(0,255,0),occludeColor=(0,0,255)):
        self.rect_sel.draw(img)
        useColor = color
        height, width = img.shape[:2]

        for track in self.tracks : 
            if not track.isActive():
                continue
            if track.isOccluded():
                useColor = occludeColor
            else:
                useColor = color
                
            x,y,w,h = track.getNewestRect()
            xt,yt,wt,ht = track.getTrackletRect()
            
            if self.doActRecog:                
                action = track.getLastAction()               
            else:
                action = 'noActRec'
                
            lbl = '#' + repr(track.getTrackId()) + ' ' + action + '(' + repr(w) + ',' + repr(h) + ')'
            
            #if self.doClassify:                        
            #    ttype = track.getLastClass()
            #else:
            #    ttype = 'noclassify'
                
            #keypoints = track.getLastKeypoints()
                
            #if self.drawKeypoints and not keypoints is None:
                
            #    for kp in keypoints:  
            #        kp.pt = (kp.pt[0]+x, kp.pt[1]+y)     

            #    cv2.drawKeypoints(img,keypoints,img)
                
            #    lbl = '#' + repr(track.getTrackId()) + ' ' + ttype + ' (' + repr(len(keypoints)) + ')'        
            #else:                
            #    lbl = '#' + repr(track.getTrackId()) + ' ' + ttype
            
            cv2.rectangle(img,(x,y),(x+w,y+h),useColor,1)
            
            if not self.doActRecog:
                cv2.rectangle(img,(xt,yt),(xt+wt,yt+ht),(0,255,255),1)
            
            if track.isOccluded() and False:
                sx,sy,sw,sh = track.getSearchWindow()
                cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,255,255),1)
            
            cv2.putText(img,lbl,(x,y+h+15),0,0.5,useColor) 
            
            if False:
                path = track.getPath() 
                predicted = track.getPredicted()
                currPoint = 0
                drawPoint = 1 #limit number of points drawn
            
                for point in path:
                    currPoint = currPoint + 1
                    if currPoint % drawPoint == 0 :                       
                         cv2.putText(img,'+',(int(point[0]), int(point[1])),0,0.3,color)  
                     
                for point in predicted:
                    cv2.putText(img,'+',(int(point[0]), int(point[1])),0,0.3,occludeColor) 
            
        if self.doTrackWrite:
            frmLbl = 'Total objects: ' + repr(len(self.tracks)) + ' WRITE'
        else:
            frmLbl = 'Total objects: ' + repr(len(self.tracks) )
        
        cv2.putText(img,title,(15,15),0,0.5,color)
        cv2.putText(img,frmLbl,(15,30),0,0.5,color)
        return img
        
    def closeOut(self):
        self.outputFile.close()
        
        #for track in self.tracks:
        #    track.cleanup()
        

            