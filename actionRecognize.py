# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 08:22:23 2016

@author: mmorel, haiming, Colin Eason
"""

import numpy as np
import cv2
import copy
import math
import os

class ActionRecognize:
       
    def __init__(self, path, vCubeDimens=(50,100,45),matchEdges=False,retrain=False,filterVelocity=True):
        self.path = path
        self.vCubeDimens = vCubeDimens # w,h,t
        self.actions = []
        self.edgeActions = []
        self.matchEdges = matchEdges
        self.outCount = 0
        self.filterVelocity = filterVelocity
        #self.trainActions() if retrain:
            self.trainActions() else:
            self.loadActions();
            rawr = self.actions
        ar = 42
        
    def getAction(self, tracklet, thresh=0.0001):
        x,y,t = self.vCubeDimens
        if not tracklet is None :
            vCube = []   
            h,w = tracklet[-t:][0].shape[:2]                  
            for frame in tracklet[-t:]:                                
                resized = cv2.resize(frame, (x,y))
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                if self.matchEdges:
                    resized = self.getSobel(resized)
                #cv2.imshow('tracklet',resized)
                vCube.append(resized)
            #cv2.imwrite('frames/' + repr(self.outCount) + 'frame.jpg', vCube[0]);  
            ev3d = self.get3dPCA(vCube);
            bestPA = 0
            label = 'unknown'
            actionset = self.edgeActions if self.matchEdges else self.actions
            for action in actionset:
                wmin, wmax = action[2]
                if (w < wmin or w > wmax) and self.filterVelocity:
                    continue
                pa = self.getAveragePrincipleAngle(ev3d, action[1])
                #print('pa for ' + action[0] + ': ' + repr(pa))
                if pa > bestPA and pa > thresh:
                    label = action[0]
                    bestPA = pa                  
            #print('best pa: ' + label + ': ' + repr(bestPA))
            return label
        else :
            return 'unknown'
            
    def get3dPCA(self, frames):
        #vcube = np.empty(shape=self.vCubeDimens,dtype=np.float32) x,y,t = self.vCubeDimens
        vcube = np.array(frames,dtype=np.float32)
        self.outCount = self.outCount + 1      
        sliceXY = np.swapaxes(vcube,1,2).reshape(t,y*x,order='F')#.swapaxes(0,1)    
        sliceXT = np.swapaxes(vcube,0,2).reshape(x,y*t)#.swapaxes(0,1)   
        sliceYT = np.swapaxes(vcube,0,1).reshape(y,x*t,order='F')#.swapaxes(0,1)   
        mean1, eigenXY = cv2.PCACompute(sliceXY, None)#,cv2.PCA_DATA_AS_COL) 
        # optional : write out eigenvectors#eign = copy.deepcopy(eigenXY)
        #cv2.normalize(eign, eign, 0, 255, cv2.NORM_MINMAX)
        #single = eign[0]
        #rolled = np.reshape(single,(-1,x))
        #frame = cv2.cvtColor(rolled, cv2.COLOR_GRAY2BGR)
        #cv2.imwrite('frames/' + repr(self.outCount) + 'frame.jpg', frames[0]);  
        #cv2.imwrite('frames/' + repr(self.outCount) + 'pca.jpg', frame);  
        mean2, eigenXT = cv2.PCACompute(sliceXT, None,cv2.PCA_DATA_AS_COL)   
        mean3, eigenYT = cv2.PCACompute(sliceYT, None,cv2.PCA_DATA_AS_COL)   
        return (eigenXY,eigenXT,eigenYT)    
        
    def getAveragePrincipleAngle(self, eigVec3d1, eigVec3d2):
        paXY = self.getPrincipleAngle(eigVec3d1[0],eigVec3d2[0])
        paXT = self.getPrincipleAngle(eigVec3d1[1],eigVec3d2[1])
        paYT = self.getPrincipleAngle(eigVec3d1[2],eigVec3d2[2])
        return abs(paXY) + abs(paXT) + abs(paYT) # / 3.0;
        
    def getPrincipleAngle(self, eigVecs1, eigVecs2):
        #e2t = cv2.transpose(eigVecs2)
        evm = cv2.gemm(eigVecs1,eigVecs2,1,None,1,flags=cv2.GEMM_2_T)
        #evm = np.outer(eigVecs1,e2t)
        w,u,vt = cv2.SVDecomp(evm)
        return float(w[0]); 
        
    def getSobel(self,img):
        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        sobel_64fdx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobel_64fdy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        abs_sobel64fdx =  cv2.convertScaleAbs(sobel_64fdx)
        abs_sobel64fdy =  cv2.convertScaleAbs(sobel_64fdy)
        sobel_8u = cv2.addWeighted(abs_sobel64fdx, 0.5, abs_sobel64fdy, 0.5, 0)
        #abs_sobel64f = np.absolute(sobel_64f)
        #sobel_8u = np.uint8(abs_sobel64f)
        #cv2.imshow('sobel',sobel_8u)
        return sobel_8u
        
    def trainActions(self):
        x,y,t = self.vCubeDimens
        for pathname in next(os.walk(self.path))[1] :      
            sampleCount = 1
            moreSamples = True
            wmax = 0
            wmin = 10000
            while moreSamples:
                if os.path.isfile(self.path + pathname + '/' + pathname + repr(sampleCount) + '.avi'):                
                    videoStream = cv2.VideoCapture(self.path + pathname + '/' + pathname + repr(sampleCount) + '.avi')          
                else:
                    moreSamples = False
                    continue                    
                    
                if videoStream.isOpened():
                    sampleCount = sampleCount + 1
                else:
                    moreSamples = False
                    continue
                frames = []
                edgeFrames = []
                frameCount = 0                     
                        
                while(videoStream.isOpened() and frameCount < t and moreSamples):
                    frameCount = frameCount + 1
                    ret, frame = videoStream.read()
                    if ret==True:  
                        h,w = frame.shape[:2]
                        wmax = w if w > wmax else wmax
                        wmin = w if w < wmin else wmin
                        #cv2.imshow('tracklet',frame)
                        #cv2.imwrite('frames/' + pathname + repr(frameCount) + 'original.jpg', frame);
                        resized = cv2.resize(frame, (x,y))
                        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        #if self.matchEdges:
                        edges = self.getSobel(copy.deepcopy(resized))
                        #cv2.imshow('tracklet',resized)
                        #cv2.imwrite('frames/' + pathname + repr(frameCount) + 'resized.jpg', resized);
                        frames.append(resized)
                        edgeFrames.append(edges)
                    #cv2.imwrite('frames/' + repr(self.outCount) + 'frame.jpg', frames[0]);
                videoStream.release()
                self.actions.append( (pathname, self.get3dPCA(frames), np.array([wmin*0.9,wmax*1.1]) ) )
                self.edgeActions.append( (pathname, self.get3dPCA(edgeFrames), np.array([wmin*0.9,wmax*1.1]) ) )
        #for action in self.actions:
        #    print('found action: ' + action[0])
            
        self.saveActions()
            
    def loadActions(self):
        for pathname in next(os.walk(self.path))[1]:
            count = 1         
            while os.path.isfile(self.path + pathname + '/' + pathname + repr(count) + 'exy.npy'):               
                eigenXY = np.load(self.path + pathname + '/' + pathname + repr(count) + 'exy.npy')
                eigenXT = np.load(self.path + pathname + '/' +pathname + repr(count) + 'ext.npy')
                eigenYT = np.load(self.path + pathname + '/' +pathname + repr(count) + 'eyt.npy')
                minmax = np.load(self.path + pathname + '/' +pathname + repr(count) + 'len.npy')           
                self.actions.append( (pathname, (eigenXY,eigenXT,eigenYT),minmax ) )
                count = count + 1
                
            count = 1            
            while os.path.isfile(self.path + pathname + '/' + pathname + repr(count) + 'exyEdges.npy'):           
                eigenXY = np.load(self.path + pathname + '/' + pathname + repr(count) + 'exyEdges.npy')
                eigenXT = np.load(self.path + pathname + '/' +pathname + repr(count) + 'extEdges.npy')
                eigenYT = np.load(self.path + pathname + '/' +pathname + repr(count) + 'eytEdges.npy')
                minmax = np.load(self.path + pathname + '/' +pathname + repr(count) + 'lenEdges.npy')            
                self.edgeActions.append( (pathname, (eigenXY,eigenXT,eigenYT),minmax ) )
                count = count + 1
            rawr = 42
            
    def saveActions(self):
        count = 0
        lastAction = 'rawr'
        for action in self.actions:
            if not action[0] == lastAction:
                count = 1
                lastAction = action[0]
            else:
                count = count + 1
            np.save(self.path + action[0] + '/' + action[0] + repr(count) + 'exy', action[1][0])  
            np.save(self.path + action[0] + '/' +action[0] + repr(count) + 'ext', action[1][1]) 
            np.save(self.path + action[0] + '/' +action[0] + repr(count) + 'eyt', action[1][2]) 
            np.save(self.path + action[0] + '/' +action[0] + repr(count) + 'len', action[2])      
            
        count = 0
        lastAction = 'rawr'            
        for action in self.edgeActions:
            if not action[0] == lastAction:
                count = 1
                lastAction = action[0]
            else:
                count = count + 1
            np.save(self.path + action[0] + '/' + action[0] + repr(count) + 'exyEdges', action[1][0])  
            np.save(self.path + action[0] + '/' +action[0] + repr(count) + 'extEdges', action[1][1]) 
            np.save(self.path + action[0] + '/' +action[0] + repr(count) + 'eytEdges', action[1][2]) 
            np.save(self.path + action[0] + '/' +action[0] + repr(count) + 'lenEdges', action[2])
        

            
