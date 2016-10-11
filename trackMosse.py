# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:38:31 2016

Based on OpenCV 3.1 sample code mosse.py

MOSSE Algorithm originally proposed by:
David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"

@author: mmorel, haiming, Colin Eason
"""
import numpy as np
import cv2
import copy
import math
from matplotlib import pyplot as plt
from trackBase import TrackBase

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in range(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 3.0
        if not self.good:
            return False, (0,0,0,0)

        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()
        
        return True, (int(self.pos[0]-self.size[0]/2),int(self.pos[1]-self.size[1]/2),int(self.size[0]),int(self.size[1]))

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1
        
    def getRect(self):      
        return (int(self.pos[0]-self.size[0]/2),int(self.pos[1]-self.size[1]/2),int(self.size[0]),int(self.size[1]))
        
    def isGood(self):
        return self.isGood
        
class TrackMosse (TrackBase):
    
    def __init__(self, trackId, roi, img, mask):
        super(TrackMosse,self).__init__(trackId,roi,img,mask)
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.Canny(frame_gray,125,175)
        frame_gray = self.getSobel(frame_gray)
        #x,y,w,h = self.resizeRect(roi,1.3,img.shape[:2])
        x,y,w,h = self.rect[0]
        mosse_roi = (x,y,x+w,y+h) # MOSSE expects top left/bottom right coordinates not x,y,w,h
        self.tracker = MOSSE(frame_gray, mosse_roi)
        self.stuckCount = 0
        
    def getSobel(self,img):
        return img
        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        sobel_64f = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
        abs_sobel64f = np.absolute(sobel_64f)
        sobel_8u = np.uint8(abs_sobel64f)
        #cv2.imshow('sobel',sobel_8u)
        return sobel_8u
            
    def doNonOccludedSearch(self, img):
        #if self.stuckCount > 5:
        #    self.active = False
        #    return True, (0,0,0,0)

        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.Canny(frame_gray,125,175)
        frame_gray = self.getSobel(frame_gray)
        good, newroi = self.tracker.update(frame_gray)
        
        if good :
            x,y,w,h = newroi
            #histMatch, score = self.histogramsMatch(img[y:y+h, x:x+w], self.getOldestTemplate(), 0.4)
            #tempMatch, tscore = self.templatesMatch(img[y:y+h, x:x+w], self.getOldestTemplate(), 0.7)
            #return not tempMatch, newroi
            #return False, newroi
                
            if newroi == self.getNewestRect():
                self.stuckCount = self.stuckCount + 1
            else:
                self.stuckCount = 0
                
                #if self.stuckCount > 15:
                #    self.stuckCount = 0
                #    return True, newroi
            #print('Track #' + repr(self.trackId) + ' found')     
            return False, newroi
                
        else : # track failed    
            #print('Track #' + repr(self.trackId) + ' becoming occluded')          
            return True, (0,0,0,0)
            #return False, newroi
            
    def doOccludedSearch(self, img): 
        occluded, newroi = super(TrackMosse,self).doOccludedSearch(img)
        
        # re-initialize tracker if we're coming out of occlusion
        if not occluded:
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x,y,w,h = newroi
            mosse_roi = (x,y,x+w,y+h) # MOSSE expects top left/bottom right coordinates not x,y,w,h
            self.tracker = MOSSE(frame_gray, mosse_roi)
        return occluded, newroi
                           