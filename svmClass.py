# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:35:22 2016

@author: mmorel, haiming, Colin Eason
"""
import cv2
import numpy as np
import sys

class SvmClass:
    root = 'classes/'
    
    def __init__(self, name,algorithm='sift'):
        self.name = name
    
        if algorithm.lower() == 'surf':
            self.detector= cv2.xfeatures2d.SURF_create(20)
        elif algorithm.lower() == 'orb':
            self.detector= cv2.ORB_create(200,1.2,8,14,0,2,0,14)
        elif algorithm.lower() == 'fast':
            self.detector = cv2.FastFeatureDetector_create()
        else: # default to sift
            self.detector = cv2.xfeatures2d.SIFT_create(edgeThreshold=10,sigma=1.6)
    
        self.numClusters = 200
        self.bow = cv2.BOWKMeansTrainer(self.numClusters)
        self.matcher = cv2.BFMatcher()
        self.descriptor = cv2.BOWImgDescriptorExtractor(self.detector, self.matcher)
        self.model = cv2.ml.SVM_create()
        self.samples = []
        self.labels = []
        self.tempWidth = 64
        self.tempHeight = 32
        
    def matches(self, img, rect):
        imgry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x,y,w,h = rect
        #center = (x+w/2, y+h/2)
        #center = (center[0]-self.tempWidth/2,center[1]-self.tempHeight/2)
        #x,y = center
        #w,h = (self,tempWidth, self.tempHeight)
        sample = imgry[y:y+h, x:x+w]
        #cv2.imshow('sample',sample)
        # extract features with desired algorithm
        kp = self.detector.detect(sample)
        # compute descriptors with desired algorithm
        des = self.descriptor.compute(sample, kp)
        if not des is None:
            pred = self.model.predict(des)
            #print('Class: ' + self.name + ' predict: ' + repr(pred))
            return pred[1] > 0.0, kp
        else:
            return False, None
            
    def getName(self):
        return self.name
        
    def trainVocab(self):
        # first, generate vocabulary
        self.addWords(self.root + self.name + '/pos')
        self.addWords(self.root + self.name + '/neg')
        self.vocab = self.bow.cluster()                                        
        
    def train(self, retrainVocab=False):
        if retrainVocab:
            self.trainVocab()
        else:
            self.vocab = np.load(self.root + self.name + 'Vocab.npy')
        self.descriptor.setVocabulary(self.vocab)
    
        # next, train SVM
        self.addSamples(self.root + self.name + '/pos')
        self.addSamples(self.root + self.name + '/neg', positive=False)
        samplesnp = np.empty(shape=[0, self.numClusters], dtype=np.float32)
        for sample in self.samples:
            row = np.array(sample, dtype=np.float32)
            samplesnp = np.vstack((samplesnp,row))
        labelsnp = np.array(self.labels, dtype=np.int32)
        self.model.train(samplesnp, cv2.ml.ROW_SAMPLE, labelsnp)    
        
    def save(self):
        np.save(self.root + self.name + 'Vocab', self.vocab)
        self.model.save(self.root + self.name + 'Svm.yml')

    def load(self):
        self.vocab = np.load(self.root + self.name + 'Vocab.npy')
        self.descriptor.setVocabulary(self.vocab)
        self.model.load(self.root + self.name + 'Svm.yml')
        
    def addWords(self,folder):    
        done = False
        count = 0
        while not done:
            count = count + 1
            sample = cv2.imread(folder + '/sample (' + repr(count) + ').png',cv2.IMREAD_GRAYSCALE)
            if sample is None:
                done = True
            else:       
                # compute the descriptors with desired algorithm
                kp, des = self.detector.detectAndCompute(sample,mask=None)
                #des = np.array(des,np.float32)
                #print (repr(des.size))
                # add to bag of words
                if not des is None : #and des.size > 1:       
                    self.bow.add(des)
                if False:
                    sample = cv2.drawKeypoints(sample,kp,sample,color=(0,255,0), flags=0)
                    cv2.imshow('keypoints',sample)
                    k = cv2.waitKey(1000) & 0xFF
                    if k == ord('q'):
                        sys.exit(0)
                
    def addSamples(self,folder, positive=True):
        done = False
        count = 0
        while not done:
            count = count + 1
            sample = cv2.imread(folder + '/sample (' + repr(count) + ').png',cv2.IMREAD_GRAYSCALE)
            if sample is None:
                done = True
            else:
                # extract features with desired algorithm
                kp = self.detector.detect(sample)
                #for k in kp:
                #    print (repr(k.pt))
                #kp, des = self.detector.detectAndCompute(sample,mask=None)
                
                #kp = np.array(kp,np.float32)
                
                # compute descriptors with desired algorithm
                des = self.descriptor.compute(sample, kp)
                if not des is None:
                    # store freshly computed descriptors
                    self.samples.append(des)
            
                    # was this a positive or negative example?
                    if positive:
                        self.labels.append(1.0)
                    else:
                        self.labels.append(0.0)
        