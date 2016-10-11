# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:53:36 2016

@author: mmorel, haiming, Colin Eason
"""
import cv2
import sys
import os
import numpy as np
from svmClass import SvmClass   

class ObjectClassifier:
    
    def __init__(self):
        self.classes = []
        for pathname in next(os.walk('classes'))[1] :            
            self.classes.append(SvmClass(pathname))
        
        #for className in classes:
        #    self.classes.append(SvmClass(className))
            
        for classInst in self.classes:
            #classInst.load()      
            print('Training SVM: ' + classInst.getName())
            classInst.train(False)
            
        print('Done.')
        
    def classify(self,img,rect):
        
        for classInst in self.classes:
            matches,kp = classInst.matches(img,rect)
            if matches:
                return classInst.getName(),kp               
            
        return 'unknown',None