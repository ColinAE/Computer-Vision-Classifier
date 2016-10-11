# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:43:02 2016

@author: mmorel, haiming, Colin Eason
"""
import cv2
import sys
import os
import numpy as np
from svmClass import SvmClass

def print_Usage_and_Exit():
    print('Usage: svmTrainer.py (class name)')
    sys.exit(-1)
    
def main(arg=None):
    classes = []
    if len(sys.argv) >= 2:
        classes.append(SvmClass(sys.argv[1]))
    else:
        #get all classes
        for pathname in next(os.walk('classes'))[1] :            
            classes.append(SvmClass(pathname))
    
    for classInst in classes:
        print('Training SVM: ' + classInst.getName())
        classInst.train(True)
        classInst.save()
        print('Done. Output to ' + classInst.getName() + 'Svm.yml')
                    
if __name__ == '__main__':
    main()
