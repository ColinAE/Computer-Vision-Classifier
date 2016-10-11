# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:51:41 2016

@author: mmorel
"""
import numpy as np
import cv2
from actionRecognize import ActionRecognize

def get3dPCA(vcube, dimens):
    x,y,t = dimens
    sliceXY = np.swapaxes(vcube,1,2).reshape(t,y*x,order='F')#.swapaxes(0,1)    
    #sliceXYn = sliceXY / sliceXY.max(axis=0)
    sliceXT = np.swapaxes(vcube,0,2).reshape(x,y*t)#.swapaxes(0,1)   
    #sliceXTn = sliceXT / sliceXT.max(axis=0)
    sliceYT = np.swapaxes(vcube,0,1).reshape(y,x*t,order='F')#.swapaxes(0,1)   
    #sliceYTn = sliceYT / sliceYT.max(axis=0)
    mean1, eigenXY = cv2.PCACompute(sliceXY, None) 
    mean2, eigenXT = cv2.PCACompute(sliceXT, None)   
    mean3, eigenYT = cv2.PCACompute(sliceYT, None)    
    #print (repr(eigenXY))
    return (eigenXY,eigenXT,eigenYT) 
    
def getPrincipleAngle(eigVecs1, eigVecs2):
    #e2t = cv2.transpose(eigVecs2)
    #e2t = cv2.transpose(eigVecs2)
    evm = cv2.gemm(eigVecs1,eigVecs2,1,None,1,flags=cv2.GEMM_2_T)
    #evm = np.outer(eigVecs1,e2t)
    w,u,vt = cv2.SVDecomp(evm)
    #print (repr(w))
    #print (repr(float(w[0])))
    return float(w[0]); 
    
def getAveragePrincipleAngle(eigVec3d1, eigVec3d2):
    paXY = getPrincipleAngle(eigVec3d1[0],eigVec3d2[0])
    paXT = getPrincipleAngle(eigVec3d1[1],eigVec3d2[1])
    paYT = getPrincipleAngle(eigVec3d1[2],eigVec3d2[2])
    return abs(paXY) + abs(paXT) + abs(paYT) # / 3.0;    
        
def main(arg=None):
    dimens = (2,3,4)
    frames = []
    frames.append([[1,2],[3,4],[5,6]])
    frames.append([[7,8],[9,10],[11,12]])
    frames.append([[13,14],[15,16],[17,18]])
    frames.append([[19,20],[21,22],[23,24]])
    vcube = np.array(frames,dtype=np.float32)
    cube1 = np.array( [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]], [[13,14],[15,16],[17,18]], [[19,20],[21,22],[23,24]]], dtype=np.float32 )
    cube2 = np.array( [[[1,1],[1,1],[1,0]], [[1,1],[1, 0],[1,  1]], [[0,  1],[1,  1],[1,  1]], [[1,  1],[0,  1],[1,  1]]], dtype=np.float32 )
    #cube1 = np.array( [[[1,1,1],[1,1,1],[1,1,0]], [[1,1,1],[1,0,1],[1,1,1]], [[0,1,1],[1,1,1],[1,1,1]], [[1,1,1],[0,1,1],[1,1,1]]], dtype=np.float32 )
    #cube2 = np.array( [[[1,1,0],[1,1,1],[1,1,0]], [[1,1,1],[1,0,1],[1,1,1]], [[0,1,1],[1,1,1],[1,1,1]], [[1,1,1],[0,1,1],[1,1,1]]], dtype=np.float32 )
    #cube2 = np.array( [[[3,1,5],[11,7,1],[0,2,9]], [[10,3,16],[1,0,31],[7,9,0]], [[4,8,1],[3,87,9],[55,0,9]], [[7,12,1],[0,1,1],[1,1,1]]], dtype=np.float32 )#cube2 = np.array( [[[13,14],[15,16]], [[17,18],[19,20]], [[21,22],[23,24]]], dtype=np.float32 )
    #print (repr(cube))
    #row_sums = cube1.sum(axis=1)
    #cube1 = cube1 / row_sums[:, np.newaxis]
    #row_sums = cube2.sum(axis=1)
    #cube2 = cube2 / row_sums[:, np.newaxis]
    #x,y,t = (2,3,4)
    #xyslice = np.swapaxes(vcube,1,2).reshape(t,y*x,order='F').swapaxes(0,1)
    #xtslice = np.swapaxes(vcube,0,2).reshape(x,y*t).swapaxes(0,1)
    
    #ytslice = np.swapaxes(vcube,0,1).reshape(y,x*t,order='F').swapaxes(0,1)
    pca1 = get3dPCA(cube1,dimens)
    pca2 = get3dPCA(cube2,dimens)
    pa = getAveragePrincipleAngle(pca1,pca2)
    print (repr(pa))
    #print (repr(xtslice))
    #print (repr(ytslice))
    #actionRecognizer = ActionRecognize('actions/',(50,100,45))
    
if __name__ == '__main__':
    main()