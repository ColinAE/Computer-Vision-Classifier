# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:44:46 2016
Modified on Tues Feb 9 2016
@author: mmorel, haiming, Colin Eason
"""

import cv2
import sys
import numpy as np
import time
import copy
import random
from objectDetector import ObjectDetector

class ObjectExtractor:
    
    def __init__(self, path,width=64,height=48):
        self.root = path
        self.fgcount = 0
        self.bgcount = 0
        self.width = width
        self.height = height
        
    def extract(self,img,mask,points):
        cntlist = points[1]
        height, width = img.shape[:2]
        for cnt in cntlist:
            self.fgcount = self.fgcount + 1
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            sample = img[y:y+h, x:x+w]
            cv2.imwrite( self.root + '/fg/' + repr(self.fgcount) + '.png',sample)
            
    def extractBg(self,img,points):
        self.bgcount = self.bgcount + 1
        height, width = img.shape[:2]
        size = (random.randint(50,100),random.randint(50,100))
        loc = (random.randint(0,width-size[0]),random.randint(0,height-size[1]))
        x,y = loc
        w,h = size
        sample = img[y:y+self.height, x:x+self.width]
        cv2.imwrite( self.root + '/bg/' + repr(self.bgcount) + '.png',sample)

def print_Usage_and_Exit():
    print('Usage: extractObjects.py (input file) (output dir)')
    sys.exit(-1)
    
#first, get command line arguments
def main(arg=None):
    if len(sys.argv) < 3:
        print_Usage_and_Exit()
    processVideo(sys.argv[1])
    cv2.destroyAllWindows()

def processVideo(src):
    
    # eat first few frames to let video stabilize
    reservedFrames = 10

    frameCount = 0    
    
    # Frame display selector
    select = 0
    
    # Open video stream
    videoStream = cv2.VideoCapture(src)
    
    width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Calculate # of milliseconds each frame is displayed
    fps = videoStream.get(cv2.CAP_PROP_FPS)
    if (fps > 0):
        spf = (1/fps)*1000
    else:
        spf = 30

    # Offset for user speed control
    delayScale = 0
    
    # Foreground/Background separator
    objdet = ObjectDetector() 
    
    #objtrk = ObjectTracker()
    
    objext = ObjectExtractor(sys.argv[2])

    displayBounds = True              
    
    suppressBgMove = True   

    while(videoStream.isOpened()):
    
        # Read next frame
        ret, frame = videoStream.read()
        
        frameCount = frameCount + 1
                   
        # Process frames until none are left
        if ret==True and frameCount >= reservedFrames:      

            start = time.time()            
            
            if width >= 1000:
                frame = cv2.pyrDown(frame)  
                      
            # Attempt tracking found objects.  Needs to be done before seeding
            #objtrk.update(frame)
                    
            # Detect objects
            if frameCount % 5 == 0:
                points = objdet.detectObjects(frame)
                
                #objtrk.seedTracks(frame, objdet.getLastFrameFiltered(), points)
            
                objext.extract(frame, objdet.getLastFrameFiltered(), points)
                
                objext.extractBg(frame, points)
                       
            dispFrame = copy.deepcopy(frame)
            title = 'Original Image'
            color = (0,255,0)
            
            # Display different stage of object detection process if desired
            if select == 1:   
                dispFrame = objdet.getLastFrameBgSubtract()
                title = 'BG Subtract'
                color = (255,255,255) 
            elif select == 2:
                dispFrame = objdet.getLastFrameFiltered()
                title = 'Filtered'
                color = (255,255,255)  
            elif select == 3:
                dispFrame = objdet.getLastFrameContours()
                title = 'Contours'
                color = (255,255,255) 
                
            if displayBounds == True:
                #objtrk.drawBounds(title,dispFrame,color)
                objdet.drawBounds(title,points,dispFrame,color)
                    
            cv2.imshow('frame',dispFrame)
                
            # Calculate delay until next frame is shown
            end = time.time()
            elapsed = (end - start) * 1000
            delay = int(spf - elapsed)
            delay += delayScale
            if delay <= 0:
                delay = 1
        
            ## User controls
            # ~
            # Wait for user input
            k = cv2.waitKey(delay) & 0xFF

            # quit
            if k == ord('q'):
                break
            # Cycle through stages of object detection
            elif k == ord('d'):
                select = select + 1
                select = select % 4
            # Show object-detected input
            elif k == ord('1'):
                select = 0
            # Show uncleaned background separation frames
            elif k == ord('2'):
                select = 1
            # Show cleaned background separated frames
            elif k == ord('3'):
                select = 2
            # Show frames that contain blobs of objects
            elif k == ord('4'):
                select = 3
            # Show or hide bounding rectangles and labels
            elif k == ord('s'): 
                displayBounds = not displayBounds
            # Enable/disable background movement suppression
            elif k == ord('v'): 
                suppressBgMove = not suppressBgMove
            # Slow playback down
            elif k == ord('-'):
                delayScale += 10
            # Speed playback up
            elif k == ord('+'):
                delayScale -= 10
            # Pause/Unpause
            elif k == ord(' ') :
                while True:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord(' '):
                        break  
            # ~         
        elif ret == False:
            break
      
    videoStream.release()
   
    return
    
    
if __name__ == '__main__':
    main()
