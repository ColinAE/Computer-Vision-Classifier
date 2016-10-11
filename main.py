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
from objectDetector import ObjectDetector
from objectTracker import ObjectTracker
from common import RectSelector

#objtrk = None

#lastFrame = None

def print_Usage_and_Exit():
    print('Usage: cs510hw4.py (input file) [options]')
    sys.exit(-1)
    
# Get all command line arguments.
def main(arg = None):
    if len(sys.argv) < 2:
        print_Usage_and_Exit()   
    processVideo(sys.argv[1])
    cv2.destroyAllWindows()    

def processVideo(src):
    
    # Skips first few frames to let the video stabilize.
    # When physically activating video recording on a camera, 
    # the the force will create small movements in the video.
    # Since the program cannot handle camera movement,
    # the first few frames must be skipped to allow the stream to stabilize and
    # avoid classification errors.
    reservedFrames = 10

    frameCount = 0
    
    # Frame display selector
    select = 0
    
    # Open video stream
    videoStream = cv2.VideoCapture(src)
    
    # Video width
    width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Video height
    height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video codec
    codec = int(videoStream.get(cv2.CAP_PROP_FOURCC))
    
    # Calculates # of milliseconds each frame is displayed
    fps = videoStream.get(cv2.CAP_PROP_FPS)
    if (fps > 0):
        spf = (1/fps)*1000
    else:
        spf = 30

    # Offset for user speed control
    delayScale = 0
    
    # Foreground/Background separator
    objdet = ObjectDetector()   
    
    # Select tracking algorithm
    # Mosse is default
    trackAlgorithm = 'mosse'
    trackAlgorithm = 'mflow' if 'mflow' in sys.argv else trackAlgorithm
    trackAlgorithm = 'boost' if 'boost' in sys.argv else trackAlgorithm
    
    matchedges = True if 'edges' in sys.argv else False 
    filterVelocity = False if 'novel' in sys.argv else True 
    replayOnly = True if 'replay' in sys.argv else False 
    writeVideo = True if 'write' in sys.argv and not replayOnly else False 
    actRec = False if 'noact' in sys.argv and not replayOnly else True
    retrain = True if 'retrain' in sys.argv else False 
    dummy = np.zeros((height, width), np.uint8)
    cv2.imshow('frame',dummy)
    
    # Object Tracker
    objtrk = ObjectTracker('output_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.txt', trackAlgorithm, drawKp='drawkp' in sys.argv, doClassify=False, doActRecog=actRec,outFps=fps,matchEdges=matchedges,retrain=retrain,filterVelocity=filterVelocity)

    # Displays boundaries of identified objects
    displayBounds = True              
    
    suppressBgMove = True   
    
    # Opens output stream
    if writeVideo:
        outputStream = cv2.VideoWriter(time.strftime("%Y-%m-%d_%H-%M-%S") +'.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
        if not outputStream.isOpened():
            outputStream = None
            print('Error opening output video stream')  
            
    # Allows users to pause video 
    paused = False
    
    #lastFrame = None
    
    #lastRect = None

    while(videoStream.isOpened()):
    
        # Read next frame
        if not paused:
            ret, frame = videoStream.read()
            frameCount = frameCount + 1
                   
        # Process frames until none are left
        if ret==True and frameCount >= reservedFrames:      
            lastFrame = frame
            start = time.time()            
            if width >= 1000:
                frame = cv2.pyrDown(frame)  
                      
            # Attempt tracking found objects.  Needs to be done before seeding
            if not paused and not replayOnly:
                objtrk.update(frame)
                    
                # Detect objects
                if frameCount % 1 == 0:
                    points = objdet.detectObjects(frame)
                
                    objtrk.seedTracks(frame, objdet.getLastFrameFiltered(), points)
                       
            dispFrame = copy.deepcopy(frame)
            title = 'Original Image'
            color = (0,255,0)
            
            # Allos user to display different stages of object detection
            if select == 1 and not replayOnly:   
                dispFrame = objdet.getLastFrameBgSubtract()
                title = 'BG Subtract'
                color = (255,255,255) 
            elif select == 2 and not replayOnly:
                dispFrame = objdet.getLastFrameFiltered()
                title = 'Filtered'
                color = (255,255,255)  
            elif select == 3 and not replayOnly:
                dispFrame = objdet.getLastFrameContours()
                title = 'Contours'
                color = (255,255,255) 
            
            # Displays boundaries of detected objects
            if displayBounds == True and not replayOnly:
                objtrk.drawBounds(title,dispFrame,color)
                
            if not replayOnly:    
                objtrk.writeTracks(frameCount)
            
            # Displays the frame        
            cv2.imshow('frame',dispFrame)
            
            # Writes video to output stream frame-by frame
            if writeVideo and not paused and not replayOnly:               
                outputStream.write(dispFrame)
                
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
            if k == ord('w'):
                objtrk.saveTracks()
            if k == ord('t'):
                objtrk.clearTracks()   

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
                paused = not paused
               # while True:
               #     k = cv2.waitKey(1) & 0xFF
               #     if k == ord(' '):
               #         break  
            # ~         
        elif ret == False:
            break
    
    # Cleanup
    if writeVideo:
        if not outputStream is None:
            outputStream.release()
    videoStream.release()
    objtrk.closeOut()
   
    return
    
    
if __name__ == '__main__':
    main()
