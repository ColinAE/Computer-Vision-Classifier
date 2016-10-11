~~Final Project~~
Python 3.5
OpenCV 3.1

Samples:
Many good condition samples have been hosted at ~cs510/data/haiming

Features:
-Object detection, tracking, and classification

Procedure to run :
1. Extract files
2. chmod +x *.py
3. python cs510hw4.py (input filename)

Command line options:

noact : do not perform action recognition

retrain : refreshes training state on action snippets instead of saved eigenvectors

edges : match on edge filtered images instead of spatial (required retrain flag as well to set or remove this mode)

mosse : select mosse tracking algorithm (default)

boost : select boosting tracking algorithm

mflow : select medianflow tracking algorithm


The video stream will open in a window with rectangles around detected objects and labels for actions


User keyboard options:

q: quit

space: Pause/Unpause 

d: Cycle through stages of object detection

1-4: Jump to specific debug frame playback

-: Slow down playback

+: Speed up playback (limited by frame processing delay)

s: Show/hide labels and bounding rectangles
