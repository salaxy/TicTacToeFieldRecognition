'''
Created on 02.02.2015

@author: Andy Klay
'''

import cv2
import numpy as np
from cmath import rect, pi
import time


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    # three frames per secound are enough
    # for gamestate recoqnition
    #time.sleep(0.33)
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv,(5,5),0)
    
    
    edges = cv2.Canny(frame,100,255)
    dilation = cv2.dilate(edges,np.ones((3,3),np.uint8),iterations = 1)
    
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    
    
    for cnt in contours:
        
        
        #if(cv2.isContourConvex(cnt)):
        if(True):
            
            #cv2.drawContours(frame, [cnt], -1, (255,0,0), 3)
            
            M = cv2.moments(cnt)
            #cx = int(M['m10']/M['m00'])
            #cy = int(M['m01']/M['m00'])
            area = abs(cv2.contourArea(cnt))
            perimeter = cv2.arcLength(cnt,True)
            #print perimeter
            
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            
            #cv2.drawContours(frame, [approx], -1, (0,0,255), 3)
            
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            rect_area = w*h
            extent = float(area)/rect_area
            
            if (area > 0 and area< 50000):
                formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
                print formFactor
                
                if(formFactor>0.01 and formFactor<1.0):
                    cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)
                    #cv2.drawContours(frame, contours, -1, (0,0,255), 3)
                #berechne Flaeche des aktuellen Objektes
                #realArea = ABS(cvContourArea(c));
                #realArea = abs(cv2.contourArea(cnt))
                #convex = cv2.isContourConvex(cnt)
                
                    a,b,c,d = cv2.boundingRect(cnt)
                #rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
                #rect = rect.reshape((-1,1,2))
                #rectArea = abs(cv2.contourArea(rect))
            
    
    

    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()