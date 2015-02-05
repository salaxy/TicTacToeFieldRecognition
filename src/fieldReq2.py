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
    time.sleep(0.5)
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame2 = np.copy(frame)
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
            
            if (area > 20000 and area< 60000):
                formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
                #print formFactor
                #print area
                cv2.drawContours(frame, [cnt], -1, (0,255,255), 3)
                
                if(formFactor>0.55 and formFactor<0.7):
                    
                    cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)

                    #TODO we need the corners for the robot actions

                
                    # This is the Region of Interest, so next step is to isolate it
                    rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
                    rect = rect.reshape((-1,1,2))
                    cv2.drawContours(frame, [rect], -1, (0,0,255), 3)
                    
                    roi = dilation[y:y+h,x:x+w]
                    roiFrame = frame2[y:y+h,x:x+w]
                    #roi = edges[100:200,100:300]
                    print roi.shape
                    #imgheader = cv2.cv.CreateImageHeader((roi[0], roi[1]), cv2.cv.IPL_DEPTH_8U, 1)
                    #opencvImg = np.asarray(imgheader[:,:])
                    cv2.imshow("roi", roi)
                    
                    #get the eges for rotation
                    cntsRoi, hierarchy = cv2.findContours(roi,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    #cv2.drawContours(roiFrame, cntsRoi, -1, (0,255,255), 3)
                    cv2.imshow("roiFrame", roiFrame)
                    
                    for found in cntsRoi:
                        area = abs(cv2.contourArea(found))
                        perimeter = cv2.arcLength(found,True)
                        
                        if (area > 20000 and area< 60000):
                        
                            formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
                            #print formFactor
                            #print area

                            if(formFactor>0.55 and formFactor<0.7):
                        
                                epsilon = 0.01*cv2.arcLength(found,True)
                                approx = cv2.approxPolyDP(found,epsilon,True)
                        
                                #cv2.drawContours(roiFrame, [found], -1, (255,0,255), 3)                        
                        
                                moment = cv2.moments(found)
    
                                #print len(approx)
                                #cv2.drawContours(roiFrame, [approx], -1, (255,0,255), 3)     
                                if(len(approx)==4):
                                    cv2.drawContours(roiFrame, [approx], -1, (255,0,255), 3)       
                        
                                    #rotating the roi for analysis of the sub sections
                                    (h, w) = roi.shape[:2]
                                    cx = int(moment['m10']/moment['m00'])
                                    cy = int(moment['m01']/moment['m00'])
                                    center = (cx, cy)
                                    #center = (w / 2, h / 2)
                                    matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
                                    rotated = cv2.warpAffine(roi, matrix, (w, h))
                                    cv2.imshow("rotated", rotated)
                        
                                    print approx[0]
                                    print approx[1]
                                    print approx[2]
                                    print approx[3]
                        
                                    #prepare correct transform.
                                    #pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
                                    pts1 = np.float32(approx)
                                    pts2 = np.float32([[0,0],[h,0],[0,w],[w,h]])
                        
                                    transformMatrix = cv2.getPerspectiveTransform(pts1,pts2)
                                    dst = cv2.warpPerspective(roiFrame,transformMatrix,(w,h))
                                    #dst = cv2.warpAffine(roiFrame, transformMatrix, (w, h))
                                    cv2.imshow("dst", dst)

    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()