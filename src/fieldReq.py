'''
Created on 02.02.2015

@author: Andy Klay
'''

import cv2
import numpy as np
from cmath import rect

'''
Created on 02.02.2015

@author: Andy Klay
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(frame,100,200)
    #http://en.wikipedia.org/wiki/Canny_edge_detector
    
    #contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    
    #cnt = contours[4]
    #cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
    
    for cnt in contours:
        #print cnt
        
        #berechne Flaeche des aktuellen Objektes
        #realArea = ABS(cvContourArea(c));
        realArea = abs(cv2.contourArea(cnt))
        convex = cv2.isContourConvex(cnt)
        #print realArea
        
        #TODO vorher villt noch das weisse Blatt papier
        # isolieren und dann weiter arbeiten!
        
        x,y,w,h = cv2.boundingRect(cnt)
        rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
        rect = rect.reshape((-1,1,2))
        rectArea = abs(cv2.contourArea(rect))
        
        if ((rectArea > 25000) and (rectArea < 50000)):
            print rectArea
            crop_img = cv2.polylines(frame,[rect],True,(0,255,255))
            
            #crop_img = edges[x:y,w:h]
            
            #imgheader = cv2.cv.CreateImageHeader((crop_img[0], crop_img[1])),
            #opencvImg = np.asarray(imgheader[:,:])
            #cv2.imwrite('roi.png', opencvImg)
            #cv2.imshow("cropped", crop_img)
       
       
        #cv2.drawContours(frame, rect, -1, (0,0,255), 3)
        #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #TODO hier nur die grossen druch lassen
        
        #rect = cv2.minAreaRect(cnt)
        #box = cv2.boxPoints(rect)
        #box = np.int0(rect)
        #im = cv2.drawContours(frame,[rect],0,(0,0,255),2)
        
        #cv2.approxPolyDP(curve, epsilon, closed)s
        #cv2.approxPolyDP(cnt, output, True)
        
        #skip small contours
        #if( (realArea < 50) or (not convex)):
        #if(realArea < 0):
        #    continue
        
        #cv2.drawContours(frame, cnt, -1, (255,255,0), 3)
        
        if(not convex):
            continue
        
        cv2.drawContours(frame, cnt, -1, (0,255,255), 3)
        
        #if (cv2.isContourConvex(cnt)):
        
            #cv2.drawContours(frame, cnt, -1, (0,0,255), 3)
        
        #if (realArea > minAreaBlue && realArea< maxAreaBlue)
        #if ((realArea > 100) and (realArea< 1000)):
        
            #//berechne Umfang des aktuellen Objektes
            #arcLength = cvArcLength(c);
            #arcLength = cv2.arcLength(curve, closed)
        
            #//berechne Formfaktor des aktuellen Objektes
            #formFactor = ABS(1/ ((arcLength*arcLength) / (4*PI*realArea )));
        
            #if(formFactor>0.73 && formFactor<0.85){
        
                #//berechne Rahmen des aktuellen Objektes
                #rect = cvContourBoundingRect(c);
                #rectArea=rect.width*rect.height;
                #cvDrawContours( blueImage[3], c, aussereFarbe, innereFarbe, 1, 3, 8 );
                #//objekt in vector speichern
                #blueRectObjects.push_back(rect);
        
        
        cv2.drawContours(frame, cnt, -1, (255,0,0), 3)

    # define range of blue color in HSV
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    #cv2.imshow('edges',edges)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()