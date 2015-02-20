'''
Created on 07.02.2015

@author: Andy Klay
'''

import cv2
import numpy as np
from cmath import rect, pi

def findASign(image):

    edges = cv2.Canny(image,100,255)
    dilation = cv2.dilate(edges,np.ones((3,3),np.uint8),iterations = 2)
    cnts, hierarchy = cv2.findContours(dilation,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    foundaSignFlag=0
    sign=0
    signArea=0


    for ct in cnts:

        x,y,w,h = cv2.boundingRect(ct)
        rect_area = w*h
        #cv2.drawContours(image, [ct], -1, (255,50,200), 3)
        #print rect_area
        if (rect_area > 1000 and rect_area< 6000):
            #print area
            #print rect_area
            
            cv2.drawContours(image, [ct], -1, (0,255,255), 3)
            
            area = abs(cv2.contourArea(ct))
            perimeter = cv2.arcLength(ct,True)
            formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
            aspect_ratio = float(w)/h
            extent = float(area)/rect_area
            
            rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
            rect = rect.reshape((-1,1,2))
            cv2.drawContours(image, [rect], -1, (0,0,255), 3)

            signArea=rect_area
            foundaSignFlag=foundaSignFlag+1
            
            
            if (formFactor<0.5):
            #if (formFactor<0.5 and extent<0.5):
                sign = 1
            #elif(formFactor>0.5 and extent>0.5):
            elif(formFactor>0.5):
                sign = 2
            
            debug = False
            if(debug):
                print "formFactor: " + str(formFactor)
                print "aspect_ratio: " + str(aspect_ratio)
                print "perimeter: " + str(perimeter)
                print "extent: " + str(extent)
                print "area: " + str(area)
                print "countour_len: " + str(len(ct))

                if (sign == 1):
                    print "found a cross"
                elif(sign == 2):
                    print "found a circle"
        else:       
            # wenn kein rechteck in dieser groessenordung
            # gefunden wurde,
            #dann ist das Feld leer
            pass
        
    #print "foundaSignFlag: " + str(foundaSignFlag)
    #print "signArea: " + str(signArea)
    #print ""
    
    return sign
    pass

