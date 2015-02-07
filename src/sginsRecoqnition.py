'''
Created on 07.02.2015

@author: Andy Klay
'''

import cv2
import numpy as np
from cmath import rect, pi

def findSign(edgesImage, image):

    edges = cv2.Canny(image,100,255)
    dilation = cv2.dilate(edges,np.ones((3,3),np.uint8),iterations = 2)
    cnts, hierarchy = cv2.findContours(dilation,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    foundaSignFlag=0
    signArea=0


    for ct in cnts:

        x,y,w,h = cv2.boundingRect(ct)
        rect_area = w*h
        #cv2.drawContours(image, [ct], -1, (255,50,200), 3)
        
        if (rect_area > 1000 and rect_area< 5000):
            #print area
            #print rect_area
            
            #cv2.drawContours(image, [ct], -1, (0,255,255), 3)
            
            area = abs(cv2.contourArea(ct))
            perimeter = cv2.arcLength(ct,True)
            formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
            aspect_ratio = float(w)/h

            #extent = float(area)/rect_area
            
            rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
            rect = rect.reshape((-1,1,2))
            cv2.drawContours(image, [rect], -1, (0,0,255), 3)

            signArea=rect_area
            foundaSignFlag=foundaSignFlag+1
        else:       
            # wenn kein rechteck in dieser groessenordung
            # gefunden wurde,
            #dann ist das Feld leer
            pass
        
        
    print "foundaSignFlag: " + str(foundaSignFlag)
    print "signArea: " + str(signArea)
    return foundaSignFlag
    pass

if __name__ == '__main__':

    roiDilat1 = cv2.imread("roiDilat1.png")
    roiDilat2 = cv2.imread("roiDilat2.png")

    roiOrg1 = cv2.imread("roiOrg1.png")
    roiOrg2 = cv2.imread("roiOrg2.png")


    findSign(roiDilat1, roiOrg1)
    findSign(roiDilat2, roiOrg2)


    cv2.imshow("roiOrg1", roiOrg1)
    cv2.imshow("roiOrg2", roiOrg2)
    cv2.waitKey(0)
