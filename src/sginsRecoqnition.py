'''
Created on 07.02.2015

@author: Salaxy
'''

import cv2
import numpy as np
from cmath import rect, pi
import time
from array import array


def findSign(edgesImage, image):


    edges = cv2.Canny(image,100,255)
    dilation = cv2.dilate(edges,np.ones((1,1),np.uint8),iterations = 2)
    cnts, hierarchy = cv2.findContours(dilation,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    for ct in cnts:
        area = abs(cv2.contourArea(ct))
        perimeter = cv2.arcLength(ct,True)
        
        cv2.drawContours(image, [ct], -1, (255,50,200), 3)
        if (area > 50 and area< 500):
            print area
            formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
            cv2.drawContours(image, [ct], -1, (0,255,255), 3)



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
