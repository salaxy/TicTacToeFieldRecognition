'''
Created on 16.02.2015

@author: Salaxy
'''
import cv2

from signRecognition import findASign


if __name__ == '__main__':

    roiDilat1 = cv2.imread("roiDilat1.png")
    roiDilat2 = cv2.imread("roiDilat2.png")

    roiOrg1 = cv2.imread("roiOrg1.png")
    roiOrg2 = cv2.imread("roiOrg2.png")
    roiOrg8 = cv2.imread("roiOrg8.png")


    findASign(roiOrg1)
    findASign(roiOrg2)
    findASign(roiOrg8)


    cv2.imshow("roiOrg1", roiOrg1)
    cv2.imshow("roiOrg2", roiOrg2)
    cv2.imshow("roiOrg8", roiOrg8)
    cv2.waitKey(0)