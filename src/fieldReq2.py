'''
Created on 02.02.2015

@author: Andy Klay
'''

import cv2
import numpy as np
from cmath import rect, pi
import time

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
    
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
     
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    
    return hnew


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
     
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
     
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
     
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    #rect = order_points(pts)
    rect = rectify(pts)
    (tl, tr, br, bl) = rect
     
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[0]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
     
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
     
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
     
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
     
    # return the warped image
    return warped

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    # three frames per secound are enough
    # for gamestate recoqnition
    #time.sleep(0.33)
    time.sleep(0.6)
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
    
                                #print len(approx)
                                #cv2.drawContours(roiFrame, [approx], -1, (255,0,255), 3)     
                                if(len(approx)==4):
                                    cv2.drawContours(roiFrame, [approx], -1, (255,0,255), 3)       
                        
                                    #rotating the roi for analysis of the sub sections
                                    ( width, height) = roi.shape[:2]
                                    #moment = cv2.moments(approx)
                                    #cx = int(moment['m10']/moment['m00'])
                                    #cy = int(moment['m01']/moment['m00'])
                                    #center = (cx, cy)
                                    #center = (w / 2, h / 2)
                                    #matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
                                    #rotated = cv2.warpAffine(roi, matrix, (w, h))
                                    #cv2.imshow("rotated", rotated)
                        
                                    print approx
                                    #print corners
                                    cv2.drawContours(roiFrame, [approx[0]], -1, (0,255,0), 10)
                                    cv2.drawContours(roiFrame, [approx[1]], -1, (0,255,0), 10)
                                    cv2.drawContours(roiFrame, [approx[2]], -1, (0,255,0), 10)
                                    cv2.drawContours(roiFrame, [approx[3]], -1, (0,255,0), 10)
                                    
                                    
                        
                                    #prepare correct transform.
                                    pts1 = np.array(approx, dtype = "float32")
                                    #pts2 = np.float32([[0,0],[height,0],[0,width],[width,height]])
                                    #transformMatrix = cv2.getPerspectiveTransform(pts1,pts2)
                                    #dst = cv2.warpPerspective(roiFrame,transformMatrix,(height,width))
                                    #cv2.imshow("dst", dst)
                                    
                                    warped = four_point_transform(roiFrame, pts1)
                                    cv2.imshow("warped", warped)

    cv2.imshow('frame',frame)
    

    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


