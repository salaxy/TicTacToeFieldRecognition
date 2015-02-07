'''
Created on 02.02.2015

@author: Andy Klay
'''

import cv2
import numpy as np
from cmath import rect, pi
import time
from array import array
from sginsRecoqnition import findSign


def schnibbidiSchnapp(wrapedEdges, warpedImage):

    # villt noch mal sicher gehen das es auchdas Feld mit 9 Einzelfeldern ist

    wrapedEdges
    ( height, width) = wrapedEdges.shape[:2]
    widthPart = width/3
    heightPart = height/3
    
    heightRowTwo = heightPart
    heightRowThree = heightPart*2
    
    widthColumnTwo = widthPart
    widthColumnThree = widthPart*2
    
    edgeImageList = []
    orgImageList = []

    
    edgeImageList.append(wrapedEdges[0:heightRowTwo,0:widthColumnTwo])
    orgImageList.append(warpedImage[0:heightRowTwo,0:widthColumnTwo])
    cv2.imshow("roiDilat 1", edgeImageList[-1])
    eins=findSign(warpedImage)
    #cv2.imwrite("roiDilat1.png", edgeImageList[-1])

    edgeImageList.append(wrapedEdges[0:heightRowTwo,widthColumnTwo:widthColumnThree])
    orgImageList.append(warpedImage[0:heightRowTwo,widthColumnTwo:widthColumnThree])
    cv2.imshow("roiDilat 2", edgeImageList[-1])
    zwei=findSign(warpedImage)
    #cv2.imwrite("roiDilat2.png", edgeImageList[-1])

    edgeImageList.append(wrapedEdges[0:heightRowTwo,widthColumnThree:width])
    orgImageList.append(warpedImage[0:heightRowTwo,widthColumnThree:width])
    cv2.imshow("roiDilat 3", edgeImageList[-1])
    eins=findSign(warpedImage)
    #cv2.imwrite("roiDilat3.png", edgeImageList[-1])

    
    #edgeImageList.append(wrapedEdges[heightRowTwo:heightRowThree,0:widthColumnTwo])
    #cv2.imshow("roiDilat 4", edgeImageList[-1])
    #edgeImageList.append(wrapedEdges[heightRowTwo:heightRowThree,widthColumnTwo:widthColumnThree])
    #cv2.imshow("roiDilat 5", edgeImageList[-1])
    #edgeImageList.append(wrapedEdges[heightRowTwo:heightRowThree,widthColumnThree:width])
    #cv2.imshow("roiDilat 6", edgeImageList[-1])
    
    #edgeImageList.append(wrapedEdges[heightRowThree: height,0:widthColumnTwo])
    #cv2.imshow("roiDilat 7", edgeImageList[-1])
    #edgeImageList.append(wrapedEdges[heightRowThree: height,widthColumnTwo:widthColumnThree])
    #cv2.imshow("roiDilat 8", edgeImageList[-1])
    #edgeImageList.append(wrapedEdges[heightRowThree: height,widthColumnThree:width])
    #cv2.imshow("roiDilat 9", edgeImageList[-1])
    
    
    listOfCnts = []
    n=0
    for elem in edgeImageList:
        
        cnts, hierarchy = cv2.findContours(elem,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
         
        listOfCnts.append(cnts)
        
        for ct in cnts:
            area = abs(cv2.contourArea(ct))
            perimeter = cv2.arcLength(ct,True)
            
            #cv2.drawContours(warpedFrame, [found], -1, (255,50,200), 3)
            if (area > 50 and area< 500):
                print area
                formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
                cv2.drawContours(orgImageList[n], [ct], -1, (0,255,255), 3)
        
        #print "nr."+ str(n)
        #print len(cnts)
        n=n+1
    
    cv2.imshow("roiOrg1", orgImageList[0])
    #cv2.imwrite("roiOrg1.png", orgImageList[0])
    cv2.imshow("roiOrg 2", orgImageList[1])
    #cv2.imwrite("roiOrg2.png", orgImageList[1])
    cv2.imshow("roiOrg 3", orgImageList[2])
    #cv2.imwrite("roiOrg3.png", orgImageList[2])
    
    #for elem in edgeImageList:
        
        #cv2.imshow("roiDilat"+'n', elem)
    
        #n=n+1
        #print elem
    #allFields = array([,,,
    #                   ,,,
    #                   ,,])
    
    #cv2.imshow("roi1", wrappedField[0:heightRowTwo,0:widthColumnTwo])
    #cv2.imshow("roi2", wrappedField[heightRowThree,widthColumnThree:width])
    #cv2.imshow("roi2", wrappedField[heightRowThree,widthColumnThree:width])
    
    # This is the Region of Interest, so next step is to isolate it
    #rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
    #rect = rect.reshape((-1,1,2))
    #cv2.drawContours(frame, [rect], -1, (0,0,255), 3)
    
    #roiDilat = dilation[y:y+h,x:x+w]
    #roiFrame = frameCopy[y:y+h,x:x+w]
    #roiDilat = edges[100:200,100:300]
    #print roiDilat.shape
    #imgheader = cv2.cv.CreateImageHeader((roiDilat[0], roiDilat[1]), cv2.cv.IPL_DEPTH_8U, 1)
    #opencvImg = np.asarray(imgheader[:,:])
    #cv2.imshow("roiDilat", roiDilat)
    
    #return hnew


def schnibbidiSchnapp2(warpedEdges, warpedFrame):

    # villt noch mal sicher gehen das es auchdas Feld mit 9 Einzelfeldern ist

    cntsRoi, hierarchy = cv2.findContours(warpedEdges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(roiFrame, cntsRoi, -1, (0,255,255), 3)
    
    list = []
    
    for found in cntsRoi:
        area = abs(cv2.contourArea(found))
        perimeter = cv2.arcLength(found,True)
        
        print area
        
        #cv2.drawContours(warpedFrame, [found], -1, (255,50,200), 3)
        
        if (area > 100 and area< 100000):
        
            formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
            #print formFactor
            #print area
            #cv2.drawContours(warpedFrame, [found], -1, (200,50,255), 3)

            #print formFactor
            if(formFactor>0.55 and formFactor<0.7):
        
                epsilon = 0.01*cv2.arcLength(found,True)
                approx = cv2.approxPolyDP(found,epsilon,True)
                
                cv2.drawContours(warpedFrame, [found], -1, (150,100,100), 3)
            
                list.append(found)
            
    if(len(list)>=9):
        print "all fields req"
    cv2.imshow("warpedFrame2", warpedFrame)
    cv2.imshow("warpedEdges2", warpedEdges)           

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
    time.sleep(0.4)
    _, frame = cap.read()

    # Convert BGR to HSV
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frameCopy = np.copy(frame)
    #blur = cv2.GaussianBlur(hsv,(5,5),0)
    
    
    edges = cv2.Canny(frame,100,255)
    dilation = cv2.dilate(edges,np.ones((1,1),np.uint8),iterations = 2)
    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    
    for cnt in contours:
        
        #if(cv2.isContourConvex(cnt)):
        #cv2.drawContours(frame, [cnt], -1, (255,0,0), 3)
        
        #M = cv2.moments(cnt)
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        area = abs(cv2.contourArea(cnt))
        #epsilon = 0.1*cv2.arcLength(cnt,True)
        
        if (area > 20000 and area < 70000):
            
            perimeter = cv2.arcLength(cnt,True)
            formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
            #print formFactor
            #print area
            #cv2.drawContours(frame, [cnt], -1, (0,255,255), 3)
            
            if(formFactor>0.55 and formFactor<0.7):
                
                x,y,w,h = cv2.boundingRect(cnt)
                #aspect_ratio = float(w)/h
                #rect_area = w*h
                #extent = float(area)/rect_area

                #TODO we need the corners for the robot actions
                # This is the Region of Interest, so next step is to isolate it
                rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
                rect = rect.reshape((-1,1,2))
                cv2.drawContours(frameCopy, [rect], -1, (0,0,255), 3)
                cv2.drawContours(frameCopy, [cnt], -1, (0,255,0), 3)
                #verstaerkender Rhamen
                cv2.drawContours(frame, [cnt], -1, (0,255,0), 5)
                #roiDilat = dilation[y:y+h,x:x+w]
                roiFrame = np.copy(frame[y:y+h,x:x+w])
                #roiEdges = edges[y:y+h,x:x+w]
                #print roiDilat.shape
                #imgheader = cv2.cv.CreateImageHeader((roiDilat[0], roiDilat[1]), cv2.cv.IPL_DEPTH_8U, 1)
                #opencvImg = np.asarray(imgheader[:,:])
                #cv2.imshow("roiDilat", roiDilat)
                
                #get the eges for rotation
                #cv2.drawContours(roiFrame, cntsRoi, -1, (0,255,255), 3)
                roiEdges = cv2.Canny(roiFrame,50,255)
                roiEdges = cv2.dilate(roiEdges,np.ones((2,2),np.uint8),iterations = 1)
                cntsRoi, hierarchy = cv2.findContours(roiEdges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)    
                    

                    
                #secound stage ******************************************************************************* 
                #wrapping of whole gamefield in a new image
                for found in cntsRoi:
                    area = abs(cv2.contourArea(found))
                    #print area
                    
                    if (area > 20000 and area < 100000):
                        perimeter = cv2.arcLength(found,True)                        
                        formFactor = abs(1/ ((perimeter*perimeter) / (4*pi*area )));
                        #print formFactor
                        #print area

                        #cv2.drawContours(roiFrame, [found], -1, (255,0,255), 3)    
                        if(formFactor>0.55 and formFactor<0.7):
                    
                            epsilon = 0.01*cv2.arcLength(found,True)
                            approx = cv2.approxPolyDP(found,epsilon,True)
                    
                            #cv2.drawContours(roiFrame, [found], -1, (255,0,255), 3)                        
                
                            #print len(approx)
                            #cv2.drawContours(roiFrame, [approx], -1, (255,0,255), 3)     
                            if(len(approx)==4):
                                     
                    
                                #rotating the roiDilat for analysis of the sub sections
                                ( width, height) = roiEdges.shape[:2]
                                #moment = cv2.moments(approx)
                                #cx = int(moment['m10']/moment['m00'])
                                #cy = int(moment['m01']/moment['m00'])
                                #center = (cx, cy)
                                #center = (w / 2, h / 2)
                                #matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
                                #rotated = cv2.warpAffine(roiDilat, matrix, (w, h))
                                #cv2.imshow("rotated", rotated)
   
                                if(False):
                                    print approx
                                    #draw corners and shape
                                    cv2.drawContours(roiFrame, [approx], -1, (255,0,255), 3)  
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
                                
                                warpedFrame = four_point_transform(roiFrame, pts1)
                                warpedEdges = four_point_transform(roiEdges, pts1)
                                #warpedDilat = four_point_transform(roiDilat, pts1)
                                #cv2.imshow("warpedFrame", warpedFrame)
                                #cv2.imshow("warpedEdges", warpedEdges)
                                
                                
                                
                                # TODO schnibbidischapp into nine parts
                                schnibbidiSchnapp(warpedEdges, warpedFrame)
                                #schnibbidiSchnapp2(warpedEdges, warpedFrame)
                                
                #cv2.imshow("roiDilat", roiDilat)
                cv2.imshow("roiFrame", roiFrame)

        
    cv2.imshow('frameCopy',frameCopy)
    #cv2.imshow('frame',frame)
    #cv2.imshow('edges',edges)
    #cv2.imshow('dilation',dilation)
    

    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


