

import numpy as np
import cv2

import dlib

#import pyautogui

from ahk import AHK
ahk = AHK()

#for the right camera
cap = cv2.VideoCapture(1)
#fish, chicken = cap.read()
#if not fish:
    #cap = cv2.VideoCapture(1)
#------------------------------
    
#face detection   
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
#nose_cascade = cv2.CascadeClassifier('data/haarcascade_nose.xml')
#nose_cascade = cv2.CascadeClassifier('data/Nariz.xml')
#lucas kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

canny_thresh = 150
accum_thresh = 20
minR = 0
maxR = 30

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


def setDim(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return (width, height)          

def rect_to_bb(rect):
    #This code is shamelessly stolen from https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def main():
    dim = (0, 0)
    while(True): 
        ###ret is true if there is video. 
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if dim == (0, 0):
            dim = setDim(frame, 200)
        #frame = cv2.resize(frame, dim)
        
        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #s = 0.1
        #m = 1 - s*8 
        #kernel = np.array([[s,s,s], [s,m,s], [s,s,s]])
        #frame = cv2.filter2D(frame, -1, kernel)     
        
        #prtsc = pyautogui.screenshot()
        #cv2.imshow('prtsc', prtsc)
        
        #faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            print("fish")
        #ret, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        
        faces = face_cascade.detectMultiScale(gray)
        
        
        #detections = detector(frame)        
        #for detection in (detections): 
            #(x, y, w, h) = rect_to_bb(detection)
            #cv2.rectangle(frame,(x,y),(w,h),(0,255,0), 2 )
            
        
        # Read the parameters from the GUI
        param1 = cv2.getTrackbarPos('Canny Threshold', 'Hough Circle Transform')
        param2 = cv2.getTrackbarPos('Accumulator Threshold', 'Hough Circle Transform')
        minRadius = cv2.getTrackbarPos('Min Radius', 'Hough Circle Transform')
        maxRadius = cv2.getTrackbarPos('Max Radius', 'Hough Circle Transform')
            
        
        #for (x, y, w, h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  # Draw a blue box around each face.
            ## Define a "Region of Interest" for the eye detector, since eyes are typically found on a face.
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = frame[y:y+h, x:x+w]        
            #eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            
            ##print(type(h*0.5))   #<class 'numpy.intc'>
            ##print(np.intc(h*0.5)) 
            #cv2.rectangle(frame,(x,y),(x+w,y+(np.intc(h*0.5))),(0,0,255),2)

            #for (ex,ey,ew,eh) in eyes:
                
                #if (ey+eh < np.intc( h *0.5)) :
                    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  # Draw a green box around each eye.    
                    #roi_eyes = roi_color[ey:ey+eh, ex:ex+ew]
                    #roi_eyesgray = cv2.cvtColor(roi_eyes, cv2.COLOR_BGR2GRAY)
                    #circles = cv2.HoughCircles(roi_eyesgray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1,
                                                #param2=param2, minRadius=minRadius, maxRadius=maxRadius)                                    
                    #if circles is not None:
                        #circles = np.uint16(np.around(circles))
                
                        #for i in circles[0,:]:
                            ## Draw the outer circle
                            #cv2.circle(roi_eyes,(i[0],i[1]),i[2],(0,255,0),2)
                            ## Draw the center of the circle
                            #cv2.circle(roi_eyes,(i[0],i[1]),2,(0,0,255),3)
                        
                    
            #####noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
            #####for (nx,ny,nw,nh) in noses:
                #####cv2.rectangle(roi_color,(ex,ey),(nx+nw,ny+nh),(0,0,255),2)  # Draw a red box around each nose.    
                #####roi_nose = roi_color[ny:ny+nh, nx:nx+nw]
            
    
            #print(x, y, w, h)
        cv2.imshow('frame', frame)
        #cv2.imshow('gray', gray)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return


# Create the GUI elements
cv2.namedWindow('Hough Circle Transform')
cv2.createTrackbar('Canny Threshold', 'Hough Circle Transform', 1, 500, nothing)
cv2.createTrackbar('Accumulator Threshold', 'Hough Circle Transform', 1, 500, nothing)
cv2.createTrackbar("Min Radius", 'Hough Circle Transform', 0, 100, nothing)
cv2.createTrackbar("Max Radius", 'Hough Circle Transform', 1, 100, nothing)
# Set some default parameters
cv2.setTrackbarPos("Max Radius", 'Hough Circle Transform', maxR)
cv2.setTrackbarPos("Canny Threshold", 'Hough Circle Transform', canny_thresh)
cv2.setTrackbarPos("Accumulator Threshold", 'Hough Circle Transform', accum_thresh)


##dlib
#detector = dlib.simple_object_detector("detector.svm")

main()