# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

import time


import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import models



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class mynn(torch.nn.Module):
    def __init__(self):
        super(mynn, self).__init__()
        depth = 1
        self.layer = nn.Sequential(
            nn.Conv2d(1, depth, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(depth)#,
            ###nn.Flatten(),            
            ###nn.Linear(40*40, 1)
            )
        
        ###f2 = 1 #number of output channels
        ###self.layer = nn.Sequential(
            ###nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            ###nn.ReLU(),
            ###nn.BatchNorm2d(f2),
            ###nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(36 * 36 * depth, 160) #fully connected layer
        self.fc2 = nn.Linear(160, 20)
        self.fc3 = nn.Linear(20, 2)
        #self.fc3 = nn.Sequential(
            #nn.Flatten(),
            #nn.Linear(20, 2)
            #)
    def forward(self,x):
        batch_size = 1
        #print(x.size())
        #x = x.view(batch_size, -1)
        
        ## use ^^^ to get the number 1600
        x = self.layer(x)
        #print(x.size())
        ###x = self.layer(x)
        ###x = x.reshape(x.size(0), -1)
        x = x.view(1, -1)
        #print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

            


from ahk import AHK
ahk = AHK()

#set up image taking thing
import ctypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
print(w, h)
#w = 1920
#h = 1080


xvals = list(range(0,w,100))
tx = len(xvals)
yvals = list(range(0,h,100))
ty = len(yvals)

mspeed = 0



#

def nextPos(indx, indy): 
    xxx = xvals[indx]
    yyy = yvals[indy]    
    
    ahk.mouse_move(x=xxx, y=yyy, speed=mspeed) 
    print(ahk.mouse_position)
    indx = indx + 1
    if (indx >= tx):
        indx = 0
        indy = indy + 1
    if (indy >= ty):
        print("done")
        indx = 0
        indy = 0        
    return xxx, yyy, indx, indy



import ctypes
user32 = ctypes.windll.user32
screensize = ( user32.GetSystemMetrics(0)    ,    user32.GetSystemMetrics(1) )

camera = 1

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


def convert_to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def blobParam(thr=(0,255,10), 
              col=(False, 200), 
              area=(False, 30, 2000), 
              cir=(False, 0.5), 
              conv=(False, 0.5), 
              iner=(False, 0.5)):
    '''
    '''
    if (True):
        params = cv2.SimpleBlobDetector_Params()
        
        # Set thresholds for the image binarization.
        params.minThreshold = thr[0]
        params.maxThreshold = thr[1]
        params.thresholdStep = thr[2]
        
        # Filter by colour.
        params.filterByColor = col[0]
        params.blobColor = col[1]
        
        # Filter by Area.
        params.filterByArea = area[0]
        params.minArea = area[1]
        params.maxArea = area[2]
        
        # Filter by Circularity
        params.filterByCircularity = cir[0]
        params.minCircularity = cir[1]
        
        # Filter by Convexity
        params.filterByConvexity = conv[0]
        params.minConvexity = conv[1]
        
        # Filter by Inertia
        params.filterByInertia = iner[0]
        params.minInertiaRatio = iner[1]
        
        blobdetector = cv2.SimpleBlobDetector_create(params) 
        return blobdetector

#Unused
def houghCircle(img):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1,
                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)    
    img = img_original.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # Draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # Draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)    


def setDim(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return (width, height)     



###roi_gray_l = cv2.resize(roi_gray_l, dim)       
###roi_color_l = cv2.resize(roi_color_l, dim) 
###roi_blur_l = cv2.GaussianBlur(roi_gray_l, (5,5), 0)

###ret, roi_T_l = cv2.threshold(roi_blur_l, size2, 255, cv2.THRESH_BINARY)

def otsuThreshold(img):
    #blur = cv2.GaussianBlur(img,(5,5),0)
    ret, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret, th
    

def detect_pupil(roi_color, param1, param2, minRadius, maxRadius, input1, input2):
    dim = setDim(roi_color, 200)
    roi_color = cv2.resize(roi_color, dim)          
    roi_gray = convert_to_gray_scale(roi_color)
    
    roi_blur = cv2.GaussianBlur(roi_gray, (3,3), 0)
    #ret, roi_T = cv2.threshold(roi_blur, input2, 255, cv2.THRESH_BINARY)
    ret, roi_T = otsuThreshold(roi_blur)
    
    
    des = cv2.bitwise_not(roi_T)
    contours,hierarchy = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(des,[cnt],0,255,-1)
    kernel = np.ones((2,2), np.uint8)
    des = cv2.erode(des, kernel, iterations=input1)    
    roi_T2 = cv2.bitwise_not(des)
    
    testin1 = cv2.getTrackbarPos('test1', 'blob')
    if (testin1 < 0):
        cv2.namedWindow('blob')
        cv2.createTrackbar("test1", 'blob', 0, 3000, nothing)
        testin1 = cv2.getTrackbarPos('test1', 'blob')
        
    roi_T2BLOB = detect_blob(roi_T2, blobParam(cir=(False, float(55/255)), area=(True, 500, 1000)))
    cv2.imshow('final2', roi_T2BLOB)
    
    contours,hierarchy = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)    
    final1 = cv2.drawContours(roi_gray, contours, -1, (100), 3)
    final = np.concatenate((roi_T, des, roi_T2, final1),axis=1)
    cv2.imshow('final', final)    
    
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1,
                                #param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    #if circles is not None:
        #circles = np.uint16(np.around(circles))

        #for i in circles[0,:]:
            ## Draw the outer circle
            #cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            ## Draw the center of the circle
            #cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)    
    
    #cv2.imshow('frame1', frame)
    
circx = [0]
circy = [0]
circI = 0
circMax = 5

def circBuff(xxx, yyy):
    global circx
    global circy
    global circI
    global circMax
    try:
        circx[circI] = xxx
        circy[circI] = yyy
    except:
        circx.append(xxx)
        circy.append(yyy)
    circI += 1
    if circI >= circMax:
        circI = 0
    avex = sum(circx) / len(circx)
    avey = sum(circy) / len(circy)
    return [avex, avey]
    
    
def eyeControl(roi_gray_l, model, control=True):
    nnim = torch.tensor([[roi_gray_l]]).to(dtype=torch.float,device=device)
    #print(type(roi_gray_l))
    output = model(nnim)
    #print(output[0][0].item())
    outx = output[0][0].item()
    outy = output[0][1].item()
    #print(outx, outy)
    [outx, outy] = circBuff(outx, outy)
    if control:
        ahk.mouse_move(x=outx, y=outy, speed=1)     
    else:
        print(outx, outy)
        return (outx, outy)

def detect_blob(frame, blobdetector, color=(0,0,255)):
    #gray = convert_to_gray_scale(frame)
    #gray=  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
    #gray = cv2.bitwise_not(gray)
    keypoints = blobdetector.detect(frame)
    detected = cv2.drawKeypoints(frame, keypoints, None, color,
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)        
    #cv2.imshow('detected', detected)
    return detected

red_c = (0, 0, 255)
gre_c = (0, 255, 0)
blu_c = (255, 0, 0)
cya_c = (255, 255, 0)
yel_c = (0, 255, 255)




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")

args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor


def sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        print("error")



def main():
    model = mynn().to(device)
    #model.load_state_dict(torch.load("model.pth"))  
    model.load_state_dict(torch.load("model(218).pth"), strict=False)
    #checkpoint = torch.load("model.pth") # ie, model_best.pth.tar
    #model.load_state_dict(checkpoint['state_dict'])    
    model.eval()
    
    
    detector = dlib.get_frontal_face_detector() 
    
    predictor = dlib.shape_predictor(args["shape_predictor"])

    cap = cv2.VideoCapture(camera)
    
    indx = 0
    indy = 0
    picnum = 0
    
    data_mode = False
    ai_mode = False
    continuous_capture_mode = False
    capdelay = 0
    
    
    #error calculations;
    totalSample = 0
    per1=0
    per2=0
    per3=0
    error = 0
    
    while(True): 
        # Read the parameters from the GUI
        param1 = cv2.getTrackbarPos('Canny Threshold', 'Parameters')
        param2 = cv2.getTrackbarPos('Accumulator Threshold', 'Parameters')
        minRadius = cv2.getTrackbarPos('Min Radius', 'Parameters')
        maxRadius = cv2.getTrackbarPos('Max Radius', 'Parameters')
        sizeInc = cv2.getTrackbarPos('Size', 'Parameters')
        input1 = cv2.getTrackbarPos('input1', 'Parameters')
        input2 = cv2.getTrackbarPos('input2', 'Parameters')
        
        ret, frame = cap.read() #ret is true if there is video. 
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        
        rects = detector(gray, 1)
        frame2 = frame.copy()
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)  # convert dlib's rectangle to a OpenCV-style bounding box  [i.e., (x, y, w, h)], then draw the face bounding box
            #^ just an array of points. 
            
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            
            cv2.rectangle(frame2, (x, y), (x + w, y + h), gre_c, 2)
            
            # show the face number
            cv2.putText(frame2, "Face #{}".format(i + 1), (x - 10, y - 10),   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gre_c, 2)
            
            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the frame
            #####for (x, y) in shape:
                #####cv2.circle(frame2, (x, y), 1, red_c, -1)
                
            
                
            #print("start")
            left_centre = [0, 0]
            leftMin = [np.intc(0), np.intc(0)]
            leftMax = [np.intc(0), np.intc(0)]
            for i in range (36, 42): # left eye
                cv2.circle(frame2, (shape[i,0], shape[i,1]), 1, red_c, -1)
                left_centre[0] += shape[i,0] 
                left_centre[1] += shape[i,1]
                if (leftMin[0] == 0 or shape[i,0] < leftMin[0]):
                    leftMin[0] =  shape[i,0]
                if (leftMin[1] == 0 or shape[i,1] < leftMin[1]):
                    leftMin[1] =  shape[i,1]
                if (leftMax[0] == 0 or shape[i,0] > leftMax[0]):
                    leftMax[0] =  shape[i,0]
                if (leftMax[1] == 0 or shape[i,1] > leftMax[1]):
                    leftMax[1] =  shape[i,1]
            leftMin[1] -= np.intc(sizeInc)
            leftMax[1] += np.intc(sizeInc)
            left_centre [0]= np.intc(left_centre [0]/6)
            left_centre [1]= np.intc(left_centre [1]/6)
            cv2.circle(frame2, (np.intc(left_centre[0]), np.intc(left_centre[1])), 1, yel_c, -1)
        
            
            
            
            leftMin[1] = left_centre[1] - np.intc(20)
            leftMax[1] = left_centre[1] + np.intc(20)
            leftMin[0] = left_centre[0] - np.intc(20)
            leftMax[0] = left_centre[0] + np.intc(20)
            
            roi_gray_l   = gray[leftMin[1]:leftMax[1], leftMin[0]:leftMax[0]]
            roi_color_l  = frame[leftMin[1]:leftMax[1], leftMin[0]:leftMax[0]]   
            ####roi_color_l  = frame[left_centre[0]-sizeInc:left_centre[0]+sizeInc, left_centre[1]-sizeInc:left_centre[1]+sizeInc]  
            
            #cv2.imshow("newwin", roi_color_l)
            #detect_pupil(roi_color_l, param1, param2, minRadius, maxRadius, input1, input2)
            cv2.rectangle(frame2, (leftMin[0], leftMin[1], leftMax[0] - leftMin[0] ,leftMax[1] - leftMin[1]), cya_c,2)  # Draw a blue box around each face.
            '''machine learning part'''
            cv2.imshow('eye', roi_gray_l)
            #######if (ai_mode):
                #######eyeControl(roi_gray_l, model)
            #######else:
                #######[xread, yread] = eyeControl(roi_gray_l, model, False)
                #######totalSample = totalSample +1
                #######xerr = -ahk.mouse_position[0] + xread
                #######yerr = -ahk.mouse_position[1] + yread
                #######error = (xerr**2 + yerr**2)**0.5
                #######ser = ""
                #######if error < 100:
                    #######per1 = per1 + 1
                    #######ser = ":<100"
                #######elif error < 200:
                    #######per2 = per2 + 1
                    #######ser = ":100-200"
                #######else:
                    #######per3 = per3 + 1
                    #######ser = ":>200"
                #######print("Error = {}, {}".format(error, ser))
                
                
            ###ret, roi_T_l = cv2.threshold(roi_blur_l, size2, 255, cv2.THRESH_BINARY)
            #detect_blob(roi_T_l , roi_color_l)
            ##kernel = np.ones((3,3), np.uint8)
            ##roi_T_l = cv2.erode(roi_T_l, kernel, iterations=size2)
            ##roi_T_l = cv2.morphologyEx(roi_T_l, cv2.MORPH_CLOSE, kernel, iterations=size2)
                        

            ###if (True):
                ###right_centre = [0, 0]
                ###for i in range (42, 48): # right eye
                    ###cv2.circle(frame2, (shape[i,0], shape[i,1]), 1, red_c, -1)
                    ###right_centre[0] += shape[i,0] 
                    ###right_centre[1] += shape[i,1]
                ###right_centre [0]/= 6
                ###right_centre [1]/= 6
                ###cv2.circle(frame2, (np.intc(right_centre[0]), np.intc(right_centre[1])), 1, yel_c, -1)        
                    
                ###for i in range (27, 31): # nose
                    ###cv2.circle(frame2, (shape[i,0], shape[i,1]), 1, blu_c, -1)
            
        #end of if loop
        #final = np.concatenate((frame2, frame),axis=1)
        cv2.imshow('maincam', frame2)
        
        ###if(data_mode):
            ###if (time.perf_counter() -  starttime > 2):
                ###starttime = time.perf_counter()
                ###xxx, yyy, indx, indy = nextPos(indx, indy)
            ###else:
                ###if (time.perf_counter() -  starttime > 1):
                    ###if (time.perf_counter() -  capdelay > 0.2):
                        ###capdelay = time.perf_counter()
                        ###xx = ahk.mouse_position[0]
                        ###yy = ahk.mouse_position[1]
                        ###print("capture'{0}.{1}.{2}.png".format(xx, yy,picnum))
                        ###cv2.imwrite('{0}.{1}.{2}.png'.format(xx, yy,picnum), roi_gray_l, [cv2.IMWRITE_JPEG_QUALITY, 10])
                        ###picnum = picnum + 1                         
        
        ###if continuous_capture_mode:
            ###if (time.perf_counter() -  starttime > 2):
                ###starttime = time.perf_counter()
            ###else:
                ###if (time.perf_counter() -  starttime > 1):
                    ###if (time.perf_counter() -  capdelay > 0.2):
                        ###capdelay = time.perf_counter()
                        ###xx = ahk.mouse_position[0]
                        ###yy = ahk.mouse_position[1]
                        ###print("capture'{0}.{1}.{2}.png".format(xx, yy,picnum))
                        ###cv2.imwrite('{0}.{1}.{2}.png'.format(xx, yy,picnum), roi_gray_l, [cv2.IMWRITE_JPEG_QUALITY, 10])
                        ###picnum = picnum + 1                                     
              
        ###if cv2.waitKey(1) & 0xFF == ord('m'):
            ###if not continuous_capture_mode:
                ###print("start continuous_capture_mode mode")
            ###else:
                ###print("stop continuous_capture_mode mode")
            ###starttime = time.perf_counter()
            ###continuous_capture_mode = not continuous_capture_mode    

        ###if cv2.waitKey(1) & 0xFF == ord('n'):
            ###if not data_mode:
                ###print("start data mode")
            ###else:
                ###print("stop data mode")            
            ###xxx, yyy, indx, indy = nextPos(indx, indy)
            ###starttime = time.perf_counter()
            ###data_mode = not data_mode
        
        ###if cv2.waitKey(1) & 0xFF == ord('a'):
            ###if not ai_mode:
                ###print("start AI mode")
            ###else:
                ###print("stop AI mode")            
            ###ai_mode = not ai_mode        
        
        ###if cv2.waitKey(1) & 0xFF == ord('c'):
            ###xx = ahk.mouse_position[0]
            ###yy = ahk.mouse_position[1]
            ###print("capture'{0}.{1}.{2}.png".format(xx, yy,picnum))
            ###cv2.imwrite('{0}.{1}.{2}.png'.format(xx, yy,picnum), roi_gray_l, [cv2.IMWRITE_JPEG_QUALITY, 10])
            ###picnum = picnum + 1
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            
        
    cap.release()
    cv2.destroyAllWindows()
    
    print("Score: \n000 - 100 :{:.2f}\n100 - 200 :{:.2f}\n    > 200 :{:.2f}\n".format(per1/totalSample, per2/totalSample, per3/totalSample))
    #end of main



#start of code kinda

cv2.namedWindow('Parameters')
cv2.createTrackbar('Canny Threshold', 'Parameters', 1, 500, nothing)
cv2.createTrackbar('Accumulator Threshold', 'Parameters', 1, 500, nothing)
cv2.createTrackbar("Min Radius", 'Parameters', 0, 100, nothing)
cv2.createTrackbar("Max Radius", 'Parameters', 1, 100, nothing)
cv2.createTrackbar("Size", 'Parameters', 0, 100, nothing)
cv2.createTrackbar("input1", 'Parameters', 0, 255, nothing)
cv2.createTrackbar("input2", 'Parameters', 0, 255, nothing)

# Set some default parameters
cv2.setTrackbarPos("Max Radius", 'Parameters', 100)
cv2.setTrackbarPos("Canny Threshold", 'Parameters', 100)
cv2.setTrackbarPos("Accumulator Threshold", 'Parameters', 20)
cv2.setTrackbarPos("input2", 'Parameters', 20)
cv2.setTrackbarPos("Size", 'Parameters', 10)
main()
    
#cv2.imshow("Output", image)
#cv2.waitKey(0)
