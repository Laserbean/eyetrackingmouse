# blob_detection.py

import cv2

cap = cv2.VideoCapture(1)  # Open the first camera connected to the computer.

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Set thresholds for the image binarization.
params.minThreshold = 10
params.maxThreshold = 200
params.thresholdStep = 10

# Filter by colour.
params.filterByColor = False
params.blobColor = 200

# Filter by Area.
params.filterByArea = False
params.minArea = 15
params.maxArea = 2000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.3

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.8

detector1 = cv2.SimpleBlobDetector_create(params)


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor




def main():

    while True:
        #param1 = cv2.getTrackbarPos('Canny Threshold', 'Parameters')
        

        ret, frame = cap.read()  # Read an image from the frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        keypoints = detector1.detect(gray)
        detected = cv2.drawKeypoints(frame, keypoints, None, (0,0,255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('detected', detected)  # Show the image on the display.
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the script when q is pressed.
            break
    cap.release()
    cv2.destroyAllWindows()    

# Release the camera device and close the GUI.
main()

